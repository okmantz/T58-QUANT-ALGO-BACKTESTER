"""Data manifest + coverage gate (2026-10 audit).

``docs/DATA_COVERAGE.md`` used to be a report nobody enforced. This module turns it into a gate:

  * ``load_manifest()``  reads ``data/data_manifest.json`` (rows: file, symbol, years, synthetic, flags);
    ``scripts/build_data_manifest.py`` regenerates it from the dataset folder.
  * ``check_dataset(df, label)`` decides, for the data a pipeline run is about to use, whether the
    data can support a READY / HIGH CONFIDENCE verdict:
        - SYNTHETIC data (manifest flag, a SYNTHETIC token in the file name/label) -> never
        - fewer than ``MIN_YEARS_FOR_READY`` (5) years of history in the frame -> never
        - a manifest row flagged unreadable -> never
        - futures-style data that still carries contract-roll jumps (splices / interleaved flips) -> never
          (re-import it back-adjusted; the importer does this automatically for continuous-future files)
    and returns reasons. ``check_dataset`` never raises; if it cannot decide it says so.
  * ``assert_manifest_fresh()`` is the pipeline-start check: when a manifest exists it warns about
    manifest rows whose files are missing on disk.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

MIN_YEARS_FOR_READY = 5.0
MIN_ROLL_EVENTS_TO_FLAG = 2     # one stray jump can be a real gap; two or more is a roll pattern
MANIFEST_RELATIVE = Path("data") / "data_manifest.json"
_SYNTH = re.compile(r"synthetic|\bfake\b|\bsim(ulated)?\b|\bdemo\b", re.I)


_FUTURES_ROOTS = (
    "es|nq|ym|rty|mes|mnq|mym|m2k|gc|mgc|si|hg|pl|pa|cl|mcl|ng|rb|ho|zn|zb|zf|zt|zc|zs|zw|zl|zm|6e|6j|6b|6a|6c|6s|le|he|vx"
)
_FUTURES_TOKEN = re.compile(rf"^(?:{_FUTURES_ROOTS})\d*$|^(?:futures?|fut)$")


def looks_like_futures(label: str | None) -> bool:
    """True when a file/label looks like a (continuous) futures series, e.g. ``NQ1!/NQ_1m.parquet``,
    ``futures_ES.F_1m.parquet``, ``GC_1min.csv``. Spot FX / crypto / equities return False."""
    if not label:
        return False
    tokens = [t for t in re.split(r"[^a-z0-9]+", str(label).lower().replace("\\", "/")) if t]
    return any(_FUTURES_TOKEN.match(t) for t in tokens)


def roll_gap_counts(df: pd.DataFrame | None) -> tuple[int, int] | None:
    """(isolated contract splices, interleaved contract flips) still present in ``df``, or None if it
    cannot be measured. Back-adjusted data returns (0, 0)."""
    try:
        from app.data.continuous_contract import detect_splices_and_interleaves
        if df is None or len(df) < 1000 or not {"timestamp", "open", "high", "low", "close"} <= set(df.columns):
            return None
        rep = detect_splices_and_interleaves(df)
        return len(rep.splice_rolls), len(rep.interleave_events)
    except Exception:  # noqa: BLE001 -- the gate must never raise
        return None


@dataclass
class DataGateResult:
    ok_for_ready: bool
    years: float | None
    synthetic: bool
    reasons: list = field(default_factory=list)
    manifest_row: dict | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def manifest_path(root: Path | None = None) -> Path:
    return (root or _repo_root()) / MANIFEST_RELATIVE


def load_manifest(path: Path | None = None) -> dict:
    p = path or manifest_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- missing/corrupt manifest means "no manifest", handled by callers
        return {}


def _find_row(manifest: dict, label: str | None) -> dict | None:
    if not label:
        return None
    rows = manifest.get("files") or []
    lab = str(label).replace("\\", "/").lower()
    base = lab.rsplit("/", 1)[-1]
    best = None
    for r in rows:
        f = str(r.get("file", "")).lower()
        if f == lab or f.endswith("/" + base) or f.rsplit("/", 1)[-1] == base:
            best = r
            if f == lab:
                break
    return best


def frame_years(df: pd.DataFrame) -> float | None:
    try:
        ts = pd.to_datetime(df["timestamp"])
        return float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / (365.25 * 86400))
    except Exception:  # noqa: BLE001
        return None


def check_dataset(df: pd.DataFrame | None, label: str | None = None, *, manifest: dict | None = None,
                  min_years: float = MIN_YEARS_FOR_READY) -> DataGateResult:
    """Can this data support a READY verdict? See module docstring."""
    reasons: list[str] = []
    row = _find_row(manifest if manifest is not None else load_manifest(), label)
    years = frame_years(df) if df is not None else (row or {}).get("years")
    synthetic = bool(label and _SYNTH.search(str(label))) or bool(row and row.get("synthetic"))
    if synthetic:
        reasons.append(f"DATA GATE: '{label}' is synthetic/simulated data -- results on it say nothing about the market, so READY is not allowed.")
    if row and "UNREADABLE" in str(row.get("flags", "")).upper():
        reasons.append(f"DATA GATE: '{label}' is flagged unreadable in the data manifest.")
    if years is None:
        reasons.append("DATA GATE: could not measure the history length of this dataset; READY is not allowed on unmeasured data.")
    elif years < min_years:
        reasons.append(f"DATA GATE: only {years:.1f} years of history (< {min_years:g}); too short to cover more than one volatility regime, so READY is not allowed.")
    if df is not None and looks_like_futures(label):
        counts = roll_gap_counts(df)
        if counts is not None and counts[0] + counts[1] >= MIN_ROLL_EVENTS_TO_FLAG:
            reasons.append(f"DATA GATE: '{label}' carries unadjusted contract-roll gaps ({counts[0]} splice(s), {counts[1]} interleaved flip(s)); "
                           "every roll is a fake price jump the strategy can 'trade', so READY is not allowed. Re-import it back-adjusted.")
    return DataGateResult(ok_for_ready=not reasons, years=years, synthetic=synthetic, reasons=reasons, manifest_row=row)


def assert_manifest_fresh(root: Path | None = None) -> list:
    """Pipeline-start check. Returns warnings (never raises): missing manifest, or manifest rows whose
    file is not on disk (stale manifest)."""
    root = root or _repo_root()
    m = load_manifest(manifest_path(root))
    if not m:
        return ["No data/data_manifest.json found -- run scripts/build_data_manifest.py. Synthetic/short-history data is then only caught by file name and measured span."]
    warns = []
    raw = root / "data" / "raw"
    if raw.exists():
        missing = [r["file"] for r in m.get("files", []) if not (raw / r["file"]).exists()]
        if missing:
            warns.append(f"data manifest lists {len(missing)} file(s) that are not on disk (first: {missing[0]}); regenerate it.")
    return warns


def manifest_from_coverage_markdown(md_text: str) -> dict:
    """Parse the table in docs/DATA_COVERAGE.md into manifest rows."""
    rows = []
    for line in md_text.splitlines():
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| file"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        f, sym, bars, start, end, years, bar, flags = cells[:8]

        def _num(x):
            try:
                return float(x)
            except ValueError:
                return None
        rows.append({
            "file": f, "symbol": None if sym == "-" else sym,
            "bars": int(_num(bars)) if _num(bars) is not None else None,
            "start": None if start == "-" else start, "end": None if end == "-" else end,
            "years": _num(years), "bar": bar, "flags": flags,
            "synthetic": "SYNTHETIC" in flags.upper(),
        })
    return {"schema": 1, "min_years_for_ready": MIN_YEARS_FOR_READY, "files": rows}
