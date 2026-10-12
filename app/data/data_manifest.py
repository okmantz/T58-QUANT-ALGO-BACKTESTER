"""Data manifest + coverage gate (2026-10 audit).

``docs/DATA_COVERAGE.md`` used to be a report nobody enforced. This module turns it into a gate:

  * ``load_manifest()``  reads ``data/data_manifest.json`` (rows: file, symbol, years, synthetic, flags);
    ``scripts/build_data_manifest.py`` regenerates it from the dataset folder.
  * ``check_dataset(df, label)`` decides, for the data a pipeline run is about to use, whether the
    data can support a READY / HIGH CONFIDENCE verdict:
        - SYNTHETIC data (manifest flag, a SYNTHETIC token in the file name/label) -> never
        - fewer than ``MIN_YEARS_FOR_READY`` (5) years of history in the frame -> never
        - a manifest row flagged unreadable -> never
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
MANIFEST_RELATIVE = Path("data") / "data_manifest.json"
_SYNTH = re.compile(r"synthetic|\bfake\b|\bsim(ulated)?\b|\bdemo\b", re.I)


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
