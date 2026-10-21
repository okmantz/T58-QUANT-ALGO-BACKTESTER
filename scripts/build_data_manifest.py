"""Regenerate data/data_manifest.json by scanning data/raw (2026-10 audit).

    python scripts/build_data_manifest.py            # scan data/raw, write data/data_manifest.json
    python scripts/build_data_manifest.py --from-coverage-doc   # just re-parse docs/DATA_COVERAGE.md

Flags written per file: SYNTHETIC (file name contains SYNTHETIC/FAKE), <2y history, UNREADABLE,
RAW ROLL GAPS (futures-style files whose raw bytes still carry contract-roll jumps; informational -- the importer
back-adjusts continuous-future files on load, and the pipeline data gate measures the LOADED frame, so only a frame that
still carries roll gaps after import blocks READY).
The pipeline's data gate (app.data.data_manifest.check_dataset) refuses READY on synthetic data or on
less than 5 years of history, using this manifest plus the actual span of the loaded frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.data.data_manifest import (  # noqa: E402
    MIN_ROLL_EVENTS_TO_FLAG, MIN_YEARS_FOR_READY, looks_like_futures, manifest_from_coverage_markdown, roll_gap_counts,
)


def _scan_file(p: Path, raw: Path) -> dict:
    rel = p.relative_to(raw).as_posix()
    row = {"file": rel, "symbol": None, "bars": None, "start": None, "end": None, "years": None, "bar": None, "flags": "", "synthetic": False}
    flags = []
    if "synthetic" in p.name.lower() or "fake" in p.name.lower():
        flags.append("SYNTHETIC")
        row["synthetic"] = True
    try:
        if p.stat().st_size < 1024 and p.read_bytes()[:40].startswith(b"version https://git-lfs"):
            flags.append("GIT-LFS POINTER (run `git lfs pull`; file content not downloaded)")
            row["error"] = "git-lfs pointer"
            row["flags"] = ", ".join(flags)
            return row
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
            # Header-less exports (TradingView/MT "2025-09-06 08:50<TAB>o<TAB>h<TAB>l<TAB>c<TAB>v"): the first "column name" is a timestamp.
            first = str(df.columns[0])
            if str(first)[:2] in ("19", "20") and pd.notna(pd.to_datetime(first.replace("\t", " ").split(" ")[0], errors="coerce")):
                df = pd.read_csv(p, header=None, sep=r"[\t,;]", engine="python")
                df = df.iloc[:, :6]
                df.columns = ["timestamp", "open", "high", "low", "close", "volume"][: df.shape[1]]
        cols = {c.lower(): c for c in df.columns}
        tcol = cols.get("timestamp") or cols.get("time") or cols.get("datetime") or cols.get("date") or df.columns[0]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = pd.to_datetime(df[tcol], errors="coerce", utc=True).dropna()
        row["bars"] = int(len(ts))
        row["start"], row["end"] = str(ts.min().date()), str(ts.max().date())
        row["years"] = round(float((ts.max() - ts.min()).total_seconds() / (365.25 * 86400)), 2)
        row["bar"] = str(ts.diff().median())
        if row["years"] < 2:
            flags.append("<2y history")
        if row["years"] < MIN_YEARS_FOR_READY:
            flags.append(f"<{MIN_YEARS_FOR_READY:g}y (not READY-eligible)")
        if looks_like_futures(rel) or looks_like_futures(p.parent.name):
            frame = pd.DataFrame({
                "timestamp": pd.to_datetime(df[tcol], errors="coerce", utc=True),
                **{c: pd.to_numeric(df[cols[c]], errors="coerce") for c in ("open", "high", "low", "close") if c in cols},
            }).dropna().reset_index(drop=True)
            counts = roll_gap_counts(frame)
            if counts is not None:
                row["roll_gaps"] = f"{counts[0]} splice(s), {counts[1]} interleaved flip(s)"
                if counts[0] + counts[1] >= MIN_ROLL_EVENTS_TO_FLAG:
                    flags.append("RAW ROLL GAPS (back-adjusted at import)")
    except Exception as exc:  # noqa: BLE001
        flags.append("UNREADABLE")
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    # 2026-10-20 (report section 5): every file must resolve to an instrument spec with MEASURED costs, otherwise the backtest silently
    # runs with zero costs. Unresolved files are flagged loudly and listed by scripts/bootstrap_data.py.
    try:
        from app.data.cost_model import resolve_cost
        from app.data.session_clock import infer_symbol
        sym = infer_symbol(rel) or infer_symbol(p.parent.name)
        line = resolve_cost(sym or rel) or resolve_cost(p.parent.name)
        row["symbol"] = row["symbol"] or (line.symbol if line else (sym or None))
        row["spec_resolved"] = bool(line)
        if line:
            row["round_trip_cost_usd"] = round(line.round_trip_dollars(), 2)
            row["cost_source"] = line.source
        else:
            flags.append("NO INSTRUMENT SPEC (costs unmeasured)")
    except Exception as exc:  # noqa: BLE001
        row["spec_resolved"] = False
        flags.append(f"spec check failed: {type(exc).__name__}")
    if row.get("bars") == 0 and "UNREADABLE" not in flags:
        flags.append("NO PARSEABLE TIMESTAMPS (0 bars)")
    row["flags"] = ", ".join(flags) if flags else "ok"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-coverage-doc", action="store_true")
    args = ap.parse_args()
    out = ROOT / "data" / "data_manifest.json"
    if args.from_coverage_doc:
        m = manifest_from_coverage_markdown((ROOT / "docs" / "DATA_COVERAGE.md").read_text(encoding="utf-8"))
    else:
        raw = ROOT / "data" / "raw"
        files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in (".csv", ".parquet"))
        m = {"schema": 1, "min_years_for_ready": MIN_YEARS_FOR_READY, "files": [_scan_file(p, raw) for p in files]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(m['files'])} files)")


if __name__ == "__main__":
    main()
