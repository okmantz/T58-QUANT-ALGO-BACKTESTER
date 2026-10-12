"""Regenerate data/data_manifest.json by scanning data/raw (2026-10 audit).

    python scripts/build_data_manifest.py            # scan data/raw, write data/data_manifest.json
    python scripts/build_data_manifest.py --from-coverage-doc   # just re-parse docs/DATA_COVERAGE.md

Flags written per file: SYNTHETIC (file name contains SYNTHETIC/FAKE), <2y history, UNREADABLE.
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

from app.data.data_manifest import MIN_YEARS_FOR_READY, manifest_from_coverage_markdown  # noqa: E402


def _scan_file(p: Path, raw: Path) -> dict:
    rel = p.relative_to(raw).as_posix()
    row = {"file": rel, "symbol": None, "bars": None, "start": None, "end": None, "years": None, "bar": None, "flags": "", "synthetic": False}
    flags = []
    if "synthetic" in p.name.lower() or "fake" in p.name.lower():
        flags.append("SYNTHETIC")
        row["synthetic"] = True
    try:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        tcol = cols.get("timestamp") or cols.get("time") or cols.get("datetime") or cols.get("date") or df.columns[0]
        ts = pd.to_datetime(df[tcol], errors="coerce", utc=True).dropna()
        row["bars"] = int(len(ts))
        row["start"], row["end"] = str(ts.min().date()), str(ts.max().date())
        row["years"] = round(float((ts.max() - ts.min()).total_seconds() / (365.25 * 86400)), 2)
        row["bar"] = str(ts.diff().median())
        if row["years"] < 2:
            flags.append("<2y history")
        if row["years"] < MIN_YEARS_FOR_READY:
            flags.append(f"<{MIN_YEARS_FOR_READY:g}y (not READY-eligible)")
    except Exception as exc:  # noqa: BLE001
        flags.append("UNREADABLE")
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
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
