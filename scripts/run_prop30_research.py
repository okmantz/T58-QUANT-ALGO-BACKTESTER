#!/usr/bin/env python3
"""Reproducible 30-day prop evaluation research. Usage: python scripts/run_prop30_research.py [--out reports/prop30] [--instruments NQ ES GC]"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from app.data.importer import import_csv  # noqa: E402
from app.research.prop30_pipeline import Prop30Config, run, to_markdown  # noqa: E402

FILES = {"NQ": "data/raw/NQ1!/futures_NQ.F_1m.parquet", "ES": "data/raw/ES1!/futures_ES.F_1m.parquet",
         "GC": "data/raw/GC1!/futures_GC.F_1m.parquet", "CL": "data/raw/CL1!/futures_CL.F_1m.parquet"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/prop30")
    ap.add_argument("--instruments", nargs="*", default=["NQ", "ES", "GC", "CL"])
    ap.add_argument("--assume-untouched-after", default=None, help="date after which data is declared never previously observed")
    a = ap.parse_args()
    frames = {}
    for k in a.instruments:
        p = ROOT / FILES[k]
        if p.exists():
            frames[k] = import_csv(str(p)).dataframe.reset_index(drop=True)
        else:
            print(f"[skip] {k}: {p} not found", file=sys.stderr)
    res = run(frames, Prop30Config(assume_untouched_after=a.assume_untouched_after))
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "prop30_report.json").write_text(json.dumps(res, indent=2, default=str))
    (out / "prop30_report.md").write_text(to_markdown(res))
    print(to_markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
