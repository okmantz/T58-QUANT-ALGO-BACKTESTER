"""Date range, bar spacing, gaps and instrument-spec coverage for every dataset under data/raw.

    python scripts/data_coverage_audit.py [--root data/raw] [--out docs/DATA_COVERAGE.md]

Answers, per file: does it span enough history for cross-market tests, is the bar
size what the file name says, are there long gaps, is it flagged synthetic, and does
the instrument have a tick/point/cost spec (otherwise cost-aware tests are unsafe).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            if "ts" in df.columns and "timestamp" not in df.columns:
                df = df.rename(columns={"ts": "timestamp"})
            if "timestamp" not in df.columns:
                df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        else:
            from app.data.importer import import_csv
            res = import_csv(str(path))
            if not res.is_valid:
                return None
            df = res.dataframe
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if getattr(df["timestamp"].dt, "tz", None) is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        return df.sort_values("timestamp")
    except Exception:
        return None


def audit(root: Path) -> list[dict]:
    from app.data.instrument_specs import get_any_instrument_spec as get_instrument_spec, guess_any_instrument_symbol as guess_instrument_symbol
    rows = []
    for p in sorted(list(root.rglob("*.parquet")) + list(root.rglob("*.csv"))):
        df = _load(p)
        market = p.parent.name
        sym = guess_instrument_symbol(market) or guess_instrument_symbol(p.stem)
        spec = get_instrument_spec(sym) if sym else None
        if df is None or len(df) < 2:
            rows.append(dict(file=str(p.relative_to(root)), market=market, status="UNREADABLE"))
            continue
        ts = df["timestamp"]
        step = ts.diff().median()
        gaps = ts.diff()
        long_gaps = int((gaps > max(step * 50, pd.Timedelta(days=5))).sum())
        years = (ts.iloc[-1] - ts.iloc[0]).days / 365.25
        flags = []
        if "synthetic" in p.name.lower():
            flags.append("SYNTHETIC")
        if spec is None:
            flags.append("no instrument spec")
        if years < 2:
            flags.append("<2y history")
        if long_gaps:
            flags.append(f"{long_gaps} long gaps")
        rows.append(dict(file=str(p.relative_to(root)), market=market, symbol=sym or "-", bars=len(df), start=str(ts.iloc[0].date()),
                         end=str(ts.iloc[-1].date()), years=round(years, 2), step=str(step), flags=", ".join(flags) or "ok"))
    return rows


def to_markdown(rows: list[dict]) -> str:
    out = ["# Dataset coverage audit", "", "| file | symbol | bars | start | end | years | bar | flags |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| {file} | {symbol} | {bars} | {start} | {end} | {years} | {step} | {flags} |".format(**{**dict(symbol="-", bars="-", start="-", end="-", years="-", step="-", flags=r.get("status", "")), **r}))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--out", default="docs/DATA_COVERAGE.md")
    a = ap.parse_args()
    md = to_markdown(audit(Path(a.root)))
    Path(a.out).write_text(md, encoding="utf-8")
    print(md)
