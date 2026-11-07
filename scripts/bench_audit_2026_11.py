"""Reproduce the 2026-11-06 audit's speed numbers on the repo's own ES 1-minute data.

    python scripts/bench_audit_2026_11.py memo            # skip-memo off vs on, identical-output check (full history by default)
    python scripts/bench_audit_2026_11.py memo --bars 500000 --stop 80
    python scripts/bench_audit_2026_11.py sweep           # 8 independent backtests: serial vs threads vs processes
    python scripts/bench_audit_2026_11.py sweep --bars 200000 --jobs 8

Run it on an idle machine: a busy CPU is what made the audit's first sweep numbers misleading.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

from app.backtest import execution as ex  # noqa: E402
from app.backtest.risk import RiskConfig  # noqa: E402
from app.data.instrument_specs import apply_instrument_spec  # noqa: E402

DATA = ROOT / "data" / "raw" / "ES1!" / "futures_ES.F_1m.parquet"


def load(bars: int | None) -> pd.DataFrame:
    df = pd.read_parquet(DATA).rename(columns={"ts": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None) if df["timestamp"].dt.tz is not None else df["timestamp"]
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    return df.iloc[:bars].reset_index(drop=True) if bars else df


def signals(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    fast, slow = c.ewm(span=20, adjust=False).mean(), c.ewm(span=80, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    out = np.where((fast > slow) & (rsi > 50), 1, np.where((fast < slow) & (rsi < 50), -1, 0))
    return pd.Series(out, index=df.index)


def mes_risk() -> RiskConfig:
    r = RiskConfig(initial_balance=50_000.0, risk_mode="percent", risk_value=0.5, max_trades_per_day=1000, sizing_mode="skip")
    return apply_instrument_spec(r, "MES")


def run(df, sig, risk, stop_points, memo):
    old = ex._SKIP_MEMO_ENABLED
    ex._SKIP_MEMO_ENABLED = memo
    try:
        t0 = time.perf_counter()
        out = ex.run_execution(df, sig, risk, stop_points / risk.pip_size, 2 * stop_points / risk.pip_size)
        return time.perf_counter() - t0, out
    finally:
        ex._SKIP_MEMO_ENABLED = old


def cmd_memo(a):
    df = load(a.bars)
    sig, risk = signals(df), mes_risk()
    print(f"{len(df):,} ES 1-minute bars, stop {a.stop} points, MES spec, $50k, 0.5% risk")
    t_off, off = run(df, sig, risk, a.stop, memo=False)
    t_on, on = run(df, sig, risk, a.stop, memo=True)
    same = ([asdict(t) for t in off[0]] == [asdict(t) for t in on[0]]
            and np.array_equal(off[1]["equity"].to_numpy(), on[1]["equity"].to_numpy())
            and off[1].attrs["sizing_summary"] == on[1].attrs["sizing_summary"]
            and off[1].attrs["entry_block_counts"] == on[1].attrs["entry_block_counts"])
    print(f"memo off {t_off:7.2f}s | memo on {t_on:7.2f}s | speedup {t_off / t_on:4.2f}x | trades {len(on[0]):,} | identical output: {same}")
    if not same:
        sys.exit(1)


def _one(args):
    df, sig, risk, stop = args
    return len(ex.run_execution(df, sig, risk, stop / risk.pip_size, 2 * stop / risk.pip_size)[0])


def cmd_sweep(a):
    df = load(a.bars)
    sig, risk = signals(df), mes_risk()
    jobs = [(df, sig, risk, 30 + 2 * i) for i in range(a.jobs)]
    t0 = time.perf_counter(); serial = [_one(j) for j in jobs]; ts = time.perf_counter() - t0
    print(f"{a.jobs} backtests x {len(df):,} bars on {os.cpu_count()} cores")
    print(f"serial      {ts:6.2f}s  1.00x")
    for n in (2, 4):
        with ThreadPoolExecutor(n) as p:
            t0 = time.perf_counter(); r = list(p.map(_one, jobs)); t = time.perf_counter() - t0
        print(f"{n} threads   {t:6.2f}s  {ts / t:4.2f}x  same={r == serial}")
    with ProcessPoolExecutor(2) as p:
        list(p.map(_one, jobs[:2]))  # warm the workers so spawn/import is not counted
        t0 = time.perf_counter(); r = list(p.map(_one, jobs)); t = time.perf_counter() - t0
    print(f"2 processes {t:6.2f}s  {ts / t:4.2f}x  same={r == serial}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["memo", "sweep"])
    ap.add_argument("--bars", type=int, default=None)
    ap.add_argument("--stop", type=float, default=40.0)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    {"memo": cmd_memo, "sweep": cmd_sweep}[args.mode](args)
