"""Measure Speed Run Phase 2 (one Full Pipeline per top candidate): serial vs threads vs processes.

Speed Run's Phase 2 runs ``run_full_pipeline`` per candidate on a ThreadPoolExecutor. The 2026-11-06 audit said threads are
likely GIL-bound there too and asked for it to be measured before changing it. This reproduces that phase's shape: each
validation gets ``parallel_search_max_workers = total_workers // n_concurrent``.

    python scripts/bench_speed_run_phase2.py --bars 30000 --candidates 4 --concurrent 2

Run it on an idle machine.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

from app.backtest.risk import RiskConfig  # noqa: E402
from app.orchestration.full_pipeline import FullPipelineConfig, run_full_pipeline  # noqa: E402
from app.prop.simulator import PropRules  # noqa: E402
from app.strategy.manual import ManualStrategy  # noqa: E402


def frame(n: int, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-03", periods=n, freq="1h")
    step = 0.00015 * np.where((np.arange(n) // 5) % 2 == 0, 1, -1) + rng.normal(0, 0.00006, n)
    close = 1.1 + np.cumsum(step)
    open_ = np.concatenate([[1.1], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.00003, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.00003, n))
    return pd.DataFrame(dict(timestamp=ts, open=open_, high=high, low=low, close=close, volume=100.0))


def cfg_for(workers: int) -> FullPipelineConfig:
    return FullPipelineConfig(n_folds=3, ga_population=8, ga_generations=2, ga_search_mc_sims=100, final_mc_sims=1500,
                              oos_check_folds=3, holdout_frac=0.2, parallel_search=True, parallel_search_max_workers=workers,
                              save_to_library=False)


def strategy(i: int) -> ManualStrategy:
    fast, slow = 2 + i, 6 + 2 * i
    return ManualStrategy({
        "name": f"sma {fast}/{slow}",
        "indicators": [{"type": "sma", "period": fast, "column": "close", "as": "f"},
                       {"type": "sma", "period": slow, "column": "close", "as": "s"}],
        "long_entry": "f > s", "long_exit": "f < s", "short_entry": "f < s", "short_exit": "f > s",
        "stop_loss_pips": 20, "take_profit_pips": 40,
    })


def _validate(args):
    df, i, workers = args
    out = Path(tempfile.mkdtemp(prefix="t58_p2bench_"))
    try:
        r = run_full_pipeline(df, strategy(i), RiskConfig(), PropRules(), out, cfg_for(workers), progress_cb=None)
        return (r.verdict, round(float(r.final_mc.evaluation_pass_probability), 6))
    finally:
        shutil.rmtree(out, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=30000)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--concurrent", type=int, default=2)
    a = ap.parse_args()
    df = frame(a.bars)
    total = os.cpu_count() or 2
    per_job = max(1, total // a.concurrent)
    jobs = [(df, i, per_job) for i in range(a.candidates)]
    print(f"{a.candidates} validations x {a.bars:,} bars, {a.concurrent} concurrent, {per_job} GA worker(s) each, {total} cores")

    t0 = time.perf_counter(); serial = [_validate(j) for j in jobs]; ts = time.perf_counter() - t0
    print(f"serial        {ts:7.1f}s  1.00x")
    with ThreadPoolExecutor(a.concurrent) as p:
        t0 = time.perf_counter(); thr = list(p.map(_validate, jobs)); t = time.perf_counter() - t0
    print(f"threads       {t:7.1f}s  {ts / t:4.2f}x  same={thr == serial}")
    with ProcessPoolExecutor(a.concurrent) as p:
        list(p.map(_validate, [(frame(2000), 0, per_job)] * a.concurrent))   # warm workers (spawn + import) so it is not counted
        t0 = time.perf_counter(); proc = list(p.map(_validate, jobs)); t = time.perf_counter() - t0
    print(f"processes     {t:7.1f}s  {ts / t:4.2f}x  same={proc == serial}")


if __name__ == "__main__":
    main()
