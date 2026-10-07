"""v9.4: the Full Pipeline PBO step must not take forever, and speeding
it up must not change what it measures.

Two levers, both estimand-preserving:
1. compute_pbo evaluates its independent CSCV paths across a process
   pool (results re-keyed by path index -- bit-identical aggregates).
2. Full Pipeline's PBO inner Monte Carlo runs at a reduced simulation
   count (pbo_mc_sims=50): PBO consumes only each path's candidate
   RANKING, so 200-sim precision was wasted motion.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRISTINE = Path.home() / "workspace" / "tmp-pristine"

from app.backtest.risk import RiskConfig  # noqa: E402
from app.monte_carlo.engine import MonteCarloConfig  # noqa: E402
from app.prop.simulator import PropRules  # noqa: E402
from app.validation.cpcv import compute_pbo  # noqa: E402


def _trending_df(n=1800, seed=3):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    price = 100.0
    rows = []
    for i in range(n):
        step = 0.02 + rng.normal(0, 0.05)
        o = price
        c = o + step
        rows.append((ts[i], o, max(o, c) + 0.01, min(o, c) - 0.01, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_spec(fast: int, slow: int):
    return {
        "source_type": "manual",
        "config": {
            "name": f"sma {fast}/{slow}",
            "indicators": [
                {"type": "sma", "period": fast, "column": "close", "as": "sma_fast"},
                {"type": "sma", "period": slow, "column": "close", "as": "sma_slow"},
            ],
            "long_entry": "sma_fast > sma_slow",
            "long_exit": "sma_fast < sma_slow",
            "stop_loss_pips": 20,
            "take_profit_pips": 40,
        },
    }


SPECS = [_sma_spec(5, 15), _sma_spec(10, 30), _sma_spec(3, 50)]


def test_parallel_paths_match_serial_exactly(monkeypatch):
    df = _trending_df()
    parallel = compute_pbo(df, SPECS, RiskConfig(), n_groups=6, n_test_groups=2,
                           max_paths=6, metric="sharpe_ratio")
    monkeypatch.setattr("app.validation.cpcv.os.cpu_count", lambda: 1)
    serial = compute_pbo(df, SPECS, RiskConfig(), n_groups=6, n_test_groups=2,
                         max_paths=6, metric="sharpe_ratio")
    assert parallel.pbo == serial.pbo
    assert parallel.mean_is_by_candidate == serial.mean_is_by_candidate
    assert parallel.mean_oos_by_candidate == serial.mean_oos_by_candidate
    assert parallel.is_best_candidate_per_path == serial.is_best_candidate_per_path
    assert parallel.n_paths == 6


def test_pbo_with_eval_pass_metric_still_runs():
    df = _trending_df(n=1200)
    result = compute_pbo(
        df, SPECS, RiskConfig(), n_groups=6, n_test_groups=2, max_paths=4,
        metric="eval_pass_probability", prop_rules=PropRules(),
        mc_cfg=MonteCarloConfig(n_simulations=25, random_seed=3),
    )
    assert result.n_candidates == 3
    assert result.n_paths == 4
    assert 0.0 <= result.pbo <= 1.0


def test_full_pipeline_pbo_mc_sims_is_the_reduced_count():
    from app.orchestration.full_pipeline import FullPipelineConfig

    cfg = FullPipelineConfig()
    assert cfg.pbo_mc_sims == 50
    assert cfg.pbo_mc_sims < cfg.ga_search_mc_sims


def _run_against(tree: Path):
    probe = "tests/test_v94_pbo_speed.py"
    dest = tree / probe
    shutil.copy2(ROOT / probe, dest)
    try:
        return subprocess.run(
            [sys.executable, "-m", "pytest", probe, "-x", "-q",
             "-k", "not fails_on_upstream and not _run_against"],
            cwd=tree, capture_output=True, text=True, timeout=900)
    finally:
        dest.unlink(missing_ok=True)


@pytest.mark.skipif(not PRISTINE.exists(), reason="pristine upstream copy not available")
def test_fails_on_upstream_v92():
    r = _run_against(PRISTINE)
    assert r.returncode != 0, r.stdout[-1200:]
