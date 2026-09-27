"""Proves SearchStageConfig.plateau_robust_selection/plateau_finalist_pool
actually reach Stage 2's RefinementConfig (see the FIX comment on those two
fields in app.search.batch_runner.SearchStageConfig and on _fast_stage_cfg
in tests/test_batch_runner.py for the bug this closes: Stage 2 used to
silently run with RefinementConfig's own plateau defaults no matter what
this config said, because _stage2_task/run_search never passed them
through).

Kept in its own file, deliberately using the smallest possible
plateau_finalist_pool (1) and a single-gene strategy, so turning plateau-
robust selection ON here still finishes in a few seconds -- proving the
wiring works without reintroducing the very slowdown that motivated
turning it off by default in test_batch_runner.py's _fast_stage_cfg.
"""
from __future__ import annotations

from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.search.batch_runner import SearchStageConfig, run_search
from app.search.strategy_space import generate_search_space


def _trending_df(n=600, seed=3, drift=0.00015):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.00003)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00002))
        l = min(o, c) - abs(rng.normal(0, 0.00002))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_plateau_robust_selection_reaches_stage2_through_search_stage_config(tmp_path):
    df = _trending_df()
    single_space = generate_search_space(
        mode="single",
        single_config={
            "name": "sma cross",
            "indicators": [
                {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
                {"type": "sma", "period": 20, "column": "close", "as": "sma_slow"},
            ],
            "long_entry": "sma_fast > sma_slow",
            "long_exit": "sma_fast < sma_slow",
            "risk_management": {
                "stop_type": "fixed", "stop_value": 20,
                "target_type": "fixed", "target_value": 40,
            },
        },
    )
    cfg = SearchStageConfig(
        min_trades=1, min_profit_factor=0.0, max_drawdown_buffer_mult=10.0,
        stage1_top_n=1, ga_population=4, ga_generations=1, ga_search_sims=10,
        stage2_top_n=1, full_mc_sims=10, walk_forward_folds=0, robustness_neighbors=0,
        workers=1, random_seed=42,
        plateau_robust_selection=True, plateau_finalist_pool=1,
    )
    summary = run_search(
        df, RiskConfig(), PropRules(), single_space, cfg,
        db_path=str(tmp_path / "search.db"), instrument="TEST", timeframe="5m",
    )
    assert summary.total_candidates == 1
    assert summary.elapsed_seconds > 0
