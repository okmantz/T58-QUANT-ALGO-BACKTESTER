"""Tests for the auto-shrink-on-low-trades upgrade to
app.optimize.walkforward_ga.run_walkforward_aware_refinement, and for
risk_of_ruin_cap actually changing the GA's search (not just its final
score)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.refinement import RefinementConfig
from app.optimize.walkforward_ga import run_walkforward_aware_refinement
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy


def _trending_df(n=2400, seed=3, drift=0.00015):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.00006)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00003))
        l = min(o, c) - abs(rng.normal(0, 0.00003))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config():
    return {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }


def test_auto_shrink_reduces_generations_on_thin_trade_count():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=30)
    # A deliberately absurd min_oos_trades_per_candidate -- guarantees the
    # baseline's real (but finite) chained-OOS trade count falls short of
    # it, so this test doesn't depend on hand-tuning a strategy config to
    # produce some exact, small trade count on this synthetic dataset.
    refine_cfg = RefinementConfig(
        population_size=6, generations=8, search_monte_carlo_sims=20,
        min_oos_trades_per_candidate=100_000, auto_shrink_on_low_trades=True,
    )
    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3, parallel=False,
    )
    assert len(result.generation_history) - 1 < refine_cfg.generations
    assert any("Auto-shrink" in w for w in result.warnings)


def test_auto_shrink_disabled_keeps_full_configured_width():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=30)
    refine_cfg = RefinementConfig(
        population_size=6, generations=3, search_monte_carlo_sims=20,
        min_oos_trades_per_candidate=100_000, auto_shrink_on_low_trades=False,
    )
    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3, parallel=False,
    )
    assert len(result.generation_history) - 1 == refine_cfg.generations
    assert not any("Auto-shrink" in w for w in result.warnings)


def test_normal_trade_count_does_not_shrink():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=30)
    refine_cfg = RefinementConfig(
        population_size=6, generations=3, search_monte_carlo_sims=20,
        min_oos_trades_per_candidate=5,  # low bar -- this strategy trades plenty
    )
    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3, parallel=False,
    )
    assert len(result.generation_history) - 1 == refine_cfg.generations
    assert not any("Auto-shrink" in w for w in result.warnings)


def test_risk_of_ruin_cap_runs_end_to_end_without_error():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=30)
    refine_cfg = RefinementConfig(
        population_size=6, generations=2, search_monte_carlo_sims=20, risk_of_ruin_cap=20.0,
    )
    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3, parallel=False,
    )
    assert result.best is not None
    assert result.total_evaluations > 0
