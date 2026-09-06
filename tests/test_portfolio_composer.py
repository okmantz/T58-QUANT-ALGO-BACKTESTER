from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.portfolio.composer import (
    PortfolioComposerError,
    compose_portfolio,
)
from app.portfolio.portfolio import InstrumentLeg, PortfolioConfig
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy


def _trending_df(n=1200, seed=3, drift=0.00015, start_price=1.10):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    price = start_price
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.0006)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.0003))
        l = min(o, c) - abs(rng.normal(0, 0.0003))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config(fast=5, slow=15):
    return {
        "name": f"sma_{fast}_{slow}",
        "indicators": [
            {"type": "sma", "period": fast, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": slow, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }


def _candidates(n_legs=4):
    legs = []
    for i in range(n_legs):
        df = _trending_df(seed=i + 1, drift=0.00012 * (1 if i % 2 == 0 else -1), start_price=1.0 + i * 0.5)
        legs.append(InstrumentLeg(
            name=f"leg_{i}", df=df, strategy=ManualStrategy(_sma_config(5 + i, 15 + i * 2)), risk=RiskConfig(),
        ))
    return legs


def test_too_few_candidates_raises():
    with pytest.raises(PortfolioComposerError):
        compose_portfolio(_candidates(1), min_legs=2, max_legs=3)


def test_duplicate_names_raise():
    legs = _candidates(2)
    legs[1].name = legs[0].name
    with pytest.raises(PortfolioComposerError):
        compose_portfolio(legs, min_legs=2, max_legs=2)


def test_exhaustive_search_small_pool():
    legs = _candidates(4)
    result = compose_portfolio(legs, min_legs=2, max_legs=3, max_evaluations=100)
    assert result.search_mode == "exhaustive"
    assert result.best_combo_names
    assert result.best_result is not None
    assert result.n_combinations_evaluated > 0
    assert result.leaderboard[0].score >= result.leaderboard[-1].score


def test_greedy_fallback_when_search_space_too_large():
    legs = _candidates(6)
    result = compose_portfolio(legs, min_legs=2, max_legs=4, max_evaluations=8)
    assert result.search_mode == "greedy"
    assert result.n_combinations_evaluated <= 8 + 4  # small slack for the seeding pass bookkeeping
    assert result.best_combo_names
    assert any("greedy forward selection" in w for w in result.warnings)


def test_scores_with_eval_pass_probability_when_prop_rules_supplied():
    legs = _candidates(4)
    cfg = PortfolioConfig(
        initial_balance=50_000,
        prop_rules=PropRules(account_size=50_000, evaluation_profit_target_pct=8.0),
        mc_config=MonteCarloConfig(n_simulations=200),
    )
    result = compose_portfolio(legs, min_legs=2, max_legs=2, max_evaluations=100, portfolio_config=cfg)
    assert result.score_metric == "eval_pass_probability"
    for c in result.leaderboard:
        if c.score_metric == "eval_pass_probability":
            assert 0.0 <= c.score <= 100.0


def test_falls_back_to_profit_factor_without_prop_rules():
    legs = _candidates(3)
    result = compose_portfolio(legs, min_legs=2, max_legs=2, max_evaluations=100)
    assert result.score_metric == "profit_factor"


def test_render_table_and_to_dict():
    legs = _candidates(3)
    result = compose_portfolio(legs, min_legs=2, max_legs=2, max_evaluations=100)
    d = result.to_dict()
    assert d["best_combo_names"]
    table = result.render_table()
    assert "Portfolio Composer" in table
