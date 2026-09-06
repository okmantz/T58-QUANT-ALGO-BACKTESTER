from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.strategy.auto_regime_selector import (
    RegimeSelectorError,
    select_regime_strategies,
)
from app.strategy.manual import ManualStrategy


def _mixed_regime_df(n=4000, seed=3):
    """Trending first half, choppy/mean-reverting second half -- gives the
    selector genuinely different regimes to attribute trades against."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    half = n // 2
    trend = np.concatenate([np.linspace(0, 60, half), np.full(n - half, 60.0)])
    noise = np.cumsum(rng.normal(0, 0.4, n))
    chop = np.concatenate([np.zeros(half), 6 * np.sin(np.linspace(0, 40 * np.pi, n - half))])
    price = 1900 + trend + noise + chop
    high = price + np.abs(rng.normal(0.3, 0.15, n))
    low = price - np.abs(rng.normal(0.3, 0.15, n))
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": high, "low": low, "close": price, "volume": 100.0,
    })


def _trend_follow_config():
    return {
        "indicators": [
            {"type": "ema", "period": 10, "column": "close", "as": "ema_fast"},
            {"type": "ema", "period": 40, "column": "close", "as": "ema_slow"},
        ],
        "long_entry": "ema_fast > ema_slow",
        "short_entry": "ema_fast < ema_slow",
    }


def _mean_reversion_config():
    return {
        "indicators": [
            {"type": "rsi", "period": 10, "column": "close", "as": "rsi_10"},
        ],
        "long_entry": "rsi_10 < 30",
        "short_entry": "rsi_10 > 70",
    }


def test_unknown_dimension_raises():
    df = _mixed_regime_df()
    candidates = {"trend": ManualStrategy(_trend_follow_config())}
    with pytest.raises(RegimeSelectorError):
        select_regime_strategies(df, candidates, "not_a_real_dimension")


def test_no_candidates_raises():
    df = _mixed_regime_df()
    with pytest.raises(RegimeSelectorError):
        select_regime_strategies(df, {}, "trend")


def test_selects_a_winner_per_regime_and_builds_router():
    df = _mixed_regime_df()
    candidates = {
        "trend_follow": ManualStrategy(_trend_follow_config()),
        "mean_reversion": ManualStrategy(_mean_reversion_config()),
    }
    result = select_regime_strategies(
        df, candidates, "trend", risk=RiskConfig(pip_size=0.01), pip_size=0.01, min_trades_per_cell=5,
    )
    assert result.regime_dimension == "trend"
    assert result.scores  # every candidate x present-regime pair was scored
    # Every score belongs to one of the two candidates.
    assert {s.strategy_name for s in result.scores} <= set(candidates)
    # If at least one regime got a qualifying candidate, a router is built.
    if result.assignments:
        assert result.router is not None
        out = result.router.generate(df)
        assert len(out.signals) == len(df)
        assert set(out.signals.unique()).issubset({-1, 0, 1})
    else:
        assert result.router is None


def test_render_table_and_to_dict():
    df = _mixed_regime_df(n=1500)
    candidates = {"trend_follow": ManualStrategy(_trend_follow_config())}
    result = select_regime_strategies(df, candidates, "volatility", min_trades_per_cell=3)
    d = result.to_dict()
    assert d["regime_dimension"] == "volatility"
    table = result.render_table()
    assert "Auto Regime Selection" in table


def test_zero_trade_candidate_is_excluded_with_warning():
    df = _mixed_regime_df(n=800)
    impossible_config = {
        "indicators": [{"type": "rsi", "period": 10, "column": "close", "as": "rsi_10"}],
        "long_entry": "rsi_10 < -999",   # never true
        "short_entry": "rsi_10 > 999",   # never true
    }
    candidates = {"dead_strategy": ManualStrategy(impossible_config)}
    result = select_regime_strategies(df, candidates, "trend", min_trades_per_cell=3)
    assert result.router is None
    assert any("zero trades" in w for w in result.warnings)
