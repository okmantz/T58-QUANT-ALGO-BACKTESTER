import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.ensemble.auto_builder import (
    AutoEnsembleError,
    _avg_abs_correlation,
    _Leg,
    build_diversified_ensemble,
    strategy_from_record,
)
from app.strategy.manual import ManualStrategy


def _choppy_df(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    price = 100.0
    rows = []
    for i in range(n):
        step = rng.normal(0, 0.5)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.2))
        l = min(o, c) - abs(rng.normal(0, 0.2))
        rows.append((ts[i], o, h, l, c, 1000.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_cross_config(fast=5, slow=15):
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


def _rsi_reversion_config(period=14, low=30, high=70):
    return {
        "name": f"rsi_{period}",
        "indicators": [{"type": "rsi", "period": period, "column": "close", "as": "rsi_val"}],
        "long_entry": f"rsi_val < {low}",
        "long_exit": f"rsi_val > 50",
        "short_entry": f"rsi_val > {high}",
        "short_exit": f"rsi_val < 50",
        "stop_loss_pips": 15,
        "take_profit_pips": 15,
    }


def _donchian_breakout_config(period=20):
    return {
        "name": f"donchian_{period}",
        "indicators": [
            {"type": "donchian", "period": period, "column": "close", "as": "dc"},
        ],
        "long_entry": "close > dc",
        "long_exit": "close < dc",
        "short_entry": "close < dc",
        "short_exit": "close > dc",
        "stop_loss_pips": 25,
        "take_profit_pips": 50,
    }


def _record(family, config, score, cid):
    return {
        "candidate_id": cid,
        "family": family,
        "source_type": "manual",
        "config": config,
        "composite_score": score,
    }


def test_strategy_from_record_manual():
    rec = _record("macd_cross_trend", _sma_cross_config(), 0.5, "c1")
    strat = strategy_from_record(rec)
    assert isinstance(strat, ManualStrategy)


def test_strategy_from_record_missing_config_raises():
    with pytest.raises(AutoEnsembleError):
        strategy_from_record({"candidate_id": "c1", "source_type": "manual"})


def test_strategy_from_record_unsupported_source_type_raises():
    with pytest.raises(AutoEnsembleError):
        strategy_from_record({"candidate_id": "c1", "source_type": "made_up", "code_text": "x"})


def test_avg_abs_correlation_identical_series_is_one():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    returns = pd.Series(np.random.default_rng(1).normal(0, 1, 30), index=idx)
    leg_a = _Leg(record={}, name="a", family="f", strategy=None, composite_score=1.0, daily_returns=returns)
    leg_b = _Leg(record={}, name="b", family="f", strategy=None, composite_score=1.0, daily_returns=returns.copy())
    assert _avg_abs_correlation(leg_a, [leg_b]) == pytest.approx(1.0, abs=1e-9)


def test_avg_abs_correlation_too_little_overlap_treated_as_zero():
    idx_a = pd.date_range("2024-01-01", periods=3, freq="D")
    idx_b = pd.date_range("2030-01-01", periods=3, freq="D")
    leg_a = _Leg(record={}, name="a", family="f", strategy=None, composite_score=1.0,
                 daily_returns=pd.Series([1.0, 2.0, 3.0], index=idx_a))
    leg_b = _Leg(record={}, name="b", family="f", strategy=None, composite_score=1.0,
                 daily_returns=pd.Series([1.0, 2.0, 3.0], index=idx_b))
    assert _avg_abs_correlation(leg_a, [leg_b]) == 0.0


def test_build_diversified_ensemble_insufficient_candidates():
    df = _choppy_df()
    records = [_record("macd_cross_trend", _sma_cross_config(), 0.9, "c1")]
    result = build_diversified_ensemble(df, records, RiskConfig(initial_balance=50_000), min_legs=3)
    assert result.status == "insufficient_candidates"
    assert result.notes


def test_build_diversified_ensemble_respects_family_cap_and_builds_basket():
    df = _choppy_df()
    records = [
        # Two near-duplicate trend-following legs -- family cap should keep only the better-scoring one.
        _record("macd_cross_trend", _sma_cross_config(5, 15), 0.9, "trend_a"),
        _record("macd_cross_trend", _sma_cross_config(6, 16), 0.7, "trend_b"),
        # Two near-duplicate mean-reversion legs -- same story.
        _record("rsi_extreme_reversion", _rsi_reversion_config(14), 0.85, "mr_a"),
        _record("rsi_extreme_reversion", _rsi_reversion_config(10), 0.6, "mr_b"),
        # A genuinely different hypothesis (breakout family).
        _record("trend_breakout", _donchian_breakout_config(20), 0.8, "brk_a"),
    ]
    result = build_diversified_ensemble(
        df, records, RiskConfig(initial_balance=50_000),
        min_legs=2, max_legs=5, max_per_family=1, max_pairwise_correlation=0.9,
    )
    assert result.status == "built"
    assert 2 <= len(result.basket_names) <= 3
    # max_per_family=1 -- no two legs in the basket should share a family.
    assert len(result.basket_families) == len(set(result.basket_families))
    # The lower-scoring duplicate from each family should have been rejected by the family cap.
    rejected_ids = {r.candidate_id for r in result.rejected}
    assert "trend_b" in rejected_ids or "mr_b" in rejected_ids
    assert result.portfolio_result is not None
    assert result.portfolio_result.combined_statistics is not None


def test_build_diversified_ensemble_min_legs_validation():
    with pytest.raises(AutoEnsembleError):
        build_diversified_ensemble(_choppy_df(), [], RiskConfig(), min_legs=1)
    with pytest.raises(AutoEnsembleError):
        build_diversified_ensemble(_choppy_df(), [], RiskConfig(), min_legs=4, max_legs=3)
