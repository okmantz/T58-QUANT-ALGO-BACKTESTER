"""Tests for app.prop.rolling_evaluation -- Owen's "slide the exact prop
evaluation across every possible start bar" ask."""
import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.prop.rolling_evaluation import run_rolling_evaluation
from app.prop.simulator import PropRules


def _mock_trades(n=400, seed=1, mean=25.0, std=150.0, per_day=3):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2022-01-01")
    trades = []
    for i in range(n):
        t = base + pd.Timedelta(days=i // per_day)
        pnl = float(rng.normal(mean, std))
        trades.append(Trade(
            entry_time=t, exit_time=t, direction=1, entry_price=1.1, exit_price=1.1,
            size=1000, pnl=pnl, pnl_pct=0.1, exit_reason="signal", commission=0, equity_after=0,
        ))
    return trades


def _rules(**overrides):
    base = dict(
        account_size=50_000, evaluation_profit_target_pct=8, daily_loss_limit_pct=5,
        max_drawdown_pct=10, min_trading_days=3, consistency_rule_pct=None, payout_frequency_days=14,
    )
    base.update(overrides)
    return PropRules(**base)


def test_rolling_evaluation_basic_shape():
    trades = _mock_trades(n=600, seed=1, mean=30.0)
    result = run_rolling_evaluation(trades, _rules(), window_trading_days=30)
    assert result.n_windows > 0
    assert result.n_passed + result.n_failed == result.n_windows
    assert 0.0 <= result.pass_rate_pct <= 100.0
    assert 0.0 <= result.first_payout_rate_pct <= 100.0
    # first payout can only happen among windows that passed
    assert result.first_payout_rate_pct <= result.pass_rate_pct + 1e-6


def test_rolling_evaluation_pass_rate_differs_from_single_backtest_verdict():
    """The whole point: a strategy that looks like a clean single-shot
    PASS can still have a mediocre rolling pass rate once you slide the
    window -- this uses a strategy with one big early win followed by
    choppy/negative trades, so the FIRST start bar passes trivially but
    most later start bars (which miss the early win) do not."""
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2022-01-01")
    trades = []
    pnls = [8000.0] + list(rng.normal(-20.0, 80.0, 500))
    for i, pnl in enumerate(pnls):
        t = base + pd.Timedelta(days=i // 3)
        trades.append(Trade(
            entry_time=t, exit_time=t, direction=1, entry_price=1.1, exit_price=1.1,
            size=1000, pnl=float(pnl), pnl_pct=0.1, exit_reason="signal", commission=0, equity_after=0,
        ))
    rules = _rules(evaluation_profit_target_pct=8.0, account_size=50_000)
    result = run_rolling_evaluation(trades, rules, window_trading_days=30)

    # The very first window (which includes the $8,000 head start) should
    # pass; the overall rate across all windows should be far lower --
    # exactly Owen's "a single equity curve hides this" point.
    assert result.windows[0].passed is True
    assert result.pass_rate_pct < 50.0


def test_rolling_evaluation_failure_breakdown_categories_are_known():
    trades = _mock_trades(n=500, seed=2, mean=-5.0, std=200.0)  # rough enough to generate real failures
    result = run_rolling_evaluation(trades, _rules(), window_trading_days=20)
    allowed = {"daily_loss_limit", "max_drawdown", "target_not_reached", "too_many_days"}
    assert set(result.failure_breakdown.keys()) <= allowed
    assert sum(result.failure_breakdown.values()) == result.n_failed


def test_rolling_evaluation_respects_max_windows_cap():
    trades = _mock_trades(n=1500, seed=3, per_day=1)
    result = run_rolling_evaluation(trades, _rules(), window_trading_days=15, max_windows=50)
    assert result.n_windows <= 50


def test_rolling_evaluation_stride_reduces_window_count():
    trades = _mock_trades(n=600, seed=4, per_day=1)
    dense = run_rolling_evaluation(trades, _rules(), window_trading_days=20, stride=1, max_windows=None)
    sparse = run_rolling_evaluation(trades, _rules(), window_trading_days=20, stride=5, max_windows=None)
    assert sparse.n_windows < dense.n_windows


def test_rolling_evaluation_rejects_empty_trades():
    with pytest.raises(ValueError):
        run_rolling_evaluation([], _rules(), window_trading_days=30)


def test_rolling_evaluation_rejects_nonpositive_window():
    trades = _mock_trades(n=50)
    with pytest.raises(ValueError):
        run_rolling_evaluation(trades, _rules(), window_trading_days=0)


def test_rolling_evaluation_render_includes_key_fields():
    trades = _mock_trades(n=400, seed=5)
    result = run_rolling_evaluation(trades, _rules(), window_trading_days=25)
    text = result.render()
    assert "Eval Pass Rate" in text
    assert "Windows Tested" in text
    assert "First Payout Rate" in text


def test_rolling_evaluation_best_and_worst_period_are_populated_with_enough_data():
    trades = _mock_trades(n=1200, seed=6, per_day=2)
    result = run_rolling_evaluation(trades, _rules(), window_trading_days=20, max_windows=500)
    if result.n_windows >= 20:
        assert result.worst_starting_period is not None
        assert result.best_starting_period is not None
