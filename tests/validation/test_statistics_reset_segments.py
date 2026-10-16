"""
FIX (2026-09-18): compute_statistics' drawdown metrics (max_drawdown_pct,
average_drawdown_pct, max_daily/weekly_drawdown_pct, and calmar_ratio,
which divides by max_drawdown_pct) used a single running-max (cummax)
across the WHOLE equity curve. That is correct for a normal, never-reset
account, but as soon as RiskConfig.reset_on_breach (see
app.backtest.execution / app.backtest.risk) causes the raw engine to
mechanically "buy a new account" mid-run, a naive whole-curve cummax
treats the OLD account's peak as still being the new account's high-water
mark -- reporting a "drawdown" that has nothing to do with how much either
actual account itself ever drew down. These tests pin down that
compute_statistics now measures drawdown PER SIMULATED ACCOUNT (segmented
at each entry in equity_curve.attrs["account_reset_events"]), and that a
curve with no reset events at all is completely unaffected (byte-identical
to the pre-fix calculation).
"""
import math

import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.backtest.statistics import compute_statistics


def _dummy_trade(entry_time, exit_time, pnl, equity_after) -> Trade:
    return Trade(
        entry_time=entry_time, exit_time=exit_time, direction=1,
        entry_price=100.0, exit_price=100.0 + pnl, size=1.0,
        pnl=pnl, pnl_pct=0.0, exit_reason="signal", commission=0.0,
        equity_after=equity_after,
    )


def test_drawdown_is_measured_per_account_segment_not_across_a_reset():
    ts = pd.date_range("2024-01-01 00:00", periods=6, freq="h")
    equity_values = [10_000.0, 10_500.0, 10_200.0, 10_000.0, 4_000.0, 6_000.0]
    equity_df = pd.DataFrame({"timestamp": ts, "equity": equity_values})
    # One reset, right before row 3 (the restart back to initial_balance).
    equity_df.attrs["account_reset_events"] = [{
        "reset_at": ts[3], "equity_before_reset": 10_200.0,
        "equity_after_forced_close": 10_200.0, "drawdown_pct": 0.0,
    }]
    trades = [
        _dummy_trade(ts[0], ts[1], 500.0, 10_500.0),
        _dummy_trade(ts[1], ts[2], -300.0, 10_200.0),
        _dummy_trade(ts[3], ts[4], -6_000.0, 4_000.0),
        _dummy_trade(ts[4], ts[5], 2_000.0, 6_000.0),
    ]
    stats = compute_statistics(trades, equity_df, initial_balance=10_000.0)

    # Segment 1's own worst drawdown: (10_200 - 10_500) / 10_500 = -2.857...%
    # Segment 2's own worst drawdown: (4_000 - 10_000) / 10_000 = -60%
    # The correct, segment-aware max is 60% -- NOT the ~61.9% a naive
    # whole-curve cummax would report by carrying segment 1's 10,500 peak
    # forward as segment 2's high-water mark.
    assert stats.max_drawdown_pct == pytest.approx(60.0, abs=1e-6)
    assert stats.account_reset_count == 1


def test_no_reset_events_reproduces_the_original_whole_curve_calculation():
    """No account_reset_events at all (the byte-identical default path,
    covering every existing caller/report today) must compute drawdown
    exactly as before this fix -- a single whole-curve cummax."""
    ts = pd.date_range("2024-01-01 00:00", periods=4, freq="h")
    equity_values = [10_000.0, 10_500.0, 9_000.0, 9_500.0]
    equity_df = pd.DataFrame({"timestamp": ts, "equity": equity_values})
    trades = [
        _dummy_trade(ts[0], ts[1], 500.0, 10_500.0),
        _dummy_trade(ts[1], ts[2], -1_500.0, 9_000.0),
        _dummy_trade(ts[2], ts[3], 500.0, 9_500.0),
    ]
    stats = compute_statistics(trades, equity_df, initial_balance=10_000.0)
    # Whole-curve peak is 10,500; worst trough is 9,000 -> -14.2857...%
    assert stats.max_drawdown_pct == pytest.approx((1500.0 / 10_500.0) * 100, abs=1e-6)
    assert stats.account_reset_count == 0


def test_multiple_resets_each_get_their_own_fresh_high_water_mark():
    ts = pd.date_range("2024-01-01 00:00", periods=6, freq="h")
    equity_values = [10_000.0, 2_000.0, 10_000.0, 1_000.0, 10_000.0, 9_000.0]
    equity_df = pd.DataFrame({"timestamp": ts, "equity": equity_values})
    equity_df.attrs["account_reset_events"] = [
        {"reset_at": ts[2], "equity_before_reset": 2_000.0, "equity_after_forced_close": 2_000.0, "drawdown_pct": 80.0},
        {"reset_at": ts[4], "equity_before_reset": 1_000.0, "equity_after_forced_close": 1_000.0, "drawdown_pct": 90.0},
    ]
    trades = [
        _dummy_trade(ts[0], ts[1], -8_000.0, 2_000.0),
        _dummy_trade(ts[2], ts[3], -9_000.0, 1_000.0),
        _dummy_trade(ts[4], ts[5], -1_000.0, 9_000.0),
    ]
    stats = compute_statistics(trades, equity_df, initial_balance=10_000.0)
    assert stats.account_reset_count == 2
    # Segment 3's own worst drawdown is only 10% (10,000 -> 9,000) -- the
    # worst SEGMENT-LOCAL drawdown across all three is segment 2's 90%.
    assert stats.max_drawdown_pct == pytest.approx(90.0, abs=1e-6)


def test_default_backtest_statistics_reset_count_is_zero():
    """Constructing BacktestStatistics directly (as many existing tests
    across the suite do) without account_reset_count must still work --
    purely additive field with a safe default."""
    from app.backtest.statistics import BacktestStatistics
    stats = BacktestStatistics(
        net_profit=0, gross_profit=0, gross_loss=0, return_pct=0, average_trade=0,
        win_rate=0, loss_rate=0, average_winner=0, average_loser=0,
        largest_winner=0, largest_loser=0,
        max_drawdown=0, max_drawdown_pct=0, average_drawdown_pct=0,
        max_daily_drawdown_pct=0, max_weekly_drawdown_pct=0,
        max_losing_streak=0, max_winning_streak=0,
        profit_factor=0, expectancy=0, average_r=0, risk_reward=0,
        sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, total_trades=0,
    )
    assert stats.account_reset_count == 0
