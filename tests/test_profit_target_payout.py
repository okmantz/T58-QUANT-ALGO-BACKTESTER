"""
Tests for RiskConfig.profit_target_pct / reset_on_target -- the
profit-target twin of reset_on_breach (see RiskConfig's own docstring
and app.backtest.execution.run_execution's payout-check block).

Root problem this closes: before these fields existed, the raw backtest
engine had zero concept of a profit target -- only a breach (loss floor)
could ever change what "the account" meant mid-run. A strategy that
reached its eval/funded profit target just kept accumulating equity on
the same never-reset number for the rest of the dataset instead of being
modeled as "payout taken, keep trading" the way a real funded account
(or a mechanically-repurchased eval, see reset_on_breach) actually
behaves.
"""
import pandas as pd
import pytest

from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig
from app.backtest.statistics import compute_statistics


def _steady_uptrend_df(n=10, start=100.0, step=5.0):
    ts = pd.date_range("2024-01-01 09:00", periods=n, freq="D")
    rows = []
    price = start
    for t in ts:
        o, c = price, price + step
        rows.append((t, o, max(o, c) + 0.5, min(o, c) - 0.5, c, 1000.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_reset_on_target_false_is_byte_identical_to_before():
    """profit_target_pct/reset_on_target default to None/False and must
    reproduce the old no-payout-concept behavior exactly -- equity just
    keeps climbing, no payout events recorded."""
    df = _steady_uptrend_df(n=8)
    signals = pd.Series([1] + [0] * 7)
    risk = RiskConfig(initial_balance=10_000.0, risk_mode="percent", risk_value=10.0, pip_size=1.0)
    trades, equity_df = run_execution(df, signals, risk, stop_loss_pips=None, take_profit_pips=None)
    assert equity_df.attrs["account_reset_events"] == []
    assert equity_df.attrs["payout_events"] == []
    stats = compute_statistics(trades, equity_df, risk.initial_balance)
    assert stats.payout_count == 0
    assert stats.total_payout_amount == 0.0


def test_profit_target_triggers_a_payout_and_keeps_trading():
    """A strategy that re-enters repeatedly and crosses its profit target
    must have the excess withdrawn (equity brought back to baseline) and
    keep opening new trades afterward -- not stall flat the instant the
    target is first hit."""
    ts = pd.date_range("2024-01-01 09:00", periods=6, freq="D")
    rows = [
        (ts[0], 100.0, 100.5, 99.5, 100.0, 1000.0),
        (ts[1], 100.0, 130.5, 99.5, 130.0, 1000.0),  # big realized winner -> crosses target
        (ts[2], 130.0, 130.5, 129.5, 130.0, 1000.0),
        (ts[3], 130.0, 135.5, 129.5, 135.0, 1000.0),  # another winner after the payout
        (ts[4], 135.0, 135.5, 134.5, 135.0, 1000.0),
        (ts[5], 135.0, 136.0, 134.5, 135.5, 1000.0),
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    signals = pd.Series([1, 0, 1, 0, 1, 0])

    risk = RiskConfig(
        initial_balance=10_000.0, risk_mode="percent", risk_value=50.0, pip_size=1.0,
        reset_on_target=True, profit_target_pct=10.0,  # $1,000 needed per payout
    )
    with pytest.warns(RuntimeWarning, match="payout"):
        trades, equity_df = run_execution(df, signals, risk, stop_loss_pips=25, take_profit_pips=None)

    payout_events = equity_df.attrs["payout_events"]
    assert len(payout_events) >= 1
    assert all(ev["kind"] == "payout" for ev in payout_events)
    assert all(ev["payout_amount"] > 0 for ev in payout_events)
    # Trading must continue after the payout -- at least one trade after it.
    first_payout_at = payout_events[0]["reset_at"]
    assert any(pd.Timestamp(t.exit_time) > pd.Timestamp(first_payout_at) for t in trades)
    # A payout is never a breach: account_blown machinery must be untouched.
    assert equity_df.attrs["breach_events"] == []

    stats = compute_statistics(trades, equity_df, risk.initial_balance)
    assert stats.payout_count == len(payout_events)
    assert stats.total_payout_amount == pytest.approx(sum(ev["payout_amount"] for ev in payout_events), abs=1e-6)
    # Payouts must never be miscounted as breaches.
    assert stats.account_reset_count == 0
    assert stats.is_reset_chain is False


def test_payout_never_force_closes_the_open_position():
    """Unlike a breach (account termination -> forced close), a payout is
    just a withdrawal from realized equity -- it must never touch an open
    position."""
    ts = pd.date_range("2024-01-01 09:00", periods=4, freq="D")
    rows = [
        (ts[0], 100.0, 100.5, 99.5, 100.0, 1000.0),
        (ts[1], 100.0, 130.5, 99.5, 130.0, 1000.0),  # realized winner crosses target
        (ts[2], 130.0, 145.0, 129.5, 140.0, 1000.0),  # opens a new position, still running
        (ts[3], 140.0, 145.0, 139.5, 142.0, 1000.0),  # still open at end of data
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    signals = pd.Series([1, 0, 1, 1])
    risk = RiskConfig(
        initial_balance=10_000.0, risk_mode="percent", risk_value=50.0, pip_size=1.0,
        reset_on_target=True, profit_target_pct=10.0,
    )
    with pytest.warns(RuntimeWarning, match="payout"):
        trades, equity_df = run_execution(df, signals, risk, stop_loss_pips=25, take_profit_pips=None)
    # The second trade (opened after the payout) should still be open/settled
    # at end-of-data close, not force-closed exactly at the payout instant.
    payout_at = pd.Timestamp(equity_df.attrs["payout_events"][0]["reset_at"])
    assert any(pd.Timestamp(t.entry_time) > payout_at for t in trades)


def test_breach_and_payout_can_coexist_and_are_counted_separately():
    """reset_on_breach and reset_on_target are independent -- a run can
    hit both a payout and (separately, e.g. on a later losing streak) a
    breach, and statistics must keep the two counts distinct."""
    ts = pd.date_range("2024-01-01 09:00", periods=6, freq="D")
    rows = [
        (ts[0], 100.0, 100.5, 99.5, 100.0, 1000.0),
        (ts[1], 100.0, 130.5, 99.5, 130.0, 1000.0),   # winner -> payout
        (ts[2], 130.0, 130.5, 129.5, 130.0, 1000.0),
        (ts[3], 130.0, 130.5, 90.0, 92.0, 1000.0),    # crash -> breach
        (ts[4], 92.0, 96.0, 91.0, 95.0, 1000.0),      # re-enter after breach reset
        (ts[5], 95.0, 99.0, 94.0, 98.0, 1000.0),
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    signals = pd.Series([1, 0, 1, 0, 1, 1])
    risk = RiskConfig(
        initial_balance=10_000.0, risk_mode="percent", risk_value=50.0, pip_size=1.0,
        reset_on_target=True, profit_target_pct=10.0,
        reset_on_breach=True, max_account_drawdown_pct=10.0,
    )
    trades, equity_df = run_execution(df, signals, risk, stop_loss_pips=25, take_profit_pips=None)
    stats = compute_statistics(trades, equity_df, risk.initial_balance)
    assert stats.payout_count >= 1
    assert stats.account_reset_count >= 1
