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


def test_zero_size_contract_floor_warns_instead_of_silently_reporting_zero_trades():
    """REGRESSION TEST (Owen's "T58 Gold Trend Breakout" strategy, Sep
    2026): a strategy risking only 0.6% of a $100k account against GC's
    real contract_size (100 -- $100/point) with an ATR-based stop wide
    enough that one whole contract's risk exceeds that 0.6% budget floors
    EVERY entry to 0 contracts and silently reports 0 trades, with
    nothing telling the user why. The exact same strategy backtested
    WITHOUT contract_size set (continuous/fractional sizing) trades
    completely normally -- this is not a strategy-logic bug, it's a
    risk-value-vs-instrument-lot-size mismatch that the engine must
    surface, not swallow silently."""
    ts = pd.date_range("2024-01-01 09:00", periods=5, freq="D")
    rows = [
        (ts[0], 2000.0, 2010.0, 1990.0, 2005.0, 1000.0),
        (ts[1], 2005.0, 2020.0, 1995.0, 2015.0, 1000.0),
        (ts[2], 2015.0, 2030.0, 2005.0, 2025.0, 1000.0),
        (ts[3], 2025.0, 2040.0, 2015.0, 2035.0, 1000.0),
        (ts[4], 2035.0, 2050.0, 2025.0, 2045.0, 1000.0),
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    signals = pd.Series([1, 0, 1, 0, 1])

    # A wide (88-point) stop with GC's real contract_size at a small 0.6%
    # risk budget on $100k: $600 risk / (88 * $100/pt) << 1 contract.
    risk_realistic = RiskConfig(
        initial_balance=100_000.0, risk_mode="percent", risk_value=0.6,
        pip_size=1.0, contract_size=100.0,
    )
    with pytest.warns(RuntimeWarning, match="rounded DOWN to 0 whole"):
        trades, equity_df = run_execution(df, signals, risk_realistic, stop_loss_pips=88, take_profit_pips=None)
    assert len(trades) == 0
    assert equity_df.attrs["zero_size_contract_floor_count"] > 0

    # The identical strategy WITHOUT contract_size (continuous sizing)
    # must trade normally -- proving the strategy itself is fine and this
    # is purely a whole-contract-realism vs. risk-value mismatch.
    risk_fractional = RiskConfig(
        initial_balance=100_000.0, risk_mode="percent", risk_value=0.6, pip_size=1.0,
    )
    trades2, equity_df2 = run_execution(df, signals, risk_fractional, stop_loss_pips=88, take_profit_pips=None)
    assert len(trades2) > 0
    assert equity_df2.attrs["zero_size_contract_floor_count"] == 0
