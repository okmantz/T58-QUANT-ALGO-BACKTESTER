"""
The backtest engine must trade until the market data ends.

Regression tests for the "took a few trades at the start, then never traded
again for years" bug. Every stall path found in the raw engine is covered:
  1. account breach          -> log it, start a fresh account, keep trading
  2. profit target reached   -> log it, bank the payout, keep trading
  3. whole-contract sizing dead-lock (equity just under 1 contract's risk)
  4. adaptive-risk 0.0 multipliers on triggers that can never release
  5. adaptive state leaking from one account into the next
plus the opt-in legacy halt and the stall diagnostic that explains a halt.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskRule
from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig, with_prop_safety_defaults
from app.prop.simulator import PropRules


def _walk_df(n=6000, seed=7, freq="h", start=1000.0, vol=1.5):
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0, vol, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, vol / 2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, vol / 2, n))
    ts = pd.date_range("2020-01-01", periods=n, freq=freq)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": 1000.0})


def _trend_signals(df, fast=10, slow=30):
    f = df["close"].rolling(fast).mean()
    s = df["close"].rolling(slow).mean()
    return pd.Series(np.where(f > s, 1, -1)).where(s.notna(), 0)


def _run(df, risk, sig=None, **kw):
    sig = _trend_signals(df) if sig is None else sig
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trades, eq = run_execution(df, sig, risk, stop_loss_pips=kw.pop("sl", 5), take_profit_pips=kw.pop("tp", 10), **kw)
    return trades, eq, [str(w.message) for w in caught]


def _reckless(**over):
    base = dict(initial_balance=10_000.0, risk_mode="percent", risk_value=6.0, pip_size=1.0,
                max_account_drawdown_pct=10.0)
    base.update(over)
    return RiskConfig(**base)


# 1. breach -----------------------------------------------------------------
def test_default_config_keeps_trading_to_the_last_bar_after_breaches():
    df = _walk_df()
    trades, eq, _ = _run(df, _reckless())
    assert len(eq.attrs["breach_events"]) >= 1, "scenario must actually breach for this test to mean anything"
    last_bar = df["timestamp"].iloc[-1]
    assert last_bar - trades[-1].entry_time < pd.Timedelta(days=5)   # traded right to the end of the data


def test_explicit_reset_on_breach_false_no_longer_halts_the_raw_engine():
    df = _walk_df()
    trades, eq, _ = _run(df, _reckless(reset_on_breach=False))
    assert len(eq.attrs["breach_events"]) >= 1
    assert df["timestamp"].iloc[-1] - trades[-1].entry_time < pd.Timedelta(days=5)


def test_every_reset_is_logged_in_plain_language():
    df = _walk_df()
    _, eq, msgs = _run(df, _reckless())
    log = eq.attrs["account_reset_log"]
    assert len(log) == len(eq.attrs["account_reset_events"]) >= 1
    assert all("trading continues" in line for line in log)
    assert any("ACCOUNT BREACH" in line for line in log)
    assert any("account reset(s) occurred" in m for m in msgs)


def test_halt_on_breach_true_is_the_opt_in_legacy_halt_and_explains_the_stall():
    df = _walk_df()
    trades, eq, msgs = _run(df, _reckless(halt_on_breach=True))
    assert df["timestamp"].iloc[-1] - trades[-1].entry_time > pd.Timedelta(days=30)
    stall = [m for m in msgs if "TRADING STALL" in m]
    assert stall and "engine halted" in stall[0]


# 2. profit target ------------------------------------------------------------
def test_profit_target_is_wired_from_prop_rules_and_trading_continues():
    df = _walk_df(seed=3)
    rules = PropRules(account_size=10_000, evaluation_profit_target_pct=3.0, max_drawdown_pct=50.0, daily_loss_limit_pct=50.0)
    risk = with_prop_safety_defaults(RiskConfig(initial_balance=10_000, risk_value=3.0, pip_size=1.0), rules)
    assert risk.profit_target_pct == 3.0 and risk.reset_on_target is True
    trades, eq, _ = _run(df, risk)
    payouts = eq.attrs["payout_events"]
    assert len(payouts) >= 1
    assert any("PROFIT TARGET REACHED" in line for line in eq.attrs["account_reset_log"])
    assert df["timestamp"].iloc[-1] - trades[-1].entry_time < pd.Timedelta(days=5)


def test_explicit_profit_target_is_never_overridden_by_prop_rules():
    rules = PropRules(account_size=10_000, evaluation_profit_target_pct=8.0)
    risk = with_prop_safety_defaults(RiskConfig(initial_balance=10_000, profit_target_pct=2.5), rules)
    assert risk.profit_target_pct == 2.5


def test_no_target_configured_means_no_payouts_and_no_halt():
    df = _walk_df(seed=3)
    risk = RiskConfig(initial_balance=10_000, risk_value=3.0, pip_size=1.0)  # no rules, no target
    trades, eq, _ = _run(df, risk)
    assert eq.attrs["payout_events"] == []
    assert df["timestamp"].iloc[-1] - trades[-1].entry_time < pd.Timedelta(days=5)


# 3. whole-contract dead-lock -------------------------------------------------
def test_one_contract_minimum_prevents_the_sizing_dead_lock():
    # B2-2 (2026-10-04): entries fill at the NEXT bar's open, so the loss
    # that knocks equity under the 1-contract threshold must come from a
    # gap AFTER the fill bar -- the old 8-bar layout relied on the
    # same-bar-close fill to be inside the position for the gap bar.
    ts = pd.date_range("2024-01-01 09:00", periods=9, freq="D")
    rows = [
        (ts[0], 100.0, 100.5, 99.5, 100.0, 1000.0),   # signal -> fills at bar 1's open
        (ts[1], 100.0, 100.5, 99.5, 100.0, 1000.0),   # entry @ 100.0, stop 80.0
        (ts[2], 75.0, 75.5, 60.0, 75.0, 1000.0),      # gap down THROUGH the stop -> -$1,250
        (ts[3], 76.0, 76.5, 75.5, 76.0, 1000.0),      # rescue: 1 contract (equity 48,750 < 50,000)
        (ts[4], 77.0, 77.5, 76.5, 77.0, 1000.0),
        (ts[5], 78.0, 78.5, 77.5, 78.0, 1000.0),
        (ts[6], 79.0, 79.5, 78.5, 79.0, 1000.0),
        (ts[7], 80.0, 80.5, 79.5, 80.0, 1000.0),
        (ts[8], 81.0, 81.5, 80.5, 81.0, 1000.0),
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    sig = pd.Series([1] * 9)
    risk = RiskConfig(initial_balance=50_000.0, risk_mode="percent", risk_value=2.0, pip_size=1.0, contract_size=50.0)
    trades, eq, msgs = _run(df, sig=sig, risk=risk, sl=20, tp=None)
    assert trades[0].pnl < 0                      # equity is now just under the 1-contract threshold
    assert len(trades) >= 2                       # ...and the run still keeps trading
    assert eq.attrs["min_contract_rescue_count"] >= 1
    assert any("1-contract minimum" in m for m in msgs)


def test_unaffordable_config_still_reports_zero_size_instead_of_being_rescued():
    df = _walk_df(n=500)
    risk = RiskConfig(initial_balance=50_000.0, risk_value=0.1, pip_size=1.0, contract_size=50.0)
    trades, eq, msgs = _run(df, risk, sl=20)
    assert trades == []                           # a fresh account can't afford a contract: config problem, not a stall
    assert eq.attrs["min_contract_rescue_count"] == 0


# 4 + 5. adaptive risk --------------------------------------------------------
def test_adaptive_zero_multiplier_on_a_self_locking_trigger_cannot_freeze_the_run():
    df = _walk_df(seed=11)
    cfg = AdaptiveRiskConfig(enabled=True, rules=[AdaptiveRiskRule(trigger="drawdown_pct", threshold=1.0, risk_multiplier=0.0)])
    risk = RiskConfig(initial_balance=10_000, risk_value=2.0, pip_size=1.0)
    trades, eq, _ = _run(df, risk, adaptive_risk=cfg)
    assert df["timestamp"].iloc[-1] - trades[-1].entry_time < pd.Timedelta(days=5)


def test_adaptive_state_resets_with_the_account():
    from app.backtest.adaptive_risk import AdaptiveRiskState
    st = AdaptiveRiskState(initial_balance=10_000.0)
    st.record_trade_close(-900.0, is_new_day=True)
    st.record_trade_close(-400.0, is_new_day=False)
    assert st.consecutive_losses == 2 and st.current_drawdown_pct() > 10
    st.reset()
    assert st.consecutive_losses == 0 and st.current_drawdown_pct() == 0.0 and st.cumulative_realized_pnl == 0.0


def test_daily_triggers_may_still_be_zero():
    st_cfg = AdaptiveRiskConfig(enabled=True, rules=[AdaptiveRiskRule(trigger="daily_loss_pct", threshold=1.0, risk_multiplier=0.0)])
    from app.backtest.adaptive_risk import AdaptiveRiskState
    st = AdaptiveRiskState(initial_balance=10_000.0)
    st.record_trade_close(-500.0, is_new_day=True)
    assert st.active_multiplier(st_cfg) == 0.0    # releases by itself tomorrow, so 0.0 is safe
