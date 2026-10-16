"""Regression tests for the multi-year trading stall (2026-10-06).

Owen's Full Pipeline run of the v9 Momentum Continuation champion on
ES 1m resampled to 5m (2020-2026) produced a trade chart with thousands
of trades through Sep 2021 and then a flat line for the remaining ~5
years. The run's own log said why: the strategy kept signalling (55,798
later bars) but every entry was blocked -- 55,795 of them by "adaptive
risk zero size".

Two engine defects compounded into that flat line:

1. STALE-DAY LATCH (app.backtest.adaptive_risk / execution): daily
   adaptive-risk triggers (daily_loss_pct / daily_profit_pct) read
   AdaptiveRiskState.day_realized_pnl, which only reset when a trade
   CLOSED on a new day. On any day where no trade closes -- precisely
   what happens once entries are throttled to zero -- the accumulator
   kept its last active day's value forever. If that value had tripped
   the limit-aware preset's daily-profit lock (multiplier 0.0), every
   entry for the rest of the dataset was blocked at exactly x0.0, with
   no possible recovery (no trades -> no closes -> no reset).

2. SUB-CONTRACT SIZES (app.backtest.execution): the adaptive multiplier
   was applied AFTER whole-contract flooring, so a throttled whole-
   contract instrument (contract_size set) either traded fractional
   contracts (not real) or, once the product reached 0, stopped
   trading entirely. Adaptive throttling may shrink size toward the
   1-contract minimum; it must never silently end a multi-year run --
   the hard circuit breakers (daily loss limit, account blow floor)
   are the mechanisms allowed to stop trading outright.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.adaptive_risk import (
    AdaptiveRiskConfig,
    AdaptiveRiskRule,
    AdaptiveRiskState,
)
from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.strategy.manual import ManualStrategy


def _synthetic_df(n=6000, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    walk = np.cumsum(rng.normal(0, 0.3, n))
    close = 100 + walk - 0.15 * (walk - pd.Series(walk).rolling(50, min_periods=1).mean().to_numpy())
    high = close + rng.random(n) * 0.4
    low = close - rng.random(n) * 0.4
    openp = close + rng.normal(0, 0.05, n)
    return pd.DataFrame({"timestamp": ts, "open": openp, "high": high, "low": low, "close": close})


_FREQUENT_TRADER = {
    "name": "RSI flip-flop (trades often, on purpose, for stall testing)",
    "entry_conditions": {
        "long": [{"left": {"type": "rsi", "period": 5}, "operator": "<", "right": {"type": "value", "value": 50}}],
        "short": [{"left": {"type": "rsi", "period": 5}, "operator": ">", "right": {"type": "value", "value": 50}}],
    },
    "exit_conditions": {"long": [], "short": []},
    "risk_management": {
        "stop_type": "atr", "stop_value": 1.0, "target_type": "atr", "target_value": 1.5,
        "opposite_signal_exit": True,
    },
}


# ---------------------------------------------------------------------------
# Defect 1: the daily accumulator belongs to the clock, not the trade ledger
# ---------------------------------------------------------------------------

def test_daily_profit_lock_releases_on_a_new_day_without_any_trade_close():
    cfg = AdaptiveRiskConfig(enabled=True, rules=[
        AdaptiveRiskRule(trigger="daily_profit_pct", threshold=1.0, risk_multiplier=0.0),
    ])
    state = AdaptiveRiskState(initial_balance=10_000.0)
    state.record_trade_close(pnl=150.0, is_new_day=True)   # +1.5% day -> lock trips
    assert state.active_multiplier(cfg) == 0.0
    # Days pass with no trades at all, so no trade ever closes to reset
    # the accumulator. "Today" is still a new day: the lock must release.
    state.begin_new_day()
    assert state.day_realized_pnl == 0.0
    assert state.active_multiplier(cfg) == 1.0


def test_trading_continues_past_a_profit_locked_day():
    """End-to-end form of Owen's flat line: a daily-profit lock that
    trips on day 1 must mute only the REST of day 1, not the remaining
    ~2 months of data."""
    df = _synthetic_df(n=6000, seed=3)
    risk = RiskConfig(initial_balance=5_000.0, risk_mode="percent", risk_value=2.0)
    lock_only = AdaptiveRiskConfig(enabled=True, rules=[
        AdaptiveRiskRule(trigger="daily_profit_pct", threshold=0.5, risk_multiplier=0.0,
                         label="Locked for the day (test)"),
    ])
    result = run_backtest(df, ManualStrategy(_FREQUENT_TRADER), risk, adaptive_risk=lock_only)
    assert result.trades, "expected the frequent trader to trade at all"
    entry_days = {t.entry_time.normalize() for t in result.trades}
    assert len(entry_days) >= 10, (
        f"trading only happened on {len(entry_days)} day(s) -- the daily lock latched "
        "across days instead of releasing each morning"
    )
    last_entry = max(t.entry_time for t in result.trades)
    cutoff = df["timestamp"].iloc[int(len(df) * 0.75)]
    assert last_entry >= cutoff, (
        f"last entry {last_entry} is before the final quarter of the data (stall)"
    )


# ---------------------------------------------------------------------------
# Defect 2: throttling toward (never past) the whole-contract minimum
# ---------------------------------------------------------------------------

def test_throttled_contract_sizes_stay_whole_and_keep_trading():
    df = _synthetic_df(n=6000, seed=11)
    risk = RiskConfig(
        initial_balance=50_000.0, risk_mode="fixed", risk_value=250.0,
        pip_size=1.0, contract_size=50.0,
    )
    heavy_throttle = AdaptiveRiskConfig(enabled=True, rules=[
        AdaptiveRiskRule(trigger="consecutive_losses", threshold=1, risk_multiplier=0.02,
                         label="Heavy throttle after one loss (test)"),
    ])
    result = run_backtest(df, ManualStrategy(_FREQUENT_TRADER), risk, adaptive_risk=heavy_throttle)
    assert result.trades, "expected trades"
    # No fractional contracts on a whole-contract instrument, ever.
    fractional = [t for t in result.trades if abs(t.size % 50.0) > 1e-9]
    assert not fractional, (
        f"{len(fractional)} trade(s) sized in fractional contracts, e.g. size={fractional[0].size}"
    )
    # The throttle really engaged (multiplier < 1 on some trade) but the
    # run was NOT silenced: trades continue into the back half of the data.
    assert any(t.adaptive_risk_multiplier < 1.0 for t in result.trades)
    last_entry = max(t.entry_time for t in result.trades)
    midpoint = df["timestamp"].iloc[len(df) // 2]
    assert last_entry >= midpoint, f"last entry {last_entry} before dataset midpoint (stall)"
