"""Regression pins for the exit engine questions raised in the 2026-11-10 audit (intrabar ordering, gaps, slippage)."""
import warnings

import pandas as pd

from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig

FLAT = (100, 100.2, 99.8, 100)


def _run(bars, sl=10, tp=20):
    n = len(bars)
    ts = pd.date_range("2024-01-02 15:00", periods=n, freq="1min")
    df = pd.DataFrame([(t, *b, 1000.0) for t, b in zip(ts, bars)], columns=["timestamp", "open", "high", "low", "close", "volume"])
    sig = [0] * n
    sig[1:n - 1] = [1] * (n - 2)
    risk = RiskConfig(initial_balance=50000.0, risk_mode="fixed", risk_value=100.0, pip_size=1.0, contract_size=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trades, _ = run_execution(df, pd.Series(sig), risk, stop_loss_pips=sl, take_profit_pips=tp)
    return trades[0]


def test_stop_is_assumed_first_when_stop_and_target_share_a_bar():
    t = _run([FLAT, FLAT, FLAT, (100, 125, 85, 100), FLAT, FLAT])
    assert t.exit_reason == "stop_loss" and t.pnl < 0


def test_gap_through_the_stop_fills_at_the_open_not_the_stop_price():
    t = _run([FLAT, FLAT, FLAT, (80, 81, 79, 80), FLAT, FLAT])
    assert t.exit_reason == "stop_loss" and t.exit_price < 90.0          # worse than the stop level 90, near the 80 open


def test_target_fills_at_the_target_and_costs_reduce_the_pnl():
    t = _run([FLAT, FLAT, FLAT, (130, 131, 129, 130), FLAT, FLAT])
    assert t.exit_reason == "take_profit" and abs(t.exit_price - 120.0) < 1e-9
    assert t.entry_price > 100.0                                           # entry paid spread/slippage against the trade
