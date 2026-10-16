"""Fill-order test from the accuracy plan: a 1h bar whose 1-minute path hits
the target first books a win; the bar-only path books a loss."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig


def _frames(target_first: bool):
    ts = pd.date_range("2024-01-02 15:00", periods=12, freq="1h")
    px = np.full(12, 100.0)
    df = pd.DataFrame(dict(timestamp=ts, open=px, high=px + 0.2, low=px - 0.2, close=px, volume=1))
    # bar 4 spans both stop (98) and target (102)
    df.loc[4, ["high", "low"]] = [102.5, 97.5]
    m_ts = pd.date_range(ts[4], periods=60, freq="1min")
    mp = np.full(60, 100.0)
    m = pd.DataFrame(dict(timestamp=m_ts, open=mp, high=mp + 0.1, low=mp - 0.1, close=mp, volume=1))
    hi_i, lo_i = (10, 40) if target_first else (40, 10)
    m.loc[hi_i, "high"] = 102.5
    m.loc[lo_i, "low"] = 97.5
    # the 1m extremes must reproduce the 1h bar
    return df, m


def _run(intrabar: bool, target_first: bool):
    df, m = _frames(target_first)
    sig = pd.Series(np.zeros(len(df), dtype=int))
    sig.iloc[1:4] = 1
    risk = RiskConfig(initial_balance=50_000.0, risk_mode="fixed", risk_value=500.0, pip_size=1.0, contract_size=1.0,
                      intrabar_replay=intrabar, max_trades_per_day=20)
    tr, _ = run_execution(df, sig, risk, stop_loss_pips=2.0, take_profit_pips=2.0, intrabar_df=m if intrabar else None)
    return tr


def test_target_first_books_win_with_intrabar_and_loss_without():
    with_ib = _run(True, True)
    without = _run(False, True)
    assert with_ib and without
    assert with_ib[0].pnl > 0 and with_ib[0].exit_reason == "take_profit"
    assert without[0].pnl < 0


def test_stop_first_still_books_loss_with_intrabar():
    tr = _run(True, False)
    assert tr and tr[0].pnl < 0
