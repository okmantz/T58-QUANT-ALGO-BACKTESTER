"""Audit item 8 (2026-10-17): the random-entry null runs as columns of one vectorised pass and must give the SAME numbers
the real engine gives -- the fast path is a speed-up, never a different answer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.research.director import random_entry_distribution

try:
    from tests.backtest.test_vectorized_fastpath import _gapless_df
except Exception:  # pragma: no cover
    from backtest.test_vectorized_fastpath import _gapless_df  # type: ignore


def _risk(**kw):
    return RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=0.01, spread_pips=1.0, slippage_pips=0.5,
                      commission_per_trade=0.5, max_trades_per_day=999, **kw)


@pytest.mark.parametrize("metric", ["net_profit", "expectancy", "profit_factor"])
def test_fast_null_equals_engine_null(metric):
    df = _gapless_df(n=500, seed=4)
    kw = dict(n_entries=15, hold_bars=10, stop_loss_pips=20.0, take_profit_pips=30.0, metric=metric, n_seeds=12, seed=3)
    fast = random_entry_distribution(df, _risk(), 0.0, fast=True, **kw)
    slow = random_entry_distribution(df, _risk(), 0.0, fast=False, **kw)
    assert fast["n_seeds"] == slow["n_seeds"]
    assert np.allclose(fast["null"], slow["null"], rtol=1e-9, atol=1e-6)
    assert fast["p_value"] == slow["p_value"]


def test_perbar_atr_exits_also_match():
    df = _gapless_df(n=500, seed=9)
    sl = pd.Series(0.8 + 0.2 * np.sin(np.arange(len(df)) / 20.0))
    kw = dict(n_entries=12, hold_bars=8, stop_loss_distance=sl, take_profit_distance=sl * 2, n_seeds=8, seed=1)
    fast = random_entry_distribution(df, _risk(), 0.0, fast=True, **kw)
    slow = random_entry_distribution(df, _risk(), 0.0, fast=False, **kw)
    assert np.allclose(fast["null"], slow["null"], rtol=1e-9, atol=1e-6)


def test_run_config_the_fast_path_ignores_forces_the_engine():
    from app.research.director import _null_fast_eligible
    df = _gapless_df(n=100)
    assert _null_fast_eligible(df, _risk(), None, None)
    assert not _null_fast_eligible(df, _risk(block_weekend_hold=True), None, None)
    assert not _null_fast_eligible(df, _risk(flatten_time_ct="15:45"), None, None)
    assert not _null_fast_eligible(df, _risk(time_stop_hours=6.0), None, None)
