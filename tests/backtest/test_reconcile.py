"""The canonical-engine reconciliation harness: fast path must reproduce the scalar reference trade for trade."""
import numpy as np
import pandas as pd
import pytest

from app.backtest.reconcile import compare_trades, reconcile_fastpath
from app.backtest.risk import RiskConfig
from tests.backtest.test_vectorized_fastpath import _gapless_df


def _signals(df, seed=1, every=9):
    rng = np.random.default_rng(seed)
    s = np.zeros(len(df), dtype=np.int8)
    s[::every] = rng.choice([-1, 1], size=len(s[::every]))
    return pd.Series(s, index=df.index)


def test_fastpath_matches_scalar_reference_on_gapless_data():
    df = _gapless_df()
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=0.01, spread_pips=1.0)
    rep = reconcile_fastpath(df, _signals(df), risk, stop_loss_pips=60.0, take_profit_pips=120.0)
    assert not rep.skipped and rep.n_scalar > 10
    assert rep.agrees, rep.to_dict()


def test_compare_trades_flags_a_missing_trade():
    from app.backtest.execution import run_execution
    df = _gapless_df()
    tr, _ = run_execution(df, _signals(df), RiskConfig(initial_balance=10_000.0, pip_size=0.01), 60.0, 120.0)
    rep = compare_trades(tr, tr[:-1])
    assert not rep.agrees and rep.mismatches
