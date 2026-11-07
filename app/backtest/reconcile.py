"""Canonical execution reconciliation.

``run_execution`` (scalar) is THE reference engine: Quick Optimizer, Full Pipeline, the discovery engine and every report ultimately quote its
trades. The vectorized fast path and the numba screener exist only to *rank* candidates quickly. This module proves they agree with the
reference on identical inputs (signals, fills, costs, slippage, sizing, intrabar behaviour) and reports every trade-level difference, so
a disagreement is a visible bug instead of a silent ranking error. ``canonical_replay`` re-runs any shortlisted candidate on the
reference engine so what is reported is always the canonical result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.execution import run_execution


@dataclass
class ReconcileReport:
    n_scalar: int
    n_fast: int
    n_matched: int
    max_entry_price_diff: float
    max_exit_price_diff: float
    max_pnl_diff: float
    net_pnl_scalar: float
    net_pnl_fast: float
    mismatches: list = field(default_factory=list)
    skipped: str = ""

    @property
    def agrees(self) -> bool:
        return (not self.skipped and self.n_scalar == self.n_fast and self.n_matched == self.n_scalar
                and self.max_pnl_diff <= 0.01 + 1e-9 * max(1.0, abs(self.net_pnl_scalar)))

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["agrees"] = self.agrees
        return d


def _key(t):
    return pd.Timestamp(t.entry_time)


def compare_trades(scalar, fast, price_tol: float = 1e-9) -> ReconcileReport:
    fs = {_key(t): t for t in fast}
    matched, mism = 0, []
    me = mx = mp = 0.0
    for t in scalar:
        f = fs.get(_key(t))
        if f is None:
            mism.append({"entry_time": str(_key(t)), "issue": "missing in fast path"})
            continue
        matched += 1
        de, dx, dp = abs(t.entry_price - f.entry_price), abs(t.exit_price - f.exit_price), abs(t.pnl - f.pnl)
        me, mx, mp = max(me, de), max(mx, dx), max(mp, dp)
        if de > price_tol or dx > price_tol or dp > 0.01:
            mism.append({"entry_time": str(_key(t)), "entry_diff": de, "exit_diff": dx, "pnl_diff": dp,
                         "scalar_exit_reason": getattr(t, "exit_reason", None), "fast_exit_reason": getattr(f, "exit_reason", None)})
    return ReconcileReport(len(scalar), len(fast), matched, me, mx, mp, float(sum(t.pnl for t in scalar)), float(sum(t.pnl for t in fast)), mism[:50])


def reconcile_fastpath(df: pd.DataFrame, signals: pd.Series, risk, stop_loss_pips=None, take_profit_pips=None,
                       stop_loss_distance=None, take_profit_distance=None) -> ReconcileReport:
    """Run one candidate on the scalar reference and on the vectorized fast path, and compare trade by trade."""
    from app.backtest.vectorized_fastpath import VectorizedCandidate, risk_is_vectorizable, run_vectorized_batch
    if not risk_is_vectorizable(risk):
        return ReconcileReport(0, 0, 0, 0, 0, 0, 0, 0, skipped="risk config is not vectorizable (screening falls back to the scalar engine)")
    scalar, _ = run_execution(df, signals, risk, stop_loss_pips, take_profit_pips, stop_loss_distance=stop_loss_distance,
                              take_profit_distance=take_profit_distance)
    cand = VectorizedCandidate("c", signals.to_numpy(dtype=np.int8), stop_loss_pips, take_profit_pips,
                               None if stop_loss_distance is None else np.asarray(stop_loss_distance, dtype=float),
                               None if take_profit_distance is None else np.asarray(take_profit_distance, dtype=float))
    out = run_vectorized_batch(df, [cand], risk)["c"]
    return compare_trades(scalar, out.trades)


def canonical_replay(df, signals, risk, **kw):
    """Re-run a shortlisted candidate on the reference engine. Anything shown to the user (reports, lifecycle, prop sims) should use THIS."""
    return run_execution(df, signals, risk, kw.pop("stop_loss_pips", None), kw.pop("take_profit_pips", None), **kw)
