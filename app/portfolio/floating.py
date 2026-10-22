"""Floating-equity overlay for COMBINED trade lists.

``run_portfolio_backtest`` merges independently simulated legs by close time, so its equity curve is
realised-only: two trades that were open together and each dipped $800 look like two small, separate
closed losses. Every ``Trade`` records its intratrade extremes (``worst_price`` / ``mfe_price``), so
without the full engine we can still bound what the COMBINED account really went through:

  * lay every trade's [entry_time, exit_time] on one shared clock;
  * at each event time, equity_low  = closed balance + sum over open trades of their worst open P&L (MAE),
                       equity_high = closed balance + sum over open trades of their best open P&L (MFE);
  * trailing peak = running max of equity_high, worst-case drawdown = max(peak - equity_low).

This is a deliberately conservative bound (it assumes every open trade was at its extreme at the same
instant) -- the same bound the unified engine applies inside a bar. Trades without recorded extremes fall
back to the closed-P&L proxy (``app.prop.excursions``) and are counted so the result can disclose it.
For exact answers use the unified engine (``run_unified_portfolio``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.prop.excursions import trade_excursions


@dataclass
class FloatingOverlay:
    closed_max_dd: float
    worst_case_max_dd: float
    closed_max_dd_pct: float
    worst_case_max_dd_pct: float
    max_concurrent: int
    pct_time_overlapping: float
    n_trades: int
    n_measured: int
    n_proxied: int
    worst_equity: float
    curve: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)   # timestamp, closed, low, high

    def to_dict(self) -> dict:
        return {
            "closed_only_max_drawdown": self.closed_max_dd,
            "worst_case_max_drawdown": self.worst_case_max_dd,
            "closed_only_max_drawdown_pct": self.closed_max_dd_pct,
            "worst_case_max_drawdown_pct": self.worst_case_max_dd_pct,
            "drawdown_understatement": self.worst_case_max_dd - self.closed_max_dd,
            "max_concurrent_trades": self.max_concurrent,
            "pct_time_overlapping": self.pct_time_overlapping,
            "trades": self.n_trades,
            "trades_with_measured_excursions": self.n_measured,
            "trades_using_closed_pnl_proxy": self.n_proxied,
            "lowest_floating_equity": self.worst_equity,
        }


def _utc(values) -> np.ndarray:
    s = pd.to_datetime(pd.Series(list(values)))
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.to_numpy(dtype="datetime64[ns]").astype("int64")


def floating_equity_overlay(trades: list, initial_balance: float) -> FloatingOverlay | None:
    n = len(trades)
    if n == 0:
        return None
    ex = trade_excursions(trades)
    mae = np.asarray(ex.mae, dtype=float)
    mfe = np.asarray(ex.mfe, dtype=float)
    pnl = np.array([t.pnl if math.isfinite(t.pnl) else 0.0 for t in trades], dtype=float)
    t_in = _utc([t.entry_time for t in trades])
    t_out = _utc([t.exit_time for t in trades])
    t_out = np.maximum(t_out, t_in)

    # events: (time, kind, idx); entries before exits at the same instant so touching trades count as overlapping
    times = np.unique(np.concatenate([t_in, t_out]))
    order_in = np.argsort(t_in, kind="stable")
    order_out = np.argsort(t_out, kind="stable")
    ip = op = 0
    closed = float(initial_balance)
    open_mae = 0.0
    open_mfe = 0.0
    n_open = 0
    peak = float(initial_balance)
    closed_peak = float(initial_balance)
    closed_dd = worst_dd = 0.0
    max_conc = 0
    overlap_span = 0
    total_span = max(int(times[-1] - times[0]), 1)
    prev_t = times[0]
    lows = np.empty(len(times)); highs = np.empty(len(times)); closeds = np.empty(len(times))
    for k, tt in enumerate(times):
        if n_open > 1:
            overlap_span += int(tt - prev_t)
        prev_t = tt
        while ip < n and t_in[order_in[ip]] <= tt:
            j = order_in[ip]
            open_mae += mae[j]; open_mfe += mfe[j]; n_open += 1
            ip += 1
        low = closed + open_mae
        high = closed + open_mfe
        peak = max(peak, high)
        worst_dd = max(worst_dd, peak - low)
        max_conc = max(max_conc, n_open)
        # trades closing at this instant leave the open set and realise their P&L
        while op < n and t_out[order_out[op]] <= tt:
            j = order_out[op]
            if t_in[j] <= tt:
                open_mae -= mae[j]; open_mfe -= mfe[j]; n_open -= 1
                closed += pnl[j]
                closed_peak = max(closed_peak, closed)
                closed_dd = max(closed_dd, closed_peak - closed)
                peak = max(peak, closed)
            op += 1
        lows[k], highs[k], closeds[k] = low, high, closed
    base = max(float(initial_balance), 1e-9)
    pk_pct = lambda dd: dd / base * 100.0
    curve = pd.DataFrame({"timestamp": pd.to_datetime(times), "closed": closeds, "low": lows, "high": highs})
    return FloatingOverlay(
        closed_max_dd=float(closed_dd), worst_case_max_dd=float(max(worst_dd, closed_dd)),
        closed_max_dd_pct=pk_pct(closed_dd), worst_case_max_dd_pct=pk_pct(max(worst_dd, closed_dd)),
        max_concurrent=int(max_conc), pct_time_overlapping=100.0 * overlap_span / total_span,
        n_trades=n, n_measured=ex.n_measured, n_proxied=ex.n_proxied, worst_equity=float(lows.min()), curve=curve,
    )
