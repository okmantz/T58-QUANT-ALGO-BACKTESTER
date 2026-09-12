"""
Vectorized Stage 1 fast-path: many candidates, one pass over the bars.

WHY THIS EXISTS
----------------
Search Lab's Stage 1 ("cheap filter") today calls the real bar-by-bar
`run_execution()` once per candidate. Each call is its own Python-level
loop over every bar, allocating a `Trade` and touching several dict/attr
lookups per open position. That is the right, honest way to backtest ONE
strategy -- but at Stage 1 scale (hundreds to thousands of candidates,
each looping the SAME dataframe) the fixed Python-loop overhead is paid
once per candidate instead of once per search.

This module runs a single bar loop that is vectorized ACROSS a batch of
candidates (columns) instead of across time. Per-bar state (position,
direction, stop, take, equity) becomes a length-K numpy array instead of
a handful of Python scalars, and every candidate advances one bar at a
time together. This is the same "loop over time, vectorize over columns"
idea vectorbt/mheloy's VectorBT fork use for their fast paths -- but
implemented here in plain numpy (no vectorbt dependency) for two reasons:

  1. vectorbt's OSS license (Apache 2.0 + Commons Clause) restricts
     selling a product whose value is substantially this software --
     a real consideration for a commercial app like T58, not just a
     style preference.
  2. It keeps this fast path auditable against the exact fill/cost
     logic in app.backtest.execution, instead of trusting a third
     party's semantics to match ours.

SCOPE -- READ BEFORE TRUSTING A NUMBER FROM THIS MODULE
--------------------------------------------------------
This is a Stage 1 filter, not a second execution engine. It intentionally
does NOT implement everything app.backtest.execution.run_execution does:

  - Eligible candidates only: fixed pip stop-loss/take-profit (or none)
    -- no per-bar ATR-style stop_loss_distance/take_profit_distance, no
    trailing stop, no breakeven trigger. See `is_vectorizable()`.
  - No adaptive-risk overlay (Stage 1 never uses one today anyway).
  - No daily-loss-limit forced-close, no account-blown circuit breaker,
    no per-trade loss clamp. (max_trades_per_day IS enforced, matching
    the real engine exactly -- it defaults to 10 on RiskConfig, so
    skipping it would have materially overstated trade frequency for
    the common case, not just an edge case.)
  - Drawdown is computed from REALIZED equity at trade-close events only
    (no intrabar mark-to-market) -- a slightly optimistic proxy, fine for
    ranking candidates against each other, not a substitute for the real
    number.

Any candidate ineligible for this fast path (or that this module errors
on) MUST fall back to the existing scalar `_stage1_task` path -- see
`app.search.batch_runner._stage1_task_batch`. Every survivor of Stage 1,
vectorized or not, is re-validated by the real engine at Stage 2/3 exactly
as before this module existed. Nothing here is ever the final answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.execution import DEFAULT_STOP_PCT_OF_PRICE, Trade
from app.backtest.risk import RiskConfig


@dataclass
class VectorizedCandidate:
    candidate_id: str
    signals: np.ndarray          # int8, shape (n_bars,), values in {-1, 0, 1}
    stop_loss_pips: float | None
    take_profit_pips: float | None


@dataclass
class VectorizedOutcome:
    candidate_id: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame | None = None  # None when zero trades
    scale_mismatch: bool = False
    # True if any entry's fixed-pips stop looked implausible against this
    # instrument's own price level or recent ATR -- the same
    # pip_scale_mismatch / atr_scale_mismatch heuristic
    # app.backtest.execution.run_execution applies, kept here so the
    # fast path doesn't silently drop this diagnostic (see
    # app.search.batch_runner._stage1_task_batch, which surfaces it
    # exactly like the scalar path always has).


def is_vectorizable(strat_result) -> bool:
    """True only for the simplest case this fast path can honestly
    handle: a fixed-pips stop/target (or none at all). Anything with a
    per-bar dynamic distance, a trailing stop, a breakeven trigger, or a
    partial-exit/scale-out config must go through the real engine instead
    -- see module docstring."""
    return (
        strat_result.stop_loss_distance is None
        and strat_result.take_profit_distance is None
        and strat_result.trailing_stop_distance is None
        and strat_result.breakeven_trigger_r is None
        and getattr(strat_result, "partial_exit", None) is None
    )


def _risk_amount_vec(equity: np.ndarray, risk: RiskConfig) -> np.ndarray:
    """Vectorized mirror of RiskConfig.risk_amount. Kept deliberately
    tiny and side-by-side testable against the scalar version --
    see tests/test_vectorized_fastpath.py::test_risk_amount_parity."""
    eq_for_sizing = np.maximum(equity, 0.0)
    if risk.risk_mode == "fixed":
        return np.full_like(eq_for_sizing, max(risk.risk_value, 0.0))
    return np.maximum(eq_for_sizing * (risk.risk_value / 100.0), 0.0)


def _position_size_vec(equity: np.ndarray, sizing_pips: np.ndarray, risk: RiskConfig) -> np.ndarray:
    """Vectorized mirror of RiskConfig.position_size -- see
    tests/test_vectorized_fastpath.py::test_position_size_parity."""
    safe_pips = np.where(sizing_pips > 0, sizing_pips, 10.0)
    stop_distance = safe_pips * risk.pip_size
    risk_amt = _risk_amount_vec(equity, risk)
    units = np.where(stop_distance > 0, risk_amt / stop_distance, 0.0)
    if risk.max_position_size is not None:
        units = np.minimum(units, risk.max_position_size)
    return np.maximum(units, 0.0)


def run_vectorized_batch(
    df: pd.DataFrame,
    candidates: list[VectorizedCandidate],
    risk: RiskConfig,
) -> dict[str, VectorizedOutcome]:
    """
    Runs every candidate in `candidates` against `df` in one pass over
    the bars, vectorized across candidates. Returns a dict keyed by
    candidate_id, each holding a real `Trade` list (identical shape to
    what run_execution() produces) plus a sparse (trade-event-only)
    equity curve suitable for app.backtest.statistics.compute_statistics.

    Every candidate here must have passed `is_vectorizable()` on its own
    StrategyResult -- this function does not check that itself, so
    passing an ineligible candidate silently gives WRONG numbers (a
    fixed-pips-only stop is assumed for every column).
    """
    n = len(df)
    k = len(candidates)
    if k == 0:
        return {}

    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy()

    # Same 14-bar True Range mean app.backtest.execution uses purely as a
    # volatility yardstick for the fixed-pips-vs-instrument sanity check
    # below -- kept in sync deliberately so a strategy that trips this
    # heuristic through the fast path trips it identically through the
    # scalar path (and vice versa).
    _prev_close = np.empty_like(closes)
    _prev_close[0] = np.nan
    _prev_close[1:] = closes[:-1]
    _true_range = np.maximum(
        highs - lows, np.maximum(np.abs(highs - _prev_close), np.abs(lows - _prev_close)),
    )
    _atr = pd.Series(_true_range).rolling(14, min_periods=1).mean().to_numpy()

    sig = np.stack([c.signals for c in candidates], axis=1).astype(np.int8)  # (n, k)
    sl_pips = np.array([c.stop_loss_pips or 0.0 for c in candidates], dtype=np.float64)
    tp_pips = np.array([c.take_profit_pips or 0.0 for c in candidates], dtype=np.float64)

    # Per-day trade-count bookkeeping, mirroring app.backtest.execution's
    # own day-index approach -- max_trades_per_day defaults to 10 on
    # RiskConfig, so this is not an edge case to skip: silently ignoring
    # it would materially overstate trade frequency (and therefore every
    # downstream stat) for the common case, not just an unusual one.
    bar_dates = pd.DatetimeIndex(ts).normalize().to_numpy()
    _unique_days, day_idx = np.unique(bar_dates, return_inverse=True)
    trades_today_count = np.zeros((len(_unique_days), k), dtype=np.int64)

    spread_price = risk.spread_pips * risk.pip_size
    slip_price = risk.slippage_pips * risk.pip_size

    equity = np.full(k, risk.initial_balance, dtype=np.float64)
    in_pos = np.zeros(k, dtype=bool)
    direction = np.zeros(k, dtype=np.int8)
    entry_price = np.zeros(k, dtype=np.float64)
    entry_ts = np.empty(k, dtype=ts.dtype)
    equity_at_entry = np.zeros(k, dtype=np.float64)
    size_arr = np.zeros(k, dtype=np.float64)
    initial_risk = np.zeros(k, dtype=np.float64)
    stop_price = np.full(k, np.nan)
    take_price = np.full(k, np.nan)

    bar_dates = pd.DatetimeIndex(ts).normalize().to_numpy()
    _unique_days, day_idx = np.unique(bar_dates, return_inverse=True)
    trades_today_count = np.zeros((len(_unique_days), k), dtype=np.int64)

    trade_events: list[list[dict]] = [[] for _ in range(k)]
    scale_mismatch = np.zeros(k, dtype=bool)

    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        sig_i = sig[i]

        if in_pos.any():
            active = in_pos
            long_m = active & (direction == 1)
            short_m = active & (direction == -1)

            exit_price = np.full(k, np.nan)
            reason = np.empty(k, dtype=object)

            # Honest gap-through fill: a resting stop the bar gapped
            # straight past fills at the open, not the stop level --
            # mirrors app.backtest.execution's own stop-fill logic.
            stop_hit_long = long_m & ~np.isnan(stop_price) & (l <= stop_price)
            stop_hit_short = short_m & ~np.isnan(stop_price) & (h >= stop_price)
            exit_price[stop_hit_long] = np.minimum(stop_price[stop_hit_long], o)
            exit_price[stop_hit_short] = np.maximum(stop_price[stop_hit_short], o)
            reason[stop_hit_long] = "stop_loss"
            reason[stop_hit_short] = "stop_loss"

            remaining = active & np.isnan(exit_price)
            take_hit_long = remaining & long_m & ~np.isnan(take_price) & (h >= take_price)
            take_hit_short = remaining & short_m & ~np.isnan(take_price) & (l <= take_price)
            exit_price[take_hit_long] = take_price[take_hit_long]
            exit_price[take_hit_short] = take_price[take_hit_short]
            reason[take_hit_long] = "take_profit"
            reason[take_hit_short] = "take_profit"

            remaining2 = active & np.isnan(exit_price)
            signal_exit = remaining2 & (sig_i != direction)
            exit_price[signal_exit] = c
            reason[signal_exit] = "signal"

            exit_mask = active & ~np.isnan(exit_price)
            if exit_mask.any():
                filled = exit_price[exit_mask] - (spread_price + slip_price) * direction[exit_mask]
                pnl = (filled - entry_price[exit_mask]) * size_arr[exit_mask] * direction[exit_mask]
                pnl = pnl - risk.commission_per_trade
                pnl = np.where(np.isfinite(pnl), pnl, 0.0)
                new_equity = equity[exit_mask] + pnl
                idxs = np.nonzero(exit_mask)[0]
                for pos_j, col in enumerate(idxs):
                    entry_eq = equity_at_entry[col]
                    trade_events[col].append(dict(
                        entry_time=pd.Timestamp(entry_ts[col]),
                        exit_time=pd.Timestamp(ts[i]),
                        direction=int(direction[col]),
                        entry_price=float(entry_price[col]),
                        exit_price=float(filled[pos_j]),
                        size=float(size_arr[col]),
                        pnl=float(pnl[pos_j]),
                        pnl_pct=(float(pnl[pos_j]) / entry_eq) * 100 if entry_eq else 0.0,
                        exit_reason=str(reason[col]),
                        commission=risk.commission_per_trade,
                        equity_after=float(new_equity[pos_j]),
                        initial_risk=float(initial_risk[col]) if initial_risk[col] else None,
                    ))
                equity[exit_mask] = new_equity
                in_pos[exit_mask] = False
                direction[exit_mask] = 0
                stop_price[exit_mask] = np.nan
                take_price[exit_mask] = np.nan

        flat = ~in_pos
        under_daily_cap = trades_today_count[day_idx[i]] < risk.max_trades_per_day
        want_entry = flat & under_daily_cap & (sig_i != 0)
        if want_entry.any():
            idxs = np.nonzero(want_entry)[0]
            d = sig_i[idxs].astype(np.int8).astype(np.float64)
            raw_price = c
            entry_fill = raw_price + (spread_price + slip_price) * d

            sl_p = sl_pips[idxs]
            has_fixed_sl = sl_p > 0
            fallback_dist = abs(raw_price) * DEFAULT_STOP_PCT_OF_PRICE
            sl_dist = np.where(has_fixed_sl, sl_p * risk.pip_size, fallback_dist)
            sizing_pips = np.where(
                has_fixed_sl, sl_p,
                (fallback_dist / risk.pip_size) if risk.pip_size else 0.0,
            )

            tp_p = tp_pips[idxs]
            has_tp = tp_p > 0
            tp_dist = tp_p * risk.pip_size

            # Same pip_scale_mismatch / atr_scale_mismatch heuristic as
            # app.backtest.execution: only meaningful for a genuinely
            # fixed-pips stop (not the no-stop-defined fallback).
            if has_fixed_sl.any() and raw_price:
                fixed_dist = sl_p[has_fixed_sl] * risk.pip_size
                price_ratio = np.abs(fixed_dist) / abs(raw_price)
                price_mismatch = (price_ratio < 0.0002) | (price_ratio > 0.25)
                bar_atr = _atr[i]
                if bar_atr and bar_atr > 0:
                    atr_ratio = np.abs(fixed_dist) / bar_atr
                    atr_mismatch = atr_ratio < 0.15
                else:
                    atr_mismatch = np.zeros_like(price_mismatch)
                mismatched_cols = idxs[has_fixed_sl][price_mismatch | atr_mismatch]
                if len(mismatched_cols):
                    scale_mismatch[mismatched_cols] = True

            sized = _position_size_vec(equity[idxs], sizing_pips, risk)
            valid = np.isfinite(sized) & (sized > 0)
            final_idxs = idxs[valid]
            if len(final_idxs):
                fd = d[valid]
                in_pos[final_idxs] = True
                direction[final_idxs] = fd.astype(np.int8)
                entry_price[final_idxs] = entry_fill[valid]
                entry_ts[final_idxs] = ts[i]
                equity_at_entry[final_idxs] = equity[final_idxs]
                size_arr[final_idxs] = sized[valid]
                stop_price[final_idxs] = entry_fill[valid] - fd * sl_dist[valid]
                take_price[final_idxs] = np.where(
                    has_tp[valid], entry_fill[valid] + fd * tp_dist[valid], np.nan,
                )
                initial_risk[final_idxs] = sl_dist[valid]
                trades_today_count[day_idx[i], final_idxs] += 1

    # Close any still-open position at the final bar's close, same as
    # the real engine's end-of-data handling.
    if in_pos.any():
        idxs = np.nonzero(in_pos)[0]
        c = closes[n - 1]
        d = direction[idxs].astype(np.float64)
        filled = c - (spread_price + slip_price) * d
        pnl = (filled - entry_price[idxs]) * size_arr[idxs] * d
        pnl = pnl - risk.commission_per_trade
        pnl = np.where(np.isfinite(pnl), pnl, 0.0)
        new_equity = equity[idxs] + pnl
        for pos_j, col in enumerate(idxs):
            entry_eq = equity_at_entry[col]
            trade_events[col].append(dict(
                entry_time=pd.Timestamp(entry_ts[col]),
                exit_time=pd.Timestamp(ts[n - 1]),
                direction=int(direction[col]),
                entry_price=float(entry_price[col]),
                exit_price=float(filled[pos_j]),
                size=float(size_arr[col]),
                pnl=float(pnl[pos_j]),
                pnl_pct=(float(pnl[pos_j]) / entry_eq) * 100 if entry_eq else 0.0,
                exit_reason="end_of_data",
                commission=risk.commission_per_trade,
                equity_after=float(new_equity[pos_j]),
                initial_risk=float(initial_risk[col]) if initial_risk[col] else None,
            ))
        equity[idxs] = new_equity

    outcomes: dict[str, VectorizedOutcome] = {}
    for col, cand in enumerate(candidates):
        events = trade_events[col]
        trades = [Trade(**e) for e in events]
        mismatch = bool(scale_mismatch[col])
        if not trades:
            outcomes[cand.candidate_id] = VectorizedOutcome(
                cand.candidate_id, trades=[], equity_curve=None, scale_mismatch=mismatch,
            )
            continue
        eq_ts = [pd.Timestamp(ts[0])] + [t.exit_time for t in trades]
        eq_vals = [risk.initial_balance] + [t.equity_after for t in trades]
        curve = pd.DataFrame({"timestamp": eq_ts, "equity": eq_vals})
        outcomes[cand.candidate_id] = VectorizedOutcome(
            cand.candidate_id, trades=trades, equity_curve=curve, scale_mismatch=mismatch,
        )
    return outcomes
