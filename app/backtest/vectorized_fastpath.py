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

  - Eligible candidates only: a fixed-pip stop-loss/take-profit (or
    none), OR per-bar stop_loss_distance / take_profit_distance arrays
    in raw price units (e.g. an ATR-multiple stop -- the realistic
    strategy class the old fixed-pips-only gate excluded). Still
    excluded: trailing stops, breakeven triggers, partial exits /
    scale-outs. See `is_vectorizable()`.
  - Fill-lag parity (v5): market entries and signal-driven exits fill at
    the FILL bar's open -- `open[i + entry_fill_lag_bars]`, default 1,
    read via `getattr(risk, "entry_fill_lag_bars", 1)` -- mirroring the
    scalar engine, never at the signal bar's close. Entries decided on
    the last `entry_fill_lag_bars` bars are SKIPPED (the tail has no next
    open to fill at); a signal-exit decided on a tail bar cannot fill
    either, so the position stays open and settles via the end-of-data
    close, exactly like the scalar engine. Resting stop-loss /
    take-profit orders are NOT lagged -- they trigger intrabar, same as
    the scalar engine.
  - No adaptive-risk overlay (Stage 1 never uses one today anyway).
  - Commission parity (C1): every exit charges flat per-trade PLUS
    per-contract x whole contracts closed (commission_per_trade +
    commission_per_contract x size/contract_size), exactly like the
    scalar engine's _settle_exit -- the old flat-only charge
    undercharged every multi-contract position.
  - Daily-loss-limit forced close + day-entry block (C2): the
    floating-adverse mark-to-market check (liquidated AT the floor,
    mirroring the scalar engine post-C3), per-day realized P&L
    bookkeeping keyed off the same days the scalar engine uses
    (signal exits book on the fill bar's day), and the account-blown
    circuit breaker (blown_floor threaded; reset-on-breach default
    honored, halt_on_breach latches).
  - No per-trade loss clamp. (max_trades_per_day IS enforced, matching
    the real engine exactly -- it defaults to 10 on RiskConfig, so
    skipping it would have materially overstated trade frequency for
    the common case, not just an edge case.)
  - Drawdown is computed from REALIZED equity at trade-close events only
    (no intrabar mark-to-market) -- a slightly optimistic proxy, fine for
    ranking candidates against each other, not a substitute for the real
    number.

Documented fill-lag semantic choices (kept identical for every column so
candidates still rank against each other fairly; recorded here so the
scalar engine's own fill-lag implementation can be checked against them):
  - Sizing and the reentry-cooldown clock are evaluated at the DECISION
    bar (the bar whose close produced the signal); only the fill price
    and fill time move to the fill bar. The per-day trade CAP, however,
    is checked and counted at the FILL bar's session day -- matching the
    scalar engine (execution.py: fill_bar_date = day_idx[fill_idx));
    counting it at the decision bar disagrees near the 17:00 CT session
    roll and the cap then binds differently on the two engines.
  - The per-bar stop/target distance consumed at entry is the DECISION
    bar's value (same bar the scalar engine reads it from).
  - The reentry-cooldown clock starts at the DECISION bar of a
    signal-exit (the bar the scalar engine settles it on).

Whole-contract flooring (RiskConfig.contract_size) IS mirrored here --
see _position_size_vec: a trade whose intended risk doesn't reach one
whole contract sizes to 0 whole contracts and is skipped, exactly like
the real engine. (This was the one previously-undocumented divergence;
it is now parity, verified by
tests/test_vectorized_fastpath.py::test_position_size_parity_with_contract_size.)

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

try:
    # v5 B2-3 (session day): the fastpath's max_trades_per_day bookkeeping
    # must key off the SAME session day (17:00 CT roll) as the scalar
    # engine, or the daily cap binds differently near the roll boundary
    # and trade counts diverge. Guarded like execution.py's own import.
    from app.data.trading_day import trading_day as _trading_day
except ImportError:  # pragma: no cover
    _trading_day = None


@dataclass
class VectorizedCandidate:
    candidate_id: str
    signals: np.ndarray          # int8, shape (n_bars,), values in {-1, 0, 1}
    stop_loss_pips: float | None
    take_profit_pips: float | None
    # Per-bar stop/target distances in raw price units, shape (n_bars,)
    # -- e.g. an ATR-multiple stop. A bar whose value is NaN or <= 0 is
    # treated as "no per-bar distance for this bar" and falls back to the
    # fixed-pips fields (then to the no-stop fallback), mirroring
    # StrategyResult.stop_loss_distance / take_profit_distance precedence
    # in app.backtest.execution.run_execution.
    stop_distances: np.ndarray | None = None
    take_distances: np.ndarray | None = None


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


def _is_perbar_distance(x) -> bool:
    """True for array-like per-bar distances (pd.Series / np.ndarray /
    list / tuple). A bare scalar float here would crash the scalar engine
    too (it calls `.values` on it), so anything else is NOT vectorizable
    -- it falls back to the scalar path rather than failing here."""
    return isinstance(x, (pd.Series, np.ndarray, list, tuple))


def is_vectorizable(strat_result) -> bool:
    """True for the shapes this fast path can honestly handle: a
    fixed-pips stop/target (or none at all), and -- since v5 -- per-bar
    stop_loss_distance / take_profit_distance arrays in raw price units
    (e.g. ATR-multiple stops, the realistic strategy class the old
    fixed-pips-only gate excluded). Anything with a trailing stop, a
    breakeven trigger, or a partial-exit/scale-out config must go through
    the real engine instead -- see module docstring."""
    sl = strat_result.stop_loss_distance
    tp = strat_result.take_profit_distance
    dynamic_ok = (sl is None or _is_perbar_distance(sl)) and (tp is None or _is_perbar_distance(tp))
    return (
        dynamic_ok
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
    if risk.contract_size:
        # Parity with RiskConfig.position_size: floor to whole contracts, never round up.
        lots = np.floor(units / risk.contract_size + 1e-9)
        units = np.maximum(lots, 0.0) * risk.contract_size
    return np.maximum(units, 0.0)


# ------------------------------------------------------------------
# ADAPTIVE EVALUATION BUDGET (v5)
# ------------------------------------------------------------------
# Budget rule: every candidate starts with the full n-bar pass, but a
# column stops consuming bars the moment it provably cannot trade again
# -- i.e. it holds no open position AND the loop has passed its last
# non-zero signal bar. Two mechanisms, both EXACT (results are
# bit-identical to a full pass; this is a pure cost saving, never an
# approximation):
#   1. Up-front skip: a candidate with zero non-zero signals can never
#      open a position (entries require sig != 0), so it resolves
#      immediately to an empty outcome without entering the bar loop.
#   2. In-loop compaction: every BUDGET_CHECK_BARS bars, provably-dead
#      columns are compacted out of the working arrays; when no live
#      column remains the loop breaks early.
# What this deliberately does NOT do: performance-based early-kill
# (e.g. "stop evaluating a candidate that looks bad after 30% of the
# data"). That would be an approximation -- a candidate killed early on
# the fast path could disagree with the scalar engine -- and Stage 1's
# contract is to be a faithful filter, not a judge. Cheap candidates
# (silent/exhausted) cost less; expensive ones still get the full,
# honest evaluation.
# ------------------------------------------------------------------
_BUDGET_CHECK_BARS = 256


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
    passing an ineligible candidate silently gives WRONG numbers (only
    fixed-pips or per-bar-array stops/targets are honored for every
    column; trailing stops, breakeven triggers and partial exits are
    silently ignored).

    Fill-lag parity: market entries and signal-driven exits fill at
    `open[i + entry_fill_lag_bars]` (default 1, read via
    `getattr(risk, "entry_fill_lag_bars", 1)`), never at the signal bar's
    close. Entries that cannot fill (tail bars) are skipped.
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

    sig_full = np.stack(
        [np.asarray(c.signals, dtype=np.int8).reshape(n) for c in candidates], axis=1
    )  # (n, k)

    # Per-bar stop/target distance matrices, in raw price units; NaN
    # where the candidate carries no per-bar array. A length mismatch
    # raises here -> the caller (app.search.batch_runner._stage1_task_batch)
    # catches it and falls back to the scalar path for the whole chunk.
    sl_full = np.full((n, k), np.nan)
    tp_full = np.full((n, k), np.nan)
    for col, c in enumerate(candidates):
        if c.stop_distances is not None:
            sl_full[:, col] = np.asarray(c.stop_distances, dtype=float).reshape(n)
        if c.take_distances is not None:
            tp_full[:, col] = np.asarray(c.take_distances, dtype=float).reshape(n)

    # Fill-lag parity with the scalar engine (v5): market entries and
    # signal-driven exits fill at the fill bar's open. Read via getattr
    # so this module also works against a RiskConfig that predates the
    # field (default 1, matching the field's own default).
    fill_lag = getattr(risk, "entry_fill_lag_bars", 1)
    try:
        fill_lag = int(fill_lag)
    except (TypeError, ValueError):
        fill_lag = 1
    fill_lag = max(fill_lag, 0)

    # ---- adaptive budget: up-front exact skip of silent candidates ----
    has_sig = sig_full != 0
    any_sig = has_sig.any(axis=0)
    # Last bar index carrying a non-zero signal per column (-1 = silent).
    rev_pos = has_sig[::-1].argmax(axis=0)
    last_sig_bar_all = np.where(any_sig, (n - 1) - rev_pos, -1)

    outcomes: dict[str, VectorizedOutcome] = {}
    trade_events: list[list[dict]] = [[] for _ in range(k)]
    scale_mismatch = np.zeros(k, dtype=bool)

    live_cols = np.nonzero(any_sig)[0]  # original column indices still in the loop
    for col in np.nonzero(~any_sig)[0]:
        # Exact: zero signals -> zero possible entries -> zero trades,
        # provably. Never enters the bar loop.
        outcomes[candidates[col].candidate_id] = VectorizedOutcome(
            candidates[col].candidate_id, trades=[], equity_curve=None, scale_mismatch=False,
        )
    last_sig_bar = last_sig_bar_all[live_cols]  # working-indexed from here on

    # Per-day trade-count bookkeeping, mirroring app.backtest.execution's
    # own day-index approach -- max_trades_per_day defaults to 10 on
    # RiskConfig, so this is not an edge case to skip: silently ignoring
    # it would materially overstate trade frequency (and therefore every
    # downstream stat) for the common case, not just an unusual one.
    # Indexed by ORIGINAL column (never compacted); working indices map
    # through live_cols.
    # v5 B2-3: key off the SESSION day (17:00 CT roll, naive = UTC),
    # exactly like the scalar engine -- a calendar-day key disagrees with
    # it near the roll boundary and the daily cap then binds differently.
    _bar_ts = pd.DatetimeIndex(ts)
    _ts_tz = getattr(df["timestamp"].dtype, "tz", None)
    if _trading_day is not None:
        _ts_for_days = _bar_ts
        if _ts_tz is not None:
            _ts_for_days = _ts_for_days.tz_localize("UTC").tz_convert(_ts_tz)
        bar_dates = np.array(
            [_trading_day(t, tz="America/Chicago", roll_hour=17) for t in _ts_for_days],
            dtype=object,
        )
    else:
        bar_dates = _bar_ts.normalize().to_numpy()
    _unique_days, day_idx = np.unique(bar_dates, return_inverse=True)
    trades_today_count = np.zeros((len(_unique_days), k), dtype=np.int64)

    spread_price = risk.spread_pips * risk.pip_size
    slip_price = risk.slippage_pips * risk.pip_size

    # C2: daily-loss-limit + account-blown handling, mirroring the scalar
    # engine (app.backtest.execution). daily_limit_amount is None when no
    # daily-loss limit is configured; blown_floor is None when no
    # max-drawdown floor is configured (see RiskConfig.account_blown_floor).
    daily_limit_amount = (
        risk.initial_balance * (risk.daily_loss_limit_pct / 100.0)
        if getattr(risk, "daily_loss_limit_pct", None) is not None
        else None
    )
    blown_floor = risk.account_blown_floor()
    halt_on_breach = bool(getattr(risk, "halt_on_breach", False))

    # C2: per-day REALIZED P&L, mirroring the scalar engine's
    # pnl_today_sum -- feeds both the floating-adverse daily-loss check
    # and the day-entry block below. Original-column-indexed
    # (n_days, k), never compacted, exactly like trades_today_count.
    pnl_today_sum = np.zeros((len(_unique_days), k), dtype=np.float64)

    # Working state arrays below are indexed by WORKING column
    # (0..m-1); live_cols maps working -> original column.
    m = len(live_cols)
    sl_pips = np.array([candidates[c].stop_loss_pips or 0.0 for c in live_cols], dtype=np.float64)
    tp_pips = np.array([candidates[c].take_profit_pips or 0.0 for c in live_cols], dtype=np.float64)
    equity = np.full(m, risk.initial_balance, dtype=np.float64)
    in_pos = np.zeros(m, dtype=bool)
    direction = np.zeros(m, dtype=np.int8)
    # P2-1 parity with app.backtest.execution: per-candidate bar index
    # of the most recent full close, so RiskConfig.reentry_cooldown_bars
    # blocks same-bar reentry here exactly like the scalar engine.
    last_close_bar = np.full(m, -10 ** 9, dtype=np.int64)
    entry_price = np.zeros(m, dtype=np.float64)
    entry_ts = np.empty(m, dtype=ts.dtype)
    equity_at_entry = np.zeros(m, dtype=np.float64)
    size_arr = np.zeros(m, dtype=np.float64)
    initial_risk = np.zeros(m, dtype=np.float64)
    stop_price = np.full(m, np.nan)
    take_price = np.full(m, np.nan)
    # C2: per-column account-blown latch (prop_daily_loss_is_breach, or
    # halt_on_breach after a floor breach) -- blocks all further entries
    # for that column, mirroring the scalar engine. Compacted with the
    # other working-indexed state in _compact below.
    account_blown = np.zeros(m, dtype=bool)

    def _compact(alive: np.ndarray) -> None:
        """Adaptive budget, mechanism 2: drop provably-dead columns from
        every working-indexed array. trade_events / scale_mismatch /
        trades_today_count / pnl_today_sum stay original-indexed and are
        untouched."""
        nonlocal m, live_cols, last_sig_bar
        nonlocal sl_pips, tp_pips, equity, in_pos, direction, last_close_bar
        nonlocal entry_price, entry_ts, equity_at_entry, size_arr
        nonlocal initial_risk, stop_price, take_price, account_blown
        keep = np.nonzero(alive)[0]
        live_cols = live_cols[keep]
        last_sig_bar = last_sig_bar[keep]
        sl_pips = sl_pips[keep]
        tp_pips = tp_pips[keep]
        equity = equity[keep]
        in_pos = in_pos[keep]
        direction = direction[keep]
        last_close_bar = last_close_bar[keep]
        entry_price = entry_price[keep]
        entry_ts = entry_ts[keep]
        equity_at_entry = equity_at_entry[keep]
        size_arr = size_arr[keep]
        initial_risk = initial_risk[keep]
        stop_price = stop_price[keep]
        take_price = take_price[keep]
        account_blown = account_blown[keep]
        m = len(live_cols)

    def _settle_exits(widx: np.ndarray, raw_prices: np.ndarray, reason_strs: np.ndarray,
                      ts_vals: np.ndarray, day_idxs: np.ndarray, close_bar: int) -> None:
        """Shared settlement for every exit path (stop/take, signal,
        daily-loss forced close, account-blown forced close, end of
        data): applies spread/slippage, charges commission, books equity
        AND per-day realized P&L, appends the Trade dicts, and clears the
        working position state. Single code path so a cost/bookkeeping
        fix can never land on one exit type and miss another.

        C1 commission parity with the scalar engine (B2-4): flat
        per-trade charge PLUS per-contract x whole contracts actually
        closed (size / contract_size) -- the old flat-only charge
        undercharged every multi-contract position.

        C2: pnl_today_sum[day, column] is booked on the SAME day the
        scalar engine would book it (signal exits key off the FILL bar's
        day; everything else keys off its own bar's day), so the
        floating-adverse check and the day-entry block see the same
        realized day P&L the scalar engine sees."""
        oidx = live_cols[widx]
        d = direction[widx].astype(np.float64)
        filled = raw_prices - (spread_price + slip_price) * d
        pnl = (filled - entry_price[widx]) * size_arr[widx] * d
        _contracts = size_arr[widx] / risk.contract_size if risk.contract_size else np.zeros(len(widx))
        _commission = risk.commission_per_trade + risk.commission_per_contract * _contracts
        pnl = pnl - _commission
        pnl = np.where(np.isfinite(pnl), pnl, 0.0)
        new_equity = equity[widx] + pnl
        for pos_j, (w, col) in enumerate(zip(widx, oidx)):
            entry_eq = equity_at_entry[w]
            trade_events[col].append(dict(
                entry_time=pd.Timestamp(entry_ts[w]),
                exit_time=pd.Timestamp(ts_vals[pos_j]),
                direction=int(direction[w]),
                entry_price=float(entry_price[w]),
                exit_price=float(filled[pos_j]),
                size=float(size_arr[w]),
                pnl=float(pnl[pos_j]),
                pnl_pct=(float(pnl[pos_j]) / entry_eq) * 100 if entry_eq else 0.0,
                exit_reason=str(reason_strs[pos_j]),
                commission=float(_commission[pos_j]),
                equity_after=float(new_equity[pos_j]),
                initial_risk=float(initial_risk[w]) if initial_risk[w] else None,
            ))
        equity[widx] = new_equity
        pnl_today_sum[day_idxs, oidx] += pnl
        in_pos[widx] = False
        direction[widx] = 0
        last_close_bar[widx] = close_bar  # P2-1: EXEC-002 cooldown parity (decision bar)
        stop_price[widx] = np.nan
        take_price[widx] = np.nan

    for i in range(n):
        if m == 0:
            break  # adaptive budget: every column provably done
        if i % _BUDGET_CHECK_BARS == 0:
            alive = in_pos | (i <= last_sig_bar)
            if not bool(alive.all()):
                _compact(alive)
                if m == 0:
                    break

        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        sig_i = sig_full[i, live_cols]

        if in_pos.any():
            active = in_pos
            long_m = active & (direction == 1)
            short_m = active & (direction == -1)

            exit_price = np.full(m, np.nan)
            exit_ts_i = np.full(m, ts[i], dtype=ts.dtype)
            reason = np.empty(m, dtype=object)
            exit_day = np.full(m, day_idx[i], dtype=np.int64)

            # C2: floating-adverse daily-loss check (mirrors the scalar
            # engine's mark-to-market check): a prop firm's daily-loss
            # floor is monitored on floating equity, so a trade that dips
            # deep underwater intrabar and recovers by the close still
            # gets liquidated. Runs BEFORE stop/take resolution -- the
            # floor liquidation is the first thing that would have fired
            # on this bar. Forced columns are masked out of the stop/take/
            # signal logic below via the np.isnan(exit_price) guards.
            if daily_limit_amount is not None:
                _aw = np.nonzero(active)[0]
                _adv = np.where(direction[_aw] == 1, l, h)
                _float_pnl = (
                    (_adv - entry_price[_aw]) * size_arr[_aw] * direction[_aw].astype(np.float64)
                )
                _day_real = pnl_today_sum[day_idx[i], live_cols[_aw]]
                _dl_breach = (_day_real + _float_pnl) <= -daily_limit_amount
                if _dl_breach.any():
                    _bw = _aw[_dl_breach]
                    _bd = direction[_bw].astype(np.float64)
                    # C3 parity: liquidate AT the floor (the first moment
                    # cumulative day P&L hits the limit), clamped to not
                    # exceed the bar's adverse extreme. day_realized is a
                    # SIGNED sum (negative on a losing day), so the
                    # remaining room to the floor is daily_limit_amount +
                    # day_realized -- see the scalar engine's C3 comment
                    # for the derivation.
                    _safe_size = np.where(size_arr[_bw] > 0, size_arr[_bw], np.nan)
                    _liq_dist = (daily_limit_amount + _day_real[_dl_breach]) / _safe_size
                    _liq_price = entry_price[_bw] - _bd * _liq_dist
                    _liq_price = np.where(
                        np.isnan(_liq_price),
                        _adv[_dl_breach],
                        np.where(
                            _bd == 1,
                            np.maximum(_liq_price, _adv[_dl_breach]),
                            np.minimum(_liq_price, _adv[_dl_breach]),
                        ),
                    )
                    exit_price[_bw] = _liq_price
                    reason[_bw] = "daily_loss_limit_forced_close"
                    if getattr(risk, "prop_daily_loss_is_breach", False):
                        account_blown[_bw] = True

            # Honest gap-through fill: a resting stop the bar gapped
            # straight past fills at the open, not the stop level --
            # mirrors app.backtest.execution's own stop-fill logic.
            # Resting orders are NOT fill-lagged (they trigger intrabar).
            stop_hit_long = long_m & np.isnan(exit_price) & ~np.isnan(stop_price) & (l <= stop_price)
            stop_hit_short = short_m & np.isnan(exit_price) & ~np.isnan(stop_price) & (h >= stop_price)
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

            # Signal-driven exit: a MARKET order -- it fills at the fill
            # bar's open (fill-lag parity with the scalar engine), never
            # at this bar's close. Decided here, at the close that
            # produced the signal; the fill bar's open is already known
            # (full arrays are in memory), so it settles immediately.
            # C2: its day-P&L books on the FILL bar's day, exactly like the
            # scalar engine (which settles signal exits on the fill bar).
            remaining2 = active & np.isnan(exit_price)
            signal_exit = remaining2 & (sig_i != direction)
            exit_j = i + fill_lag
            if exit_j < n:
                exit_price[signal_exit] = opens[exit_j]
                exit_ts_i[signal_exit] = ts[exit_j]
                exit_day[signal_exit] = day_idx[exit_j]
                reason[signal_exit] = "signal"
            # else: tail bar -- no next open exists, so the signal exit
            # cannot fill; the position stays open and settles via the
            # end-of-data close below, exactly like the scalar engine.

            exit_mask = active & ~np.isnan(exit_price)
            if exit_mask.any():
                _settle_exits(
                    np.nonzero(exit_mask)[0],
                    exit_price[exit_mask],
                    reason[exit_mask],
                    exit_ts_i[exit_mask],
                    exit_day[exit_mask],
                    i,
                )

        # C2: account-blown circuit breaker (threaded blown_floor
        # handling): once REALIZED equity crosses the loss floor (or
        # hits zero), the account is terminated -- any still-open
        # position is force-closed at this bar's close, exactly like
        # the scalar engine. Unless halt_on_breach, the engine
        # default (risk.reset_on_breach) starts a fresh account and
        # keeps trading; halt_on_breach=True instead latches
        # account_blown and blocks all further entries.
        # (blown_floor may be None -- build the floor mask conditionally
        # since numpy's & does not short-circuit.)
        _floor_hit = (
            (equity <= blown_floor)
            if blown_floor is not None
            else np.zeros(m, dtype=bool)
        )
        _breached = ~account_blown & ((equity <= 0) | _floor_hit)
        if _breached.any():
            _hw = np.nonzero(_breached)[0]
            _open_hit = _hw[in_pos[_hw]]
            if len(_open_hit):
                _settle_exits(
                    _open_hit,
                    np.full(len(_open_hit), c),
                    np.full(len(_open_hit), "account_blown_forced_close", dtype=object),
                    np.full(len(_open_hit), ts[i], dtype=ts.dtype),
                    np.full(len(_open_hit), day_idx[i], dtype=np.int64),
                    i,
                )
            if halt_on_breach:
                account_blown[_hw] = True
            else:
                equity[_hw] = risk.initial_balance

        flat = ~in_pos
        # v5 B2-3 parity: the cap is checked AND counted at the FILL bar's
        # session day, exactly like the scalar engine (execution.py:
        # fill_bar_date = day_idx[fill_idx]). Checking at the decision
        # bar's day disagrees near the 17:00 CT roll and the cap then
        # binds differently on the two engines.
        _fill_j = i + fill_lag
        _fill_day_idx = day_idx[_fill_j] if _fill_j < n else day_idx[i]
        under_daily_cap = trades_today_count[_fill_day_idx][live_cols] < risk.max_trades_per_day
        # P2-1: mirror the scalar engine's reentry cooldown exactly --
        # with the default reentry_cooldown_bars=1, the close bar itself
        # is blocked and the next bar is the earliest reentry.
        cooldown_ok = (i - last_close_bar) >= risk.reentry_cooldown_bars
        # C2: day-entry block -- no new entries once the day's REALIZED
        # P&L has hit the daily-loss floor, mirroring the scalar engine's
        # daily_limit_breached (keyed off the DECISION bar's session day),
        # and none for a blown (terminated) account.
        if daily_limit_amount is not None:
            _day_realized_now = pnl_today_sum[day_idx[i]][live_cols]
            daily_ok = _day_realized_now > -daily_limit_amount
        else:
            daily_ok = np.ones(m, dtype=bool)
        want_entry = flat & under_daily_cap & cooldown_ok & daily_ok & ~account_blown & (sig_i != 0)
        # Fill-lag parity: an entry decided at bar i fills at the fill
        # bar's open. The last `fill_lag` bars have no next open, so
        # entries decided there are skipped -- they could never fill.
        fill_j = i + fill_lag
        if want_entry.any() and fill_j < n:
            widx = np.nonzero(want_entry)[0]
            oidx = live_cols[widx]
            d = sig_i[widx].astype(np.float64)
            raw_price = opens[fill_j]
            entry_fill = raw_price + (spread_price + slip_price) * d

            # Per-bar distances are read at the DECISION bar (same bar
            # the scalar engine reads them from). Scalar precedence:
            # per-bar distance > fixed pips > %-of-price fallback.
            sl_bar = sl_full[i, oidx]
            sl_bar_valid = np.isfinite(sl_bar) & (sl_bar > 0)
            tp_bar = tp_full[i, oidx]
            tp_bar_valid = np.isfinite(tp_bar) & (tp_bar > 0)

            sl_p = sl_pips[widx]
            has_fixed_sl = sl_p > 0
            fallback_dist = abs(raw_price) * DEFAULT_STOP_PCT_OF_PRICE
            sl_dist = np.where(
                sl_bar_valid, sl_bar,
                np.where(has_fixed_sl, sl_p * risk.pip_size, fallback_dist),
            )
            if risk.pip_size:
                sizing_pips = np.where(
                    sl_bar_valid, sl_bar / risk.pip_size,
                    np.where(has_fixed_sl, sl_p, fallback_dist / risk.pip_size),
                )
            else:
                sizing_pips = np.zeros_like(sl_bar)

            tp_p = tp_pips[widx]
            has_tp = tp_p > 0
            has_any_tp = tp_bar_valid | has_tp
            tp_dist = np.where(tp_bar_valid, tp_bar, tp_p * risk.pip_size)

            # Same pip_scale_mismatch / atr_scale_mismatch heuristic as
            # app.backtest.execution -- which applies it ONLY to genuinely
            # fixed-pips stops (inside its `elif stop_loss_pips:` branch),
            # so per-bar distances and the no-stop fallback are excluded
            # here too.
            fixed_only = has_fixed_sl & ~sl_bar_valid
            if fixed_only.any() and raw_price:
                fidx = np.nonzero(fixed_only)[0]
                fixed_dist = sl_p[fidx] * risk.pip_size
                price_ratio = np.abs(fixed_dist) / abs(raw_price)
                price_mismatch = (price_ratio < 0.0002) | (price_ratio > 0.25)
                bar_atr = _atr[i]
                if bar_atr and bar_atr > 0:
                    atr_ratio = np.abs(fixed_dist) / bar_atr
                    atr_mismatch = atr_ratio < 0.15
                else:
                    atr_mismatch = np.zeros_like(price_mismatch)
                mismatched = widx[fidx][price_mismatch | atr_mismatch]
                if len(mismatched):
                    scale_mismatch[live_cols[mismatched]] = True

            sized = _position_size_vec(equity[widx], sizing_pips, risk)
            valid = np.isfinite(sized) & (sized > 0)
            fw = widx[valid]
            if len(fw):
                fd = d[valid]
                in_pos[fw] = True
                direction[fw] = fd.astype(np.int8)
                entry_price[fw] = entry_fill[valid]
                entry_ts[fw] = ts[fill_j]
                equity_at_entry[fw] = equity[fw]
                size_arr[fw] = sized[valid]
                stop_price[fw] = entry_fill[valid] - fd * sl_dist[valid]
                take_price[fw] = np.where(
                    has_any_tp[valid], entry_fill[valid] + fd * tp_dist[valid], np.nan,
                )
                initial_risk[fw] = sl_dist[valid]
                # v5 B2-3 parity: the per-day trade cap is counted at the
                # FILL bar's session day, exactly like the scalar engine
                # (execution.py increments trades_today_count[fill_bar_date]).
                trades_today_count[_fill_day_idx, live_cols[fw]] += 1

    # Close any still-open position at the final bar's close, same as
    # the real engine's end-of-data handling.
    if in_pos.any():
        widx = np.nonzero(in_pos)[0]
        _settle_exits(
            widx,
            np.full(len(widx), closes[n - 1]),
            np.full(len(widx), "end_of_data", dtype=object),
            np.full(len(widx), ts[n - 1], dtype=ts.dtype),
            np.full(len(widx), day_idx[n - 1], dtype=np.int64),
            n - 1,
        )

    for col, cand in enumerate(candidates):
        if cand.candidate_id in outcomes:
            continue  # adaptive-budget up-front skip already recorded
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
