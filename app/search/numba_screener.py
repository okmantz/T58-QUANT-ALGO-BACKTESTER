"""Numba screener for Search Lab's Stage-1 filter.

WHAT IT IS
    A compiled bar loop that walks thousands of candidates' signal series over the same price bars in seconds, in R-multiples
    on a unit-equity account. It exists to answer ONE question cheaply: "is this candidate so clearly dead that running the
    exact Stage-1 engine on it would be wasted compute?"

WHAT IT IS NOT
    It never produces a verdict. It cannot pass anything: a candidate it does not kill goes through the exact path
    (``app.backtest.vectorized_fastpath`` / ``run_backtest``) exactly as before, and every Stage-1 survivor is still
    re-validated at Stages 2-3 by the real engine. vectorbt-style engines cannot model the prop-firm path dependence that
    decides a verdict (trailing drawdown, daily loss, consistency, resets), and neither does this screener: it ignores the
    daily-loss limit, the account-blown/reset logic, whole-contract flooring and equity compounding. Because of those
    omissions its numbers are only used with a safety margin (see ``ScreenThresholds``) and it can only ever REMOVE
    candidates whose unconstrained R-statistics are far below the Stage-1 bar.

SEMANTICS (mirrors app.backtest.vectorized_fastpath / app.backtest.execution for the supported subset)
    * market entry and signal-exit fill at ``open[i + entry_fill_lag_bars]`` (tail entries skipped), +/- spread+slippage;
    * resting stop-loss / take-profit trigger intrabar, stop checked first, gap-through fills at the open for stops;
    * stop/target distance: per-bar array at the decision bar > fixed pips > %-of-price fallback;
    * ``max_trades_per_day`` counted at the FILL bar's session day, ``reentry_cooldown_bars`` from the decision bar;
    * result per trade in R: (price P&L per unit * units - commission) / risk budget, units = (budget - fixed) / (stop + cost).
SUPPORTED: fixed-pip or per-bar stop/target. NOT supported (the candidate is simply not screened): trailing stops,
    breakeven, partial exits, resting orders, time stops, flatten times, weekend-hold block, prop account rules.

Numba is optional (``pip install numba``). Without it ``numba_available()`` is False and ``screen_candidates`` returns None,
so Search Lab runs exactly as it did before.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.backtest.execution import DEFAULT_STOP_PCT_OF_PRICE

try:  # pragma: no cover - exercised implicitly by availability
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # noqa: BLE001 -- numba missing/broken: the screener degrades to "off"
    _HAVE_NUMBA = False

    def njit(*_a, **_k):  # type: ignore[no-redef]
        def deco(fn):
            return fn
        return deco if not (len(_a) == 1 and callable(_a[0])) else _a[0]


def numba_available() -> bool:
    return _HAVE_NUMBA and os.environ.get("T58_DISABLE_NUMBA_SCREENER", "") not in ("1", "true", "True")


# ---------------------------------------------------------------------------------------------------- compiled core
@njit(cache=True, nogil=True)
def _sim_core(opens, highs, lows, closes, day_idx, n_days, sig, sl_fixed, tp_fixed, sl_bar, tp_bar, has_bar,
              fill_lag, side_cost, fallback_pct, budget, fixed_comm, cpc, cs, max_per_day, cooldown,
              rec_r, rec_ei, rec_xi, rec_dir, record):
    """One candidate. Returns (n_trades, sum_r, sum_r2, gross_win_r, gross_loss_r). When ``record`` is True the per-trade
    R, entry bar, exit bar (bar index of the fill) and direction are written to the rec_* arrays (tests)."""
    n = opens.shape[0]
    trades_today = np.zeros(n_days, dtype=np.int64)
    pos = 0
    entry = 0.0
    stop = np.nan
    take = np.nan
    unit_den = 1.0
    entry_bar = 0
    last_close = -1000000000
    nt = 0
    s_r = 0.0
    s_r2 = 0.0
    g_win = 0.0
    g_loss = 0.0
    for i in range(n):
        o = opens[i]
        h = highs[i]
        lo = lows[i]
        if pos != 0:
            ex = np.nan
            xi = i
            if pos == 1:
                if not np.isnan(stop) and lo <= stop:
                    ex = min(stop, o)
                elif not np.isnan(take) and h >= take:
                    ex = take
            else:
                if not np.isnan(stop) and h >= stop:
                    ex = max(stop, o)
                elif not np.isnan(take) and lo <= take:
                    ex = take
            if np.isnan(ex) and sig[i] != pos:
                j = i + fill_lag
                if j < n:
                    ex = opens[j]
                    xi = j
            if not np.isnan(ex):
                filled = ex - side_cost * pos
                per_unit = (filled - entry) * pos
                units = max(budget - fixed_comm, 0.0) / unit_den
                comm = fixed_comm + (cpc * units / cs if cs > 0.0 else 0.0)
                r = (per_unit * units - comm) / budget
                nt += 1
                s_r += r
                s_r2 += r * r
                if r > 0.0:
                    g_win += r
                else:
                    g_loss += -r
                if record:
                    rec_r[nt - 1] = r
                    rec_ei[nt - 1] = entry_bar
                    rec_xi[nt - 1] = xi
                    rec_dir[nt - 1] = pos
                pos = 0
                stop = np.nan
                take = np.nan
                last_close = i
        if pos == 0 and sig[i] != 0 and (i - last_close) >= cooldown:
            j = i + fill_lag
            if j < n:
                fd = day_idx[j]
                if trades_today[fd] < max_per_day:
                    d = 1 if sig[i] > 0 else -1
                    raw = opens[j]
                    fill = raw + side_cost * d
                    sb = sl_bar[i] if has_bar else np.nan
                    if sb > 0.0 and np.isfinite(sb):
                        dist = sb
                    elif sl_fixed > 0.0:
                        dist = sl_fixed
                    else:
                        dist = abs(raw) * fallback_pct
                    tb = tp_bar[i] if has_bar else np.nan
                    if tb > 0.0 and np.isfinite(tb):
                        tdist = tb
                    elif tp_fixed > 0.0:
                        tdist = tp_fixed
                    else:
                        tdist = np.nan
                    if dist > 0.0 and budget - fixed_comm > 0.0:
                        pos = d
                        entry = fill
                        entry_bar = j
                        stop = fill - d * dist
                        take = fill + d * tdist if not np.isnan(tdist) else np.nan
                        unit_den = dist + side_cost
                        trades_today[fd] += 1
    if pos != 0:  # end-of-data close at the last close, like the real engine
        filled = closes[n - 1] - side_cost * pos
        per_unit = (filled - entry) * pos
        units = max(budget - fixed_comm, 0.0) / unit_den
        comm = fixed_comm + (cpc * units / cs if cs > 0.0 else 0.0)
        r = (per_unit * units - comm) / budget
        nt += 1
        s_r += r
        s_r2 += r * r
        if r > 0.0:
            g_win += r
        else:
            g_loss += -r
        if record:
            rec_r[nt - 1] = r
            rec_ei[nt - 1] = entry_bar
            rec_xi[nt - 1] = n - 1
            rec_dir[nt - 1] = pos
    return nt, s_r, s_r2, g_win, g_loss


@njit(cache=True, nogil=True)
def _screen_kernel(opens, highs, lows, closes, day_idx, n_days, sig_mat, sl_fixed, tp_fixed, sl_bar_mat, tp_bar_mat,
                   has_bar, fill_lag, side_cost, fallback_pct, budget, fixed_comm, cpc, cs, max_per_day, cooldown):
    k = sig_mat.shape[0]
    out = np.zeros((k, 5), dtype=np.float64)
    dummy_r = np.zeros(1, dtype=np.float64)
    dummy_i = np.zeros(1, dtype=np.int64)
    dummy_d = np.zeros(1, dtype=np.int64)
    for c in range(k):
        if has_bar[c]:
            slb = sl_bar_mat[c]
            tpb = tp_bar_mat[c]
        else:
            slb = dummy_r
            tpb = dummy_r
        nt, s_r, s_r2, gw, gl = _sim_core(
            opens, highs, lows, closes, day_idx, n_days, sig_mat[c], sl_fixed[c], tp_fixed[c], slb, tpb, has_bar[c],
            fill_lag, side_cost, fallback_pct, budget, fixed_comm, cpc, cs, max_per_day, cooldown,
            dummy_r, dummy_i, dummy_i, dummy_d, False)
        out[c, 0] = nt
        out[c, 1] = s_r
        out[c, 2] = s_r2
        out[c, 3] = gw
        out[c, 4] = gl
    return out


# ------------------------------------------------------------------------------------------------------- python API
@dataclass(frozen=True)
class ScreenThresholds:
    """How far below the Stage-1 bar a candidate must be before it is culled. Deliberately conservative: the screener
    ignores whole-contract flooring, compounding and the daily-loss/blown logic, so R-statistics are only trusted when they
    are bad by a wide margin."""
    min_trades: int = 100                 # the run's Stage-1 min_trades
    trade_count_margin: float = 0.7       # cull when screened trades < margin * min_trades
    pf_kill: float = 0.80                 # cull when R-profit-factor is below this ...
    mean_r_kill: float = -0.03            # ... AND mean R per trade is below this ...
    min_trades_for_pf_kill: int = 30      # ... AND there are at least this many trades


@dataclass
class ScreenResult:
    candidate_id: str
    n_trades: int
    mean_r: float
    profit_factor_r: float
    killed: bool
    reason: str = ""


def _day_index(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Session-day index per bar, keyed exactly like vectorized_fastpath (17:00 CT roll)."""
    from app.backtest.vectorized_fastpath import _trading_day
    ts = pd.DatetimeIndex(df["timestamp"])
    tz = getattr(df["timestamp"].dtype, "tz", None)
    if _trading_day is not None:
        t2 = ts.tz_localize("UTC").tz_convert(tz) if tz is not None else ts
        days = np.array([_trading_day(t, tz="America/Chicago", roll_hour=17) for t in t2], dtype=object)
    else:
        days = ts.normalize().to_numpy()
    _, idx = np.unique(days, return_inverse=True)
    return idx.astype(np.int64), int(idx.max()) + 1 if len(idx) else 1


def screen_supported(risk, strat_result) -> bool:
    """True when this candidate/risk combination is inside the screener's modelled subset."""
    from app.backtest.vectorized_fastpath import is_vectorizable, risk_is_vectorizable
    if not (is_vectorizable(strat_result) and risk_is_vectorizable(risk)):
        return False
    return getattr(strat_result, "trailing_stop_distance", None) is None


def _prep_arrays(df, candidates):
    n = len(df)
    k = len(candidates)
    sig = np.stack([np.asarray(c.signals, dtype=np.int8).reshape(n) for c in candidates], axis=0)
    sl_fixed = np.zeros(k)
    tp_fixed = np.zeros(k)
    has_bar = np.zeros(k, dtype=np.bool_)
    sl_bar = np.full((k, n), np.nan) if any(c.stop_distances is not None or c.take_distances is not None for c in candidates) else np.zeros((k, 1))
    tp_bar = np.full((k, n), np.nan) if sl_bar.shape[1] == n else np.zeros((k, 1))
    return n, k, sig, sl_fixed, tp_fixed, has_bar, sl_bar, tp_bar


def screen_arrays(df: pd.DataFrame, candidates: list, risk, day_idx: np.ndarray | None = None, n_days: int | None = None) -> np.ndarray:
    """Run the compiled kernel; returns a (K, 5) array [n_trades, sum_r, sum_r2, gross_win_r, gross_loss_r]."""
    if not numba_available():
        raise RuntimeError("numba is not available")
    n, k, sig, sl_fixed, tp_fixed, has_bar, sl_bar, tp_bar = _prep_arrays(df, candidates)
    pip = float(risk.pip_size or 0.0)
    for c, cand in enumerate(candidates):
        sl_fixed[c] = float(cand.stop_loss_pips or 0.0) * pip
        tp_fixed[c] = float(cand.take_profit_pips or 0.0) * pip
        if cand.stop_distances is not None:
            sl_bar[c] = np.asarray(cand.stop_distances, dtype=float).reshape(n)
            has_bar[c] = True
        if cand.take_distances is not None:
            tp_bar[c] = np.asarray(cand.take_distances, dtype=float).reshape(n)
            has_bar[c] = True
    if day_idx is None:
        day_idx, n_days = _day_index(df)
    fill_lag = max(int(getattr(risk, "entry_fill_lag_bars", 1) or 0), 0)
    budget = float(risk.risk_amount(risk.initial_balance))
    cs = float(risk.contract_size or 0.0)
    return _screen_kernel(
        df["open"].to_numpy(dtype=np.float64), df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64), df["close"].to_numpy(dtype=np.float64),
        day_idx, int(n_days), sig, sl_fixed, tp_fixed, sl_bar, tp_bar, has_bar,
        fill_lag, float(risk.side_cost_price()), float(DEFAULT_STOP_PCT_OF_PRICE), budget,
        float(risk.commission_per_trade), float(risk.commission_per_contract or 0.0), cs,
        int(risk.max_trades_per_day), int(risk.reentry_cooldown_bars))


def simulate_trades_r(df: pd.DataFrame, signals, risk, stop_loss_pips=None, take_profit_pips=None,
                      stop_distances=None, take_distances=None) -> dict:
    """Single-candidate run that RETURNS the per-trade records (parity tests / debugging)."""
    if not numba_available():
        raise RuntimeError("numba is not available")
    n = len(df)
    day_idx, n_days = _day_index(df)
    pip = float(risk.pip_size or 0.0)
    has_bar = stop_distances is not None or take_distances is not None
    slb = np.asarray(stop_distances, dtype=float).reshape(n) if stop_distances is not None else np.full(n, np.nan)
    tpb = np.asarray(take_distances, dtype=float).reshape(n) if take_distances is not None else np.full(n, np.nan)
    cap = n + 2
    rec_r = np.zeros(cap)
    rec_ei = np.zeros(cap, dtype=np.int64)
    rec_xi = np.zeros(cap, dtype=np.int64)
    rec_dir = np.zeros(cap, dtype=np.int64)
    nt, s_r, s_r2, gw, gl = _sim_core(
        df["open"].to_numpy(dtype=np.float64), df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64), df["close"].to_numpy(dtype=np.float64),
        day_idx, int(n_days), np.asarray(signals, dtype=np.int8).reshape(n),
        float(stop_loss_pips or 0.0) * pip, float(take_profit_pips or 0.0) * pip, slb, tpb, bool(has_bar),
        max(int(getattr(risk, "entry_fill_lag_bars", 1) or 0), 0), float(risk.side_cost_price()),
        float(DEFAULT_STOP_PCT_OF_PRICE), float(risk.risk_amount(risk.initial_balance)),
        float(risk.commission_per_trade), float(risk.commission_per_contract or 0.0), float(risk.contract_size or 0.0),
        int(risk.max_trades_per_day), int(risk.reentry_cooldown_bars), rec_r, rec_ei, rec_xi, rec_dir, True)
    nt = int(nt)
    return {"n_trades": nt, "r": rec_r[:nt].copy(), "entry_bar": rec_ei[:nt].copy(), "exit_bar": rec_xi[:nt].copy(),
            "direction": rec_dir[:nt].copy(), "sum_r": float(s_r), "gross_win_r": float(gw), "gross_loss_r": float(gl)}


def judge(candidate_id: str, row: np.ndarray, th: ScreenThresholds) -> ScreenResult:
    """Turn one kernel row into a keep/kill decision. A kill is only ever 'this is far below the Stage-1 bar'."""
    n = int(row[0])
    mean_r = float(row[1]) / n if n else 0.0
    gl = float(row[4])
    pf = (float(row[3]) / gl) if gl > 0 else (float("inf") if row[3] > 0 else 0.0)
    if n < th.trade_count_margin * th.min_trades:
        return ScreenResult(candidate_id, n, mean_r, pf, True,
                            f"screened: {n} trades is far below the Stage-1 minimum ({th.min_trades})")
    if n >= th.min_trades_for_pf_kill and pf < th.pf_kill and mean_r < th.mean_r_kill:
        return ScreenResult(candidate_id, n, mean_r, pf, True,
                            f"screened: profit factor {pf:.2f} and mean {mean_r:+.3f}R/trade over {n} trades are far below the Stage-1 bar")
    return ScreenResult(candidate_id, n, mean_r, pf, False)


def screen_candidates(df: pd.DataFrame, candidates: list, risk, th: ScreenThresholds) -> dict[str, ScreenResult] | None:
    """Screen a batch. Returns {candidate_id: ScreenResult}, or None when screening is unavailable/unsafe (numba missing,
    unsupported risk config, any error) -- the caller then runs every candidate through the exact path as usual."""
    if not candidates or not numba_available():
        return None
    try:
        from app.backtest.vectorized_fastpath import risk_is_vectorizable
        if not risk_is_vectorizable(risk) or not math.isfinite(float(risk.risk_amount(risk.initial_balance))):
            return None
        if float(risk.risk_amount(risk.initial_balance)) <= 0:
            return None
        rows = screen_arrays(df, candidates, risk)
        return {c.candidate_id: judge(c.candidate_id, rows[i], th) for i, c in enumerate(candidates)}
    except Exception:  # noqa: BLE001 -- the screener must never break a run
        return None
