"""v12 indicators (2026-10-24): time-of-day volatility, range estimators, trend-vs-revert gauges, round numbers, NR bars.

  range_tod_ratio / range_tod_zscore   this bar's high-low range versus the SAME clock minute's range over the previous
                                       ``period`` sessions (relative volume exists; range had no time-of-day baseline)
  parkinson_vol / garman_klass_vol     range-based per-bar volatility estimators (percent), rolling mean over ``period`` bars
  variance_ratio                       Lo-MacKinlay VR(q): ~1 random walk, >1 trending, <1 mean-reverting (``period`` = q)
  autocorr_lag1                        rolling lag-1 autocorrelation of 1-bar log returns, -1..1 (Hurst's less noisy cousin)
  round_dist_es / _nq / _gc / _auto    signed distance from the close to the nearest round number, as a fraction of the grid
                                       spacing (-0.5 .. 0.5): ES every 50 pts, NQ every 100 pts, GC every 10 dollars;
                                       ``_auto`` picks the grid from ``frame.attrs`` (instrument/symbol) or the price level
  nr4 / nr7                            1 when this bar's range is the narrowest of the last 4 / 7 bars (own timeframe)
  daily_nr4 / daily_nr7                1 for every bar of a trading day whose PRIOR COMPLETED day was NR4 / NR7

Lookahead safety: every value at bar t uses bars <= t only. The time-of-day baselines are built from strictly EARLIER
occurrences of the same clock minute (``shift(1)`` inside each minute group); the daily NR flags come from the prior
completed trading day (``_prior_day_map``); nothing is centered or negatively shifted.

``V12_BOUNDED_RANGES`` (exact bounds of the series), ``V12_VALUE_RANGES`` (soft, practical ranges for comparison
thresholds) and ``V12_PERIOD_RANGES`` (sane period windows) are consumed by app.optimize.parameter_space so a GA can
never mutate a threshold or period outside what the indicator can actually produce.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from app.strategy.indicators_v11 import _daily_table, _minute_of_day, _prior_day_map, _ts

V12IndicatorBuilder = Callable[[pd.DataFrame, int, str, int], pd.Series]

_LN2 = float(np.log(2.0))


def _nan(f: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=f.index)


def _p(period, lo: int, hi: int, default: int) -> int:
    try:
        v = int(period)
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v if v > 0 else default))


def _bar_range(f: pd.DataFrame) -> pd.Series:
    return (f["high"].astype(float) - f["low"].astype(float)).clip(lower=0.0)


# ------------------------------------------------------------------------------------ time-of-day normalised range
def _tod_baseline(f: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(range, mean, std) of the same clock minute over the previous ``period`` occurrences (current bar excluded)."""
    rng = _bar_range(f)
    key = _minute_of_day(_ts(f))
    if key.isna().all():
        return rng, _nan(f), _nan(f)
    n = max(3, period // 3)
    grp = rng.groupby(key, sort=False)
    prior = grp.shift(1)
    # rolling over a group's prior values: transform keeps the original order/index
    mean = prior.groupby(key, sort=False).transform(lambda s: s.rolling(period, min_periods=n).mean())
    std = prior.groupby(key, sort=False).transform(lambda s: s.rolling(period, min_periods=n).std())
    return rng, mean, std


def range_tod_ratio(f, p, c, l):
    """High-low range divided by the mean range of the same clock minute over the previous ``period`` sessions."""
    try:
        rng, mean, _ = _tod_baseline(f, _p(p, 5, 60, 20))
        return (rng / mean.replace(0, np.nan)).astype(float)
    except Exception:  # noqa: BLE001 -- a series that cannot be built is NaN (condition False), never a crash
        return _nan(f)


def range_tod_zscore(f, p, c, l):
    """(range - same-minute mean) / same-minute std over the previous ``period`` sessions."""
    try:
        rng, mean, std = _tod_baseline(f, _p(p, 5, 60, 20))
        return ((rng - mean) / std.replace(0, np.nan)).astype(float)
    except Exception:  # noqa: BLE001
        return _nan(f)


# ------------------------------------------------------------------------------------------- range-based volatility
def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = a.astype(float), b.astype(float)
    ok = (a > 0) & (b > 0)
    return pd.Series(np.where(ok, np.log(a.where(ok, 1.0) / b.where(ok, 1.0)), np.nan), index=a.index)


def parkinson_vol(f, p, c, l):
    """Parkinson (1980) volatility, percent per bar: sqrt(mean(ln(H/L)^2) / (4 ln 2)) over ``period`` bars."""
    try:
        n = _p(p, 2, 500, 20)
        hl = _safe_log_ratio(f["high"], f["low"])
        var = (hl ** 2).rolling(n, min_periods=n).mean() / (4.0 * _LN2)
        return (np.sqrt(var.clip(lower=0.0)) * 100.0).astype(float)
    except Exception:  # noqa: BLE001
        return _nan(f)


def garman_klass_vol(f, p, c, l):
    """Garman-Klass (1980) volatility, percent per bar: sqrt(mean(0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2)) over ``period`` bars."""
    try:
        n = _p(p, 2, 500, 20)
        hl = _safe_log_ratio(f["high"], f["low"])
        co = _safe_log_ratio(f["close"], f["open"])
        per_bar = 0.5 * hl ** 2 - (2.0 * _LN2 - 1.0) * co ** 2
        var = per_bar.rolling(n, min_periods=n).mean()
        return (np.sqrt(var.clip(lower=0.0)) * 100.0).astype(float)
    except Exception:  # noqa: BLE001
        return _nan(f)


# ------------------------------------------------------------------------------ trend-versus-revert gauges
def variance_ratio(f, p, c, l):
    """Lo-MacKinlay variance ratio VR(q) = Var(q-bar log return) / (q * Var(1-bar log return)), q = ``period`` (2..30).

    Estimated over a trailing window of ``max(10*q, 60)`` bars (``lookback`` overrides when larger than q). ~1 for a random
    walk, >1 when returns trend (positive autocorrelation), <1 when they mean-revert."""
    try:
        q = _p(p, 2, 30, 5)
        win = max(10 * q, 60)
        try:
            if l is not None and int(l) > q:
                win = int(l)
        except (TypeError, ValueError):
            pass
        px = f["close"].astype(float)
        lp = np.log(px.where(px > 0))
        r1 = lp.diff()
        rq = lp.diff(q)
        v1 = r1.rolling(win, min_periods=win).var()
        vq = rq.rolling(win, min_periods=win).var()
        return (vq / (q * v1.replace(0, np.nan))).astype(float)
    except Exception:  # noqa: BLE001
        return _nan(f)


def autocorr_lag1(f, p, c, l):
    """Rolling lag-1 autocorrelation of 1-bar log returns over ``period`` bars (10..300): >0 trending, <0 mean-reverting."""
    try:
        n = _p(p, 10, 300, 50)
        px = f["close"].astype(float)
        r = np.log(px.where(px > 0)).diff()
        return r.rolling(n, min_periods=n).corr(r.shift(1)).clip(-1.0, 1.0).astype(float)
    except Exception:  # noqa: BLE001
        return _nan(f)


# --------------------------------------------------------------------------------------------- round numbers
ROUND_GRID = {"es": 50.0, "nq": 100.0, "gc": 10.0}


def _auto_grid(f: pd.DataFrame) -> float:
    """Grid for ``round_dist_auto``: from frame.attrs instrument/symbol when it names ES/NQ/GC, else a power of ten
    one decade below the median price (e.g. a ~5000 price -> a grid of 100)."""
    attrs = getattr(f, "attrs", {}) or {}
    name = " ".join(str(attrs.get(k, "")) for k in ("instrument", "symbol", "market", "name")).upper()
    for key, grid in (("MNQ", ROUND_GRID["nq"]), ("NQ", ROUND_GRID["nq"]), ("MES", ROUND_GRID["es"]), ("ES", ROUND_GRID["es"]),
                      ("MGC", ROUND_GRID["gc"]), ("GC", ROUND_GRID["gc"])):
        if key in name:
            return grid
    med = float(pd.to_numeric(f["close"], errors="coerce").median())
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return float(10.0 ** (np.floor(np.log10(med)) - 1))


def _round_dist(f: pd.DataFrame, grid: float, column: str = "close") -> pd.Series:
    col = column if column in f.columns else "close"
    px = f[col].astype(float)
    # (price - nearest multiple of grid) / grid, in [-0.5, 0.5]; 0 = exactly on a round number
    return ((px / grid) - np.round(px / grid)).astype(float)


def round_dist_es(f, p, c, l):
    """ES: signed distance to the nearest 50-point level as a fraction of the 50-point grid (-0.5..0.5)."""
    return _round_dist(f, ROUND_GRID["es"], c)


def round_dist_nq(f, p, c, l):
    """NQ: signed distance to the nearest 100-point level as a fraction of the 100-point grid (-0.5..0.5)."""
    return _round_dist(f, ROUND_GRID["nq"], c)


def round_dist_gc(f, p, c, l):
    """GC: signed distance to the nearest $10 level as a fraction of the $10 grid (-0.5..0.5)."""
    return _round_dist(f, ROUND_GRID["gc"], c)


def round_dist_auto(f, p, c, l):
    """Grid chosen from the frame's instrument (ES/NQ/GC) or its price level; same -0.5..0.5 scale."""
    return _round_dist(f, _auto_grid(f), c)


# -------------------------------------------------------------------------------------------------- NR4 / NR7
def _nr(f: pd.DataFrame, n: int) -> pd.Series:
    rng = _bar_range(f)
    full = rng.rolling(n, min_periods=n).count().eq(n)
    narrowest = rng.rolling(n, min_periods=n).min()
    return (rng <= narrowest).astype(float).where(full, np.nan)


def nr4(f, p, c, l):
    """1 when this bar's high-low range is the narrowest of the last 4 bars (inclusive), else 0."""
    return _nr(f, 4)


def nr7(f, p, c, l):
    """1 when this bar's high-low range is the narrowest of the last 7 bars (inclusive), else 0."""
    return _nr(f, 7)


def _daily_nr(f: pd.DataFrame, n: int) -> pd.Series:
    try:
        d = _daily_table(f)
        rng = (d["high"] - d["low"]).clip(lower=0.0)
        narrowest = rng.rolling(n, min_periods=n).min()
        flag = ((rng <= narrowest) & rng.rolling(n, min_periods=n).count().eq(n)).astype(float)
        return _prior_day_map(f, flag)  # shift(1): today's bars see only the prior COMPLETED day
    except Exception:  # noqa: BLE001
        return _nan(f)


def daily_nr4(f, p, c, l):
    """1 for every bar of a trading day whose prior completed day had the narrowest range of the last 4 days."""
    return _daily_nr(f, 4)


def daily_nr7(f, p, c, l):
    """1 for every bar of a trading day whose prior completed day had the narrowest range of the last 7 days."""
    return _daily_nr(f, 7)


V12_INDICATOR_BUILDERS: dict[str, V12IndicatorBuilder] = {
    "range_tod_ratio": range_tod_ratio, "range_tod_zscore": range_tod_zscore,
    "parkinson_vol": parkinson_vol, "garman_klass_vol": garman_klass_vol,
    "variance_ratio": variance_ratio, "autocorr_lag1": autocorr_lag1,
    "round_dist_es": round_dist_es, "round_dist_nq": round_dist_nq, "round_dist_gc": round_dist_gc,
    "round_dist_auto": round_dist_auto,
    "nr4": nr4, "nr7": nr7, "daily_nr4": daily_nr4, "daily_nr7": daily_nr7,
}
V12_INDICATOR_KINDS: tuple[str, ...] = tuple(V12_INDICATOR_BUILDERS)

# 0/1 flags: the comparison constant is a logical selector, not a tunable threshold (parameter_space freezes it).
V12_FLAGS: tuple[str, ...] = ("nr4", "nr7", "daily_nr4", "daily_nr7")
# Kinds whose builder ignores period/lookback: no period gene.
V12_NO_PERIOD: tuple[str, ...] = ("round_dist_es", "round_dist_nq", "round_dist_gc", "round_dist_auto", *V12_FLAGS)

# Exact bounds of the produced series (merged into indicators.BOUNDED_OSCILLATOR_RANGES).
V12_BOUNDED_RANGES: dict[str, tuple[float, float]] = {
    "autocorr_lag1": (-1.0, 1.0),
    "round_dist_es": (-0.5, 0.5), "round_dist_nq": (-0.5, 0.5), "round_dist_gc": (-0.5, 0.5), "round_dist_auto": (-0.5, 0.5),
}  # the 0/1 NR flags are covered by manual.BOOLEAN_OPERAND_KINDS + parameter_space.FLAG_OPERAND_KINDS
# Practical (soft) ranges for comparison thresholds on the unbounded-but-scaled series.
V12_VALUE_RANGES: dict[str, tuple[float, float]] = {
    "range_tod_ratio": (0.0, 10.0),      # 1.0 = a typical range for that minute of the day
    "range_tod_zscore": (-4.0, 8.0),
    "parkinson_vol": (0.0, 5.0),         # percent per bar
    "garman_klass_vol": (0.0, 5.0),
    "variance_ratio": (0.0, 3.0),        # 1.0 = random walk
}
# Period windows (inclusive) a period gene may take; kinds not listed keep the generic period rule.
V12_PERIOD_RANGES: dict[str, tuple[int, int]] = {
    "range_tod_ratio": (5, 60), "range_tod_zscore": (5, 60),
    "parkinson_vol": (5, 200), "garman_klass_vol": (5, 200),
    "variance_ratio": (2, 30),
    "autocorr_lag1": (10, 300),
}

V12_DESCRIPTIONS: dict[str, str] = {
    "range_tod_ratio": "Bar range divided by the mean range of the same clock minute over the previous N sessions (1.0 = typical).",
    "range_tod_zscore": "Bar range z-score against the same clock minute's range over the previous N sessions.",
    "parkinson_vol": "Parkinson high-low volatility, percent per bar, over N bars.",
    "garman_klass_vol": "Garman-Klass OHLC volatility, percent per bar, over N bars.",
    "variance_ratio": "Lo-MacKinlay variance ratio VR(q): >1 trending, <1 mean-reverting, ~1 random walk (period = q).",
    "autocorr_lag1": "Rolling lag-1 autocorrelation of 1-bar returns: >0 trending, <0 mean-reverting.",
    "round_dist_es": "Signed distance to the nearest 50-point ES level, as a fraction of the grid (-0.5..0.5).",
    "round_dist_nq": "Signed distance to the nearest 100-point NQ level, as a fraction of the grid (-0.5..0.5).",
    "round_dist_gc": "Signed distance to the nearest $10 GC level, as a fraction of the grid (-0.5..0.5).",
    "round_dist_auto": "Distance to the nearest round number, grid chosen from the instrument (ES/NQ/GC) or price level.",
    "nr4": "1 when this bar has the narrowest range of the last 4 bars.",
    "nr7": "1 when this bar has the narrowest range of the last 7 bars.",
    "daily_nr4": "1 on days following an NR4 day (narrowest daily range of the last 4).",
    "daily_nr7": "1 on days following an NR7 day (narrowest daily range of the last 7).",
}
