"""
v7 indicator extensions (worker D, 2026-10-05).

New lookahead-safe indicator kinds for the discovery grammar, covering the
Workstream-C list that had no builder yet at HEAD 9a98914:

    stochrsi_k / stochrsi_d   StochRSI (RSI of RSI, 0-100 bounded)
    plus_di / minus_di         Wilder's +DI / -DI (the directional halves of
                               the existing adx() -- same +DM/-DM/TR math)
    linreg_slope               rolling least-squares slope of `column`
                               (price units per bar)
    hurst_exponent             rolling R/S Hurst exponent (0-1; 0.5 = random
                               walk, >0.5 trending, <0.5 mean-reverting)
    kst                        Know Sure Thing (Pring's weighted ROC sum)
    coppock                     Coppock Curve (WMA10 of ROC14 + ROC11)

Lookahead safety: every series is built only from the current and past
bars -- rolling windows, ewm with adjust=False, and .shift(+n) (past bars)
only. No negative shifts, no centered windows, no whole-frame aggregates.
Each is covered by tests/test_indicators_v7.py (reference-value agreement
on synthetic data + the repo's behavioral lookahead check,
app.strategy.lookahead_check, via a Manual strategy that trades on the
kind).

Wiring: app.strategy.indicators._build_indicator_series_uncached falls back
to V7_INDICATOR_BUILDERS here for any kind it doesn't know (clearly-marked
v7 hook), and app.strategy.manual._series_from_operand routes the new kind
names into build_indicator_series -- so Manual/evolution/Search-Lab configs
can reference these by name exactly like the built-ins.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from app.strategy.indicators import _period, _wilder_smooth, roc, rsi, sma, true_range, wma


def stochrsi(frame: pd.DataFrame, period: int = 14, column: str = "close",
             smooth_k: int = 3, smooth_d: int = 3) -> tuple[pd.Series, pd.Series]:
    """StochRSI (Tushar Chande / Stanley Kroll).

    Applies the stochastic formula to RSI instead of price: where the RSI
    sits inside its own trailing `period`-bar range, 0-100. %K is the
    SMA(`smooth_k`) of that raw value, %D the SMA(`smooth_d`) of %K.
    All windows are trailing-only (causal).
    """
    p = _period(period)
    source = frame[column] if column in frame.columns else frame["close"]
    rsi_s = rsi(source, p)
    lowest = rsi_s.rolling(p, min_periods=p).min()
    highest = rsi_s.rolling(p, min_periods=p).max()
    denom = (highest - lowest).replace(0, np.nan)
    raw = 100 * (rsi_s - lowest) / denom
    # Floating point can push a value a hair outside [0, 100] (e.g. when
    # rsi == highest but the division rounds up); StochRSI is definitionally
    # bounded, so clip -- this also keeps the grammar's 0-100 threshold
    # bounds honest.
    raw = raw.clip(0, 100)
    k = sma(raw.fillna(50.0), smooth_k).clip(0, 100)
    d = sma(k, smooth_d).clip(0, 100)
    return k, d


def _directional_movement(frame: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
    """Wilder's +DI / -DI, transcribed from the same +DM/-DM/TR math as the
    repo's adx() so the three agree by construction: +DM/-DM from
    consecutive high/low deltas (each zeroed unless positive AND larger
    than the other side), Wilder-smoothed and normalized by smoothed True
    Range, times 100. Trailing-only (causal)."""
    p = _period(period)
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=frame.index)
    tr_smooth = _wilder_smooth(true_range(frame), p)
    plus_di = 100 * _wilder_smooth(plus_dm, p) / tr_smooth.replace(0, np.nan)
    minus_di = 100 * _wilder_smooth(minus_dm, p) / tr_smooth.replace(0, np.nan)
    return plus_di, minus_di


def plus_di(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's +DI (0-100): bullish directional movement strength."""
    return _directional_movement(frame, period)[0]


def minus_di(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's -DI (0-100): bearish directional movement strength."""
    return _directional_movement(frame, period)[1]


def linreg_slope(series: pd.Series, period: int = 14) -> pd.Series:
    """Rolling least-squares slope of `series` over a trailing `period`-bar
    window, in series units per bar. Positive = rising, negative = falling.

    Vectorized form of the textbook fit: for window positions j = 0..N-1
    and values y_j, slope = (N*sum(j*y_j) - sum(j)*sum(y_j)) /
    (N*sum(j^2) - sum(j)^2). With global index g and window start s =
    g - N + 1, sum(j*y_j) = sum(g*y) - s*sum(y), all three sums being plain
    trailing rolling sums -- so each bar's slope uses only bars at or
    before it. Verified term-by-term against np.polyfit in
    tests/test_indicators_v7.py.
    """
    n = _period(period)
    y = pd.Series(series).astype(float)
    g = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    sum_y = y.rolling(n, min_periods=n).sum()
    sum_gy = (g * y).rolling(n, min_periods=n).sum()
    s = g - n + 1  # window start (global index) for each bar
    sum_j = n * (n - 1) / 2.0
    sum_j2 = (n - 1) * n * (2 * n - 1) / 6.0
    sum_jy = sum_gy - s * sum_y
    denom = n * sum_j2 - sum_j * sum_j
    slope = (n * sum_jy - sum_j * sum_y) / denom
    return slope


def hurst_exponent(series: pd.Series, period: int = 100) -> pd.Series:
    """Rolling Hurst exponent via the rescaled-range (R/S) method.

    0.5 ~= random walk, >0.5 trending/persistent, <0.5 mean-reverting/
    anti-persistent.

    The R/S statistic is computed on the series' first DIFFERENCES
    (returns), not on the levels -- this is the standard construction
    (cf. the `hurst` package's R/S estimator): price levels are an
    integrated process, and R/S applied to levels of a random walk
    converges to 1.0, not 0.5, which would make the indicator useless.
    On differences, a random walk's increments are white noise and the
    estimator centers correctly at ~0.5.

    Per trailing `period`-bar window of differences w (length m): demean,
    Z = cumsum(w - mean(w)), R = max(Z) - min(Z), S = sample std of w;
    H = log(R/S) / log(m). Each window uses only past bars (causal).
    Degenerate windows (zero variance) yield NaN rather than a fabricated
    0.5 -- documented, and asserted in tests.
    """
    n = _period(period)
    diffs = pd.Series(series).astype(float).diff()

    def _rs(w: np.ndarray) -> float:
        if np.isnan(w).any():
            return np.nan
        m = len(w)
        dev = w - w.mean()
        s = w.std(ddof=1)
        if s == 0 or not np.isfinite(s):
            return np.nan
        z = np.cumsum(dev)
        r = z.max() - z.min()
        if r <= 0 or not np.isfinite(r):
            return np.nan
        return float(np.log(r / s) / np.log(m))

    return diffs.rolling(n, min_periods=n).apply(_rs, raw=True)


def kst(frame: pd.DataFrame, column: str = "close") -> pd.Series:
    """Know Sure Thing (Martin Pring): a weighted sum of four smoothed
    rates-of-change, KST = 1*SMA10(ROC10) + 2*SMA10(ROC15) +
    3*SMA10(ROC20) + 4*SMA15(ROC30). Standard fixed parameters (they are
    the indicator's definition, not a tunable period). All components are
    trailing-only (causal)."""
    roc10 = roc(frame, 10, column)
    roc15 = roc(frame, 15, column)
    roc20 = roc(frame, 20, column)
    roc30 = roc(frame, 30, column)
    return (sma(roc10, 10) + 2 * sma(roc15, 10)
            + 3 * sma(roc20, 10) + 4 * sma(roc30, 15))


def coppock(frame: pd.DataFrame, column: str = "close") -> pd.Series:
    """Coppock Curve (E.S.C. Coppock): WMA(10) of (ROC(14) + ROC(11)) --
    a long-term momentum oscillator. Standard fixed parameters (the
    indicator's definition). Trailing-only (causal)."""
    return wma(roc(frame, 14, column) + roc(frame, 11, column), 10)


# ---------------------------------------------------------------------------
# Dispatch table: kind -> (frame, period, column, lookback) -> Series.
# Mirrors the signature of
# app.strategy.indicators._build_indicator_series_uncached so the v7
# fallback hook there can call straight through. `period` drives the
# indicator's main window; the fixed-parameter indicators (kst, coppock)
# ignore it, exactly like the existing connors_rsi/awesome_oscillator do.
# ---------------------------------------------------------------------------
V7IndicatorBuilder = Callable[[pd.DataFrame, int, str, int | None], pd.Series]

V7_INDICATOR_BUILDERS: dict[str, V7IndicatorBuilder] = {
    "stochrsi_k": lambda frame, period, column, lookback: stochrsi(frame, period, column)[0],
    "stochrsi_d": lambda frame, period, column, lookback: stochrsi(frame, period, column)[1],
    "plus_di": lambda frame, period, column, lookback: plus_di(frame, period),
    "minus_di": lambda frame, period, column, lookback: minus_di(frame, period),
    "linreg_slope": lambda frame, period, column, lookback: linreg_slope(
        frame[column] if column in frame.columns else frame["close"], period),
    "hurst_exponent": lambda frame, period, column, lookback: hurst_exponent(
        frame[column] if column in frame.columns else frame["close"], period),
    "kst": lambda frame, period, column, lookback: kst(frame, column),
    "coppock": lambda frame, period, column, lookback: coppock(frame, column),
}

# Bounded 0-100 oscillators among the new kinds -- the grammar's
# THRESHOLD_BOUNDS (app.search.grammar) references this so random
# thresholds stay inside the indicators' real ranges. (Deliberately kept
# as a v7-local constant rather than editing the app-wide
# BOUNDED_OSCILLATOR_RANGES map -- the grammar-local bounds rule is
# enforced by the grammar's own draw/validate path.)
V7_BOUNDED_RANGES: dict[str, tuple[float, float]] = {
    "stochrsi_k": (0.0, 100.0),
    "stochrsi_d": (0.0, 100.0),
    "plus_di": (0.0, 100.0),
    "minus_di": (0.0, 100.0),
}

V7_INDICATOR_KINDS: tuple[str, ...] = tuple(V7_INDICATOR_BUILDERS)
