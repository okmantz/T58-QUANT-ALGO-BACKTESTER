"""
v8 indicator extensions (2026-10-05).

Eight new lookahead-safe indicator kinds for the discovery grammar,
covering gaps, efficiency, channel position, bandwidth, volume and
trend-linearity concepts that had no terminal yet:

    atr_percentile        trailing percentile rank of ATR (0-100): is
                          current volatility high or low *relative to its
                          own recent history*, dimensionless.
    efficiency_ratio      Kaufman's Efficiency Ratio
                          (|close - close[period]| / sum(|bar-to-bar|),
                          0-1): how efficiently price travelled vs how
                          much it wandered. The `period` operand knob is
                          the ER window.
    donchian_mid_distance (close - donchian_mid) / (upper - lower):
                          dimensionless channel position, roughly
                          -0.5..+0.5; 0 at the channel midpoint.
    bollinger_bandwidth   (upper - lower) / mid: the classic squeeze
                          gauge, dimensionless and scale-free.
    volume_zscore         (volume - sma(volume)) / std(volume): how
                          unusual participation is, in standard
                          deviations (vs relative_volume's plain ratio).
    overnight_gap_atr     (open - prev_close) / ATR: the bar-to-bar gap
                          normalized by volatility -- dimensionless, so a
                          grammar threshold means the same thing on ES
                          and on EURUSD.
    fractal_strength      ATR-normalized size of the most recent
                          CONFIRMED fractal swing envelope: a causal
                          multi-bar "how big are the swings getting"
                          exhaustion gauge.
    linreg_r2             rolling R-squared of a least-squares fit of
                          price on time (0-1): trend *linearity*,
                          complementing v7's linreg_slope (trend rate).

Lookahead safety: every series is built only from the current and past
bars -- trailing rolling windows, .shift(+n) (past bars) only, no
negative shifts, no centered windows, no whole-frame aggregates.
fractal_strength's swing confirmation deliberately mirrors
app.strategy.manual's swing_high/swing_low semantics (a centered
2w+1 window whose confirmation is shifted forward by w bars, so the
signal at bar i only uses swings that would have been knowable at i).

Wiring: app.strategy.indicators._build_indicator_series_uncached falls
back to V8_INDICATOR_BUILDERS here for any kind neither it nor the v7
module knows (additive only), app.strategy.manual._series_from_operand
routes the new kind names into build_indicator_series, and
app.search.grammar.INDICATOR_KINDS registers them as grammar terminals.
Covered by tests/test_indicators_v8.py (reference-value agreement on
synthetic data, the prefix-invariance property the repo's behavioral
lookahead check rests on, and the behavioral check itself via a Manual
strategy that trades on each kind).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from app.strategy.indicators import _period, atr, sma


def atr_percentile(frame: pd.DataFrame, period: int = 14, rank_window: int = 100) -> pd.Series:
    """Percentile rank (0-100) of the current ATR within its own trailing
    `rank_window`-bar history. 100 = the most volatile bar in the window,
    0 = the calmest. The rolling window INCLUDES the current bar -- the
    rank of bar i is a pure function of bars i-rank_window+1..i, so it is
    causal (this is the same trailing-inclusive convention Connors RSI
    uses for its own percentile-rank leg)."""
    p = _period(period)
    w = max(int(rank_window or 100), 2)
    atr_s = atr(frame, p)
    out = atr_s.rolling(w, min_periods=w).apply(
        lambda x: float((x <= x[-1]).mean()) * 100.0, raw=True
    )
    return out.clip(0.0, 100.0)


def efficiency_ratio(series: pd.Series, period: int = 14) -> pd.Series:
    """Kaufman's Efficiency Ratio: net change over the window divided by
    the sum of absolute bar-to-bar changes, 0-1. 1.0 = price moved in a
    straight line; near 0 = price chopped sideways. All windows are
    trailing-only (change uses series.shift(period) -- past bars)."""
    p = _period(period)
    change = (series - series.shift(p)).abs()
    volatility = series.diff().abs().rolling(p, min_periods=p).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0.0)
    return er.clip(0.0, 1.0)


def donchian_mid_distance(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    """Dimensionless channel position: (close - donchian_mid) /
    (donchian_upper - donchian_lower). +0.5 at the upper band, -0.5 at
    the lower band, 0 at the midpoint. Trailing rolling max/min only --
    causal. A flat channel (upper == lower) reads as 0.0, not NaN."""
    p = _period(period)
    upper = frame["high"].rolling(p, min_periods=p).max()
    lower = frame["low"].rolling(p, min_periods=p).min()
    mid = (upper + lower) / 2.0
    width = (upper - lower).replace(0, np.nan)
    return ((frame["close"] - mid) / width).fillna(0.0)


def bollinger_bandwidth(frame: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger bandwidth: (upper - lower) / mid, dimensionless and
    scale-free (the same 0.02 threshold means the same thing on ES and
    EURUSD). Trailing mean/std only -- causal."""
    p = _period(period)
    mid = frame["close"].rolling(p, min_periods=p).mean()
    sd = frame["close"].rolling(p, min_periods=p).std()
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    return ((upper - lower) / mid.replace(0, np.nan)).fillna(0.0)


def volume_zscore(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    """(volume - trailing mean) / trailing std: participation in standard
    deviations. Complements the existing relative_volume ratio terminal
    (volume / average_volume) -- a z-score knows the difference between
    "2x on a quiet day" and "2x on a normally wild day". Causal."""
    p = _period(period)
    vol = frame["volume"]
    mean = vol.rolling(p, min_periods=p).mean()
    sd = vol.rolling(p, min_periods=p).std()
    return ((vol - mean) / sd.replace(0, np.nan)).fillna(0.0)


def overnight_gap_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """(open - prev_close) / ATR: the bar-to-bar gap in volatility units.
    On daily bars this is the classic overnight gap; on intraday bars it
    is the per-bar discontinuity, still dimensionless and comparable
    across instruments. prev_close is shift(1) -- past data only."""
    p = _period(period)
    gap = frame["open"] - frame["close"].shift(1)
    atr_s = atr(frame, p)
    return (gap / atr_s.replace(0, np.nan)).fillna(0.0)


def fractal_strength(frame: pd.DataFrame, period: int = 14, swing_window: int = 5) -> pd.Series:
    """ATR-normalized size of the most recent CONFIRMED fractal swing
    envelope: (|last confirmed swing high - last confirmed swing low|) /
    ATR. A swing point is only confirmed `swing_window` bars after it
    prints (centered 2w+1 window, shifted forward by w -- the exact
    convention app.strategy.manual uses for swing_high/swing_low), so
    the series at bar i never sees a swing that would not have been
    knowable at bar i. The recorded extreme is the swing's OWN high/low
    (not the confirmation bar's price -- see the shift(w) below). Reads
    0.0 until both a swing high and a swing low have been confirmed.
    Growing values = the market's swings are getting bigger in
    volatility units (exhaustion/blow-off territory);
    shrinking values = compression."""
    p = _period(period)
    w = max(int(swing_window or 5), 1)
    high, low = frame["high"], frame["low"]
    hi_raw = high == high.rolling(2 * w + 1, center=True, min_periods=w + 1).max()
    lo_raw = low == low.rolling(2 * w + 1, center=True, min_periods=w + 1).min()
    hi_conf = hi_raw.shift(w)
    lo_conf = lo_raw.shift(w)
    # The swing confirmed at bar i printed at bar i-w: shift the price
    # series back by the same lag so the recorded extreme is the swing's
    # own high/low, not the confirmation bar's. Still causal -- bar i-w
    # is in the past at bar i.
    last_hi = high.shift(w).where(hi_conf.fillna(False)).ffill()
    last_lo = low.shift(w).where(lo_conf.fillna(False)).ffill()
    atr_s = atr(frame, p)
    strength = ((last_hi - last_lo).abs() / atr_s.replace(0, np.nan))
    return strength.fillna(0.0)


def linreg_r2(series: pd.Series, period: int = 14) -> pd.Series:
    """Rolling R-squared of an OLS fit of price on time within the
    trailing `period`-bar window (0-1): how LINEAR the recent trend is.
    Complements v7's linreg_slope (how FAST price is moving): a steep
    slope with low R2 is a jagged lurch; the same slope with high R2 is
    a clean trend. Fully vectorized via trailing rolling sums (no
    per-window apply), and the window is trailing-only -- causal.

    Math: within a window ending at bar i, the within-window time index
    is x_k = k for k = 0..n-1. cov(x, y) comes from rolling sums of y
    and of (global_index * y): S_xy(window) = sum(t_j*y_j) - start*sum(y_j)
    where start = t - n + 1 is the window's first global index.
    """
    p = _period(period)
    n = float(p)
    y = series.astype(float)
    t = pd.Series(np.arange(len(y)), index=y.index, dtype=float)
    sx = n * (n - 1.0) / 2.0
    sxx = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0
    var_x = sxx / n - (sx / n) ** 2
    sy = y.rolling(p, min_periods=p).sum()
    syy = (y * y).rolling(p, min_periods=p).sum()
    sty = (t * y).rolling(p, min_periods=p).sum()
    start = t - n + 1.0
    sxy = sty - start * sy
    cov = sxy / n - (sx / n) * (sy / n)
    var_y = syy / n - (sy / n) ** 2
    denom = var_x * var_y
    r2 = (cov * cov / denom.where(denom > 0, np.nan)).fillna(0.0)
    return r2.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Dispatch table: kind -> (frame, period, column, lookback) -> Series.
# Mirrors the signature of
# app.strategy.indicators._build_indicator_series_uncached so the v8
# fallback hook there can call straight through. `period` drives the
# indicator's main window; atr_percentile's rank window and
# fractal_strength's swing window use fixed defaults, exactly like the
# existing connors_rsi/awesome_oscillator fixed-parameter kinds.
# ---------------------------------------------------------------------------
V8IndicatorBuilder = Callable[[pd.DataFrame, int, str, int | None], pd.Series]

V8_INDICATOR_BUILDERS: dict[str, V8IndicatorBuilder] = {
    "atr_percentile": lambda frame, period, column, lookback: atr_percentile(frame, period),
    "efficiency_ratio": lambda frame, period, column, lookback: efficiency_ratio(
        frame[column] if column in frame.columns else frame["close"], period),
    "donchian_mid_distance": lambda frame, period, column, lookback: donchian_mid_distance(frame, period),
    "bollinger_bandwidth": lambda frame, period, column, lookback: bollinger_bandwidth(frame, period),
    "volume_zscore": lambda frame, period, column, lookback: volume_zscore(frame, period),
    "overnight_gap_atr": lambda frame, period, column, lookback: overnight_gap_atr(frame, period),
    "fractal_strength": lambda frame, period, column, lookback: fractal_strength(frame, period),
    "linreg_r2": lambda frame, period, column, lookback: linreg_r2(
        frame[column] if column in frame.columns else frame["close"], period),
}

# Bounded oscillators among the new kinds -- the grammar's
# THRESHOLD_BOUNDS (app.search.grammar) references this so random
# thresholds stay inside the indicators' real ranges. (Kept as a
# v8-local constant rather than editing the app-wide
# BOUNDED_OSCILLATOR_RANGES map -- the grammar-local bounds rule is
# enforced by the grammar's own draw/validate path.)
V8_BOUNDED_RANGES: dict[str, tuple[float, float]] = {
    "atr_percentile": (0.0, 100.0),
    "efficiency_ratio": (0.0, 1.0),
    "linreg_r2": (0.0, 1.0),
}

V8_INDICATOR_KINDS: tuple[str, ...] = tuple(V8_INDICATOR_BUILDERS)
