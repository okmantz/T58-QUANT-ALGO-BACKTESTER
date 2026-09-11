"""
Composable threshold / signal engine -- ported from the HyperTA project's
Thresholds module (HyperTA/Thresholds/thresholds.py: crossLevel,
crossLines, inRange, holdLevel, stdvBandsThreshold, skew/kurtosis/
derivative triggers, mixThresholds), which was real working code, not a
stub -- see that project's own analysis for the full picture.

Adapted here to reuse this app's OWN existing indicator math
(app.strategy.indicators.build_indicator_series -- RSI, EMA, MACD,
Bollinger, ATR, VWAP, etc. are already implemented and tested there)
rather than re-deriving indicator values from scratch the way HyperTA's
version does. Every threshold function below takes a plain OHLCV
DataFrame (this app's standard timestamp/open/high/low/close/volume
shape) and an indicator `kind` string (any value build_indicator_series
already accepts), and returns a normalized Signal.

The composable idea itself IS the point of porting this: rather than
every strategy family hand-rolling its own "RSI oversold AND price above
200 EMA" condition from scratch, mix_thresholds() combines any number of
these Signals via AND/OR set logic into one first-class composite
signal -- usable standalone (see the Quant Lab "Composite Signal
Builder" tool, web + desktop) today, and a natural building block for a
future Strategy DSL condition type.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.strategy.indicators import build_indicator_series


class ThresholdError(Exception):
    """Raised for a bad configuration (unknown indicator kind, empty
    data, an impossible parameter combination)."""


@dataclass
class Signal:
    """Normalized output of every threshold function below: which bars
    triggered (a boolean Series aligned to the input DataFrame's index),
    plus a human-readable label for display/logging."""
    label: str
    triggered: pd.Series  # bool, same index as the input df

    @property
    def n_triggers(self) -> int:
        return int(self.triggered.sum())

    def trigger_timestamps(self, df: pd.DataFrame) -> pd.Series:
        return df.loc[self.triggered, "timestamp"] if "timestamp" in df.columns else self.triggered[self.triggered].index.to_series()


def _series_for(df: pd.DataFrame, kind: str, period: int, column: str = "close") -> pd.Series:
    try:
        return build_indicator_series(df, kind, period=period, column=column)
    except Exception as exc:  # noqa: BLE001 -- surface as ThresholdError, this module's own exception type
        raise ThresholdError(f"Could not compute indicator '{kind}' (period={period}): {exc}") from exc


# ---------------------------------------------------------------------------
# Threshold primitives
# ---------------------------------------------------------------------------

def cross_level(df: pd.DataFrame, kind: str, period: int, threshold: float, *, direction: str = "above", column: str = "close") -> Signal:
    """Fires the bar an indicator crosses a fixed numeric level.
    direction='above': fires going from <= threshold to > threshold (e.g. RSI crossing above 30).
    direction='below': fires going from >= threshold to < threshold (e.g. RSI crossing below 70)."""
    series = _series_for(df, kind, period, column)
    if direction == "above":
        triggered = (series > threshold) & (series.shift(1) <= threshold)
        label = f"{kind}({period}) crosses above {threshold:g}"
    elif direction == "below":
        triggered = (series < threshold) & (series.shift(1) >= threshold)
        label = f"{kind}({period}) crosses below {threshold:g}"
    else:
        raise ThresholdError(f"direction must be 'above' or 'below', got {direction!r}.")
    return Signal(label=label, triggered=triggered.fillna(False))


def cross_lines(df: pd.DataFrame, fast_kind: str, fast_period: int, slow_kind: str, slow_period: int, *, direction: str = "above", column: str = "close") -> Signal:
    """Fires when a fast line crosses a slow line -- e.g. EMA(9) crossing above EMA(21),
    or MACD crossing above its signal line (fast_kind='macd', slow_kind='macd_signal')."""
    fast = _series_for(df, fast_kind, fast_period, column)
    slow = _series_for(df, slow_kind, slow_period, column)
    if direction == "above":
        triggered = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        label = f"{fast_kind}({fast_period}) crosses above {slow_kind}({slow_period})"
    elif direction == "below":
        triggered = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        label = f"{fast_kind}({fast_period}) crosses below {slow_kind}({slow_period})"
    else:
        raise ThresholdError(f"direction must be 'above' or 'below', got {direction!r}.")
    return Signal(label=label, triggered=triggered.fillna(False))


def in_range(df: pd.DataFrame, kind: str, period: int, lower: float, upper: float, *, column: str = "close") -> Signal:
    """Fires on every bar where the indicator sits inside [lower, upper] -- e.g. RSI between 40 and 60 (a 'no edge' filter)."""
    if lower >= upper:
        raise ThresholdError(f"lower ({lower}) must be < upper ({upper}).")
    series = _series_for(df, kind, period, column)
    triggered = (series >= lower) & (series <= upper)
    return Signal(label=f"{kind}({period}) in [{lower:g}, {upper:g}]", triggered=triggered.fillna(False))


def hold_level(df: pd.DataFrame, kind: str, period: int, level: float, *, direction: str = "above", min_bars: int = 3, column: str = "close") -> Signal:
    """Fires on the bar an indicator has stayed above (or below) `level`
    for at least `min_bars` consecutive bars -- e.g. 'RSI above 50 for 3+ bars'
    (a persistence filter, distinct from a one-bar cross_level)."""
    if min_bars < 1:
        raise ThresholdError("min_bars must be >= 1.")
    series = _series_for(df, kind, period, column)
    holding = (series > level) if direction == "above" else (series < level)
    if direction not in ("above", "below"):
        raise ThresholdError(f"direction must be 'above' or 'below', got {direction!r}.")
    # Consecutive-True run length ending at each bar, via the classic
    # "reset a counter to 0 on every False" grouping trick.
    run_id = (~holding).cumsum()
    run_length = holding.groupby(run_id).cumcount() + 1
    run_length = run_length.where(holding, 0)
    triggered = run_length >= min_bars
    return Signal(label=f"{kind}({period}) held {direction} {level:g} for {min_bars}+ bars", triggered=triggered.fillna(False))


def stdv_bands_threshold(df: pd.DataFrame, *, ema_period: int = 10, window: int = 50, sigma: float = 0.8, column: str = "close") -> Signal:
    """Fires when price breaches an EMA +/- sigma*rolling-stdev band --
    a volatility-relative version of a Bollinger breakout, using this app's
    own ema()/stdev() math rather than a fixed-lookback Bollinger Band."""
    from app.strategy.indicators import ema as _ema, stdev as _stdev

    price = df[column] if column in df.columns else df["close"]
    center = _ema(price, ema_period)
    band = _stdev(price, window) * sigma
    upper, lower = center + band, center - band
    triggered = (price > upper) | (price < lower)
    return Signal(label=f"price outside EMA({ema_period}) +/- {sigma:g}*stdev({window})", triggered=triggered.fillna(False))


def skew_threshold(df: pd.DataFrame, *, window: int = 20, lower: float = -2.0, upper: float = 1.0, column: str = "close") -> Signal:
    """Fires when a rolling window's return-distribution skew falls
    outside [lower, upper] -- an asymmetric-tail filter (e.g. a sharp
    one-sided move building inside an otherwise calm range)."""
    price = df[column] if column in df.columns else df["close"]
    returns = price.pct_change()
    skew = returns.rolling(window, min_periods=max(3, window // 2)).skew()
    triggered = (skew < lower) | (skew > upper)
    return Signal(label=f"return skew({window}) outside [{lower:g}, {upper:g}]", triggered=triggered.fillna(False))


def kurtosis_threshold(df: pd.DataFrame, *, window: int = 20, lower: float = -2.0, upper: float = 1.0, column: str = "close") -> Signal:
    """Fires when a rolling window's return-distribution excess kurtosis
    falls outside [lower, upper] -- a fat-tails/regime-change filter."""
    price = df[column] if column in df.columns else df["close"]
    returns = price.pct_change()
    kurt = returns.rolling(window, min_periods=max(4, window // 2)).kurt()
    triggered = (kurt < lower) | (kurt > upper)
    return Signal(label=f"return kurtosis({window}) outside [{lower:g}, {upper:g}]", triggered=triggered.fillna(False))


def derivative_threshold(df: pd.DataFrame, *, k: int = 20, threshold: float = 0.0, direction: str = "above", column: str = "close") -> Signal:
    """Fires when a smoothed rate-of-change (a k-bar centered slope of an
    EMA-smoothed price, not a raw 1-bar diff, so a single noisy tick can't
    trigger it) crosses `threshold` -- e.g. 'rate of change turns positive'
    as a lightweight momentum-inflection filter."""
    from app.strategy.indicators import ema as _ema

    price = df[column] if column in df.columns else df["close"]
    smoothed = _ema(price, max(2, k // 4))
    slope = (smoothed - smoothed.shift(k)) / max(k, 1)
    if direction == "above":
        triggered = (slope > threshold) & (slope.shift(1) <= threshold)
    elif direction == "below":
        triggered = (slope < threshold) & (slope.shift(1) >= threshold)
    else:
        raise ThresholdError(f"direction must be 'above' or 'below', got {direction!r}.")
    return Signal(label=f"{k}-bar smoothed slope crosses {direction} {threshold:g}", triggered=triggered.fillna(False))


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def mix_thresholds(signals: list[Signal], *, mode: str = "and") -> Signal:
    """Combines any number of Signals into one composite Signal via set
    logic: mode='and' fires only where EVERY signal fired on that bar
    (intersection); mode='or' fires where ANY signal fired (union).
    This is the actual point of porting HyperTA's threshold engine: e.g.
    mix_thresholds([rsi_oversold, above_200ema], mode='and') gives you
    'RSI oversold AND price above the 200 EMA' as one first-class signal,
    instead of every strategy family hand-rolling this combination
    itself."""
    if not signals:
        raise ThresholdError("mix_thresholds needs at least one signal.")
    if mode not in ("and", "or"):
        raise ThresholdError(f"mode must be 'and' or 'or', got {mode!r}.")
    combined = signals[0].triggered.copy()
    for s in signals[1:]:
        combined = (combined & s.triggered) if mode == "and" else (combined | s.triggered)
    joiner = " AND " if mode == "and" else " OR "
    label = joiner.join(f"({s.label})" for s in signals)
    return Signal(label=label, triggered=combined)


@dataclass
class SignalPreview:
    label: str
    n_triggers: int
    trigger_timestamps: list
    warnings: list = field(default_factory=list)

    def render_summary(self) -> str:
        lines = [f"Signal: {self.label}", f"Triggers: {self.n_triggers}"]
        if self.n_triggers:
            shown = self.trigger_timestamps[-20:]
            lines.append(f"Most recent {len(shown)} trigger timestamp(s):")
            lines.extend(f"  {ts}" for ts in shown)
        for w in self.warnings:
            lines.append(f"  note: {w}")
        return "\n".join(lines)


def preview_signal(df: pd.DataFrame, signal: Signal) -> SignalPreview:
    """Turns a Signal into a display-ready preview (trigger count + the
    most recent trigger timestamps) -- what the Quant Lab tool and CLI
    actually show, rather than a raw boolean Series."""
    warnings: list[str] = []
    if signal.n_triggers == 0:
        warnings.append("No triggers -- try loosening the threshold(s) or a longer dataset.")
    timestamps = list(signal.trigger_timestamps(df)) if "timestamp" in df.columns else []
    return SignalPreview(label=signal.label, n_triggers=signal.n_triggers, trigger_timestamps=timestamps, warnings=warnings)
