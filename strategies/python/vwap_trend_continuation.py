"""
VWAP Trend Continuation (ema 50/100, rsi7)
--------------------------------------------
Written for the T58 Quant Algo Backtester's Python strategy adapter
(app/strategy/python.py), which requires a top-level
`generate_signals(df) -> pd.Series` of -1/0/1 values.

This mirrors app/strategy/manual.py + app/strategy/indicators.py exactly,
so it reproduces the same trades as the "manual" JSON version of this
strategy that already ran through your Full Pipeline report:

- VWAP ignores the "period" field in your engine -- it's a SESSION-
  ANCHORED cumulative VWAP that resets every calendar day, not a rolling
  N-bar average. (This is the #1 reason the previous version produced no
  trades: it used a rolling window, which is a different indicator.)
- "close" operands ignore their "period"/"field" -- they're just the raw
  close price.
- RSI/EMA/ATR use Wilder-style smoothing (ewm alpha=1/period), same as
  your indicators.py.
- Entries/exits run through the same stateful long/flat/short loop as
  app.strategy.base.signals_from_conditions (opposite-signal flip is
  always on, matching risk_management.opposite_signal_exit=true), plus
  the same max-bars-in-trade flattening as
  ManualStrategy._apply_signal_exits.
- ATR-based stop/target distances are attached via .attrs so the engine
  sizes and protects each trade exactly as risk_management specifies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "VWAP Trend Continuation (ema 50/100, rsi7)"

PARAMS = dict(
    ema_long_slow=294,
    ema_long_fast=58,
    ema_short_fast=35,
    ema_short_slow=159,
    rsi_long_period=18,
    rsi_long_thresh=57.6102629644959,
    rsi_short_period=42,
    rsi_short_thresh=66.008585331847,
    rsi_exit_period=5,
    rsi_exit_long_thresh=66.68737207001313,
    rsi_exit_short_thresh=64.42442560822603,
    atr_stop_period=16,
    atr_stop_mult=4.806786275425379,
    atr_target_period=11,
    atr_target_mult=5.054720923946611,
    max_bars_in_trade=63,
)


# ---------------------------------------------------------------------------
# Indicators -- deliberately identical to app/strategy/indicators.py
# ---------------------------------------------------------------------------
def _ema(series: pd.Series, period: int) -> pd.Series:
    p = max(int(period), 1)
    return series.ewm(span=p, adjust=False, min_periods=p).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    p = max(int(period), 1)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / p, adjust=False, min_periods=p).mean()
    avg_loss = loss.ewm(alpha=1 / p, adjust=False, min_periods=p).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss.ne(0), 100)
    return result.fillna(50)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    p = max(int(period), 1)
    return _true_range(df).ewm(alpha=1 / p, adjust=False, min_periods=p).mean()


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative, day-anchored VWAP -- resets at the start of each
    calendar day, exactly like app.strategy.indicators.vwap(). The
    strategy JSON's "period" field on the VWAP operand is not used by
    your engine at all, for either side, so long/short share one series.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    ts = pd.to_datetime(df["timestamp"])
    day = ts.dt.normalize()
    pv = typical * volume
    return pv.groupby(day).cumsum() / volume.groupby(day).cumsum().replace(0, np.nan)


# ---------------------------------------------------------------------------
# Signal state machine -- mirrors app.strategy.base.signals_from_conditions
# ---------------------------------------------------------------------------
def _signals_from_conditions(
    index: pd.Index,
    long_entry: pd.Series,
    long_exit: pd.Series,
    short_entry: pd.Series,
    short_exit: pd.Series,
    allow_opposite_signal_flip: bool = True,
) -> pd.Series:
    le, lx = long_entry.to_numpy(), long_exit.to_numpy()
    se, sx = short_entry.to_numpy(), short_exit.to_numpy()

    position = 0
    out = np.zeros(len(index), dtype=int)
    for i in range(len(index)):
        if position == 0:
            if le[i]:
                position = 1
            elif se[i]:
                position = -1
        elif position == 1:
            if lx[i]:
                position = 0
            elif allow_opposite_signal_flip and se[i]:
                position = -1
        elif position == -1:
            if sx[i]:
                position = 0
            elif allow_opposite_signal_flip and le[i]:
                position = 1
        out[i] = position

    return pd.Series(out, index=index)


def _apply_max_bars(signals: pd.Series, max_bars: int) -> pd.Series:
    """Mirrors ManualStrategy._apply_signal_exits's max_bars_in_trade block:
    once a position has been held for `max_bars` bars, flatten it."""
    vals = signals.to_numpy(copy=True)
    position = 0
    bars = 0
    for i in range(len(vals)):
        if position == 0 and vals[i] != 0:
            position = vals[i]
            bars = 0
        elif position != 0:
            if vals[i] != position:
                position = vals[i]
                bars = 0
            else:
                bars += 1
                if bars >= max_bars:
                    vals[i] = 0
                    position = 0
                    bars = 0
    return pd.Series(vals, index=signals.index)


# ---------------------------------------------------------------------------
# Public entry point required by app/strategy/python.py
# ---------------------------------------------------------------------------
def generate_signals(df: pd.DataFrame, p: dict = PARAMS) -> pd.Series:
    work = df.copy()

    ema_long_slow = _ema(work["close"], p["ema_long_slow"])
    ema_long_fast = _ema(work["close"], p["ema_long_fast"])
    ema_short_fast = _ema(work["close"], p["ema_short_fast"])
    ema_short_slow = _ema(work["close"], p["ema_short_slow"])

    rsi_long = _rsi(work["close"], p["rsi_long_period"])
    rsi_short = _rsi(work["close"], p["rsi_short_period"])
    rsi_exit = _rsi(work["close"], p["rsi_exit_period"])

    vwap = _session_vwap(work)  # same series feeds both long and short sides

    long_entry = (ema_long_slow > ema_long_fast) & (work["close"] > vwap) & (rsi_long < p["rsi_long_thresh"])
    short_entry = (ema_short_fast < ema_short_slow) & (work["close"] < vwap) & (rsi_short > p["rsi_short_thresh"])
    long_exit = rsi_exit > p["rsi_exit_long_thresh"]
    short_exit = rsi_exit < p["rsi_exit_short_thresh"]

    long_entry = long_entry.fillna(False)
    short_entry = short_entry.fillna(False)
    long_exit = long_exit.fillna(False)
    short_exit = short_exit.fillna(False)

    raw_signals = _signals_from_conditions(
        work.index, long_entry, long_exit, short_entry, short_exit,
        allow_opposite_signal_flip=True,  # risk_management.opposite_signal_exit = true
    )
    signals = _apply_max_bars(raw_signals, p["max_bars_in_trade"])

    atr_stop = _atr(work, p["atr_stop_period"])
    atr_target = _atr(work, p["atr_target_period"])
    stop_loss_distance = atr_stop * p["atr_stop_mult"]
    take_profit_distance = atr_target * p["atr_target_mult"]

    signals.attrs["stop_loss_distance"] = stop_loss_distance
    signals.attrs["take_profit_distance"] = take_profit_distance

    return signals


if __name__ == "__main__":
    print(
        "This module is meant to be uploaded to the T58 Backtester's Python "
        "strategy slot. It exposes generate_signals(df) per app/strategy/python.py's "
        "contract; it isn't meant to be run standalone."
    )
