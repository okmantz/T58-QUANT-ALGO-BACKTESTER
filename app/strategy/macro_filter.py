"""
Macro (higher-timeframe) bias filter -- works for EVERY strategy source (manual JSON, Python, Pine, MQL5).

"Analyze the 1-hour, execute on the 15-minute": the strategy runs on the execution timeframe exactly as written; this layer
reads the bias of a coarser timeframe and only lets entries through that agree with it. The bias of an HTF bar is used only
once that bar has CLOSED (never a forming bar), so it cannot look ahead.

Modes (bias is +1 up / -1 down / 0 neutral):
  ema         price above/below an EMA(length) of the macro timeframe
  ema_slope   EMA(length) rising/falling
  supertrend  SuperTrend(length, 3x ATR) direction
  donchian    breaks of the prior `length`-bar high/low (persistent state)
  adx_ema     EMA side, but neutral while ADX(14) < 20 (no trend = no bias)
  macd        MACD(12,26,9) histogram sign
  rsi         RSI(length) above/below 50
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy import indicators as I
from app.strategy.base import StrategyResult

MODES = {
    "off": "Off",
    "ema": "Price vs EMA",
    "ema_slope": "EMA slope",
    "supertrend": "SuperTrend direction",
    "donchian": "Donchian breakout state",
    "adx_ema": "EMA side, neutral when ADX < 20",
    "macd": "MACD histogram",
    "rsi": "RSI above/below 50",
}

_CACHE: dict = {}


def _bias_series(htf: pd.DataFrame, mode: str, length: int) -> np.ndarray:
    c = htf["close"]
    if mode == "ema":
        b = np.sign(c - I.ema(c, length))
    elif mode == "ema_slope":
        e = I.ema(c, length)
        b = np.sign(e.diff())
    elif mode == "supertrend":
        _line, d = I.supertrend(htf, period=max(int(length), 2), atr_mult=3.0)
        b = np.sign(d)
    elif mode == "donchian":
        hi = htf["high"].rolling(length, min_periods=length).max().shift(1)
        lo = htf["low"].rolling(length, min_periods=length).min().shift(1)
        up, dn = (c > hi).to_numpy(), (c < lo).to_numpy()
        out = np.zeros(len(c))
        state = 0.0
        for k in range(len(c)):
            if up[k]:
                state = 1.0
            elif dn[k]:
                state = -1.0
            out[k] = state
        b = pd.Series(out, index=c.index)
    elif mode == "adx_ema":
        side = np.sign(c - I.ema(c, length))
        b = side.where(I.adx(htf, 14) >= 20, 0.0)
    elif mode == "macd":
        _m, _s, h = I.macd(c, 12, 26, 9)
        b = np.sign(h)
    elif mode == "rsi":
        b = np.sign(I.rsi(c, max(int(length), 2)) - 50.0)
    else:
        raise ValueError(f"Unknown macro filter mode '{mode}'.")
    return pd.Series(b).fillna(0.0).to_numpy(float)


def macro_bias(df: pd.DataFrame, macro_tf: str, mode: str, length: int) -> np.ndarray:
    """Per-bar bias of `df` (the execution frame) from the macro timeframe, aligned without look-ahead."""
    from app.data.timeframe_resample import parse_timeframe_label
    from app.strategy.interp_common import bar_minutes, completed_map, resample_to, time_index
    htf_min = parse_timeframe_label(macro_tf)
    base_min = bar_minutes(df)
    if htf_min <= base_min * 1.0001:
        raise ValueError(f"The macro timeframe ({macro_tf}) must be coarser than the execution timeframe (~{base_min:.0f}m).")
    ts = time_index(df)
    key = (len(df), int(ts[0].value), int(ts[-1].value), round(float(df["close"].sum()), 6), macro_tf, mode, int(length))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    htf = resample_to(df, htf_min)
    bias_htf = _bias_series(htf, mode, int(length))
    pos = completed_map(ts, base_min, time_index(htf), htf_min)
    out = np.where(pos >= 0, bias_htf[np.clip(pos, 0, len(bias_htf) - 1)], 0.0)
    if len(_CACHE) > 64:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out


def apply_macro_filter(result: StrategyResult, df: pd.DataFrame, macro_tf: str, mode: str, length: int = 50,
                       strict: bool = False, exit_on_flip: bool = False) -> tuple[StrategyResult, str]:
    """Returns (result with gated signals, one-line description). Entries are only taken on the bar the strategy itself
    changes its signal, and only when the macro bias allows that direction (strict: bias must equal the direction; default:
    bias must not be the opposite direction). A blocked entry stays blocked until the strategy changes its signal again."""
    if getattr(result, "entry_orders", None):
        return result, "Macro filter skipped: this strategy places resting limit orders, which the filter cannot gate."
    bias = macro_bias(df, macro_tf, mode, length)
    sig = result.signals.to_numpy(int)
    out = np.zeros(len(sig), dtype=int)
    pos, blocked, prev = 0, False, 0
    entries = allowed = 0
    for i in range(len(sig)):
        s = sig[i]
        b = bias[i]
        if s != prev:                                   # the strategy changed its mind on this bar
            blocked = False
            if s != 0:
                entries += 1
                ok = (b == s) if strict else (b != -s)
                if ok:
                    allowed += 1
                    pos = s
                else:
                    blocked = True
                    pos = 0
            else:
                pos = 0
        elif s != 0 and not blocked and pos == 0:
            pos = 0                                      # never enter a hold that began while blocked
        if pos != 0 and exit_on_flip and b == -pos:
            pos, blocked = 0, True
        prev = s
        out[i] = pos
    res = StrategyResult(**{**result.__dict__, "signals": pd.Series(out, index=result.signals.index)})
    note = (f"Macro filter ({macro_tf} {MODES.get(mode, mode)}, length {length}"
            f"{', strict' if strict else ''}{', exits on flip' if exit_on_flip else ''}): "
            f"{allowed} of {entries} entries agreed with the higher-timeframe bias; {entries - allowed} were blocked.")
    return res, note


# ---------------------------------------------------------------------------------------------------------------------
# Regime / volatility gate -- the audit's "all four strategies weakened in 2025" finding: edges are regime dependent, so let
# any strategy be restricted to the conditions it actually works in. Causal: uses only bars up to and including the signal bar.
REGIME_MODES = {
    "off": "Off",
    "vol_normal_or_high": "Trade only when volatility is not in the lowest third (ATR percentile >= 33)",
    "vol_high": "Trade only in the top third of volatility (ATR percentile >= 67)",
    "vol_low": "Trade only in the lowest two thirds of volatility (ATR percentile <= 67)",
    "trending": "Trade only when ADX(14) >= 20 (trend regime)",
    "ranging": "Trade only when ADX(14) < 20 (range regime)",
}


def regime_mask(df: pd.DataFrame, mode: str, lookback: int = 500) -> np.ndarray:
    atr = I.atr(df, 14)
    if mode in ("vol_normal_or_high", "vol_high", "vol_low"):
        # rolling percentile rank of ATR versus its own trailing window (no future data)
        pct = atr.rolling(lookback, min_periods=max(50, lookback // 5)).apply(lambda w: (w[:-1] < w[-1]).mean() * 100.0, raw=True)
        if mode == "vol_normal_or_high":
            m = pct >= 33
        elif mode == "vol_high":
            m = pct >= 67
        else:
            m = pct <= 67
    elif mode == "trending":
        m = I.adx(df, 14) >= 20
    elif mode == "ranging":
        m = I.adx(df, 14) < 20
    else:
        raise ValueError(f"Unknown regime filter '{mode}'.")
    return m.fillna(False).to_numpy(bool)


def apply_regime_filter(result: StrategyResult, df: pd.DataFrame, mode: str, lookback: int = 500) -> tuple[StrategyResult, str]:
    """Blocks NEW entries on bars where the regime test is false. An open position is left to its own exits."""
    if getattr(result, "entry_orders", None):
        return result, "Regime filter skipped: this strategy places resting limit orders."
    ok = regime_mask(df, mode, lookback)
    sig = result.signals.to_numpy(int)
    out = np.zeros(len(sig), dtype=int)
    prev, pos, entries, allowed = 0, 0, 0, 0
    for i in range(len(sig)):
        s = sig[i]
        if s != prev:
            if s != 0:
                entries += 1
                if ok[i]:
                    allowed += 1
                    pos = s
                else:
                    pos = 0
            else:
                pos = 0
        prev = s
        out[i] = pos
    res = StrategyResult(**{**result.__dict__, "signals": pd.Series(out, index=result.signals.index)})
    return res, f"Regime filter ({REGIME_MODES.get(mode, mode)}): {allowed} of {entries} entries kept."
