"""Regime-adaptive RSI zones + causal divergence detector.

Port of astra-quant-agent's ``classify_rsi_zone``
(``scripts/factors/okx_quant_factors.py:452-483``) and ``detect_divergence``
(``:324-356``), re-expressed as per-bar, fully causal pandas series for the
T58 strategy engine.

Port rules (per the Oct-4 deep analysis, Part A ports #2):
* Trend regime comes from T58's EXISTING detectors --
  ``calculate_hh_ll_structure`` in ``app.quant_lab.market_structure`` --
  never a parallel regime system. The per-swing trend is forward-filled onto
  bars and shifted by the swing confirmation window (``right``) so a bar's
  regime only ever uses confirmed swings.
* RSI comes from T58's own ``app.strategy.indicators.rsi`` -- no second RSI
  definition is introduced (same rule the astra source holds itself to).
* Everything is causal: bar ``i``'s outputs use bars ``<= i`` only. The
  behavioral lookahead gate (``app.strategy.lookahead_check``) must stay
  clean -- no centered windows, no negative shifts, no full-frame groupby
  statistics leaking future bars into the past.
* ``zone_bounds`` and ``lookback`` are exposed parameters so the search can
  tune them (they are plain numbers/dicts on the Manual operand).

Why it matters: static 30/70 RSI fails in trends (bull markets park RSI at
65-80). Regime-conditional bands turn that known false-discovery class into
regime-aware entries: buy pullbacks in bull regimes, sell rallies in bear
regimes, fade extremes only in ranges.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.quant_lab.market_structure import calculate_hh_ll_structure
from app.strategy.indicators import rsi as t58_rsi

# ---------------------------------------------------------------------------
# Zone bounds -- astra's defaults, exposed so the search can tune them.
# ---------------------------------------------------------------------------

DEFAULT_ZONE_BOUNDS: dict[str, object] = {
    # Bull regime: buy the pullback, don't chase the extension.
    "bull_pullback": (38.0, 52.0),   # BULL_PULLBACK_BUY zone [lo, hi)
    "bull_healthy": (52.0, 72.0),    # BULL_HEALTHY zone [lo, hi]
    "bull_overbought": 75.0,         # > this: OVERBOUGHT_NO_CHASE
    # Bear regime: sell the rally, don't chase the flush.
    "bear_rally": (46.0, 62.0),      # BEAR_RALLY_SELL zone (lo, hi]
    "bear_healthy": (28.0, 46.0),    # BEAR_HEALTHY zone [lo, hi]
    "bear_oversold": 25.0,           # < this: OVERSOLD_NO_CHASE
    # Range regime: classic mean-reversion extremes.
    "range_oversold": 32.0,          # <= this: RANGE_OVERSOLD
    "range_overbought": 68.0,        # >= this: RANGE_OVERBOUGHT
}

# Zone labels (kept identical to the astra source).
BULL_PULLBACK_BUY = "BULL_PULLBACK_BUY"
BULL_HEALTHY = "BULL_HEALTHY"
OVERBOUGHT_NO_CHASE = "OVERBOUGHT_NO_CHASE"
BEAR_RALLY_SELL = "BEAR_RALLY_SELL"
BEAR_HEALTHY = "BEAR_HEALTHY"
OVERSOLD_NO_CHASE = "OVERSOLD_NO_CHASE"
RANGE_OVERSOLD = "RANGE_OVERSOLD"
RANGE_OVERBOUGHT = "RANGE_OVERBOUGHT"
NEUTRAL = "NEUTRAL"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# Numeric encodings for the Manual builder (conditions compare numbers).
TREND_BULL = 1.0
TREND_BEAR = -1.0
TREND_RANGE = 0.0

DIVERGENCE_BULLISH = 1.0
DIVERGENCE_BEARISH = -1.0
DIVERGENCE_NONE = 0.0


def _merged_bounds(zone_bounds: dict | None) -> dict:
    """DEFAULT_ZONE_BOUNDS with any caller overrides applied (shallow)."""
    bounds = dict(DEFAULT_ZONE_BOUNDS)
    if zone_bounds:
        for key, value in zone_bounds.items():
            if key in bounds:
                bounds[key] = value
    return bounds


def _normalize_trend(trend: str | None) -> str:
    """Map T58 detector vocabulary ('up'/'down'/'range') and astra
    vocabulary ('bull'/'bear') onto 'bull' / 'bear' / 'range'."""
    t = str(trend or "").strip().lower()
    if "bull" in t or t == "up":
        return "bull"
    if "bear" in t or t == "down":
        return "bear"
    return "range"


def classify_rsi_zone_scalar(
    rsi_value: float | None,
    trend: str | None,
    zone_bounds: dict | None = None,
) -> str:
    """Scalar port of astra's ``classify_rsi_zone`` -- one RSI value, one
    trend label -> one zone label. Missing RSI never degrades to a neutral
    reading; it is explicitly INSUFFICIENT_DATA."""
    if rsi_value is None:
        return INSUFFICIENT_DATA
    try:
        r = float(rsi_value)
    except (TypeError, ValueError):
        return INSUFFICIENT_DATA
    if not np.isfinite(r):
        return INSUFFICIENT_DATA
    b = _merged_bounds(zone_bounds)
    t = _normalize_trend(trend)
    if t == "bull":
        if r > float(b["bull_overbought"]):
            return OVERBOUGHT_NO_CHASE
        lo, hi = b["bull_pullback"]
        if float(lo) <= r < float(hi):
            return BULL_PULLBACK_BUY
        lo, hi = b["bull_healthy"]
        if float(lo) <= r <= float(hi):
            return BULL_HEALTHY
        return NEUTRAL
    if t == "bear":
        if r < float(b["bear_oversold"]):
            return OVERSOLD_NO_CHASE
        lo, hi = b["bear_healthy"]
        if float(lo) <= r <= float(hi):
            return BEAR_HEALTHY
        lo, hi = b["bear_rally"]
        if float(lo) < r <= float(hi):
            return BEAR_RALLY_SELL
        return NEUTRAL
    if r <= float(b["range_oversold"]):
        return RANGE_OVERSOLD
    if r >= float(b["range_overbought"]):
        return RANGE_OVERBOUGHT
    return NEUTRAL


def trend_regime_series(
    df: pd.DataFrame,
    *,
    left: int = 5,
    right: int = 5,
) -> pd.Series:
    """Per-bar trend regime from T58's EXISTING swing-structure detector.

    ``calculate_hh_ll_structure`` labels each confirmed fractal swing
    up/down/range. A swing at bar ``i`` is only *confirmed* once ``right``
    further bars have printed, so each swing's trend is planted at its
    confirmation bar (``index + right``) and forward-filled -- bar ``i``'s
    regime therefore only ever reflects swings confirmed at or before ``i``.
    Bars before the first confirmation read ``"range"`` (unknown, not bull).
    """
    n = len(df)
    out = pd.Series("range", index=df.index, dtype=object)
    if n < left + right + 2:
        return out
    hh_ll = calculate_hh_ll_structure(df, left=left, right=right)
    if hh_ll.empty:
        return out
    # Plant each swing's trend at its confirmation bar, then forward-fill.
    planted = pd.Series(np.nan, index=np.arange(n), dtype=object)
    for _, row in hh_ll.iterrows():
        try:
            pos = int(row["index"]) + int(right)
        except (TypeError, ValueError):
            continue
        if 0 <= pos < n:
            planted.iloc[pos] = _normalize_trend(row.get("trend"))
    planted = planted.ffill()
    mask = planted.notna()
    out = out.where(~mask, planted)
    return out.astype(object)


def rsi_series(df: pd.DataFrame, *, rsi_period: int = 14) -> pd.Series:
    """Per-bar RSI using T58's own indicator (Wilder-style ewm, causal)."""
    return t58_rsi(df["close"], period=max(int(rsi_period), 1))


def rsi_zone_series(
    df: pd.DataFrame,
    *,
    rsi_period: int = 14,
    zone_bounds: dict | None = None,
    trend_left: int = 5,
    trend_right: int = 5,
    trend: pd.Series | None = None,
) -> pd.Series:
    """Per-bar RSI zone label (``classify_rsi_zone_scalar`` vectorized over
    bars). ``trend`` may be supplied directly (a per-bar 'bull'/'bear'/
    'range' -- or 'up'/'down'/'range' -- series); otherwise it is derived
    from the existing market-structure detector."""
    rsi = rsi_series(df, rsi_period=rsi_period)
    trend_s = trend if trend is not None else trend_regime_series(df, left=trend_left, right=trend_right)
    bounds = _merged_bounds(zone_bounds)
    labels = [
        classify_rsi_zone_scalar(r, t, bounds)
        for r, t in zip(rsi.to_numpy(), trend_s.to_numpy())
    ]
    return pd.Series(labels, index=df.index, dtype=object)


def regime_flag_series(
    df: pd.DataFrame,
    kind: str,
    *,
    rsi_period: int = 14,
    zone_bounds: dict | None = None,
    trend_left: int = 5,
    trend_right: int = 5,
    trend: pd.Series | None = None,
) -> pd.Series:
    """Numeric Manual-builder operands derived from the RSI zones.

    kind:
      "rsi_regime"    -> +1 bull / -1 bear / 0 range (per-bar, causal)
      "rsi_zone_buy"  -> 1 when the bar is in BULL_PULLBACK_BUY, else 0
      "rsi_zone_sell" -> 1 when the bar is in BEAR_RALLY_SELL, else 0
    """
    kind = str(kind).lower()
    if kind == "rsi_regime":
        trend_s = trend if trend is not None else trend_regime_series(df, left=trend_left, right=trend_right)
        code = trend_s.map({"bull": TREND_BULL, "bear": TREND_BEAR, "range": TREND_RANGE})
        return code.fillna(TREND_RANGE).astype(float)
    if kind in ("rsi_zone_buy", "rsi_zone_sell"):
        zones = rsi_zone_series(
            df, rsi_period=rsi_period, zone_bounds=zone_bounds,
            trend_left=trend_left, trend_right=trend_right, trend=trend,
        )
        want = BULL_PULLBACK_BUY if kind == "rsi_zone_buy" else BEAR_RALLY_SELL
        return (zones == want).astype(float)
    raise ValueError(f"Unknown regime flag kind '{kind}'.")


def divergence_signals(
    closes: np.ndarray,
    oscillator: np.ndarray,
    *,
    lookback: int = 20,
    eps: float = 1e-9,
) -> np.ndarray:
    """Vectorized, causal port of astra's ``detect_divergence``.

    For every bar ``i`` (with at least ``lookback`` bars of history), the
    trailing ``lookback``-bar window is split in halves and the extrema of
    the two halves are compared -- exactly the scalar algorithm:

    * price makes a LOWER low while the oscillator's low RISES -> +1
      (bullish: downside momentum is exhausted);
    * price makes a HIGHER high while the oscillator's high FALLS -> -1
      (bearish: upside momentum is exhausted);
    * otherwise 0.

    ``argmin``/``argmax`` return the FIRST extremum, matching the scalar
    ``min(range(...), key=...)`` tie-break. Chunked so the strided window
    view stays memory-bounded on long frames. Bullish takes precedence on
    the (rare) bar where both fire, as in the source.
    """
    closes = np.asarray(closes, dtype=float)
    oscillator = np.asarray(oscillator, dtype=float)
    n = min(len(closes), len(oscillator))
    out = np.zeros(n, dtype=np.float64)
    lookback = max(6, int(lookback))
    if n < lookback:
        return out
    half = lookback // 2
    second_len = lookback - half
    chunk = 50_000
    ar = None
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        # Windows ending inside [start, end) need `lookback - 1` bars of
        # history before `start`.
        s0 = max(0, start - (lookback - 1))
        c = closes[s0:end]
        o = oscillator[s0:end]
        m = len(c)
        if m < lookback:
            continue
        n_win = m - lookback + 1
        strides = (c.strides[0], c.strides[0])
        wc = np.lib.stride_tricks.as_strided(c, shape=(n_win, lookback), strides=strides)
        wo = np.lib.stride_tricks.as_strided(o, shape=(n_win, lookback), strides=strides)
        first_c, second_c = wc[:, :half], wc[:, half:]
        first_o, second_o = wo[:, :half], wo[:, half:]
        if ar is None or len(ar) < n_win:
            ar = np.arange(n_win)
        else:
            ar = ar[:n_win]
        # Bullish: lower low in price, higher low in the oscillator.
        i1 = np.argmin(first_c, axis=1)
        i2 = half + np.argmin(second_c, axis=1)
        bull = (
            (second_c[ar, i2 - half] < first_c[ar, i1] - eps)
            & (second_o[ar, i2 - half] > first_o[ar, i1] + eps)
        )
        # Bearish: higher high in price, lower high in the oscillator.
        j1 = np.argmax(first_c, axis=1)
        j2 = half + np.argmax(second_c, axis=1)
        bear = (
            (second_c[ar, j2 - half] > first_c[ar, j1] + eps)
            & (second_o[ar, j2 - half] < first_o[ar, j1] - eps)
        )
        vals = np.where(bull, DIVERGENCE_BULLISH, np.where(bear, DIVERGENCE_BEARISH, DIVERGENCE_NONE))
        # Window row r ends at bar s0 + r + lookback - 1.
        out[s0 + lookback - 1: end] = vals
    return out


def rsi_divergence_series(
    df: pd.DataFrame,
    *,
    rsi_period: int = 14,
    lookback: int = 20,
    eps: float = 1e-9,
) -> pd.Series:
    """Per-bar price-vs-RSI divergence: +1 bullish / -1 bearish / 0 none.

    The oscillator is T58's own RSI series; the detector only ever looks at
    the trailing ``lookback`` bars ending at (and including) each bar, so it
    is causal by construction.
    """
    rsi = rsi_series(df, rsi_period=rsi_period).to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    vals = divergence_signals(closes, rsi, lookback=lookback, eps=eps)
    return pd.Series(vals, index=df.index, dtype=float)
