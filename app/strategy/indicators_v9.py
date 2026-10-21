"""
v9 indicator extensions (2026-10-17): gamma-exposure + price-structure terminals.

These are the indicator "capabilities" from the Gamma Exposure video this release was modelled on, rebuilt so they can
run inside every strategy-creation tool (Search Lab, Evolution Lab, Manual builder, Cross-instrument, Research Loop):

    gex_regime                +1 positive-gamma day (dealers dampen moves) / -1 negative-gamma day (they amplify them)
    gex_flip_distance         (close - gamma flip) / ATR           >0 = above the flip
    gex_call_wall_distance    (call wall - close) / ATR            small = pressing on gamma resistance
    gex_put_wall_distance     (close - put wall) / ATR             small = sitting on gamma support
    theta_clock               0..1 progress through the 08:30-15:00 CT session (0DTE decay builds into the close)
    gamma_convexity           ATR-normalised price acceleration (close - 2*close[p] + close[2p]) / ATR
    ema_stack_score           -1..+1 trend stack of EMA(p) / EMA(2p) / EMA(4p)
    bull_flag_score           impulse-pole strength (ATR) when the last `p` bars form a tight flag, signed (+bull/-bear)
    volume_expansion          3-bar mean volume / 20-bar mean volume

Data honesty. The gex_* kinds use REAL levels when they exist (frame columns gex_flip / gex_call_wall / gex_put_wall /
gex_net, or -- when frame.attrs["instrument"] is set -- the daily snapshots in data/gamma via
app.data.gamma_data.merge_gamma_levels, applied with a one-day lag so there is no lookahead). Any bar without a real level
falls back to a price-derived PROXY so a backtest never fails on missing data:
    flip  -> prior-day typical price (H+L+C)/3     call wall -> prior-day high     put wall -> prior-day low
    regime-> positive when ATR percentile < 50 (calm tape ~ long-gamma), negative otherwise
The proxies are ordinary prior-day levels; they are labelled as proxies in the Gamma Center and in the strategy library.

Lookahead safety: only trailing windows and .shift(+n) are used; prior-day levels come from completed days only.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from app.strategy.indicators import _period, atr, ema


def _ts(frame: pd.DataFrame) -> pd.Series | None:
    """Exchange-clock timestamps (P0, report 2026-10-20): stamped ``exchange_time`` / source_timezone conversion when the
    frame has clock provenance, else the raw stamps (legacy)."""
    if "timestamp" in frame.columns or isinstance(frame.index, pd.DatetimeIndex):
        from app.data.session_clock import frame_exchange_time
        return frame_exchange_time(frame)
    return None


def prior_day_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """Previous COMPLETED calendar day's high / low / typical price for every bar (NaN on the first day)."""
    ts = _ts(frame)
    if ts is None:  # no clock: use the previous 288 bars as a stand-in "day"
        hi = frame["high"].rolling(288, min_periods=48).max().shift(1)
        lo = frame["low"].rolling(288, min_periods=48).min().shift(1)
        cl = frame["close"].shift(1)
        return pd.DataFrame({"pd_high": hi, "pd_low": lo, "pd_typ": (hi + lo + cl) / 3.0}, index=frame.index)
    from app.data.session_clock import session_day_key
    day = session_day_key(frame) if "timestamp" in frame.columns else (ts.dt.tz_localize(None).dt.normalize() if ts.dt.tz is not None else ts.dt.normalize())
    g = pd.DataFrame({"day": day.values, "high": frame["high"].values, "low": frame["low"].values,
                      "close": frame["close"].values})
    daily = g.groupby("day").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    daily["typ"] = (daily["high"] + daily["low"] + daily["close"]) / 3.0
    prev = daily.shift(1)
    mapped = pd.DataFrame({"pd_high": day.map(prev["high"]).values, "pd_low": day.map(prev["low"]).values,
                           "pd_typ": day.map(prev["typ"]).values}, index=frame.index)
    return mapped


def _real_gex_columns(frame: pd.DataFrame) -> pd.DataFrame | None:
    cols = ["gex_net", "gex_flip", "gex_call_wall", "gex_put_wall"]
    if all(c in frame.columns for c in cols[1:]):
        return frame[[c for c in cols if c in frame.columns]]
    inst = frame.attrs.get("instrument") if hasattr(frame, "attrs") else None
    if not inst or _ts(frame) is None:
        return None
    try:
        from app.data import gamma_data as gd
        if gd.load_levels(inst).empty and gd.load_squeezemetrics_history().empty:
            return None
        base = frame if "timestamp" in frame.columns else frame.assign(timestamp=_ts(frame))
        merged = gd.merge_gamma_levels(base, inst)
        return merged[[c for c in cols if c in merged.columns]]
    except Exception:  # noqa: BLE001 - gamma data must never break a backtest
        return None


def gex_levels(frame: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """flip / call_wall / put_wall / regime per bar: real where available, proxy elsewhere. Also returns `is_real`."""
    pdl = prior_day_levels(frame)
    out = pd.DataFrame(index=frame.index)
    out["flip"], out["call_wall"], out["put_wall"] = pdl["pd_typ"], pdl["pd_high"], pdl["pd_low"]
    atr_s = atr(frame, _period(period))
    rank = atr_s.rolling(100, min_periods=20).apply(lambda x: float((x <= x[-1]).mean()), raw=True)
    out["regime"] = np.where(rank.fillna(0.5) < 0.5, 1.0, -1.0)
    out["is_real"] = 0.0
    real = _real_gex_columns(frame)
    if real is not None:
        for src, dst in (("gex_flip", "flip"), ("gex_call_wall", "call_wall"), ("gex_put_wall", "put_wall")):
            if src in real.columns:
                m = real[src].notna()
                out.loc[m, dst] = real.loc[m, src]
                out.loc[m, "is_real"] = 1.0
        if "gex_net" in real.columns:
            m = real["gex_net"].notna()
            out.loc[m, "regime"] = np.where(real.loc[m, "gex_net"] >= 0, 1.0, -1.0)
    return out


def gex_regime(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    return gex_levels(frame, period)["regime"].astype(float)


def gex_flip_distance(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    lv = gex_levels(frame, period)
    return ((frame["close"] - lv["flip"]) / atr(frame, _period(period)).replace(0, np.nan)).fillna(0.0)


def gex_call_wall_distance(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    lv = gex_levels(frame, period)
    return ((lv["call_wall"] - frame["close"]) / atr(frame, _period(period)).replace(0, np.nan)).fillna(0.0)


def gex_put_wall_distance(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    lv = gex_levels(frame, period)
    return ((frame["close"] - lv["put_wall"]) / atr(frame, _period(period)).replace(0, np.nan)).fillna(0.0)


def theta_clock(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Progress through the 08:30-15:00 CT cash session, 0 before the open and 1 after the close. Uses the bar's own
    clock only, so it is causal by construction. Without timestamps it reads 0."""
    ts = _ts(frame)
    if ts is None:
        return pd.Series(0.0, index=frame.index)
    minutes = ts.dt.hour * 60 + ts.dt.minute
    from app.data.session_clock import has_clock_provenance
    # 08:30 CT == 09:30 ET: exchange-clock frames (stamped) are on the New York clock, legacy frames on the old CT reading.
    open_min = 570 if has_clock_provenance(frame) else 510
    return ((minutes - open_min) / 390.0).clip(0.0, 1.0).astype(float).set_axis(frame.index)


def gamma_convexity(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    p = _period(period)
    c = frame["close"]
    accel = c - 2.0 * c.shift(p) + c.shift(2 * p)
    return (accel / atr(frame, p).replace(0, np.nan)).fillna(0.0)


def ema_stack_score(frame: pd.DataFrame, period: int = 9, series: pd.Series | None = None) -> pd.Series:
    p = _period(period)
    c = series if series is not None else frame["close"]
    e1, e2, e3 = ema(c, p), ema(c, 2 * p), ema(c, 4 * p)
    return ((np.sign(e1 - e2) + np.sign(e2 - e3)) / 2.0).astype(float).fillna(0.0)


def bull_flag_score(frame: pd.DataFrame, period: int = 6) -> pd.Series:
    """Impulse pole (previous 3p bars, in ATR) when the latest p bars are a tight consolidation (range < half the pole).
    Positive = bull flag, negative = bear flag, 0 = no flag. Pole and flag windows end at the current bar, so the
    series only uses past and current data."""
    p = _period(period)
    a = atr(frame, 14).replace(0, np.nan)
    pole = (frame["close"].shift(p) - frame["close"].shift(4 * p)) / a
    flag_range = (frame["high"].rolling(p, min_periods=p).max() - frame["low"].rolling(p, min_periods=p).min()) / a
    tight = flag_range < (pole.abs() * 0.5)
    return pole.where(tight & (pole.abs() >= 2.0), 0.0).fillna(0.0)


def volume_expansion(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    p = _period(period)
    vol = frame["volume"].astype(float)
    return (vol.rolling(3, min_periods=3).mean() / vol.rolling(p, min_periods=p).mean().replace(0, np.nan)).fillna(1.0)


V9IndicatorBuilder = Callable[[pd.DataFrame, int, str, int | None], pd.Series]

V9_INDICATOR_BUILDERS: dict[str, V9IndicatorBuilder] = {
    "gex_regime": lambda f, p, c, l: gex_regime(f, p),
    "gex_flip_distance": lambda f, p, c, l: gex_flip_distance(f, p),
    "gex_call_wall_distance": lambda f, p, c, l: gex_call_wall_distance(f, p),
    "gex_put_wall_distance": lambda f, p, c, l: gex_put_wall_distance(f, p),
    "theta_clock": lambda f, p, c, l: theta_clock(f, p),
    "gamma_convexity": lambda f, p, c, l: gamma_convexity(f, p),
    "ema_stack_score": lambda f, p, c, l: ema_stack_score(f, p, f[c] if c in f.columns else None),
    "bull_flag_score": lambda f, p, c, l: bull_flag_score(f, p),
    "volume_expansion": lambda f, p, c, l: volume_expansion(f, p),
}

V9_BOUNDED_RANGES: dict[str, tuple[float, float]] = {
    "gex_regime": (-1.0, 1.0),
    "ema_stack_score": (-1.0, 1.0),
    "theta_clock": (0.0, 1.0),
}

V9_INDICATOR_KINDS: tuple[str, ...] = tuple(V9_INDICATOR_BUILDERS)

# One-line plain-English meaning of each kind, shown by the Gamma Center and the clickable metric popups.
V9_DESCRIPTIONS: dict[str, str] = {
    "gex_regime": "+1 when dealers are net long gamma (moves get dampened), -1 when net short (moves get amplified).",
    "gex_flip_distance": "ATR distance of price from the gamma-flip level; positive = above it.",
    "gex_call_wall_distance": "ATR distance up to the largest call-gamma strike (gamma resistance).",
    "gex_put_wall_distance": "ATR distance down to the largest put-gamma strike (gamma support).",
    "theta_clock": "0-1 progress through the cash session; option decay and pinning intensify into the close.",
    "gamma_convexity": "ATR-normalised price acceleration: is the move speeding up or fading?",
    "ema_stack_score": "-1 to +1 alignment of EMA(p) / EMA(2p) / EMA(4p).",
    "bull_flag_score": "Strength of the impulse pole (in ATR) when the last bars form a tight flag; sign = direction.",
    "volume_expansion": "Recent 3-bar volume relative to the 20-bar average.",
}
