"""v10 indicator extensions: the research labs' findings, exposed as ordinary indicators.

Every strategy-creation engine (Search Lab, Evolution Lab, Manual builder, Cross-instrument, Research Loop) builds
indicators through ``build_indicator_series``; registering the research labs' causal features here is what lets them
*feed* those engines instead of living on a dashboard.

    vp_dist_poc        (close - PRIOR session POC) / ATR
    vp_location        +1 above the prior value area, -1 below it, 0 inside
    vp_in_lvn          1 when price sits at one of the prior session's low-volume nodes (thin air), else 0
    flow_delta_z       z-score of bar delta (TRUE bid/ask delta when the frame has bid_volume/ask_volume, else a labelled
                       close-location PROXY)
    flow_cvd_slope     cumulative-delta slope normalised by volume
    flow_absorption    +1 sell aggression that failed to push price down, -1 buy aggression that failed to lift it
    spread_zscore      z-score of the spread vs ``frame.attrs['spread_partner']`` (a second OHLCV frame), else 0
    leader_return      the leader's return (``frame.attrs['leader']``) over the last ``p`` bars, in its own ATR, else 0
    event_window       1 on the bars right after a release in ``frame.attrs['events']`` (``p`` bars), else 0

Lookahead safety: volume-profile levels come from the PRIOR session; flow features use only the current and earlier bars;
the partner / leader frames are aligned with ``merge_asof(direction='backward')``; events flag only bars at/after release.
Missing auxiliary data never raises -- the indicator returns the neutral value (0) so a backtest cannot crash on it.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np
import pandas as pd

V10IndicatorBuilder = Callable[[pd.DataFrame, int, str, int], pd.Series]
_CACHE: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
_CACHE_MAX = 2


def _key(frame: pd.DataFrame, tag: str) -> tuple | None:
    try:
        n = len(frame)
        ts = frame["timestamp"]
        sample = frame["close"].to_numpy()[:: max(1, n // 32)]
        return (tag, n, str(ts.iloc[0]), str(ts.iloc[-1]), hash(np.round(sample, 8).tobytes()), "bid_volume" in frame.columns)
    except Exception:  # noqa: BLE001
        return None


def _cached(frame: pd.DataFrame, tag: str, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    k = _key(frame, tag)
    if k is not None and k in _CACHE:
        _CACHE.move_to_end(k)
        out = _CACHE[k]
        if len(out) == len(frame):
            return out
    out = fn(frame)
    if k is not None:
        _CACHE[k] = out
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return out


def _base(frame: pd.DataFrame) -> pd.DataFrame | None:
    """A copy with a real ``timestamp`` column the labs can use (None when the frame has no clock)."""
    if "timestamp" in frame.columns:
        d = frame.copy()
    elif isinstance(frame.index, pd.DatetimeIndex):
        d = frame.reset_index().rename(columns={frame.index.name or "index": "timestamp"})
    else:
        return None
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if getattr(d["timestamp"].dt, "tz", None) is not None:
        d["timestamp"] = d["timestamp"].dt.tz_localize(None)
    return d.reset_index(drop=True)


def _zeros(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=frame.index)


def _vp(frame: pd.DataFrame) -> pd.DataFrame:
    def build(f: pd.DataFrame) -> pd.DataFrame:
        from app.research_labs.volume_profile import profile_feature_frame

        d = _base(f)
        if d is None or "volume" not in d.columns or float(d["volume"].sum()) <= 0:
            return pd.DataFrame(index=f.index)
        out = profile_feature_frame(d)
        out.index = f.index
        return out

    return _cached(frame, "vp", build)


def _flow(frame: pd.DataFrame) -> pd.DataFrame:
    def build(f: pd.DataFrame) -> pd.DataFrame:
        from app.research_labs.order_flow import flow_frame

        d = f.reset_index(drop=True)
        if "volume" not in d.columns:
            return pd.DataFrame(index=f.index)
        out = flow_frame(d)
        out.index = f.index
        return out

    return _cached(frame, "flow", build)


def _col(frame: pd.DataFrame, source: pd.DataFrame, name: str) -> pd.Series:
    if name not in source.columns:
        return _zeros(frame)
    return source[name].astype(float).fillna(0.0)


def vp_dist_poc(f, p, c, l):
    return _col(f, _vp(f), "vp_dist_poc")


def vp_location(f, p, c, l):
    return _col(f, _vp(f), "vp_location")


def vp_in_lvn(f, p, c, l):
    return _col(f, _vp(f), "vp_in_lvn")


def flow_delta_z(f, p, c, l):
    return _col(f, _flow(f), "delta_z")


def flow_cvd_slope(f, p, c, l):
    return _col(f, _flow(f), "cvd_slope")


def flow_absorption(f, p, c, l):
    return _col(f, _flow(f), "absorption")


def _aux(frame: pd.DataFrame, key: str) -> pd.DataFrame | None:
    other = frame.attrs.get(key) if hasattr(frame, "attrs") else None
    if not isinstance(other, pd.DataFrame) or other.empty:
        return None
    return _base(other)


def spread_zscore(f, p, c, l):
    other, base = _aux(f, "spread_partner"), _base(f)
    if other is None or base is None:
        return _zeros(f)
    try:
        from app.research_labs.stat_arb import spread_zscore_feature

        z = spread_zscore_feature(base, other, freq="1h", z_window=max(10, int(p) or 60))
        z.index = f.index
        return z.fillna(0.0)
    except Exception:  # noqa: BLE001
        return _zeros(f)


def leader_return(f, p, c, l):
    other, base = _aux(f, "leader"), _base(f)
    if other is None or base is None:
        return _zeros(f)
    try:
        n = max(1, int(p))
        o = other.sort_values("timestamp")
        tr = (o["high"] - o["low"]).rolling(14, min_periods=3).mean().shift(1)
        o["lead_ret"] = (o["close"] - o["close"].shift(n)) / tr.replace(0, np.nan)
        m = pd.merge_asof(base[["timestamp"]].sort_values("timestamp"), o[["timestamp", "lead_ret"]], on="timestamp", direction="backward")
        return pd.Series(m["lead_ret"].fillna(0.0).to_numpy(), index=f.index)
    except Exception:  # noqa: BLE001
        return _zeros(f)


def event_window(f, p, c, l):
    events, base = f.attrs.get("events") if hasattr(f, "attrs") else None, _base(f)
    if events is None or base is None:
        return _zeros(f)
    try:
        from app.research_labs.event_study import event_flag_feature

        flag = event_flag_feature(base, events, window_bars=max(1, int(p)))
        return pd.Series(flag.to_numpy(), index=f.index)
    except Exception:  # noqa: BLE001
        return _zeros(f)


V10_INDICATOR_BUILDERS: dict[str, V10IndicatorBuilder] = {
    "vp_dist_poc": vp_dist_poc, "vp_location": vp_location, "vp_in_lvn": vp_in_lvn,
    "flow_delta_z": flow_delta_z, "flow_cvd_slope": flow_cvd_slope, "flow_absorption": flow_absorption,
    "spread_zscore": spread_zscore, "leader_return": leader_return, "event_window": event_window,
}
V10_INDICATOR_KINDS: tuple[str, ...] = tuple(V10_INDICATOR_BUILDERS)
# Kinds that need only the frame itself -> safe to hand to the random-search grammar.
V10_SELF_CONTAINED: tuple[str, ...] = ("vp_dist_poc", "vp_location", "vp_in_lvn", "flow_delta_z", "flow_cvd_slope", "flow_absorption")
V10_BOUNDED_RANGES: dict[str, tuple[float, float]] = {"vp_location": (-1.0, 1.0), "vp_in_lvn": (0.0, 1.0),
                                                      "flow_absorption": (-1.0, 1.0), "event_window": (0.0, 1.0)}
V10_DESCRIPTIONS: dict[str, str] = {
    "vp_dist_poc": "ATR distance of price from the prior session's point of control (most-traded price).",
    "vp_location": "+1 above / -1 below / 0 inside the prior session's value area.",
    "vp_in_lvn": "1 when price is at a low-volume node of the prior session (little acceptance -> fast moves).",
    "flow_delta_z": "z-score of bar delta (true bid/ask when the data has it, otherwise a labelled proxy).",
    "flow_cvd_slope": "Cumulative-delta slope per unit volume: is aggression building in one direction?",
    "flow_absorption": "+1 sell aggression absorbed (bullish) / -1 buy aggression absorbed (bearish).",
    "spread_zscore": "Z-score of the spread to a partner instrument (needs frame.attrs['spread_partner']).",
    "leader_return": "Leader instrument's recent move in its own ATR (needs frame.attrs['leader']).",
    "event_window": "1 for the bars right after a scheduled release (needs frame.attrs['events']).",
}
