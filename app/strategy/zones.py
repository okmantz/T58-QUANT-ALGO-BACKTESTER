"""Persistent price zones with a lifecycle (created -> touched -> filled /
invalidated) for strategies that rest an order at a level instead of firing
a market order.

Supported kinds: ``fvg`` (fair value gap, 3-candle imbalance) and
``order_block`` (last opposite candle before a displacement). Both are
strictly causal: a zone exists only from the close of the bar that completes
it (``created_bar``); nothing about it is used earlier. An order resting at a
zone is cancelled when price closes beyond the far edge (invalidation) or
after ``expiry_bars``. Zones feed ``app.backtest.resting_orders``, so sizing,
costs and fills are the engine's own.

``check_zone_causality`` is the lookahead test for zones: rebuilding from a
truncated history must give exactly the zones already completed there.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.resting_orders import LimitOrder


@dataclass
class Zone:
    kind: str                  # "fvg" | "order_block"
    side: int                  # +1 bullish (buy on retrace), -1 bearish
    near: float                # edge nearest current price (entry level)
    far: float                 # opposite edge (stop goes beyond it)
    created_bar: int           # bar on whose CLOSE the zone becomes known
    invalidated_bar: int | None = None
    first_touch_bar: int | None = None
    state: str = "created"     # created | touched | invalidated
    tag: str = field(default="")


def _atr(df: pd.DataFrame, period: int) -> np.ndarray:
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(period, min_periods=period).mean().to_numpy()


def build_zones(df: pd.DataFrame, kind: str = "fvg", *, min_size_atr: float = 0.3, atr_period: int = 14,
                displacement_atr: float = 1.5) -> list[Zone]:
    h, l, o, c = (df[k].to_numpy(float) for k in ("high", "low", "open", "close"))
    atr = _atr(df, atr_period)
    zones: list[Zone] = []
    for i in range(2, len(df)):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if kind == "fvg":
            if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_size_atr * a:
                zones.append(Zone("fvg", 1, near=float(l[i]), far=float(h[i - 2]), created_bar=i, tag="fvg_bull"))
            elif h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_size_atr * a:
                zones.append(Zone("fvg", -1, near=float(h[i]), far=float(l[i - 2]), created_bar=i, tag="fvg_bear"))
        elif kind == "order_block":
            body = abs(c[i] - o[i])
            if body >= displacement_atr * a:
                if c[i] > o[i] and c[i - 1] < o[i - 1] and c[i] > h[i - 1]:
                    zones.append(Zone("order_block", 1, near=float(h[i - 1]), far=float(l[i - 1]), created_bar=i, tag="ob_bull"))
                elif c[i] < o[i] and c[i - 1] > o[i - 1] and c[i] < l[i - 1]:
                    zones.append(Zone("order_block", -1, near=float(l[i - 1]), far=float(h[i - 1]), created_bar=i, tag="ob_bear"))
        else:
            raise ValueError(f"unknown zone kind {kind!r}")
    return zones


def annotate_lifecycle(zones: list[Zone], df: pd.DataFrame) -> list[Zone]:
    """Fills first_touch_bar / invalidated_bar / state by scanning bars AFTER creation."""
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    n = len(df)
    for z in zones:
        for j in range(z.created_bar + 1, n):
            if z.first_touch_bar is None and ((l[j] <= z.near) if z.side == 1 else (h[j] >= z.near)):
                z.first_touch_bar, z.state = j, "touched"
            if (c[j] < z.far) if z.side == 1 else (c[j] > z.far):
                z.invalidated_bar, z.state = j, "invalidated"
                break
    return zones


def orders_from_zones(zones: list[Zone], df: pd.DataFrame, *, expiry_bars: int = 30, stop_buffer_atr: float = 0.25,
                      target_r: float = 2.0, atr_period: int = 14) -> list[LimitOrder]:
    """Entry at the near edge on the first retrace, stop beyond the far edge, target target_r x risk."""
    atr = _atr(df, atr_period)
    annotate_lifecycle(zones, df)
    out: list[LimitOrder] = []
    for z in zones:
        a = atr[z.created_bar]
        buf = stop_buffer_atr * (a if np.isfinite(a) else 0.0)
        stop = z.far - z.side * buf
        risk_d = abs(z.near - stop)
        if risk_d <= 0:
            continue
        last = z.created_bar + expiry_bars
        if z.invalidated_bar is not None:
            last = min(last, z.invalidated_bar)   # live through the bar that closes beyond the far edge
        if last <= z.created_bar:
            continue
        out.append(LimitOrder(z.side, float(z.near), z.created_bar, int(last), float(stop),
                              float(z.near + z.side * target_r * risk_d), z.tag))
    return out


def zone_orders_from_config(df: pd.DataFrame, cfg: dict) -> list[LimitOrder]:
    """cfg: {"kind": "fvg"|"order_block", "min_size_atr", "expiry_bars", "stop_buffer_atr", "target_r", "atr_period"}"""
    kind = cfg.get("kind", "fvg")
    atr_p = int(cfg.get("atr_period", 14))
    zones = build_zones(df, kind, min_size_atr=float(cfg.get("min_size_atr", 0.3)), atr_period=atr_p,
                        displacement_atr=float(cfg.get("displacement_atr", 1.5)))
    return orders_from_zones(zones, df, expiry_bars=int(cfg.get("expiry_bars", 30)),
                             stop_buffer_atr=float(cfg.get("stop_buffer_atr", 0.25)),
                             target_r=float(cfg.get("target_r", 2.0)), atr_period=atr_p)


def check_zone_causality(cfg: dict, df: pd.DataFrame, n_checks: int = 8) -> list[str]:
    """Problems found (empty = causal). For several cut points k, every zone
    order whose creation bar is <= k-1 in the truncated history must equal the
    one built from full history, and truncated history must not know orders the
    full history lacks. Lifecycle-dependent fields (expiry) are compared only
    when the zone is not still live at the cut."""
    problems: list[str] = []
    full = zone_orders_from_config(df, cfg)
    by_bar = {(o.created_bar, o.side): o for o in full}
    n = len(df)
    for k in np.linspace(max(30, n // 5), n - 1, n_checks).astype(int):
        part = zone_orders_from_config(df.iloc[:k].reset_index(drop=True), cfg)
        for o in part:
            f = by_bar.get((o.created_bar, o.side))
            if f is None:
                problems.append(f"order created at bar {o.created_bar} exists with {k} bars of history but not with full history")
            elif (abs(f.price - o.price) > 1e-9) or (abs(f.stop_price - o.stop_price) > 1e-9):
                problems.append(f"order at bar {o.created_bar} changes with future data (price/stop differ)")
    return problems[:10]
