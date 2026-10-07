"""
RESTING (LIMIT) ORDERS and PERSISTENT ZONES (2026-10-07 discovery layer).

The bar engine (app.backtest.execution) trades a per-bar signal series: a
market order at the next open. Many real ideas are not that -- "buy the
retest of the fair-value gap", "fade back to the opening-range edge" are
RESTING orders that sit at a price for N bars and fill only if price comes to
them. Modeling those as a next-open market order would book trades that never
happened (price never returned) and fills at prices that were never offered.

This module simulates them honestly, with the same sizing and costs as the
engine (RiskConfig.size_for_stop / side_cost_price / commission_for):

  * an order is live from created_bar+1 to expire_bar (inclusive) and is
    cancelled the moment it is filled or expires;
  * a BUY limit fills only if price trades THROUGH it (low < limit; set
    require_trade_through=False to fill on a touch). A bar that opens below
    the limit fills at the open (never worse than the limit, never better
    than what the market offered);
  * limit fills pay no spread/slippage (they provide liquidity); stop exits
    pay the full spread+slippage; target (limit) exits pay none;
  * on the fill bar the stop is checked too, and when a bar could have hit
    both stop and target the STOP is booked first (conservative);
  * one position at a time; orders whose creation falls inside an open
    position are skipped;
  * sizing is whole contracts through size_for_stop, so risk_at_stop and the
    skip reasons match the main engine.

`zones_fvg` builds persistent fair-value-gap zones (a 3-candle imbalance)
and turns each into a resting order at the zone edge nearest price.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.backtest.execution import Trade


@dataclass
class LimitOrder:
    side: int                 # +1 buy limit, -1 sell limit
    price: float              # limit price
    created_bar: int          # bar on whose CLOSE the order was placed (live from the next bar)
    expire_bar: int           # last bar the order is live
    stop_price: float
    target_price: float | None = None
    tag: str = ""


def simulate_resting_orders(
    df: pd.DataFrame,
    orders: list[LimitOrder],
    risk,
    *,
    require_trade_through: bool = True,
    initial_equity: float | None = None,
) -> list[Trade]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    ts = pd.to_datetime(df["timestamp"]).to_numpy()
    n = len(df)
    equity = float(initial_equity if initial_equity is not None else risk.initial_balance)
    trades: list[Trade] = []
    side_cost = risk.side_cost_price()
    busy_until = -1

    for od in sorted(orders, key=lambda x: x.created_bar):
        if od.created_bar <= busy_until:
            continue
        d = int(od.side)
        fill_bar = None
        fill_px = None
        last = min(od.expire_bar, n - 1)
        for j in range(od.created_bar + 1, last + 1):
            if d == 1:
                hit = (l[j] < od.price) if require_trade_through else (l[j] <= od.price)
                if hit:
                    fill_bar, fill_px = j, min(od.price, o[j])
                    break
            else:
                hit = (h[j] > od.price) if require_trade_through else (h[j] >= od.price)
                if hit:
                    fill_bar, fill_px = j, max(od.price, o[j])
                    break
        if fill_bar is None:
            continue
        stop_dist = abs(fill_px - od.stop_price)
        if stop_dist <= 0 or (d == 1 and od.stop_price >= fill_px) or (d == -1 and od.stop_price <= fill_px):
            continue
        dec = risk.size_for_stop(equity, stop_dist)
        if dec.skip_reason or dec.units <= 0:
            continue
        # a capped (fit_stop) decision tightens the stop
        eff_stop = fill_px - d * dec.stop_distance
        eff_target = od.target_price
        # ---- manage the position from the fill bar onward --------------
        exit_px = None
        reason = None
        exit_bar = n - 1
        for j in range(fill_bar, n):
            hit_stop = (l[j] <= eff_stop) if d == 1 else (h[j] >= eff_stop)
            if hit_stop:
                raw = min(eff_stop, o[j]) if (d == 1 and j > fill_bar) else (
                    max(eff_stop, o[j]) if (d == -1 and j > fill_bar) else eff_stop)
                exit_px = raw - d * side_cost
                reason, exit_bar = "stop_loss", j
                break
            if eff_target is not None:
                hit_tp = (h[j] >= eff_target) if d == 1 else (l[j] <= eff_target)
                if hit_tp:
                    exit_px, reason, exit_bar = eff_target, "take_profit", j
                    break
        if exit_px is None:
            exit_px = c[n - 1] - d * side_cost
            reason, exit_bar = "end_of_data", n - 1
        cs = dec.contract_size
        contracts = (dec.units / cs) if cs else 0.0
        comm = risk.commission_for(contracts, dec.commission_per_contract)
        pnl = (exit_px - fill_px) * dec.units * d - comm
        equity += pnl
        busy_until = exit_bar
        trades.append(Trade(
            entry_time=pd.Timestamp(ts[fill_bar]), exit_time=pd.Timestamp(ts[exit_bar]),
            direction=d, entry_price=float(fill_px), exit_price=float(exit_px), size=float(dec.units),
            pnl=float(pnl), pnl_pct=float(pnl / (equity - pnl) * 100.0) if (equity - pnl) else 0.0,
            exit_reason=reason, commission=float(comm), equity_after=float(equity),
            initial_risk=float(dec.stop_distance), intended_risk_dollars=float(dec.budget),
            risk_at_stop_dollars=float(dec.risk_at_stop), stop_capped=bool(dec.stop_capped),
            sizing_mode=risk.sizing_mode, used_micro=bool(dec.used_micro), contracts=float(dec.contracts),
        ))
    return trades


def zones_fvg(
    df: pd.DataFrame,
    *,
    min_gap_atr: float = 0.3,
    atr_period: int = 14,
    expiry_bars: int = 30,
    stop_buffer_atr: float = 0.25,
    target_r: float = 2.0,
) -> list[LimitOrder]:
    """Fair-value-gap zones -> resting orders.

    Bullish FVG at bar i: low[i] > high[i-2] (gap between candle i-2's high
    and candle i's low); the zone is [high[i-2], low[i]]. A resting BUY limit
    sits at the zone's upper edge (first touch on the retrace), stop one
    buffer below the zone, target target_r x stop distance above. Bearish FVG
    mirrors it. The order is placed on the close of bar i (no lookahead: the
    gap is only known once bar i has closed)."""
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(atr_period, min_periods=atr_period).mean().to_numpy()
    out: list[LimitOrder] = []
    for i in range(2, len(df)):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_gap_atr * a:
            top, bot = l[i], h[i - 2]
            stop = bot - stop_buffer_atr * a
            risk_d = top - stop
            out.append(LimitOrder(1, float(top), i, i + expiry_bars, float(stop),
                                  float(top + target_r * risk_d), "fvg_bull"))
        elif h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_gap_atr * a:
            bot, top = h[i], l[i - 2]
            stop = top + stop_buffer_atr * a
            risk_d = stop - bot
            out.append(LimitOrder(-1, float(bot), i, i + expiry_bars, float(stop),
                                  float(bot - target_r * risk_d), "fvg_bear"))
    return out
