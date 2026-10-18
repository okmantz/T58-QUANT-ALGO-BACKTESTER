"""Lab 2 -- Footprint & Order-Flow Lab.

Bid/ask volume per price, diagonal imbalances, bar delta, cumulative volume delta (CVD) and absorption.

Two data tiers, never blended silently:

* TRUE order flow -- trade-level data with aggressor side (a ``side`` column: buy/sell, B/A, 1/-1, plus ``price`` and
  ``size``), or bars that already carry ``bid_volume`` / ``ask_volume``. Footprints, diagonal imbalances and delta are exact.
* PROXY order flow -- plain OHLCV. True bid/ask volume CANNOT be reconstructed from candles, so the lab falls back to
  the bar-location delta  ``volume * ((close-low) - (high-close)) / (high-low)`` and labels every number "proxy".
  Useful for ranking ideas, not evidence of what the tape really did.

Default study: do entries after liquidity sweeps perform better when aggressive buying/selling FAILS to move price
(absorption), or when CVD diverges from price?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research_labs import common as C

LAB = "order_flow"
TITLE = "Footprint & Order-Flow Lab"
ASSUMPTIONS = [
    "Trade-level data with aggressor side gives true bid/ask volume. Plain OHLCV does NOT -- the lab then uses a "
    "bar-location delta PROXY and marks every result as a proxy.",
    "Diagonal imbalance: ask volume at price p vs bid volume at p-1 tick (buy imbalance), bid at p vs ask at p+1 tick "
    "(sell imbalance), flagged when the ratio reaches the chosen threshold.",
    "Absorption = unusually heavy one-sided aggression (volume z-score and |delta|/volume above thresholds) that "
    "did not move price in the aggressor's favour (close-open on the wrong side or within 10% of ATR).",
    "A sweep's order-flow context is read at the sweep bar's close, so entries never use later information.",
]
_BUY = {"b", "buy", "bid", "a_buy", "1", "ask_lift", "buyer"}
_SELL = {"s", "sell", "ask", "-1", "seller", "bid_hit"}


def has_trade_level(df: pd.DataFrame) -> bool:
    cols = {str(c).lower() for c in df.columns}
    return {"price", "size", "side"} <= cols


def has_bar_bid_ask(df: pd.DataFrame) -> bool:
    cols = {str(c).lower() for c in df.columns}
    return {"bid_volume", "ask_volume"} <= cols


def _signed_side(side: pd.Series) -> np.ndarray:
    s = side.astype(str).str.strip().str.lower()
    out = np.zeros(len(s))
    out[s.isin(_BUY).to_numpy()] = 1.0
    out[s.isin(_SELL).to_numpy()] = -1.0
    return out


def footprint_from_trades(trades: pd.DataFrame, bar_minutes: int = 5, tick_size: float = 0.25,
                          imbalance_ratio: float = 3.0, max_bars: int = 400) -> dict:
    """Builds bars + per-price bid/ask footprint from trade prints. Aggressor side must be on every row."""
    t = trades.copy()
    t.columns = [str(c).lower() for c in t.columns]
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    sign = _signed_side(t["side"])
    keep = sign != 0
    if keep.sum() == 0:
        raise C.ResearchLabError("No row has a recognisable aggressor side (use buy/sell, B/A or 1/-1).")
    t, sign = t[keep], sign[keep]
    t["ask_vol"] = np.where(sign > 0, t["size"], 0.0)     # aggressive buys lift the ask
    t["bid_vol"] = np.where(sign < 0, t["size"], 0.0)     # aggressive sells hit the bid
    t["bar"] = t["timestamp"].dt.floor(f"{int(bar_minutes)}min")
    t["px"] = (t["price"] / tick_size).round() * tick_size
    g = t.groupby(["bar", "px"], sort=True)[["ask_vol", "bid_vol"]].sum().reset_index()
    bars = t.groupby("bar").agg(open=("price", "first"), high=("price", "max"), low=("price", "min"),
                                close=("price", "last"), volume=("size", "sum"),
                                ask_volume=("ask_vol", "sum"), bid_volume=("bid_vol", "sum")).reset_index()
    bars["delta"] = bars["ask_volume"] - bars["bid_volume"]
    bars["cvd"] = bars["delta"].cumsum()
    stacked, buy_imb, sell_imb = [], [], []
    for bar, grp in g.groupby("bar"):
        a = grp.set_index("px")["ask_vol"]
        b = grp.set_index("px")["bid_vol"]
        bi = [(p, float(a[p]), float(b.get(round(p - tick_size, 10), 0.0))) for p in a.index
              if a[p] > 0 and a[p] >= imbalance_ratio * max(float(b.get(round(p - tick_size, 10), 0.0)), 1e-9)]
        si = [(p, float(b[p]), float(a.get(round(p + tick_size, 10), 0.0))) for p in b.index
              if b[p] > 0 and b[p] >= imbalance_ratio * max(float(a.get(round(p + tick_size, 10), 0.0)), 1e-9)]
        buy_imb.append(len(bi))
        sell_imb.append(len(si))
        prices = sorted(p for p, *_ in bi)
        run = best = 1 if prices else 0
        for i in range(1, len(prices)):
            run = run + 1 if abs(prices[i] - prices[i - 1] - tick_size) < 1e-9 else 1
            best = max(best, run)
        stacked.append(best)
    bars = bars.merge(pd.DataFrame({"bar": sorted(g["bar"].unique()), "buy_imbalances": buy_imb, "sell_imbalances": sell_imb,
                                    "stacked_buy_run": stacked}), on="bar", how="left")
    tail = bars.tail(max_bars)
    last = g[g["bar"] == bars["bar"].iloc[-1]].sort_values("px", ascending=False)
    return {"bars_df": bars.rename(columns={"bar": "timestamp"}),
            "bars": tail.assign(bar=tail["bar"].astype(str)).to_dict(orient="records"),
            "last_bar_footprint": last.assign(bar=last["bar"].astype(str)).to_dict(orient="records"),
            "bars_total": int(len(bars)), "imbalance_ratio": imbalance_ratio, "tick_size": tick_size}


def flow_frame(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Per-bar delta / CVD / absorption columns. TRUE when bid_volume+ask_volume exist, otherwise a labelled PROXY."""
    d = df
    out = pd.DataFrame(index=d.index)
    true_flow = has_bar_bid_ask(d)
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    if true_flow:
        out["delta"] = d["ask_volume"].astype(float) - d["bid_volume"].astype(float)
        vol = (d["ask_volume"] + d["bid_volume"]).astype(float)
    else:
        clv = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / rng
        out["delta"] = (d["volume"].astype(float) * clv).fillna(0.0)
        vol = d["volume"].astype(float)
    out["cvd"] = out["delta"].cumsum()
    out["volume_z"] = (vol - vol.rolling(period, min_periods=5).mean()) / vol.rolling(period, min_periods=5).std().replace(0, np.nan)
    out["delta_ratio"] = (out["delta"] / vol.replace(0, np.nan)).fillna(0.0)
    out["delta_z"] = (out["delta"] - out["delta"].rolling(period, min_periods=5).mean()) / out["delta"].rolling(period, min_periods=5).std().replace(0, np.nan)
    atr = C.atr_series(d).replace(0, np.nan)
    move = (d["close"] - d["open"]) / atr
    heavy = (out["volume_z"] >= 1.0) & (out["delta_ratio"].abs() >= 0.25)
    # buy-side aggression that failed to lift price => sellers absorbed (bearish); sell-side that failed => bullish
    out["absorb_bearish"] = (heavy & (out["delta_ratio"] > 0) & (move <= 0.10)).fillna(False)
    out["absorb_bullish"] = (heavy & (out["delta_ratio"] < 0) & (move >= -0.10)).fillna(False)
    out["absorption"] = np.where(out["absorb_bullish"], 1.0, np.where(out["absorb_bearish"], -1.0, 0.0))
    lb = max(5, period)
    out["cvd_slope"] = out["cvd"].diff(lb) / vol.rolling(lb, min_periods=3).sum().replace(0, np.nan)
    out["is_true_flow"] = 1.0 if true_flow else 0.0
    return out


def sweep_flow_study(df: pd.DataFrame, flow: pd.DataFrame, horizon: int = 12, lookback: int = 20,
                     oos_fraction: float = 0.3, min_n: int = 15, seed: int = C.DEFAULT_SEED) -> dict:
    sweeps = C.detect_liquidity_sweeps(df, lookback=lookback)
    outcome = C.sweep_directional_return(df, sweeps, horizon) * 1e4
    ev = sweeps["ssl_sweep"] | sweeps["bsl_sweep"]
    ssl, bsl = sweeps["ssl_sweep"], sweeps["bsl_sweep"]
    # reversal-aligned order-flow evidence, read at (or one bar before) the sweep bar's close
    absorb = ((ssl & ((flow["absorb_bullish"]) | flow["absorb_bullish"].shift(1).fillna(False)))
              | (bsl & ((flow["absorb_bearish"]) | flow["absorb_bearish"].shift(1).fillna(False))))
    aggr = ((ssl & (flow["delta_z"] <= -1.0)) | (bsl & (flow["delta_z"] >= 1.0)))
    low_ll = df["low"] <= df["low"].rolling(lookback, min_periods=lookback).min()
    high_hh = df["high"] >= df["high"].rolling(lookback, min_periods=lookback).max()
    cvd_lo = flow["cvd"] > flow["cvd"].rolling(lookback, min_periods=lookback).min()
    cvd_hi = flow["cvd"] < flow["cvd"].rolling(lookback, min_periods=lookback).max()
    diverge = ((ssl & cvd_lo & low_ll.shift(0, fill_value=False)) | (bsl & cvd_hi & high_hh.shift(0, fill_value=False)))
    groups = {
        "absorption at the sweep (aggression failed)": absorb,
        "no absorption": ~absorb,
        "aggressive delta into the sweep": aggr,
        "CVD divergence (price extreme, CVD not)": diverge,
        "no CVD divergence": ~diverge,
    }
    rows = C.group_compare(outcome, ev, groups, oos_fraction=oos_fraction, min_n=min_n, seed=seed)
    return {"events": int((ev & outcome.notna()).sum()), "horizon_bars": horizon,
            "baseline_all_sweeps": C.describe_group(outcome[ev].to_numpy(), seed=seed), "groups": rows}


def run(df: pd.DataFrame, *, trades: pd.DataFrame | None = None, label: str = "", horizon: int = 12,
        sweep_lookback: int = 20, oos_fraction: float = 0.3, bar_minutes: int = 5, tick_size: float = 0.25,
        imbalance_ratio: float = 3.0, seed: int = C.DEFAULT_SEED, save: bool = True) -> C.ExperimentRecord:
    warnings: list[str] = []
    footprint = None
    if trades is not None and has_trade_level(trades):
        footprint = footprint_from_trades(trades, bar_minutes, tick_size, imbalance_ratio)
        d = C.prepare_ohlcv(footprint.pop("bars_df"))
        mode = "true (trade-level aggressor side)"
    else:
        d = C.prepare_ohlcv(df)
        if has_bar_bid_ask(d):
            mode = "true (bar bid/ask volume)"
        else:
            mode = "PROXY (OHLCV only -- bar-location delta)"
            warnings.append("This is OHLCV only: true bid/ask order flow cannot be reconstructed from candles. Delta, "
                            "CVD and absorption below are PROXIES; load trade-level data with aggressor side for real order flow.")
    flow = flow_frame(d)
    study = sweep_flow_study(d, flow, horizon, sweep_lookback, oos_fraction, seed=seed)
    if study["events"] < 30:
        warnings.append(f"Only {study['events']} sweeps in this sample -- treat the table as exploratory.")
    promising = [g["group"] for g in study["groups"] if str(g["verdict"]).startswith("promising")]
    headline = (f"[{mode}] promising: {', '.join(promising)}" if promising
                else f"[{mode}] {study['events']} sweeps; no order-flow condition survived correction + out-of-sample")
    tail = flow.tail(200).assign(timestamp=d["timestamp"].tail(200).astype(str).to_numpy())
    results = {"headline": headline, "mode": mode, "sweep_study": study, "footprint": footprint,
               "flow_tail": tail[["timestamp", "delta", "cvd", "delta_z", "absorption"]].to_dict(orient="records"),
               "needs_for_true_order_flow": "trade-level prints with a side column (buy/sell), or bars with bid_volume and ask_volume"}
    return C.record_from(LAB, TITLE, {"horizon": horizon, "sweep_lookback": sweep_lookback, "oos_fraction": oos_fraction,
                                      "bar_minutes": bar_minutes, "tick_size": tick_size, "imbalance_ratio": imbalance_ratio,
                                      "mode": mode}, ASSUMPTIONS, d, results, warnings, seed, label, save=save)
