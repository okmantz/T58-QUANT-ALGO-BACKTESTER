"""Lab 3 -- Liquidity & Order-Book Lab.

Studies resting depth, queue imbalance, liquidity replenishment, size depletion and the price response that follows
large market orders -- and asks the honest question: *is visible liquidity associated with rejection, continuation, or
no measurable effect at all?*

This lab REFUSES to run on candles. Resting liquidity cannot be inferred from OHLCV, so without Level-2 snapshots the
lab returns ``status="needs_data"`` with the exact schema it needs instead of inventing numbers. (A clearly labelled
SYNTHETIC demo book is available so you can see what the output looks like.)

Accepted Level-2 layout (wide snapshots, one row per book update):
    timestamp, bid_px_1..bid_px_N, bid_sz_1..bid_sz_N, ask_px_1..ask_px_N, ask_sz_1..ask_sz_N
(``bid_price_k`` / ``bid_size_k`` spellings are accepted too.) Optional trade prints: timestamp, price, size, side.
Market-by-order data (order ids, adds / cancels / modifies) is richer still -- cancellations are then exact rather than
the net-depletion estimate used here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research_labs import common as C

LAB = "order_book"
TITLE = "Liquidity & Order-Book Lab"
ASSUMPTIONS = [
    "Requires real Level-2 snapshots. Resting depth is never inferred from candles.",
    "Queue imbalance QI = (bid size - ask size) / (bid size + ask size) at the best level; depth imbalance does the "
    "same over the top N levels.",
    "Without market-by-order data, cancellations cannot be separated from executions: 'depletion without trades' is "
    "reported as a cancellation ESTIMATE.",
    "Forward mid-price changes are measured a fixed number of snapshots ahead, in basis points; signed so that "
    "positive = in the direction the imbalance / aggressor pointed (continuation), negative = rejection.",
    "Chronological in-sample / out-of-sample halves are reported for every effect.",
]
NEEDS_DATA_SCHEMA = ("timestamp, bid_px_1..N, bid_sz_1..N, ask_px_1..N, ask_sz_1..N  (N >= 1)  "
                     "[+ optional trades: timestamp, price, size, side]")


def _find_cols(df: pd.DataFrame, side: str, kind: str) -> list[str]:
    """Level-ordered column names for one side ('bid'/'ask') and kind ('px'/'sz'); stops at the first missing level."""
    spellings = ("px", "price") if kind == "px" else ("sz", "size")
    names: list[str] = []
    for k in range(1, 51):
        hit = next((f"{side}_{sp}_{k}" for sp in spellings if f"{side}_{sp}_{k}" in df.columns), None)
        if hit is None:
            break
        names.append(hit)
    return names


def detect_levels(df: pd.DataFrame) -> int:
    cols = {str(c).lower() for c in df.columns}
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    if {"bid_px", "ask_px", "bid_sz", "ask_sz"} <= cols:
        return 1
    n = min(len(_find_cols(d, s, k)) for s in ("bid", "ask") for k in ("px", "sz"))
    return n


def synthetic_book(n: int = 6000, levels: int = 5, seed: int = C.DEFAULT_SEED, effect: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SYNTHETIC demo data with a planted, weak imbalance->price effect (so the demo shows how a real effect looks)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-05 14:30", periods=n, freq="500ms")
    mid = 5000.0
    tick = 0.25
    rows, mids = [], []
    imb_prev = 0.0
    for i in range(n):
        mid += tick * np.round(rng.normal(0, 0.6) + effect * imb_prev)
        b1, a1 = rng.integers(20, 200), rng.integers(20, 200)
        imb_prev = (b1 - a1) / (b1 + a1)
        row = {"timestamp": ts[i]}
        for k in range(1, levels + 1):
            row[f"bid_px_{k}"] = mid - tick * k + tick / 2
            row[f"ask_px_{k}"] = mid + tick * k - tick / 2
            row[f"bid_sz_{k}"] = float(b1 if k == 1 else rng.integers(20, 200))
            row[f"ask_sz_{k}"] = float(a1 if k == 1 else rng.integers(20, 200))
        rows.append(row)
        mids.append(mid)
    book = pd.DataFrame(rows)
    big = rng.random(n) < 0.02
    tr = pd.DataFrame({"timestamp": ts[big], "price": np.array(mids)[big], "size": rng.integers(50, 400, big.sum()).astype(float),
                       "side": rng.choice(["buy", "sell"], big.sum())})
    return book, tr


def book_features(book: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    b = book.copy()
    b.columns = [str(c).lower() for c in b.columns]
    b["timestamp"] = pd.to_datetime(b["timestamp"])
    b = b.sort_values("timestamp").reset_index(drop=True)
    if {"bid_px", "ask_px", "bid_sz", "ask_sz"} <= set(b.columns):
        bpx, apx, bsz, asz = ["bid_px"], ["ask_px"], ["bid_sz"], ["ask_sz"]
    else:
        bpx, apx = _find_cols(b, "bid", "px"), _find_cols(b, "ask", "px")
        bsz, asz = _find_cols(b, "bid", "sz"), _find_cols(b, "ask", "sz")
    n = min(levels, len(bpx), len(apx), len(bsz), len(asz))
    if n < 1:
        raise C.ResearchLabError("Could not find Level-2 columns. Expected: " + NEEDS_DATA_SCHEMA)
    out = pd.DataFrame({"timestamp": b["timestamp"]})
    out["bid1"], out["ask1"] = b[bpx[0]].astype(float), b[apx[0]].astype(float)
    out["mid"] = (out["bid1"] + out["ask1"]) / 2.0
    out["spread"] = out["ask1"] - out["bid1"]
    b1, a1 = b[bsz[0]].astype(float), b[asz[0]].astype(float)
    out["b1"], out["a1"] = b1, a1
    out["qi"] = (b1 - a1) / (b1 + a1).replace(0, np.nan)
    bd = b[bsz[:n]].astype(float).sum(axis=1)
    ad = b[asz[:n]].astype(float).sum(axis=1)
    out["bid_depth"], out["ask_depth"] = bd, ad
    out["depth_imb"] = (bd - ad) / (bd + ad).replace(0, np.nan)
    out["microprice"] = (out["ask1"] * b1 + out["bid1"] * a1) / (b1 + a1).replace(0, np.nan)
    return out


def replenishment_stats(f: pd.DataFrame, drop_frac: float = 0.5, k: int = 5) -> dict:
    """After the best bid / ask loses >= drop_frac of its size with the price unchanged, how often is the size back?"""
    res = {}
    for side, size, px in (("bid", "b1", "bid1"), ("ask", "a1", "ask1")):
        s, p = f[size].to_numpy(float), f[px].to_numpy(float)
        prev = np.roll(s, 1)
        dropped = (s <= (1 - drop_frac) * prev) & (p == np.roll(p, 1)) & (prev > 0)
        dropped[0] = False
        idx = np.flatnonzero(dropped)
        idx = idx[idx + k < len(s)]
        if len(idx) == 0:
            res[side] = {"events": 0, "replenished_within_k": None, "mean_recovery": None}
            continue
        back = np.array([(s[i + 1:i + 1 + k] >= 0.8 * prev[i]).any() and (p[i + 1:i + 1 + k] == p[i]).all() for i in idx])
        rec = np.array([s[i + k] / prev[i] for i in idx])
        res[side] = {"events": int(len(idx)), "replenished_within_k": float(back.mean()), "mean_recovery": float(np.mean(rec))}
    return res


def cancellation_estimate(f: pd.DataFrame, trades: pd.DataFrame | None, drop_frac: float = 0.5) -> dict:
    """Best-level size drops with the price unchanged and no trade print in that interval = estimated cancellations."""
    out = {}
    ts = f["timestamp"].to_numpy()
    tr_ts = None if trades is None else np.sort(pd.to_datetime(trades["timestamp"]).to_numpy())
    for side, size, px in (("bid", "b1", "bid1"), ("ask", "a1", "ask1")):
        s, p = f[size].to_numpy(float), f[px].to_numpy(float)
        prev = np.roll(s, 1)
        dropped = (s <= (1 - drop_frac) * prev) & (p == np.roll(p, 1)) & (prev > 0)
        dropped[0] = False
        idx = np.flatnonzero(dropped)
        if tr_ts is None:
            out[side] = {"depletions": int(len(idx)), "estimated_cancellations": None,
                         "note": "no trade prints supplied: cannot separate executions from cancellations"}
            continue
        lo = np.searchsorted(tr_ts, ts[idx - 1], side="right")
        hi = np.searchsorted(tr_ts, ts[idx], side="right")
        no_trade = (hi - lo) == 0
        out[side] = {"depletions": int(len(idx)), "estimated_cancellations": int(no_trade.sum()),
                     "cancel_share": float(no_trade.mean()) if len(idx) else None}
    return out


def _mid_ahead_bps(f: pd.DataFrame, horizon: int) -> pd.Series:
    return (f["mid"].shift(-horizon) / f["mid"] - 1.0) * 1e4


def imbalance_effect(f: pd.DataFrame, horizon: int = 20, oos_fraction: float = 0.3, seed: int = C.DEFAULT_SEED) -> dict:
    out_bps = _mid_ahead_bps(f, horizon)
    res = {"horizon_snapshots": horizon}
    cut = int(len(f) * (1 - oos_fraction))
    for name in ("qi", "depth_imb"):
        x = f[name]
        ok = x.notna() & out_bps.notna()
        if ok.sum() < 50:
            res[name] = {"n": int(ok.sum()), "verdict": "too little data"}
            continue
        xr, yr = x[ok].rank(), out_bps[ok].rank()
        corr = float(np.corrcoef(xr, yr)[0, 1])
        n = int(ok.sum())
        r_is = float(np.corrcoef(x[ok & (np.arange(len(f)) < cut)].rank(), out_bps[ok & (np.arange(len(f)) < cut)].rank())[0, 1]) if (ok & (np.arange(len(f)) < cut)).sum() > 30 else None
        r_oos = float(np.corrcoef(x[ok & (np.arange(len(f)) >= cut)].rank(), out_bps[ok & (np.arange(len(f)) >= cut)].rank())[0, 1]) if (ok & (np.arange(len(f)) >= cut)).sum() > 30 else None
        lo, hi = C.fisher_ci(corr, max(n // max(horizon, 1), 5))
        top = out_bps[ok & (x >= x[ok].quantile(0.8))]
        bot = out_bps[ok & (x <= x[ok].quantile(0.2))]
        w = C.welch_test(top.to_numpy(), bot.to_numpy())
        if lo is not None and (lo > 0 or hi < 0) and r_oos is not None and np.sign(r_oos) == np.sign(corr):
            verdict = "continuation" if corr > 0 else "rejection"
        else:
            verdict = "no measurable effect"
        res[name] = {"n": n, "spearman": corr, "ci_low": lo, "ci_high": hi, "in_sample_corr": r_is, "oos_corr": r_oos,
                     "top_quintile_mean_bps": float(top.mean()) if len(top) else None,
                     "bottom_quintile_mean_bps": float(bot.mean()) if len(bot) else None,
                     "top_vs_bottom_p": w["p"], "verdict": verdict,
                     "note": "CI uses an effective sample size n/horizon because forward windows overlap"}
    return res


def large_order_response(f: pd.DataFrame, trades: pd.DataFrame | None, quantile: float = 0.98, horizon: int = 20,
                         seed: int = C.DEFAULT_SEED) -> dict:
    if trades is None or len(trades) < 50:
        return {"status": "skipped", "reason": "needs trade prints (timestamp, price, size, side)"}
    t = trades.copy()
    t.columns = [str(c).lower() for c in t.columns]
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    sign = np.where(t["side"].astype(str).str.lower().isin(["buy", "b", "1"]), 1.0,
                    np.where(t["side"].astype(str).str.lower().isin(["sell", "s", "-1"]), -1.0, 0.0))
    t = t[sign != 0].assign(sign=sign[sign != 0])
    thr = t["size"].quantile(quantile)
    big = t[t["size"] >= thr]
    if len(big) < 10:
        return {"status": "skipped", "reason": f"only {len(big)} large orders at the {quantile:.0%} size quantile"}
    pos = np.searchsorted(f["timestamp"].to_numpy(), big["timestamp"].to_numpy(), side="left")
    ok = pos < len(f) - horizon
    pos, big = pos[ok], big[ok]
    move = (f["mid"].to_numpy()[pos + horizon] / f["mid"].to_numpy()[pos] - 1.0) * 1e4 * big["sign"].to_numpy()
    d = C.describe_group(move, seed=seed)
    tt = C.one_sample_t(move)
    verdict = ("continuation (price kept moving with the aggressor)" if d["ci_low"] is not None and d["ci_low"] > 0
               else "rejection (price reverted)" if d["ci_high"] is not None and d["ci_high"] < 0
               else "no measurable effect")
    return {"status": "ok", "size_threshold": float(thr), "large_orders": int(len(move)), "signed_mid_move_bps": d,
            "t": tt["t"], "p": tt["p"], "verdict": verdict}


def run(book: pd.DataFrame | None = None, trades: pd.DataFrame | None = None, *, label: str = "", levels: int = 5,
        horizon: int = 20, large_quantile: float = 0.98, oos_fraction: float = 0.3, demo: bool = False,
        seed: int = C.DEFAULT_SEED, save: bool = True) -> C.ExperimentRecord:
    params = {"levels": levels, "horizon": horizon, "large_quantile": large_quantile, "oos_fraction": oos_fraction, "demo": demo}
    warnings: list[str] = []
    if demo:
        book, trades = synthetic_book(seed=seed)
        label = label or "SYNTHETIC demo book"
        warnings.append("SYNTHETIC demo data (with a planted weak imbalance effect). Nothing here says anything about a real market.")
    if book is None or detect_levels(book) < 1:
        stub = pd.DataFrame({"timestamp": pd.to_datetime(["2000-01-01"]), "open": [0.0], "high": [0.0], "low": [0.0],
                             "close": [0.0], "volume": [0.0]})
        results = {"headline": "needs Level-2 data -- resting liquidity cannot be inferred from candles",
                   "needs_data": NEEDS_DATA_SCHEMA,
                   "why": "Candles hide the order book. Visible depth, queue position, cancellations and replenishment only exist in "
                          "Level-2 (or market-by-order) data. Download it from your futures data vendor and drop it in, or tick "
                          "'demo' to see the lab run on synthetic data."}
        return C.ExperimentRecord(lab=LAB, title=TITLE, params=params, assumptions=ASSUMPTIONS,
                                  data=C.data_fingerprint(stub, label), results=results, warnings=warnings, seed=seed,
                                  status="needs_data")
    f = book_features(book, levels)
    if len(f) > C.MAX_BARS:
        f = f.iloc[-C.MAX_BARS:].reset_index(drop=True)
    effect = imbalance_effect(f, horizon, oos_fraction, seed)
    big = large_order_response(f, trades, large_quantile, horizon, seed)
    rep = replenishment_stats(f)
    canc = cancellation_estimate(f, trades)
    verdicts = [v.get("verdict") for v in effect.values() if isinstance(v, dict)]
    headline = ("queue imbalance: " + str(effect.get("qi", {}).get("verdict")) + "; depth imbalance: "
                + str(effect.get("depth_imb", {}).get("verdict")) + "; large orders: " + str(big.get("verdict", big.get("status"))))
    results = {"headline": headline, "imbalance_effect": effect, "large_order_response": big, "replenishment": rep,
               "cancellation_estimate": canc, "spread_bps_median": float((f["spread"] / f["mid"]).median() * 1e4),
               "snapshots": int(len(f)), "demo": bool(demo)}
    snap = f.rename(columns={"timestamp": "timestamp"}).assign(open=f["mid"], high=f["mid"], low=f["mid"], close=f["mid"], volume=0.0)
    return C.record_from(LAB, TITLE, params, ASSUMPTIONS, snap, results, warnings, seed, label, save=save)
