"""Portfolio algos: prices (+ optional current value scores) -> target weights. Pure functions, so they backtest and trade identically."""
from __future__ import annotations

import numpy as np
import pandas as pd

ALGOS = {
    "momentum": "Own what is trending: 12-1 month momentum, skipping the latest month.",
    "mean_reversion": "Buy short-term oversold names inside long-term uptrends (RSI-style pullbacks).",
    "low_vol_trend": "Steady risers: positive trend ranked by low volatility (smoother ride).",
    "value": "Cheapest quality companies by SEC fundamentals (current-snapshot only; not backtestable without look-ahead).",
    "t58_composite": "Blend of value, quality/low-vol and momentum, weighted by your profile.",
}


def _z(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    sd = s.std()
    return ((s - s.mean()) / sd).fillna(0.0) if sd and sd > 0 else s.fillna(0.0) * 0


def factor_table(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.ffill()
    ret = p.pct_change()
    f = pd.DataFrame(index=p.columns)
    f["mom"] = (p.shift(21).iloc[-1] / p.shift(252).iloc[-1] - 1) if len(p) > 260 else (p.iloc[-1] / p.iloc[0] - 1)
    f["vol"] = ret.tail(126).std() * np.sqrt(252)
    d = p.diff()
    up, dn = d.clip(lower=0).tail(14).mean(), (-d.clip(upper=0)).tail(14).mean()
    f["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    f["trend_up"] = (p.iloc[-1] > p.tail(200).mean()).astype(float)
    f["dist_high"] = p.iloc[-1] / p.tail(252).max() - 1
    return f


def target_weights(algo: str, prices: pd.DataFrame, cfg: dict, value: dict | None = None, crypto: set | None = None) -> dict:
    """Returns {symbol: weight}; weights sum to <= 1 - cash_floor. Respects max_positions, max_weight, crypto_cap."""
    prices = prices.dropna(axis=1, how="all")
    if prices.shape[0] < 30 or prices.shape[1] == 0:
        return {}
    f = factor_table(prices)
    crypto = crypto or set()
    if algo == "momentum":
        score = _z(f["mom"]) + 0.5 * f["trend_up"]
        eligible = f["trend_up"] > 0
    elif algo == "mean_reversion":
        score = _z(-f["rsi"])
        eligible = (f["trend_up"] > 0) & (f["rsi"] < 45)
    elif algo == "low_vol_trend":
        score = _z(-f["vol"]) + 0.5 * _z(f["mom"])
        eligible = (f["trend_up"] > 0) & (f["mom"] > 0)
    elif algo == "value":
        v = pd.Series({k: (x.get("score") if isinstance(x, dict) else x) for k, x in (value or {}).items()}).reindex(f.index)
        score = _z(v.fillna(v.min() if v.notna().any() else 0))
        eligible = v.fillna(0) >= 50
    else:  # t58_composite
        wv, wm, wq = cfg.get("value_weight", 0.4), cfg.get("momentum_weight", 0.3), cfg.get("quality_weight", 0.3)
        v = pd.Series({k: (x.get("score") if isinstance(x, dict) else x) for k, x in (value or {}).items()}).reindex(f.index)
        has_v = v.notna().any()
        if not has_v:                              # no fundamentals (e.g. backtest, ETFs, crypto): renormalise onto price factors
            wm, wq, wv = wm / max(wm + wq, 1e-9), wq / max(wm + wq, 1e-9), 0.0
        vz = _z(v.fillna(v.median())) if has_v else 0.0
        score = wv * vz + wm * _z(f["mom"]) + wq * (_z(-f["vol"]))
        eligible = (f["trend_up"] > 0) | (f["dist_high"] < -0.25)   # uptrend, or deeply discounted
        eligible &= f["mom"].fillna(-1) > -0.35                       # skip falling knives
    score = score[eligible.reindex(score.index).fillna(False)].sort_values(ascending=False)
    picks = list(score.index[: int(cfg.get("max_positions", 10))])
    if not picks:
        return {}
    budget = 1.0 - float(cfg.get("cash_floor", 0.1))
    cap = float(cfg.get("max_weight", 0.15))
    vols = f["vol"].reindex(picks).replace(0, np.nan).fillna(f["vol"].median() if f["vol"].notna().any() else 0.3)
    raw = (1 / vols); raw = raw / raw.sum() * budget       # inverse-vol sizing
    w = raw.clip(upper=cap)
    cc = float(cfg.get("crypto_cap", 0.0))
    cw = w[[s for s in w.index if s in crypto]].sum()
    if cw > cc:
        for s in [s for s in w.index if s in crypto]:
            w[s] = w[s] * (cc / cw) if cw else 0
    return {k: round(float(v), 5) for k, v in w.items() if v > 0.001}
