"""Walk-forward backtest of an investing algo on price history (price-based factors only: no look-ahead on fundamentals)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.investing.algos import target_weights


def backtest(prices: pd.DataFrame, algo: str, cfg: dict, crypto: set | None = None, cost_bps: float = 5.0,
             start_cash: float = 10_000.0, warmup: int = 260) -> dict:
    prices = prices.sort_index().ffill()
    if len(prices) < warmup + 30:
        raise ValueError(f"Need at least {warmup + 30} daily bars; got {len(prices)}.")
    step = max(1, int(cfg.get("rebalance_days", 21)))
    stop = float(cfg.get("stop_loss_pct", 100)) / 100
    ret = prices.pct_change().fillna(0.0)
    w = pd.Series(0.0, index=prices.columns)
    entry = pd.Series(np.nan, index=prices.columns)
    equity, eq = [], start_cash
    turnover_cost = cost_bps / 1e4
    trades = 0
    for i in range(warmup, len(prices)):
        if (i - warmup) % step == 0:
            tw = pd.Series(target_weights(algo, prices.iloc[: i + 1], cfg, None, crypto)).reindex(prices.columns).fillna(0.0)
            eq -= eq * float((tw - w).abs().sum()) * turnover_cost
            trades += int(((tw - w).abs() > 1e-6).sum())
            newly = (w <= 0) & (tw > 0)
            entry[newly] = prices.iloc[i][newly]
            w = tw
        eq *= 1 + float((w * ret.iloc[i]).sum())
        if stop < 1:                                    # protective stop on each holding
            hit = (w > 0) & (prices.iloc[i] / entry - 1 < -stop)
            if hit.any():
                eq -= eq * float(w[hit].sum()) * turnover_cost
                w[hit] = 0.0
        equity.append(eq)
    eqs = pd.Series(equity, index=prices.index[warmup:])
    bench = prices.mean(axis=1).pct_change().fillna(0.0).iloc[warmup:]       # equal-weight buy&hold of the same universe
    beq = start_cash * (1 + bench).cumprod()
    r = eqs.pct_change().dropna()
    yrs = len(eqs) / 252
    dd = (eqs / eqs.cummax() - 1).min()
    bdd = (beq / beq.cummax() - 1).min()
    out = {
        "algo": algo, "start": str(eqs.index[0].date()), "end": str(eqs.index[-1].date()), "years": round(yrs, 2),
        "total_return_pct": round((eqs.iloc[-1] / start_cash - 1) * 100, 2),
        "cagr_pct": round(((eqs.iloc[-1] / start_cash) ** (1 / yrs) - 1) * 100, 2) if yrs > 0 else None,
        "max_drawdown_pct": round(float(dd) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else None,
        "trades": trades,
        "benchmark_total_return_pct": round((beq.iloc[-1] / start_cash - 1) * 100, 2), "benchmark_max_drawdown_pct": round(float(bdd) * 100, 2),
        "beats_benchmark": bool(eqs.iloc[-1] > beq.iloc[-1]),
        "equity": [[str(d.date()), round(float(v), 2)] for d, v in eqs.iloc[:: max(1, len(eqs) // 300)].items()],
        "benchmark": [[str(d.date()), round(float(v), 2)] for d, v in beq.iloc[:: max(1, len(beq) // 300)].items()],
        "notes": ["Price-based factors only (fundamentals are not back-tested: today's SEC data would leak the future).",
                  f"Costs {cost_bps:.0f} bps per unit of turnover. Benchmark = equal-weight buy-and-hold of the same universe.",
                  "Past performance does not predict future results; universe survivorship can flatter results."],
    }
    return out
