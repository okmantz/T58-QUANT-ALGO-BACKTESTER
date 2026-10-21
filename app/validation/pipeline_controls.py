"""Positive and negative pipeline controls (report 2026-10-20, items 5 and 7).

A validation pipeline that rejects everything is indistinguishable from a broken one. These controls measure the EDGE-TEST LAYER itself:

  NEGATIVE control  trade streams with a TRUE mean of 0R (random entries / shuffled signals have exactly this property).
                    The pooled test must call PASS on no more than ~alpha of them. Measured: false-positive rate.
  POSITIVE control  trade streams with a PLANTED edge of +0.10R ... +0.40R on a realistic trade count. Measured: detection rate
                    (power), single market versus pooled over ``n_markets``.

Both are Monte Carlo over R-multiples (sd 1.3R, same-day correlation across markets via a shared daily shock), so they check the statistics
the verdict depends on - day-block bootstrap, cluster t, trade-count floor - not the strategy engine. ``results`` are cached to
``reports/pipeline_controls.json`` and published on the dashboard; a positive control that cannot detect +0.30R is flagged as a PIPELINE
problem, not a strategy problem.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.validation.pooled_edge import analytic_power_table, pooled_edge_test

_PATH = Path(__file__).resolve().parents[2] / "reports" / "pipeline_controls.json"
SD_R = 1.3


def _simulate(rng, edge, n_markets, trades_per_market, rho_day=0.25):
    days = pd.bdate_range("2019-01-02", periods=max(trades_per_market * 2, 50))
    per = {}
    shock = rng.normal(0.0, 1.0, len(days))             # shared daily shock -> cross-market correlation
    for m in range(n_markets):
        idx = np.sort(rng.choice(len(days), size=trades_per_market, replace=False))
        eps = rng.normal(0.0, 1.0, trades_per_market)
        r = edge + SD_R * (np.sqrt(rho_day) * shock[idx] + np.sqrt(1 - rho_day) * eps)
        per[f"M{m}"] = (list(r), list(days[idx].strftime("%Y-%m-%d")))
    return per


def negative_control(n_sims: int = 200, n_markets: int = 8, trades_per_market: int = 60, seed: int = 7, n_boot: int = 400) -> dict:
    rng = np.random.default_rng(seed)
    passes = sum(pooled_edge_test(_simulate(rng, 0.0, n_markets, trades_per_market), n_boot=n_boot, seed=int(rng.integers(1 << 30))).passed
                 for _ in range(n_sims))
    fpr = passes / n_sims
    return {"n_sims": n_sims, "false_positive_rate": fpr, "target_max": 0.10, "ok": fpr <= 0.10}


def positive_control(edges=(0.10, 0.20, 0.30, 0.40), n_sims: int = 100, n_markets: int = 8, trades_per_market: int = 60, seed: int = 8,
                     n_boot: int = 400) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for e in edges:
        single = pooled = 0
        for _ in range(n_sims):
            sim = _simulate(rng, e, n_markets, trades_per_market)
            first = next(iter(sim))
            single += pooled_edge_test({first: sim[first]}, min_trades=30, n_boot=n_boot, seed=int(rng.integers(1 << 30))).passed
            pooled += pooled_edge_test(sim, n_boot=n_boot, seed=int(rng.integers(1 << 30))).passed
        rows.append({"edge_r": e, "single_market": single / n_sims, "pooled": pooled / n_sims})
    detects_30 = next((r["pooled"] for r in rows if abs(r["edge_r"] - 0.30) < 1e-9), None)
    return {"n_sims": n_sims, "n_markets": n_markets, "trades_per_market": trades_per_market, "rows": rows,
            "ok": detects_30 is None or detects_30 >= 0.80,
            "note": "pooled detection of a planted +0.30R edge must be >= 80%, otherwise the pipeline (not the strategy) is the problem"}


def run_controls(n_sims: int = 150, save: bool = True) -> dict:
    out = {"t": time.time(), "negative": negative_control(n_sims=n_sims), "positive": positive_control(n_sims=max(40, n_sims // 2)),
           "analytic_power": analytic_power_table()}
    out["ok"] = bool(out["negative"]["ok"] and out["positive"]["ok"])
    if save:
        try:
            _PATH.parent.mkdir(parents=True, exist_ok=True)
            _PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    return out


def load_controls(max_age_days: float = 30.0, compute_if_missing: bool = True) -> dict | None:
    """Cached result for the dashboard card; recomputed (a few seconds) when missing or older than ``max_age_days``."""
    try:
        d = json.loads(_PATH.read_text(encoding="utf-8"))
        if time.time() - float(d.get("t", 0)) <= max_age_days * 86400:
            return d
    except Exception:  # noqa: BLE001
        pass
    return run_controls() if compute_if_missing else None
