#!/usr/bin/env python3
"""Build the starter Strategy Graveyard seed (app/search/seed_data/graveyard_seed.jsonl).

Every row is a REAL measured failure: a candidate from the app's own strategy families is generated with
app.search.strategy_space, backtested with the real engine (app.backtest.engine.run_backtest, non-zero spread /
slippage / commission) on a dataset bundled in data/raw/, on an in-sample slice and a held-out out-of-sample slice,
and only candidates that actually fail are kept. Nothing is invented: the numbers in each row's reason text are the
measured numbers from that run, and each row's `notes` records the dataset, date range and costs it was measured on.

Scope of the claim (also written into every row): "no edge on THIS dataset slice under THESE costs" -- not a proof the
idea can never work elsewhere. That is why seed rows are filed under the exact dataset label + timeframe they were
measured on (the graveyard is per-instrument by design -- see app.search.graveyard.graveyard_path_for).

Usage:
    python scripts/build_graveyard_seed.py                 # build (takes ~10-25 min on 2 cores)
    python scripts/build_graveyard_seed.py --quick         # tiny smoke run
    python scripts/build_graveyard_seed.py --target 600    # stop once this many failures collected

To fold YOUR OWN local graveyard (data/evolution/strategy_graveyard__*.jsonl) into the shipped seed, use
scripts/export_graveyard_seed.py instead -- that one never runs backtests, it only copies rows you already have.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

OUT = ROOT / "app" / "search" / "seed_data" / "graveyard_seed.jsonl"

# dataset label (as Strategy-Lab dataset pickers show it) -> (file, pip_size, spread_pips, slippage_pips, bars)
DATASETS = [
    ("EURUSD/EURUSD60.csv", "data/raw/EURUSD/EURUSD60.csv", 0.0001, 0.8, 0.2, 30000),
    ("EURUSD/EURUSD240.csv", "data/raw/EURUSD/EURUSD240.csv", 0.0001, 0.8, 0.2, 12000),
    ("GBPUSD/GBPUSD60.csv", "data/raw/GBPUSD/GBPUSD60.csv", 0.0001, 1.0, 0.3, 30000),
    ("GBPUSD/GBPUSD240.csv", "data/raw/GBPUSD/GBPUSD240.csv", 0.0001, 1.0, 0.3, 12000),
    ("XAUUSD (GOLD)/XAUUSD60.csv", "data/raw/XAUUSD (GOLD)/XAUUSD60.csv", 0.1, 3.0, 1.0, 30000),
    ("XAUUSD (GOLD)/XAUUSD240.csv", "data/raw/XAUUSD (GOLD)/XAUUSD240.csv", 0.1, 3.0, 1.0, 12000),
]
COMMISSION_PER_TRADE = 3.0
ACCOUNT = 100_000.0
RISK_PCT = 0.5
MAX_DD_LIMIT_PCT = 10.0
MIN_TRADES = 30


def _run(args):
    label, path, pip, spread, slip, bars, family, per_family, seed = args
    import pandas as pd

    from app.backtest.engine import run_backtest
    from app.backtest.risk import RiskConfig
    from app.data.importer import import_csv
    from app.data.timeframe_resample import infer_timeframe_label
    from app.search.graveyard import param_signature
    from app.search.strategy_space import build_strategy_from_spec, generate_search_space

    df = import_csv(str(ROOT / path)).dataframe.tail(bars).reset_index(drop=True)
    tf = infer_timeframe_label(df)
    split = int(len(df) * 0.6)
    d_is, d_oos = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
    t0, t1, t2 = str(df["timestamp"].iloc[0])[:10], str(df["timestamp"].iloc[split])[:10], str(df["timestamp"].iloc[-1])[:10]
    risk = RiskConfig(initial_balance=ACCOUNT, risk_value=RISK_PCT, pip_size=pip, spread_pips=spread,
                      slippage_pips=slip, commission_per_trade=COMMISSION_PER_TRADE)
    try:
        space = generate_search_space("family", family=family, max_candidates=per_family, seed=seed)
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for cid, spec in space.candidates.items():
        params = (space.meta.get(cid) or {}).get("params") or {}
        try:
            r_is = run_backtest(d_is, build_strategy_from_spec(spec), risk, lookahead_check="skip")
            r_oos = run_backtest(d_oos, build_strategy_from_spec(spec), risk, lookahead_check="skip")
            s_is, s_oos = r_is.statistics, r_oos.statistics
            if s_is is None or s_oos is None:
                continue
        except Exception:  # noqa: BLE001 -- a candidate the engine can't run is not a measured failure
            continue
        n_is, n_oos = int(s_is.total_trades), int(s_oos.total_trades)
        ret_is, ret_oos = float(s_is.return_pct), float(s_oos.return_pct)
        pf_is, pf_oos = float(s_is.profit_factor or 0), float(s_oos.profit_factor or 0)
        dd = max(float(s_is.max_drawdown_pct), float(s_oos.max_drawdown_pct))
        n_all = n_is + n_oos
        if n_all < MIN_TRADES:
            primary, reason = "too few trades", (
                f"Only {n_all} trades across {t0} to {t2} -- too few to tell an edge from luck.")
            oos = "positive" if ret_oos > 0 else "negative"
        elif ret_oos <= 0 and ret_is > 0:
            primary, reason = "out-of-sample collapse", (
                f"Profitable in-sample ({ret_is:+.1f}%, PF {pf_is:.2f}) but lost money on held-out data "
                f"({ret_oos:+.1f}%, PF {pf_oos:.2f}) -- an overfit edge that did not carry forward.")
            oos = "negative"
        elif ret_oos <= 0 and ret_is <= 0:
            primary, reason = "no edge after costs", (
                f"Lost money in both halves (in-sample {ret_is:+.1f}%, out-of-sample {ret_oos:+.1f}%; "
                f"PF {pf_is:.2f}/{pf_oos:.2f}) once spread, slippage and commission were charged.")
            oos = "negative"
        elif dd > MAX_DD_LIMIT_PCT:
            primary, reason = "drawdown breach", (
                f"Profitable ({ret_is:+.1f}% / {ret_oos:+.1f}%) but max drawdown reached {dd:.1f}%, "
                f"through a {MAX_DD_LIMIT_PCT:.0f}% prop-style limit.")
            oos = "positive"
        elif pf_oos < 1.1:
            primary, reason = "marginal edge", (
                f"Held-out profit factor only {pf_oos:.2f} ({ret_oos:+.1f}%) -- inside the noise band after costs.")
            oos = "positive"
        else:
            continue  # survived this simple screen -- not a graveyard row
        rows.append({
            "_seed_instrument": label, "_seed_timeframe": tf,
            "candidate_id": f"seed-{cid}-{tf}", "family": family, "generation": None,
            "stage_died": "not_ranked", "reason": reason, "oos_result": oos,
            "neighbor_robustness_pct": None, "monte_carlo_failure_pct": None, "prop_sim_pass_pct": None,
            "cpcv_oos_pass_pct": None, "pbo": None, "fitness_score": round(pf_oos, 3),
            "param_signature": param_signature(family, params),
            "notes": [
                f"starter seed: measured with the T58 engine on {label} ({tf}), {t0} to {t2}; "
                f"in-sample {t0}-{t1}, held-out {t1}-{t2}",
                f"costs: spread {spread}p, slippage {slip}p, ${COMMISSION_PER_TRADE:g}/trade; {RISK_PCT}% risk on ${ACCOUNT:,.0f}",
                "claim is scoped to this dataset slice and these costs -- not a proof the idea can never work elsewhere",
                f"params: {json.dumps(params, sort_keys=True, default=str)}",
            ],
            "primary_failure": primary,
            "weakness": f"{n_all} trades; max drawdown {dd:.1f}%",
        })
    return rows


def build(per_family: int, target: int, quick: bool) -> int:
    from app.search.strategy_space import FAMILIES

    fams = list(FAMILIES)
    datasets = DATASETS[:1] if quick else DATASETS
    if quick:
        fams = fams[:6]
    tasks = [(l, p, pip, sp, sl, b, f, per_family, 1000 + i)
             for i, f in enumerate(fams) for (l, p, pip, sp, sl, b) in datasets]
    seen, rows = set(), []
    with Pool(processes=2) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_run, tasks, chunksize=1), 1):
            for r in chunk:
                key = (r["_seed_instrument"], r["_seed_timeframe"], r["param_signature"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(tasks)} tasks, {len(rows)} failures", flush=True)
            if target and len(rows) >= target * 1.15:
                pool.terminate()
                break
    rows.sort(key=lambda r: (r["_seed_instrument"], r["_seed_timeframe"], r["family"], r["candidate_id"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} failed variants -> {OUT.relative_to(ROOT)}")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    build(a.per_family, a.target, a.quick)
