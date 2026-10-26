"""Run the ONE Strategy Discovery Engine from the command line.

    python -m scripts.run_discovery_engine --preset search_lab --data data/raw/NQ_5m.csv --symbol NQ [--families a,b | ready | all]
                                           [--pool ES=data/raw/ES_5m.csv --pool YM=...] [--max-candidates 2000] [--seed 42]
                                           [--out reports/discovery_run.json] [--retest]

Presets (the old labs): search_lab, evolution_lab, evolution_loop, forge, speed_run, discovery.
Every preset shares ONE objective (edge_lb_r), ONE entry floor (100+ trades, t>=1, cost<0.15R), ONE lifetime trial ledger, a sealed slice
that is read once, and the mandatory promotion sequence (reproduction -> sequence-deflated Sharpe -> random-timing null -> pooled
cross-market edge -> sealed slice). A passing champion is a FORWARD-TEST CANDIDATE, nothing more.

--retest first re-runs every earlier champion/near-miss on the bars that arrived after its window and prints the lab's ranking.
Before the first run, ``python -m scripts.bootstrap_data`` shows which pooled markets you actually have.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: str):
    from app.data.importer import import_csv
    res = import_csv(path)
    if not res.is_valid:
        raise SystemExit(f"could not import {path}: {[i.message for i in res.issues][:2]}")
    return res.dataframe


def _risk(symbol: str):
    from app.backtest.risk import RiskConfig
    from app.data.instrument_specs import apply_any_instrument_spec, micro_equivalent
    m = micro_equivalent(symbol.upper())
    return apply_any_instrument_spec(RiskConfig(initial_balance=100000, risk_mode="percent", risk_value=0.5), (m.symbol if m else symbol.upper()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="search_lab")
    ap.add_argument("--data", required=True)
    ap.add_argument("--symbol", default="UNKNOWN")
    ap.add_argument("--timeframe", default="unknown")
    ap.add_argument("--families", default="")
    ap.add_argument("--pool", action="append", default=[], help="SYMBOL=path for the pooled cross-market test (repeatable)")
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--max-promote", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sealed-frac", type=float, default=0.10)
    ap.add_argument("--out", default="reports/discovery_run.json")
    ap.add_argument("--retest", action="store_true")
    a = ap.parse_args(argv)

    from app.discovery.hypothesis import preregister_v13_families
    from app.discovery_engine.engine import DiscoveryEngine, EngineRequest
    from app.discovery_engine.retest import RetestStore, lab_ranking, run_retests
    from app.prop.simulator import PropRules

    preregister_v13_families()
    df = _load(a.data)
    risk = _risk(a.symbol)
    if a.retest:
        for key, out in run_retests(df, risk, RetestStore(), instrument=a.symbol, progress=print):
            pass
        for row in lab_ranking()[:15]:
            print(f"  {row['name']:34s} retests={row['retests']:.0f} hits={row['hits']:.0f} shrunk={row['shrunk_hit_rate']:.0%} meanR={row['mean_oos_r']:+.3f}")
    frames, risks = {a.symbol.upper(): df}, {a.symbol.upper(): risk}
    for item in a.pool:
        sym, _, path = item.partition("=")
        frames[sym.upper()], risks[sym.upper()] = _load(path), _risk(sym)
    fams = None
    if a.families:
        fams = a.families if a.families in ("all", "ready") else [f.strip() for f in a.families.split(",") if f.strip()]
    req = EngineRequest(preset=a.preset, df=df, risk=risk, prop_rules=PropRules(), instrument=a.symbol.upper(), timeframe=a.timeframe, families=fams,
                        seed=a.seed, sealed_frac=a.sealed_frac, frames=frames, risks=risks, max_candidates=a.max_candidates or None,
                        max_promote=a.max_promote or None, progress_cb=print)
    res = DiscoveryEngine().run(req)
    for note in res.notes:
        print("note:", note)
    print(f"\n{res.preset}: {res.n_evaluations} evaluations, {len(res.champions)} ranked champion(s), "
          f"{len(res.forward_test_candidates)} forward-test candidate(s). Effective trials: {res.trial_inputs.get('n_tests')}")
    for c, p in zip(res.champions, res.promotions):
        verdict = "PASS -> forward test" if p.get("passed") else f"FAIL at {p.get('failed_step')}: {p.get('failed_reason')}"
        lb = c.get("edge_lb_r")
        print(f"  {c.get('family') or '?':30s} edge_lb_r={'n/a' if lb is None else format(lb, '+.3f')}  {verdict}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res.to_dict(), indent=1, default=str), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
