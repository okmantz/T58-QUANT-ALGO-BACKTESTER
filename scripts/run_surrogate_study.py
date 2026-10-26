"""Surrogate study: does the Strategy Discovery Engine hand out forward-test candidates on data with NO edge, and does it find a planted one?

    python -m scripts.run_surrogate_study [--null-runs 60] [--planted-runs 5] [--families a,b,c | --batch 1] [--null-seeds 200]
                                          [--days 80] [--markets 8] [--max-candidates 120] [--out reports/surrogate_gate.json]

Runs the REAL engine (all mandatory promotion steps) on ``--null-runs`` random-walk surrogates and on planted-edge surrogates of
+0.10R / +0.20R / +0.30R, then reports

  * the share of no-edge runs that yield a forward-test candidate  (target < 5%)
  * the share of planted-edge runs where the planted edge is found

and saves the report. The rollout gate for new family batches (app.discovery_engine.surrogate.gate_status) opens ONLY when the saved
report is fresh, was produced by the current engine code, shows < 5% false discoveries over >= 20 null runs and >= 50% power on the
+0.20R / +0.30R plants. ~60 null runs are needed to *prove* < 5% at 95% confidence (0 hits in 60 -> upper bound 4.9%).
Re-run after every major change to the engine.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--null-runs", type=int, default=60)
    ap.add_argument("--planted-runs", type=int, default=5)
    ap.add_argument("--families", default="", help="comma-separated; default = the active (open) families")
    ap.add_argument("--batch", type=int, default=0, help="include this v13 rollout batch's ready families")
    ap.add_argument("--null-seeds", type=int, default=200)
    ap.add_argument("--days", type=int, default=80)
    ap.add_argument("--markets", type=int, default=8)
    ap.add_argument("--max-candidates", type=int, default=120)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    from app.discovery.hypothesis import preregister_v13_families
    from app.discovery_engine import surrogate as sg
    from app.search import families_v13 as f13
    from app.search import strategy_space as ss

    preregister_v13_families()
    if a.families:
        fams = [f.strip() for f in a.families.split(",") if f.strip()]
    else:
        fams = list(ss.active_family_names())
    if a.batch:
        fams += [n for n in f13.families_in_batch(a.batch) if f13.V13_META[n].status == "ready" and n not in f13.RETIRED_V13 and n not in fams]
    print(f"Surrogate study: {len(fams)} families, {a.null_runs} null runs, {a.planted_runs} planted runs per target size.")
    pipe = sg.engine_pipeline(fams, null_seeds=a.null_seeds, max_candidates=a.max_candidates)
    rep = sg.run_surrogate_study(pipe, n_null=a.null_runs, n_planted=a.planted_runs, n_markets=a.markets, days=a.days, progress=print,
                                 min_null_runs=min(a.null_runs, 20))
    if a.batch and a.batch not in rep.opened_batches:
        rep.opened_batches = sorted(set(rep.opened_batches) | {a.batch})
    path = sg.save_report(rep, a.out or None)
    print()
    print(f"False-discovery rate: {rep.null_candidates}/{rep.n_null} = {rep.null_rate:.1%} (95% upper bound {rep.null_rate_upper95:.1%}); target < 5%")
    for row in rep.planted:
        print(f"Planted +{row['target_r']:.2f}R (measured {row['measured_r']:+.3f}R): found {row['found']}/{row['runs']}")
    for n in rep.notes:
        print("note:", n)
    st = sg.gate_status(path)
    print("Gate:", "OPEN -- batches " + ",".join(map(str, sg.enabled_batches(path))) if st["open"] else "CLOSED -- " + "; ".join(st["reasons"]))
    print(f"Report saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
