"""Reproducible research-integrity benchmark (fast statistical-chain version).

    python -m scripts.run_integrity_benchmark [--null-runs 60] [--planted-runs 10] [--seed 0] [--days 250] [--no-save]

Runs a fixed in-sample-selection / out-of-sample-confirmation chain on random-walk surrogates (false discoveries) and on planted-edge
surrogates of +0.10R / +0.20R / +0.30R (detection). The same arguments always give the same numbers. This is NOT the production discovery
engine (that is scripts/run_surrogate_study.py, which is slow); it is a quick, exact check of the statistics. Passing needs < 5% false
discoveries over >= 60 null runs and >= 50% power on the +0.20R and +0.30R plants with >= 10 runs per size. The false-discovery rate on
real data is not measured by either script.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--null-runs", type=int, default=60)
    ap.add_argument("--planted-runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=250)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)
    from app.validation import research_integrity as ri
    res = ri.run_benchmark(n_null=a.null_runs, n_planted=a.planted_runs, seed=a.seed, days=a.days)
    print(res["label"])
    print(f"version {res['version']}  dataset {res['dataset_fingerprint']}  seed {res['seed']}")
    print(f"False discoveries: {res['null_candidates']}/{res['n_null']} = {res['null_rate']:.1%} (95% upper bound {res['null_rate_upper95']:.1%}); target < 5%")
    for p in res["planted"]:
        print(f"Planted +{p['target_r']:.2f}R (measured {p['measured_r']:+.3f}R): found {p['found']}/{p['runs']}")
    print("PASS" if res["passed"] else "FAIL: " + "; ".join(res["reasons"]))
    if not a.no_save:
        print("saved to", ri.save_benchmark(res))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
