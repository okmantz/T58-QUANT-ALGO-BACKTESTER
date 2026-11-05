"""Run the GA-vs-random proof on the built-in landscapes.   python -m scripts.run_ga_vs_random [--runs 30] [--out reports/ga_vs_random.json]"""
import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    from app.optimize.ga_vs_random import BENCHMARKS, compare, make_genes
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--genes", type=int, default=6)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    out = {}
    for name, fn in BENCHMARKS.items():
        r = compare(fn, make_genes(a.genes), population=a.population, generations=a.generations, runs=a.runs)
        out[name] = r.to_dict()
        print(f"{name:10s} GA {r.ga_mean_best:+.4f} vs random {r.random_mean_best:+.4f}  wins {r.ga_wins}-{r.random_wins}  p={r.sign_test_p:.4f}  -> {r.verdict}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
