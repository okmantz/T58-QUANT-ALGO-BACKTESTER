"""Benchmark the optimizer modes (genetic / tpe / cma_es / auto) on synthetic noisy landscapes of several sizes, at an EQUAL evaluation budget.

2026-11-05 audit: ``auto`` picks TPE for small spaces and CMA-ES for larger ones. That rule is a hypothesis; this script is how it is checked.
Scores are the TRUE (noise-free) value of each mode's reported best point, so a mode that merely got lucky on the noise does not win.

    python tools/benchmark_optimizer_modes.py [--seeds 8] [--budget-per-param 25]
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.optimize import refinement as R  # noqa: E402
from app.optimize.parameter_space import GeneMeta  # noqa: E402


def _landscape(n: int, seed: int):
    rng = random.Random(seed)
    centre = [rng.uniform(0.25, 0.75) for _ in range(n)]
    rot = [rng.uniform(-0.4, 0.4) for _ in range(n)]

    def true(x01):
        s = 0.0
        for i, v in enumerate(x01):
            d = v - centre[i] - rot[i] * (x01[i - 1] - 0.5)   # mild correlation between parameters
            s += d * d
        return 1.0 - math.sqrt(s / n) * 2.0 + 0.15 * math.sin(14 * x01[0])   # one ripple = local optima

    return true


def run(mode: str, n: int, seed: int, budget_per_param: int) -> float:
    genes = [GeneMeta(path=(i,), kind="x", is_int=False, lo=0.0, hi=1.0, base_value=0.5, label=f"x{i}") for i in range(n)]
    true = _landscape(n, seed)
    noise = random.Random(seed + 99)
    tracked: list = []

    def evaluate(genome, gen, keep_full=False, track=True):
        c = R.Candidate(generation=gen, genome=list(genome), fitness=true(genome) + noise.gauss(0, 0.05))
        if track:
            tracked.append(c)
        return c

    budget = budget_per_param * n
    pop = 12
    cfg = R.RefinementConfig(population_size=pop, generations=max(budget // pop - 1, 1), optimizer_mode=mode, random_seed=seed)
    log = lambda *_: None  # noqa: E731
    eff = R.resolve_optimizer_mode(mode, n)
    base = evaluate([g.base_value for g in genes], 0)
    if eff == "genetic":
        R._run_genetic_search(genes, evaluate, cfg, log, random.Random(seed), base)
    elif eff == "tpe":
        R._run_tpe_search(genes, evaluate, cfg, log)
    else:
        R._run_cma_es_search(genes, evaluate, cfg, log)
    best = max(tracked, key=lambda c: c.fitness)
    return true(best.genome)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--budget-per-param", type=int, default=25)
    a = ap.parse_args()
    print(f"{'params':>6} {'mode':>8} {'mean true score':>16} {'sd':>7}   (higher is better; equal budget of {a.budget_per_param}/param)")
    for n in (3, 6, 10, 16):
        row = {}
        for mode in ("genetic", "tpe", "cma_es", "auto"):
            vals = [run(mode, n, s, a.budget_per_param) for s in range(a.seeds)]
            row[mode] = statistics.mean(vals)
            print(f"{n:>6} {mode:>8} {row[mode]:>16.4f} {statistics.pstdev(vals):>7.4f}")
        print(f"       -> best: {max(row, key=row.get)}")


if __name__ == "__main__":
    main()
