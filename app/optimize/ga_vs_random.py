"""GA-vs-random proof harness.

Question it answers: *given the same evaluation budget, does the genetic search beat plain random search?*  If it does
not, the GA is just an expensive way to overfit and the search budget is better spent elsewhere.

``compare(objective, genes, ...)`` runs the REAL ``_run_genetic_search`` and a uniform random search on identical
budgets (``population_size * (generations + 1)`` evaluations) over many paired seeds, then reports the mean best score
of each, the paired win rate and an exact one-sided sign-test p-value.  ``objective(genome) -> float`` is any callable
(a synthetic landscape, or a wrapper around a real backtest).  Built-in landscapes (``BENCHMARKS``) give a quick,
dependency-free check.  The harness is deterministic for a given ``seed``.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Callable, Sequence

from app.optimize.refinement import Candidate, RefinementConfig, _random_gene_value, _run_genetic_search


def make_genes(n: int, lo: float = 0.0, hi: float = 1.0):
    return [SimpleNamespace(lo=lo, hi=hi, is_int=False, name=f"g{i}") for i in range(n)]


def _sphere(g):  # smooth single peak at 0.7
    return -sum((x - 0.7) ** 2 for x in g) / len(g)


def _rastrigin(g):  # many local peaks
    return -sum(((x - 0.5) * 10) ** 2 - 10 * math.cos(2 * math.pi * (x - 0.5) * 10) + 10 for x in g) / len(g)


def _needle(g):  # flat with a narrow ridge: GA should have little advantage
    return math.exp(-sum((x - 0.31) ** 2 for x in g) / 0.002)


BENCHMARKS: dict[str, Callable] = {"sphere": _sphere, "rastrigin": _rastrigin, "needle": _needle}


@dataclass
class ComparisonResult:
    runs: int
    budget: int
    ga_mean_best: float
    random_mean_best: float
    ga_wins: int
    random_wins: int
    ties: int
    sign_test_p: float
    verdict: str
    per_seed: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n)


def run_random(objective: Callable, genes: Sequence, budget: int, rng: random.Random) -> float:
    return max(objective([_random_gene_value(g, rng) for g in genes]) for _ in range(budget))


def run_ga(objective: Callable, genes: Sequence, cfg: RefinementConfig, rng: random.Random) -> float:
    def evaluate(genome, gen):
        return Candidate(generation=gen, genome=list(genome), fitness=float(objective(genome)))

    baseline = evaluate([_random_gene_value(g, rng) for g in genes], 0)
    _, _, best = _run_genetic_search(list(genes), evaluate, cfg, lambda *_: None, rng, baseline)
    return best.fitness


def compare(objective: Callable, genes: Sequence, *, population: int = 16, generations: int = 8, runs: int = 30, seed: int = 0,
            alpha: float = 0.05) -> ComparisonResult:
    cfg = RefinementConfig(population_size=population, generations=generations)
    budget = population * (generations + 1)
    rows, w, l, t = [], 0, 0, 0
    for i in range(runs):
        ga = run_ga(objective, genes, cfg, random.Random(seed * 100003 + i))
        rd = run_random(objective, genes, budget, random.Random(seed * 100003 + 50000 + i))
        rows.append({"seed": i, "ga": ga, "random": rd})
        if abs(ga - rd) < 1e-12:
            t += 1
        elif ga > rd:
            w += 1
        else:
            l += 1
    p = _sign_test_p(w, l)
    if p < alpha:
        verdict = "GA beats random search at equal budget"
    elif _sign_test_p(l, w) < alpha:
        verdict = "Random search beats the GA at equal budget"
    else:
        verdict = "No significant difference at equal budget"
    return ComparisonResult(runs, budget, sum(r["ga"] for r in rows) / runs, sum(r["random"] for r in rows) / runs, w, l, t, p, verdict, rows)
