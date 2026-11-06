"""2026-11-05 audit: auto optimizer mode, adaptive GA budget + plateau early stop with restart, warm starts, drawdown-aware edge objective."""
from __future__ import annotations

import math
import random

from app.optimize import refinement as R
from app.optimize.parameter_space import GeneMeta


def _genes(n):
    return [GeneMeta(path=(i,), kind="x", is_int=False, lo=0.0, hi=1.0, base_value=0.5, label=f"x{i}") for i in range(n)]


def test_auto_picks_tpe_for_small_spaces_and_genetic_for_large():
    assert R.resolve_optimizer_mode("genetic", 3) == "genetic"
    assert R.resolve_optimizer_mode("cma_es", 30) == "cma_es"
    assert R.resolve_optimizer_mode("auto", R.AUTO_TPE_MAX_PARAMS + 1) == "genetic"
    small = R.resolve_optimizer_mode("auto", 3)
    assert small in ("tpe", "genetic")        # genetic only when optuna is not installed
    assert "auto" in R.OPTIMIZER_MODES
    R.RefinementConfig(optimizer_mode="auto")  # accepted by validation


def test_drawdown_penalty_free_inside_comfort_band_and_grows_beyond():
    assert R.edge_drawdown_penalty(2.0, 10.0) == 0.0
    assert R.edge_drawdown_penalty(5.0, 10.0) == 0.0
    mid, full = R.edge_drawdown_penalty(7.5, 10.0), R.edge_drawdown_penalty(10.0, 10.0)
    assert 0.0 < mid < full and math.isclose(full, 0.5)
    assert R.edge_drawdown_penalty(None, 10.0) == 0.0 and R.edge_drawdown_penalty(5.0, None) == 0.0
    assert R.edge_drawdown_penalty(15.0, 10.0) > full


def _run_ga(cfg, fitness_fn):
    genes = _genes(3)
    seen = []

    def evaluate(genome, gen, keep_full=False, track=True):
        c = R.Candidate(generation=gen, genome=list(genome), fitness=fitness_fn(genome))
        seen.append(c)
        return c

    base = evaluate([0.5, 0.5, 0.5], 0)
    logs = []
    R._run_genetic_search(genes, evaluate, cfg, logs.append, random.Random(1), base)
    return seen, logs


def test_early_stop_halts_a_flat_search_and_restarts_once():
    cfg = R.RefinementConfig(population_size=6, generations=20, early_stop_patience=3, early_stop_restarts=1, random_seed=1)
    seen, logs = _run_ga(cfg, lambda g: 1.0)                         # perfectly flat: nothing ever improves
    full_budget = 6 * 21
    assert len(seen) < full_budget
    assert any("restarting from a fresh random population" in m for m in logs)
    assert any("plateaued" in m for m in logs)


def test_without_patience_the_search_spends_its_whole_budget():
    cfg = R.RefinementConfig(population_size=6, generations=5, random_seed=1)
    seen, logs = _run_ga(cfg, lambda g: 1.0)
    assert not any("plateaued" in m for m in logs)
    assert len(seen) >= 6 * 4   # baseline + initial population + every bred generation


def test_warm_start_genomes_seed_the_first_population():
    target = [0.9, 0.1, 0.4]
    cfg = R.RefinementConfig(population_size=6, generations=1, warm_start_genomes=[target], random_seed=1)
    seen, _ = _run_ga(cfg, lambda g: -sum((a - b) ** 2 for a, b in zip(g, target)))
    assert any(c.genome == target for c in seen[:6])
