"""Tests for plateau-robust champion selection (app.optimize.refinement):
_plateau_robustness_score, _neighbor_genomes, and _select_plateau_robust.

The whole point of this feature is to stop the GA from handing back a
thin, one-point fitness spike (classic in-sample curve-fit noise) just
because it happens to be the single highest number seen anywhere in the
search, when a nearby, slightly-lower-scoring region would hold up under
a small nudge to any one parameter."""
from __future__ import annotations

import math

import pytest

from app.optimize.parameter_space import GeneMeta
from app.optimize.refinement import (
    Candidate,
    RefinementConfig,
    _neighbor_genomes,
    _plateau_robustness_score,
    _select_plateau_robust,
)


def _gene(kind="period", lo=1.0, hi=100.0, is_int=True) -> GeneMeta:
    return GeneMeta(path=("x",), kind=kind, is_int=is_int, lo=lo, hi=hi, base_value=(lo + hi) / 2, label=kind)


def _candidate(genome: list, fitness: float, generation: int = 0) -> Candidate:
    return Candidate(generation=generation, genome=genome, fitness=fitness)


# ---------------------------------------------------------------------------
# _plateau_robustness_score
# ---------------------------------------------------------------------------

def test_plateau_score_rewards_a_flat_neighborhood_over_a_spike():
    spike = _plateau_robustness_score(center_fitness=2.0, neighbor_fitnesses=[0.1, 0.1, 0.1, 0.1])
    plateau = _plateau_robustness_score(center_fitness=1.5, neighbor_fitnesses=[1.4, 1.45, 1.4, 1.5])
    assert plateau > spike


def test_plateau_score_ignores_non_finite_neighbor_values():
    score = _plateau_robustness_score(center_fitness=1.0, neighbor_fitnesses=[float("-inf"), float("nan"), 1.0])
    assert math.isfinite(score)


def test_plateau_score_is_negative_inf_when_nothing_is_finite():
    score = _plateau_robustness_score(center_fitness=float("nan"), neighbor_fitnesses=[float("-inf")])
    assert score == float("-inf")


# ---------------------------------------------------------------------------
# _neighbor_genomes
# ---------------------------------------------------------------------------

def test_neighbor_genomes_perturbs_one_gene_at_a_time_both_directions():
    genes = [_gene(lo=0.0, hi=100.0), _gene(lo=0.0, hi=10.0)]
    genome = [50.0, 5.0]
    neighbors = _neighbor_genomes(genome, genes, step_frac=0.1)
    # 2 genes * 2 directions = 4 neighbor points, each differing from the
    # base genome in exactly one position.
    assert len(neighbors) == 4
    for n in neighbors:
        diffs = [i for i in range(len(genome)) if n[i] != genome[i]]
        assert diffs == [0] or diffs == [1]


def test_neighbor_genomes_clamps_to_gene_bounds_at_the_edge():
    genes = [_gene(lo=0.0, hi=100.0, is_int=False)]
    genome = [99.0]
    neighbors = _neighbor_genomes(genome, genes, step_frac=0.5)  # would overshoot to 149
    assert all(n[0] <= 100.0 for n in neighbors)


def test_neighbor_genomes_skips_a_step_that_collapses_to_the_same_integer():
    # An integer gene with a tiny span and a tiny step_frac can round back
    # to the exact same value -- that direction contributes no new probe
    # point rather than a duplicate of the center.
    genes = [_gene(lo=1.0, hi=2.0, is_int=True)]
    genome = [1.0]
    neighbors = _neighbor_genomes(genome, genes, step_frac=0.01)
    assert all(n[0] != genome[0] for n in neighbors)


# ---------------------------------------------------------------------------
# _select_plateau_robust
# ---------------------------------------------------------------------------

def test_select_plateau_robust_swaps_away_from_a_thin_spike():
    genes = [_gene(lo=0.0, hi=100.0)]
    spike = _candidate([50.0], fitness=3.0)          # highest raw fitness...
    plateau_center = _candidate([20.0], fitness=2.0)  # ...but this region holds up
    candidates = [spike, plateau_center]
    cfg = RefinementConfig(plateau_finalist_pool=2, plateau_neighbor_step_frac=0.1)

    def evaluate_cheap(genome):
        # The spike's fitness collapses one step away in either direction;
        # the plateau region barely moves.
        if genome[0] in (spike.genome[0] - 10.0, spike.genome[0] + 10.0):
            return _candidate(genome, fitness=0.05)
        if genome[0] in (plateau_center.genome[0] - 10.0, plateau_center.genome[0] + 10.0):
            return _candidate(genome, fitness=1.9)
        return _candidate(genome, fitness=0.0)

    chosen, report = _select_plateau_robust(candidates, genes, cfg, evaluate_cheap)
    assert chosen.genome == plateau_center.genome
    assert report is not None
    assert report["swapped"] is True
    assert report["raw_best_fitness"] == pytest.approx(3.0)
    assert report["chosen_fitness"] == pytest.approx(2.0)


def test_select_plateau_robust_keeps_the_raw_best_when_it_is_already_flat():
    genes = [_gene(lo=0.0, hi=100.0)]
    flat_best = _candidate([50.0], fitness=2.0)
    worse = _candidate([20.0], fitness=1.0)
    candidates = [flat_best, worse]
    cfg = RefinementConfig(plateau_finalist_pool=2, plateau_neighbor_step_frac=0.1)

    def evaluate_cheap(genome):
        # Every neighborhood is equally flat -- nothing should look more
        # "robust" than the genuinely-best point, so no swap should occur.
        return _candidate(genome, fitness=genome[0] / 25.0)

    chosen, report = _select_plateau_robust(candidates, genes, cfg, evaluate_cheap)
    assert chosen.genome == flat_best.genome
    assert report["swapped"] is False


def test_select_plateau_robust_returns_raw_best_with_no_report_when_nothing_finite():
    genes = [_gene(lo=0.0, hi=100.0)]
    candidates = [_candidate([1.0], fitness=float("-inf")), _candidate([2.0], fitness=float("nan"))]
    cfg = RefinementConfig()
    chosen, report = _select_plateau_robust(candidates, genes, cfg, evaluate_cheap=lambda g: _candidate(g, 0.0))
    assert report is None
    assert chosen in candidates


def test_select_plateau_robust_deduplicates_identical_genomes_in_the_finalist_pool():
    genes = [_gene(lo=0.0, hi=100.0)]
    # Same genome appears twice (e.g. carried over by elitism across
    # generations) -- it must only occupy one finalist slot.
    dup_a = _candidate([50.0], fitness=2.0, generation=0)
    dup_b = _candidate([50.0], fitness=2.0, generation=3)
    other = _candidate([10.0], fitness=1.0)
    cfg = RefinementConfig(plateau_finalist_pool=2, plateau_neighbor_step_frac=0.1)
    calls = []

    def evaluate_cheap(genome):
        calls.append(tuple(genome))
        return _candidate(genome, fitness=1.5)

    _select_plateau_robust([dup_a, dup_b, other], genes, cfg, evaluate_cheap)
    # Two finalists (the deduped 50.0 genome + the 10.0 genome) * 2
    # neighbor probes each (one gene, two directions) = 4 calls, not 6.
    assert len(calls) == 4
