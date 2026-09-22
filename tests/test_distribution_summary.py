"""Tests for app.optimize.distribution_summary -- the "distributions, not
just a winner" summary generalized from app.optimize.refinement's original
(refinement-only) implementation so Search Lab and Evolution Lab's own
batch-level results can report the same median-vs-best table."""
from __future__ import annotations

import math

from app.optimize.distribution_summary import compute_distribution_summary


def _record(fitness, eval_pass, payout_pass, max_dd=None):
    return {
        "fitness": fitness,
        "mc_summary": {"evaluation_pass_probability": eval_pass, "first_payout_probability": payout_pass},
        "statistics": {"max_drawdown_pct": max_dd} if max_dd is not None else None,
    }


def test_median_and_best_computed_correctly():
    records = [_record(0.5, 40.0, 20.0, 8.0), _record(0.9, 70.0, 55.0, 4.0), _record(0.3, 20.0, 10.0, 12.0)]
    d = compute_distribution_summary(records, total_tested=100)
    assert d["candidates_tested"] == 100
    assert d["candidates_with_trades"] == 3
    assert d["eval_pass_probability"] == {"median": 40.0, "best": 70.0}
    assert d["first_payout_probability"] == {"median": 20.0, "best": 55.0}
    assert d["max_drawdown_pct"] == {"median": 8.0, "best": 4.0}  # best = lowest drawdown


def test_total_tested_defaults_to_record_count_when_omitted():
    d = compute_distribution_summary([_record(0.5, 40.0, 20.0)])
    assert d["candidates_tested"] == 1


def test_records_without_mc_summary_are_excluded_not_fabricated():
    records = [_record(0.5, 40.0, 20.0), {"fitness": 0.9, "mc_summary": None, "statistics": None}]
    d = compute_distribution_summary(records)
    assert d["candidates_with_trades"] == 1


def test_records_without_max_drawdown_omit_that_key_entirely():
    d = compute_distribution_summary([_record(0.5, 40.0, 20.0, max_dd=None)])
    assert d["max_drawdown_pct"] is None


def test_robust_candidates_uses_80_percent_of_best_threshold():
    # best=1.0 -> threshold=0.8; only fitnesses >= 0.8 count
    records = [_record(1.0, 50, 25), _record(0.85, 50, 25), _record(0.5, 50, 25)]
    d = compute_distribution_summary(records)
    assert d["robust_candidates"] == 2
    assert d["robust_threshold_fraction_of_best"] == 0.8


def test_negative_best_fitness_uses_120_percent_threshold():
    # best=-1.0 (all negative) -> threshold = -1.0 * 1.2 = -1.2, so both -1.0 and -1.1 qualify
    records = [_record(-1.0, 10, 5), _record(-1.1, 10, 5), _record(-5.0, 10, 5)]
    d = compute_distribution_summary(records)
    assert d["robust_threshold_fraction_of_best"] == 1.2
    assert d["robust_candidates"] == 2


def test_empty_input_returns_none():
    assert compute_distribution_summary([]) is None


def test_all_records_missing_mc_summary_returns_none():
    assert compute_distribution_summary([{"fitness": 0.5, "mc_summary": None, "statistics": None}]) is None


def test_non_finite_fitness_excluded():
    records = [_record(0.5, 40, 20), _record(math.inf, 90, 90), _record(math.nan, 90, 90)]
    d = compute_distribution_summary(records)
    assert d["candidates_with_trades"] == 1
