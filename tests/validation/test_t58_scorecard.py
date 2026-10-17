"""Tests for app.scoring.t58_scorecard."""
from __future__ import annotations

import pytest
from dataclasses import dataclass

from app.scoring.t58_scorecard import (
    T58ScorecardInputs,
    compute_t58_score,
    score_from_results,
    tier_for_score,
)


def test_all_components_maxed_scores_100_and_elite():
    inputs = T58ScorecardInputs(
        pass_probability=100, first_payout_probability=100, risk_of_ruin=100,
        walk_forward_stability=100, monte_carlo_robustness=100, parameter_stability=100,
        expectancy_lower_bound=100, drawdown=100,
    )
    result = compute_t58_score(inputs)
    assert result.score == 100.0
    assert result.tier == "Elite"
    assert result.n_components_used == 8


def test_all_components_floored_scores_0_and_reject():
    inputs = T58ScorecardInputs(
        pass_probability=0, first_payout_probability=0, risk_of_ruin=0,
        walk_forward_stability=0, monte_carlo_robustness=0, parameter_stability=0,
        expectancy=0, drawdown=0,
    )
    result = compute_t58_score(inputs)
    assert result.score == 0.0
    assert result.tier == "Reject"


def test_missing_components_are_renormalized_not_penalized():
    # Only pass_probability and first_payout_probability supplied, both maxed --
    # a partial-but-perfect scorecard should still score 100, not be dragged
    # down by six "missing" components treated as zeros.
    inputs = T58ScorecardInputs(pass_probability=100, first_payout_probability=100)
    result = compute_t58_score(inputs)
    assert result.score == 100.0
    assert result.n_components_used == 2
    assert result.n_components_total == 12   # 2026-10-17: funded survival, replay and expectancy LB joined the scorecard
    assert result.notes


def test_no_components_scores_0_with_a_note():
    result = compute_t58_score(T58ScorecardInputs())
    assert result.score == 0.0
    assert result.tier == "Reject"
    assert result.notes


def test_tier_thresholds_match_masterclass_table():
    assert tier_for_score(92) == "Elite"
    assert tier_for_score(85) == "Elite"
    assert tier_for_score(84.9) == "Strong"
    assert tier_for_score(75) == "Strong"
    assert tier_for_score(74.9) == "Promising"
    assert tier_for_score(65) == "Promising"
    assert tier_for_score(64.9) == "Reject"


def test_values_are_clipped_into_0_100_range():
    inputs = T58ScorecardInputs(pass_probability=150, first_payout_probability=-50)
    result = compute_t58_score(inputs)
    assert result.components["pass_probability"]["value"] == 100.0
    assert result.components["first_payout_probability"]["value"] == 0.0


@dataclass
class _FakeMC:
    evaluation_pass_probability: float = 80.0
    first_payout_probability: float = 60.0
    risk_of_ruin_pct: float = 10.0
    return_percentiles: dict = None

    def __post_init__(self):
        if self.return_percentiles is None:
            self.return_percentiles = {25: 8.0, 50: 10.0, 75: 12.0}


@dataclass
class _FakeWF:
    walk_forward_efficiency: float = 0.9


@dataclass
class _FakeRobustness:
    parameter_robustness_score: float = 70.0


@dataclass
class _FakeStats:
    average_r: float = 0.5
    max_drawdown_pct: float = 4.0


def test_score_from_results_maps_real_objects_end_to_end():
    result = score_from_results(
        mc_result=_FakeMC(), walk_forward_result=_FakeWF(), robustness_result=_FakeRobustness(),
        statistics=_FakeStats(), prop_max_drawdown_pct=10.0,
    )
    # 2026-10-15: Monte Carlo robustness is no longer derived from the (degenerate) median-return ratio,
    # so it is MISSING here (no seed-dispersion measurement) -> 7 components, not 8.
    # 2026-10-17: the old point-estimate "expectancy" component was replaced by the replay-independent lower bound
    # (expectancy_lower_bound), which needs a bootstrap this helper is not given -> also MISSING here -> 6.
    assert result.n_components_used == 6
    assert result.components["monte_carlo_robustness"]["value"] is None
    assert result.components["expectancy_lower_bound"]["value"] is None
    # risk_of_ruin_pct=10 -> inverted score 90; drawdown 4/10 used -> score 60;
    # expectancy 0.5R on the rescaled 0.6R-full-score scale -> 83.3; everything else passed through directly.
    assert result.components["risk_of_ruin"]["value"] == 90.0
    assert result.components["drawdown"]["value"] == 60.0
    assert 0.0 < result.score <= 100.0


def test_score_from_results_handles_all_none_gracefully():
    result = score_from_results()
    assert result.score == 0.0
    assert result.n_components_used == 0


def test_conformance_is_informational_and_never_changes_the_score():
    base = compute_t58_score(T58ScorecardInputs(pass_probability=80, first_payout_probability=60))
    with_conf = compute_t58_score(T58ScorecardInputs(pass_probability=80, first_payout_probability=60,
                                                     t58_conformance=0.0))
    assert with_conf.score == base.score
    assert with_conf.weight_coverage == base.weight_coverage
    assert with_conf.t58_conformance == 0.0 and base.t58_conformance is None
    assert "t58_conformance" not in with_conf.components
