"""Tests for the GA-searches-what-it's-graded-on upgrade: compute_fitness's
risk_of_ruin_cap penalty (app.optimize.refinement) and its wiring through
app.optimize.walkforward_ga."""
from __future__ import annotations

import math
from dataclasses import replace

from app.monte_carlo.engine import MonteCarloResult
from app.optimize.refinement import RefinementConfig, _apply_ruin_penalty, compute_fitness


def _mc(risk_of_ruin_pct: float, eval_pass=60.0, first_payout=40.0) -> MonteCarloResult:
    return MonteCarloResult(
        n_simulations=1000,
        evaluation_pass_probability=eval_pass,
        first_payout_probability=first_payout,
        failure_before_payout_probability=0.0,
        multiple_payout_probability=0.0,
        median_days_to_pass=None,
        median_days_to_first_payout=None,
        average_days_to_first_payout=None,
        median_return_pct=0.0,
        mean_return_pct=0.0,
        expected_payout=1000.0,
        median_payout=1000.0,
        total_simulated_withdrawals=0.0,
        median_drawdown_pct=5.0,
        p95_drawdown_pct=8.0,
        worst_drawdown_pct=10.0,
        risk_of_ruin_pct=risk_of_ruin_pct,
        median_max_losing_streak=3.0,
        worst_max_losing_streak=5,
    )


def test_no_cap_means_no_change_to_eval_pass_probability():
    mc = _mc(risk_of_ruin_pct=40.0)
    with_no_cap = compute_fitness({}, None, mc, "eval_pass_probability")
    assert with_no_cap == mc.evaluation_pass_probability


def test_cap_erodes_fitness_when_ruin_present():
    mc = _mc(risk_of_ruin_pct=10.0)
    baseline = compute_fitness({}, None, mc, "eval_pass_probability")
    penalized = compute_fitness({}, None, mc, "eval_pass_probability", risk_of_ruin_cap=20.0)
    assert penalized < baseline
    assert penalized > 0  # gentle below the cap, not zeroed out


def test_penalty_is_much_steeper_once_ruin_exceeds_the_cap():
    cap = 20.0
    at_cap = compute_fitness({}, None, _mc(risk_of_ruin_pct=cap), "eval_pass_probability", risk_of_ruin_cap=cap)
    double_cap = compute_fitness({}, None, _mc(risk_of_ruin_pct=cap * 2), "eval_pass_probability", risk_of_ruin_cap=cap)
    baseline = 60.0
    erosion_at_cap = baseline - at_cap
    erosion_double_cap = baseline - double_cap
    assert erosion_double_cap > erosion_at_cap
    # Above the cap the search should already look near-hopeless, well
    # before the actual hard veto -- the whole point is not wasting the
    # search budget converging on a doomed candidate.
    assert double_cap < baseline * 0.3


def test_ruin_aware_metrics_are_never_double_penalized():
    mc = _mc(risk_of_ruin_pct=50.0)
    for metric in ("composite_prop_score", "prop_guide_score"):
        unpenalized = compute_fitness({"total_trades": 100, "profit_factor": 1.4, "win_rate": 55.0}, None, mc, metric)
        penalized = compute_fitness(
            {"total_trades": 100, "profit_factor": 1.4, "win_rate": 55.0}, None, mc, metric, risk_of_ruin_cap=20.0,
        )
        assert unpenalized == penalized


def test_zero_or_negative_fitness_is_left_alone():
    mc = _mc(risk_of_ruin_pct=90.0)
    assert _apply_ruin_penalty(0.0, mc, 20.0, "eval_pass_probability") == 0.0
    assert _apply_ruin_penalty(-5.0, mc, 20.0, "eval_pass_probability") == -5.0


def test_zero_ruin_is_unpenalized():
    mc = _mc(risk_of_ruin_pct=0.0)
    baseline = compute_fitness({}, None, mc, "eval_pass_probability")
    penalized = compute_fitness({}, None, mc, "eval_pass_probability", risk_of_ruin_cap=20.0)
    assert baseline == penalized


def test_refinement_config_defaults_keep_old_behavior():
    cfg = RefinementConfig()
    assert cfg.risk_of_ruin_cap is None
    assert cfg.auto_shrink_on_low_trades is True
    assert cfg.min_oos_trades_per_candidate == 30
