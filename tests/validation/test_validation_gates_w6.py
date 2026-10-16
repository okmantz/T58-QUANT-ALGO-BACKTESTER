"""Tests for the v5 validation-gate wiring (work item w6-gates).

Covers:
  - Full Pipeline _make_verdict: WF / CPCV / ICIR promoted from diagnostic
    to HARD gates (fail -> NOT READY with the reason recorded), CPCV-as-
    primary NOT TESTED -> NOT READY, and the advisory-only flag.
  - DSR as a reject-capable gate incl. the n_trials fix (count_all_trials
    counts GA inner-loop evaluations, not just leaderboard survivors).
  - PBO as a reject-capable gate.
  - Quick Optimize's READY acceptance gate
    (_evaluate_quick_optimize_ready_gate): per-attempt pass >= threshold,
    no lookahead leak, min OOS trades.
"""
from __future__ import annotations

import pytest

from app.monte_carlo.engine import MonteCarloResult
from app.orchestration.full_pipeline import _make_verdict
from app.orchestration.quick_optimize import _evaluate_quick_optimize_ready_gate
from app.search.robustness import (
    WalkForwardResult,
    count_all_trials,
    deflated_sharpe_gate,
)
from app.validation.cpcv import CPCVResult, PBOResult, pbo_gate
from app.validation.icir import HalfLifeResult, ICIRGateResult, SignificanceResult


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _mc(risk_of_ruin_pct: float = 5.0) -> MonteCarloResult:
    return MonteCarloResult(
        n_simulations=1000,
        evaluation_pass_probability=80.0,
        first_payout_probability=60.0,
        failure_before_payout_probability=10.0,
        multiple_payout_probability=5.0,
        median_days_to_pass=20.0,
        median_days_to_first_payout=40.0,
        average_days_to_first_payout=45.0,
        median_return_pct=10.0,
        mean_return_pct=11.0,
        expected_payout=500.0,
        median_payout=450.0,
        total_simulated_withdrawals=1000.0,
        median_drawdown_pct=3.0,
        p95_drawdown_pct=6.0,
        worst_drawdown_pct=9.0,
        risk_of_ruin_pct=risk_of_ruin_pct,
        median_max_losing_streak=4.0,
        worst_max_losing_streak=8,
        per_attempt_pass_probability=75.0,
        per_attempt_payout_probability=55.0,
        total_independent_attempts=1000,
    )


def _wf(is_stable: bool) -> WalkForwardResult:
    return WalkForwardResult(
        folds=[],
        n_folds=4,
        metric="eval_pass_probability",
        mean_train_metric=60.0,
        mean_test_metric=50.0 if is_stable else 10.0,
        walk_forward_efficiency=0.83 if is_stable else 0.17,
        is_stable=is_stable,
        stability_threshold=0.4,
    )


def _cpcv(is_robust: bool) -> CPCVResult:
    return CPCVResult(
        metric="eval_pass_probability",
        n_groups=6,
        n_test_groups=2,
        n_paths=15,
        paths=[],
        mean_oos_metric=50.0,
        median_oos_metric=50.0,
        std_oos_metric=5.0,
        pct_paths_oos_negative=0.0,
        pct_paths_oos_below_is=20.0 if is_robust else 90.0,
        mean_is_metric=60.0,
        degradation=10.0,
        is_robust=is_robust,
        robustness_threshold=0.5,
    )


def _icir(ok: bool) -> ICIRGateResult:
    return ICIRGateResult(
        in_sample_icir=0.10,
        out_sample_icir=0.08 if ok else 0.01,
        icir_retention_pct=80.0 if ok else 10.0,
        in_sample_half_life=HalfLifeResult(),
        out_sample_half_life=HalfLifeResult(),
        significance=SignificanceResult(significant=ok),
        ok=ok,
        reasons=[] if ok else ["Bonferroni-corrected p-value above alpha"],
    )


def _icir_unmeasurable() -> ICIRGateResult:
    # The gate ran but couldn't compute an ICIR (too few distinct periods
    # with trades) -- UNPROVEN, not a genuine failure.
    return ICIRGateResult(
        in_sample_icir=None,
        out_sample_icir=None,
        icir_retention_pct=None,
        in_sample_half_life=HalfLifeResult(),
        out_sample_half_life=HalfLifeResult(),
        significance=SignificanceResult(significant=False),
        ok=False,
        reasons=["not enough distinct time periods with trades to compute a reliable ICIR"],
    )


def _pbo_result(pbo: float) -> PBOResult:
    return PBOResult(
        n_candidates=8,
        n_groups=6,
        n_test_groups=2,
        n_paths=10,
        metric="eval_pass_probability",
        pbo=pbo,
        logits=[],
        is_best_candidate_per_path=[],
        oos_rank_of_is_best_per_path=[],
        overall_best_candidate_index=0,
        mean_is_by_candidate=[],
        mean_oos_by_candidate=[],
        note="test",
    )


def _hard_gate_reasons(reasons: list[str]) -> list[str]:
    return [r for r in reasons if "HARD VALIDATION GATE FAILED" in r]


# ---------------------------------------------------------------------------
# Full Pipeline: WF / CPCV / ICIR hard gates
# ---------------------------------------------------------------------------

def test_wf_fail_rejects():
    verdict, reasons, _scorecard, _ruin_fail, _lh_fail = _make_verdict(
        _mc(), _wf(is_stable=False), _icir(ok=True),
    )
    assert verdict == "NOT READY"
    fails = _hard_gate_reasons(reasons)
    assert len(fails) == 1
    assert "walk-forward" in fails[0].lower()


def test_wf_pass_does_not_reject():
    verdict, reasons, _scorecard, _ruin_fail, _lh_fail = _make_verdict(
        _mc(), _wf(is_stable=True), _icir(ok=True),
    )
    assert _hard_gate_reasons(reasons) == []
    # A stable WF must never be the thing forcing NOT READY via a gate.
    assert not any("walk-forward" in r.lower() for r in _hard_gate_reasons(reasons))


def test_cpcv_primary_fail_rejects():
    verdict, reasons, _scorecard, _ruin_fail, _lh_fail = _make_verdict(
        _mc(), _wf(is_stable=True), _icir(ok=True),
        cpcv_primary_result=_cpcv(is_robust=False),
        primary_robustness_method="cpcv",
    )
    assert verdict == "NOT READY"
    fails = _hard_gate_reasons(reasons)
    assert len(fails) == 1
    assert "cpcv" in fails[0].lower()


def test_cpcv_primary_not_tested_rejects():
    # CPCV selected as the primary robustness method but produced no
    # result -> NOT TESTED -> rejected, not silently kept.
    verdict, reasons, _scorecard, _ruin_fail, _lh_fail = _make_verdict(
        _mc(), None, _icir(ok=True),
        cpcv_primary_result=None,
        primary_robustness_method="cpcv",
    )
    assert verdict == "NOT READY"
    assert any("NOT TESTED" in r for r in _hard_gate_reasons(reasons))


def test_cpcv_supporting_fail_does_not_reject():
    # CPCV as a supporting diagnostic stays diagnostic -- it must not gate.
    verdict, reasons, _scorecard, _ruin_fail, _lh_fail = _make_verdict(
        _mc(), _wf(is_stable=True), _icir(ok=True),
        cpcv_supporting_result=_cpcv(is_robust=False),
        primary_robustness_method="walk_forward",
    )
    assert _hard_gate_reasons(reasons) == []


def test_icir_fail_rejects():
    verdict, reasons, _scorecard, _ruin_fail, _lh_fail = _make_verdict(
        _mc(), _wf(is_stable=True), _icir(ok=False),
    )
    assert verdict == "NOT READY"
    fails = _hard_gate_reasons(reasons)
    assert len(fails) == 1
    assert "icir" in fails[0].lower()
    # The old "Supporting diagnostic" note must not double-report the same failure.
    assert not any(r.startswith("Supporting diagnostic: did NOT pass the ICIR") for r in reasons)


def test_icir_unmeasurable_does_not_reject():
    # "Couldn't compute" is UNPROVEN, not FAILED -- a gate that can't
    # measure can't convict (same couldn't-run/UNPROVEN distinction as
    # the walk-forward gate).
    verdict, reasons, _scorecard, _ruin_fail, _lh_fail = _make_verdict(
        _mc(), _wf(is_stable=True), _icir_unmeasurable(),
    )
    assert _hard_gate_reasons(reasons) == []
    assert any("did NOT pass the ICIR" in r for r in reasons)  # still on the record


def test_multiple_gate_failures_all_recorded():
    verdict, reasons, _scorecard, _ruin_fail, _lh_fail = _make_verdict(
        _mc(), _wf(is_stable=False), _icir(ok=False),
        cpcv_primary_result=_cpcv(is_robust=False),
        primary_robustness_method="cpcv",
    )
    assert verdict == "NOT READY"
    assert len(_hard_gate_reasons(reasons)) == 3


def test_gates_advisory_only_does_not_reject():
    verdict, reasons, _scorecard, _ruin_fail, _lh_fail = _make_verdict(
        _mc(), _wf(is_stable=False), _icir(ok=False),
        cpcv_primary_result=_cpcv(is_robust=False),
        primary_robustness_method="cpcv",
        gates_advisory_only=True,
    )
    # No hard-gate rejection recorded...
    assert _hard_gate_reasons(reasons) == []
    # ...but every failure is still on the record as ADVISORY (3 gates failed).
    assert sum(1 for r in reasons if "ADVISORY ONLY" in r) == 3


# ---------------------------------------------------------------------------
# DSR gate + n_trials fix
# ---------------------------------------------------------------------------

def test_count_all_trials_includes_ga_inner_loop():
    # The old undercount: n_trials = len(stage1_records) only. The fix:
    # baseline + EVERY genome the GA's inner loop actually backtested.
    assert count_all_trials(baseline_count=1, ga_total_evaluations=0) == 1
    assert count_all_trials(baseline_count=1, ga_total_evaluations=500) == 501
    assert count_all_trials(baseline_count=1, ga_total_evaluations=500,
                            extra_trial_counts=(12, 8)) == 521
    assert count_all_trials() >= 1  # never zero -- a computed Sharpe had >= 1 trial


def test_dsr_gate_rejects_overfit_champion():
    # A modest Sharpe selected from hundreds of trials: after deflation it
    # is indistinguishable from the best-by-chance -> REJECT.
    import numpy as np
    rng = np.random.default_rng(7)
    trial_sharpes = list(rng.normal(0.0, 0.6, 500))
    gate = deflated_sharpe_gate(
        observed_sharpe=0.9,
        trial_sharpes=trial_sharpes,
        n_trials=count_all_trials(baseline_count=1, ga_total_evaluations=499),
        n_trade_returns=200,
        min_probabilistic_sharpe=0.95,
    )
    assert gate.n_trials == 500
    assert not gate.passed
    assert "REJECTED" in gate.reason


def test_dsr_gate_passes_genuine_edge():
    import numpy as np
    rng = np.random.default_rng(7)
    trial_sharpes = list(rng.normal(0.0, 0.4, 60))
    gate = deflated_sharpe_gate(
        observed_sharpe=2.5,
        trial_sharpes=trial_sharpes,
        n_trials=count_all_trials(baseline_count=1, ga_total_evaluations=59),
        n_trade_returns=400,
        min_probabilistic_sharpe=0.95,
    )
    assert gate.passed
    assert "PASSED" in gate.reason


def test_dsr_gate_failure_rejects_in_verdict():
    import numpy as np
    rng = np.random.default_rng(7)
    gate = deflated_sharpe_gate(
        observed_sharpe=0.9,
        trial_sharpes=list(rng.normal(0.0, 0.6, 500)),
        n_trials=500,
        n_trade_returns=200,
    )
    assert not gate.passed
    verdict, reasons, _scorecard, _ruin_fail, _lh_fail = _make_verdict(
        _mc(), _wf(is_stable=True), _icir(ok=True),
        dsr_gate_result=gate,
    )
    assert verdict == "NOT READY"
    assert any("deflated sharpe" in r.lower() for r in _hard_gate_reasons(reasons))


# ---------------------------------------------------------------------------
# PBO gate
# ---------------------------------------------------------------------------

def test_pbo_gate_rejects_noise_selection():
    gate = pbo_gate(_pbo_result(pbo=0.72), max_pbo=0.5)
    assert not gate.passed
    assert "REJECTED" in gate.reason


def test_pbo_gate_passes_signal_selection():
    gate = pbo_gate(_pbo_result(pbo=0.28), max_pbo=0.5)
    assert gate.passed
    assert "PASSED" in gate.reason


def test_pbo_gate_missing_result_is_not_a_rejection():
    gate = pbo_gate(None)
    assert gate.passed  # unmeasurable != failed


def test_pbo_gate_failure_rejects_in_verdict():
    gate = pbo_gate(_pbo_result(pbo=0.8), max_pbo=0.5)
    verdict, reasons, _scorecard, _ruin_fail, _lh_fail = _make_verdict(
        _mc(), _wf(is_stable=True), _icir(ok=True),
        pbo_gate_result=gate,
    )
    assert verdict == "NOT READY"
    assert any("pbo gate" in r.lower() for r in _hard_gate_reasons(reasons))


# ---------------------------------------------------------------------------
# Quick Optimize READY acceptance gate
# ---------------------------------------------------------------------------

def test_ready_gate_passes():
    passed, reasons = _evaluate_quick_optimize_ready_gate(
        per_attempt_pass_probability=75.0,
        lookahead_bug_detected=False,
        oos_trade_count=150,
    )
    assert passed is True
    assert reasons == []


def test_ready_gate_rejects_low_per_attempt_pass():
    passed, reasons = _evaluate_quick_optimize_ready_gate(
        per_attempt_pass_probability=60.0,
        lookahead_bug_detected=False,
        oos_trade_count=150,
    )
    assert passed is False
    assert any("per-attempt" in r for r in reasons)


def test_ready_gate_rejects_lookahead_leak():
    passed, reasons = _evaluate_quick_optimize_ready_gate(
        per_attempt_pass_probability=85.0,
        lookahead_bug_detected=True,
        oos_trade_count=150,
    )
    assert passed is False
    assert any("lookahead" in r for r in reasons)


def test_ready_gate_rejects_thin_sample():
    passed, reasons = _evaluate_quick_optimize_ready_gate(
        per_attempt_pass_probability=85.0,
        lookahead_bug_detected=False,
        oos_trade_count=40,
    )
    assert passed is False
    assert any("trade" in r for r in reasons)


def test_ready_gate_boundary_values_pass():
    # Exactly on the bar counts as clearing it.
    passed, reasons = _evaluate_quick_optimize_ready_gate(
        per_attempt_pass_probability=70.0,
        lookahead_bug_detected=False,
        oos_trade_count=100,
    )
    assert passed is True
    assert reasons == []


def test_ready_gate_custom_thresholds():
    passed, _ = _evaluate_quick_optimize_ready_gate(
        per_attempt_pass_probability=75.0,
        lookahead_bug_detected=False,
        oos_trade_count=150,
        min_per_attempt_pass_pct=80.0,
    )
    assert passed is False
