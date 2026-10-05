"""B2(d) regression tests (v6, 2026-10-05): the Full Pipeline acceptance
verdict gates on PER-ATTEMPT pass odds (Wilson-95% CI lower bound,
fallback: per-attempt point estimate), not the inflated chain-level
number. A strategy with per-attempt lower bound < min_per_attempt_pass_pct
(70.0) is NOT READY. Legacy MC results with no per-attempt information
at all skip the gate (a gate that can't measure can't convict)."""

from dataclasses import dataclass

import pytest

from app.monte_carlo.engine import MonteCarloResult
from app.orchestration.full_pipeline import _make_verdict, FullPipelineConfig


def _mc(**kw):
    base = dict(
        n_simulations=1000,
        evaluation_pass_probability=96.0,  # inflated chain-level number (the trap)
        first_payout_probability=80.0,
        failure_before_payout_probability=5.0,
        multiple_payout_probability=40.0,
        median_days_to_pass=10.0,
        median_days_to_first_payout=20.0,
        average_days_to_first_payout=22.0,
        median_return_pct=10.0,
        mean_return_pct=11.0,
        expected_payout=500.0,
        median_payout=450.0,
        total_simulated_withdrawals=1000.0,
        median_drawdown_pct=3.0,
        p95_drawdown_pct=6.0,
        worst_drawdown_pct=9.0,
        risk_of_ruin_pct=5.0,
        median_max_losing_streak=4.0,
        worst_max_losing_streak=8,
        return_percentiles={25: 8.0, 50: 10.0, 75: 12.0},
    )
    base.update(kw)
    return MonteCarloResult(**base)


def _gate_reason(reasons):
    return [r for r in reasons if "per-attempt eval pass odds too low" in r]


def test_low_per_attempt_ci_lower_bound_rejects():
    # Chain-level 96% but per-attempt CI lower bound only 20% -> NOT READY.
    mc = _mc(per_attempt_pass_ci95=(20.0, 45.0), per_attempt_pass_probability=33.0)
    verdict, reasons, scorecard, hard_fail, lookahead = _make_verdict(mc, None)
    assert verdict == "NOT READY"
    assert _gate_reason(reasons), "expected the B2(d) gate reason"
    assert "20.0%" in _gate_reason(reasons)[0]
    assert hard_fail is False


def test_high_per_attempt_ci_lower_bound_does_not_fire_gate():
    mc = _mc(per_attempt_pass_ci95=(75.0, 92.0), per_attempt_pass_probability=85.0)
    verdict, reasons, *_ = _make_verdict(mc, None)
    assert not _gate_reason(reasons)


def test_point_estimate_fallback_used_when_ci_unknown():
    # CI unknown (0,0) but per-attempt point estimate present -> gates on it.
    mc = _mc(per_attempt_pass_ci95=(0.0, 0.0), per_attempt_pass_probability=40.0)
    verdict, reasons, *_ = _make_verdict(mc, None)
    assert verdict == "NOT READY"
    assert _gate_reason(reasons)


def test_legacy_result_without_per_attempt_fields_skips_gate():
    # A pre-v6 MC-shaped object carries no per-attempt info at all: the
    # gate must not convict (and must not error).
    @dataclass
    class _LegacyMC:
        evaluation_pass_probability: float = 96.0
        first_payout_probability: float = 80.0
        risk_of_ruin_pct: float = 5.0
        return_percentiles: dict = None

        def __post_init__(self):
            if self.return_percentiles is None:
                self.return_percentiles = {25: 8.0, 50: 10.0, 75: 12.0}

    verdict, reasons, *_ = _make_verdict(_LegacyMC(), None)
    assert not _gate_reason(reasons)
    assert verdict != "NOT READY"


def test_gate_respects_advisory_mode():
    mc = _mc(per_attempt_pass_ci95=(20.0, 45.0), per_attempt_pass_probability=33.0)
    verdict, reasons, *_ = _make_verdict(mc, None, gates_advisory_only=True)
    # Advisory mode records the failure but never rejects on it: no HARD
    # gate tag for the per-attempt reason (the verdict itself may still be
    # NOT READY from the scorecard tier -- that is the score, not the gate).
    assert any("ADVISORY ONLY" in r and "per-attempt eval pass odds" in r for r in reasons)
    assert not any("HARD VALIDATION GATE FAILED" in r and "per-attempt eval pass odds" in r for r in reasons)


def test_threshold_configurable_via_config():
    assert FullPipelineConfig().min_per_attempt_pass_pct == 70.0
    assert FullPipelineConfig(min_per_attempt_pass_pct=50.0).min_per_attempt_pass_pct == 50.0
    mc = _mc(per_attempt_pass_ci95=(55.0, 70.0), per_attempt_pass_probability=62.0)
    verdict_default, *_ = _make_verdict(mc, None)
    assert verdict_default == "NOT READY"
    verdict_loose, reasons_loose, *_ = _make_verdict(mc, None, min_per_attempt_pass_pct=50.0)
    assert not _gate_reason(reasons_loose)
