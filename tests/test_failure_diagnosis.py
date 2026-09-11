"""Tests for app.search.failure_diagnosis -- the per-candidate "Why did
this strategy fail?" engine that backs Forge Strategy's graveyard entries.
"""
from __future__ import annotations

from app.search.failure_diagnosis import diagnose_candidate


def test_rolling_failure_breakdown_drives_primary_and_secondary():
    rolling = {
        "pass_rate_pct": 20.0,
        "first_payout_rate_pct": 5.0,
        "failure_breakdown": {"daily_loss_limit": 12, "max_drawdown": 3, "target_not_reached": 1},
    }
    diag = diagnose_candidate(
        candidate_id="fam-abc123", family="liquidity_sweep_reversal",
        verdict="FAILED prop validation", rolling=rolling,
    )
    assert diag.primary_failure == "daily loss"
    assert diag.secondary_failure == "max drawdown"
    assert "volatility filter" in diag.suggested_mutation


def test_falls_back_to_structural_gates_when_no_rolling_or_stats():
    diag = diagnose_candidate(
        candidate_id="fam-def456", family="trend_breakout",
        verdict="FAILED CPCV/regime gate",
        cpcv={"is_robust": False},
    )
    assert diag.primary_failure == "backtest-overfit vs. peers"


def test_walk_forward_instability_detected():
    diag = diagnose_candidate(
        candidate_id="fam-ghi789", family="mtf_pullback",
        verdict="FAILED prop validation (Stage 3 gate)",
        walk_forward={"is_stable": False},
    )
    assert diag.primary_failure == "walk-forward instability"
    assert "regime-specific" in diag.suggested_mutation


def test_excessive_losing_streak_becomes_secondary_failure():
    rolling = {"failure_breakdown": {"max_drawdown": 5}}
    stats = {"total_trades": 40, "max_losing_streak": 10}  # 25% > 15% threshold
    diag = diagnose_candidate(
        candidate_id="fam-jkl000", family="rsi_extreme_reversion",
        verdict="FAILED prop validation", rolling=rolling, statistics=stats,
    )
    assert diag.primary_failure == "max drawdown"
    assert diag.secondary_failure == "excessive losing streak"
    assert "losing streak" in diag.weakness


def test_strength_identifies_high_target_low_conversion():
    mc_summary = {"evaluation_pass_probability": 70.0, "first_payout_probability": 10.0}
    diag = diagnose_candidate(
        candidate_id="fam-mno111", family="vwap_reversion",
        verdict="FAILED prop validation", mc_summary=mc_summary,
        rolling={"failure_breakdown": {"target_not_reached": 1}},
    )
    assert diag.strength is not None
    assert "target achievement" in diag.strength


def test_related_successful_family_excludes_self():
    family_performance = {"Trend Breakout": 12.0, "Vwap Reversion": 40.0}
    diag = diagnose_candidate(
        candidate_id="fam-pqr222", family="trend_breakout",
        verdict="FAILED prop validation",
        rolling={"failure_breakdown": {"daily_loss_limit": 2}},
        family_performance=family_performance,
    )
    assert diag.related_successful_family == "Vwap Reversion"


def test_no_signal_falls_back_to_generic_reason():
    diag = diagnose_candidate(candidate_id="fam-stu333", family="unknown", verdict="FAILED")
    assert diag.primary_failure == "failed validation thresholds"
    assert diag.related_successful_family is None
