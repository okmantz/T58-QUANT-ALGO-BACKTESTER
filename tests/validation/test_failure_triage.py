"""Tests for app.search.failure_triage -- aggregating why a batch of
candidates failed instead of only logging per-candidate."""
from __future__ import annotations

from app.search.failure_triage import aggregate_failure_reasons


def test_all_passed_reports_no_failures():
    records = [{"passed_stage1": True, "family": "trend_breakout"} for _ in range(5)]
    summary = aggregate_failure_reasons(records, "Stage 1", "passed_stage1")
    assert summary.total_failed == 0
    assert summary.format_log_lines() == ["  Stage 1: no failures to triage -- all 5 scored candidate(s) passed."]


def test_zero_records_is_not_reported_as_everything_passed():
    """Regression test: a stage that scored NOTHING (every candidate was
    skipped upstream -- e.g. a worker-pool stall/crash) must be reported
    distinctly from a stage that scored candidates and all of them passed.
    Conflating the two previously made a worker crash look identical to
    'this search space has no viable strategy' in the log."""
    summary = aggregate_failure_reasons([], "Stage 1", "passed_stage1")
    assert summary.total_records == 0
    assert summary.total_failed == 0
    lines = summary.format_log_lines()
    assert len(lines) == 1
    assert "0 candidates were actually scored" in lines[0]
    assert "NOT the same as everything passing" in lines[0]
    assert "everything passed" not in lines[0]


def test_zero_trades_is_classified_and_counted():
    records = [
        {"passed_stage1": False, "error": "no trades generated on this data", "family": "fam_a"},
        {"passed_stage1": False, "error": "no trades generated on this data", "family": "fam_a"},
        {"passed_stage1": True, "family": "fam_a"},
    ]
    summary = aggregate_failure_reasons(records, "Stage 1", "passed_stage1")
    assert summary.total_records == 3
    assert summary.total_failed == 2
    assert summary.reason_counts["zero trades generated"] == 2


def test_too_few_trades_uses_the_configured_floor_in_the_message():
    records = [
        {"passed_stage1": False, "statistics": {"total_trades": 4, "net_profit": 10}, "family": "fam_a"},
    ]
    summary = aggregate_failure_reasons(records, "Stage 1", "passed_stage1", min_trades=20)
    reasons = dict(summary.top_reasons())
    assert any("too few trades" in r and "4" in r and "20" in r for r in reasons)


def test_low_profit_factor_is_classified():
    records = [
        {"passed_stage1": False, "statistics": {"total_trades": 50, "profit_factor": 0.8, "net_profit": -5}, "family": "fam_a"},
    ]
    summary = aggregate_failure_reasons(records, "Stage 1", "passed_stage1", min_trades=1, min_profit_factor=1.05)
    reasons = dict(summary.top_reasons())
    assert any("profit factor too low" in r for r in reasons)


def test_lookahead_bug_is_classified_distinctly():
    records = [
        {"passed_stage3_gate": False, "lookahead": {"bug_detected": True}, "family": "fam_a"},
    ]
    summary = aggregate_failure_reasons(records, "Stage 3", "passed_stage3_gate")
    assert summary.reason_counts["lookahead bug detected"] == 1


def test_early_kill_floor_error_is_classified_distinctly():
    records = [
        {"passed_stage3_gate": False, "error": "failed Stage 3 early-kill floor (before Monte Carlo)", "family": "fam_a"},
    ]
    summary = aggregate_failure_reasons(records, "Stage 3", "passed_stage3_gate")
    assert summary.reason_counts["failed Stage 3 early-kill floor (before Monte Carlo)"] == 1


def test_reasons_are_broken_down_by_family():
    records = [
        {"passed_stage1": False, "error": "no trades generated on this data", "family": "fam_a"},
        {"passed_stage1": False, "error": "no trades generated on this data", "family": "fam_b"},
    ]
    summary = aggregate_failure_reasons(records, "Stage 1", "passed_stage1")
    assert "fam_a" in summary.reason_counts_by_family
    assert "fam_b" in summary.reason_counts_by_family


def test_format_log_lines_includes_percentages():
    records = [
        {"passed_stage1": False, "error": "no trades generated on this data", "family": "fam_a"},
        {"passed_stage1": False, "error": "no trades generated on this data", "family": "fam_a"},
    ]
    summary = aggregate_failure_reasons(records, "Stage 1", "passed_stage1")
    lines = summary.format_log_lines()
    assert any("100%" in line for line in lines)
