"""Problem 4: payout-time budget, frontier, and the demote-only READY cap."""
from __future__ import annotations

from types import SimpleNamespace as NS

from app.prop.payout_time import choose_for_budget, frontier_point, pass_within_budget_pct


def _mc(outcomes, days):
    return NS(attempt_outcomes=outcomes, attempt_calendar_days=days, median_calendar_days_to_pass=30)


def test_pass_within_budget_counts_only_passes_inside_the_window():
    mc = _mc([1, 1, 1, 2, 0], [10, 40, 90, 5, 100])
    assert pass_within_budget_pct(mc, 45) == 40.0         # passes at day 10 and 40 only
    assert pass_within_budget_pct(mc, 180) == 60.0
    assert pass_within_budget_pct(_mc([], []), 30) is None and pass_within_budget_pct(mc, 0) is None


def test_frontier_point_reports_pass_bust_and_budget():
    fp = frontier_point(_mc([1, 1, 2, 2, 0], [10, 20, 5, 6, 90]), 30)
    assert fp["pass_pct"] == 40.0 and fp["bust_pct"] == 40.0 and fp["within_budget_pct"] == 40.0


def test_choice_respects_bust_cap_and_prefers_within_budget():
    pts = [
        {"risk_value": 0.25, "within_budget_pct": 30.0, "bust_pct": 5.0},
        {"risk_value": 0.50, "within_budget_pct": 55.0, "bust_pct": 15.0},
        {"risk_value": 1.00, "within_budget_pct": 70.0, "bust_pct": 40.0},      # fastest but busts too often
    ]
    best, _ = choose_for_budget(pts, bust_cap_pct=20.0)
    assert best["risk_value"] == 0.50
    none, why = choose_for_budget(pts, bust_cap_pct=1.0)
    assert none is None and "bust" in why


def test_verdict_cap_only_applies_when_a_budget_is_set():
    from app.orchestration.full_pipeline import _make_verdict_inner
    import inspect
    sig = inspect.signature(_make_verdict_inner)
    assert "payout_time_budget_days" in sig.parameters and sig.parameters["payout_time_budget_days"].default is None
    from app.orchestration.full_pipeline import FullPipelineConfig
    assert FullPipelineConfig().payout_time_budget_days is None      # default behaviour unchanged
