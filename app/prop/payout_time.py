"""Time-to-payout budget (Problem 4, 2026-11).

The Monte Carlo already simulates how many calendar days each purchased attempt takes to pass (``attempt_outcomes`` /
``attempt_calendar_days``), but nothing used it: speed was a four-point score input and the risk sweep maximised a survival score with no
time term. A trader paying monthly for an evaluation cares about "pass within the time I am willing to wait, without busting", so the
budget is the user's number, and the frontier below shows pass / bust / within-budget by risk level so the trade-off is visible instead of
buried in one score.

Risk size stays rule-based by default (half the drawdown limit over eight attempts of room); this module never fits size to the development
slice on its own. It is only consulted when the user sets a budget.
"""
from __future__ import annotations

PASSED, BUSTED, OPEN = 1, 2, 0


def pass_within_budget_pct(mc, budget_days: float) -> float | None:
    """% of simulated purchased attempts that PASSED within ``budget_days`` calendar days. None when the result carries no per-path outcomes."""
    out = list(getattr(mc, "attempt_outcomes", None) or [])
    days = list(getattr(mc, "attempt_calendar_days", None) or [])
    if not out or len(out) != len(days) or budget_days is None or float(budget_days) <= 0:
        return None
    ok = sum(1 for o, d in zip(out, days) if int(o) == PASSED and float(d) <= float(budget_days))
    return 100.0 * ok / len(out)


def frontier_point(mc, budget_days: float | None) -> dict:
    """Pass / bust / within-budget for one Monte Carlo result."""
    out = list(getattr(mc, "attempt_outcomes", None) or [])
    n = len(out)
    passed = 100.0 * sum(1 for o in out if int(o) == PASSED) / n if n else None
    bust = 100.0 * sum(1 for o in out if int(o) == BUSTED) / n if n else None
    return {
        "pass_pct": passed, "bust_pct": bust,
        "within_budget_pct": pass_within_budget_pct(mc, budget_days) if budget_days else None,
        "median_calendar_days_to_pass": getattr(mc, "median_calendar_days_to_pass", None),
        "n_paths": n,
    }


def choose_for_budget(points: list, *, bust_cap_pct: float = 20.0):
    """Best risk level for the budget: among points whose bust rate is within the cap, the highest within-budget pass rate (ties go to the
    LOWER bust rate, then the lower risk). Returns (point_or_None, note). A point with no frontier data is ignored."""
    ok = [p for p in points if p.get("within_budget_pct") is not None and p.get("bust_pct") is not None]
    if not ok:
        return None, "no risk level had time-to-pass data"
    under = [p for p in ok if p["bust_pct"] <= float(bust_cap_pct)]
    if not under:
        lo = min(ok, key=lambda p: p["bust_pct"])
        return None, (f"no risk level keeps the evaluation bust rate under {bust_cap_pct:.0f}% "
                      f"(lowest seen {lo['bust_pct']:.0f}% at risk {lo.get('risk_value')})")
    best = max(under, key=lambda p: (round(p["within_budget_pct"], 6), -p["bust_pct"], -float(p.get("risk_value") or 0.0)))
    return best, ""
