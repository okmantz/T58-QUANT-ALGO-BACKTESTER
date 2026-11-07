"""Rule provenance for every prop-firm preset (2026-11-06 audit).

``presets.py`` already carries a stamp per preset (``verified_on`` / ``source_url`` / ``verification_tier``). What was missing is the
other half of honest provenance: a plain list, per preset, of the rules the simulator does NOT model, and one place that shows the source,
date and an UNVERIFIED flag next to them. Everything here is DERIVED from the preset's own fields and the simulator's known scope, so it
cannot drift from the data: a field that is switched off (``flatten_time_ct is None``, ``max_contracts is None``, ``dd_basis == "legacy"``) shows
up as "not modeled" automatically, and editing a preset updates this view with no second table to maintain.

Verdict gates are untouched: this module only reports (``ready_eligibility`` / ``rules_stamp_state`` still decide what they decided before).
"""
from __future__ import annotations

# Rules no preset can express today, because the simulator has no input for them. A firm may or may not impose them; the list is the
# honest boundary of what a pass/fail from this app proves.
GLOBAL_NOT_MODELED = (
    "News-event trading blackouts / restricted-instrument windows",
    "Maximum lot / position-size limits other than the per-firm contract cap (when one is set)",
    "Hedging, martingale, grid and copy-trading prohibitions (algo behaviour rules are reviewed in algo_policy_note, not simulated)",
    "Weekend / overnight holding restrictions beyond the configured session flatten time",
    "Platform data-feed differences, spread widening at the firm's broker, and rejected / requeued orders",
    "Profit-split changes, fee refunds and promotional discounts (dollar economics use the fee table, not live pricing)",
)


def not_modeled(preset) -> list[str]:
    """Rules this specific preset leaves unmodeled, derived from which of its fields are off."""
    out: list[str] = []
    if getattr(preset, "dd_basis", "legacy") == "legacy":
        out.append("Drawdown basis is 'legacy' (pre-overhaul behaviour): whether the firm measures drawdown on realized, end-of-day or floating equity was never confirmed")
    if getattr(preset, "flatten_time_ct", None) is None:
        out.append("Session flatten deadline: not modeled (positions held past the firm's end-of-day cutoff are not penalised)")
    if getattr(preset, "max_contracts", None) is None:
        out.append("Per-firm contract cap: not modeled")
    if getattr(preset, "max_eval_calendar_days", None) is None:
        out.append("Evaluation time limit: not modeled (no calendar-day deadline on the challenge)")
    if getattr(preset, "consistency_rule_pct", None) is None:
        out.append("Evaluation consistency rule: none modeled (the firm may have one the preset omits)")
    if getattr(preset, "funded_consistency_rule_pct", None) is None:
        out.append("Funded-stage consistency rule: none modeled")
    if getattr(preset, "trailing_lock", False) is False and getattr(preset, "drawdown_type", "") == "trailing":
        out.append("Trailing drawdown never locks: modeled as trailing for the account's whole life")
    if not getattr(preset, "max_inactive_days", 0):
        out.append("Inactivity rule: not modeled")
    if not getattr(preset, "rules_checked_on", ""):
        out.append("Drawdown / lock / contract fields have no 'rules_checked_on' date: they were never re-checked against the firm")
    return out


def provenance(preset, today=None) -> dict:
    """Source, stamp date, age, tier, state and the unmodeled-rules list for one preset."""
    from app.prop.presets import rules_stamp_age_days, rules_stamp_state
    state, why = rules_stamp_state(preset, today)
    return {
        "key": preset.key, "firm": preset.firm, "label": preset.label,
        "source_url": preset.source_url or None, "verified_on": preset.verified_on or None,
        "verification_tier": preset.verification_tier or None, "age_days": rules_stamp_age_days(preset, today),
        "state": state, "unverified": state == "unverified", "why": why or None,
        "not_modeled": not_modeled(preset), "source_note": preset.source_note or None,
    }


def provenance_table(today=None) -> dict:
    from app.prop.presets import list_presets
    rows = [provenance(p, today) for p in list_presets()]
    return {
        "n_presets": len(rows),
        "n_unverified": sum(r["unverified"] for r in rows),
        "n_stale": sum(r["state"] == "stale" for r in rows),
        "global_not_modeled": list(GLOBAL_NOT_MODELED),
        "presets": rows,
    }
