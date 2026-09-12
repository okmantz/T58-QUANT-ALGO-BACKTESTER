"""Tests for app.orchestration.prop_autotune -- the heuristic advisor that
suggests Loop Mode / risk settings from a PropRules object. Deliberately
tests monotonicity and bounds rather than exact output numbers (the
formulas are an intentionally simple, documented heuristic, not a fitted
model -- pinning exact values would make this brittle to a deliberate,
well-reasoned formula tweak later).
"""
from __future__ import annotations

from app.orchestration.prop_autotune import suggest_from_prop_rules
from app.prop.simulator import PropRules


def _rules(**overrides) -> PropRules:
    base = dict(evaluation_profit_target_pct=8.0, max_drawdown_pct=10.0, daily_loss_limit_pct=5.0)
    base.update(overrides)
    return PropRules(**base)


def test_suggestion_is_deterministic():
    rules = _rules()
    a = suggest_from_prop_rules(rules)
    b = suggest_from_prop_rules(rules)
    assert a.to_dict() == b.to_dict()


def test_risk_value_scales_with_daily_loss_limit():
    tight = suggest_from_prop_rules(_rules(daily_loss_limit_pct=2.0))
    loose = suggest_from_prop_rules(_rules(daily_loss_limit_pct=10.0))
    assert tight.risk_value_pct < loose.risk_value_pct


def test_risk_value_is_bounded():
    extreme_tight = suggest_from_prop_rules(_rules(daily_loss_limit_pct=0.1))
    extreme_loose = suggest_from_prop_rules(_rules(daily_loss_limit_pct=100.0))
    assert 0.1 <= extreme_tight.risk_value_pct <= 2.0
    assert 0.1 <= extreme_loose.risk_value_pct <= 2.0


def test_tighter_eval_gets_a_lower_suggested_target_and_more_patience():
    tight = suggest_from_prop_rules(_rules(evaluation_profit_target_pct=15.0, max_drawdown_pct=5.0))
    loose = suggest_from_prop_rules(_rules(evaluation_profit_target_pct=3.0, max_drawdown_pct=25.0))
    assert tight.tightness_label == "tight"
    assert loose.tightness_label == "loose"
    assert tight.target_eval_pass_pct < loose.target_eval_pass_pct
    assert tight.stall_rounds_before_widen >= loose.stall_rounds_before_widen


def test_target_eval_pass_pct_is_bounded():
    extreme_tight = suggest_from_prop_rules(_rules(evaluation_profit_target_pct=100.0, max_drawdown_pct=1.0))
    extreme_loose = suggest_from_prop_rules(_rules(evaluation_profit_target_pct=0.1, max_drawdown_pct=100.0))
    assert 35.0 <= extreme_tight.target_eval_pass_pct <= 75.0
    assert 35.0 <= extreme_loose.target_eval_pass_pct <= 75.0


def test_stall_rounds_is_bounded():
    extreme_tight = suggest_from_prop_rules(_rules(evaluation_profit_target_pct=100.0, max_drawdown_pct=1.0))
    extreme_loose = suggest_from_prop_rules(_rules(evaluation_profit_target_pct=0.1, max_drawdown_pct=100.0))
    assert 1 <= extreme_tight.stall_rounds_before_widen <= 4
    assert 1 <= extreme_loose.stall_rounds_before_widen <= 4


def test_tight_eval_prefers_lower_variance_family_groups():
    tight = suggest_from_prop_rules(_rules(evaluation_profit_target_pct=15.0, max_drawdown_pct=5.0))
    assert set(tight.preferred_family_groups) == {"mean_reversion", "pullback", "vwap"}
    assert len(tight.preferred_families) > 0
    assert "trend_following" not in tight.preferred_family_groups


def test_loose_eval_prefers_trend_following_family_groups():
    loose = suggest_from_prop_rules(_rules(evaluation_profit_target_pct=3.0, max_drawdown_pct=25.0))
    assert set(loose.preferred_family_groups) == {"trend_following", "breakout", "momentum"}
    assert len(loose.preferred_families) > 0


def test_preferred_families_are_all_real_registered_families():
    """Every name in preferred_families must actually be a key in
    app.search.strategy_space.FAMILIES -- otherwise a caller pre-filling
    a starting_family field with one of these names would fail."""
    from app.search.strategy_space import FAMILIES

    for label, rules in [
        ("tight", _rules(evaluation_profit_target_pct=15.0, max_drawdown_pct=5.0)),
        ("moderate", _rules()),
        ("loose", _rules(evaluation_profit_target_pct=3.0, max_drawdown_pct=25.0)),
    ]:
        suggestion = suggest_from_prop_rules(rules)
        for family_name in suggestion.preferred_families:
            assert family_name in FAMILIES, f"{family_name} ({label}) is not a registered family"


def test_rationale_is_a_nonempty_list_of_strings():
    suggestion = suggest_from_prop_rules(_rules())
    assert len(suggestion.rationale) >= 2
    assert all(isinstance(line, str) and line for line in suggestion.rationale)


def test_to_dict_round_trips_every_field():
    suggestion = suggest_from_prop_rules(_rules())
    d = suggestion.to_dict()
    assert d["risk_value_pct"] == suggestion.risk_value_pct
    assert d["target_eval_pass_pct"] == suggestion.target_eval_pass_pct
    assert d["stall_rounds_before_widen"] == suggestion.stall_rounds_before_widen
    assert d["preferred_family_groups"] == suggestion.preferred_family_groups
    assert d["preferred_families"] == suggestion.preferred_families
    assert d["tightness_label"] == suggestion.tightness_label
    assert d["rationale"] == suggestion.rationale
