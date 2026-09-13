from app.prop.presets import get_preset, list_firms, list_presets, presets_for_firm
from app.prop.simulator import PropRules


def test_list_presets_nonempty_and_unique_keys():
    presets = list_presets()
    assert len(presets) > 0
    keys = [p.key for p in presets]
    assert len(keys) == len(set(keys))


def test_get_preset_returns_expected():
    p = get_preset("ftmo_100k")
    assert p.firm == "FTMO"
    assert p.account_size == 100_000


def test_get_preset_unknown_key_raises():
    try:
        get_preset("not_a_real_key")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_to_prop_rules_round_trips_fields():
    p = get_preset("apex_50k")
    rules = p.to_prop_rules()
    assert isinstance(rules, PropRules)
    assert rules.account_size == p.account_size
    assert rules.max_drawdown_pct == p.max_drawdown_pct
    assert rules.drawdown_type == p.drawdown_type


def test_list_firms_has_multiple_firms():
    firms = list_firms()
    assert "FTMO" in firms
    assert "Apex Trader Funding" in firms
    assert len(firms) >= 5


def test_presets_for_firm_filters_correctly():
    ftmo_presets = presets_for_firm("FTMO")
    assert len(ftmo_presets) >= 2
    assert all(p.firm == "FTMO" for p in ftmo_presets)


def test_presets_for_firm_case_insensitive():
    assert len(presets_for_firm("ftmo")) == len(presets_for_firm("FTMO"))


def test_multiple_account_sizes_offered_where_expected():
    ftmo_sizes = {p.account_size for p in presets_for_firm("FTMO")}
    assert len(ftmo_sizes) >= 3


def test_lucid_payout_frequency_matches_current_lucidpro_terms():
    """Regression guard: this preset was previously modeled as a 14-day
    payout cycle when Lucid's own live pricing page (LucidPro, 50K Pro
    Funded) advertises 3 days -- see app.prop.presets' source_note."""
    for key in ("lucid_50k", "lucid_100k"):
        p = get_preset(key)
        assert p.payout_frequency_days == 3
        assert p.min_trading_days == 1


def test_apex_has_no_evaluation_stage_minimums_post_4_0():
    """Regression guard: Apex 4.0 (March 2026) removed both the minimum
    trading days and the evaluation-stage consistency rule."""
    for key in ("apex_50k", "apex_100k"):
        p = get_preset(key)
        assert p.min_trading_days == 0
        assert p.consistency_rule_pct is None


def test_every_preset_has_source_note_and_as_of():
    for p in list_presets():
        assert p.as_of, f"{p.key} is missing as_of"
        assert p.source_note, f"{p.key} is missing source_note"
