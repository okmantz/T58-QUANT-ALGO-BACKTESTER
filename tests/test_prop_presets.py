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
