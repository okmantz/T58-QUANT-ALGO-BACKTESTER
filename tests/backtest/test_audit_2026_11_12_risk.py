import pytest
from app.backtest.risk import RiskConfig, with_prop_safety_defaults, build_run_context
from app.prop.presets import PROP_FIRM_PRESETS, get_preset


def _rules(key="topstep_50k"):
    return get_preset(key).to_prop_rules()


def test_zero_fill_lag_is_floored_unless_explicitly_allowed():
    assert RiskConfig(entry_fill_lag_bars=0).entry_fill_lag_bars == 1
    assert RiskConfig(entry_fill_lag_bars=0, allow_same_bar_fill=True).entry_fill_lag_bars == 0


def test_prop_runs_clamp_risk_to_one_percent_and_record_it():
    r = with_prop_safety_defaults(RiskConfig(risk_value=2.5), _rules())
    assert r.risk_value == 1.0 and "clamped" in r.risk_cap_note
    f = with_prop_safety_defaults(RiskConfig(risk_mode="fixed", risk_value=5000.0), _rules())
    assert f.risk_value == pytest.approx(500.0)


def test_prop_runs_never_use_fit_stop():
    for mode in ("skip", "fit_stop"):
        assert with_prop_safety_defaults(RiskConfig(sizing_mode=mode), _rules()).sizing_mode == "rr_planned"
    assert build_run_context(RiskConfig(sizing_mode="skip"), _rules()).sizing_mode == "rr_planned"


def test_payout_cap_dollars_and_apex_basis_reach_prop_rules():
    assert get_preset("lucid_flex_25k").to_prop_rules().payout_cap_dollars == 1000.0
    assert get_preset("lucid_25k").to_prop_rules().payout_cap_dollars == 1000.0
    for k in ("apex_25k_eod", "apex_25k_intraday", "apex_50k_eod", "apex_50k_intraday", "apex_100k_eod", "apex_100k_intraday"):
        assert get_preset(k).to_prop_rules().trailing_distance_basis == "account"
