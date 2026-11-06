"""Problem 6: every entry point resolves the SAME costs for the same instrument (registry ticks, no double commission)."""
from __future__ import annotations

import pandas as pd
import pytest

from app.backtest.risk import RiskConfig, UntouchedDefault, describe_simulation
from app.data.instrument_specs import apply_instrument_spec, get_instrument_spec
from app.search.instrument_risk import resolve_leg_risk, resolve_run_costs

EXPECTED_PTS = {"ES": 1.084, "MES": 1.284, "NQ": 1.210, "MNQ": 1.710}


def _rt_points(r: RiskConfig) -> float:
    return 2 * (r.spread_pips + r.slippage_pips) * r.pip_size + (r.commission_per_trade + r.commission_per_contract) / r.contract_size


@pytest.mark.parametrize("sym", list(EXPECTED_PTS))
def test_all_paths_agree_with_registry(sym):
    base = RiskConfig(pip_size=get_instrument_spec(sym).pip_size)
    df = pd.DataFrame({"close": [1.0]})
    a = apply_instrument_spec(base, sym)
    b, _ = resolve_leg_risk(base, df, sym)
    c, _ = resolve_run_costs(RiskConfig(pip_size=get_instrument_spec(sym).pip_size, contract_size=get_instrument_spec(sym).contract_size), sym)
    for r in (a, b):
        assert _rt_points(r) == pytest.approx(EXPECTED_PTS[sym], abs=1e-3)
    assert c.spread_pips == a.spread_pips and c.slippage_pips == a.slippage_pips
    assert c.commission_per_trade == 0.0 and c.commission_per_contract == a.commission_per_contract


def test_explicit_user_values_are_never_overridden():
    base = RiskConfig(pip_size=1.0, spread_pips=1.0, slippage_pips=0.3, commission_per_trade=3.0)  # typed (plain floats)
    assert not isinstance(base.spread_pips, UntouchedDefault)
    out, _ = resolve_run_costs(base, "ES")
    assert out.spread_pips == 1.0 and out.slippage_pips == 0.3 and out.commission_per_trade == 3.0


def test_marker_survives_replace_and_behaves_like_float():
    from dataclasses import replace
    r = RiskConfig()
    r2 = replace(r, risk_value=2.0)
    assert isinstance(r2.spread_pips, UntouchedDefault) and r2.spread_pips == 1.0 and r2.spread_pips + 1 == 2.0


def test_report_prints_round_turn_cost():
    r = apply_instrument_spec(RiskConfig(pip_size=1.0), "ES")
    assert "Round-turn cost: 1.084 price points" in describe_simulation(r)
