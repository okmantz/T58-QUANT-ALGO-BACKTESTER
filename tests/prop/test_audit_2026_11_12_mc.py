from types import SimpleNamespace

import numpy as np
import pandas as pd

from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.presets import get_preset


def _trades(n=60):
    r = np.random.default_rng(1)
    out = []
    for i in range(n):
        t = pd.Timestamp("2026-01-05 10:00") + pd.Timedelta(days=i)
        out.append(SimpleNamespace(pnl=float(r.choice([150.0, -100.0])), entry_time=t, exit_time=t + pd.Timedelta(minutes=30), size=2.0,
                                   initial_risk=1.0, intended_risk_dollars=100.0, commission=0.0, max_adverse_excursion=None))
    return out


def test_cost_stress_is_per_contract_not_percent_of_pnl():
    rules = get_preset("topstep_50k").to_prop_rules()
    base = run_monte_carlo(_trades(), rules, MonteCarloConfig(n_simulations=200, random_seed=3, slippage_stress_pct=0.0))
    stressed = run_monte_carlo(_trades(), rules, MonteCarloConfig(n_simulations=200, random_seed=3, cost_stress_mult=3.0, cost_stress_symbol="ES"))
    assert "COST STRESS" in stressed.methodology_note
    assert stressed.evaluation_pass_probability <= base.evaluation_pass_probability


def test_calendar_limited_rules_are_disclosed():
    rules = get_preset("apex_50k_eod").to_prop_rules()
    res = run_monte_carlo(_trades(), rules, MonteCarloConfig(n_simulations=100, random_seed=3))
    assert "CALENDAR FIDELITY" in res.methodology_note
