"""30-day prop evaluation evaluator + feasibility screen + the 2026-11-10 audit fixes (units, clocks, censoring)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.prop.feasibility import expectancy_r, frontier, pass_probability_iid, required_frequency
from app.prop.presets import get_preset
from app.prop.thirty_day import (ThirtyDayConfig, TradeLedger, claim_target_achieved, dependence_aware_interval,
                                 evaluate_thirty_day, rank_key)


def _rules():
    return get_preset("lucid_50k").to_prop_rules()


def _ledger(pnls, start="2024-01-02 15:00", every_days=1, hold_h=1.0, mae_extra=0.0):
    """Weekday trades, one per ``every_days`` weekdays, at 15:00 UTC (inside the session day)."""
    days = pd.bdate_range(start, periods=len(pnls) * every_days)[::every_days]
    et = np.array([d.to_datetime64() for d in days], dtype="datetime64[ns]")
    xt = et + np.timedelta64(int(hold_h * 3600), "s")
    p = np.asarray(pnls, float)
    z = np.zeros(len(p))
    return TradeLedger(et, xt, p, np.minimum(p, 0.0) - mae_extra, np.maximum(p, 0.0), z + 5.0, z + 500.0,
                       np.ones(len(p), int), np.array(["s"] * len(p), dtype=object), np.array(["MNQ"] * len(p), dtype=object))


def _span(led, extra=70):
    a, b = led.span()
    return a - pd.Timedelta(days=1), b + pd.Timedelta(days=extra)


def test_empty_ledger_never_claims_success():
    res = evaluate_thirty_day(TradeLedger.empty(), _rules())
    assert res.p_pass_30 == 0.0 and not claim_target_achieved(res)


def test_steady_winner_passes_within_30_and_is_measured_in_calendar_days():
    led = _ledger([400.0] * 120)                       # +$400 a weekday: +$3,000 by the 8th trade, well under 30 calendar days
    s, e = _span(led, 0)
    res = evaluate_thirty_day(led, _rules(), data_start=s, data_end=e, with_sensitivity=False)
    assert res.n_windows > 0
    assert res.p_pass_30 > 0.9
    assert res.median_days_to_pass is not None and res.median_days_to_pass <= 14  # calendar days, not trading-day counts


def test_slow_winner_fails_30_but_passes_60():
    led = _ledger([300.0] * 60, every_days=3)          # one +$300 trade every 3 weekdays: needs 10 trades ~ 6 calendar weeks
    s, e = _span(led, 0)
    res = evaluate_thirty_day(led, _rules(), data_start=s, data_end=e, with_sensitivity=False)
    assert res.p_pass_30 < 0.2
    assert res.p_pass_60 > res.p_pass_30


def test_losing_ledger_has_zero_pass_and_busts():
    led = _ledger([-300.0] * 80)
    s, e = _span(led, 0)
    res = evaluate_thirty_day(led, _rules(), data_start=s, data_end=e, with_sensitivity=False)
    assert res.p_pass_30 == 0.0 and res.p_bust_30 > 0.5 and not claim_target_achieved(res)


def test_cost_stress_lowers_pass_probability():
    pn = [250.0, 250.0, -150.0] * 40
    led = _ledger(pn)
    s, e = _span(led, 0)
    res = evaluate_thirty_day(led, _rules(), data_start=s, data_end=e)
    c = res.cost_sensitivity
    assert c[1.0]["p_pass_30"] >= c[2.0]["p_pass_30"]


def test_few_independent_windows_cannot_meet_the_target_bound():
    led = _ledger([400.0] * 25)
    s, e = _span(led, 35)
    res = evaluate_thirty_day(led, _rules(), data_start=s, data_end=e, with_sensitivity=False)
    assert not res.meets_target_lower_bound                       # a handful of disjoint windows can never certify 65%
    assert res.interval["lower"] < res.p_pass_30 or res.p_pass_30 < 1.0


def test_interval_is_wider_than_naive_wilson_for_overlapping_starts():
    flags = np.array([1.0] * 30 + [0.0] * 30)
    ords = np.arange(60)
    iv = dependence_aware_interval(flags, ords, 30)
    assert iv["n_independent"] < 10 and iv["lower"] < 0.4 and iv["upper"] > 0.6


def test_rank_key_orders_by_lower_bound_not_point_estimate():
    led = _ledger([400.0] * 120)
    s, e = _span(led, 0)
    a = evaluate_thirty_day(led, _rules(), data_start=s, data_end=e, with_sensitivity=False)
    b = evaluate_thirty_day(_ledger([-50.0] * 120), _rules(), data_start=s, data_end=e, with_sensitivity=False)
    assert rank_key(a) < rank_key(b)


def test_ledger_scaling_and_cost_multiplier_are_consistent():
    led = _ledger([100.0, -50.0, 80.0])
    assert np.allclose(led.scaled(2.0).pnl, led.pnl * 2)
    assert np.allclose(led.with_cost_multiplier(2.0).pnl, led.pnl - led.cost)


# ------------------------------------------------------------------------------------------------------- feasibility screen
def test_feasibility_more_trades_help_and_no_edge_never_passes():
    r = _rules()
    lo = pass_probability_iid(trades_per_30d=4, win_prob=0.55, payoff_r=1.5, risk_pct=0.5, rules=r, n_sims=1500)
    hi = pass_probability_iid(trades_per_30d=40, win_prob=0.55, payoff_r=1.5, risk_pct=0.5, rules=r, n_sims=1500)
    assert hi.p_pass_30 > lo.p_pass_30
    none = pass_probability_iid(trades_per_30d=40, win_prob=0.3, payoff_r=1.0, risk_pct=0.5, rules=r, n_sims=1500)
    assert none.p_pass_30 < 0.05
    assert expectancy_r(0.5, 1.0, 0.0) == pytest.approx(0.0)


def test_required_frequency_and_frontier_shapes():
    r = _rules()
    rf = required_frequency(win_prob=0.5, payoff_r=1.5, rules=r, freq_grid=(2, 10, 40), n_sims=800)
    assert len(rf["curve"]) == 3
    rows = frontier(r, expectancies=(0.2,), freq_grid=(5, 40), n_sims=600)
    assert len(rows) == 2 and all(0.0 <= x["p_pass_30"] <= 1.0 for x in rows)
