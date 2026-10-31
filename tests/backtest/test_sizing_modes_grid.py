"""Standalone proof of every sizing mode on a fixed NQ grid (2026-10-31 audit, section 3, fix 3).

The audit's finding: sub-target position sizes in the Full Pipeline reports are NOT a ``max_contracts`` cap. They are
``sizing_mode="rr_planned"``, which plans ``planned_target_dollars / RR`` of risk per trade. This file pins, for each of the five
modes, the contracts, ``risk_at_stop``, skip reason, stop capping and planned risk on a fixed grid of stops and reward:risk ratios,
and proves the invariants the audit relies on:

* the ``max_contracts`` cap is counted in mini-equivalents (4 mini = 40 micro) and only binds above that,
* ``rr_planned`` risk is the plan (anchor / RR, never above the cap) and shrinks as RR grows,
* cost in R does not depend on the contract count,
* ``RiskConfig.size_for_stop`` risk reconciliation compares against the PLAN, never the cap.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.backtest.risk import (
    SIZING_MODES,
    RiskConfig,
    position_sizing_deviation_message,
    sizing_policy_line,
    sizing_policy_summary,
)
from app.backtest.statistics import compute_risk_reconciliation

# NQ: $20/pt, MNQ $2/pt. Costs set explicitly so the grid does not drift if instrument defaults are re-tuned.
SIDE_COST = 0.5          # spread 0.25 + slippage 0.25 points, per fill side
NQ_COMMISSION = 4.20
MNQ_COMMISSION = 1.42


def _nq(mode: str, **over) -> RiskConfig:
    base = RiskConfig(
        initial_balance=50_000.0, risk_mode="fixed", risk_value=500.0, max_contracts=4, planned_target_dollars=300.0,
        pip_size=1.0, contract_size=20.0, commission_per_trade=0.0, commission_per_contract=NQ_COMMISSION,
        micro_contract_size=2.0, micro_commission_per_contract=MNQ_COMMISSION, spread_pips=0.25, slippage_pips=0.25,
    )
    return replace(base, sizing_mode=mode, **over)


def _size(mode: str, stop: float, rr: float, **over):
    return _nq(mode, **over).size_for_stop(50_000.0, float(stop), tp_distance=stop * rr)


def _worst(units: float, contracts: float, stop: float, commission: float) -> float:
    return units * (stop + SIDE_COST) + commission * contracts


# (stop pts, RR) -> expected (contracts, used_micro, risk_at_stop) per mode; None contracts = skipped
GRID = {
    "skip": {
        (20, 5.0): (1, False, 414.2), (20, 2.5): (1, False, 414.2), (12, 2.0): (1, False, 254.2),
        (25, 1.0): (0, False, 0.0), (80, 2.0): (0, False, 0.0),
    },
    "fit_stop": {
        (20, 5.0): (1, False, 414.2), (20, 2.5): (1, False, 414.2), (12, 2.0): (1, False, 254.2),
        (25, 1.0): (1, False, 500.0), (80, 2.0): (1, False, 500.0),
    },
    "fixed_contracts": {
        (20, 5.0): (1, False, 414.2), (20, 2.5): (1, False, 414.2), (12, 2.0): (1, False, 254.2),
        (25, 1.0): (1, False, 514.2), (80, 2.0): (1, False, 1614.2),
    },
    "micro_fallback": {
        (20, 5.0): (1, False, 414.2), (20, 2.5): (1, False, 414.2), (12, 2.0): (1, False, 254.2),
        (25, 1.0): (9, True, 471.78), (80, 2.0): (3, True, 487.26),
    },
    "rr_planned": {
        (20, 5.0): (1, True, 42.42), (20, 2.5): (2, True, 84.84), (12, 2.0): (5, True, 132.1),
        (25, 1.0): (5, True, 262.1), (80, 2.0): (0, False, 0.0),
    },
}


def test_grid_covers_every_sizing_mode():
    assert set(GRID) == set(SIZING_MODES)


@pytest.mark.parametrize("mode", list(GRID))
def test_contracts_and_risk_at_stop_per_mode(mode):
    for (stop, rr), (contracts, micro, risk) in GRID[mode].items():
        d = _size(mode, stop, rr)
        assert d.contracts == contracts, (mode, stop, rr)
        assert d.used_micro is micro, (mode, stop, rr)
        assert d.risk_at_stop == pytest.approx(risk, abs=0.01), (mode, stop, rr)
        if contracts == 0:
            assert d.units == 0 and d.skip_reason, (mode, stop, rr, "a skip must carry its reason, never a silent zero")


def test_skip_reasons_are_named():
    assert _size("skip", 25, 1.0).skip_reason == "stop_too_wide_for_budget"
    assert _size("rr_planned", 80, 2.0).skip_reason == "stop_exceeds_planned_risk"


def test_fit_stop_shrinks_the_stop_to_the_budget_instead_of_skipping():
    d = _size("fit_stop", 80, 2.0)
    assert d.stop_capped and d.contracts == 1
    assert d.stop_distance < 80
    assert d.risk_at_stop == pytest.approx(500.0, abs=0.01)   # lands exactly on the budget


def test_fixed_contracts_flags_a_stop_that_exceeds_the_budget():
    assert _size("fixed_contracts", 80, 2.0).above_budget is True
    assert _size("fixed_contracts", 12, 2.0).above_budget is False


def test_risk_at_stop_matches_closed_form_for_every_traded_cell():
    for mode, cells in GRID.items():
        for (stop, rr), (contracts, micro, _risk) in cells.items():
            if not contracts:
                continue
            d = _size(mode, stop, rr)
            cs = 2.0 if micro else 20.0
            comm = MNQ_COMMISSION if micro else NQ_COMMISSION
            assert d.risk_at_stop == pytest.approx(_worst(d.units, d.contracts, d.stop_distance, comm), abs=1e-6)
            assert d.units == pytest.approx(d.contracts * cs)


# --- the audit's actual finding -------------------------------------------------------------------------------------

def test_rr_planned_sub_target_size_is_the_plan_not_the_cap():
    """Reproduces the Lucid report: a 20-pt NQ stop at RR 5.17-ish sizes to ONE micro and ~$42 of risk although the cap is $500."""
    d = _size("rr_planned", 20, 5.0)
    assert d.contracts == 1 and d.used_micro
    assert d.risk_at_stop == pytest.approx(42.42, abs=0.01)           # the figure in the Lucid report
    assert d.budget == pytest.approx(500.0)                           # the cap is untouched ...
    assert d.planned_risk == pytest.approx(300.0 / 5.0)               # ... the PLAN is anchor / RR
    assert d.risk_at_stop <= d.planned_risk <= d.budget


def test_rr_planned_plan_shrinks_with_rr_and_never_exceeds_the_cap():
    plans = [_size("rr_planned", 20, rr).planned_risk for rr in (0.25, 0.5, 1.0, 2.0, 2.5, 5.0)]
    assert plans == sorted(plans, reverse=True)
    assert max(plans) <= 500.0 + 1e-9
    assert _size("rr_planned", 20, 0.25).planned_risk == pytest.approx(500.0)   # capped at the budget


def test_only_rr_planned_reports_a_plan():
    for mode in SIZING_MODES:
        d = _size(mode, 20, 2.5)
        assert (d.planned_risk is not None) == (mode == "rr_planned")


def test_max_contracts_counts_mini_equivalents_and_only_binds_above_forty_micros():
    # 1-point stop, huge RR: the plan wants far more than 40 micros, so the cap must bind at exactly 4 mini = 40 micro.
    hi_budget = replace(_nq("micro_fallback"), risk_value=100_000.0, initial_balance=1_000_000.0)
    d = hi_budget.size_for_stop(1_000_000.0, 1.0, tp_distance=2.0)
    assert d.contracts == 4 and not d.used_micro                      # whole minis first, 4 = the cap
    d = _size("rr_planned", 20, 5.0, max_contracts=1)
    assert d.contracts == 1                                           # 1 micro is well under 1 mini = 10 micro
    d = _size("rr_planned", 12, 2.0, max_contracts=None)
    assert d.contracts == 5                                           # removing the cap changes nothing here: it never bound
    assert _size("rr_planned", 12, 2.0, max_contracts=4).contracts == d.contracts


def test_cost_in_r_does_not_depend_on_contract_count():
    """Round-trip cost / (stop x point value) is the same for 1 or 5 contracts; only micro commission per notional differs."""
    for contracts in (1, 5):
        cost = 2 * SIDE_COST * 2.0 * contracts + MNQ_COMMISSION * contracts       # round trip on micros
        risk = 2.0 * contracts * 20.0                                             # stop x $/pt x contracts
        assert cost / risk == pytest.approx((2 * SIDE_COST * 2.0 + MNQ_COMMISSION) / 40.0)
    micro_r = (2 * SIDE_COST * 2.0 + MNQ_COMMISSION) / (2.0 * 20)
    mini_r = (2 * SIDE_COST * 20.0 + NQ_COMMISSION) / (20.0 * 20)
    assert micro_r > mini_r                                            # a micro fallback pays more cost in R at the same stop


# --- reconciliation labels the plan, not the cap ------------------------------------------------------------------------

def _trade(planned, intended, risk_at_stop, *, pnl=-40.0, fallback=False, mode="rr_planned", micro=True) -> Trade:
    t0 = pd.Timestamp("2026-01-05 10:00")
    return Trade(
        entry_time=t0, exit_time=t0 + pd.Timedelta(minutes=5), direction=1, entry_price=22_000.0, exit_price=21_980.0,
        size=2.0, pnl=pnl, pnl_pct=-0.1, exit_reason="stop_loss", commission=1.42, equity_after=49_960.0,
        initial_risk=20.0, intended_risk_dollars=intended, risk_at_stop_dollars=risk_at_stop, sizing_mode=mode, used_micro=micro,
        planned_risk_dollars=planned, used_fallback_stop=fallback,
    )


def test_reconciliation_compares_against_the_plan_under_rr_planned():
    trades = [_trade(planned=60.0, intended=500.0, risk_at_stop=42.42) for _ in range(10)]
    recon = compute_risk_reconciliation(trades)
    assert recon["risk_baseline"] == "planned"
    assert recon["sizing_mode"] == "rr_planned"
    assert recon["avg_planned_risk_dollars"] == pytest.approx(60.0)
    assert recon["avg_intended_risk_dollars"] == pytest.approx(500.0)
    # 42.42 vs a 60.0 plan is a 29% shortfall from whole-contract rounding -- flagged against the PLAN ...
    assert recon["pct_trades_position_capped"] == pytest.approx(100.0)
    msg = position_sizing_deviation_message(recon)
    assert msg and "rr_planned" in msg and "NOT the configured cap" in msg
    assert "max-position-size cap" not in msg                          # the old mislabel is gone


def test_a_trade_on_plan_is_not_flagged_even_though_it_is_far_below_the_cap():
    trades = [_trade(planned=45.0, intended=500.0, risk_at_stop=44.7) for _ in range(10)]
    recon = compute_risk_reconciliation(trades)
    assert recon["pct_trades_position_capped"] == 0.0                   # was 100% before the fix (44.7 << 500)
    assert position_sizing_deviation_message(recon) is None


def test_legacy_trades_without_a_plan_still_reconcile_against_the_configured_target():
    trades = [_trade(planned=None, intended=500.0, risk_at_stop=250.0, mode="skip", micro=False) for _ in range(10)]
    recon = compute_risk_reconciliation(trades)
    assert recon["risk_baseline"] == "configured"
    assert recon["pct_trades_position_capped"] == pytest.approx(100.0)
    assert "BELOW" in position_sizing_deviation_message(recon)


def test_substituted_stop_trades_are_excluded_from_risk_comparisons_and_counted():
    trades = [_trade(planned=45.0, intended=500.0, risk_at_stop=44.7) for _ in range(8)]
    trades += [_trade(planned=45.0, intended=500.0, risk_at_stop=5.0, fallback=True) for _ in range(2)]
    recon = compute_risk_reconciliation(trades)
    assert recon["pct_trades_substituted_stop"] == pytest.approx(20.0)
    assert recon["pct_trades_position_capped"] == 0.0                   # the two bogus-stop trades would have read as capped
    assert recon["avg_actual_stop_risk_dollars"] == pytest.approx(44.7)


def test_sizing_policy_is_summarised_for_report_headers():
    risk = _nq("rr_planned")
    p = sizing_policy_summary(risk, {"pct_trades_micro": 100.0, "pct_trades_substituted_stop": 0.0})
    assert p["sizing_mode"] == "rr_planned" and p["planned_target_dollars"] == 300.0
    assert p["risk_cap_dollars"] == pytest.approx(500.0) and p["max_contracts"] == 4 and p["micro_fallback"] is True
    line = sizing_policy_line(risk, {"pct_trades_micro": 100.0})
    for needle in ("rr_planned", "$500", "anchor=$300", "micro fallback ON", "100% of trades on micros"):
        assert needle in line
    assert sizing_policy_summary(_nq("skip"))["planned_target_dollars"] is None
    assert "micro fallback OFF" in sizing_policy_line(_nq("skip"))
