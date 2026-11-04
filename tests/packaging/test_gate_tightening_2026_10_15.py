"""2026-10-15 audit: tighter / fairer gates, and the plain-English verdict explanations."""
from __future__ import annotations

import pytest

import app.orchestration.full_pipeline as fp
from app.scoring.verdict_explain import explain_verdict


@pytest.fixture()
def ready_core(monkeypatch):
    monkeypatch.setattr(fp, "_make_verdict_core", lambda *a, **k: ("READY", [], object(), False, False))


def _mc(**kw):
    base = {"sample_ok": True, "sample_notes": [], "per_attempt_pass_probability": 75.0,
            "per_attempt_pass_ci95": (70.0, 80.0)}
    base.update(kw)
    return type("MC", (), base)()


def _replay(rate, ci, n_eff=6.0, n=48):
    return type("R", (), {"sample_ok": True, "pass_rate": rate, "n_attempts": n, "pass_ci_honest": ci,
                          "n_effective": n_eff})()


# -- cost stress ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("net,expect_demote", [(1.0, True), (240.0, True), (260.0, False), (1000.0, False)])
def test_cost_stress_needs_a_meaningful_share_of_base_profit(ready_core, net, expect_demote):
    cs = {"multiplier": 2.0, "net_profit": net, "base_net_profit": 1000.0}
    v, reasons, *_ = fp._make_verdict(_mc(), None, cost_stress=cs)
    assert (v == "MARGINAL") is expect_demote
    if expect_demote:
        assert "COST STRESS" in reasons[-1] and "25%" in reasons[-1]


def test_cost_stress_threshold_is_configurable(ready_core):
    cs = {"multiplier": 2.0, "net_profit": 400.0, "base_net_profit": 1000.0}
    assert fp._make_verdict(_mc(), None, cost_stress=cs, cost_stress_min_retained=0.5)[0] == "MARGINAL"
    assert fp._make_verdict(_mc(), None, cost_stress=cs, cost_stress_min_retained=0.25)[0] == "READY"


# -- replay vs Monte Carlo, compared with intervals ---------------------------------------------------------
def test_replay_with_wide_overlapping_interval_no_longer_demotes(ready_core):
    # 30-point gap (the old rule would demote) but the replay's honest interval (30-90%) overlaps MC's 70-80%.
    v, _, *_ = fp._make_verdict(_mc(), None, attempt_replay=_replay(0.45, (0.30, 0.90)))
    assert v == "READY"


def test_replay_with_disjoint_intervals_demotes_even_below_25_points(ready_core):
    v, reasons, *_ = fp._make_verdict(_mc(), None, attempt_replay=_replay(0.55, (0.50, 0.60)))
    assert v == "MARGINAL" and "REPLAY DISAGREES" in reasons[-1] and "no overlap" in reasons[-1]


def test_replay_disjoint_but_tiny_gap_is_not_demoted(ready_core):
    mc = _mc(per_attempt_pass_probability=75.0, per_attempt_pass_ci95=(74.0, 76.0))
    v, _, *_ = fp._make_verdict(mc, None, attempt_replay=_replay(0.70, (0.69, 0.71)))   # 5-point gap
    assert v == "READY"


def test_replay_start_dates_raised():
    assert fp.FullPipelineConfig().attempt_replay_starts == 48


# -- holdout gate ---------------------------------------------------------------------------------------
def _holdout(n, mean, std, net=None):
    net = net if net is not None else mean * n
    return {
        "holdout_statistics": {"total_trades": n, "net_profit": net, "profit_factor": 0.8 if net < 0 else 1.3},
        "in_sample_statistics": {"total_trades": 200},
        "holdout_trade_pnl": {"n": n, "mean": mean, "std": std},
    }


def test_holdout_gate_state_rules():
    g = fp._holdout_gate_state
    assert not g(_holdout(10, -50, 300), 30)["ran"]                        # 10 trades no longer counts as a check
    thin = g(_holdout(35, -10, 300), 30)                                     # lost money, but upper bound >> 0
    assert thin["ran"] and thin["losing"] and not thin["hard_fail"] and thin["inconclusive"]
    clear = g(_holdout(200, -60, 200), 30)                                   # convincingly negative
    assert clear["hard_fail"] and not clear["inconclusive"]
    assert not g(_holdout(200, 40, 200), 30)["losing"]


def test_core_rejects_only_convincing_holdout_losses_and_caps_inconclusive_ones():
    import inspect
    src = inspect.getsource(fp._make_verdict_core)
    assert "_holdout_gate_state" in src and "_ho_losing_inconclusive" in src
    assert fp.FullPipelineConfig().min_holdout_trades_for_gate == 30
    assert inspect.signature(fp._make_verdict_core).parameters["min_holdout_trades_for_gate"].default == 30


# -- conformance ----------------------------------------------------------------------------------------
def test_conformance_is_not_a_scored_component():
    from app.scoring import t58_scorecard as sc
    assert "t58_conformance" not in sc._WEIGHTS and sum(abs(w) for w in sc._WEIGHTS.values()) == 100.0


# -- verdict explanations --------------------------------------------------------------------------------
def test_explain_translates_known_reasons_and_skips_context_lines():
    reasons = [
        "COST STRESS: at 2x spread the same trades net $5 (vs $900 at 1x) -- capped at MARGINAL.",
        "For reference, T58 Score: 88.0/100 -- Strong (10/10 checks available) (not the reason for this verdict).",
        "  pass_probability: 90.0/100 (weight +25)",
        "SAMPLE FLOOR: the evidence is too thin.",
    ]
    e = explain_verdict("MARGINAL", reasons, score=88.0, tier="Strong")
    assert e["title"] == "Why it is MARGINAL" and len(e["points"]) == 2
    assert "doubled" in e["points"][0] and "too few trades" in e["points"][1]
    assert "88/100" in e["summary"]


def test_explain_never_returns_an_empty_popup():
    assert explain_verdict("NOT READY", None, fallback="Failed its last check.")["points"] == ["Failed its last check."]
    assert explain_verdict("READY", [])["points"]
    assert explain_verdict(None, None)["title"] == "No verdict yet"
    weird = explain_verdict("MARGINAL", ["Something brand new happened. And more text."])
    assert weird["points"] == ["Something brand new happened."]


def test_champion_board_rows_carry_the_explanation(tmp_path):
    from app.scoring.champion_board import candidate_dimensions
    dims = candidate_dimensions({"last_champion_check": {"verdict": "NOT READY", "t58_score": 41.0, "t58_tier": "Reject",
                                                           "reasons": ["HARD SAFETY GATE FAILED: Monte Carlo risk of ruin is 40%"]}})
    assert dims["verdict_explain"]["verdict"] == "NOT READY" and dims["verdict_explain"]["points"]


# -- subscription EV ----------------------------------------------------------------------------------------
def test_monthly_subscription_charges_every_expected_month():
    from app.orchestration.ev_verdict import evaluate_ev, evaluation_months
    kw = dict(legacy_verdict="MARGINAL", holdout_pass_rate=0.6, window_payout_rate=0.4, payout_value=3000,
              challenge_price=49, profit_split_pct=90, activation_fee=149, bankroll_ruin_pct=5)
    one = evaluate_ev(challenge_months=1, monthly_subscription=True, **kw)
    three = evaluate_ev(challenge_months=3, monthly_subscription=True, **kw)
    flat = evaluate_ev(challenge_months=3, monthly_subscription=False, **kw)
    assert one.c == 49 and three.c == 147 and flat.c == 49
    assert one.ev - three.ev == pytest.approx(98.0)
    # trading-day input is still converted (x7/5); calendar-day input is not converted again
    assert evaluation_months(None) == 1 and evaluation_months(30) == 2 and evaluation_months(61) == 3
    assert evaluation_months(30, calendar_days=True) == 1 and evaluation_months(31, calendar_days=True) == 2 and evaluation_months(61, calendar_days=True) == 3
    from app.prop.presets import get_fee_info
    assert get_fee_info("topstep_50k").monthly_subscription is True
    assert not get_fee_info("lucid_50k").monthly_subscription


def test_per_attempt_billing_charges_each_attempt_for_its_own_calendar_days():
    from app.orchestration.ev_verdict import (billing_periods, billing_profile, attempt_cost, compute_ev_attempts,
                                              break_even_pass_rate_attempts)
    # 29 days -> 1 period, 30 -> 1, 31 -> 2, 61 -> 3 (the audit's 29/29/61 example: 1 + 1 + 3 = 5 periods, not 3 x median)
    assert [billing_periods(d) for d in (29, 30, 31, 61)] == [1, 1, 2, 3]
    prof = billing_profile([1, 1, 1], [29, 29, 61])
    assert prof["periods_pass"] == pytest.approx(5 / 3) and prof["periods_nonpass"] is None
    assert attempt_cost(prof, 1.0, 49.0) == pytest.approx(49.0 * 5 / 3)
    # fast-fail / slow-pass: failures bust inside a month, passes take 3 periods
    prof = billing_profile([2, 2, 2, 1], [5, 9, 12, 85])
    assert prof["periods_nonpass"] == 1.0 and prof["periods_pass"] == 3.0
    assert attempt_cost(prof, 0.25, 50.0) == pytest.approx(50.0 * (0.25 * 3 + 0.75 * 1))
    ev = compute_ev_attempts(0.25, 0.5, 3000.0, prof, 50.0, activation_fee=100.0)
    assert ev == pytest.approx(0.25 * (0.5 * 3000 - 100) - 50.0 * (0.25 * 3 + 0.75 * 1))
    # the numeric break-even is the exact root of EV(p)
    pstar = break_even_pass_rate_attempts(0.5, 3000.0, prof, 50.0, activation_fee=100.0)
    assert compute_ev_attempts(pstar, 0.5, 3000.0, prof, 50.0, 100.0) == pytest.approx(0.0, abs=1e-6)


def test_evaluate_ev_uses_per_attempt_billing_and_lower_bound():
    from app.orchestration.ev_verdict import billing_profile, evaluate_ev
    prof = billing_profile([2, 2, 2, 1], [5, 9, 12, 85])
    kw = dict(legacy_verdict="MARGINAL", holdout_pass_rate=0.7, window_payout_rate=0.4, payout_value=3000, challenge_price=49,
              profit_split_pct=90, activation_fee=149, bankroll_ruin_pct=5, monthly_subscription=True, billing_profile=prof)
    flat = evaluate_ev(challenge_months=3, **{k: v for k, v in kw.items() if k != "billing_profile"})
    per = evaluate_ev(**kw)
    assert per.billing and per.ev > flat.ev          # fast failures are cheaper than three months on every attempt
    lb = evaluate_ev(holdout_pass_rate_lb=0.01, **kw)  # a tiny lower bound cannot clear break-even even when the point estimate does
    assert lb.p_lb == pytest.approx(0.01) and lb.ev_lb < 0 and lb.label != per.label or lb.ev_lb < 0
