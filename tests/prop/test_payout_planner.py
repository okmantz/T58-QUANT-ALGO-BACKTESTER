"""Payout-EV planner: closed forms checked by hand, then cross-checked against a brute-force simulation of the same model."""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from app.prop import payout_planner as pp


def _inp(**kw):
    base = dict(label="X", p_pass=0.5, q_payout=0.5, challenge_price=100.0, reset_price=80.0, activation_fee=50.0,
                profit_split_pct=90.0, first_payout_gross=2000.0, days_to_pass=10.0, days_pass_to_payout=5.0,
                days_per_failed_attempt=6.0)
    base.update(kw)
    return pp.PlanInputs(**base)


def test_expected_attempts_is_one_over_p_times_q():
    assert pp.expected_attempts(0.5, 0.5) == pytest.approx(4.0)
    assert pp.expected_attempts(1.0, 1.0) == pytest.approx(1.0)
    assert pp.expected_attempts(0.0, 0.7) is None
    assert pp.expected_attempts(0.7, 0.0) is None


def test_attempts_for_confidence_matches_the_geometric_cdf():
    k = pp.attempts_for_confidence(0.5, 0.5, 0.90)         # s = .25 -> 1-(.75)^k >= .9 -> k = 9
    assert k == 9
    assert 1 - 0.75 ** k >= 0.90 > 1 - 0.75 ** (k - 1)
    assert pp.attempts_for_confidence(1.0, 1.0, 0.9) == 1
    assert pp.attempts_for_confidence(0.0, 0.5, 0.9) is None


def test_expected_fee_spend_by_hand():
    # s = .25 -> E[N] = 4; passes before payout = 1/q = 2
    # 100 (first) + 80 * 3 (re-buys) + 50 * 2 (activations) = 440
    assert pp.expected_fee_spend(_inp()) == pytest.approx(440.0)


def test_reset_price_defaults_to_the_challenge_price():
    assert pp.expected_fee_spend(_inp(reset_price=None)) == pytest.approx(100 + 100 * 3 + 100)


def test_unknown_price_gives_no_fee_numbers_instead_of_a_guess():
    plan = pp.plan_option(_inp(challenge_price=None, reset_price=None))
    assert plan.expected_fee_spend is None and plan.budget_for_90pct is None and plan.expected_net_after_fees is None
    assert plan.verdict == "NEEDS A PRICE"


def test_expected_days_by_hand_and_assumption_flag():
    days, assumed = pp.expected_days(_inp())
    assert days == pytest.approx(3 * 6.0 + 15.0) and assumed is False
    days, assumed = pp.expected_days(_inp(days_per_failed_attempt=None))
    assert days == pytest.approx(3 * 10.0 + 15.0) and assumed is True


def test_subscription_firms_are_charged_per_billing_period():
    flat = pp.expected_fee_spend(_inp(activation_fee=0.0))
    sub = pp.expected_fee_spend(_inp(activation_fee=0.0, periods_per_attempt=2.0))
    assert sub == pytest.approx(2.0 * flat)


def test_the_closed_forms_agree_with_a_brute_force_simulation():
    inp = _inp()
    rng = random.Random(7)
    n_runs, spend_total, days_total, attempts_total = 60_000, 0.0, 0.0, 0
    for _ in range(n_runs):
        spend, days, attempts = 0.0, 0.0, 0
        while True:
            attempts += 1
            spend += inp.challenge_price if attempts == 1 else inp.reset_price
            passed = rng.random() < inp.p_pass
            if passed:
                spend += inp.activation_fee
            if passed and rng.random() < inp.q_payout:
                days += inp.days_to_pass + inp.days_pass_to_payout
                break
            days += inp.days_per_failed_attempt
        spend_total += spend
        days_total += days
        attempts_total += attempts
    assert attempts_total / n_runs == pytest.approx(pp.expected_attempts(0.5, 0.5), rel=0.03)
    assert spend_total / n_runs == pytest.approx(pp.expected_fee_spend(inp), rel=0.03)
    assert days_total / n_runs == pytest.approx(pp.expected_days(inp)[0], rel=0.03)


def test_verdicts():
    assert pp.plan_option(_inp(p_pass=0.0)).verdict == "NOT VIABLE"
    assert pp.plan_option(_inp(first_payout_gross=100.0)).verdict == "NOT WORTH THE FEES"      # 90 net < 440 spend
    assert pp.plan_option(_inp(first_payout_gross=None)).verdict == "NEEDS A PAYOUT SIZE"
    assert pp.plan_option(_inp()).verdict == "WORTH PLAYING"
    assert pp.plan_option(_inp(p_pass=0.2, q_payout=0.5, first_payout_gross=50_000.0)).verdict == "LONG SHOT"   # E[N] = 10


def test_net_after_fees_and_roi():
    plan = pp.plan_option(_inp())
    assert plan.expected_first_payout_net == pytest.approx(1800.0)
    assert plan.expected_net_after_fees == pytest.approx(1800.0 - 440.0)
    assert plan.roi_multiple == pytest.approx(1800.0 / 440.0)


# ---- parallel evaluations --------------------------------------------------------------------------------------------
def test_one_lane_equals_the_single_account_plan():
    inp = _inp()
    one = pp.parallel_plan(inp, lanes=1)
    assert one.expected_days_to_first_payout == pytest.approx(pp.expected_days(inp)[0])
    assert one.expected_fee_spend == pytest.approx(pp.expected_fee_spend(inp), rel=1e-9)


def test_two_lanes_are_faster_and_cost_more():
    inp = _inp()
    two = pp.parallel_plan(inp, lanes=2, independence=1.0)
    assert two.expected_days_to_first_payout < pp.expected_days(inp)[0]
    assert two.expected_fee_spend > pp.expected_fee_spend(inp)
    assert two.days_saved > 0 and two.extra_spend > 0


def test_fully_correlated_lanes_save_no_time_but_still_cost_double():
    inp = _inp()
    two = pp.parallel_plan(inp, lanes=2, independence=0.0)
    assert two.expected_days_to_first_payout == pytest.approx(pp.expected_days(inp)[0])
    assert two.expected_fee_spend == pytest.approx(2.0 * pp.expected_fee_spend(inp), rel=1e-9)


def test_parallel_bounds_bracket_the_estimate():
    two = pp.parallel_plan(_inp(), lanes=2, independence=0.5)
    assert two.days_if_independent <= two.expected_days_to_first_payout <= two.days_if_fully_correlated


def test_parallel_matches_a_round_based_simulation_when_independent():
    inp = _inp()
    lanes, rng, n_runs, rounds_total = 3, random.Random(11), 40_000, 0
    s = inp.p_pass * inp.q_payout
    for _ in range(n_runs):
        rounds = 0
        while True:
            rounds += 1
            if any(rng.random() < s for _ in range(lanes)):
                break
        rounds_total += rounds
    plan = pp.parallel_plan(inp, lanes=lanes, independence=1.0)
    assert rounds_total / n_runs == pytest.approx(plan.expected_rounds, rel=0.03)


def test_parallel_with_no_payout_chance_says_so():
    plan = pp.parallel_plan(_inp(p_pass=0.0), lanes=2)
    assert plan.expected_days_to_first_payout is None and "Not enough information" in plan.note


# ---- comparing accounts + Monte Carlo adapter --------------------------------------------------------------------------
def test_compare_options_names_the_best_on_each_axis_and_ignores_unknowns():
    a = _inp(label="Lucid 50k", p_pass=0.6, q_payout=0.7, challenge_price=172.0, reset_price=None, days_to_pass=8.0)
    b = _inp(label="Lucid 100k", p_pass=0.5, q_payout=0.7, challenge_price=None, first_payout_gross=4000.0, days_to_pass=9.0)
    out = pp.compare_options([a, b])
    assert out["cheapest"] == "Lucid 50k"                 # the 100k has no price, so it can never win the fee axis
    assert out["fastest"] == "Lucid 50k"
    assert out["most_likely_per_attempt"] == "Lucid 50k"
    assert out["best_net_after_fees"] == "Lucid 50k"
    assert "Lucid 50k" in out["headline"]


def test_compare_options_with_nothing_priced_asks_for_prices():
    out = pp.compare_options([_inp(challenge_price=None), _inp(label="Y", challenge_price=None)])
    assert out["cheapest"] is None and "Enter the missing prices" in out["headline"]


def test_from_monte_carlo_converts_percentages_and_prefers_an_entered_price():
    mc = SimpleNamespace(per_attempt_pass_probability=60.0, per_attempt_payout_probability=70.0,
                         evaluation_pass_probability=61.0, first_payout_probability=40.0,
                         median_days_to_pass=9.0, median_days_to_first_payout=14.0, median_payout=2500.0)
    fee = SimpleNamespace(challenge_price=172.0, reset_price=None, activation_fee=0.0, profit_split_pct=90.0)
    inp = pp.from_monte_carlo("Lucid 50k", mc, fee=fee, preset_key="lucid_50k")
    assert inp.p_pass == pytest.approx(0.6) and inp.q_payout == pytest.approx(0.7)
    assert inp.days_to_pass == 9.0 and inp.days_pass_to_payout == pytest.approx(5.0)
    assert inp.challenge_price == 172.0 and inp.first_payout_gross == 2500.0
    assert pp.from_monte_carlo("x", mc, fee=fee, challenge_price=250.0).challenge_price == 250.0
    no_price = pp.from_monte_carlo("Lucid 100k", mc, fee=SimpleNamespace(challenge_price=None, reset_price=None, activation_fee=0.0, profit_split_pct=90.0))
    assert no_price.challenge_price is None and any("price" in n for n in no_price.notes)


def test_from_monte_carlo_falls_back_to_chain_level_numbers_for_old_results():
    mc = SimpleNamespace(evaluation_pass_probability=50.0, first_payout_probability=25.0,
                         median_days_to_pass=None, median_days_to_first_payout=None, median_payout=0.0)
    inp = pp.from_monte_carlo("old", mc)
    assert inp.p_pass == pytest.approx(0.5) and inp.q_payout == pytest.approx(0.5)
    assert inp.days_to_pass is None and inp.first_payout_gross is None


def test_attempt_cap_keeps_hopeless_plans_printable():
    n = pp.expected_attempts(1e-9, 1e-9)
    assert n == pp.MAX_ATTEMPTS_CAP and math.isfinite(n)
