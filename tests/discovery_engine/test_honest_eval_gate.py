"""honest_eval is a promotion GATE (pass/fail), failures are surfaced, not swallowed."""
import pytest

from app.discovery_engine import engine_steps as es
from app.discovery_engine import promotion as pr


def _td(lb=0.7, p=0.8, n=20.0, proxy=False):
    return {"lower_bound": lb, "p_pass_30": p, "n_independent": n, "p_bust_30": 0.1, "excursions_proxy": proxy}


def test_gate_passes_only_above_the_lower_bound_bar():
    cfg = pr.PromotionConfig()
    assert pr.honest_eval_step(_td(lb=0.70), cfg).passed
    r = pr.honest_eval_step(_td(lb=0.42, p=0.45), cfg)
    assert not r.passed and "lower bound" in r.detail


def test_gate_rejects_missing_errored_proxy_and_thin_evidence():
    cfg = pr.PromotionConfig()
    assert not pr.honest_eval_step(None, cfg).passed
    assert "boom" in pr.honest_eval_step({"error": "boom"}, cfg).detail
    assert not pr.honest_eval_step(_td(proxy=True), cfg).passed
    assert not pr.honest_eval_step(_td(n=3.0), cfg).passed


def test_a_high_ranking_champion_below_the_bar_is_not_promoted():
    ok = lambda n: (lambda c, ctx: pr.StepResult(n, True))  # noqa: E731
    steps = pr.Steps(**{n: ok(n) for n in pr.STEP_ORDER})
    steps.honest_eval = lambda c, ctx: pr.honest_eval_step(_td(lb=0.33, p=0.42), pr.PromotionConfig())
    res = pr.run_promotion_sequence({"spec": {}}, steps)
    assert not res.passed and res.failed_step == "honest_eval"


def test_net_profit_mode_skips_the_gate_and_prop_mode_without_rules_fails_it():
    s = es.build_steps(dev_df=None, sealed_df=None, risk=None, cfg=pr.PromotionConfig(objective_mode="net_profit"))
    assert s.honest_eval({"spec": {}}, {}).skipped
    s = es.build_steps(dev_df=None, sealed_df=None, risk=None, cfg=pr.PromotionConfig())
    r = s.honest_eval({"spec": {}}, {})
    assert not r.passed and "no prop rules" in r.detail


def test_evaluate_spec_failure_is_raised_not_swallowed():
    with pytest.raises(es.SpecEvaluationError):
        es.evaluate_spec({"bogus": True}, None, None, 1.0)
