"""Objective mode (prop firm default | net profit): the form hook, the fitness mapping and the new metrics."""
from __future__ import annotations

import math

import pytest
from flask import Flask, jsonify, request

from app.optimize.refinement import (
    DEFAULT_FITNESS, FITNESS_METRICS, NET_PROFIT_FITNESS, compute_fitness, fitness_for_objective_mode,
)
from app.web.objective_mode import apply_objective_mode_to_form, current_objective_mode, install


def _app():
    app = Flask(__name__)
    install(app)

    @app.route("/echo", methods=["POST"])
    def echo():
        return jsonify(fitness_metric=request.form.get("fitness_metric", DEFAULT_FITNESS), mode=current_objective_mode())

    return app


def test_default_fitness_is_the_prop_lifecycle_objective():
    assert DEFAULT_FITNESS == "lifecycle_gradient"
    assert "lifecycle_gradient" in FITNESS_METRICS and "net_profit_adjusted" in FITNESS_METRICS
    assert "pass_gradient" in FITNESS_METRICS          # still selectable by name


def test_form_without_objective_mode_is_untouched():
    r = _app().test_client().post("/echo", data={"fitness_metric": "edge_lb_r"})
    assert r.get_json() == {"fitness_metric": "edge_lb_r", "mode": "prop_lifecycle"}


def test_prop_mode_keeps_the_chosen_metric():
    r = _app().test_client().post("/echo", data={"objective_mode": "prop_lifecycle", "fitness_metric": "edge_lb_r_dd"})
    assert r.get_json() == {"fitness_metric": "edge_lb_r_dd", "mode": "prop_lifecycle"}


@pytest.mark.parametrize("posted,expected", [("lifecycle_gradient", NET_PROFIT_FITNESS), ("eval_pass_probability", NET_PROFIT_FITNESS),
                                             ("", NET_PROFIT_FITNESS), ("net_profit", "net_profit")])
def test_net_profit_mode_replaces_any_prop_metric(posted, expected):
    data = {"objective_mode": "net_profit"}
    if posted:
        data["fitness_metric"] = posted
    r = _app().test_client().post("/echo", data=data)
    assert r.get_json() == {"fitness_metric": expected, "mode": "net_profit"}


def test_net_profit_mode_adds_the_metric_when_the_form_had_none():
    r = _app().test_client().post("/echo", data={"objective_mode": "net_profit"})
    assert r.get_json()["fitness_metric"] == NET_PROFIT_FITNESS


def test_json_posts_are_not_touched():
    app = _app()
    r = app.test_client().post("/echo", json={"objective_mode": "net_profit"})
    assert r.status_code == 200 and r.get_json()["fitness_metric"] == DEFAULT_FITNESS


def test_fitness_for_objective_mode_table():
    assert fitness_for_objective_mode(None) == DEFAULT_FITNESS
    assert fitness_for_objective_mode("prop_lifecycle", "edge_lb_r") == "edge_lb_r"
    assert fitness_for_objective_mode("net_profit", "sharpe_ratio") == NET_PROFIT_FITNESS
    assert fitness_for_objective_mode("net_profit", "net_profit") == "net_profit"
    assert fitness_for_objective_mode("nonsense") == DEFAULT_FITNESS


class _MC:
    def __init__(self, **kw):
        self.n_source_trades = 200
        self.evaluation_pass_probability = 60.0
        self.per_attempt_pass_probability = 60.0
        self.per_attempt_payout_probability = 50.0
        self.per_attempt_paid_probability = 30.0
        self.funded_breach_within_horizon_pct = 0.0
        self.median_calendar_days_to_pass = 30.0
        self.risk_of_ruin_pct = 5.0
        self.p95_drawdown_pct = 4.0
        self.__dict__.update(kw)


STATS = {"sharpe_ratio": 1.2, "profit_factor": 1.5, "net_profit": 5000.0, "total_trades": 200, "max_drawdown_pct": 6.0}


def test_lifecycle_gradient_ranks_paid_probability_not_just_pass_probability():
    a = compute_fitness(STATS, None, _MC(per_attempt_paid_probability=45.0), "lifecycle_gradient")
    b = compute_fitness(STATS, None, _MC(per_attempt_paid_probability=15.0), "lifecycle_gradient")
    assert a > b
    # same pass probability, very different payout odds -> pass_gradient cannot tell them apart, lifecycle_gradient can
    assert compute_fitness(STATS, None, _MC(per_attempt_paid_probability=45.0), "pass_gradient") == \
        compute_fitness(STATS, None, _MC(per_attempt_paid_probability=15.0), "pass_gradient")


def test_lifecycle_gradient_rewards_speed_and_punishes_funded_breaches():
    base = compute_fitness(STATS, None, _MC(), "lifecycle_gradient")
    slow = compute_fitness(STATS, None, _MC(median_calendar_days_to_pass=120.0), "lifecycle_gradient")
    dies = compute_fitness(STATS, None, _MC(funded_breach_within_horizon_pct=60.0), "lifecycle_gradient")
    assert slow < base and dies < base


def test_lifecycle_gradient_is_zero_base_for_a_strategy_that_never_gets_paid():
    f = compute_fitness({**STATS, "net_profit": -100.0}, None,
                        _MC(per_attempt_paid_probability=0.0, per_attempt_payout_probability=0.0, per_attempt_pass_probability=0.0,
                            evaluation_pass_probability=0.0), "lifecycle_gradient")
    assert f <= 0.0


def test_net_profit_adjusted_ignores_prop_fields_and_penalises_drawdown_and_thin_samples():
    bare = type("M", (), {})()                         # no prop / Monte Carlo fields at all
    good = compute_fitness({**STATS}, None, bare, "net_profit_adjusted")
    deep = compute_fitness({**STATS, "max_drawdown_pct": 40.0}, None, bare, "net_profit_adjusted")
    thin = compute_fitness({**STATS, "total_trades": 10}, None, bare, "net_profit_adjusted")
    assert good > deep and good > thin and good > 0
    assert compute_fitness({**STATS, "total_trades": 0}, None, bare, "net_profit_adjusted") == -math.inf


def test_hook_function_never_raises_outside_a_request():
    apply_objective_mode_to_form()          # no request context -> swallowed, not raised
