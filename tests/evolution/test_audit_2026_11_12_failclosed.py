from types import SimpleNamespace

from app.evolution.engine import EvolutionRunner
from app.evolution.prop_fitness import compute_prop_fitness


def _runner(objective, metric):
    r = EvolutionRunner.__new__(EvolutionRunner)
    r.cfg = SimpleNamespace(objective=objective, cpcv_metric=metric)
    return r


def test_cpcv_metric_matches_the_objective():
    assert _runner("edge_lb_r", "mean_r")._effective_cpcv_metric() == "mean_r"
    assert _runner("prop_fitness", "mean_r")._effective_cpcv_metric() == "eval_pass_probability"
    assert _runner("prop_fitness", "profit_factor")._effective_cpcv_metric() == "profit_factor"


def test_flag_unpromotable_accumulates_reasons():
    rec = SimpleNamespace(meta={})
    r = _runner("edge_lb_r", "mean_r")
    r._flag_unpromotable(rec, "PBO computation failed")
    r._flag_unpromotable(rec, "CPCV failed: ValueError")
    assert "PBO" in rec.meta["unpromotable"] and "CPCV" in rec.meta["unpromotable"]


def test_missing_evidence_is_named_not_defaulted_to_half():
    stats = {"total_trades": 80, "max_drawdown_pct": 4.0, "net_profit": 1000.0}
    mc = {"per_attempt_pass_probability": 60.0, "per_attempt_payout_probability": 50.0}
    out = compute_prop_fitness(stats, mc, None, None, [10.0, -5.0, 8.0])
    assert set(out.evidence_missing) == {"robustness", "walk-forward"}
    assert out.robustness == 1.0 and out.penalty_param_sensitivity == 0.0
    assert any("UNSCORED" in n for n in out.notes)
