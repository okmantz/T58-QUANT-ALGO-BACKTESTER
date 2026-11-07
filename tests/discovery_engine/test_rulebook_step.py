"""Rulebook v2 G3 inside the discovery engine's mandatory promotion sequence."""
from app.discovery_engine import promotion as pr
from app.discovery_engine import objective as obj


def test_g3_is_a_mandatory_step_right_after_reproduction():
    assert pr.STEP_ORDER[:2] == ("reproduction", "rulebook_g3") and len(pr.STEP_ORDER) == 6


def test_failing_g3_stops_the_sequence_and_skips_the_rest():
    ok = lambda n: (lambda c, ctx: pr.StepResult(n, True))  # noqa: E731
    steps = pr.Steps(**{n: ok(n) for n in pr.STEP_ORDER})
    steps.rulebook_g3 = lambda c, ctx: pr.StepResult("rulebook_g3", False, detail="E_plan +0.012R < +0.05R")
    res = pr.run_promotion_sequence({"spec": {}}, steps)
    assert not res.passed and res.failed_step == "rulebook_g3"
    assert [s.skipped for s in res.steps] == [False, False, True, True, True, True]


def test_missing_g3_implementation_is_a_failure_not_a_pass():
    ok = lambda n: (lambda c, ctx: pr.StepResult(n, True))  # noqa: E731
    steps = pr.Steps(**{n: ok(n) for n in pr.STEP_ORDER if n != "rulebook_g3"})
    assert pr.run_promotion_sequence({"spec": {}}, steps).failed_step == "rulebook_g3"


def test_net_profit_mode_never_applies_the_rulebook_gate(monkeypatch):
    from app.discovery_engine import engine_steps as es
    steps = es.build_steps(dev_df=None, sealed_df=None, risk=None, cfg=pr.PromotionConfig(objective_mode="net_profit"), n_trials=50)
    r = steps.rulebook_g3({"spec": {}}, {})
    assert r.passed and r.skipped and "net-profit" in r.detail


def test_prop_mode_g3_rejects_noise(monkeypatch):
    from types import SimpleNamespace
    import numpy as np, pandas as pd
    from app.discovery_engine import engine_steps as es
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2020-01-02", periods=1500)
    tr = [SimpleNamespace(entry_time=d, exit_time=d + pd.Timedelta(minutes=30), direction=1 if i % 2 else -1,
                          pnl=(2.0 if rng.random() < 1 / 3 else -1.0) * 100.0, initial_risk=1.0, size=100.0, commission=2.0)
          for i, d in enumerate(days)]
    monkeypatch.setattr(es, "evaluate_spec", lambda *a, **k: tr)
    steps = es.build_steps(dev_df=None, sealed_df=None, risk=None, cfg=pr.PromotionConfig(), n_trials=200)
    r = steps.rulebook_g3({"spec": {}}, {})
    assert not r.passed and "E_plan" in r.detail


def test_plan_edge_r_deflates_with_more_looks():
    from types import SimpleNamespace
    import numpy as np, pandas as pd
    rng = np.random.default_rng(2)
    days = pd.bdate_range("2019-01-02", periods=1200)
    tr = [SimpleNamespace(entry_time=d, exit_time=d, direction=1, pnl=(2.0 if rng.random() < 0.42 else -1.0) * 100.0,
                          initial_risk=1.0, size=100.0, commission=0.0) for d in days]
    assert obj.plan_edge_r(tr, 1) > obj.plan_edge_r(tr, 50)
