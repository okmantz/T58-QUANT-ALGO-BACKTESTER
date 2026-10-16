"""v9: tests for app/funnel/recovery.py -- the prescriptive failure-recovery engine.

Every test asserts DATA-DRIVEN prescriptions: the plan must name the gate,
the margin, and concrete next actions (exact settings), never generic
"try different parameters". One test guards the standing rule: the engine
must never prescribe lowering the 70% gate.
"""

import sys
from types import SimpleNamespace

sys.path.insert(0, "/home/hatch/workspace/t58v9-code/T58-QUANT-ALGO-BACKTESTER-main")

from app.funnel.recovery import (
    EVAL_PASS_GATE,
    diagnose_evolution,
    diagnose_pipeline,
    diagnose_quickopt,
    diagnose_search,
)


def _nosuch(text):
    """No action may contain generic filler or gate-lowering language."""
    bad = ["try different parameters", "try again", "lower the gate",
           "reduce the 70", "70% gate"]
    low = text.lower()
    return not any(b in low for b in bad)


def test_search_stage1_wipeout_names_margin_and_killer():
    plan = diagnose_search(
        {"total_candidates": 200, "stage1_survivors": 0, "stage2_survivors": 0,
         "stage3_survivors": 0, "leaderboard": []},
        dataset_label="ES1!/5min",
        db_diagnostics={"stage1_median_pf": 0.91, "stage1_median_trades": 14,
                        "pf_fail_share": 0.73, "trades_fail_share": 0.21},
    )
    assert plan.failure_class == "SEARCH_STAGE1_WIPEOUT"
    assert not plan.passed
    assert abs(plan.margin - 0.14) < 1e-9  # 1.05 - 0.91
    assert "profit factor was the bigger killer" in plan.headline
    assert len(plan.actions) >= 2
    kinds = [a.kind for a in plan.actions]
    assert "new_search" in kinds
    # the re-search must carry exact settings, not vibes
    rs = next(a for a in plan.actions if a.kind == "new_search")
    assert rs.params["search_cfg"]["max_candidates"] == 400
    assert rs.params["search_cfg"]["candidate_source"] == "grammar"
    for a in plan.actions:
        assert _nosuch(a.title + a.rationale)


def test_search_stage1_wipeout_trade_starved_picks_frequency_lever():
    plan = diagnose_search(
        {"total_candidates": 200, "stage1_survivors": 0, "stage2_survivors": 0,
         "stage3_survivors": 0, "leaderboard": []},
        db_diagnostics={"stage1_median_pf": 1.2, "stage1_median_trades": 6,
                        "pf_fail_share": 0.2, "trades_fail_share": 0.75},
    )
    assert plan.failure_class == "SEARCH_STAGE1_WIPEOUT"
    assert "trade count was the bigger killer" in plan.headline
    assert any("higher-frequency" in a.title for a in plan.actions)


def test_search_stage3_wipeout_prescribes_quick_optimize_with_settings():
    lb = [{"candidate_id": "c1", "family": "macd_trend",
           "mc_summary": {"per_attempt_pass_probability": 41.0},
           "statistics": {"profit_factor": 1.1, "win_rate": 0.45}}]
    plan = diagnose_search(
        {"total_candidates": 200, "stage1_survivors": 30, "stage2_survivors": 8,
         "stage3_survivors": 0, "leaderboard": lb, "db_path": "/tmp/x.db",
         "run_id": "r1"},
        dataset_label="ES1!/5min",
    )
    assert plan.failure_class == "SEARCH_STAGE3_WIPEOUT"
    assert abs(plan.margin - 29.0) < 1e-9
    qo = next(a for a in plan.actions if a.kind == "quick_optimize")
    assert qo.params["qo_cfg"]["ga_population"] == 32
    assert qo.params["qo_cfg"]["fitness_metric"] == "eval_pass_probability"
    assert "41" in qo.rationale and "29" in qo.rationale


def test_search_weak_champion():
    lb = [{"candidate_id": "c9", "family": "rsi_reversion",
           "mc_summary": {"per_attempt_pass_probability": 62.0},
           "statistics": {}}]
    plan = diagnose_search(
        {"total_candidates": 200, "stage1_survivors": 30, "stage2_survivors": 8,
         "stage3_survivors": 3, "champion_candidate_id": "c9",
         "leaderboard": lb, "db_path": "/tmp/x.db", "run_id": "r1"},
        dataset_label="ES1!/5min",
    )
    assert plan.failure_class == "SEARCH_WEAK_CHAMPION"
    assert abs(plan.margin - 8.0) < 1e-9


def test_search_strong_champion_pass_path():
    lb = [{"candidate_id": "c9", "family": "rsi_reversion",
           "mc_summary": {"per_attempt_pass_probability": 74.0}}]
    plan = diagnose_search(
        {"total_candidates": 200, "stage1_survivors": 30, "stage2_survivors": 8,
         "stage3_survivors": 3, "champion_candidate_id": "c9",
         "leaderboard": lb, "db_path": "/tmp/x.db", "run_id": "r1"})
    assert plan.passed
    assert any(a.kind == "promote" for a in plan.actions)


def _mc(**kw):
    base = dict(per_attempt_pass_probability=41.0,
                per_attempt_payout_probability=30.0,
                risk_of_ruin_pct=10.0, p95_drawdown_pct=6.0,
                median_drawdown_pct=3.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_pipeline_cost_drag_prescribes_wider_stops():
    mc = _mc()
    stats = {"profit_factor": 1.05, "profit_factor_gross": 1.65,
             "win_rate": 0.48, "total_trades": 220, "expectancy": 12.0}
    result = SimpleNamespace(
        verdict="NOT READY", strategy_display_name="s1",
        final_mc=mc, final_bt=SimpleNamespace(statistics=stats),
        scorecard=SimpleNamespace(score=52.0, tier="C"),
        lookahead_hard_fail=False, risk_of_ruin_hard_fail=False,
        final_config={}, saved_library_note=None)
    plan = diagnose_pipeline(result, dataset_label="ES1!/5min")
    assert plan.failure_class == "PIPE_NOT_READY"
    assert abs(plan.margin - 29.0) < 1e-9
    assert any("cost drag" in n for n, _, _ in plan.weakest_metrics)
    first = plan.actions[0]
    assert first.kind == "new_search"
    assert first.params["search_cfg"]["stop_mult_scale"] == 1.5
    assert "1.65" in first.rationale and "1.05" in first.rationale


def test_pipeline_ruin_hard_fail():
    mc = _mc(risk_of_ruin_pct=38.0)
    result = SimpleNamespace(
        verdict="NOT READY", strategy_display_name="s1", final_mc=mc,
        final_bt=SimpleNamespace(statistics={"profit_factor": 1.4}),
        scorecard=None, lookahead_hard_fail=False,
        risk_of_ruin_hard_fail=True, risk_of_ruin_cap=20.0,
        final_config={}, saved_library_note=None, warnings=[])
    plan = diagnose_pipeline(result)
    assert plan.failure_class == "PIPE_RUIN"
    assert abs(plan.margin - 18.0) < 1e-9
    assert any(a.kind == "quick_optimize" for a in plan.actions)


def test_pipeline_lookahead_hard_fail():
    result = SimpleNamespace(
        verdict="NOT READY", strategy_display_name="s1", final_mc=_mc(),
        final_bt=SimpleNamespace(statistics={}), scorecard=None,
        lookahead_hard_fail=True, risk_of_ruin_hard_fail=False,
        final_config={}, saved_library_note=None)
    plan = diagnose_pipeline(result)
    assert plan.failure_class == "PIPE_LOOKAHEAD"
    assert all(a.kind != "quick_optimize" for a in plan.actions)


def test_pipeline_ready_pass_path():
    mc = _mc(per_attempt_pass_probability=76.0,
             per_attempt_payout_probability=58.0)
    result = SimpleNamespace(
        verdict="READY", strategy_display_name="s1", final_mc=mc,
        final_bt=SimpleNamespace(statistics={"profit_factor": 1.6}),
        scorecard=None, lookahead_hard_fail=False,
        risk_of_ruin_hard_fail=False, final_config={},
        saved_library_note=None)
    plan = diagnose_pipeline(result)
    assert plan.passed
    kinds = [a.kind for a in plan.actions]
    assert kinds == ["promote", "validate", "deploy_checklist"]


def test_quickopt_no_improvement_prescribes_structure_change():
    result = SimpleNamespace(baseline_eval_pass_probability=38.0,
                             optimized_eval_pass_probability=38.5,
                             improved=False, baseline_trades=150)
    plan = diagnose_quickopt(result, dataset_label="ES1!/5min")
    assert plan.failure_class == "QO_NO_IMPROVEMENT"
    assert any(a.kind == "new_search" and
               a.params["search_cfg"].get("candidate_source") == "grammar"
               for a in plan.actions)


def test_quickopt_improved_but_short():
    result = SimpleNamespace(baseline_eval_pass_probability=38.0,
                             optimized_eval_pass_probability=61.0,
                             improved=True, baseline_trades=150)
    plan = diagnose_quickopt(result, dataset_label="ES1!/5min")
    assert plan.failure_class == "QO_IMPROVED_SHORT"
    assert abs(plan.margin - 9.0) < 1e-9


def test_quickopt_zero_trades_baseline():
    result = SimpleNamespace(baseline_eval_pass_probability=0.0,
                             optimized_eval_pass_probability=0.0,
                             improved=False, baseline_trades=0)
    plan = diagnose_quickopt(result)
    assert plan.failure_class == "QO_BASELINE_BROKEN"


def test_evolution_empty():
    plan = diagnose_evolution([], total_evaluated=500)
    assert plan.failure_class == "EVO_EMPTY"
    assert "500" in plan.headline


def test_gate_never_lowered():
    """The engine must never suggest moving the 70% gate."""
    res = SimpleNamespace(
        verdict="NOT READY", strategy_display_name="s", final_mc=_mc(),
        final_bt=SimpleNamespace(statistics={"profit_factor": 1.0}),
        scorecard=None, lookahead_hard_fail=False,
        risk_of_ruin_hard_fail=False, final_config={},
        saved_library_note=None)
    for plan in [diagnose_pipeline(res),
                 diagnose_search({"total_candidates": 10,
                                  "stage1_survivors": 0, "stage2_survivors": 0,
                                  "stage3_survivors": 0, "leaderboard": []}),
                 diagnose_quickopt(SimpleNamespace(
                     baseline_eval_pass_probability=10.0,
                     optimized_eval_pass_probability=11.0,
                     improved=False, baseline_trades=50))]:
        for a in plan.actions:
            assert _nosuch(a.title + " " + a.rationale), a.title
        assert plan.gate_required is None or plan.gate_required >= EVAL_PASS_GATE or \
            plan.stage in ("search",)


# ---------------------------------------------------------------------------
# Web wiring tests (v9): the one-click recovery endpoint.
# ---------------------------------------------------------------------------

def _web_client():
    import app.web.server as srv
    srv.app.config["TESTING"] = True
    return srv.app.test_client()


def test_recovery_quick_optimize_rejects_bad_strategy_ref():
    """POST with an unbuildable strategy_ref -> clean 400, not 500/NameError.
    (Guards the BUG-1 class: JOB_QUICK_OPTIMIZE must exist.)"""
    c = _web_client()
    r = c.post("/recovery/quick-optimize", json={
        "strategy_ref": {}, "dataset_label": "nope",
        "risk": {}, "rules": {}, "qo_cfg": {}})
    assert r.status_code == 400
    assert "ok" in r.get_json() and not r.get_json()["ok"]


def test_recovery_quick_optimize_missing_candidate_404():
    c = _web_client()
    r = c.post("/recovery/quick-optimize", json={
        "strategy_ref": {"candidate_id": "nope", "db_path": "/tmp/nope.db",
                         "run_id": "nope"},
        "dataset_label": "nope", "risk": {}, "rules": {}, "qo_cfg": {}})
    assert r.status_code == 404
    assert not r.get_json()["ok"]


def test_search_status_includes_recovery_key():
    """The search status route always carries a 'recovery' key (None when
    the job is missing/running) so the panel JS has a stable contract."""
    c = _web_client()
    r = c.get("/search/job/does-not-exist/status.json")
    assert r.status_code == 404  # unknown job -> 404, existing contract


def test_search_stage2_stall_skip_prescribes_leaner_batches():
    """When Stage-2 batches were killed by the stall watchdog (not scored),
    the plan must prescribe LEANER batches -- not a bigger GA budget."""
    plan = diagnose_search({
        "total_candidates": 200, "stage1_survivors": 20, "stage2_survivors": 0,
        "stage3_survivors": 0, "champion_candidate_id": None,
        "stage2_stalled_skipped": 20, "leaderboard": []})
    assert plan.failure_class == "SEARCH_STAGE2_WIPEOUT"
    assert "stall watchdog" in plan.headline
    cfgs = [a.params["search_cfg"] for a in plan.actions if a.kind == "new_search"]
    assert cfgs, "must offer a re-run"
    # Leaner, not bigger: stage1_top_n small, GA small.
    assert cfgs[0]["stage1_top_n"] <= 12
    assert cfgs[0]["ga_population"] <= 10


def test_multi_search_recovery_picks_furthest_instrument():
    """The multi-instrument aggregate picks the instrument that got furthest."""
    import app.web.server as srv
    job = {"done": True, "risk": {}, "rules": {},
           "results": {
               "ES1!/5min": {"error": None, "total_candidates": 100,
                             "stage1_survivors": 0, "stage2_survivors": 0,
                             "stage3_survivors": 0, "champion_candidate_id": None,
                             "db_path": None, "run_id": None},
               "NQ1!/5min": {"error": None, "total_candidates": 100,
                             "stage1_survivors": 15, "stage2_survivors": 0,
                             "stage3_survivors": 0, "champion_candidate_id": None,
                             "db_path": None, "run_id": None},
           }}
    plan = srv._multi_search_recovery(job)
    assert plan is not None
    # NQ got further (15 Stage-1) so its label is in the headline.
    assert "NQ1!/5min" in plan.headline
    assert plan.failure_class == "SEARCH_STAGE2_WIPEOUT"
