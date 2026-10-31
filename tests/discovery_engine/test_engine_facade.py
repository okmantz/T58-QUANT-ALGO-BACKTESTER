"""The ONE Strategy Discovery Engine: presets, family guard, orchestration, surrogate gate, automatic retests, status page."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.discovery_engine import promotion as pr
from app.discovery_engine import retest as rt
from app.discovery_engine import surrogate as sg
from app.discovery_engine.engine import BackendOutput, DiscoveryEngine, EngineRequest, resolve_families
from app.discovery_engine.presets import PRESETS, STAGE3_PROP_RELAXED, apply_overrides, get_preset
from app.validation import trial_ledger as tl


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_PATH", tmp_path / "reports" / "trial_ledger.json")


def _risk():
    return RiskConfig(spread_pips=0.0, slippage_pips=0.0, commission_per_trade=0.0)


# ------------------------------------------------------------------------------------------------------------------ presets
def test_the_five_old_labs_are_presets_of_one_engine():
    assert {"search_lab", "evolution_lab", "forge", "speed_run", "discovery"} <= set(PRESETS)
    assert {p.backend for p in PRESETS.values()} == {"search", "evolution", "forge", "speed_run", "discovery"}
    assert PRESETS["evolution_loop"].loop and not PRESETS["search_lab"].loop


def test_get_preset_accepts_aliases_and_rejects_unknown_names():
    assert get_preset("Search Lab").name == "search_lab" and get_preset("speedrun").name == "speed_run"
    with pytest.raises(KeyError):
        get_preset("nope")


def test_apply_overrides_only_touches_fields_that_exist_and_relaxes_prop_floors():
    @dataclass(frozen=True)
    class Cfg:
        min_eval_pass_probability: float = 0.5
        stage3_min_profit_factor: float = 1.2
        workers: int = 4

    out = apply_overrides(Cfg(), get_preset("search_lab"), {"workers": 1, "not_a_field": 9})
    assert out.workers == 1 and out.min_eval_pass_probability == 0.0 and out.stage3_min_profit_factor == 0.0
    assert not hasattr(out, "not_a_field")
    assert set(STAGE3_PROP_RELAXED) >= {"min_eval_pass_probability"}


# ------------------------------------------------------------------------------------------------------------ family guard
def test_resolve_families_gates_unregistered_v13_families_and_unknown_names(tmp_path):
    from app.discovery.hypothesis import HypothesisStore, preregister_v13_families
    store = HypothesisStore(tmp_path / "hyp.json")
    preset = get_preset("search_lab")
    ran, gated = resolve_families(["trend_breakout", "no_such_family"], preset, store=store)
    assert gated["no_such_family"] == "unknown family"
    # a v13 family with no kill condition on record does not run, ever
    ran, gated = resolve_families(["weekly_range_breakout"], preset, store=store)
    assert ran == [] and "not pre-registered" in gated["weekly_range_breakout"]
    preregister_v13_families(store)
    ran, gated = resolve_families(["weekly_range_breakout"], preset, store=store)
    assert ran == ["weekly_range_breakout"] and not gated


# ----------------------------------------------------------------------------------------------------- orchestration (fake)
def _fake_steps(passes=True):
    def make(**kw):
        steps = {n: (lambda c, ctx, n=n: pr.StepResult(n, passes, detail="fake")) for n in pr.STEP_ORDER}
        return pr.Steps(**steps)
    return make


def _fake_backend(specs, evaluations=500):
    def backend(req, preset, families, df):
        return BackendOutput(native="x", champions=[{"spec": s, "family": f, "trades": t} for s, f, t in specs], evaluations=evaluations)
    return backend


def _trades(n, mean_r, sd=1.0, seed=0):
    from app.backtest.execution import Trade
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2022-01-03 10:00")
    out = []
    for i in range(n):
        r = float(mean_r + sd * rng.standard_normal())
        out.append(Trade(entry_time=t0 + pd.Timedelta(days=i), exit_time=t0 + pd.Timedelta(days=i, minutes=30), direction=1, entry_price=100.0,
                         exit_price=100.0 + r, size=1.0, pnl=r, pnl_pct=r / 100.0, exit_reason="take_profit" if r > 0 else "stop_loss",
                         commission=0.0, equity_after=100000.0 + r, initial_risk=1.0))
    return out


def _small_df():
    from app.discovery_engine.surrogate import make_null_frame
    return make_null_frame(days=30, seed=1)


def test_engine_ranks_by_edge_lb_r_applies_the_floor_and_carves_the_sealed_slice(tmp_path):
    good, weak, thin = _trades(300, 0.35, seed=1), _trades(300, 0.05, seed=2), _trades(20, 0.5, seed=3)
    specs = [({"config": {"id": "weak"}}, "trend_breakout", weak), ({"config": {"id": "good"}}, "trend_breakout", good),
             ({"config": {"id": "thin"}}, "trend_breakout", thin)]
    seen = {}

    def steps(**kw):
        seen.update(sealed=len(kw["sealed_df"]), dev=len(kw["dev_df"]), n_trials=kw["n_trials"])
        return _fake_steps(True)(**kw)

    df = _small_df()
    eng = DiscoveryEngine(backends={"search": _fake_backend(specs)}, steps_factory=steps)
    res = eng.run(EngineRequest(preset="search_lab", df=df, risk=_risk(), families=["trend_breakout"], work_dir=tmp_path, register_retests=False))
    ids = [c["spec"]["config"]["id"] for c in res.champions]
    assert ids[0] == "good" and "thin" not in ids                      # ranked by edge_lb_r; under the trade floor cannot be promoted
    assert res.objective == "edge_lb_r" and res.n_evaluations == 500
    assert seen["sealed"] == int(len(df) * 0.10) and seen["dev"] + seen["sealed"] == len(df)
    assert seen["n_trials"] >= 1 and res.trial_inputs
    assert all(c["provenance"] and c["provenance"].get("seed") == 42 for c in res.champions)
    assert res.passed and len(res.forward_test_candidates) == len(res.passed)
    json.dumps(res.to_dict(), default=str)


def test_a_failed_mandatory_step_means_no_forward_test_candidate(tmp_path):
    eng = DiscoveryEngine(backends={"search": _fake_backend([({"config": {"id": "a"}}, "trend_breakout", _trades(300, 0.4))])},
                          steps_factory=_fake_steps(False))
    res = eng.run(EngineRequest(preset="search_lab", df=_small_df(), risk=_risk(), families=["trend_breakout"], register_retests=False))
    assert not res.forward_test_candidates and res.promotions[0]["failed_step"] == "reproduction"


def test_loop_mode_stop_candidate_must_pass_the_whole_sequence(tmp_path):
    specs = [({"config": {"id": str(i)}}, "trend_breakout", _trades(300, 0.4, seed=i)) for i in range(3)]
    eng = DiscoveryEngine(backends={"evolution": _fake_backend(specs)}, steps_factory=_fake_steps(False))
    res = eng.run(EngineRequest(preset="evolution_loop", df=_small_df(), risk=_risk(), families=["trend_breakout"], register_retests=False))
    assert not res.passed and any("FAILED" in n for n in res.notes)


def test_engine_refuses_to_run_when_every_family_is_gated_out():
    eng = DiscoveryEngine(backends={"search": _fake_backend([])})
    with pytest.raises(ValueError):
        eng.run(EngineRequest(preset="search_lab", df=_small_df(), risk=_risk(), families=["no_such_family"]))


def test_engine_registers_champions_and_near_misses_for_later_retests(tmp_path, monkeypatch):
    path = tmp_path / "retests.json"
    monkeypatch.setattr(rt, "default_path", lambda: path)
    eng = DiscoveryEngine(backends={"search": _fake_backend([({"config": {"id": "a"}}, "trend_breakout", _trades(300, 0.4))])},
                          steps_factory=_fake_steps(True))
    res = eng.run(EngineRequest(preset="search_lab", df=_small_df(), risk=_risk(), families=["trend_breakout"]))
    assert res.passed
    recs = rt.RetestStore(path).all()
    assert len(recs) == 1 and next(iter(recs.values()))["kind"] == "champion" and next(iter(recs.values()))["window_end"]


# ----------------------------------------------------------------------------------------------- one real search-backend smoke
def test_search_backend_runs_end_to_end_on_a_tiny_family_set(tmp_path):
    from tests.search.test_batch_runner import _fast_stage_cfg, _frictionless_risk, _trending_df
    from app.prop.simulator import PropRules
    df = _trending_df(n=2500)
    eng = DiscoveryEngine(steps_factory=_fake_steps(False))
    req = EngineRequest(preset="search_lab", df=df, risk=_frictionless_risk(), prop_rules=PropRules(), families=["trend_breakout"],
                        work_dir=tmp_path, max_candidates=6, register_retests=False, promote=False,
                        overrides={k: v for k, v in _fast_stage_cfg().__dict__.items() if k in ("min_trades", "stage1_top_n", "ga_population", "ga_generations",
                                                                                               "ga_search_sims", "stage2_top_n", "full_mc_sims", "walk_forward_folds",
                                                                                               "robustness_neighbors", "workers", "stage1_min_t_stat", "stage1_max_cost_in_r")})
    res = eng.run(req)
    assert res.backend == "search" and res.families_run == ["trend_breakout"] and res.n_evaluations > 0
    assert (tmp_path / "search.db").exists()


# -------------------------------------------------------------------------------------------------------------- surrogates
def test_null_and_planted_frames_are_reproducible_and_causal():
    a, b = sg.make_null_frame(days=10, seed=5), sg.make_null_frame(days=10, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(sg.make_null_frame(days=10, seed=6))
    p = sg.make_planted_frame(days=10, seed=5, b=0.05)
    assert {"timestamp", "open", "high", "low", "close", "volume"} <= set(p.columns)
    assert (p["high"] >= p[["open", "close"]].max(axis=1) - 1e-9).all() and (p["low"] <= p[["open", "close"]].min(axis=1) + 1e-9).all()


def test_planted_edge_scales_with_the_drift_parameter():
    m0, n0 = sg.measure_planted_mean_r(sg.make_planted_frame(days=60, seed=3, b=0.0))
    m1, n1 = sg.measure_planted_mean_r(sg.make_planted_frame(days=60, seed=3, b=0.3))
    assert n0 > 20 and n1 > 20 and m1 > m0 + 0.05


def test_clopper_pearson_upper_bound():
    assert sg.clopper_pearson_upper(0, 20) == pytest.approx(1 - 0.05 ** (1 / 20), rel=0.02)
    assert sg.clopper_pearson_upper(0, 60) < 0.05 and sg.clopper_pearson_upper(3, 20) > 3 / 20
    assert sg.clopper_pearson_upper(0, 0) == 1.0


def test_run_surrogate_study_with_a_fake_pipeline_counts_false_discoveries_and_power(monkeypatch):
    monkeypatch.setattr(sg, "calibrate_planted", lambda t, **k: {"b": 0.1, "measured_r": t, "n_trades": 100})
    calls = []

    def pipeline(df, frames, seed):
        calls.append(set(frames))
        return {"candidate": str(next(iter(frames))).startswith("EDGE")}

    rep = sg.run_surrogate_study(pipeline, n_null=5, planted_targets=(0.2, 0.3), n_planted=2, n_markets=3, days=5, min_null_runs=5)
    assert rep.null_candidates == 0 and rep.null_rate == 0.0 and rep.power == {"0.20": 1.0, "0.30": 1.0}
    assert len(calls) == 5 + 4 and all(len(c) == 3 for c in calls) and rep.notes


def _report(**kw):
    base = dict(created_at=time.time(), code_hash="abc", n_null=60, null_candidates=1, null_rate=1 / 60, null_rate_upper95=0.08, planted=[{"target_r": 0.2, "runs": 10}, {"target_r": 0.3, "runs": 10}],
                power={"0.10": 0.2, "0.20": 0.7, "0.30": 0.9}, runs=[], min_null_runs=60, schema=sg.GATE_SCHEMA, opened_batches=[1, 2], notes=[])
    base.update(kw)
    return base


def test_gate_is_closed_without_a_report_and_opens_only_on_a_fresh_passing_one(tmp_path):
    p = tmp_path / "gate.json"
    assert not sg.gate_status(p)["open"] and sg.enabled_batches(p) == []
    p.write_text(json.dumps(_report()))
    st = sg.gate_status(p, current_hash="abc")
    assert st["open"]


@pytest.mark.parametrize("patch,needle", [
    ({"null_rate": 0.10}, "false-discovery rate"),
    ({"n_null": 5}, "null runs"),
    ({"n_null": 40, "min_null_runs": 20}, "null runs"),
    ({"planted": [{"target_r": 0.2, "runs": 3}, {"target_r": 0.3, "runs": 3}]}, "planted runs"),
    ({"power": {"0.20": 0.1, "0.30": 0.2}}, "power"),
    ({"created_at": time.time() - 200 * 86400}, "older than"),
    ({"schema": 999}, "schema"),
    ({"code_hash": "stale"}, "code changed"),
])
def test_gate_closes_on_each_failure_mode(tmp_path, patch, needle):
    p = tmp_path / "gate.json"
    p.write_text(json.dumps(_report(**patch)))
    st = sg.gate_status(p, current_hash="abc")
    assert not st["open"] and any(needle in r for r in st["reasons"])


def test_save_and_load_report_round_trip(tmp_path):
    rep = sg.SurrogateReport(created_at=1.0, code_hash="h", n_null=1, null_candidates=0, null_rate=0.0, null_rate_upper95=0.95)
    p = sg.save_report(rep, tmp_path / "r.json")
    assert sg.load_report(p)["code_hash"] == "h" and sg.load_report(tmp_path / "missing.json") is None


def test_code_hash_changes_when_engine_code_changes(tmp_path):
    (tmp_path / "app" / "discovery_engine").mkdir(parents=True)
    f = tmp_path / "app" / "discovery_engine" / "a.py"
    f.write_text("x=1")
    h1 = sg.code_hash(tmp_path)
    f.write_text("x=2")
    assert sg.code_hash(tmp_path) != h1


def test_new_v13_batches_open_only_through_the_gate(tmp_path, monkeypatch):
    from app.search import strategy_space as ss
    closed = set(ss.active_family_names())
    p = tmp_path / "gate.json"
    p.write_text(json.dumps(_report(opened_batches=[1])))
    monkeypatch.setattr(sg, "report_path", lambda: p)
    monkeypatch.setattr(sg, "code_hash", lambda root=None: "abc")
    opened = set(ss.active_family_names())
    assert opened >= closed and len(opened) > len(closed)


# ---------------------------------------------------------------------------------------------------------------------- retests
def _planted_pair():
    return sg.make_planted_frame(days=80, seed=7, b=0.3)


def test_retest_store_is_idempotent_and_ranks_due_candidates(tmp_path):
    store = rt.RetestStore(tmp_path / "r.json")
    spec = sg.planted_config()
    df = _planted_pair()
    end = df["timestamp"].iloc[len(df) // 2]
    k1 = store.register({"source_type": "manual", "config": spec}, kind="near_miss", window_end=end, family="f")
    k2 = store.register({"source_type": "manual", "config": spec}, kind="champion", window_end=end, family="f")
    assert k1 == k2 and store.all()[k1]["kind"] == "champion"
    assert store.due(df, min_new_bars=100) == [k1]
    assert store.due(df.iloc[:10], min_new_bars=100) == []


def test_retest_candidate_runs_only_on_bars_after_the_window_and_flags_a_real_edge(tmp_path):
    df = _planted_pair()
    cut = df["timestamp"].iloc[len(df) // 3]
    rec = {"spec": {"source_type": "manual", "config": sg.planted_config()}, "window_end": str(cut), "history": []}
    out = rt.retest_candidate(rec, df, _risk())
    assert out is not None and out["window_start"] > str(cut)[:10] and out["n_trades"] >= rt.MIN_RETEST_TRADES
    assert out["hit"] and out["mean_r"] > 0
    # not enough fresh data -> no retest at all
    rec["window_end"] = str(df["timestamp"].iloc[-10])
    assert rt.retest_candidate(rec, df, _risk()) is None


def test_run_retests_appends_to_history_and_lab_ranking_shrinks_small_samples(tmp_path):
    store = rt.RetestStore(tmp_path / "r.json")
    df = _planted_pair()
    key = store.register({"source_type": "manual", "config": sg.planted_config()}, kind="champion", window_end=str(df["timestamp"].iloc[len(df) // 3]),
                         family="planted_family", preset="search_lab")
    done = rt.run_retests(df, _risk(), store)
    assert [k for k, _ in done] == [key] and len(store.all()[key]["history"]) == 1
    assert rt.run_retests(df, _risk(), store) == []                    # the window moved forward: nothing new to test
    rows = rt.lab_ranking(store)
    assert rows[0]["name"] == "planted_family" and rows[0]["retests"] == 2.0
    # one lucky retest is shrunk toward the prior, not crowned
    assert rows[0]["shrunk_hit_rate"] == pytest.approx((rows[0]["hits"] + rt.PRIOR_HITS) / (2.0 + rt.PRIOR_N))
    assert rows[0]["shrunk_hit_rate"] < 1.0
    assert rt.lab_ranking(store, by="preset")[0]["name"] == "search_lab"


def test_inconclusive_retests_do_not_count_toward_the_ranking(tmp_path):
    store = rt.RetestStore(tmp_path / "r.json")
    key = store.register({"config": {"a": 1}}, kind="champion", window_end="2024-01-01", family="f")
    store.record(key, {"conclusive": False, "hit": False, "mean_r": 0.0, "window_end": "2024-02-01"})
    assert rt.lab_ranking(store) == []


# ---------------------------------------------------------------------------------------------------------------- status page
def test_status_api_and_page(client_or_none=None):
    from app.web.server import app
    c = app.test_client()
    r = c.get("/api/discovery-engine/status")
    assert r.status_code == 200
    d = r.get_json()
    assert d["objective"] == "edge_lb_r" and len(d["presets"]) >= 5
    assert d["floor"]["min_trades"] == 100
    assert len(d["v13"]["rows"]) == 100 and d["v13"]["blocked"] == 8
    assert d["gate"]["open"] is False                                    # shipped closed: the study has to be run on the user's machine
    page = c.get("/discovery-engine")
    assert page.status_code == 200 and b"Strategy Discovery Engine" in page.data


def test_preregister_endpoint_is_idempotent(tmp_path, monkeypatch):
    from app.discovery import hypothesis as h
    from app.web.server import app
    real = h.HypothesisStore
    monkeypatch.setattr(h, "HypothesisStore", lambda path=None: real(tmp_path / "hyp.json"))
    c = app.test_client()
    r1 = c.post("/api/discovery-engine/preregister")
    r2 = c.post("/api/discovery-engine/preregister")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r2.get_json()["registered"] == 0 and r2.get_json()["total"] == 100


# ------------------------------------------------------------------------------------- the integrated start page + job flow
def test_engine_is_inside_create_not_its_own_sidebar_entry():
    from app.web.server import app
    c = app.test_client()
    create = c.get("/start-here/create").get_data(as_text=True)
    assert 'href="/discovery-engine"' in create and "I don&rsquo;t have one yet" in create and "I have a strategy" in create
    # the old standalone sidebar entry (it sat right before "(1) Create") is gone; Create is the way in
    sidebar = create.split('class="t58-main"')[0]
    assert 'href="/discovery-engine"' not in sidebar


def test_engine_page_is_the_tool_not_a_hub_of_links():
    from app.web.server import app
    html = app.test_client().get("/discovery-engine").get_data(as_text=True)
    assert 'action="/discovery-engine/start"' in html and "Start searching" in html
    assert 'name="dataset"' in html and 'id="de_prop_preset_select"' in html and 'name="effort"' in html


def test_engine_start_validates_input():
    from app.web.server import app
    c = app.test_client()
    r = c.post("/discovery-engine/start", data={})
    assert r.status_code == 400 and b"Pick the market data" in r.data
    r = c.post("/discovery-engine/start", data={"dataset": "../../etc/passwd"})
    assert r.status_code == 400


def test_engine_job_flow_with_a_stub_engine(monkeypatch):
    """Start -> background job -> status.json carries plain-language steps -> save champion to the library."""
    import time
    from pathlib import Path

    import pandas as pd

    from app.discovery_engine import engine as eng
    from app.web.server import app

    root = Path(__file__).resolve().parents[2] / "data" / "raw" / "EURUSD" / "EURUSD240.csv"
    if not root.exists():
        import pytest
        pytest.skip("bundled sample data not present")

    class Stub:
        def run(self, req):
            req.progress_cb("stub engine ran")
            spec = {"source_type": "manual", "config": {"name": "x", "entry_conditions": [], "exit_conditions": [], "risk_management": {}}}
            champs = [{"spec": spec, "family": "rsi_extreme_reversion", "candidate_id": "c1", "edge_lb_r": 0.12, "edge": {"n_trades": 150}, "floor_reasons": []}]
            promos = [{"passed": False, "failed_step": "null_gate", "failed_reason": "no better than random timing",
                       "steps": [{"name": "reproduction", "passed": True, "detail": "same"}, {"name": "null_gate", "passed": False, "detail": "p=0.4"},
                                 {"name": "sealed_slice", "passed": False, "skipped": True, "detail": ""}]}]
            return eng.EngineResult(run_id="r", preset=req.preset, backend="x", objective="edge_lb_r", families_run=["rsi_extreme_reversion"], gated_out={},
                                    n_evaluations=10, trial_inputs={"n_tests": 3}, champions=champs, promotions=promos, provenance={}, notes=[], elapsed_seconds=1.0)

    monkeypatch.setattr("app.discovery_engine.engine.DiscoveryEngine", Stub)
    c = app.test_client()
    r = c.post("/discovery-engine/start", data={"dataset": "EURUSD/EURUSD240.csv", "method": "gauntlet", "effort": "quick", "max_bars": "3000"})   # the default method is the Fast Lab; "gauntlet" is the engine
    assert r.status_code == 302, r.get_data(as_text=True)[:400]
    job_id = r.headers["Location"].rsplit("/", 1)[-1]
    data = {}
    for _ in range(100):
        data = c.get(f"/discovery-engine/job/{job_id}/status.json").get_json()
        if data.get("done"):
            break
        time.sleep(0.1)
    assert data["done"] and not data["error"], data
    ch = data["result"]["champions"][0]
    assert ch["passed"] is False and ch["failed_step"] == "Beats random entry timing"
    assert [s["state"] for s in ch["steps"]] == ["pass", "fail", "skipped"]
    assert c.get(f"/discovery-engine/job/{job_id}").status_code == 200
    save = c.post(f"/discovery-engine/job/{job_id}/save/0")
    assert save.status_code == 200 and save.get_json()["ok"] and "load_strategy_type=manual" in save.get_json()["test_url"]
    from app.strategy.library import get_strategy_library_dir
    (get_strategy_library_dir("manual") / save.get_json()["name"]).unlink()
