"""Evolution Lab on the Strategy Discovery Engine: shared floor and objective, mean-R folds, pool P&L, budgeted holdout, mandatory stop."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.discovery_engine import promotion as pr
from app.discovery_engine.holdout import HoldoutExhausted
from app.evolution import engine as eng
from app.evolution.edge_fitness import compute_edge_fitness
from app.evolution.engine import EvolutionCandidateRecord, EvolutionConfig, EvolutionRunner
from app.prop.simulator import PropRules
from app.validation import trial_ledger as tl
from tests.evolution.test_evolution_lab import _trending_df


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_PATH", tmp_path / "reports" / "trial_ledger.json")


def _cfg(tmp_path, **kw):
    base = dict(population_size=10, elite_keep=2, max_generations=1, mc_sims=50, robustness_neighbors=1, walk_forward_folds=2,
                cpcv_top_n=2, cpcv_max_paths=3, cpcv_n_groups=3, save_to_library=False, min_trades=3, min_profit_factor=0.0,
                min_t_stat=0.0, max_cost_in_r=0.0, max_drawdown_buffer_mult=20.0,
                knowledge_graph_path=str(tmp_path / "kg.jsonl"), checkpoint_path=str(tmp_path / "checkpoint.json"),
                tested_log_path=str(tmp_path / "tested.jsonl"))
    base.update(kw)
    return EvolutionConfig(**base)


def test_defaults_are_the_one_objective_and_one_floor():
    c = EvolutionConfig()
    assert c.objective == "edge_lb_r"
    assert (c.min_trades, c.min_t_stat, c.max_cost_in_r) == (100, 1.0, 0.15)
    assert c.walk_forward_metric == "mean_r" and c.cpcv_metric == "mean_r" and c.strict_entry_floor
    with pytest.raises(ValueError):
        EvolutionConfig(objective="nonsense")


def test_edge_fitness_orders_by_edge_charges_complexity_and_rejects_unmeasurable():
    class T:  # minimal trade stand-in understood by trade_r_and_days
        pass

    def trades(mean):
        rng = np.random.default_rng(1)
        out = []
        ts = pd.Timestamp("2024-01-02")
        for i in range(200):
            t = type("Tr", (), {})()
            t.pnl = float(rng.normal(mean, 1.0)) * 100
            t.initial_risk = 100.0
            t.size = 1.0
            t.entry_time = ts + pd.Timedelta(days=i)
            t.exit_time = t.entry_time + pd.Timedelta(hours=3)
            t.r_multiple = t.pnl / 100.0
            out.append(t)
        return out

    stats = {"total_trades": 200, "max_drawdown_pct": 3.0, "max_losing_streak": 4}
    good, weak = trades(0.4), trades(0.0)
    fg = compute_edge_fitness(stats, good, {}, None, None, [t.pnl for t in good])
    fw = compute_edge_fitness(stats, weak, {}, None, None, [t.pnl for t in weak])
    assert fg.final_score > fw.final_score
    spec = {"config": {"entry_conditions": {"long": [{"a": i} for i in range(12)], "short": []}, "exit_conditions": {"long": [], "short": []}}}
    assert compute_edge_fitness(stats, good, {}, None, None, [t.pnl for t in good], spec=spec).final_score < fg.final_score
    assert compute_edge_fitness({"total_trades": 3}, good[:3], {}, None, None, []).final_score == -1e6


def test_prefilter_task_applies_the_shared_floor():
    df = _trending_df(2500)
    eng._EVO_WORKER["df"], eng._EVO_WORKER["risk"], eng._EVO_WORKER["adaptive_risk"] = df, RiskConfig(), None
    from app.search.strategy_space import generate_search_space
    space = generate_search_space(mode="family", family="trend_breakout", max_candidates=3, seed=1)
    cid, spec = next(iter(space.candidates.items()))
    meta = space.meta.get(cid, {})
    loose = eng._evo_prefilter_task(cid, spec, meta, 1, 0.0, 100.0, 20.0)
    strict_n = eng._evo_prefilter_task(cid, spec, meta, 10_000, 0.0, 100.0, 20.0)
    assert "min_trades" in strict_n[4]
    t_gate = eng._evo_prefilter_task(cid, spec, meta, 1, 0.0, 100.0, 20.0, None, 1e9, 0.0)
    if loose[6] is not None:
        assert "t_stat" in t_gate[4]


def test_one_generation_streams_pool_pnl_and_checkpoints_the_objective(tmp_path):
    df = _trending_df()
    runner = EvolutionRunner(df, RiskConfig(), PropRules(), _cfg(tmp_path), progress_cb=None)
    runner._run_loop()
    assert runner.generation == 0 and len(runner.journal) == 1
    st = runner.status()
    assert st["objective"] == "edge_lb_r" and st["holdout_budget"]["max_looks"] == 200
    from app.evolution import checkpoint as ck
    saved = ck.load_checkpoint(tmp_path / "checkpoint.json")
    assert saved.extra["objective"] == "edge_lb_r"
    store_parts = list(tmp_path.glob("evolution_pool_pnl.part*.npz"))
    if runner.status()["leaderboard_size"] or store_parts:
        from app.discovery_engine.pnl_store import PnlStore
        series, fams = PnlStore.load_series(tmp_path / "evolution_pool_pnl")
        assert all(k.startswith("g0:") for k in series) and len(series) == len(fams)
    assert runner._run_trials.total >= 10          # every child of the generation is counted for the Deflated Sharpe


def test_checkpoint_from_another_objective_is_not_resumed(tmp_path):
    df = _trending_df()
    legacy = _cfg(tmp_path, objective="eval_pass_probability")
    EvolutionRunner(df, RiskConfig(), PropRules(), legacy, progress_cb=None)._run_loop()
    logs = []
    r2 = EvolutionRunner(df, RiskConfig(), PropRules(), _cfg(tmp_path), progress_cb=logs.append)
    assert not r2.resumed and any("different scales" in m or "fresh run" in m for m in logs)


def _record(cid="c1"):
    return EvolutionCandidateRecord(candidate_id=cid, spec={"source_type": "manual", "config": {"name": "x"}}, meta={"family": "f"},
                                    stats={"total_trades": 10}, mc_summary={}, trade_pnls=[1.0] * 10)


def test_holdout_gate_fails_closed_when_the_look_budget_is_spent(tmp_path):
    df = _trending_df()
    runner = EvolutionRunner(df, RiskConfig(), PropRules(), _cfg(tmp_path, holdout_max_looks=0), progress_cb=None)
    rec = _record()
    with pytest.raises(HoldoutExhausted):
        runner.evaluate_champion_on_locked_holdout(rec)
    assert runner._apply_holdout_gate([rec]) == []
    assert rec.holdout_check["budget_exhausted"] is True


def _loop_runner(tmp_path, monkeypatch, steps):
    df = _trending_df(600)
    runner = EvolutionRunner(df, RiskConfig(), PropRules(), _cfg(tmp_path, max_generations=5, target_eval_pass_pct=70.0, target_metric="eval_pass_probability"), progress_cb=None)
    rec = _record()
    rec.mc_summary = {"per_attempt_pass_probability": 99.0}
    rec.trades = [object()]
    rec.provenance = {"seed": 1}
    runner.leaderboard = [rec]
    monkeypatch.setattr(runner, "_run_one_generation", lambda gen: gen)
    monkeypatch.setattr(runner, "_promotion_steps", lambda: steps)
    monkeypatch.setattr(runner, "pool_tests", lambda force=False: None)
    monkeypatch.setattr(runner, "prune_record", lambda r: {})
    return runner, rec


def test_loop_stop_triggers_the_mandatory_sequence_and_fails_the_run_if_a_step_fails(tmp_path, monkeypatch):
    steps = pr.Steps(**{n: (lambda c, ctx, n=n: pr.StepResult(n, True)) for n in pr.STEP_ORDER})
    steps.null_gate = lambda c, ctx: pr.StepResult("null_gate", False, detail="random timing does as well")
    runner, rec = _loop_runner(tmp_path, monkeypatch, steps)
    runner._run_loop()
    assert runner.run_failed and "null_gate" in runner.run_failed_reason
    assert rec.promotion["failed_step"] == "null_gate" and runner.status()["run_failed"]


def test_loop_stop_with_every_step_passing_keeps_the_champion(tmp_path, monkeypatch):
    steps = pr.Steps(**{n: (lambda c, ctx, n=n: pr.StepResult(n, True)) for n in pr.STEP_ORDER})
    runner, rec = _loop_runner(tmp_path, monkeypatch, steps)
    runner._run_loop()
    assert not runner.run_failed and rec.promotion["passed"] is True and rec.promotion["forward_test_required"] is True
