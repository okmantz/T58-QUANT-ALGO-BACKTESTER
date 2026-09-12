"""Tests for Evolution Lab's "loop mode" -- EvolutionConfig.target_eval_pass_pct
makes a run stop ITSELF the first generation a leaderboard candidate clears
the target, instead of running forever (or to max_generations) regardless
of what's already on the leaderboard. See EvolutionConfig's own comment
for the full rationale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.evolution.engine import EvolutionCandidateRecord, EvolutionConfig, EvolutionRunner
from app.prop.simulator import PropRules


def _trending_df(n=200, seed=3):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.linspace(0, 40, n)
    noise = np.cumsum(rng.normal(0, 0.4, n))
    price = 1900 + drift + noise
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.3, "low": price - 0.3,
        "close": price, "volume": 100.0,
    })


def _record(candidate_id, cpcv=None, raw=None):
    return EvolutionCandidateRecord(
        candidate_id=candidate_id, spec={}, meta={},
        mc_summary={"evaluation_pass_probability": raw} if raw is not None else {},
        cpcv_oos_eval_pass_probability=cpcv,
    )


def _runner(tmp_path, **cfg_overrides):
    cfg = EvolutionConfig(
        knowledge_graph_path=str(tmp_path / "kg.jsonl"),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        tested_log_path=str(tmp_path / "tested_candidates.jsonl"),
        **cfg_overrides,
    )
    return EvolutionRunner(_trending_df(), RiskConfig(), PropRules(), cfg, progress_cb=None)


# ---------------------------------------------------------------------------
# _check_target_reached -- pure function of self.leaderboard + cfg, no run needed
# ---------------------------------------------------------------------------

def test_no_target_configured_is_a_no_op(tmp_path):
    runner = _runner(tmp_path)  # target_eval_pass_pct defaults to None
    runner.leaderboard = [_record("a", cpcv=99.0)]
    assert runner._check_target_reached() is None


def test_target_reached_on_cpcv_metric_by_default(tmp_path):
    runner = _runner(tmp_path, target_eval_pass_pct=60.0)
    runner.leaderboard = [_record("below", cpcv=40.0), _record("above", cpcv=61.0)]
    hit = runner._check_target_reached()
    assert hit is not None and hit.candidate_id == "above"


def test_target_not_reached_when_every_candidate_is_below(tmp_path):
    runner = _runner(tmp_path, target_eval_pass_pct=60.0)
    runner.leaderboard = [_record("a", cpcv=40.0), _record("b", cpcv=50.0)]
    assert runner._check_target_reached() is None


def test_target_check_respects_raw_metric_when_configured(tmp_path):
    runner = _runner(tmp_path, target_eval_pass_pct=60.0, target_metric="eval_pass_probability")
    # A candidate with a great CPCV number must NOT trigger when the
    # configured metric is the raw one and the raw number is still low --
    # confirms the metric switch actually changes which field is read.
    runner.leaderboard = [_record("cpcv_only", cpcv=99.0, raw=10.0), _record("raw_winner", cpcv=5.0, raw=61.0)]
    hit = runner._check_target_reached()
    assert hit is not None and hit.candidate_id == "raw_winner"


def test_target_check_ignores_candidates_with_no_value_for_the_metric(tmp_path):
    runner = _runner(tmp_path, target_eval_pass_pct=60.0)
    runner.leaderboard = [_record("no_cpcv_computed", cpcv=None)]
    assert runner._check_target_reached() is None


# ---------------------------------------------------------------------------
# _run_loop actually stops early once the target is hit
# ---------------------------------------------------------------------------

def test_run_loop_stops_before_max_generations_once_target_reached(tmp_path, monkeypatch):
    """Fabricates a winning leaderboard entry starting at generation 2 and
    confirms the loop (configured for 10 generations) stops right there
    instead of continuing to generation 10 -- the actual behavior change
    this feature is for, not just the pure-function check above."""
    runner = _runner(tmp_path, max_generations=10, target_eval_pass_pct=60.0, population_size=4, elite_keep=2)

    call_count = {"n": 0}

    def fake_run_one_generation(gen):
        call_count["n"] += 1
        if call_count["n"] == 3:
            runner.leaderboard = [_record("winner", cpcv=75.0)]
        return gen

    monkeypatch.setattr(runner, "_run_one_generation", fake_run_one_generation)
    runner._run_loop()

    assert call_count["n"] == 3  # stopped right after the 3rd generation, not all 10
    assert runner._target_reached_by is not None
    assert runner._target_reached_by.candidate_id == "winner"
    status = runner.status()
    assert status["target_reached"] is True
    assert status["target_reached_candidate_id"] == "winner"


def test_run_loop_without_a_target_runs_to_max_generations_as_before(tmp_path, monkeypatch):
    """Regression guard: a run that never sets target_eval_pass_pct must
    behave exactly as it always did -- run every generation regardless of
    what's on the leaderboard."""
    runner = _runner(tmp_path, max_generations=4, population_size=4, elite_keep=2)

    call_count = {"n": 0}

    def fake_run_one_generation(gen):
        call_count["n"] += 1
        runner.leaderboard = [_record("great", cpcv=99.0)]  # would trigger a target if one were set
        return gen

    monkeypatch.setattr(runner, "_run_one_generation", fake_run_one_generation)
    runner._run_loop()

    assert call_count["n"] == 4
    assert runner._target_reached_by is None
    assert runner.status()["target_reached"] is False
