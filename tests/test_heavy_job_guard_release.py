"""Regression tests: every heavy job must free the shared HEAVY_JOB_GUARD
slot when it finishes -- or when starting it fails (Owen, 2026-10-06).

Follow-up to the Quick Optimize slot leak (test_quickopt_guard_release.py).
A full audit of all 18 guard slots found one more hard leak plus two
smaller holes, all fixed alongside these tests:

1. /search/start timeframe-sweep hand-off: the route releases
   JOB_SEARCH_LAB and acquires JOB_MULTI_INSTRUMENT_SEARCH, but all three
   of its exception handlers released JOB_SEARCH_LAB again -- a silent
   no-op (release() only frees the name actually held) -- so any failure
   while expanding/starting the sweep held "Multi-Instrument Search"
   forever and every other heavy job was refused until a server restart.
   Same wrong-name pattern in /evolution/start's hand-off (that slot has
   a health check, so it self-healed, but it was still wrong).
2. The overnight scheduler's deferred Full Pipeline launch acquired the
   slot, then called JOB_MANAGER.create() outside any try -- a failure
   there stranded JOB_FULL_PIPELINE.
3. The Sensitivity 2D heatmap ran completely unguarded: it could stack
   on top of any other heavy job. It now takes JOB_SENSITIVITY and
   releases it when done.

Each test below fails on the pre-fix code and passes after.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import app.web.server as server
from app.orchestration.resource_guard import (
    JOB_EVOLUTION_LAB,
    JOB_FULL_PIPELINE,
    JOB_MULTI_INSTRUMENT_EVOLUTION,
    JOB_MULTI_INSTRUMENT_SEARCH,
    JOB_SEARCH_LAB,
    JOB_SENSITIVITY,
    JOB_WFO,
)
from app.web.server import app

_GUARD = server.HEAVY_JOB_GUARD
_MANAGER = server.JOB_MANAGER
SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

_ALL_NAMES = (
    JOB_SEARCH_LAB, JOB_MULTI_INSTRUMENT_SEARCH, JOB_EVOLUTION_LAB,
    JOB_MULTI_INSTRUMENT_EVOLUTION, JOB_SENSITIVITY, JOB_WFO, JOB_FULL_PIPELINE,
)


@pytest.fixture(autouse=True)
def _clean_guard_state():
    for name in _ALL_NAMES:
        _GUARD.release(name)
    yield
    for name in _ALL_NAMES:
        _GUARD.release(name)


_SEARCH_FIELDS = {
    "search_mode": "family_named", "family": "trend_breakout",
    "max_candidates": "8", "seed": "1", "workers": "2",
    "min_trades": "1", "min_profit_factor": "0.0", "stage1_top_n": "5",
    "ga_population": "4", "ga_generations": "1", "stage2_top_n": "3",
    "full_mc_sims": "60", "walk_forward_folds": "0", "robustness_neighbors": "0",
    "fitness_metric": "composite_prop_score",
}

_EVO_FIELDS = {
    "account_size": "50000", "initial_balance": "50000",
    "profit_target": "6", "daily_loss": "2", "max_dd": "4",
    "population_size": "4", "elite_keep": "1", "mc_sims": "10", "max_generations": "1",
}


def _post_with_csv(client, url: str, fields: dict):
    with open(SAMPLE_CSV, "rb") as f:
        data = {"csv_file": (f, "EURUSD_5M_sample.csv"), **fields}
        return client.post(url, data=data, content_type="multipart/form-data")


def test_search_sweep_handoff_failure_frees_multi_search_slot(monkeypatch):
    """Regression for leak #1: if the timeframe expansion blows up after
    the hand-off, the slot must not stay held as Multi-Instrument Search."""
    called = []

    def _boom(*a, **k):
        called.append(True)
        raise RuntimeError("resample exploded")

    monkeypatch.setattr(server, "expand_dataset_across_timeframes", _boom)
    client = app.test_client()
    r = _post_with_csv(client, "/search/start", {**_SEARCH_FIELDS, "expand_timeframes": "15"})

    assert called, "request never reached the timeframe-sweep hand-off"
    assert r.status_code == 500
    assert _GUARD.active_name is None


def test_evolution_sweep_handoff_failure_frees_multi_evolution_slot(monkeypatch):
    """Same wrong-name release in /evolution/start's sweep hand-off."""
    called = []

    def _boom(*a, **k):
        called.append(True)
        raise RuntimeError("resample exploded")

    monkeypatch.setattr(server, "expand_dataset_across_timeframes", _boom)
    monkeypatch.setattr(server, "_EVOLUTION_RUNNER", None, raising=False)
    client = app.test_client()
    r = _post_with_csv(client, "/evolution/start", {**_EVO_FIELDS, "expand_timeframes": "15"})

    assert called, "request never reached the timeframe-sweep hand-off"
    assert r.status_code == 500
    assert _GUARD.active_name is None


def _make_heatmap_job() -> str:
    job_id = _MANAGER.create(log=["sensitivity test job"], instrument="TEST")
    _MANAGER.update(
        job_id,
        _ctx={"df": None, "strategy": None, "risk": None, "rules": None,
              "mc_cfg": None, "metric": "eval_pass"},
        results=[SimpleNamespace(gene_label="a"), SimpleNamespace(gene_label="b")],
    )
    return job_id


def _stub_heatmap_compute(monkeypatch):
    monkeypatch.setattr(
        server, "compute_2d_heatmap",
        lambda *a, **k: SimpleNamespace(to_dict=lambda: {}),
    )
    monkeypatch.setattr(
        server, "generate_sensitivity_report",
        lambda *a, **k: {"html": "heatmap_test_report.html"},
    )


def test_sensitivity_heatmap_route_refuses_while_slot_is_held(monkeypatch):
    """Gap fix: the 2D heatmap is heavy compute; it must not start on
    top of another heavy job."""
    _stub_heatmap_compute(monkeypatch)
    job_id = _make_heatmap_job()
    assert _GUARD.try_acquire(JOB_WFO) is True
    try:
        client = app.test_client()
        r = client.post(f"/sensitivity/job/{job_id}/heatmap", data={"param_a": "a", "param_b": "b"})
        assert r.status_code == 409
        assert r.get_json()["ok"] is False
    finally:
        _GUARD.release(JOB_WFO)


def test_sensitivity_heatmap_runner_releases_the_slot(monkeypatch):
    """The heatmap runner frees JOB_SENSITIVITY when it finishes."""
    _stub_heatmap_compute(monkeypatch)
    job_id = _make_heatmap_job()
    assert _GUARD.try_acquire(JOB_SENSITIVITY) is True

    server._run_sensitivity_heatmap_job(job_id, "a", "b", 0.5, 3)

    assert _MANAGER.get(job_id)["heatmap_done"] is True
    assert _GUARD.active_name is None


def test_scheduled_fullpipeline_create_failure_frees_the_slot(monkeypatch):
    """Minor leak: JOB_MANAGER.create() failing right after the deferred
    scheduler acquired JOB_FULL_PIPELINE must not strand the slot."""

    def _boom(*a, **k):
        raise RuntimeError("job store unavailable")

    monkeypatch.setattr(_MANAGER, "create", _boom)
    server._SCHEDULED_JOBS["sched-test"] = {"cancelled": False}
    try:
        with pytest.raises(RuntimeError):
            server._run_scheduled_fullpipeline_batch(
                "sched-test", 0.0,
                {"initial_log": [], "active_label": "TEST", "batch_items": []},
            )
        assert _GUARD.active_name is None
    finally:
        server._SCHEDULED_JOBS.pop("sched-test", None)
