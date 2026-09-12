"""End-to-end web tests for Forge Strategy's new Loop Mode -- real POST to
/forge/start with loop_mode=on, driving the actual background thread
(_run_forge_loop_job -> run_forge_loop -> run_forge, real threads, not
mocks) through to completion, same pattern as
tests/test_web_search.py's loop-mode tests.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

import app.web.server as server_module
from app.web.server import _FORGE_JOBS, app
from app.orchestration.resource_guard import HEAVY_JOB_GUARD, JOB_FORGE

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

_TINY_ADVANCED_FIELDS = {
    "advanced_mode": "on",
    "n_hypotheses": "6", "stage1_top_n": "6", "stage2_top_n": "3",
    "stage3_mc_sims": "30", "cpcv_pool_size": "2", "cpcv_survivors": "2",
    "final_mc_sims": "50", "mc_survivors": "2", "eval_window_days": "10",
    "rolling_survivors": "2", "locked_holdout_frac": "0.15",
    "ga_population": "4", "ga_generations": "1", "workers": "1", "seed": "1",
}


@pytest.fixture(autouse=True)
def _cleanup_forge_artifacts(tmp_path, monkeypatch):
    isolated_forge_dir = tmp_path / "forge"
    isolated_forge_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(server_module, "FORGE_DIR", isolated_forge_dir)
    monkeypatch.setattr("app.search.graveyard.get_app_base_dir", lambda: tmp_path)
    yield
    _FORGE_JOBS.clear()


def _poll_until_done(client, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/forge/job/{job_id}/status.json")
        data = r.get_json()
        assert data["found"] is True
        if data["done"]:
            return data
        time.sleep(0.3)
    raise AssertionError(f"forge job {job_id} did not finish within {timeout}s")


def test_forge_loop_mode_end_to_end():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "loop_mode": "on",
            "loop_target_pass_rate_pct": "0",  # trivially easy
            "loop_max_rounds": "1",
            "loop_stall_rounds": "1",
            **_TINY_ADVANCED_FIELDS,
        }
        r = client.post("/forge/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    status = _poll_until_done(client, job_id)
    assert status["error"] is None
    assert status["loop_mode"] is True
    assert status["loop_rounds"] >= 1
    assert status["loop_last_round"] is not None
    assert status["loop_result"] is not None
    assert status["loop_result"]["stopped_reason"] in {"target_reached", "max_rounds"}
    assert isinstance(status["funnel"], list)


def test_forge_loop_mode_can_be_stopped(monkeypatch):
    import app.orchestration.loop_runner as loop_runner_module

    real_run_forge = loop_runner_module.run_forge if hasattr(loop_runner_module, "run_forge") else None

    def _slow_run_forge(*args, **kwargs):
        from app.orchestration.forge import run_forge as _actual_run_forge
        time.sleep(0.5)
        return _actual_run_forge(*args, **kwargs)

    monkeypatch.setattr("app.orchestration.forge.run_forge", _slow_run_forge)

    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "loop_mode": "on",
            "loop_target_pass_rate_pct": "99.9",  # unreachable -- keep looping until stopped
            "loop_max_rounds": "50",
            "loop_stall_rounds": "1",
            **_TINY_ADVANCED_FIELDS,
        }
        r = client.post("/forge/start", data=data, content_type="multipart/form-data")
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    client.post(f"/forge/job/{job_id}/stop")
    status = _poll_until_done(client, job_id, timeout=30.0)
    assert status["cancelled"] is True
    assert status["loop_result"]["stopped_reason"] == "cancelled"
