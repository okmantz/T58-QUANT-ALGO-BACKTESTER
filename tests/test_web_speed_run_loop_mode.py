"""End-to-end web tests for Speed Run's new Loop Mode -- real POST to
/speed-run/start with loop_mode=on, driving the actual background thread
(_run_speedrun_loop_job -> run_speed_run_loop -> run_speed_run, real
threads, not mocks) through to completion. Also confirms the single-shot
(non-loop) path is unaffected, and that the new /speed-run/job/<id>/stop
route (which didn't exist before Loop Mode) works.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

import app.web.server as server_module
from app.web.server import _SPEEDRUN_JOBS, app

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

_FAST_FIELDS = {
    "max_candidates": "20", "stage1_top_n": "4", "ga_population": "4", "ga_generations": "1",
    "top_k_to_validate": "2", "max_concurrent_validations": "1",
    "validation_folds": "0", "validation_final_mc_sims": "40", "random_seed": "1",
}


@pytest.fixture(autouse=True)
def _cleanup_speedrun_artifacts(tmp_path, monkeypatch):
    isolated_dir = tmp_path / "speed_run"
    isolated_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(server_module, "SPEEDRUN_DIR", isolated_dir)
    monkeypatch.setattr(server_module, "SPEEDRUN_REPORTS_DIR", isolated_dir / "speed_run")
    yield
    _SPEEDRUN_JOBS.clear()


def _poll_until_done(client, job_id: str, timeout: float = 90.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/speed-run/job/{job_id}/status.json")
        data = r.get_json()
        assert data["found"] is True
        if data["done"]:
            return data
        time.sleep(0.3)
    raise AssertionError(f"speed run job {job_id} did not finish within {timeout}s")


def test_speed_run_loop_mode_end_to_end():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "loop_mode": "on", "loop_max_rounds": "1", "loop_stall_rounds": "1",
            **_FAST_FIELDS,
        }
        r = client.post("/speed-run/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    status = _poll_until_done(client, job_id)
    assert status["error"] is None
    assert status["loop_mode"] is True
    assert status["loop_rounds"] >= 1
    assert status["loop_result"] is not None
    assert status["loop_result"]["stopped_reason"] in {"target_reached", "max_rounds"}
    assert status["summary"] is not None
    # If a winner was found, its report must actually be servable through
    # the new loop-scoped route (not 404 via the flat single-shot one).
    winner = status["summary"].get("winner")
    if winner and winner.get("report_html"):
        assert winner["report_html"].startswith(f"/speed_run_reports_loop/{job_id}/")
        report_resp = client.get(winner["report_html"])
        assert report_resp.status_code == 200


def test_speed_run_non_loop_path_is_unaffected():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {"csv_file": (f, "EURUSD_5M_sample.csv"), **_FAST_FIELDS}
        r = client.post("/speed-run/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    status = _poll_until_done(client, job_id)
    assert status["error"] is None
    assert status["loop_mode"] is False
    assert status["loop_result"] is None


def test_speed_run_loop_mode_can_be_stopped(monkeypatch):
    def _slow_run_speed_run(*args, **kwargs):
        from app.orchestration.speed_run import run_speed_run as _actual
        time.sleep(0.5)
        return _actual(*args, **kwargs)

    monkeypatch.setattr("app.orchestration.speed_run.run_speed_run", _slow_run_speed_run)

    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "loop_mode": "on", "loop_max_rounds": "50", "loop_stall_rounds": "1",
            **_FAST_FIELDS,
        }
        r = client.post("/speed-run/start", data=data, content_type="multipart/form-data")
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    client.post(f"/speed-run/job/{job_id}/stop")
    status = _poll_until_done(client, job_id, timeout=30.0)
    assert status["cancelled"] is True
    assert status["loop_result"]["stopped_reason"] == "cancelled"


def test_speed_run_stop_route_is_a_safe_no_op_for_a_non_loop_job():
    """The single-shot path never gets a cancel_event -- confirm /stop on
    that kind of job is a harmless no-op rather than an error."""
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {"csv_file": (f, "EURUSD_5M_sample.csv"), **_FAST_FIELDS}
        r = client.post("/speed-run/start", data=data, content_type="multipart/form-data")
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    stop_resp = client.post(f"/speed-run/job/{job_id}/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.get_json()["ok"] is True
    # And the run itself still completes normally (never cancelled).
    status = _poll_until_done(client, job_id)
    assert status["cancelled"] is False
