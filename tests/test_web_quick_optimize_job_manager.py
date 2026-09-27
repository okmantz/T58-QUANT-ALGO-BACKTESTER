"""
End-to-end web tests for Quick Optimize's job routes (app.web.server)
after their migration from a hand-rolled _QUICKOPT_JOBS dict onto the
shared JOB_MANAGER (see app.web.job_manager's module docstring). Real
POST to /quick-optimize/start, driving the actual background thread
(_run_quickopt_job -> run_quick_optimize, real threads, not mocks)
through to completion -- same pattern as
tests/test_web_forge_loop_mode.py's own migration tests.
"""
from __future__ import annotations

import time
from pathlib import Path

from app.web.server import app

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

_TINY_GA_FIELDS = {
    "strategy_mode": "manual",
    "ga_population": "4", "ga_generations": "1", "n_folds": "2",
    "initial_balance": "10000", "account_size": "10000",
    "save_to_library": "",  # unchecked -- don't touch the real strategy library from a test
}


def _poll_until_done(client, job_id: str, url_prefix: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"{url_prefix}/job/{job_id}/status.json")
        data = r.get_json()
        assert data["found"] is True
        if data["done"]:
            return data
        time.sleep(0.2)
    raise AssertionError(f"quick optimize job {job_id} did not finish within {timeout}s")


def test_quick_optimize_end_to_end():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {"csv_file": (f, "EURUSD_5M_sample.csv"), **_TINY_GA_FIELDS}
        r = client.post("/quick-optimize/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    # Job page renders (not a 404) while the job is in flight.
    assert client.get(f"/quick-optimize/job/{job_id}").status_code == 200

    status = _poll_until_done(client, job_id, "/quick-optimize")
    assert status["error"] is None
    assert status["cancelled"] is False
    assert status["summary"] is not None
    assert status["summary"]["strategy_display_name"]
    assert isinstance(status["log"], list) and len(status["log"]) > 0


def test_quick_optimize_stop_endpoint_is_a_safe_noop_on_a_finished_job():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {"csv_file": (f, "EURUSD_5M_sample.csv"), **_TINY_GA_FIELDS}
        r = client.post("/quick-optimize/start", data=data, content_type="multipart/form-data")
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]
    _poll_until_done(client, job_id, "/quick-optimize")

    # Stopping an already-finished job must not error -- the cancel_event
    # is still stashed on the (finished) JOB_MANAGER entry either way.
    resp = client.post(f"/quick-optimize/job/{job_id}/stop")
    assert resp.status_code == 200
    assert resp.get_json()["found"] is True


def test_quick_optimize_unknown_job_id_is_404():
    client = app.test_client()
    assert client.get("/quick-optimize/job/does-not-exist/status.json").status_code == 404
    assert client.get("/quick-optimize/job/does-not-exist").status_code == 404
    assert client.post("/quick-optimize/job/does-not-exist/stop").status_code == 404
