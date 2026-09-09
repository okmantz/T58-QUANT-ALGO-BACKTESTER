"""
End-to-end tests for the Research Loop web routes (app.web.server) --
POSTs to /research-loop/start, polls the real background job through
/research-loop/job/<id>/status.json, and exercises stop. Not mocks of the
job system -- the real threading.Thread -> ResearchLoopRunner ->
run_research_loop path a browser would drive, with only Ollama's own
generate_strategy/_ask_ollama_next_hypothesis calls mocked out (same
mocking boundary tests/test_research_loop.py itself uses -- no real
Ollama needed).
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

import app.ai.research_loop as rl
from app.ai.strategy_generator import GenerationResult
from app.web.server import _RESEARCH_LOOP_JOBS, app

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

_SMA_CROSS_CODE = """
import numpy as np

STOP_LOSS_PIPS = 20
TAKE_PROFIT_PIPS = 40

def generate_signals(df):
    fast = df['close'].rolling(10).mean()
    slow = df['close'].rolling(30).mean()
    signal = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
    return df['close'].__class__(signal, index=df.index)
"""


@pytest.fixture(autouse=True)
def _isolated_experiment_db(tmp_path, monkeypatch):
    """Every iteration records into app.ai.experiment_memory -- isolate
    the database so these tests can't pollute (or be polluted by) real
    experiment history."""
    from app.ai import experiment_memory
    monkeypatch.setattr(experiment_memory, "_db_path", lambda: tmp_path / "experiments.db")
    yield
    _RESEARCH_LOOP_JOBS.clear()


@pytest.fixture(autouse=True)
def _mock_ollama(monkeypatch):
    monkeypatch.setattr(rl, "generate_strategy", lambda *a, **k: GenerationResult(code=_SMA_CROSS_CODE))
    monkeypatch.setattr(rl, "_ask_ollama_next_hypothesis", lambda *a, **k: ("next idea", False))


def _poll_until(client, job_id: str, predicate, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    data = None
    while time.time() < deadline:
        r = client.get(f"/research-loop/job/{job_id}/status.json")
        data = r.get_json()
        assert data["found"] is True
        if predicate(data):
            return data
        time.sleep(0.2)
    raise AssertionError(f"research loop job {job_id} never satisfied the predicate within {timeout}s; last={data}")


def test_research_loop_form_loads():
    client = app.test_client()
    r = client.get("/research-loop")
    assert r.status_code == 200
    assert b"Research loop" in r.data


def test_research_loop_bounded_run_end_to_end():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "n_iterations": "1", "keep_score_threshold": "999", "mc_sims": "20", "survival_sims": "50",
        }
        r = client.post("/research-loop/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    status_page = client.get(f"/research-loop/job/{job_id}")
    assert status_page.status_code == 200

    data = _poll_until(client, job_id, lambda d: not d["running"])
    assert data["stopped_reason"] == "completed"
    assert data["n_iterations_run"] == 1
    assert len(data["iterations"]) == 1
    # "code" is deliberately stripped from the poll payload to keep it bounded.
    assert "code" not in data["iterations"][0]


def test_research_loop_unbounded_run_can_be_stopped():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "n_iterations": "", "keep_score_threshold": "999", "mc_sims": "20", "survival_sims": "50",
        }
        r = client.post("/research-loop/start", data=data, content_type="multipart/form-data")
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    # Let it run a little, confirm it's genuinely unbounded (still running
    # after at least one iteration), then stop it.
    _poll_until(client, job_id, lambda d: d["n_iterations_run"] >= 1)
    assert client.get(f"/research-loop/job/{job_id}/status.json").get_json()["running"] is True

    stop_resp = client.post(f"/research-loop/job/{job_id}/stop")
    assert stop_resp.status_code in (302, 200)
    final = client.get(f"/research-loop/job/{job_id}/status.json").get_json()
    assert final["running"] is False
    assert final["stopped_reason"] == "cancelled"


def test_research_loop_missing_job_404s():
    client = app.test_client()
    assert client.get("/research-loop/job/does-not-exist").status_code == 404
    assert client.get("/research-loop/job/does-not-exist/status.json").status_code == 404


def test_research_loop_requires_a_dataset():
    client = app.test_client()
    r = client.post("/research-loop/start", data={"n_iterations": "1"}, content_type="multipart/form-data")
    assert r.status_code == 400
