"""
End-to-end tests for the Generate Strategies (AI) web routes
(app.web.server) after their migration from a hand-rolled
_GENSTRAT_JOBS dict onto the shared JOB_MANAGER (see
app.web.job_manager's module docstring). POSTs to
/generate-strategies/start and polls the real background job through
/generate-strategies/job/<id>/status.json -- only Ollama's own
generate_strategy call is mocked out (no real Ollama needed), same
mocking boundary tests/test_web_research_loop.py uses.
"""
from __future__ import annotations

import time

import pytest

import app.ai.strategy_generator as strategy_generator
from app.ai.strategy_generator import GenerationResult
from app.web.server import app

_SMA_CROSS_CODE = """
import numpy as np

def generate_signals(df):
    fast = df['close'].rolling(10).mean()
    slow = df['close'].rolling(30).mean()
    signal = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
    return df['close'].__class__(signal, index=df.index)
"""


@pytest.fixture(autouse=True)
def _mock_ollama(monkeypatch):
    monkeypatch.setattr(
        strategy_generator, "generate_strategy",
        lambda *a, **k: GenerationResult(code=_SMA_CROSS_CODE, filename_hint="sma_cross"),
    )


@pytest.fixture(autouse=True)
def _isolated_ollama_settings(tmp_path, monkeypatch):
    """save_ollama_settings persists to disk -- isolate it like every
    other web test that touches app-wide settings files."""
    import app.ai.ollama_settings as ollama_settings_mod
    monkeypatch.setattr(ollama_settings_mod, "_settings_path", lambda: tmp_path / "ollama_settings.json")


def _poll_until(client, job_id: str, predicate, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    status = {}
    while time.time() < deadline:
        resp = client.get(f"/generate-strategies/job/{job_id}/status.json")
        status = resp.get_json()
        if predicate(status):
            return status
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for job {job_id}; last status: {status}")


def test_generate_strategies_start_and_status_round_trip():
    client = app.test_client()
    resp = client.post("/generate-strategies/start", data={
        "idea": "Buy when a fast SMA crosses above a slow SMA.",
        "language": "python",
        "ai_host": "http://localhost:11434",
        "ai_model": "llama3.1",
    })
    assert resp.status_code == 302
    job_id = resp.headers["Location"].rstrip("/").split("/")[-1]

    # Job page itself renders (not a 404) while/after the job runs.
    job_page = client.get(f"/generate-strategies/job/{job_id}")
    assert job_page.status_code == 200

    status = _poll_until(client, job_id, lambda s: s.get("done") is True)
    assert status["found"] is True
    assert status["error"] is None
    assert status["code"] is not None
    assert "generate_signals" in status["code"]
    assert status["filename_hint"] == "sma_cross"
    assert status["language"] == "python"
    assert status["idea"] == "Buy when a fast SMA crosses above a slow SMA."


def test_generate_strategies_unknown_job_id_is_404():
    client = app.test_client()
    resp = client.get("/generate-strategies/job/does-not-exist/status.json")
    assert resp.status_code == 404
    assert resp.get_json() == {"found": False}

    page = client.get("/generate-strategies/job/does-not-exist")
    assert page.status_code == 404


def test_generate_strategies_requires_an_idea():
    client = app.test_client()
    resp = client.post("/generate-strategies/start", data={"idea": "   "})
    assert resp.status_code == 400
