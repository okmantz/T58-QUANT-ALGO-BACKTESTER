"""
End-to-end tests for the Multi-Instrument Evolution Lab web routes
(app.web.server) -- POSTs to /evolution/multi-instrument/start, polls the
real background groups through
/evolution/multi-instrument/job/<id>/status.json, and exercises stop/
promote. Not mocks -- the real MultiInstrumentEvolutionGroup ->
EvolutionRunner path a browser would drive. Kept small-scale (tiny
population, max_generations=1) so this finishes quickly.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import app.web.server as server_module
from app.web.server import app
from app.orchestration.resource_guard import (
    HEAVY_JOB_GUARD, JOB_EVOLUTION_LAB, JOB_MULTI_INSTRUMENT_EVOLUTION,
)


def _trending_csv(path, seed=1, n=600):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.linspace(0, 40, n)
    noise = np.cumsum(rng.normal(0, 0.4, n))
    price = 1900 + drift + noise
    df = pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.3, "low": price - 0.3,
        "close": price, "volume": 100.0,
    })
    df.to_csv(path, index=False)
    return path


_FAST_FIELDS = {
    "population_size": "8", "elite_keep": "2", "mc_sims": "20", "max_generations": "1",
    "initial_balance": "100000",
}


@pytest.fixture
def _isolated_raw_data_dir(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(server_module, "get_raw_data_dir", lambda: raw_dir)
    return raw_dir


@pytest.fixture(autouse=True)
def _isolated_evolution_base_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("app.evolution.multi_instrument.get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr("app.search.family_health.get_app_base_dir", lambda: tmp_path / "family_health_base")
    yield
    HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_EVOLUTION)
    HEAVY_JOB_GUARD.release(JOB_EVOLUTION_LAB)
    monkeypatch.setattr(server_module, "_EVOLUTION_RUNNER", None, raising=False)


def _poll_until_stopped(client, group_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/evolution/multi-instrument/job/{group_id}/status.json")
        data = r.get_json()
        assert data["found"] is True
        if not data["running"]:
            return data
        time.sleep(0.3)
    raise AssertionError(f"multi-instrument evolution group {group_id} did not stop within {timeout}s")


def test_multi_instrument_evolution_form_loads():
    client = app.test_client()
    r = client.get("/evolution/multi-instrument")
    assert r.status_code == 200
    assert b"Multi-instrument evolution lab" in r.data


def test_multi_instrument_evolution_requires_at_least_two_datasets(_isolated_raw_data_dir):
    _trending_csv(_isolated_raw_data_dir / "a.csv", seed=1)
    client = app.test_client()
    r = client.post("/evolution/multi-instrument/start", data={"datasets": ["a.csv"], **_FAST_FIELDS})
    assert r.status_code == 400
    assert b"at least 2" in r.data


def test_multi_instrument_evolution_end_to_end_start_poll_stop(_isolated_raw_data_dir):
    _trending_csv(_isolated_raw_data_dir / "eurusd.csv", seed=1)
    _trending_csv(_isolated_raw_data_dir / "gbpusd.csv", seed=2)

    client = app.test_client()
    r = client.post(
        "/evolution/multi-instrument/start",
        data={"datasets": ["eurusd.csv", "gbpusd.csv"], **_FAST_FIELDS},
    )
    assert r.status_code == 302
    group_id = r.headers["Location"].rstrip("/").split("/")[-1]

    status_page = client.get(f"/evolution/multi-instrument/job/{group_id}")
    assert status_page.status_code == 200

    # max_generations=1 means each runner should stop on its own fairly
    # quickly -- confirms the group as a whole correctly reports "not
    # running" once every instrument's runner has finished, and that the
    # guard self-heals via the status.json poll (same pattern as
    # single-instrument Evolution Lab).
    data = _poll_until_stopped(client, group_id)
    assert set(data["labels"]) == {"eurusd/eurusd", "gbpusd/gbpusd"}
    assert HEAVY_JOB_GUARD.active_name is None


def test_multi_instrument_evolution_stop_route(_isolated_raw_data_dir):
    _trending_csv(_isolated_raw_data_dir / "a.csv", seed=1)
    _trending_csv(_isolated_raw_data_dir / "b.csv", seed=2)

    client = app.test_client()
    # No max_generations this time -- runs until stopped.
    fields = {k: v for k, v in _FAST_FIELDS.items() if k != "max_generations"}
    r = client.post("/evolution/multi-instrument/start", data={"datasets": ["a.csv", "b.csv"], **fields})
    group_id = r.headers["Location"].rstrip("/").split("/")[-1]

    stop_resp = client.post(f"/evolution/multi-instrument/job/{group_id}/stop")
    assert stop_resp.status_code in (302, 200)

    data = client.get(f"/evolution/multi-instrument/job/{group_id}/status.json").get_json()
    assert data["running"] is False
    assert HEAVY_JOB_GUARD.active_name is None


def test_multi_instrument_evolution_blocked_while_single_evolution_running(monkeypatch, _isolated_raw_data_dir):
    """Regression-style test using the same simulation technique as
    tests/test_web_search.py's analogous test: HEAVY_JOB_GUARD's health
    check for JOB_EVOLUTION_LAB looks at the real _EVOLUTION_RUNNER's
    is_running flag, so simulating 'genuinely running' means patching
    that global to a fake runner reporting is_running=True -- merely
    calling try_acquire() without it would trigger the guard's own
    (correct) self-heal, since nothing would back up the claim that
    Evolution Lab is actually still running."""
    _trending_csv(_isolated_raw_data_dir / "a.csv", seed=1)
    _trending_csv(_isolated_raw_data_dir / "b.csv", seed=2)

    class _FakeRunner:
        is_running = True

    monkeypatch.setattr(server_module, "_EVOLUTION_RUNNER", _FakeRunner())
    assert HEAVY_JOB_GUARD.try_acquire(JOB_EVOLUTION_LAB)

    client = app.test_client()
    r = client.post("/evolution/multi-instrument/start", data={"datasets": ["a.csv", "b.csv"], **_FAST_FIELDS})
    assert r.status_code == 409
    assert b"Evolution Lab is already running" in r.data
