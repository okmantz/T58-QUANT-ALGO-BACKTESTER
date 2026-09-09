"""
End-to-end tests for the Multi-Instrument Speed Run web routes
(app.web.server) -- POSTs to /speed-run/multi-instrument/start, polls the
real background job through
/speed-run/multi-instrument/job/<id>/status.json until it finishes. Not
mocks of the job system -- the real threading.Thread ->
app.orchestration.multi_instrument_speed_run.run_multi_instrument_speed_run
path a browser would drive. Kept small-scale so this finishes quickly.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import app.web.server as server_module
from app.web.server import app
from app.orchestration.resource_guard import (
    HEAVY_JOB_GUARD, JOB_MULTI_INSTRUMENT_SPEED_RUN, JOB_SPEED_RUN,
)


def _trending_csv(path, seed=1, n=1200):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = 0.00015 + rng.normal(0, 0.00003)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00002))
        l = min(o, c) - abs(rng.normal(0, 0.00002))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.to_csv(path, index=False)
    return path


_FAST_FIELDS = {
    "max_candidates": "6", "stage1_top_n": "2", "ga_population": "4", "ga_generations": "1",
    "top_k_to_validate": "1", "max_concurrent_validations": "1",
    "validation_folds": "1", "validation_final_mc_sims": "50",
    "fitness_metric": "eval_pass_probability", "random_seed": "1",
    "max_concurrent_instruments": "2",
}


@pytest.fixture
def _isolated_raw_data_dir(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(server_module, "get_raw_data_dir", lambda: raw_dir)
    return raw_dir


@pytest.fixture(autouse=True)
def _isolated_speedrun_dir(tmp_path, monkeypatch):
    isolated = tmp_path / "speed_run" / "multi_instrument"
    monkeypatch.setattr(server_module, "MULTI_SPEEDRUN_DIR", isolated)
    yield
    HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SPEED_RUN)
    HEAVY_JOB_GUARD.release(JOB_SPEED_RUN)


def _poll_until_done(client, job_id: str, timeout: float = 90.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/speed-run/multi-instrument/job/{job_id}/status.json")
        data = r.get_json()
        assert data["found"] is True
        if data["done"] or data["error"]:
            return data
        time.sleep(0.3)
    raise AssertionError(f"multi-instrument speed run job {job_id} did not finish within {timeout}s")


def test_multi_instrument_speed_run_form_loads():
    client = app.test_client()
    r = client.get("/speed-run/multi-instrument")
    assert r.status_code == 200
    assert b"Multi-instrument speed run" in r.data


def test_multi_instrument_speed_run_requires_at_least_two_datasets(_isolated_raw_data_dir):
    _trending_csv(_isolated_raw_data_dir / "a.csv", seed=1)
    client = app.test_client()
    r = client.post("/speed-run/multi-instrument/start", data={"datasets": ["a.csv"], **_FAST_FIELDS})
    assert r.status_code == 400
    assert b"at least 2" in r.data


def test_multi_instrument_speed_run_end_to_end(_isolated_raw_data_dir):
    _trending_csv(_isolated_raw_data_dir / "eurusd.csv", seed=1)
    _trending_csv(_isolated_raw_data_dir / "gbpusd.csv", seed=2)

    client = app.test_client()
    r = client.post(
        "/speed-run/multi-instrument/start",
        data={"datasets": ["eurusd.csv", "gbpusd.csv"], **_FAST_FIELDS},
    )
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    status_page = client.get(f"/speed-run/multi-instrument/job/{job_id}")
    assert status_page.status_code == 200

    data = _poll_until_done(client, job_id)
    assert data["error"] is None
    assert set(data["labels"]) == {"eurusd/eurusd", "gbpusd/gbpusd"}
    assert set(data["results"].keys()) == set(data["labels"])
    assert HEAVY_JOB_GUARD.active_name is None


def test_multi_instrument_speed_run_blocked_while_speed_run_running(_isolated_raw_data_dir):
    _trending_csv(_isolated_raw_data_dir / "a.csv", seed=1)
    _trending_csv(_isolated_raw_data_dir / "b.csv", seed=2)
    assert HEAVY_JOB_GUARD.try_acquire(JOB_SPEED_RUN)

    client = app.test_client()
    r = client.post(
        "/speed-run/multi-instrument/start",
        data={"datasets": ["a.csv", "b.csv"], **_FAST_FIELDS},
    )
    assert r.status_code == 409
    assert b"Speed Run is already running" in r.data
