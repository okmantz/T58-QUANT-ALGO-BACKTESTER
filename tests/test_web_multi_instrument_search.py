"""
End-to-end tests for the Multi-Instrument Search web routes
(app.web.server) -- POSTs to /search/multi-instrument/start, polls the
real background job through /search/multi-instrument/job/<id>/status.json
until it finishes, and confirms both per-instrument results and the guard/
error paths. Not mocks of the job system -- the real threading.Thread ->
app.orchestration.multi_instrument_search.run_multi_instrument_search path
a browser would drive. Kept small-scale so this finishes quickly.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import app.web.server as server_module
from app.web.server import app
from app.orchestration.resource_guard import HEAVY_JOB_GUARD, JOB_MULTI_INSTRUMENT_SEARCH, JOB_SEARCH_LAB


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
    "family": "trend_breakout", "max_candidates": "6", "seed": "1",
    "min_trades": "1", "min_profit_factor": "0.0", "stage1_top_n": "5",
    "ga_population": "4", "ga_generations": "1", "stage2_top_n": "3",
    "full_mc_sims": "50", "walk_forward_folds": "0", "robustness_neighbors": "0",
    "fitness_metric": "eval_pass_probability", "max_concurrent": "2",
}


@pytest.fixture
def _isolated_raw_data_dir(tmp_path, monkeypatch):
    """Isolates every test in this file from the real data/raw/ directory
    -- get_raw_data_dir() also seeds bundled example datasets and, on a
    real user's machine, may contain many previously-uploaded files;
    redirecting it to a controlled tmp_path with exactly the CSVs each
    test creates keeps these tests deterministic and side-effect-free."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(server_module, "get_raw_data_dir", lambda: raw_dir)
    return raw_dir


@pytest.fixture(autouse=True)
def _isolated_search_dir(tmp_path, monkeypatch):
    isolated = tmp_path / "search"
    isolated.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(server_module, "SEARCH_DIR", isolated)
    yield
    HEAVY_JOB_GUARD.release(JOB_MULTI_INSTRUMENT_SEARCH)
    HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)


def _poll_until_done(client, job_id: str, timeout: float = 90.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/search/multi-instrument/job/{job_id}/status.json")
        data = r.get_json()
        assert data["found"] is True
        if data["done"] or data["error"]:
            return data
        time.sleep(0.3)
    raise AssertionError(f"multi-instrument job {job_id} did not finish within {timeout}s")


def test_multi_instrument_form_loads():
    client = app.test_client()
    r = client.get("/search/multi-instrument")
    assert r.status_code == 200
    assert b"Multi-instrument search" in r.data


def test_multi_instrument_start_requires_at_least_two_datasets(_isolated_raw_data_dir):
    csv_a = _trending_csv(_isolated_raw_data_dir / "a.csv", seed=1)
    client = app.test_client()
    r = client.post("/search/multi-instrument/start", data={"datasets": ["a.csv"], **_FAST_FIELDS})
    assert r.status_code == 400
    assert b"at least 2" in r.data


def test_multi_instrument_search_end_to_end(_isolated_raw_data_dir):
    _trending_csv(_isolated_raw_data_dir / "eurusd.csv", seed=1)
    _trending_csv(_isolated_raw_data_dir / "gbpusd.csv", seed=2)

    client = app.test_client()
    r = client.post(
        "/search/multi-instrument/start",
        data={"datasets": ["eurusd.csv", "gbpusd.csv"], **_FAST_FIELDS},
    )
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    status_page = client.get(f"/search/multi-instrument/job/{job_id}")
    assert status_page.status_code == 200

    data = _poll_until_done(client, job_id)
    assert data["error"] is None
    assert set(data["labels"]) == {"eurusd/eurusd", "gbpusd/gbpusd"}
    assert set(data["results"].keys()) == set(data["labels"])
    for label, res in data["results"].items():
        assert res["error"] is None
        assert res["total_candidates"] == 6

    # HEAVY_JOB_GUARD must be released once the job finishes.
    assert HEAVY_JOB_GUARD.active_name is None


def test_multi_instrument_search_blocked_while_search_lab_running(_isolated_raw_data_dir):
    _trending_csv(_isolated_raw_data_dir / "a.csv", seed=1)
    _trending_csv(_isolated_raw_data_dir / "b.csv", seed=2)
    assert HEAVY_JOB_GUARD.try_acquire(JOB_SEARCH_LAB)

    client = app.test_client()
    r = client.post(
        "/search/multi-instrument/start",
        data={"datasets": ["a.csv", "b.csv"], **_FAST_FIELDS},
    )
    assert r.status_code == 409
    assert b"Search Lab is already running" in r.data
