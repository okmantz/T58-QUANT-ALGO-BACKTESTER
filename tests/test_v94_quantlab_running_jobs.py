"""v9.4: Quant Lab tools can run on market data already in the app (the
stored-dataset dropdown) instead of forcing a fresh upload, and any
running job can be re-entered from the sidebar's "Running now" list."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PRISTINE = Path.home() / "workspace" / "tmp-pristine"

from app.web.job_manager import JobManager  # noqa: E402
from app.web.server import app as flask_app  # noqa: E402


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def _ohlcv_csv_bytes(n=400, seed=3) -> bytes:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    price = 4000 + np.cumsum(rng.normal(0, 1.0, n))
    df = pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.5,
        "low": price - 0.5, "close": price, "volume": 100.0,
    })
    return df.to_csv(index=False).encode()


@pytest.fixture
def stored_dataset():
    """A dataset sitting in data/raw/ exactly like a user-imported file."""
    from app.data.storage import get_raw_data_dir

    name = "ZZ_QA_V94.csv"
    path = get_raw_data_dir() / name
    path.write_bytes(_ohlcv_csv_bytes())
    try:
        yield name
    finally:
        path.unlink(missing_ok=True)


def _error_text(body: str):
    m = re.search(r'result-error">(.*?)</div>', body, re.S)
    return m.group(1) if m else None


def test_market_structure_accepts_stored_dataset(client, stored_dataset):
    r = client.post("/quant-lab/market-structure", data={"dataset_label": stored_dataset})
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert _error_text(body) is None
    assert 'class="card result-ok"' in body


def test_composite_signal_accepts_stored_dataset(client, stored_dataset):
    r = client.post("/quant-lab/composite-signal", data={
        "dataset_label": stored_dataset,
        "a_kind": "rsi", "a_period": "14", "a_type": "cross_level_below", "a_level": "30",
        "b_kind": "rsi", "b_period": "14", "b_type": "in_range", "b_level": "20", "b_level2": "80",
        "mode": "and",
    })
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert _error_text(body) is None
    assert 'class="card result-ok"' in body


def test_market_structure_without_data_explains_both_options(client):
    r = client.post("/quant-lab/market-structure", data={})
    assert r.status_code == 200
    assert "Choose a dataset from the dropdown" in r.get_data(as_text=True)


def test_quant_lab_forms_offer_stored_datasets(client, stored_dataset):
    for url in ("/quant-lab/market-structure", "/quant-lab/composite-signal"):
        body = client.get(url).get_data(as_text=True)
        assert 'name="dataset_label"' in body
        assert stored_dataset in body


def test_create_resolves_page_url_from_template():
    jm = JobManager()
    job_id = jm.create(tool="Quick Optimize", page_template="/quick-optimize/job/{job_id}", log=[])
    job = jm.get(job_id)
    assert job["page_url"] == f"/quick-optimize/job/{job_id}"
    assert job["tool"] == "Quick Optimize"


def test_running_jobs_endpoint_lists_live_jobs_only(client):
    from app.web.job_manager import JOB_MANAGER

    live_id = JOB_MANAGER.create(tool="Full Pipeline", page_template="/full-pipeline/job/{job_id}", log=[])
    done_id = JOB_MANAGER.create(tool="Full Pipeline", page_template="/full-pipeline/job/{job_id}", log=[])
    JOB_MANAGER.finish(done_id)
    try:
        body = client.get("/api/running-jobs.json").get_json()
        ids = {j["job_id"] for j in body["running"]}
        assert live_id in ids
        assert done_id not in ids
        entry = next(j for j in body["running"] if j["job_id"] == live_id)
        assert entry["page_url"] == f"/full-pipeline/job/{live_id}"
        assert entry["tool"] == "Full Pipeline"
    finally:
        JOB_MANAGER.finish(live_id)


def test_sidebar_contains_running_now_poller(client):
    body = client.get("/dashboard").get_data(as_text=True)
    assert 'id="t58-running-now"' in body
    assert "/api/running-jobs.json" in body


def run_against(tree: Path):
    probe = "tests/test_v94_quantlab_running_jobs.py"
    dest = tree / probe
    shutil.copy2(ROOT / probe, dest)
    try:
        return subprocess.run(
            [sys.executable, "-m", "pytest", probe, "-x", "-q",
             "-k", "not run_against and not fails_on_upstream"],
            cwd=tree, capture_output=True, text=True, timeout=600)
    finally:
        dest.unlink(missing_ok=True)


@pytest.mark.skipif(not PRISTINE.exists(), reason="pristine upstream copy not available")
def test_fails_on_upstream_v92():
    r = run_against(PRISTINE)
    assert r.returncode != 0, r.stdout[-1200:]
