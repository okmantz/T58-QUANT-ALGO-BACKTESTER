"""'Your idea' screen: page renders, status of unknown job is handled, and the worker runs an idea
end to end on a synthetic dataset and publishes result HTML."""
from __future__ import annotations

import numpy as np
import pandas as pd

import app.web.discover_routes as dr
from app.web.job_manager import JOB_MANAGER
from app.web.server import app


def _df(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-02", periods=n, freq="15min")
    close = 4000 + np.cumsum(rng.normal(0, 2.0, n))
    hi = close + rng.random(n) * 3
    lo = close - rng.random(n) * 3
    op = close + rng.normal(0, 0.5, n)
    return pd.DataFrame({"timestamp": ts, "open": op, "high": hi, "low": lo, "close": close, "volume": 1})


def test_form_renders_and_sidebar_links():
    c = app.test_client()
    r = c.get("/discover")
    assert r.status_code == 200 and b"Your idea" in r.data
    # v9.8: sidebar lifecycle sections are single links (Owen's spec), so
    # tool pages are reached from the section page's Individual tools grid.
    assert b"/discover" in c.get("/validate-simple").data  # Your Idea tool card


def test_unknown_job_status():
    r = app.test_client().get("/discover/status/nope")
    assert r.status_code == 404


def test_worker_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(dr, "_load_dataset", lambda name, max_bars: _df())
    import app.ai.experiment_memory as em
    import app.discovery.hypothesis as hy
    for mod, attr in ((em, "DB_PATH"), (hy, "STORE_PATH")):
        if hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, tmp_path / f"{attr}.x")
    jid = JOB_MANAGER.create(tool="t")
    dr._worker(jid, "buy when RSI(14) is below 30, stop 1 ATR, target 2 ATR", ["ES_test"], [], 4000, 10)
    j = JOB_MANAGER.get(jid)
    assert j["done"], j
    assert j["error"] is None, j["error"]
    assert "Grid" in j["html"]
