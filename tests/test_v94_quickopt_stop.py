"""v9.4: Quick Optimize's Stop button must actually stop the run.

Pre-fix, the cancel event was only checked between GA generations --
but one generation evaluates a whole population (backtest + Monte Carlo
per candidate, in a process pool that waited for EVERY future), so Stop
looked dead for minutes. Now evaluate_batch drains on the first
completed candidate after the event is set, the status payload reports
`stopping` while it drains, and the job finishes as cancelled.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRISTINE = Path.home() / "workspace" / "tmp-pristine"

from app.web.server import app  # noqa: E402

SAMPLE_CSV = ROOT / "data" / "examples" / "EURUSD_5M_sample.csv"


def _start(client, **overrides):
    fields = {
        "strategy_mode": "manual",
        # Big enough that the run cannot finish before Stop lands.
        "ga_population": "32", "ga_generations": "12", "n_folds": "4",
        "initial_balance": "50000", "account_size": "50000",
        "save_to_library": "",
    }
    fields.update(overrides)
    with open(SAMPLE_CSV, "rb") as f:
        r = client.post(
            "/quick-optimize/start",
            data={"csv_file": (f, "EURUSD_5M_sample.csv"), **fields},
            content_type="multipart/form-data",
        )
    assert r.status_code == 302, r.get_data(as_text=True)[:400]
    return r.headers["Location"].rstrip("/").split("/")[-1]


def test_stop_actually_cancels_a_running_job():
    client = app.test_client()
    job_id = _start(client)
    resp = client.post(f"/quick-optimize/job/{job_id}/stop")
    assert resp.status_code == 200
    assert resp.get_json()["found"] is True

    deadline = time.time() + 120
    data = None
    while time.time() < deadline:
        data = client.get(f"/quick-optimize/job/{job_id}/status.json").get_json()
        if data["done"]:
            break
        time.sleep(0.25)
    assert data is not None and data["done"], "job never finished after Stop"
    assert data["cancelled"] is True
    assert data["error"] is None


def test_status_reports_stopping_while_draining():
    client = app.test_client()
    job_id = _start(client)
    client.post(f"/quick-optimize/job/{job_id}/stop")
    # Immediately after Stop the flag is visible even before the runner
    # has drained -- this is what the page uses to say "Stopping...".
    data = client.get(f"/quick-optimize/job/{job_id}/status.json").get_json()
    assert "stopping" in data
    assert data["stopping"] is True or data["done"]


def test_ga_refinement_honours_a_preset_cancel_event():
    """The refinement itself must bail on the first candidate batch when
    the event is already set -- no full generation of work first."""
    import numpy as np
    import pandas as pd

    from app.backtest.risk import RiskConfig
    from app.monte_carlo.engine import MonteCarloConfig
    from app.optimize.walkforward_ga import (
        RefinementConfig,
        WalkforwardGACancelled,
        run_walkforward_aware_refinement,
    )
    from app.prop.simulator import PropRules
    from app.strategy.manual import ManualStrategy

    rng = np.random.default_rng(3)
    ts = pd.date_range("2024-01-01", periods=1200, freq="5min")
    price = 1.1000
    rows = []
    for i in range(1200):
        step = 0.00015 + rng.normal(0, 0.00006)
        o = price
        c = o + step
        rows.append((ts[i], o, max(o, c), min(o, c), c, 100.0))
        price = c
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    strategy = ManualStrategy({
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    })
    event = threading.Event()
    event.set()
    with pytest.raises(WalkforwardGACancelled):
        run_walkforward_aware_refinement(
            df, strategy, RiskConfig(), PropRules(),
            MonteCarloConfig(n_simulations=10),
            refinement_config=RefinementConfig(population_size=4, generations=3),
            n_folds=2, cancel_event=event,
        )


def _run_against(tree: Path):
    probe = "tests/test_v94_quickopt_stop.py"
    dest = tree / probe
    shutil.copy2(ROOT / probe, dest)
    try:
        return subprocess.run(
            [sys.executable, "-m", "pytest", probe, "-x", "-q",
             "-k", "not fails_on_upstream and not _run_against"],
            cwd=tree, capture_output=True, text=True, timeout=900)
    finally:
        dest.unlink(missing_ok=True)


@pytest.mark.skipif(not PRISTINE.exists(), reason="pristine upstream copy not available")
def test_fails_on_upstream_v92():
    r = _run_against(PRISTINE)
    assert r.returncode != 0, r.stdout[-1200:]
