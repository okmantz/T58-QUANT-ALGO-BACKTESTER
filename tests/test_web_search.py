"""
End-to-end tests for the Search Lab web routes (app.web.server).

These actually POST to /search/start, poll the real background job through
/search/job/<id>/status.json until it finishes, and exercise the promote
endpoint -- not mocks of the job system, the real threading.Thread +
run_search() path a browser would drive. Kept small-scale (few candidates,
low Monte Carlo sim counts, walk-forward/robustness disabled) so the whole
file finishes in a reasonable time under CI.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

import app.web.server as server_module
from app.web.server import _SEARCH_JOBS, app
from app.orchestration.resource_guard import HEAVY_JOB_GUARD, JOB_EVOLUTION_LAB, JOB_SEARCH_LAB

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

_PYTHON_SRC = '''STRATEGY_NAME = "Web Test EMA Cross"
EMA_FAST = 6
EMA_SLOW = 18
STOP_LOSS_PIPS = 18
TAKE_PROFIT_PIPS = 36

def generate_signals(df):
    fast = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    slow = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    return (fast > slow).astype(int) - (fast < slow).astype(int)
'''

_MQL5_SRC = '''void OnTick() {
   double fastMA = iMA(_Symbol, PERIOD_CURRENT, 6, 0, MODE_EMA, PRICE_CLOSE);
   double slowMA = iMA(_Symbol, PERIOD_CURRENT, 18, 0, MODE_EMA, PRICE_CLOSE);
   if (fastMA > slowMA) { trade.Buy(0.1, _Symbol); }
   if (fastMA < slowMA) { trade.Sell(0.1, _Symbol); }
   // T58_SL_PIPS=18
   // T58_TP_PIPS=36
}
'''

_LOOSE_STAGE_FIELDS = {
    "max_candidates": "8", "seed": "1", "workers": "2",
    "min_trades": "1", "min_profit_factor": "0.0", "stage1_top_n": "5",
    "ga_population": "4", "ga_generations": "1", "stage2_top_n": "3",
    "full_mc_sims": "60", "walk_forward_folds": "0", "robustness_neighbors": "0",
    "fitness_metric": "composite_prop_score",
}


@pytest.fixture(autouse=True)
def _cleanup_search_artifacts(tmp_path, monkeypatch):
    """
    Isolate every test in this file from the real reports/search/
    directory. SEARCH_DIR resolves to <repo_root>/reports/search in normal
    (non-frozen) runs -- that's where a real user's past search-run
    databases and leaderboard reports live. Redirecting SEARCH_DIR to a
    pytest tmp_path for the duration of each test (rather than rmtree-ing
    the real one, which was this file's original approach) means these
    tests can freely create and destroy search artifacts with zero risk to
    a real user's search history.
    """
    isolated_search_dir = tmp_path / "search"
    isolated_search_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(server_module, "SEARCH_DIR", isolated_search_dir)
    yield
    _SEARCH_JOBS.clear()


def _poll_until_done(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/search/job/{job_id}/status.json")
        data = r.get_json()
        assert data["found"] is True
        if data["done"]:
            return data
        time.sleep(0.3)
    raise AssertionError(f"search job {job_id} did not finish within {timeout}s")


def test_search_form_loads():
    client = app.test_client()
    r = client.get("/search")
    assert r.status_code == 200
    assert b"Search Lab" in r.data


def test_search_job_status_404_for_unknown_job():
    client = app.test_client()
    r = client.get("/search/job/does-not-exist/status.json")
    assert r.status_code == 404
    assert r.get_json()["found"] is False


def test_search_job_page_404_for_unknown_job():
    client = app.test_client()
    r = client.get("/search/job/does-not-exist")
    assert r.status_code == 404


def test_search_named_family_manual_end_to_end():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "search_mode": "family_named", "family": "trend_breakout",
            **_LOOSE_STAGE_FIELDS,
        }
        r = client.post("/search/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302  # redirected to the job page
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    job_page = client.get(f"/search/job/{job_id}")
    assert job_page.status_code == 200
    assert b"Search running" in job_page.data or b"Search complete" in job_page.data

    status = _poll_until_done(client, job_id)
    assert status["error"] is None
    assert status["summary"]["mode"] == "family"
    assert status["summary"]["family"] == "trend_breakout"
    assert status["summary"]["total_candidates"] == 8
    assert isinstance(status["leaderboard"], list)

    if status["summary"]["champion_candidate_id"]:
        candidate_id = status["summary"]["champion_candidate_id"]
        promo = client.post(f"/search/job/{job_id}/promote", data={"candidate_id": candidate_id})
        assert promo.status_code == 200
        promo_data = promo.get_json()
        assert promo_data["ok"] is True
        report_url = promo_data["report_html"]
        report_resp = client.get(report_url)
        assert report_resp.status_code == 200


def test_search_loop_mode_end_to_end():
    """Real POST to /search/start with loop_mode=on, driving the actual
    background thread (_run_search_loop_job -> run_search_loop ->
    run_search, real threads, not mocks) through to completion, then
    checks the loop-specific fields on status.json."""
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "search_mode": "family_named", "family": "trend_breakout",
            "loop_mode": "on",
            "loop_target_eval_pass_pct": "0",  # trivially easy -- any passer clears it
            "loop_max_rounds": "2",
            "loop_stall_rounds": "1",
            **_LOOSE_STAGE_FIELDS,
        }
        r = client.post("/search/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    status = _poll_until_done(client, job_id)
    assert status["error"] is None
    assert status["loop_mode"] is True
    assert status["loop_rounds"] >= 1
    assert status["loop_last_round"] is not None
    assert status["loop_result"] is not None
    assert status["loop_result"]["stopped_reason"] in {"target_reached", "max_rounds"}
    # The status page's leaderboard/summary rendering (unchanged from the
    # non-loop path) must still be populated with the LATEST round's data.
    assert status["summary"] is not None
    assert isinstance(status["leaderboard"], list)


def test_search_loop_mode_can_be_stopped(monkeypatch):
    """The existing /search/job/<id>/stop button must also work for a
    loop-mode job -- it should cancel the loop between rounds via the same
    cancel_event a normal job uses."""
    import app.orchestration.loop_runner as loop_runner_module

    real_run_search = loop_runner_module.run_search

    def _slow_run_search(*args, **kwargs):
        cancel_event = kwargs.get("cancel_event")
        # Give the test time to call /stop before this round's run_search
        # call itself checks cancellation internally.
        time.sleep(0.5)
        return real_run_search(*args, **kwargs)

    monkeypatch.setattr(loop_runner_module, "run_search", _slow_run_search)

    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "search_mode": "family_named", "family": "trend_breakout",
            "loop_mode": "on",
            "loop_target_eval_pass_pct": "99.9",  # unreachable -- keep looping until stopped
            "loop_max_rounds": "50",
            "loop_stall_rounds": "1",
            **_LOOSE_STAGE_FIELDS,
        }
        r = client.post("/search/start", data=data, content_type="multipart/form-data")
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    client.post(f"/search/job/{job_id}/stop")
    status = _poll_until_done(client, job_id, timeout=30.0)
    assert status["cancelled"] is True
    assert status["loop_result"]["stopped_reason"] == "cancelled"


def test_search_family_grid_python_end_to_end():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "search_mode": "family_grid", "strategy_mode": "python",
            "strategy_code": _PYTHON_SRC, "grid_points": "2",
            **_LOOSE_STAGE_FIELDS,
        }
        r = client.post("/search/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    status = _poll_until_done(client, job_id)
    assert status["error"] is None
    assert status["summary"]["family"] == "python_grid"
    for row in status["leaderboard"]:
        assert row["source_type"] == "python"


def test_search_single_mode_mql5_end_to_end():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "search_mode": "single", "strategy_mode": "mql5",
            "strategy_code": _MQL5_SRC,
            **_LOOSE_STAGE_FIELDS,
        }
        r = client.post("/search/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    status = _poll_until_done(client, job_id)
    assert status["error"] is None
    assert status["summary"]["mode"] == "single"
    assert status["summary"]["total_candidates"] == 1


def test_search_start_with_no_dataset_shows_error():
    client = app.test_client()
    r = client.post(
        "/search/start",
        data={"search_mode": "family_named", "family": "trend_breakout", **_LOOSE_STAGE_FIELDS},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    assert b"Please upload at least one valid CSV" in r.data


def test_search_promote_requires_candidate_id():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "search_mode": "family_named", "family": "trend_breakout",
            **_LOOSE_STAGE_FIELDS,
        }
        r = client.post("/search/start", data=data, content_type="multipart/form-data")
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]
    _poll_until_done(client, job_id)
    promo = client.post(f"/search/job/{job_id}/promote", data={})
    assert promo.status_code == 400
    assert promo.get_json()["ok"] is False


class _FakeEvolutionRunner:
    """Minimal stand-in for EvolutionRunner -- only .is_running matters
    for HEAVY_JOB_GUARD's registered health check."""
    def __init__(self, is_running: bool):
        self.is_running = is_running


@pytest.fixture
def _isolated_heavy_job_guard(monkeypatch):
    """HEAVY_JOB_GUARD is a process-wide singleton shared with every other
    heavy-job route in the app (Evolution Lab, Full Pipeline, Speed Run,
    Search Lab). Force it clear before and after each test here so a
    failure in one of these tests can't leave the slot stuck for the rest
    of the suite."""
    HEAVY_JOB_GUARD.release(JOB_EVOLUTION_LAB)
    HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)
    yield
    HEAVY_JOB_GUARD.release(JOB_EVOLUTION_LAB)
    HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)
    monkeypatch.setattr(server_module, "_EVOLUTION_RUNNER", None, raising=False)


def test_search_blocked_while_evolution_lab_genuinely_running(monkeypatch, _isolated_heavy_job_guard):
    """End-to-end reproduction of the reported bug's second half: Evolution
    Lab holds the shared HEAVY_JOB_GUARD slot while genuinely still
    running -- Search Lab must be refused with a clear message (not fail
    silently, hang, or crash)."""
    monkeypatch.setattr(server_module, "_EVOLUTION_RUNNER", _FakeEvolutionRunner(is_running=True))
    assert HEAVY_JOB_GUARD.try_acquire(JOB_EVOLUTION_LAB)

    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "search_mode": "family_named", "family": "trend_breakout",
            **_LOOSE_STAGE_FIELDS,
        }
        r = client.post("/search/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 409
    assert b"Evolution Lab is already running" in r.data


def test_search_self_heals_after_evolution_lab_left_a_stale_guard_slot(monkeypatch, _isolated_heavy_job_guard):
    """Regression test for the actual reported bug: Owen's Evolution Lab
    run stopped progressing and showed RUNNING with a STOP button that did
    nothing; Search Lab then wouldn't start at all. Root cause traced to
    HEAVY_JOB_GUARD's Evolution Lab slot only ever being released by the
    web app's /evolution/status.json poll noticing is_running had gone
    False -- if the run loop never got there (wedged, or the page/tab
    stopped polling), the slot stayed held forever and every other heavy
    job -- Search Lab included -- was refused indefinitely with no
    recovery short of restarting the server.

    This simulates exactly that stale state (guard held for Evolution Lab,
    but the runner itself reports not running) and confirms Search Lab's
    own try_acquire now self-heals the slot via HeavyJobGuard's registered
    health check and runs normally end to end."""
    monkeypatch.setattr(server_module, "_EVOLUTION_RUNNER", _FakeEvolutionRunner(is_running=False))
    assert HEAVY_JOB_GUARD.try_acquire(JOB_EVOLUTION_LAB)  # simulate the stale hold

    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "search_mode": "family_named", "family": "trend_breakout",
            **_LOOSE_STAGE_FIELDS,
        }
        r = client.post("/search/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302  # redirected to the job status page -- it actually started
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    status = _poll_until_done(client, job_id)
    assert status["error"] is None
    assert HEAVY_JOB_GUARD.active_name is None  # Search Lab released its own slot on completion
