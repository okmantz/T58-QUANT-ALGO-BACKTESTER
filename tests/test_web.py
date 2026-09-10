from pathlib import Path

from app.web.server import app, REPORTS_DIR
from app.data import storage


def test_index_loads():
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"T58" in r.data


def test_manifest_served():
    client = app.test_client()
    r = client.get("/manifest.json")
    assert r.status_code == 200
    assert r.content_type == "application/manifest+json"


def test_resources_page_loads():
    """The Resources tab is a static page (no engine dependency) linking
    out to the T58 30-Day Trading Quickstart Guide plus a few other free
    beginner resources -- just confirms it renders and links to the real
    guide, not that any particular external site is reachable."""
    client = app.test_client()
    r = client.get("/resources")
    assert r.status_code == 200
    assert b"30-Day Trading Quickstart Guide" in r.data
    assert b"docs.google.com/document/d/17tdRY_tzpHOdw8_a1EpAgOmradTbu02Dlo6BOAnK0Wg" in r.data


def test_resources_page_links_mslsd_strategy_doc_instead_of_pasting_it():
    """The MS-LSD strategy write-up used to be pasted in full on this page
    -- now it's a clean link-out to the real doc (kept in sync there
    instead of drifting from a stale copy pasted here)."""
    client = app.test_client()
    r = client.get("/resources")
    assert r.status_code == 200
    assert b"docs.google.com/document/d/14jubETVbumncdLJrTrw2ke-NxFPXkT34qzH4_Q1BguA" in r.data
    assert b"Open the MS-LSD strategy doc" in r.data
    # The old pasted walkthrough paragraphs should be gone.
    assert b"A valid OB needs" not in r.data


def test_resources_link_appears_in_sidebar():
    client = app.test_client()
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert b'href="/resources"' in r.data


def test_user_manual_page_loads_with_decision_guide():
    """The User Manual used to be a purely linear walkthrough with no
    branching logic. This confirms the new decision-guide section (real
    "if this happened, do that" branches, not just numbered steps) is
    present alongside the original step-by-step content."""
    client = app.test_client()
    r = client.get("/user-manual")
    assert r.status_code == 200
    assert b"Decision guide" in r.data
    assert b"Zero trades generated" in r.data
    assert b"Evolution Lab winners pass on its own leaderboard" in r.data


def test_mobile_access_page_loads_without_tailscale(monkeypatch):
    """On a box with no Tailscale installed (the normal CI/dev sandbox),
    the page must still render cleanly with setup instructions -- never
    500 just because the optional Tailscale card has nothing to show."""
    monkeypatch.setattr("app.web.server.tailscale_url", lambda: None)
    client = app.test_client()
    r = client.get("/mobile-access")
    assert r.status_code == 200
    assert b"tailscale.com/download" in r.data


def test_mobile_access_page_shows_tailscale_address_when_present(monkeypatch):
    monkeypatch.setattr("app.web.server.tailscale_url", lambda: "http://100.64.0.1:5000")
    client = app.test_client()
    r = client.get("/mobile-access")
    assert r.status_code == 200
    assert b"100.64.0.1" in r.data


def test_full_pipeline_via_manual_strategy(tmp_path):
    client = app.test_client()
    sample_csv = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

    with open(sample_csv, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "strategy_mode": "manual",
            "sma_fast": "20", "sma_slow": "50", "sl_pips": "20", "tp_pips": "40",
            "account_size": "100000", "profit_target": "8", "daily_loss": "5", "max_dd": "10",
            "dd_type": "trailing", "consistency": "30", "min_days": "5", "payout_freq": "14",
            "payout_threshold": "0", "buffer": "0", "payout_cap": "",
            "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
            "max_trades_day": "10", "commission": "0", "slippage_pips": "0.5",
            "spread_pips": "1.0", "pip_size": "0.0001",
            "n_sims": "100", "mc_method": "bootstrap",
        }
        r = client.post("/run", data=data, content_type="multipart/form-data")

    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Eval Pass Probability" in body

    # clean up any report files this test produced
    for f in REPORTS_DIR.glob("report_*"):
        f.unlink()


def test_full_pipeline_accepts_eod_drawdown_check_mode(tmp_path):
    """
    A firm documented as EOD-drawdown (e.g. FundedNext, Lucid Trading --
    see prop-algo-backtester research notes) must be selectable end-to-end
    through the web form, not just via a script. dd_check_mode=eod should
    run cleanly through the same pipeline as the default intrabar mode.
    """
    client = app.test_client()
    sample_csv = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

    with open(sample_csv, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "strategy_mode": "manual",
            "sma_fast": "20", "sma_slow": "50", "sl_pips": "20", "tp_pips": "40",
            "account_size": "100000", "profit_target": "8", "daily_loss": "5", "max_dd": "10",
            "dd_type": "static", "dd_check_mode": "eod", "consistency": "40", "min_days": "1",
            "payout_freq": "5",
            "payout_threshold": "0", "buffer": "0", "payout_cap": "",
            "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
            "max_trades_day": "10", "commission": "0", "slippage_pips": "0.5",
            "spread_pips": "1.0", "pip_size": "0.0001",
            "n_sims": "100", "mc_method": "bootstrap",
        }
        r = client.post("/run", data=data, content_type="multipart/form-data")

    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Eval Pass Probability" in body

    for f in REPORTS_DIR.glob("report_*"):
        f.unlink()


def test_missing_csv_returns_error():
    client = app.test_client()
    r = client.post("/run", data={"strategy_mode": "manual"}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert b"CSV" in r.data or b"dataset" in r.data


def test_detect_pip_size_against_uploaded_csv():
    """Web counterpart to the desktop's DETECT PIP SIZE FROM DATA button
    (see app.ui.main_window._detect_pip_size_from_data): given a freshly
    uploaded CSV (not yet a stored dataset), the endpoint should return a
    suggested pip_size derived from the data's own price scale rather than
    requiring a backtest run first."""
    client = app.test_client()
    import io
    r = client.post(
        "/data/detect-pip-size",
        data={"csv_file": (io.BytesIO(_make_bigger_csv(7)), "eurusd.csv")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["pip_size"] == 0.0001  # FX-scale prices (~1.10) in _make_bigger_csv
    assert "eurusd.csv" in payload["message"]


def test_detect_pip_size_against_stored_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "get_app_base_dir", lambda: tmp_path)
    storage.store_csv_bytes(_make_bigger_csv(9), "stored_fx.csv")
    client = app.test_client()
    r = client.post("/data/detect-pip-size", data={"existing_dataset": "stored_fx.csv"})
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["pip_size"] == 0.0001
    assert "stored_fx.csv" in payload["message"]


def test_detect_pip_size_with_nothing_selected_returns_error():
    client = app.test_client()
    r = client.post("/data/detect-pip-size", data={})
    assert r.status_code == 400
    assert "error" in r.get_json()


_BIGGER_CSV = None


def _make_bigger_csv(seed: int) -> bytes:
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    n = 200
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    df = pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.0005, "low": price - 0.0005,
        "close": price, "volume": 100.0,
    })
    return df.to_csv(index=False).encode()


def test_multi_file_upload_stores_all_and_uses_last_as_active(tmp_path, monkeypatch):
    # CRITICAL: get_raw_data_dir() resolves to <repo_root>/data/raw in normal
    # (non-frozen) runs -- the SAME directory a real user's stored
    # market-data CSVs live in. This test used to shutil.rmtree() that real
    # directory directly (before AND in `finally`), which means simply
    # running this test permanently deleted a user's actual uploaded
    # datasets. Redirecting get_app_base_dir() to a pytest tmp_path gives
    # this test its own throwaway data/raw/ with zero risk to the real one.
    monkeypatch.setattr(storage, "get_app_base_dir", lambda: tmp_path)
    raw_dir = storage.get_raw_data_dir()
    import io
    client = app.test_client()
    csv_a = _make_bigger_csv(1)
    csv_b = _make_bigger_csv(2)
    data = {
        "csv_file": [(io.BytesIO(csv_a), "setA.csv"), (io.BytesIO(csv_b), "setB.csv")],
        "strategy_mode": "manual", "sma_fast": "5", "sma_slow": "15", "sl_pips": "20", "tp_pips": "40",
        "account_size": "100000", "profit_target": "8", "daily_loss": "5", "max_dd": "10",
        "dd_type": "trailing", "consistency": "30", "min_days": "5", "payout_freq": "14",
        "payout_threshold": "0", "buffer": "0", "payout_cap": "",
        "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
        "max_trades_day": "10", "commission": "0", "slippage_pips": "0.5",
        "spread_pips": "1.0", "pip_size": "0.0001",
        "n_sims": "50", "mc_method": "bootstrap",
    }
    r = client.post("/run", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    stored_names = sorted(p.name for p in raw_dir.glob("*.csv"))
    assert stored_names == ["setA.csv", "setB.csv"]
    body = r.get_data(as_text=True)
    assert "setB.csv" in body  # most recently uploaded file becomes active
    for f in REPORTS_DIR.glob("report_*"):
        f.unlink()


def test_run_against_existing_stored_dataset_without_new_upload(tmp_path, monkeypatch):
    # See the CRITICAL note in test_multi_file_upload_stores_all_and_uses_last_as_active
    # -- same real-data-loss hazard, same fix.
    monkeypatch.setattr(storage, "get_app_base_dir", lambda: tmp_path)
    storage.store_csv_bytes(_make_bigger_csv(3), "stored.csv")
    client = app.test_client()
    data = {
        "existing_dataset": "stored.csv",
        "strategy_mode": "manual", "sma_fast": "5", "sma_slow": "15", "sl_pips": "20", "tp_pips": "40",
        "account_size": "100000", "profit_target": "8", "daily_loss": "5", "max_dd": "10",
        "dd_type": "trailing", "consistency": "30", "min_days": "5", "payout_freq": "14",
        "payout_threshold": "0", "buffer": "0", "payout_cap": "",
        "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
        "max_trades_day": "10", "commission": "0", "slippage_pips": "0.5",
        "spread_pips": "1.0", "pip_size": "0.0001",
        "n_sims": "50", "mc_method": "bootstrap",
    }
    r = client.post("/run", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert "stored.csv" in r.get_data(as_text=True)
    for f in REPORTS_DIR.glob("report_*"):
        f.unlink()


_LEAKY_PYTHON_STRATEGY = '''
import pandas as pd

def generate_signals(df, config=None):
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    h1 = x.set_index("timestamp").resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    out = pd.Series(0, index=x.index, dtype="int8")
    for i in range(60, len(x)):
        ts = x["timestamp"].iloc[i]
        h1c = h1[h1.index < ts]  # BUG: includes the still-forming current-hour bar
        if len(h1c) < 2:
            continue
        if h1c["close"].iloc[-1] > h1c["close"].iloc[-2]:
            out.iloc[i] = 1
        elif h1c["close"].iloc[-1] < h1c["close"].iloc[-2]:
            out.iloc[i] = -1
    return out
'''


def _make_leaky_strategy_csv() -> bytes:
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(11)
    n = 1500
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    price = 1900 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.5, "low": price - 0.5,
        "close": price, "volume": 100.0,
    })
    return df.to_csv(index=False).encode()


def test_python_strategy_with_lookahead_bug_shows_warning_banner():
    import io
    client = app.test_client()
    data = {
        "csv_file": (io.BytesIO(_make_leaky_strategy_csv()), "leaky.csv"),
        "strategy_mode": "python",
        "strategy_code": _LEAKY_PYTHON_STRATEGY,
        "account_size": "100000", "profit_target": "8", "daily_loss": "5", "max_dd": "10",
        "dd_type": "trailing", "consistency": "30", "min_days": "5", "payout_freq": "14",
        "payout_threshold": "0", "buffer": "0", "payout_cap": "",
        "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
        "max_trades_day": "10", "commission": "0", "slippage_pips": "0.5",
        "spread_pips": "1.0", "pip_size": "0.0001",
        "n_sims": "50", "mc_method": "bootstrap",
    }
    try:
        r = client.post("/run", data=data, content_type="multipart/form-data")
        body = r.get_data(as_text=True)
        assert r.status_code in (200, 400)
        if r.status_code == 200:
            assert "LOOKAHEAD BIAS DETECTED" in body
    finally:
        for f in REPORTS_DIR.glob("report_*"):
            f.unlink()


def test_view_saved_strategy_code_route(tmp_path, monkeypatch):
    from app.strategy import library

    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    library.save_strategy_text("STOP_LOSS_PIPS = 20\n", "view_me.py", "python")

    client = app.test_client()
    r = client.get("/strategies/view-code?strategy_type=python&filename=view_me.py")
    assert r.status_code == 200
    assert "STOP_LOSS_PIPS" in r.get_data(as_text=True)


def test_view_saved_strategy_code_route_missing_file(tmp_path, monkeypatch):
    from app.strategy import library

    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    client = app.test_client()
    r = client.get("/strategies/view-code?strategy_type=python&filename=nope.py")
    assert r.status_code == 404


def test_batch_test_route_produces_one_report_per_checked_strategy(tmp_path, monkeypatch):
    from app.strategy import library

    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    src = (
        "STRATEGY_NAME = \"Web Batch Test\"\nEMA_FAST = 5\nEMA_SLOW = 15\n"
        "STOP_LOSS_PIPS = 20\nTAKE_PROFIT_PIPS = 40\n\n"
        "def generate_signals(df):\n"
        "    fast = df[\"close\"].ewm(span=EMA_FAST, adjust=False).mean()\n"
        "    slow = df[\"close\"].ewm(span=EMA_SLOW, adjust=False).mean()\n"
        "    return (fast > slow).astype(int) - (fast < slow).astype(int)\n"
    )
    library.save_strategy_text(src, "web_batch_a.py", "python")
    library.save_strategy_text(src, "web_batch_b.py", "python")

    client = app.test_client()
    sample_csv = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"
    with open(sample_csv, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "batch_items": ["python::web_batch_a.py", "python::web_batch_b.py"],
            "account_size": "100000", "profit_target": "8", "daily_loss": "5", "max_dd": "10",
            "dd_type": "trailing", "consistency": "30", "min_days": "5", "payout_freq": "14",
            "payout_threshold": "0", "buffer": "0", "payout_cap": "",
            "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
            "max_trades_day": "10", "commission": "0", "slippage_pips": "0.5",
            "spread_pips": "1.0", "pip_size": "0.0001",
            "n_sims": "50", "mc_method": "bootstrap",
        }
        r = client.post("/strategies/batch-test", data=data, content_type="multipart/form-data")

    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Batch test results" in body
    assert "web_batch_a.py" in body and "web_batch_b.py" in body

    for f in REPORTS_DIR.glob("webbatch_*"):
        f.unlink()


def test_search_lab_stop_button_cancels_running_job():
    """Regression test for a real gap: the Search Lab web job never had a
    stop endpoint at all (Evolution Lab and the Research Loop did), even
    though app.search.batch_runner.run_search already supports a
    cancel_event. Starts a small real Search Lab run, immediately hits the
    new /search/job/<id>/stop route, and confirms the job actually stops
    (cancelled, not left running or reported as a crash)."""
    import time as _time

    client = app.test_client()
    sample_csv = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

    with open(sample_csv, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "search_mode": "family_named", "family": "all",
            "seed": "42", "max_candidates": "10",
            "min_trades": "5", "min_profit_factor": "1.0", "stage1_top_n": "5",
            "ga_population": "4", "ga_generations": "1", "stage2_top_n": "2",
            "full_mc_sims": "50", "walk_forward_folds": "2", "robustness_neighbors": "2",
            "fitness_metric": "eval_pass_probability",
            "account_size": "100000", "profit_target": "8", "daily_loss": "5", "max_dd": "10",
            "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
            "pip_size": "0.0001",
        }
        r = client.post("/search/start", data=data, content_type="multipart/form-data")

    assert r.status_code == 302
    job_url = r.headers["Location"]
    job_id = job_url.rstrip("/").split("/")[-1]

    # Hit the new stop route right away.
    stop_r = client.post(f"/search/job/{job_id}/stop")
    assert stop_r.status_code == 200
    assert stop_r.get_json()["ok"] is True

    deadline = _time.time() + 20
    status = {}
    while _time.time() < deadline:
        status = client.get(f"/search/job/{job_id}/status.json").get_json()
        if status.get("done"):
            break
        _time.sleep(0.5)

    assert status.get("done") is True
    assert status.get("cancelled") is True
    assert not status.get("error")

    from app.orchestration.resource_guard import HEAVY_JOB_GUARD, JOB_SEARCH_LAB
    HEAVY_JOB_GUARD.release(JOB_SEARCH_LAB)  # tidy up regardless of guard state at exit


def test_search_job_stop_route_on_unknown_job_returns_404():
    client = app.test_client()
    r = client.post("/search/job/does-not-exist/stop")
    assert r.status_code == 404


def test_full_pipeline_batch_route_runs_multiple_library_strategies():
    """Regression test for a real gap: the web app's Full Pipeline only
    ever ran one strategy at a time, while the desktop app has had a
    'RUN FULL PIPELINE (BATCH)' button (backed by
    app.orchestration.full_pipeline.run_full_pipeline_batch) for a while.
    Saves two manual strategies to the library, submits both to the new
    /full-pipeline/start-batch route, and confirms the batch job actually
    runs both and reports a per-strategy outcome for each."""
    import json as _json
    import time as _time

    from app.strategy.library import save_strategy_text

    manual_cfg = {
        "name": "batch-test-strategy",
        "indicators": [
            {"type": "sma", "period": 10, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 30, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }
    names = []
    for i in range(2):
        name = f"web_batch_test_strategy_{i}.json"
        save_strategy_text(_json.dumps(manual_cfg), name, "manual", overwrite=True)
        names.append(name)

    client = app.test_client()
    sample_csv = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

    try:
        with open(sample_csv, "rb") as f:
            data = {
                "csv_file": (f, "EURUSD_5M_sample.csv"),
                "batch_items": [f"manual::{n}" for n in names],
                "n_folds": "2", "window_mode": "rolling",
                "ga_population": "4", "ga_generations": "1", "ga_search_mc_sims": "20",
                "final_mc_sims": "50", "baseline_mc_sims": "20",
                "holdout_frac": "0.2", "oos_check_folds": "2", "random_seed": "42",
                "account_size": "100000", "profit_target": "8", "daily_loss": "5", "max_dd": "10",
                "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
                "max_trades_day": "10", "commission": "0", "slippage_pips": "0.5",
                "spread_pips": "1.0", "pip_size": "0.0001",
                "save_to_library": "off", "parallel_search": "off",
            }
            r = client.post(
                "/full-pipeline/start-batch", data=data,
                content_type="multipart/form-data",
            )

        assert r.status_code == 302
        job_id = r.headers["Location"].rstrip("/").split("/")[-1]

        deadline = _time.time() + 90
        status = {}
        while _time.time() < deadline:
            status = client.get(f"/full-pipeline/batch-job/{job_id}/status.json").get_json()
            if status.get("done"):
                break
            _time.sleep(1.0)

        assert status.get("done") is True
        assert not status.get("error")
        outcomes = status.get("outcomes") or []
        assert len(outcomes) == 2
        assert {o["label"] for o in outcomes} == set(names)
    finally:
        from app.strategy.library import delete_saved_strategy
        for n in names:
            try:
                delete_saved_strategy("manual", n)
            except Exception:
                pass
        from app.orchestration.resource_guard import HEAVY_JOB_GUARD, JOB_FULL_PIPELINE
        HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)


def test_full_pipeline_start_batch_with_no_selection_shows_error():
    client = app.test_client()
    sample_csv = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"
    with open(sample_csv, "rb") as f:
        data = {"csv_file": (f, "EURUSD_5M_sample.csv")}
        r = client.post("/full-pipeline/start-batch", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert b"No strategies were selected" in r.data

    from app.orchestration.resource_guard import HEAVY_JOB_GUARD, JOB_FULL_PIPELINE
    HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)


def test_full_pipeline_batch_job_can_be_stopped_mid_run():
    """Regression test for a real gap: there was no way at all to stop a
    running Full Pipeline batch job once started (see
    app.orchestration.full_pipeline.run_full_pipeline_batch's cancel_event /
    FullPipelineBatchCancelled, and the new /full-pipeline/batch-job/<id>/stop
    route). Starts a 3-strategy batch, hits stop immediately, and confirms
    the job reports 'cancelled' rather than either hanging or silently
    finishing the whole batch anyway."""
    import json as _json
    import time as _time

    from app.strategy.library import delete_saved_strategy, save_strategy_text

    manual_cfg = {
        "name": "stop-test-strategy",
        "indicators": [
            {"type": "sma", "period": 10, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 30, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow", "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow", "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20, "take_profit_pips": 40,
    }
    names = []
    for i in range(3):
        name = f"web_batch_stop_test_strategy_{i}.json"
        save_strategy_text(_json.dumps(manual_cfg), name, "manual", overwrite=True)
        names.append(name)

    client = app.test_client()
    sample_csv = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

    try:
        with open(sample_csv, "rb") as f:
            data = {
                "csv_file": (f, "EURUSD_5M_sample.csv"),
                "batch_items": [f"manual::{n}" for n in names],
                "n_folds": "2", "window_mode": "rolling",
                "ga_population": "4", "ga_generations": "1", "ga_search_mc_sims": "20",
                "final_mc_sims": "50", "baseline_mc_sims": "20",
                "holdout_frac": "0.2", "oos_check_folds": "2", "random_seed": "42",
                "account_size": "100000", "profit_target": "8", "daily_loss": "5", "max_dd": "10",
                "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
                "max_trades_day": "10", "commission": "0", "slippage_pips": "0.5",
                "spread_pips": "1.0", "pip_size": "0.0001",
                "save_to_library": "off", "parallel_search": "off",
            }
            r = client.post("/full-pipeline/start-batch", data=data, content_type="multipart/form-data")
        assert r.status_code == 302
        job_id = r.headers["Location"].rstrip("/").split("/")[-1]

        stop_r = client.post(f"/full-pipeline/batch-job/{job_id}/stop")
        assert stop_r.status_code == 200
        assert stop_r.get_json()["ok"] is True

        deadline = _time.time() + 90
        status = {}
        while _time.time() < deadline:
            status = client.get(f"/full-pipeline/batch-job/{job_id}/status.json").get_json()
            if status.get("done"):
                break
            _time.sleep(0.5)

        assert status.get("done") is True
        assert status.get("cancelled") is True
        # Stopping this early (right after the job starts) should mean the
        # batch never got through all 3 -- either 0 or 1 outcomes recorded,
        # never all 3, which would indicate stop had no real effect.
        assert len(status.get("outcomes") or []) < 3

        # Stopping an already-finished job is a no-op, not an error.
        again = client.post(f"/full-pipeline/batch-job/{job_id}/stop")
        assert again.status_code == 200
        assert again.get_json().get("already_done") is True
    finally:
        for n in names:
            try:
                delete_saved_strategy("manual", n)
            except Exception:
                pass
        from app.orchestration.resource_guard import HEAVY_JOB_GUARD, JOB_FULL_PIPELINE
        HEAVY_JOB_GUARD.release(JOB_FULL_PIPELINE)


def test_full_pipeline_batch_stop_route_404_for_unknown_job():
    client = app.test_client()
    r = client.post("/full-pipeline/batch-job/does-not-exist/stop")
    assert r.status_code == 404
    assert r.get_json()["ok"] is False


# ---------------------------------------------------------------------------
# Evolution Lab PROMOTE overfitting-gap guard -- regression coverage for a
# real report: strategies promoted at a raw "40% pass / 30% payout" coming
# back 2%/1% out of Full Pipeline. Everything upstream of
# app.evolution.engine.EvolutionRunner._cpcv_and_pbo scores each candidate
# against the SAME data the GA searched against, so a big gap between the
# raw in-sample number and the honest, held-out cpcv_oos_eval_pass_probability
# is the early warning sign -- this was computed and even logged, but never
# surfaced anywhere PROMOTE itself would show it before saving.
# ---------------------------------------------------------------------------

def _fake_checkpoint_with_candidate(record: dict):
    class _FakeCheckpoint:
        leaderboard = [record]
    return _FakeCheckpoint()


def test_evolution_promote_requires_confirmation_on_large_overfitting_gap(monkeypatch, tmp_path):
    from app.evolution import checkpoint as evo_checkpoint
    from app.strategy import library

    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    record = {
        "candidate_id": "gap-test-0001",
        "spec": {"config": {"name": "Gap Test", "market": {"instrument": "XAUUSD"}}},
        "meta": {"family": "test_family"},
        "mc_summary": {"evaluation_pass_probability": 40.0, "first_payout_probability": 30.0},
        "cpcv_oos_eval_pass_probability": 2.0,  # 38-point gap -- well over the 15-point threshold
    }
    monkeypatch.setattr(
        evo_checkpoint, "load_checkpoint", lambda *a, **k: _fake_checkpoint_with_candidate(record),
    )

    client = app.test_client()
    r = client.post("/evolution/promote", data={"candidate_id": "gap-test-0001"})
    assert r.status_code == 409
    payload = r.get_json()
    assert payload["ok"] is False
    assert payload["needs_confirmation"] is True
    assert payload["raw_pass_probability"] == 40.0
    assert payload["cpcv_oos_eval_pass_probability"] == 2.0

    # No file should have been saved yet -- the gate must block the save,
    # not just warn after the fact.
    assert not any(
        i.name.startswith("evolab_promoted_test_family_")
        for i in library.list_saved_strategies("manual")
    )

    # force=1 (the client sends this only after the user confirms the
    # confirm() dialog) bypasses the gate and actually saves.
    r2 = client.post("/evolution/promote", data={"candidate_id": "gap-test-0001", "force": "1"})
    assert r2.status_code == 200
    assert r2.get_json()["ok"] is True
    assert any(
        i.name.startswith("evolab_promoted_test_family_")
        for i in library.list_saved_strategies("manual")
    )


def test_evolution_promote_no_confirmation_needed_when_gap_is_small(monkeypatch, tmp_path):
    from app.evolution import checkpoint as evo_checkpoint
    from app.strategy import library

    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    record = {
        "candidate_id": "small-gap-0001",
        "spec": {"config": {"name": "Small Gap", "market": {"instrument": "XAUUSD"}}},
        "meta": {"family": "test_family_2"},
        "mc_summary": {"evaluation_pass_probability": 40.0, "first_payout_probability": 30.0},
        "cpcv_oos_eval_pass_probability": 32.0,  # only an 8-point gap -- under the threshold
    }
    monkeypatch.setattr(
        evo_checkpoint, "load_checkpoint", lambda *a, **k: _fake_checkpoint_with_candidate(record),
    )

    client = app.test_client()
    r = client.post("/evolution/promote", data={"candidate_id": "small-gap-0001"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


