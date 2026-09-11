import io
import json

import numpy as np
import pandas as pd

from app.web.server import app
from app.strategy.library import save_strategy_text, delete_saved_strategy


def _synthetic_csv_bytes(n=3000, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 1.10 + np.cumsum(rng.normal(0, 0.0006, n))
    df = pd.DataFrame({
        "timestamp": idx,
        "open": close + rng.normal(0, 0.0001, n),
        "high": close + np.abs(rng.normal(0, 0.0004, n)),
        "low": close - np.abs(rng.normal(0, 0.0004, n)),
        "close": close,
        "volume": rng.integers(100, 1000, n),
    })
    return df.to_csv(index=False).encode()


def _save_test_strategy(name="pytest_research_smoke.json"):
    config = {
        "entry_conditions": {
            "long": [
                {"left": {"type": "time_of_day", "session_start": "08:30", "session_end": "11:00"}, "operator": "==", "right": 1},
                {"left": {"type": "rsi", "period": 14}, "operator": "<", "right": 30},
            ],
            "long_connectors": ["AND"],
            "short": [],
        },
        "exit_conditions": {"long": [], "short": []},
        "stop_loss_pips": 20,
        "take_profit_pips": 30,
    }
    save_strategy_text(json.dumps(config), name, "manual", overwrite=True)
    return name


def test_research_form_loads():
    client = app.test_client()
    r = client.get("/research")
    assert r.status_code == 200
    assert b"Research Director" in r.data


def test_forge_form_has_pip_detect_wiring():
    """Regression guard for the missing-pip-size-detect bug: forge.html
    must expose the same detect button/field/JS hook evolution.html
    already has, not a bare hardcoded 0.0001 field."""
    client = app.test_client()
    r = client.get("/forge")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "forgeDetectPipSize" in body
    assert 'id="forge_pip_size_field"' in body
    assert "/data/detect-pip-size" in body


def test_research_run_requires_strategy_selection():
    client = app.test_client()
    data = {"csv_file": (io.BytesIO(_synthetic_csv_bytes()), "smoke.csv")}
    r = client.post("/research/run", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert b"strategy" in r.data.lower()


def test_research_run_end_to_end():
    strategy_name = _save_test_strategy()
    try:
        client = app.test_client()
        data = {
            "csv_file": (io.BytesIO(_synthetic_csv_bytes()), "smoke.csv"),
            "strategy_file": strategy_name,
            "account_size": "100000", "pip_size": "0.0001", "risk_value": "1.0",
            "profit_target": "8", "daily_loss": "5", "max_dd": "10", "window_trading_days": "20",
            "run_decomposition": "on", "run_ablation": "on", "run_null": "on",
            "run_degradation": "on", "run_contribution": "on", "run_conditional": "on", "run_regime": "on",
        }
        r = client.post("/research/run", data=data, content_type="multipart/form-data")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert "class=\"error\"" not in body
        for heading in (
            "Edge Decomposition", "Ablation Testing", "Null Strategy Benchmark",
            "Execution Fragility Score", "Trade Contribution Analysis",
            "Conditional Expectancy Maps", "Regime Discovery",
        ):
            assert heading in body, f"missing section: {heading}"
    finally:
        try:
            delete_saved_strategy("manual", strategy_name)
        except Exception:
            pass


def test_research_run_bad_dataset_shows_error_not_crash():
    client = app.test_client()
    data = {
        "csv_file": (io.BytesIO(b"not,a,valid,csv\n1,2"), "bad.csv"),
        "strategy_file": "does_not_exist.json",
    }
    r = client.post("/research/run", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert b"Research Director" in r.data
