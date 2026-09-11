from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from app.strategy.library import delete_saved_strategy, save_strategy_text
from app.web.server import app as flask_app

TEST_STRATEGY_NAME = "_test_hedge_fund_web_sma.py"
TEST_STRATEGY_CODE = """
import numpy as np

def generate_signals(df, config=None):
    fast = df["close"].rolling(5).mean()
    slow = df["close"].rolling(20).mean()
    sig = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
    return sig.tolist()
"""


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


@pytest.fixture
def saved_python_strategy():
    save_strategy_text(TEST_STRATEGY_CODE, TEST_STRATEGY_NAME, "python", overwrite=True)
    yield TEST_STRATEGY_NAME
    delete_saved_strategy("python", TEST_STRATEGY_NAME)


def _csv_bytes(n=400, seed=1, drift=0.0005):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="1D")
    price = 100.0
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.01)
        o = price
        c = o * (1 + step)
        rows.append((ts[i], o, max(o, c) * 1.001, min(o, c) * 0.999, c, 1000))
        price = c
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def test_dashboard_loads(client):
    resp = client.get("/hedge-fund/")
    assert resp.status_code == 200
    assert b"Hedge Fund Manager" in resp.data


def test_run_requires_at_least_two_assets(client):
    resp = client.post("/hedge-fund/run", data={"asset1_csv": (_csv_bytes(), "AAA.csv")}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert b"at least 2 assets" in resp.data


def test_run_with_two_bootstrap_assets_succeeds(client):
    data = {"asset1_csv": (_csv_bytes(seed=1, drift=0.0008), "AAA.csv"), "asset2_csv": (_csv_bytes(seed=2, drift=0.0002), "BBB.csv")}
    resp = client.post("/hedge-fund/run", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert 'class="error"' not in body
    assert "Total return" in body
    assert "<svg" in body


def test_saved_strategy_picker_drives_that_assets_view(client, saved_python_strategy):
    """The flagship integration: picking a saved strategy for one asset
    slot should make THAT asset's view method 'strategy_signal' while an
    asset left on the default stays 'bootstrap' -- proving the picker
    actually reaches app.hedge_fund.research.strategy_signal_view, not
    just that the form accepts the field."""
    data = {
        "asset1_csv": (_csv_bytes(seed=1, drift=0.001), "AAA.csv"),
        "asset1_library_mode": "python",
        "asset1_library_name": saved_python_strategy,
        "asset2_csv": (_csv_bytes(seed=2, drift=0.0002), "BBB.csv"),
        "lookback_bars": "150",
    }
    resp = client.post("/hedge-fund/run", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert 'class="error"' not in body
    assert "Using a saved strategy as the view generator for: AAA.csv" in body
    assert "<td>AAA.csv</td><td>" in body
    # AAA.csv's row in the last-cycle table should say strategy_signal; BBB.csv's should say bootstrap.
    aaa_row_start = body.index("<td>AAA.csv</td>")
    bbb_row_start = body.index("<td>BBB.csv</td>")
    assert "strategy_signal" in body[aaa_row_start:aaa_row_start + 300]
    assert "bootstrap" in body[bbb_row_start:bbb_row_start + 300]


def test_unknown_saved_strategy_name_errors_cleanly(client):
    data = {
        "asset1_csv": (_csv_bytes(seed=1), "AAA.csv"),
        "asset1_library_mode": "python",
        "asset1_library_name": "does_not_exist.py",
        "asset2_csv": (_csv_bytes(seed=2), "BBB.csv"),
    }
    resp = client.post("/hedge-fund/run", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert b"Could not load saved strategy" in resp.data
