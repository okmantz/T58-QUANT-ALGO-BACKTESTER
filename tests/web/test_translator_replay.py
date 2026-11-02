"""Universal Translator: hand-written Python (no directive) must export a trade replay instead of failing."""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import pytest

from app.strategy.library import delete_saved_strategy, save_strategy_text
from app.web.server import app as flask_app

HANDWRITTEN = '''
import numpy as np
import pandas as pd

def generate_signals(df):
    fast = df["close"].rolling(5).mean()
    slow = df["close"].rolling(20).mean()
    sig = pd.Series(0, index=df.index)
    sig[fast > slow] = 1
    sig[fast < slow] = -1
    atr = (df["high"] - df["low"]).rolling(14).mean()
    sig.attrs["stop_loss_distance"] = atr * 2
    sig.attrs["take_profit_distance"] = atr * 3
    return sig
'''


def _csv(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-02", periods=n, freq="15min")
    price = 1900 + np.cumsum(rng.normal(0, 0.8, n))
    df = pd.DataFrame({"timestamp": ts, "open": price, "high": price + 0.5, "low": price - 0.5, "close": price, "volume": 100.0})
    return df.to_csv(index=False).encode()


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


@pytest.fixture
def saved():
    name = "zz_replay_test_v1.py"
    save_strategy_text(HANDWRITTEN, name, "python", overwrite=True)
    yield name
    try:
        delete_saved_strategy("python", name)
    except Exception:
        pass


def _post(client, name, target, with_data=True):
    data = {"strategy_choice": f"python:{name}", "target": target, "data_tz": "UTC"}
    if with_data:
        data["data_csv"] = (io.BytesIO(_csv()), "d.csv")
    return client.post("/quant-lab/translator", data=data, content_type="multipart/form-data")


def test_handwritten_python_to_pine_replay(client, saved):
    r = _post(client, saved, "pinescript")
    body = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Trade replay" in body and "//@version=5" in body and "strategy.entry" in body
    assert "no T58_MANUAL_CONFIG_JSON" not in body


def test_handwritten_python_to_mql5_replay(client, saved):
    body = _post(client, saved, "mql5").get_data(as_text=True)
    assert "CTrade" in body and "OnTick" in body and "AddChunk" in body


def test_no_data_gives_actionable_message(client, saved):
    body = _post(client, saved, "pinescript", with_data=False).get_data(as_text=True)
    assert "trade replay" in body.lower() and "choose market data" in body.lower()


def test_replay_events_match_signal_changes():
    from app.strategy.python import PythonStrategy
    from app.strategy.replay_export import collect_replay, to_pinescript_replay
    import tempfile, os
    df = pd.read_csv(io.BytesIO(_csv()), parse_dates=["timestamp"])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(HANDWRITTEN)
    try:
        strat = PythonStrategy(f.name)
        rd = collect_replay(strat, df, "t")
        sig = strat.generate(df).signals.values
        assert len(rd.events) == int((np.diff(np.r_[0, sig]) != 0).sum())
        assert rd.events[0][1] in (1, -1) and rd.events[0][2] > 0
        pine = to_pinescript_replay(rd)
        assert all(len(m) < 4000 for m in re.findall(r'f_load\("([^"]*)"\)', pine))
    finally:
        os.unlink(f.name)
