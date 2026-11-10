"""/data-center/review + /data-center/fix: cards render, a fix writes a NEW file and leaves the original untouched."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.web import data_health_routes as routes
from app.web.server import app


@pytest.fixture
def client():
    return app.test_client()


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    # The importer already drops duplicate/out-of-order bars, so the file the page sees can only carry what survives import:
    # here a 60-point contract splice halfway through.
    rng = np.random.default_rng(1)
    days = [pd.date_range(d + pd.Timedelta(hours=1), periods=92, freq="15min") for d in pd.bdate_range("2024-02-01", periods=30)]
    ts = pd.DatetimeIndex(np.concatenate([x.values for x in days]))
    c = 4000 + np.cumsum(rng.normal(0, 1, len(ts)))
    df = pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1})
    df.loc[(len(ts) // 2 // 92) * 92:, ["open", "high", "low", "close"]] += 60.0
    df.to_csv(tmp_path / "ES_15m.csv", index=False)
    monkeypatch.setattr(routes, "resolve_stored_dataset", lambda name: tmp_path / name if (tmp_path / name).is_file() else None)
    saved = {}
    monkeypatch.setattr(routes, "store_dataset_bytes", lambda content, rel: saved.setdefault(rel, content))
    return tmp_path, saved


def test_review_shows_the_duplicate_card_with_a_fix_button(client, raw_dir):
    html = client.get("/data-center/review?name=ES_15m.csv").get_data(as_text=True)
    assert "contract-roll splice" in html and 'value="back_adjust"' in html


def test_fix_saves_a_new_file_and_never_touches_the_original(client, raw_dir):
    tmp, saved = raw_dir
    before = (tmp / "ES_15m.csv").read_bytes()
    resp = client.post("/data-center/fix", data={"name": "ES_15m.csv", "fix": "back_adjust"})
    assert resp.status_code == 302 and (tmp / "ES_15m.csv").read_bytes() == before
    assert list(saved) == ["ES_15m_back_adjust.csv"]
    assert len(pd.read_csv(__import__("io").BytesIO(saved["ES_15m_back_adjust.csv"]))) == 2760


def test_unknown_file_and_unknown_fix_are_handled(client, raw_dir):
    assert b"not in your data folder" in client.get("/data-center/review?name=nope.csv").data
    assert client.post("/data-center/fix", data={"name": "ES_15m.csv", "fix": "evil"}).status_code == 302
