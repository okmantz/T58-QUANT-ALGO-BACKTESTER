"""Unified market-data provider dropdown: providers, fetch-to-library, routes, page wiring."""
from __future__ import annotations

import pandas as pd
import pytest

from app.data import api_sources, free_sources
from app.web.server import app


@pytest.fixture(autouse=True)
def lib(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr("app.data.storage.get_raw_data_dir", lambda: raw)
    return raw


def _keys(**kw):
    from app.accounts.api_keys import ApiKeysSettings
    return ApiKeysSettings(**kw)


def test_provider_list_flags_missing_keys(monkeypatch):
    monkeypatch.setattr("app.accounts.api_keys.load_settings", lambda: _keys(tiingo_api_key="k"))
    monkeypatch.setattr("app.data.alpaca_credentials.load_credentials", lambda: None)
    got = {p["id"]: p for p in api_sources.provider_list()}
    assert got["tiingo"]["ready"] and not got["polygon"]["ready"] and got["dukascopy"]["ready"]
    assert "1m" in got["tiingo"]["timeframes"]


def test_tiingo_intraday_and_save(monkeypatch, lib):
    monkeypatch.setattr("app.accounts.api_keys.load_settings", lambda: _keys(tiingo_api_key="k"))
    seen = {}

    def http(url, params):
        seen["url"], seen["params"] = url, params
        return [{"date": f"2025-01-02T14:3{i}:00.000Z", "open": 1 + i, "high": 2 + i, "low": 0.5, "close": 1.5 + i, "volume": 10}
                for i in range(5)]

    res = api_sources.fetch_to_library("tiingo", "aapl", "1m", "2025-01-02", "2025-01-03", http_get=http)
    assert seen["params"]["resampleFreq"] == "1min" and "/iex/aapl/" in seen["url"]
    assert res["rows"] == 5 and res["full_name"] == "AAPL/AAPL_1m_tiingo.csv"
    df = pd.read_csv(lib / res["full_name"])
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_alpha_vantage_and_databento_parse(monkeypatch):
    monkeypatch.setattr("app.accounts.api_keys.load_settings",
                        lambda: _keys(alpha_vantage_api_key="k", databento_api_key="d"))
    av = {"Time Series (5min)": {"2025-01-02 10:05:00": {"1. open": "2", "2. high": "3", "3. low": "1", "4. close": "2.5", "5. volume": "9"},
                                  "2025-01-02 10:00:00": {"1. open": "1", "2. high": "2", "3. low": "1", "4. close": "2", "5. volume": "8"}}}
    df = api_sources.fetch("alpha_vantage", "IBM", "5m", "2025-01-01", "2025-01-03", http_get=lambda u, p: av)
    assert len(df) == 2 and df["timestamp"].is_monotonic_increasing
    rows = ["ts_event,open,high,low,close,volume"] + [f"2025-01-02T14:{m:02d}:00.000000000Z,{100+m},{101+m},{99+m},{100.5+m},5" for m in range(0, 10)]
    db = api_sources.fetch("databento", "ES.c.0", "5m", "2025-01-02", "2025-01-03", http_get=lambda u, p: "\n".join(rows))
    assert len(db) == 2 and db["volume"].iloc[0] == 25


def test_missing_key_and_bad_input(monkeypatch):
    monkeypatch.setattr("app.accounts.api_keys.load_settings", lambda: _keys())
    with pytest.raises(free_sources.DataSourceError, match="Settings"):
        api_sources.fetch("polygon", "AAPL", "5m")
    with pytest.raises(free_sources.DataSourceError):
        api_sources.fetch("tiingo", "", "5m")
    with pytest.raises(free_sources.DataSourceError):
        api_sources.fetch("nope", "X", "5m")


def test_routes_and_script_on_pages(monkeypatch):
    monkeypatch.setattr("app.accounts.api_keys.load_settings", lambda: _keys())
    c = app.test_client()
    j = c.get("/data/api/providers").get_json()
    assert {p["id"] for p in j["providers"]} >= {"alpaca", "lse", "tiingo", "dukascopy"}
    assert j["timeframes"][0] == "1m"
    bad = c.post("/data/api/fetch", json={"provider": "polygon", "symbol": "AAPL", "timeframe": "5m"})
    assert bad.status_code == 400 and "Settings" in bad.get_json()["error"]
    js = c.get("/static/api-source.js")
    assert js.status_code == 200 and b"__api__:" in js.data


def test_every_dataset_page_loads_the_script():
    import glob
    import re
    for f in glob.glob("app/web/templates/*.html"):
        s = open(f, encoding="utf-8").read()
        if re.search(r'<select[^>]*name="[^"]*existing_dataset"', s) and not f.endswith("_pip_size_autodetect.html"):
            assert "_sidebar.html" in s or "api-source.js" in s, f
    assert "api-source.js" in open("app/web/templates/_sidebar.html", encoding="utf-8").read()
