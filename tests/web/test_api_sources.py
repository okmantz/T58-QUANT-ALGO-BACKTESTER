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
    assert res["rows"] == 5 and res["full_name"] == "AAPL/AAPL_1m_tiingo_2025-01-02_2025-01-03.csv"
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


# ---------------------------------------------------------------- "use data from a provider" picks (tokens)

def _bars(n=60):
    return [{"date": f"2025-01-02T14:{i:02d}:00.000Z", "open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i, "volume": 10}
            for i in range(n)]


def test_token_round_trip():
    t = api_sources.make_token("alpaca", " AAPL ", "5m", "2025-01-02", "2025-02-01")
    assert t == "__api__:alpaca|AAPL|5m|2025-01-02|2025-02-01" and api_sources.is_token(t)
    assert api_sources.parse_token(t) == {"provider": "alpaca", "symbol": "AAPL", "timeframe": "5m", "start": "2025-01-02", "end": "2025-02-01"}
    assert api_sources.parse_token("__api__:alpaca") is None          # provider picked, no instrument yet
    assert api_sources.parse_token("EURUSD/EURUSD5.csv") is None and not api_sources.is_token("EURUSD/EURUSD5.csv")


def test_repeat_download_reuses_the_saved_file(monkeypatch):
    monkeypatch.setattr("app.accounts.api_keys.load_settings", lambda: _keys(tiingo_api_key="k"))
    calls = []

    def http(url, params):
        calls.append(url)
        return _bars()

    a = api_sources.fetch_to_library("tiingo", "aapl", "1m", "2025-01-02", "2025-01-03", http_get=http)
    b = api_sources.fetch_to_library("tiingo", "aapl", "1m", "2025-01-02", "2025-01-03", http_get=http)
    assert len(calls) == 1 and not a["cached"] and b["cached"] and a["full_name"] == b["full_name"]
    c = api_sources.fetch_to_library("tiingo", "aapl", "1m", "2025-01-02", "2025-01-03", http_get=http, reuse_seconds=0)
    assert len(calls) == 2 and not c["cached"]                         # reuse can be switched off
    d = api_sources.fetch_to_library("tiingo", "aapl", "1m", "2025-01-02", "2025-01-10", http_get=http)
    assert d["full_name"] != a["full_name"] and len(calls) == 3        # a different range is a different file


def test_storage_resolves_a_provider_pick(monkeypatch, lib):
    monkeypatch.setattr("app.accounts.api_keys.load_settings", lambda: _keys(tiingo_api_key="k"))
    monkeypatch.setattr(free_sources, "_default_get", lambda url, params: _bars())
    from app.data.storage import resolve_stored_dataset
    path = resolve_stored_dataset("__api__:tiingo|AAPL|1m|2025-01-02|2025-01-03")
    assert path is not None and path.is_file() and path.parent.name == "AAPL"
    assert resolve_stored_dataset("__api__:tiingo") is None            # no instrument typed yet
    assert resolve_stored_dataset("__api__:polygon|AAPL|5m||") is None  # no key saved -> "no such dataset", never a crash


def test_run_form_accepts_a_provider_pick(monkeypatch):
    """The Detect-pip-size call (and every run that bypasses the page script) sends the pick itself; the server downloads
    the data and answers as if that saved file had been chosen."""
    monkeypatch.setattr("app.accounts.api_keys.load_settings", lambda: _keys(tiingo_api_key="k"))
    monkeypatch.setattr(free_sources, "_default_get", lambda url, params: _bars())
    c = app.test_client()
    r = c.post("/data/detect-pip-size", data={"existing_dataset": "__api__:tiingo|AAPL|1m|2025-01-02|2025-01-03"})
    assert r.status_code == 200 and r.get_json().get("pip_size") is not None
    bad = c.post("/data/detect-pip-size", data={"existing_dataset": "__api__:polygon|AAPL|5m||"})
    assert bad.status_code == 400 and "data provider" in bad.get_json()["error"] and "Settings" in bad.get_json()["error"]


def test_provider_picker_script_covers_every_kind_of_picker():
    js = open("app/web/static/api-source.js", encoding="utf-8").read()
    for needle in ('select[name$="existing_dataset"]', 'select[name="dataset_label"]', 'input[type="checkbox"][name="datasets"]',
                   'select[multiple][name="ds_datasets"]', "requestSubmit"):
        assert needle in js, needle
