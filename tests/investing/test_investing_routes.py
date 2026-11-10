import pandas as pd
import numpy as np

from app.investing import market_data as md


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("T58_INVEST_DIR", str(tmp_path))
    from app.web.server import app
    app.config["TESTING"] = True
    return app.test_client()


def _fake_hist(sym, range_="5y", provider=None):
    n = 800
    rng = np.random.default_rng(abs(hash(sym)) % 1000)
    return pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))}, index=pd.bdate_range("2021-01-01", periods=n))


def test_page_and_profile(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.get("/investing")
    assert r.status_code == 200 and b"Investing" in r.data and b"Build my plan" in r.data
    j = c.post("/investing/api/profile", json={"answers": {"risk": "hold", "horizon": "y3_10", "goal": "growth", "max_loss": "20"}}).get_json()
    assert j["ok"] and j["profile"]["label"] in ("Moderate", "Growth", "Aggressive")


def test_paper_order_flow(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(md, "history", _fake_hist)
    assert c.post("/investing/api/paper/reset", json={"cash": 5000}).get_json()["ok"]
    o = c.post("/investing/api/order", json={"mode": "paper", "symbol": "AAPL", "side": "buy", "dollars": 1000}).get_json()
    assert o["ok"]
    a = c.get("/investing/api/account?mode=paper").get_json()["account"]
    assert a["positions"][0]["symbol"] == "AAPL" and abs(a["cash"] - 4000) < 1e-6


def test_live_requires_confirmation(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(md, "history", _fake_hist)
    r = c.post("/investing/api/order", json={"mode": "live", "symbol": "AAPL", "side": "buy", "dollars": 10}).get_json()
    assert not r["ok"] and "confirmation" in r["error"]


def test_sec_agent_validation(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    assert c.post("/investing/api/sec-agent", json={"user_agent": "x"}).status_code == 400


def test_close_position_in_one_click(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(md, "history", _fake_hist)
    assert c.post("/investing/api/paper/reset", json={"cash": 5000}).get_json()["ok"]
    assert c.post("/investing/api/order", json={"mode": "paper", "symbol": "AAPL", "side": "buy", "dollars": 1000}).get_json()["ok"]
    half = c.post("/investing/api/close", json={"mode": "paper", "symbol": "aapl", "fraction": 0.5}).get_json()
    assert half["ok"] and half["order"]["side"] == "sell"
    left = c.get("/investing/api/account?mode=paper").get_json()["account"]["positions"]
    assert len(left) == 1 and abs(left[0]["market_value"] - 500) < 1
    full = c.post("/investing/api/close", json={"mode": "paper", "symbol": "AAPL"}).get_json()
    assert full["ok"] and full["order"]["note"] == "close position"
    acct = c.get("/investing/api/account?mode=paper").get_json()["account"]
    assert acct["positions"] == [] and abs(acct["cash"] - 5000) < 1e-6


def test_close_rejects_unknown_holding_and_bad_fraction(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(md, "history", _fake_hist)
    c.post("/investing/api/paper/reset", json={"cash": 1000})
    assert not c.post("/investing/api/close", json={"mode": "paper", "symbol": "MSFT"}).get_json()["ok"]
    assert not c.post("/investing/api/close", json={"mode": "paper", "symbol": "MSFT", "fraction": 2}).get_json()["ok"]
    live = c.post("/investing/api/close", json={"mode": "live", "symbol": "AAPL"}).get_json()
    assert not live["ok"] and "confirmation" in live["error"]


def test_page_has_position_drawer(monkeypatch, tmp_path):
    html = _client(monkeypatch, tmp_path).get("/investing").get_data(as_text=True)
    assert 'id="drawer"' in html and "Sell all" in html and "investing/api/" in html
