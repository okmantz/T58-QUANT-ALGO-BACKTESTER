"""T58 demo prop account: default mode, rules, win/loss book, relay feed, header pill, Live guard."""
import pytest

import app.web.server as server
from app.demo_account import account as D


@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def test_new_users_default_to_demo_and_toggle_persists(client):
    assert D.get_mode() == "demo"
    assert client.get("/api/header-pnl").get_json()["mode"] == "demo"
    r = client.post("/settings/account/mode", json={"mode": "live"})
    assert r.get_json() == {"ok": True, "mode": "live"} and D.get_mode() == "live"
    assert client.get("/api/header-pnl").get_json()["mode"] == "live"
    assert client.post("/settings/account/mode", json={"mode": "paper"}).status_code == 400
    client.post("/settings/account/mode", json={"mode": "demo"})
    assert D.get_mode() == "demo"


def test_tiers_scale_the_rules():
    for size in D.TIERS:
        a = D.set_tier(size)
        assert a.balance == size and a.floor == pytest.approx(size - a.rules.loss_limit)
    with pytest.raises(ValueError):
        D.set_tier(75_000)
    d = D.set_tier(100_000).to_dict()
    assert d["profit_target"] == pytest.approx(2 * D.set_tier(50_000).to_dict()["profit_target"])


def test_wins_and_losses_are_booked_visibly():
    D.set_tier(50_000)
    r1 = D.record_trade(300.0, root="NQ", side="long", closed_at="2026-11-02T15:00:00+00:00")
    r2 = D.record_trade(-150.0, root="NQ", side="short", closed_at="2026-11-02T16:00:00+00:00")
    assert (r1["result"], r2["result"]) == ("win", "loss") and r2["balance"] == 50_150.0
    d = D.load().to_dict()
    assert d["n_wins"] == 1 and d["n_losses"] == 1 and d["net_pnl"] == 150.0 and d["win_rate"] == 50.0
    assert d["recent_trades"][0]["n"] == 2                       # newest first


def test_daily_soft_limit_flags_but_does_not_fail():
    a = D.set_tier(50_000)
    soft = a.rules.soft_daily_limit
    D.record_trade(-(soft + 10), closed_at="2026-11-02T15:00:00+00:00")
    d = D.load().to_dict()
    assert d["day_stopped"] and d["status"] == "active" and d["soft_limit_hits"] == 1


def test_breach_at_the_trailing_floor_then_reset():
    a = D.set_tier(25_000)
    row = D.record_trade(-(a.rules.loss_limit + 1), closed_at="2026-11-02T15:00:00+00:00")
    assert row["status"] == "breached" and D.load().status == "breached"
    assert D.record_trade(500.0) is None                          # a finished evaluation stops booking
    assert D.reset().attempt == 2 and D.load().status == "active"


def test_floor_trails_end_of_day_highs_and_locks():
    a = D.set_tier(50_000)
    lim, off = a.rules.loss_limit, a.rules.lock_floor_offset
    D.record_trade(600.0, closed_at="2026-11-02T15:00:00+00:00")
    D.record_trade(100.0, closed_at="2026-11-03T15:00:00+00:00")        # day roll banks day one's close
    a = D.load()
    assert a.floor == pytest.approx(50_000 + 600 - lim)
    D.record_trade(5_000.0, closed_at="2026-11-04T15:00:00+00:00")
    D.record_trade(1.0, closed_at="2026-11-05T15:00:00+00:00")
    assert D.load().floor <= 50_000 + off + 1e-6                          # never trails above the lock


def test_pass_needs_target(monkeypatch):
    a = D.set_tier(50_000)
    n = int(a.rules.profit_target // 3) + 1
    for i in range(4):
        D.record_trade(a.rules.profit_target / 4 + 1, closed_at=f"2026-11-0{i + 2}T15:00:00+00:00")
    assert D.load().status == "passed" and "target" in D.load().status_reason.lower()


def test_relay_alerts_feed_the_demo_book_like_a_real_account(client):
    out = D.deploy_strategy("Opening drive", size=50_000, tickers=["NQ"], public_url="https://example.com")
    assert out["webhook_url"].startswith("https://example.com/webhook/tradingview/") and "{{strategy.order.action}}" in out["alert_message"]
    from app.webhook_relay import config as rc, engine
    import json as _json
    from app.webhook_relay.signals import parse_alert
    relay = rc.get_relay(out["relay_id"])
    mk = lambda action, pos, price: parse_alert(_json.dumps({"ticker": "NQ1!", "action": action, "position": pos, "quantity": "1", "price": str(price)}))  # noqa: E731
    sig_in, sig_out = mk("buy", "long", 20000.0), mk("sell", "flat", 20010.0)
    engine.handle(relay, sig_in)
    engine.handle(relay, sig_out)
    d = D.load().to_dict()
    assert d["n_trades"] == 1 and d["recent_trades"][0]["result"] == "win" and d["recent_trades"][0]["source"] == "tradingview"
    assert client.get("/api/header-pnl").get_json()["pnl"] == d["net_pnl"] > 0


def test_demo_page_renders_and_deploy_route(client):
    r = client.get("/forward-test?view=demo")
    assert r.status_code == 200 and "T58 Prop Firm demo" in r.get_data(as_text=True)
    j = client.post("/demo-account/deploy", json={"strategy": "S1", "size": 25000}).get_json()
    assert j["ok"] and j["size"] == 25_000 and j["path"].startswith("/webhook/tradingview/")
    assert client.post("/demo-account/sim-trade", json={"pnl": 120}).get_json()["trade"]["result"] == "win"
    client.post("/settings/account/mode", json={"mode": "live"})
    assert client.post("/demo-account/sim-trade", json={"pnl": 5}).status_code == 409


def test_real_account_connect_blocked_in_demo_and_warning_shown(client):
    r = client.post("/deploy-live/connect/save", json={"platform": "tradovate", "values": {}})
    assert r.status_code in (409, 401, 403)
    if r.status_code == 409:
        assert r.get_json()["demo_mode"]
    page = client.get("/deploy-live").get_data(as_text=True)
    assert "Demo mode" in page and "Live" in page
    assert 'id="t58-pnl-chip"' in client.get("/settings/account").get_data(as_text=True)
