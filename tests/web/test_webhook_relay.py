"""TradingView webhook relay: parsing, rules, paper ledger, broker safety, HTTP."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.web import server as server_module
from app.web.server import app
from app.webhook_relay import config as relay_config
from app.webhook_relay import engine, journal, paper, trade_book
from app.webhook_relay.paper import PaperLedger
from app.webhook_relay.signals import SignalError, parse_alert, pine_alert_message


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(paper, "relay_dir", lambda: tmp_path)
    monkeypatch.setattr(journal, "relay_dir", lambda: tmp_path)
    monkeypatch.setattr(trade_book, "relay_dir", lambda: tmp_path)
    monkeypatch.setattr(relay_config, "_dir", lambda: tmp_path)
    engine._DEDUPE.clear()
    engine._ADAPTERS.clear()
    server_module._license_ok_cached = True
    yield
    server_module._license_ok_cached = None


def _alert(**fields):
    base = {"ticker": "CME_MINI:NQ1!", "price": "18000", "time": "t0"}
    base.update(fields)
    return json.dumps(base)


# ------------------------------------------------------------------ parsing

def test_strategy_position_wins_over_order_action():
    s = parse_alert(_alert(action="sell", position="flat"))
    assert s.target == "flat" and s.quantity is None


def test_reversal_uses_position_size_not_order_contracts():
    s = parse_alert(_alert(action="sell", position="short", quantity="2", position_size="-1"))
    assert s.target == "short" and s.quantity == 1.0


def test_plain_text_and_bad_bodies():
    assert parse_alert("buy", default_ticker="NQ1!").target == "long"
    for bad in ("", "{not json", '{"ticker":"NQ1!"}', '["x"]'):
        with pytest.raises(SignalError):
            parse_alert(bad, default_ticker="" if bad != "" else "")


def test_unfilled_placeholders_are_ignored():
    s = parse_alert('{"ticker":"NQ1!","action":"buy","price":"{{strategy.order.price}}"}')
    assert s.price is None


def test_alert_message_template_has_the_strategy_placeholders():
    msg = pine_alert_message("pw")
    for token in ("{{ticker}}", "{{strategy.order.action}}", "{{strategy.market_position}}",
                  "{{strategy.position_size}}", "{{strategy.order.price}}", '"passphrase": "pw"'):
        assert token in msg


def test_root_symbol():
    for raw, root in [("CME_MINI:NQ1!", "NQ"), ("MNQZ2026", "MNQ"), ("ESH6", "ES"), ("EURUSD", "EURUSD"), ("GC1!", "GC")]:
        assert relay_config.root_symbol(raw) == root


# ------------------------------------------------------------------ rules + paper ledger

def _relay(**kw):
    return relay_config.create_relay(kw)


def test_long_reverse_flat_cycle_pnl():
    r = _relay()
    assert engine.handle(r, parse_alert(_alert(position="long", position_size="2", time="1")))["status"] == "executed"
    assert engine.handle(r, parse_alert(_alert(position="long", position_size="2", time="2")))["status"] == "no_change"
    engine.handle(r, parse_alert(_alert(position="short", position_size="-1", price="18010", time="3")))
    engine.handle(r, parse_alert(_alert(position="flat", price="17990", time="4")))
    snap = PaperLedger(r.id).snapshot()
    assert snap["open"] == [] and snap["trades"] == 2
    # NQ = $20/point, $4.20 round-turn commission: long 2 +10pt, short 1 +20pt
    assert snap["realized"] == pytest.approx(2 * 10 * 20 - 2 * 4.2 + 1 * 20 * 20 - 4.2)


def test_open_position_unrealized():
    r = _relay()
    engine.handle(r, parse_alert(_alert(position="long", position_size="1", time="1")))
    engine.handle(r, parse_alert(_alert(position="long", position_size="1", price="18005", time="2")))
    snap = PaperLedger(r.id).snapshot()
    assert snap["open"][0]["unrealized"] == pytest.approx(5 * 20)


def test_rules_reject_and_cap():
    r = _relay(passphrase="pw", max_qty="3", allowed_tickers="NQ", allow_short="0")
    assert engine.handle(r, parse_alert(_alert(position="long")))["status"] == "rejected"          # no passphrase
    ok = engine.handle(r, parse_alert(_alert(position="long", position_size="9", passphrase="pw")))
    assert ok["status"] == "executed" and "capped" in ok["message"]
    assert PaperLedger(r.id).snapshot()["open"][0]["qty"] == 3
    assert engine.handle(r, parse_alert(_alert(position="short", passphrase="pw", time="9")))["status"] == "rejected"
    other = engine.handle(r, parse_alert(_alert(ticker="ES1!", position="long", passphrase="pw")))
    assert other["status"] == "rejected"


def test_duplicate_alert_and_missing_price():
    r = _relay(dedupe_seconds="30")
    body = _alert(position="long", position_size="1")
    assert engine.handle(r, parse_alert(body))["status"] == "executed"
    assert engine.handle(r, parse_alert(body))["status"] == "duplicate"
    assert engine.handle(r, parse_alert('{"ticker":"ES1!","position":"long","time":"x"}'))["status"] == "error"


def test_disabled_relay_rejects():
    r = _relay(enabled="0")
    assert engine.handle(r, parse_alert(_alert(position="long")))["status"] == "rejected"


# ------------------------------------------------------------------ demo-only broker safety

class _FakeAdapter:
    def __init__(self):
        self.orders, self.positions = [], []

    def ensure_connected(self):
        return SimpleNamespace(ok=True, message="ok")

    def get_open_positions(self, symbol=None):
        return list(self.positions)

    def close_position(self, ticket, comment=""):
        self.positions = [p for p in self.positions if p.ticket != ticket]
        self.orders.append(("close", ticket))
        return SimpleNamespace(ok=True, message="ok")

    def place_market_order(self, symbol, direction, volume, **kw):
        self.orders.append((symbol, direction, volume))
        self.positions.append(SimpleNamespace(ticket=str(len(self.orders)), volume=volume, direction=direction))
        return SimpleNamespace(ok=True, message="ok")

    def disconnect(self):
        pass


def _account(platform="Tradovate", live=False):
    extra = {"is_live": "true" if live else "false"} if platform in ("Tradovate", "TradeLocker") else {}
    return SimpleNamespace(id="acc1", nickname="Demo", platform=platform, extra_credentials=extra)


def _wire_broker(monkeypatch, account, adapter):
    monkeypatch.setattr("app.live_deploy.live_settings.load_accounts", lambda: [account])
    monkeypatch.setattr(engine, "_adapter_for", lambda a: adapter)


def test_live_account_is_refused(monkeypatch):
    adapter = _FakeAdapter()
    _wire_broker(monkeypatch, _account(live=True), adapter)
    r = _relay(broker_account_id="acc1")
    out = engine.handle(r, parse_alert(_alert(position="long", position_size="1")), wait_for_broker=True)
    assert out["broker"]["state"] == "refused" and adapter.orders == []


def test_mt5_needs_demo_confirmation(monkeypatch):
    adapter = _FakeAdapter()
    _wire_broker(monkeypatch, _account(platform="MT4/MT5"), adapter)
    r = _relay(broker_account_id="acc1")
    out = engine.handle(r, parse_alert(_alert(position="long", position_size="1")), wait_for_broker=True)
    assert out["broker"]["state"] == "refused"
    relay_config.update_relay(r.id, {"confirm_demo": "1"})
    out = engine.handle(relay_config.get_relay(r.id), parse_alert(_alert(position="short", time="2", position_size="-1")),
                        wait_for_broker=True)
    assert out["broker"]["state"] == "sent"


def test_demo_broker_mirrors_open_reverse_and_flat(monkeypatch):
    adapter = _FakeAdapter()
    _wire_broker(monkeypatch, _account(), adapter)
    r = _relay(broker_account_id="acc1", symbol_map="NQ=NQZ6")
    engine.handle(r, parse_alert(_alert(position="long", position_size="2", time="1")), wait_for_broker=True)
    assert adapter.orders == [("NQZ6", 1, 2.0)]
    engine.handle(r, parse_alert(_alert(position="short", position_size="-1", time="2")), wait_for_broker=True)
    assert adapter.orders[1][0] == "close" and adapter.orders[2] == ("NQZ6", -1, 1.0)
    engine.handle(r, parse_alert(_alert(position="flat", time="3")), wait_for_broker=True)
    assert adapter.positions == []
    assert journal.recent(r.id)[0]["status"] == "executed"


# ------------------------------------------------------------------ HTTP

def test_webhook_endpoint_end_to_end():
    r = _relay(passphrase="pw")
    c = app.test_client()
    url = f"/webhook/tradingview/{r.token}"
    ok = c.post(url, data=_alert(position="long", position_size="1", passphrase="pw"), content_type="application/json")
    assert ok.status_code == 200 and ok.get_json()["status"] == "executed"
    assert c.post(url, data=_alert(position="long", time="b"), content_type="application/json").status_code == 403
    assert c.post(url, data="{bad", content_type="text/plain").status_code == 400
    assert c.post("/webhook/tradingview/wrong-token", data=_alert(position="long")).status_code == 404
    assert c.get(url + "/ping").get_json()["relay"] == r.name
    state = c.get(f"/webhook-relay/relays/{r.id}/state.json").get_json()
    assert state["paper"]["open"][0]["qty"] == 1 and len(state["signals"]) >= 3


def test_webhook_refuses_when_unlicensed():
    r = _relay()
    server_module._license_ok_cached = False
    out = app.test_client().post(f"/webhook/tradingview/{r.token}", data=_alert(position="long"))
    assert out.status_code == 403


def test_webhook_works_without_session_or_lock(monkeypatch):
    """TradingView has no cookies: the account-lock gate must not block it."""
    from app.accounts import settings as account_settings

    monkeypatch.setattr(account_settings, "load_account_settings", lambda: SimpleNamespace(has_password=True))
    r = _relay()
    out = app.test_client().post(f"/webhook/tradingview/{r.token}", data=_alert(position="long", position_size="1"),
                                 content_type="application/json")
    assert out.status_code == 200


def test_setup_page_and_management_routes():
    c = app.test_client()
    assert c.post("/webhook-relay/public-url", data={"public_url": "https://abc.ngrok-free.app"}).get_json()["ok"]
    assert c.post("/webhook-relay/public-url", data={"public_url": "not a url"}).status_code == 400
    rid = c.post("/webhook-relay/relays", data={"name": "My Pine"}).get_json()["id"]
    page = c.get("/webhook-relay").get_data(as_text=True)
    assert "My Pine" in page and "https://abc.ngrok-free.app/webhook/tradingview/" in page
    assert "{{strategy.market_position}}" in page
    assert c.post(f"/webhook-relay/relays/{rid}/save", data={"name": "Renamed", "default_qty": "3", "enabled": "1"}).get_json()["ok"]
    saved = relay_config.get_relay(rid)
    assert saved.name == "Renamed" and saved.default_qty == 3 and saved.allow_short is False  # unchecked box = off
    test = c.post(f"/webhook-relay/relays/{rid}/test", data={"target": "long", "ticker": "NQ1!", "price": "18000"}).get_json()
    assert test["ok"] and test["status"] == "executed"
    old = saved.token
    c.post(f"/webhook-relay/relays/{rid}/rotate")
    assert relay_config.get_relay(rid).token != old
    assert c.post(f"/webhook-relay/relays/{rid}/delete").get_json()["ok"]
    assert relay_config.get_relay(rid) is None


def test_relay_page_linked_from_deploy_section():
    body = app.test_client().get("/forward-test").get_data(as_text=True)
    assert 'href="/webhook-relay"' in body


# ------------------------------------------------------------------ TradingView only accepts port 80 / 443

@pytest.mark.parametrize("bad", [
    "http://127.0.0.1:5004",            # what the page used to show: this computer only, wrong port
    "http://100.104.34.41:5004",        # a Tailscale address (private VPN range) on the app port
    "http://localhost:5000",
    "http://192.168.1.20:5000",
    "https://abc123.ngrok-free.app:8443",   # public host, but a port TradingView refuses
])
def test_public_address_tradingview_would_reject_is_refused(bad):
    clean, problem = relay_config.check_public_url(bad)
    assert clean == "" and problem
    with pytest.raises(ValueError):
        relay_config.set_public_url(bad)


@pytest.mark.parametrize("good,expected", [
    ("https://abc123.ngrok-free.app", "https://abc123.ngrok-free.app"),
    ("abc123.ngrok-free.app/", "https://abc123.ngrok-free.app"),             # scheme added, slash trimmed
    ("https://my-box.tail1234.ts.net", "https://my-box.tail1234.ts.net"),
    ("http://example.com:80", "http://example.com:80"),
    ("https://example.com:443", "https://example.com:443"),
])
def test_public_address_that_tradingview_accepts_is_kept(good, expected):
    clean, problem = relay_config.check_public_url(good)
    assert problem == "" and clean == expected


def test_page_never_hands_out_a_url_tradingview_will_reject(monkeypatch):
    """Regression: with no public address saved the page showed http://127.0.0.1:5004/webhook/..., which
    TradingView's alert box rejects with "Only port 80 is allowed for HTTP"."""
    c = app.test_client()
    monkeypatch.delenv("T58_PUBLIC_URL", raising=False)
    rid = c.post("/webhook-relay/relays", data={"name": "NQ1! strategy"}).get_json()["id"]
    page = c.get("/webhook-relay").get_data(as_text=True)
    token = relay_config.get_relay(rid).token
    assert token not in page                                   # no URL until a valid address is saved
    assert "Finish step 1 first" in page
    assert c.post("/webhook-relay/public-url", data={"public_url": "http://127.0.0.1:5004"}).status_code == 400
    assert c.post("/webhook-relay/public-url", data={"public_url": "https://abc.ngrok-free.app"}).get_json()["ok"]
    page = c.get("/webhook-relay").get_data(as_text=True)
    assert f"https://abc.ngrok-free.app/webhook/tradingview/{token}" in page


def test_a_bad_address_saved_by_an_older_version_is_not_shown(monkeypatch):
    (relay_config._dir() / relay_config.PUBLIC_URL_FILE).write_text(json.dumps({"url": "http://127.0.0.1:5004"}), encoding="utf-8")
    c = app.test_client()
    rid = c.post("/webhook-relay/relays", data={"name": "old"}).get_json()["id"]
    page = c.get("/webhook-relay").get_data(as_text=True)
    assert relay_config.get_relay(rid).token not in page


def test_address_check_endpoint_reports_clearly_for_bad_input():
    c = app.test_client()
    out = c.post("/webhook-relay/check-public-url", data={"public_url": "http://100.104.34.41:5004"})
    assert out.status_code == 400 and "TradingView" in out.get_json()["error"]


# ------------------------------------------------------------------ one-click tunnel (cloudflared quick tunnel)

def _fake_cloudflared(tmp_path, body: str):
    import stat
    import sys
    script = tmp_path / "fake_cloudflared.py"
    script.write_text(body, encoding="utf-8")
    launcher = tmp_path / "cloudflared"
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n", encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    return str(launcher)


def _wait_state(tunnel, want, timeout=10.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        st = tunnel.status()
        if st["state"] == want:
            return st
        time.sleep(0.05)
    return tunnel.status()


def test_tunnel_starts_saves_the_public_address_and_stops(monkeypatch, tmp_path):
    from app.webhook_relay import tunnel
    fake = _fake_cloudflared(tmp_path, (
        "import sys, time\n"
        "print('2026-10-14 INF Requesting new quick Tunnel on trycloudflare.com...', flush=True)\n"
        "print('2026-10-14 INF |  https://quiet-river-demo-1234.trycloudflare.com  |', flush=True)\n"
        "print('ARGS', ' '.join(sys.argv[1:]), flush=True)\n"
        "time.sleep(30)\n"))
    monkeypatch.setattr(tunnel, "find_binary", lambda: fake)
    try:
        st = tunnel.start(5004, relay_config.set_public_url)
        assert st["state"] in ("starting", "running")
        st = _wait_state(tunnel, "running")
        assert st["state"] == "running" and st["url"] == "https://quiet-river-demo-1234.trycloudflare.com"
        assert relay_config.get_public_url() == "https://quiet-river-demo-1234.trycloudflare.com"
        assert any("tunnel --url http://127.0.0.1:5004" in line for line in st["log"])   # forwards to THIS app's port
        assert tunnel.start(5004, relay_config.set_public_url)["state"] == "running"        # second click: no second process
    finally:
        assert tunnel.stop()["state"] == "stopped"


def test_tunnel_reports_a_clear_error_when_cloudflared_exits_early(monkeypatch, tmp_path):
    from app.webhook_relay import tunnel
    fake = _fake_cloudflared(tmp_path, "print('failed to connect to the edge', flush=True)\n")
    monkeypatch.setattr(tunnel, "find_binary", lambda: fake)
    try:
        tunnel.start(5004, relay_config.set_public_url)
        st = _wait_state(tunnel, "error")
        assert st["state"] == "error" and "cloudflared exited" in st["error"] and st["url"] == ""
    finally:
        tunnel.stop()


def test_tunnel_routes_without_cloudflared_explain_how_to_install(monkeypatch):
    from app.webhook_relay import tunnel
    monkeypatch.setattr(tunnel, "find_binary", lambda: None)
    monkeypatch.setattr(tunnel, "find_ssh", lambda: None)
    c = app.test_client()
    out = c.post("/webhook-relay/tunnel/start")
    body = out.get_json()
    assert out.status_code == 400 and body["ok"] is False and "cloudflared" in body["error"] and body["install_hint"]
    assert c.get("/webhook-relay/tunnel/status").get_json()["available"] is False
    page = c.get("/webhook-relay").get_data(as_text=True)
    assert "wr-tunnel-start" in page and "Start tunnel" in page


# ------------------------------------------------------------------ no-install option: ssh -> localhost.run

def test_tunnel_falls_back_to_ssh_when_cloudflared_is_missing(monkeypatch, tmp_path):
    from app.webhook_relay import tunnel
    fake = _fake_cloudflared(tmp_path, (
        "import sys, time\n"
        "print('ARGS', ' '.join(sys.argv[1:]), flush=True)\n"
        "print('abc123def.lhr.life tunneled with tls termination, https://abc123def.lhr.life', flush=True)\n"
        "time.sleep(30)\n"))
    monkeypatch.setattr(tunnel, "find_binary", lambda: None)
    monkeypatch.setattr(tunnel, "find_ssh", lambda: fake)
    try:
        assert tunnel.status()["available"] is True          # ssh alone is enough
        st = tunnel.start(5004, relay_config.set_public_url)
        st = _wait_state(tunnel, "running")
        assert st["state"] == "running" and st["provider"] == "ssh" and st["url"] == "https://abc123def.lhr.life"
        assert relay_config.get_public_url() == "https://abc123def.lhr.life"
        args = " ".join(st["log"])
        assert "-R 80:127.0.0.1:5004" in args and "nokey@localhost.run" in args and "BatchMode=yes" in args
    finally:
        assert tunnel.stop()["state"] == "stopped"


def test_tunnel_prefers_cloudflared_and_ssh_can_be_forced(monkeypatch):
    from app.webhook_relay import tunnel
    monkeypatch.setattr(tunnel, "find_binary", lambda: "/x/cloudflared")
    monkeypatch.setattr(tunnel, "find_ssh", lambda: "/x/ssh")
    assert tunnel.pick_provider() == "cloudflared" and tunnel.pick_provider("ssh") == "ssh"
    monkeypatch.setattr(tunnel, "find_ssh", lambda: None)
    assert tunnel.pick_provider("ssh") is None
    try:
        tunnel.start(5004, relay_config.set_public_url, provider="ssh")
        assert False, "expected TunnelError"
    except tunnel.TunnelError as exc:
        assert "ssh was not found" in str(exc)


def test_tunnel_ssh_exit_gives_a_clear_error(monkeypatch, tmp_path):
    from app.webhook_relay import tunnel
    fake = _fake_cloudflared(tmp_path, "print('connection refused', flush=True)\n")
    monkeypatch.setattr(tunnel, "find_binary", lambda: None)
    monkeypatch.setattr(tunnel, "find_ssh", lambda: fake)
    try:
        tunnel.start(5004, relay_config.set_public_url)
        st = _wait_state(tunnel, "error")
        assert st["state"] == "error" and "ssh (localhost.run) exited" in st["error"]
    finally:
        tunnel.stop()


# ------------------------------------------------------------------ trade book (what was actually taken)

def test_trade_book_records_broker_fills_and_paper_trades(monkeypatch):
    class Adapter(_FakeAdapter):
        def place_market_order(self, symbol, direction, volume, **kw):
            self.orders.append((symbol, direction, volume))
            t = str(len(self.orders))
            self.positions.append(SimpleNamespace(ticket=t, volume=volume, direction=direction,
                                                  open_price=18001.0, open_time=None))
            return SimpleNamespace(ok=True, message="ok", ticket=t, price=18001.0)

        def close_position(self, ticket, comment=""):
            super().close_position(ticket, comment)
            return SimpleNamespace(ok=True, message="ok", price=18011.0)

    adapter = Adapter()
    _wire_broker(monkeypatch, _account(), adapter)
    r = _relay(broker_account_id="acc1")
    engine.handle(r, parse_alert(_alert(position="long", position_size="1", price="18000", time="1")), wait_for_broker=True)
    engine.handle(r, parse_alert(_alert(position="flat", price="18010", time="2")), wait_for_broker=True)
    rows = trade_book.all_trades(r.id)
    by = {t["source"]: t for t in rows}
    assert by["paper"]["entry"] == 18000 and by["paper"]["exit"] == 18010
    assert by["broker"]["entry"] == 18001.0 and by["broker"]["exit"] == 18011.0
    assert by["broker"]["price_source"] == "broker fill" and by["broker"]["pnl"] is not None
    assert "source,symbol,side" in trade_book.trades_csv(r.id)
    c = app.test_client()
    st = c.get(f"/webhook-relay/relays/{r.id}/state.json").get_json()
    assert len(st["trades"]) == 2 and st["broker_open"] == []
    assert c.get(f"/webhook-relay/relays/{r.id}/trades.csv").status_code == 200
