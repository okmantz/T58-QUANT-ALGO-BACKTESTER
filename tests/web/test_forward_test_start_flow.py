"""Forward Test page flow: Start returns at once, a background job reports progress, failures arrive as plain-language
problem cards, and the Ready check explains what is missing. Uses a fake MT5 connector: no terminal, no Wine."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pandas as pd
import pytest

from app.forward_test import mt5_connector as mc
from app.forward_test import mt5_diagnostics as diag
from app.forward_test.journal import ForwardTestJournal as RealJournal
from app.forward_test.start_job import StartJob
from app.web import server
from app.web.server import app


class FakeConnector:
    """Quacks like MT5Connector. `outcome` is 'ok' or a Problem to fail with."""
    outcome = "ok"
    last = None

    def __init__(self, login, password, server, terminal_path="", **kw):
        self.login, self.server_name, self.terminal_path = login, server, terminal_path
        self.server = server
        self.on_progress = None
        FakeConnector.last = self

    def connect(self):
        for step in ("prepare", "helper", "terminal", "login", "trading"):
            if self.on_progress:
                self.on_progress(step, f"doing {step}")
        if self.outcome == "ok":
            return mc.ConnectionResult(ok=True, message="Connected.", account_login=1, account_server="Demo", balance=1000.0, equity=1000.0)
        return mc._fail_result(self.outcome)

    def check_symbol(self, symbol): return None
    def get_open_positions(self, symbol=None): return []
    def is_alive(self): return True
    def ensure_connected(self, **k): return SimpleNamespace(ok=True, message="")
    def account_summary(self): return {"balance": 1000.0, "equity": 1000.0}
    def fetch_completed_bars(self, *a, **k): return pd.DataFrame()
    def disconnect(self): pass


@pytest.fixture()
def client(tmp_path, monkeypatch):
    acct = SimpleNamespace(id="a1", nickname="Demo", login="1234", server="Demo-Server", password="pw", terminal_path="",
                           platform="MT4/MT5", is_usable=True)
    import app.live_deploy.live_settings as ls
    monkeypatch.setattr(ls, "load_accounts", lambda: [acct])
    monkeypatch.setattr(mc, "MT5Connector", FakeConnector)
    monkeypatch.setattr("app.forward_test.journal.ForwardTestJournal", lambda: RealJournal(db_path=tmp_path / "ft.db"))
    monkeypatch.setattr(server, "_build_strategy", lambda mode, form, files: (object(), None))
    server._FT_START_JOB.clear()
    server._FORWARD_TEST_SESSION.update({"session": None, "log": []})
    FakeConnector.outcome = "ok"
    yield app.test_client()
    c = app.test_client()
    c.post("/forward-test/stop")
    server._FT_START_JOB.clear()
    server._FORWARD_TEST_SESSION.update({"session": None, "log": []})


FORM = {"account_id": "a1", "symbol": "XAUUSD", "timeframe_minutes": "15", "strategy_mode": "python",
        "existing_strategy_python": "my_strat.py", "initial_balance": "100000", "risk_value": "1.0", "risk_mode": "percent",
        "poll_seconds": "1", "commission": "3"}


def _wait(client, states, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        snap = client.get("/forward-test/start-status.json").get_json()
        if snap["state"] in states:
            return snap
        time.sleep(0.1)
    raise AssertionError(f"job never reached {states}: {snap}")


def test_start_returns_immediately_then_job_succeeds(client):
    r = client.post("/forward-test/start", data=FORM)
    assert r.get_json() == {"ok": True, "starting": True}
    snap = _wait(client, {"ok"})
    assert [s["status"] for s in snap["steps"]] == ["done"] * 6
    assert client.get("/forward-test/status.json").get_json()["running"] is True
    assert client.post("/forward-test/stop").get_json()["ok"]


def test_failure_arrives_as_a_structured_plain_language_problem(client):
    FakeConnector.outcome = diag.algo_trading_off()
    client.post("/forward-test/start", data=FORM)
    snap = _wait(client, {"error"})
    p = snap["problem"]
    assert p["kind"] == "algo_trading_off" and p["steps"] and "Algo Trading" in p["title"]
    failed = [s for s in snap["steps"] if s["status"] == "error"]
    assert failed and failed[0]["id"] == "trading"           # the checklist shows WHERE it stopped
    assert client.get("/forward-test/status.json").get_json()["running"] is False


def test_progress_text_reaches_the_page(client):
    FakeConnector.outcome = diag.wine_missing()
    client.post("/forward-test/start", data=FORM)
    snap = _wait(client, {"error"})
    assert any(s["text"] == "doing helper" for s in snap["steps"])


def test_missing_account_and_strategy_get_problem_cards_not_bare_strings(client):
    r = client.post("/forward-test/start", data={**FORM, "account_id": "nope"})
    assert r.status_code == 400 and r.get_json()["problem"]["kind"] == "account_missing"
    r = client.post("/forward-test/start", data={**FORM, "existing_strategy_python": ""})
    assert r.status_code == 400 and r.get_json()["problem"]["kind"] == "strategy_missing"


def test_second_start_while_starting_is_refused(client):
    import threading
    gate = threading.Event()
    orig = FakeConnector.connect
    FakeConnector.connect = lambda self: (gate.wait(5), orig(self))[1]
    try:
        assert client.post("/forward-test/start", data=FORM).get_json()["ok"]
        r = client.post("/forward-test/start", data=FORM)
        assert r.status_code == 409
    finally:
        gate.set()
        FakeConnector.connect = orig
        _wait(client, {"ok", "error"})


def test_journal_gets_the_real_strategy_name_and_gold_spec(client):
    client.post("/forward-test/start", data=FORM)
    _wait(client, {"ok"})
    sess = server._FORWARD_TEST_SESSION["session"]
    assert sess.strategy_filename == "my_strat.py"                       # was the bare mode word "python"
    assert sess.cfg.risk.pip_size == 0.1                                 # XAUUSD spec, not the EURUSD 0.0001 default


def test_page_renders_ready_check_and_problem_container(client):
    body = client.get("/forward-test").get_data(as_text=True)
    assert 'id="ready-card"' in body and 'id="problem-box"' in body and 'id="start-card"' in body
    assert "Fixed $ per trade" in body


def test_preflight_route_reports_account_and_actions(client, monkeypatch):
    monkeypatch.setattr(mc, "platform_kind", lambda: "linux")
    monkeypatch.setattr(mc, "find_terminal_candidates", lambda: [])
    monkeypatch.setattr("app.forward_test.mt5_bridge_manager.find_wine", lambda: None)
    data = client.get("/forward-test/preflight.json?account_id=a1").get_json()
    by = {c["id"]: c for c in data["checks"]}
    assert by["account"]["status"] == "ok" and by["wine"]["status"] == "action" and data["ready"] is False


def test_start_job_unit_survives_a_crashing_session():
    class Boom:
        last_problem = None
        def start(self, progress=None): raise RuntimeError("kaboom")
    job = StartJob()
    assert job.begin(Boom(), lambda s: None)
    for _ in range(50):
        if job.snapshot()["state"] != "running":
            break
        time.sleep(0.05)
    snap = job.snapshot()
    assert snap["state"] == "error" and snap["problem"]["kind"] == "unknown" and "kaboom" in snap["problem"]["technical"]
