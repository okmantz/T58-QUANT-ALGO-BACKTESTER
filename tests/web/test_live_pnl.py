"""/api/live-pnl feeds the floating Live P&L widget."""
from types import SimpleNamespace

import app.web.server as srv


def _status(**kw):
    base = dict(running=True, connected=True, platform="MT5", balance=100000.0, equity=100250.0, n_trades_closed=4,
                win_rate=0.5, net_pnl=300.0, halted_reason=None, drift_flag=None, alert=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_empty_when_nothing_running():
    r = srv.app.test_client().get("/api/live-pnl")
    assert r.status_code == 200 and r.get_json() == {"sessions": []}


def test_reports_running_demo_session(monkeypatch):
    monkeypatch.setitem(srv._FORWARD_TEST_SESSION, "session", SimpleNamespace(status=_status()))
    rows = srv.app.test_client().get("/api/live-pnl").get_json()["sessions"]
    assert len(rows) == 1 and rows[0]["kind"] == "demo"
    assert rows[0]["floating"] == 250.0 and rows[0]["net_pnl"] == 300.0


def test_halted_session_still_listed(monkeypatch):
    monkeypatch.setitem(srv._FORWARD_TEST_SESSION, "session",
                        SimpleNamespace(status=_status(running=False, halted_reason="max drawdown")))
    rows = srv.app.test_client().get("/api/live-pnl").get_json()["sessions"]
    assert rows and rows[0]["halted_reason"] == "max drawdown"


def test_widget_script_included_in_pages():
    html = srv.app.test_client().get("/settings/account").get_data(as_text=True)
    assert "t58-livepnl.js" in html
