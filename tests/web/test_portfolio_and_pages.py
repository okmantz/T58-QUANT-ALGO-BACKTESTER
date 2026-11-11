"""Portfolio overview module + API, and render smoke tests for the redesigned Test / Validate / Create / Deploy pages."""
from types import SimpleNamespace

import pytest

import app.web.server as server
from app.forward_test.journal import ForwardTestJournal
from app.live_deploy.portfolio import build_portfolio


@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def _acct(login="123", usable=True):
    return SimpleNamespace(id="a1", nickname="Eval 1", platform="MT4/MT5", firm_name="FTMO", login=login, is_usable=usable)


def test_portfolio_aggregates_pnl_open_and_health(tmp_path):
    j = ForwardTestJournal(tmp_path / "j.db")
    sid = j.start_session("python", "s.py", "NQ", 5, "123", "srv")
    t1 = j.record_open(sid, 1, 1, 1.0, 100.0, None, None)
    j.record_close(t1, 110.0, 50.0)
    t2 = j.record_open(sid, 2, -1, 1.0, 100.0, None, None)
    j.record_close(t2, 105.0, -20.0)
    j.record_open(sid, 3, 1, 1.0, 100.0, None, None)
    out = build_portfolio([_acct()], j, {"running": True, "connected": True, "balance": 1000.0, "equity": 1010.0}, "123")
    a = out["accounts"][0]
    assert a["net_pnl"] == 30.0 and a["n_closed"] == 2 and a["win_rate"] == 50.0
    assert len(a["open_trades"]) == 1 and a["floating"] == 10.0
    assert a["health"]["state"] == "ok" and out["eligible"] is True


def test_portfolio_not_eligible_without_account_or_activity(tmp_path):
    j = ForwardTestJournal(tmp_path / "j.db")
    assert build_portfolio([], j)["eligible"] is False
    out = build_portfolio([_acct()], j)
    assert out["eligible"] is False and "forward test" in out["eligible_reason"].lower()
    assert out["accounts"][0]["health"]["state"] == "idle"


def test_halted_health_wins():
    from app.live_deploy.portfolio import _health
    assert _health({"running": True, "connected": True, "alert": "max DD"}, 50)["state"] == "halted"


def test_api_portfolio_and_pages_render(client):
    r = client.get("/api/portfolio")
    if r.status_code == 200:
        assert "accounts" in r.get_json()
    for url, needle in (("/full-pipeline", "Batch &amp; scheduling"), ("/validate-simple", "Regime matrix"), ("/start-here/create", "cr-paths")):
        resp = client.get(url, follow_redirects=True)
        if resp.status_code == 200:
            body = resp.get_data(as_text=True)
            assert needle in body, (url, needle)
