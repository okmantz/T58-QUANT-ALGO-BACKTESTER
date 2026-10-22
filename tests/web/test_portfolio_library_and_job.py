from app.portfolio import library as plib
from app.web.server import app
from app.web.job_manager import JOB_MANAGER


def test_portfolio_library_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(plib, "portfolio_library_dir", lambda: tmp_path)
    p = plib.save_portfolio("FX + Gold", {"engine": "unified", "legs": [{"name": "A", "weight": 1.0}]},
                            {"net_profit": 1234.5, "max_dd": 3.2})
    assert p.name.endswith(".portfolio.json")
    rows = plib.list_portfolios()
    assert rows[0]["name"] == "FX + Gold" and rows[0]["engine"] == "unified" and rows[0]["net_profit"] == 1234.5
    again = plib.save_portfolio("FX + Gold", {"engine": "legacy", "legs": []})
    assert again != p                                         # no silent overwrite
    assert plib.load_portfolio(p.name)["spec"]["legs"][0]["name"] == "A"
    assert plib.delete_portfolio(p.name) and not plib.delete_portfolio(p.name)


def test_portfolio_job_routes(monkeypatch):
    c = app.test_client()
    assert c.get("/portfolio/job/doesnotexist/status.json").status_code == 404
    jid = JOB_MANAGER.create(tool="Portfolio backtest")
    JOB_MANAGER.finish(jid, result={"trades": 3})
    body = c.get(f"/portfolio/job/{jid}/status.json").get_json()
    assert body["done"] is True and body["result"] == {"trades": 3}
    assert c.get(f"/portfolio/job/{jid}").status_code == 200
    assert "unified" in c.get("/portfolio").get_data(as_text=True)
