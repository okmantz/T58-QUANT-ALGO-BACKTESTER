import io
import json
import zipfile

import pytest

from app.web import library_detail_routes as ldr
from app.web.server import app


def _snap(heat_cell=(3, 11, -900.0)):
    g = [[0.0] * 24 for _ in range(7)]
    c = [[0] * 24 for _ in range(7)]
    d, h, v = heat_cell
    g[d][h], c[d][h] = v, 12
    g[2][10], c[2][10] = 1500.0, 20
    return {"recorded_at": 1.0, "instrument": "ES1!", "verdict": "NOT READY", "reasons": ["eval pass 40%"],
            "stats": {"net_profit": 1200.0, "total_trades": 90, "win_rate": 62.0, "profit_factor": 1.1, "max_drawdown_pct": 12.0},
            "eval_pass_probability": 40.0, "risk_of_ruin_pct": 8.0, "equity": [100.0, 101.0, 105.0], "heatmap": g, "heatmap_counts": c,
            "reports": {}}


def test_rundown_prefills_time_filter_and_risk():
    r = ldr.build_optimize_rundown("manual", "x.json", {"last_validation": {"dataset": "es.csv"}}, _snap(), [])
    goals = [f["goal"] for f in r["focus"]]
    assert goals[0] == "timefilter"
    assert "risk" in goals and "eval" in goals and "exits" in goals
    assert "blocked_slots=THU+11" in r["href"] and "goal=timefilter" in r["href"]
    assert "strategy_pick=manual%3A%3Ax.json" in r["href"] and "existing_dataset=es.csv" in r["href"]


def test_rundown_without_results_says_run_pipeline():
    r = ldr.build_optimize_rundown("manual", "x.json", {}, None, [])
    assert r["focus"][0]["title"].startswith("Run the Full Pipeline")


def test_rundown_does_not_resuggest_already_blocked_slot():
    r = ldr.build_optimize_rundown("manual", "x.json", {"blocked_entry_slots": "THU 11"}, _snap(), [])
    assert all(f["goal"] != "timefilter" for f in r["focus"])


def test_detail_route_and_downloads(monkeypatch, tmp_path):
    meta = {"name": "Demo", "pipeline_snapshot": _snap()}
    monkeypatch.setattr(ldr, "_meta", lambda t, f: meta)
    monkeypatch.setattr(ldr, "_folder_entries", lambda t, f, n: [])
    monkeypatch.setattr(ldr, "validation_reports", lambda t, f, n, limit=20: [])
    monkeypatch.setattr("app.data.storage.get_app_base_dir", lambda: tmp_path)
    c = app.test_client()
    d = c.get("/api/library/detail?type=manual&filename=demo.json").get_json()
    assert d["ok"] and d["backtest"]["has"] and d["backtest"]["equity"] == [100.0, 101.0, 105.0]
    assert d["validation"]["has"] is False
    assert d["downloads"]["full_pipeline"] == {}
    assert c.get("/library/download/full_pipeline?type=manual&filename=demo.json").status_code == 404
    z = c.get("/library/download-all.zip?type=manual&filename=demo.json")
    assert z.status_code == 200
    assert any(n.endswith("README.txt") for n in zipfile.ZipFile(io.BytesIO(z.data)).namelist())
    assert c.get("/api/library/detail?type=manual&filename=../x").status_code == 400


def test_pipeline_download_serves_snapshot_report(monkeypatch, tmp_path):
    (tmp_path / "rep.html").write_text("<html>ok</html>")
    meta = {"name": "Demo", "pipeline_snapshot": {**_snap(), "reports": {"html": "/full_pipeline_reports/rep.html"}}}
    monkeypatch.setattr(ldr, "_meta", lambda t, f: meta)
    monkeypatch.setattr(ldr, "_folder_entries", lambda t, f, n: [])
    monkeypatch.setattr(ldr, "_full_pipeline_dir", lambda: tmp_path)
    monkeypatch.setattr(ldr, "validation_reports", lambda t, f, n, limit=20: [])
    c = app.test_client()
    r = c.get("/library/download/full_pipeline?type=manual&filename=demo.json&fmt=html")
    assert r.status_code == 200 and b"ok" in r.data
    assert "attachment" in r.headers.get("Content-Disposition", "")
    d = c.get("/api/library/detail?type=manual&filename=demo.json").get_json()
    assert "html" in d["downloads"]["full_pipeline"]
