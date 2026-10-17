"""Evolution Lab live feed routes + gallery wiring (audit item 2)."""
from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from app.evolution import lab_extras as lx
from app.web import evolution_lab_routes as routes
from app.web.evolution_lab_routes import parse_reference_trades


class _FakeRunner:
    is_running = True
    generation = 3
    leaderboard: list = []

    def __init__(self):
        self.curve_store = lx.EquityCurveStore(10)
        self.control_tracker = lx.ControlTracker()
        self.parity = {}
        self.attached = None

    def lab_status(self):
        return {"controls": self.control_tracker.report(), "parity": self.parity, "tribes_active": True,
                "tribe_merge_generation": 10, "generation": 3}

    def attach_parity(self, cid, trades):
        if cid != "c1":
            raise KeyError(f"{cid} is not on the leaderboard")
        self.attached = trades
        return {"status": "PASS", "reasons": []}

    def finalists_report(self, top_n=5):
        return [{"candidate_id": "c1", "readiness": "READY"}]


@pytest.fixture()
def client():
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(routes.evolution_lab_bp)
    runner = _FakeRunner()
    routes.set_runner_source(lambda: runner)
    yield app.test_client(), runner
    routes.set_runner_source(None)


def test_feed_is_incremental_and_flags_restart(client):
    c, r = client
    r.curve_store.add({"id": "a", "curve": [1, 2]}); r.curve_store.add({"id": "b", "curve": [1, 3]})
    d = c.get("/api/evolution/lab/feed?since=0").get_json()
    assert [i["id"] for i in d["items"]] == ["a", "b"] and d["seq"] == 2 and d["lab"]["tribes_active"] is True
    assert c.get("/api/evolution/lab/feed?since=1").get_json()["items"][0]["id"] == "b"
    assert c.get("/api/evolution/lab/feed?since=99").get_json()["reset"] is True


def test_feed_without_runner():
    from flask import Flask
    app = Flask(__name__); app.register_blueprint(routes.evolution_lab_bp)
    routes.set_runner_source(lambda: None)
    assert app.test_client().get("/api/evolution/lab/feed").get_json()["items"] == []


def test_parity_upload_csv_and_errors(client):
    c, r = client
    csv_text = "Time,Type,Profit\n2026-01-05 10:00,buy,12.5\n2026-01-05 12:00,sell,-3\n2026-01-05 13:00,balance,100\n"
    res = c.post("/api/evolution/lab/parity", data={"candidate_id": "c1", "file": (io.BytesIO(csv_text.encode()), "x.csv")},
                 content_type="multipart/form-data")
    assert res.status_code == 200 and res.get_json()["parity"]["status"] == "PASS"
    assert [t["direction"] for t in r.attached] == [1, -1]          # the 'balance' row is skipped
    assert c.post("/api/evolution/lab/parity", json={"candidate_id": "nope", "trades": []}).status_code == 404
    assert c.post("/api/evolution/lab/parity", json={"trades": []}).status_code == 400
    assert c.post("/api/evolution/lab/parity", json={"candidate_id": "c1", "csv": "a,b\n1,2"}).status_code == 400


def test_finalists_is_computed_off_thread(client):
    import time
    c, _ = client
    out = c.get("/api/evolution/lab/finalists").get_json()
    for _ in range(50):
        if out["status"] == "done":
            break
        time.sleep(0.05)
        out = c.get("/api/evolution/lab/finalists").get_json()
    assert out["status"] == "done" and out["rows"][0]["readiness"] == "READY"


def test_parse_reference_trades_needs_columns():
    with pytest.raises(ValueError):
        parse_reference_trades("foo,bar\n1,2")
    assert parse_reference_trades("") == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_gallery_curve_geometry_under_node():
    js = Path(__file__).resolve().parents[2] / "app" / "web" / "static" / "evolution-lab.js"
    harness = ("global.window=undefined;const src=require('fs').readFileSync(process.argv[1],'utf8');"
               "const vm=require('vm');const ctx={globalThis:null};ctx.globalThis=ctx;vm.createContext(ctx);vm.runInContext(src,ctx);"
               "const g=ctx.T58EvolutionLab.curveGeometry([100,110,90,120],100,50);"
               "console.log(JSON.stringify({up:g.up,n:g.pts.length,flat:ctx.T58EvolutionLab.curveGeometry([5,5,5],100,50).up,none:ctx.T58EvolutionLab.curveGeometry([1],10,10)}));")
    p = subprocess.run(["node", "-e", harness, str(js)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["up"] is True and out["n"] == 4 and out["flat"] is True and out["none"] is None


def test_evolution_page_includes_gallery():
    html = (Path(__file__).resolve().parents[2] / "app" / "web" / "templates" / "evolution.html").read_text(encoding="utf-8")
    assert "_evolution_lab_live.html" in html
