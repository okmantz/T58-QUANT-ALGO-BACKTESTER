"""2026-10-17 release: Journal, Gamma Center, clickable metrics, library control feed, prop-sim presets, PWA, runs."""
from __future__ import annotations

import json

import pytest

from app.data import gamma_data as gd
from app.web import journal_store
from app.web.server import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(journal_store, "journal_path", lambda: tmp_path / "journal.json")
    monkeypatch.setattr(gd, "gamma_dir", lambda: tmp_path)
    return app.test_client()


@pytest.mark.parametrize("url", ["/journal", "/gamma", "/runs", "/mobile-access", "/sw.js", "/manifest.json",
                                 "/static/t58-metrics.js", "/static/t58-pwa.js", "/static/t58-runs.js"])
def test_pages_render(client, url):
    r = client.get(url)
    assert r.status_code == 200, url


def test_sidebar_links_new_pages(client):
    html = client.get("/dashboard").get_data(as_text=True)
    assert 'href="/journal"' in html and 'href="/gamma"' in html and 'href="/runs"' in html


def test_service_worker_has_root_scope(client):
    r = client.get("/sw.js")
    assert r.headers.get("Service-Worker-Allowed") == "/"


def test_journal_preloaded_with_screenshot_playbook_and_tags(client):
    d = client.get("/api/journal").get_json()
    pb = d["settings"]["playbooks"][0]
    assert pb["title"] == "M15 Pro Trend Intraday Model"
    assert [s["title"] for s in pb["sections"]] == ["BIAS", "AREA OF INTEREST", "ENTRY CRITERIA",
                                                    "ADDITIONAL CONFLUENCES", "MANAGEMENT", "NEWS"]
    groups = {g["name"]: [i["name"] for i in g["items"]] for g in d["settings"]["tagGroups"]}
    assert list(groups) == ["Setups", "Area of Interest", "Trend Phase", "Structure Flow", "Execution Window", "Instrument"]
    assert len(groups["Setups"]) == 8 and len(groups["Execution Window"]) == 14 and "ES1!" in groups["Instrument"]
    assert "1800–1900" in groups["Execution Window"]


def test_journal_trade_roundtrip_and_stats(client):
    for pnl, risk in ((1000, 500), (-300, 300), (500, 250)):
        r = client.post("/api/journal/trades", json={"date": "2026-10-01", "instrument": "ES1!", "pnl": pnl, "riskAmount": risk})
        assert r.get_json()["ok"]
    s = client.get("/api/journal").get_json()["stats"]
    assert s["trades"] == 3 and s["total_pnl"] == 1200 and s["wins"] == 2 and s["profit_factor"] == 5.0
    assert s["win_rate"] == 66.7 and s["max_drawdown"] == 300.0
    tid = client.get("/api/journal").get_json()["trades"][0]["id"]
    assert client.delete(f"/api/journal/trades/{tid}").get_json()["ok"]
    assert client.get("/api/journal").get_json()["stats"]["trades"] == 2


def test_journal_checklist_compliance(client):
    t = client.post("/api/journal/trades", json={"checklist": {"a": True, "b": False, "c": True, "d": True}}).get_json()["trade"]
    assert t["compliancePct"] == 75.0


def test_journal_settings_save(client):
    r = client.put("/api/journal/settings", json={"accountBalance": 25000})
    assert r.get_json()["settings"]["accountBalance"] == 25000.0
    assert client.put("/api/journal/settings", json={"accountBalance": "abc"}).status_code == 400


def test_metric_definitions(client):
    d = client.get("/api/metrics/profit_factor").get_json()["metric"]
    assert "gross profit" in d["formula"].lower() and d["inputs"]
    assert client.get("/api/metrics/Prop Pass Rate").get_json()["metric"]["key"] == "eval_pass_probability"
    assert client.get("/api/metrics/nonsense").status_code == 404
    assert client.get("/api/metrics/profit_factor/history").get_json()["ok"]
    labels = client.get("/api/metrics").get_json()["labels"]
    assert labels["t58 score"] == "t58_score" and labels["max drawdown"] == "max_drawdown"


def test_library_control_feed(client):
    d = client.get("/api/library/control").get_json()
    assert d["ok"] and set(d["summary"]["lanes"]) == {"Research", "Developing", "Ready", "Forward Testing"}


def test_prop_sim_presets(client):
    p = client.get("/api/prop-sim/presets").get_json()["presets"]
    assert p and {"target_pct", "daily_loss_pct", "max_dd_pct", "account_size"} <= set(p[0])


def _chain(spot=6800.0):
    from datetime import date, timedelta
    exp = (date.today() + timedelta(days=1)).strftime("%y%m%d")
    opts = []
    for k, c_oi, p_oi in ((6700, 100, 4000), (6750, 500, 2000), (6800, 3000, 3000), (6850, 5000, 400), (6900, 6000, 100)):
        g = 0.01 * (1.0 - abs(k - spot) / 400.0)
        opts.append({"option": f"SPX{exp}C{k * 1000:08d}", "open_interest": c_oi, "gamma": g, "volume": 10})
        opts.append({"option": f"SPX{exp}P{k * 1000:08d}", "open_interest": p_oi, "gamma": g, "volume": 10})
    return {"data": {"current_price": spot, "options": opts}}


def test_gamma_levels_from_chain(client, monkeypatch):
    import json as _j
    monkeypatch.setattr(gd, "fetcher", lambda url: _j.dumps(_chain()).encode())
    r = client.get("/api/gamma/live/ES").get_json()
    g = r["gex"]
    assert r["ok"] and g["call_wall"] == 6900 and g["put_wall"] == 6700
    assert g["gamma_flip"] is None or 6700 <= g["gamma_flip"] <= 6900
    assert g["regime"] in ("positive", "negative") and len(g["strikes"]) == 5
    s = client.post("/api/gamma/snapshot/ES").get_json()
    assert s["ok"] and gd.load_levels("ES").shape[0] == 1


def test_gamma_import_and_merge_is_lagged(client):
    import pandas as pd
    csv_text = "date,gamma_flip,call_wall,put_wall,spot\n2026-09-01,6700,6900,6600,6750\n"
    assert client.post("/api/gamma/import/ES", json={"csv": csv_text}).get_json()["imported"] == 1
    ts = pd.date_range("2026-09-01", periods=3 * 288, freq="5min", tz="America/Chicago")
    df = pd.DataFrame({"timestamp": ts, "open": 6760.0, "high": 6761.0, "low": 6759.0, "close": 6760.0, "volume": 1})
    out = gd.merge_gamma_levels(df, "ES")
    day1 = out[ts.normalize() == ts[0].normalize()]
    assert day1["gex_flip"].isna().all(), "same-day levels must not leak into that day's bars (lookahead)"
    assert out["gex_flip"].notna().any()


def test_gamma_bad_input(client):
    assert client.post("/api/gamma/import/ES", json={"csv": "x,y\n1,2"}).status_code == 400
    assert client.get("/api/gamma/live/EURUSD").status_code == 502


def test_library_page_has_control_center(client):
    html = client.get("/library").get_data(as_text=True)
    for needle in ("filter-instrument", "filter-family", "filter-timeframe", "filter-validation", "cc-card", "/api/library/control"):
        assert needle in html
    for lane in ("Research", "Developing", "Ready", "Forward Testing"):
        assert lane in html


def test_library_control_row_fields():
    from app.strategy import library_control as lc
    meta = {"last_run": {"profit_factor": 1.6, "net_profit": 5000, "trades": 100, "max_dd": 4.2},
            "last_champion_check": {"verdict": "READY", "t58_score": 80},
            "validation_methods_run": {"cpcv": {"efficiency": 70, "pbo": 20, "recorded_at": 1760000000}}}
    row = lc.build_row("manual", "ES gap fade 5m.json", meta, "validated", {"eval_pass_probability": 0.42, "equity_curve": list(range(100))})
    assert row["instrument"] == "ES" and row["timeframe"] == "5m" and row["status"] == "Ready"
    assert row["profit_factor"] == 1.6 and row["expectancy"] == 50.0 and row["max_drawdown_pct"] == 4.2
    assert row["pass_rate"] == 42.0 and row["robustness"] == 76.7 and row["validated_on"] == "2025-10-09"
    assert len(row["curve"]) == 40
    assert lc.lane_for({"promotion": {"stage": "forward_testing"}}, "draft") == "Forward Testing"
    assert lc.lane_for({}, "draft") == "Research"
    assert lc.normalize_instrument("MNQ1!") == "NQ" and lc.normalize_instrument("XAUUSD") == "GC"


def test_replay_page_has_propsim(client):
    html = client.get("/replay").get_data(as_text=True)
    assert "propsim-host" in html and "/static/t58-propsim.js" in html
    assert client.get("/static/t58-propsim.js").status_code == 200


def test_replay_page_splits_replay_and_propsim_into_two_tabs(client):
    html = client.get("/replay").get_data(as_text=True)
    assert 'id="rp-tab-replay"' in html and 'id="rp-tab-propsim"' in html
    assert 'id="rp-panel-replay"' in html and 'id="rp-panel-propsim"' in html
    # the simulator lives only in its own panel; the replay form lives only in the other
    replay_panel = html.split('id="rp-panel-replay"', 1)[1]
    propsim_panel = html.split('id="rp-panel-propsim"', 1)[1].split('id="rp-panel-replay"', 1)[0]
    assert "propsim-host" in propsim_panel and "/replay/prepare" not in propsim_panel
    assert "/replay/prepare" in replay_panel and 'id="propsim-host"' not in replay_panel
    assert html.count("<div") == html.count("</div>")


def test_propsim_engine_under_node(tmp_path):
    import shutil
    import subprocess
    from pathlib import Path
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    js = Path(__file__).resolve().parents[2] / "app" / "web" / "static" / "t58-propsim.js"
    prog = (
        "const S=require(process.argv[1]);"
        "const P={account:100000,targetPct:8,dailyLossPct:5,maxDdPct:10,ddType:'trailing',consistencyPct:30,minDays:5,winRate:0.45,payoffR:1.8,tradesPerDay:3,riskPct:0.5,maxDays:60,payoutPct:4,fundedDays:30,payoutMinDays:5};"
        "const out={};for(const r of [0.1,0.5,2]){P.riskPct=r;const m=S.monteCarlo(P,400,7);out[r]=[m.passPct,m.failPct,m.timeoutPct,m.payoutPct];}"
        "P.riskPct=0.5;const a=S.monteCarlo(P,400,7),b=S.monteCarlo(P,400,7);out.det=a.passPct===b.passPct;"
        "const st=S.newState(P);P.riskPct=1;out.floor=st.floor;"
        "console.log(JSON.stringify(out));"
    )
    p = subprocess.run(["node", "-e", prog, str(js)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    import json as _j
    o = _j.loads(p.stdout.strip().splitlines()[-1])
    assert o["det"] is True and o["floor"] == 90000
    assert o["2"][1] > 50 > o["0.5"][1], "raising risk per trade must raise the fail rate"
    assert o["0.1"][2] > 50, "tiny risk should mostly time out"


def test_education_has_interactive_labs(client):
    html = client.get("/education").get_data(as_text=True)
    assert 'id="labs-host"' in html and "/static/t58-edu-labs.js" in html and 'data-module="labs"' in html
    assert client.get("/static/t58-edu-labs.js").status_code == 200


def test_education_labs_are_sprinkled_through_the_modules(client):
    from app.education import content

    html = client.get("/education").get_data(as_text=True)
    # every slot references a real lesson, and every lab (1..5) is placed in some lesson
    lesson_ids = {l.id for s in content.list_sections() for l in s.lessons}
    assert set(content.LESSON_LAB_SLOTS) <= lesson_ids
    assert sorted(content.LESSON_LAB_SLOTS.values()) == [1, 2, 3, 4, 5]
    assert html.count('class="edu-inline-lab-host"') == len(content.LESSON_LAB_SLOTS)


def test_knowledge_check_is_its_own_page_not_in_the_labs_tab(client):
    page = client.get("/education/knowledge-check")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "quiz-host" in body and "mountQuiz" in body
    assert "labs-host" not in body
    assert 'href="/education/knowledge-check"' in client.get("/education").get_data(as_text=True)


def test_edu_labs_logic_under_node():
    import shutil
    import subprocess
    from pathlib import Path
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    js = Path(__file__).resolve().parents[2] / "app" / "web" / "static" / "t58-edu-labs.js"
    prog = ("const L=require(process.argv[1]);const r=L.rng(5);let px=[100];for(let i=1;i<1500;i++)px.push(px[i-1]+(r()-0.5)*2);"
            "const v=L.backtest(px,5,30,0,900);console.log(JSON.stringify({labs:L.labs.length,finite:Number.isFinite(v)}));")
    p = subprocess.run(["node", "-e", prog, str(js)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    import json as _j
    assert _j.loads(p.stdout.strip().splitlines()[-1]) == {"labs": 5, "finite": True}


def test_squeezemetrics_history_merge_is_lagged(client, monkeypatch):
    import pandas as pd
    csv_text = "date,price,dix,gex\n2026-09-01,6700,0.4,2000000000\n2026-09-02,6710,0.4,-1000000000\n"
    monkeypatch.setattr(gd, "fetcher", lambda url: csv_text.encode())
    r = client.post("/api/gamma/history/refresh").get_json()
    assert r["ok"] and r["rows"] == 2
    ts = pd.date_range("2026-09-01", periods=3 * 288, freq="5min", tz="America/Chicago")
    df = pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1})
    out = gd.merge_gamma_levels(df, "ES")
    d = ts.normalize()
    assert out.loc[d == d[0], "gex_net"].isna().all()                 # Sep 1 bars: no prior day yet
    assert (out.loc[d == d[288], "gex_net"] == 2e9).all()             # Sep 2 bars see Sep 1's value
    assert (out.loc[d == d[576], "gex_net"] == -1e9).all()            # Sep 3 bars see Sep 2's value
