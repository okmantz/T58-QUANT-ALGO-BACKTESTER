"""2026-11-05: trader profile, per-workflow memory, pre-flight, time estimate, run comparison."""
from __future__ import annotations

import json

import pytest

from app.web import workflow_support as WS


@pytest.fixture()
def cfg_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(WS, "_config_dir", lambda: tmp_path)
    return tmp_path


def test_profile_cleaning_drops_unknown_and_bad_values():
    out = WS.clean_profile({"account_size": "$100,000", "risk_per_trade_pct": "50", "instruments": "nq, es mnq", "typical_hold": "intraday",
                            "preset_key": "does_not_exist", "evil": "x", "session_start_hour": "9"})
    assert out["account_size"] == 100000.0 and "risk_per_trade_pct" not in out and "evil" not in out and "preset_key" not in out
    assert out["instruments"] == ["NQ", "ES", "MNQ"] and out["typical_hold"] == "intraday" and out["session_start_hour"] == 9.0


def test_profile_defaults_fill_account_and_risk(cfg_dir):
    (cfg_dir / "trader_profile.json").write_text(json.dumps({"account_size": 50000, "risk_per_trade_pct": 0.5}))
    d = WS.profile_defaults()
    assert d["account_size"] == 50000 and d["initial_balance"] == 50000 and d["risk_per_trade_pct"] == 0.5


def test_workflow_memory_round_trips_and_never_stores_secrets(cfg_dir):
    assert WS.remember("full_pipeline", {"dataset": "NQ/NQ5.csv", "ga_population": "20", "api_key": "SECRET", "webhook_url": "http://x", "email": "a@b.c"})
    row = WS.recall("full_pipeline")
    assert row["values"] == {"dataset": "NQ/NQ5.csv", "ga_population": "20"}
    assert WS.recall("unknown") == {}
    assert not WS.remember("../../etc", {"a": "b"}) or WS.recall("etc")   # path-like names are sanitised, never used as a path


def test_time_estimate_prefers_measured_runs(cfg_dir):
    prior = WS.estimate_seconds("full_pipeline", 50_000)
    assert "typical machine" in prior["basis"]
    for _ in range(3):
        WS.record_duration("full_pipeline", 50_000, 60.0)
    got = WS.estimate_seconds("full_pipeline", 100_000)
    assert "your last 3 run" in got["basis"] and 100 <= got["seconds"] <= 140
    WS.record_duration("quick_optimize", None, 90.0)
    assert "your last 1 run" in WS.estimate_seconds("quick_optimize", None)["basis"]


def test_preflight_flags_missing_dataset_and_adaptive_off(cfg_dir):
    checks = {c["id"]: c for c in WS.preflight("full_pipeline", dataset="nope/none.csv", adaptive=False, optimizer_mode="auto")}
    assert checks["dataset"]["status"] == "fail" and checks["adaptive"]["status"] == "warn" and checks["time"]["status"] == "ok"


def test_routes_respond(cfg_dir):
    from app.web.server import app
    c = app.test_client()
    assert c.get("/profile").status_code == 404  # the Trader profile page was removed; defaults still feed other pages
    WS._write("trader_profile.json", WS.clean_profile({"account_size": "25000", "instruments": "NQ"}))
    assert c.get("/api/profile/defaults").get_json()["defaults"]["account_size"] == 25000
    assert c.post("/api/workflow-memory/quick_optimize", json={"dataset": "x"}).get_json()["saved"] is True
    assert c.get("/api/workflow-memory/quick_optimize").get_json()["values"]["dataset"] == "x"
    assert c.get("/api/preflight?workflow=full_pipeline").get_json()["checks"]
    assert c.get("/api/time-estimate?workflow=full_pipeline&bars=20000").get_json()["seconds"] > 0
    assert c.get("/runs/compare").status_code == 200
    assert c.get("/runs/compare?ids=nope,nada").status_code == 200
    assert c.get("/api/runs/recent").status_code == 200


def test_every_adaptive_risk_checkbox_is_checked_by_default():
    import re
    from pathlib import Path
    for name in ("full_pipeline", "quick_optimize", "index", "refine", "evolution"):
        html = (Path("app/web/templates") / f"{name}.html").read_text(encoding="utf-8")
        tags = re.findall(r'<input[^>]*name="adaptive_risk_enabled"[^>]*>', html)
        assert tags and all(" checked" in t for t in tags), name


def test_adaptive_risk_defaults_on_in_configs():
    from app.evolution.engine import EvolutionConfig
    from app.orchestration.full_pipeline import FullPipelineConfig
    from app.orchestration.quick_optimize import QuickOptimizeConfig
    assert FullPipelineConfig().adaptive_risk_enabled and QuickOptimizeConfig().adaptive_risk_enabled
    assert EvolutionConfig.__dataclass_fields__["adaptive_risk_enabled"].default is True
