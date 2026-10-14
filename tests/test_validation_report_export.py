"""The Validate run's downloadable report: document content, rendering, saving, and the web routes."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import app.web.server as server
from app.validation import report_export, simple_report


def _results():
    return [
        {"check": "cpcv", "ok": True, "error": None, "summary": {
            "n_paths": 6, "metric": "profit_factor", "mean_oos_metric": 0.8, "median_oos_metric": 0.82,
            "mean_is_metric": 1.9, "is_robust": False}},
        {"check": "regime", "ok": True, "error": None, "summary": {
            "n_cells": 25, "n_disable_recommended": 2, "notes": [],
            "disable_cells": [{"label": "High / Ranging", "dims": {"volatility": "high", "environment": "ranging"},
                               "n_trades": 54, "profit_factor": 0.86, "net_profit": -3835.0, "t_stat": -0.4}],
            "cells": []}},
        {"check": "walk_forward", "ok": True, "error": None, "summary": {
            "out_of_sample_efficiency": 0.62, "n_folds_completed": 3,
            "folds": [{"fold_index": 0, "test_trade_count": 21, "test_net_profit": 1200.0}]}},
        {"check": "pbo", "ok": False, "error": "PBO needs at least 2 candidates; this strategy has no tunable parameters to perturb."},
    ]


def _doc(**kw):
    results = _results()
    report = simple_report.build_report(results, context={})
    report["strategy"] = "My Strategy.json"
    raw = {r["check"]: r["summary"] for r in results if r["ok"]}
    return report_export.build_report_document(
        report, raw=raw, baseline={"trades": 120, "net_profit": 5000.0, "win_rate": 44.0, "max_drawdown_pct": 6.1},
        meta={"strategy": "My Strategy.json", "dataset": "NQ 5m", "timeframe": "5m", "prop_preset": "topstep_50k"},
        form={"cpcv_groups": "4", "mc_sims": "500", "api_key": "SECRET", "check_cpcv": "on"},
        requested_checks=["cpcv", "regime", "walk_forward", "pbo", "wfga"], simulation_text="sim text", **kw)


def test_document_lists_every_check_with_status_bar_and_raw_numbers():
    doc = _doc()
    by = {c["check"]: c for c in doc["checks"]}
    assert set(by) == {"cpcv", "regime", "walk_forward", "pbo"}
    assert by["cpcv"]["status"] == "fail" and by["cpcv"]["verdict"] == "FAIL"
    assert by["walk_forward"]["status"] == "pass"
    assert by["pbo"]["status"] == "error" and by["pbo"]["why_not_pass"]          # did-not-run keeps its reason
    assert by["cpcv"]["bar"] == simple_report.CHECK_BARS["cpcv"]                 # bar shown even for passing checks
    assert by["walk_forward"]["bar"] == simple_report.CHECK_BARS["walk_forward"]
    assert by["cpcv"]["raw"]["mean_oos_metric"] == 0.8                            # the numbers behind the score
    assert by["regime"]["raw"]["disable_cells"][0]["label"] == "High / Ranging"
    assert doc["checks_not_run"] == ["Walk-Forward GA"]                           # ticked, but produced nothing
    assert doc["overall"]["counts"]["fail"] >= 1


def test_form_snapshot_drops_secrets_and_keeps_settings():
    doc = _doc()
    assert "api_key" not in doc["form_snapshot"]
    assert doc["form_snapshot"]["cpcv_groups"] == "4"
    assert doc["settings"]["parameters"]["mc_sims"] == "500"


def test_html_is_self_contained_and_escapes_values():
    doc = _doc()
    doc["checks"][0]["headline"] = "<script>alert(1)</script>"
    page = report_export.render_html(doc)
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    for needle in ("Scorecard", "CPCV", "Regime Matrix", "Bar to pass", "High / Ranging", "Raw numbers", "My Strategy.json"):
        assert needle in page, needle
    assert "http://" not in page and "https://" not in page          # no external assets: opens offline


def test_text_report_has_scorecard_and_reasons():
    txt = report_export.render_text(_doc())
    assert "SCORECARD" in txt and "[FAIL" in txt and "[PASS" in txt and "DID NOT RUN" in txt
    assert "WHY:" in txt


def test_save_load_list_roundtrip_and_traversal_blocked(tmp_path):
    doc = _doc()
    paths = report_export.save_report(tmp_path, doc)
    assert set(paths) == {"json", "html", "txt"} and all(p.is_file() for p in paths.values())
    rid = doc["report_id"]
    assert report_export.load_report(tmp_path, rid)["strategy"]["name"] == "My Strategy.json"
    assert report_export.report_path(tmp_path, rid, "html") is not None
    rows = report_export.list_reports(tmp_path)
    assert rows and rows[0]["report_id"] == rid and rows[0]["has_html"]
    # path traversal / odd ids are refused
    for bad in ("../x", "a/b", "..", "", "x" * 400, "a b"):
        assert report_export.safe_report_id(bad) is None
        assert report_export.report_path(tmp_path, bad, "json") is None
    assert report_export.report_path(tmp_path, rid, "exe") is None


def test_clean_json_handles_nan_and_numpy():
    import numpy as np
    out = report_export.clean_json({"a": float("nan"), "b": np.float64(1.5), "c": (1, 2), "d": np.int64(3)})
    assert out == {"a": None, "b": 1.5, "c": [1, 2], "d": 3}
    json.dumps(out)


def test_autofix_section_renders_in_html_and_text():
    af = {"accepted": True, "headline": "Fixed: Regime Matrix now passes.", "source_report_id": "orig",
          "stages": [{"label": "Regime filter", "accepted": True, "headline": "kept", "reasons": [], "details": {"x": 1}}],
          "before": [{"check": "regime", "label": "Regime Matrix", "status": "fail", "headline": "4 regimes"}],
          "after": [{"check": "regime", "label": "Regime Matrix", "status": "pass", "headline": "none"}], "notes": ["n1"]}
    doc = _doc(auto_fix=af)
    page, txt = report_export.render_html(doc), report_export.render_text(doc)
    assert "Auto-Fix" in page and "Same checks, before vs after" in page and "Regime filter" in page
    assert "AUTO-FIX: KEPT" in txt


# ---------------------------------------------------------------- web routes

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "VALIDATION_RUNS_DIR", tmp_path)
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def test_validate_run_saves_a_downloadable_report(client, tmp_path, monkeypatch):
    import pandas as pd
    from tests.validate_job_helper import wait_for_validate_job

    df = pd.DataFrame({"open": [1.0] * 30, "high": [1.0] * 30, "low": [1.0] * 30, "close": [1.0] * 30, "volume": [1.0] * 30})
    monkeypatch.setattr(server, "_resolve_dataset", lambda form, files: (df, "tiny.csv", "", None))
    monkeypatch.setattr(server, "load_strategy_text", lambda t, f: "code")
    monkeypatch.setattr(server, "build_strategy_from_code", lambda t, code, **k: SimpleNamespace(name="Stub"))
    stats = SimpleNamespace(net_profit=10.0, win_rate=50.0, max_drawdown_pct=1.0)
    monkeypatch.setattr(server, "run_backtest", lambda df_, s, risk: SimpleNamespace(trades=[], statistics=stats))
    import app.validation.runner as vrunner
    monkeypatch.setattr(vrunner, "run_cpcv", lambda *a, **k: SimpleNamespace(
        n_paths=3, metric="profit_factor", mean_oos_metric=0.7, median_oos_metric=0.7, mean_is_metric=1.8, is_robust=False))

    r = client.post("/validate-simple/start", data={
        "strategy_pick": "python::stub.py", "existing_dataset": "tiny.csv", "check_cpcv": "on",
        "initial_balance": "100000", "account_size": "100000", "cpcv_groups": "4"})
    job = wait_for_validate_job(client, r)
    assert not job["error"], job["error"]
    rep = job["report"]
    assert rep["report_id"] and set(rep["downloads"]) == {"json", "html", "txt"}
    assert "cpcv" in rep["fixable_checks"]                                  # failed overfit check -> Auto-Fix offered

    html = client.get(rep["downloads"]["html"])
    assert html.status_code == 200 and "CPCV" in html.get_data(as_text=True)
    assert "attachment" in client.get(rep["downloads"]["html"] + "?download=1").headers.get("Content-Disposition", "")
    data = client.get(rep["downloads"]["json"])
    assert data.status_code == 200
    doc = json.loads(data.get_data(as_text=True))
    assert doc["checks"][0]["raw"]["mean_oos_metric"] == 0.7                # the raw numbers made it to disk
    assert doc["form_snapshot"]["cpcv_groups"] == "4"                       # settings kept so Auto-Fix can re-run them
    assert client.get(rep["downloads"]["txt"]).status_code == 200
    listing = client.get("/validate-simple/reports").get_data(as_text=True)
    assert rep["report_id"] in listing


def test_report_route_404s_for_unknown_or_unsafe_ids(client):
    assert client.get("/validate-simple/report/nope/html").status_code == 404
    assert client.get("/validate-simple/report/..%2F..%2Fetc/html").status_code == 404
    assert client.get("/validate-simple/report/x/exe").status_code == 404
