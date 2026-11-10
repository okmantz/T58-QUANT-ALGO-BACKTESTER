import json

import pytest

from app.discovery_engine import surrogate as sg
from app.validation import research_integrity as ri
from app.web.server import app


@pytest.fixture(autouse=True)
def tmp_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(sg, "report_path", lambda: tmp_path / "surrogate_gate.json")


def test_benchmark_is_reproducible_and_small_runs_are_flagged():
    a = ri.run_benchmark(n_null=6, n_planted=2, targets=(0.30,), days=60, seed=3)
    b = ri.run_benchmark(n_null=6, n_planted=2, targets=(0.30,), days=60, seed=3)
    for k in ("null_candidates", "planted", "dataset_fingerprint", "version"):
        assert a[k] == b[k]
    assert a["passed"] is False and any("null runs" in r for r in a["reasons"])     # 6 runs can never pass the gate
    assert a["kind"] == "statistical_chain_benchmark" and "not the production engine" in a["label"]


def test_chain_has_no_lookahead_and_finds_a_strong_planted_edge():
    null_hits = sum(ri.fast_chain({"a": sg.make_null_frame(250, 100 + i)})["candidate"] for i in range(20))
    assert null_hits <= 2
    f = sg.make_planted_frame(250, 7, 0.20)
    assert ri.fast_chain({"a": f})["candidate"] is True


def test_fingerprint_changes_with_settings():
    assert ri.dataset_fingerprint(250) != ri.dataset_fingerprint(80) and ri.dataset_fingerprint() == ri.dataset_fingerprint(250)


def test_dashboard_with_nothing_measured_says_so():
    d = ri.dashboard()
    assert d["gate"]["open"] is False and d["false_positive"]["measured"] is False
    assert d["real_data_fdr"]["measured"] is False and "Not measured" in d["real_data_fdr"]["text"]
    assert d["benchmark"] is None and d["last_successful_validation"] is None


def test_dashboard_with_a_saved_study(tmp_path):
    rep = sg.SurrogateReport(created_at=__import__("time").time(), code_hash=sg.code_hash(), n_null=60, null_candidates=1, null_rate=1 / 60,
                             null_rate_upper95=sg.clopper_pearson_upper(1, 60), planted=[{"target_r": 0.2, "measured_r": 0.21, "b": 0.05, "runs": 10, "found": 7}],
                             power={"0.20": 0.7, "0.30": 0.9})
    sg.save_report(rep)
    d = ri.dashboard()
    assert d["gate"]["open"] is True and d["false_positive"]["measured"] and d["engine"]["report_code_matches_current"]
    assert d["detection"]["by_size"][0]["detection_rate"] == 0.7 and d["last_successful_validation"] is not None
    ri.write_progress(10, 100, True)
    assert ri.dashboard()["null_test_progress"] == {"done": 10, "total": 100, "running": True, "completed_runs": 60, "required_runs": 60}


def test_stale_progress_is_not_running(tmp_path):
    (tmp_path / "surrogate_progress.json").write_text(json.dumps({"done": 1, "total": 5, "running": True, "updated_at": 0}))
    assert ri.dashboard()["null_test_progress"]["running"] is False


def test_benchmark_save_load_and_currency():
    res = ri.run_benchmark(n_null=3, n_planted=1, targets=(0.30,), days=60)
    ri.save_benchmark(res)
    assert ri.load_benchmark()["version"] == ri.engine_version()
    assert ri.dashboard()["benchmark"]["current"] is True
    res["version"] = "bench1-oldcode"
    ri.save_benchmark(res)
    assert ri.dashboard()["benchmark"]["current"] is False        # a result from older benchmark code is never shown as current


def test_routes():
    c = app.test_client()
    assert c.get("/research-integrity").status_code == 200
    j = c.get("/api/research-integrity").json
    assert j["ok"] and "real_data_fdr" in j
