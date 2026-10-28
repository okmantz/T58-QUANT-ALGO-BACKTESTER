from __future__ import annotations

import time

from app.web.job_manager import JobManager


def test_compact_keeps_unsaved_results_and_evicts_saved():
    jm = JobManager()
    old = time.time() - 7200
    unsaved = jm.create(tool="Discovery", instrument="ES", started_at=old)
    jm.update(unsaved, lab_result={"winners": [1]})
    jm.finish(unsaved)
    saved = jm.create(tool="Full Pipeline", instrument="ES", started_at=old)
    jm.update(saved, report_html="/reports/x.html", result={"k": 1})
    jm.finish(saved)
    young = jm.create(tool="Validate", instrument="ES")
    jm.update(young, report_html="/reports/y.html")
    jm.finish(young)
    running = jm.create(tool="Optimize", instrument="ES", started_at=old)

    out = jm.compact(1800)
    ids = {j["job_id"] for j in jm.list_jobs()}
    assert unsaved in ids and young in ids and running in ids
    assert saved not in ids and out["evicted"] == 1
    assert [p["job_id"] for p in out["protected"]] == [unsaved]


def test_free_memory_report_shape_and_refusal(tmp_path, monkeypatch):
    from app.web.server import app

    monkeypatch.setenv("T58_JOB_HISTORY_PATH", str(tmp_path / "jh.json"))
    app.config["TESTING"] = True
    c = app.test_client()
    r = c.post("/api/memory/free", json={"unload_ollama": False})
    assert r.status_code in (200, 409)
    if r.status_code == 200:
        d = r.get_json()
        assert d["ok"] and d["headline"] and isinstance(d["steps"], list)
        assert any("never touched" in k for k in d["kept"])
        assert "before" in d and "after" in d
    assert c.get("/runs").status_code == 200
