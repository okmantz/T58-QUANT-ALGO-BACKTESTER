"""Regression tests for: persisted run history, equity curves for every run, retire-to-graveyard, runtime license URL."""
from __future__ import annotations

import pytest


def test_job_history_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("T58_JOB_HISTORY_PATH", str(tmp_path / "jh.json"))
    from app.web import job_manager as jm
    mgr = jm.JobManager() if hasattr(jm, "JobManager") else jm.JOB_MANAGER
    ids = [mgr.create(page_template="/x/{job_id}", tool="t", instrument="ES") for _ in range(7)]
    for i in ids:
        mgr.finish(i)
    assert len([h for h in jm.load_job_history() if h.get("job_id") in ids or h.get("id") in ids]) == 7


def test_top_equity_not_capped_at_five():
    import inspect

    from app.reports import run_history
    src = inspect.getsource(run_history)
    assert "[:5]" not in src.split("top_equity")[1][:600]


def test_sheet_url_runtime_save_and_resolve(tmp_path, monkeypatch):
    from app.licensing import client
    monkeypatch.delenv("T58_LICENSE_SHEET_URL", raising=False)
    monkeypatch.setattr(client, "_sheet_url_files", lambda: [tmp_path / "license_sheet_url.txt"])
    ok, _ = client.save_sheet_url("https://script.google.com/macros/s/ABC123/exec")
    assert ok
    assert client.resolve_sheet_url() == "https://script.google.com/macros/s/ABC123/exec"
    ok, _ = client.save_sheet_url("not a url")
    assert not ok


def test_retire_requires_known_strategy():
    from app.strategy.retire import retire_to_graveyard
    with pytest.raises((ValueError, FileNotFoundError)):
        retire_to_graveyard("python", "does_not_exist.py", "test")
