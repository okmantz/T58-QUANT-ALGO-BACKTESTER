"""Tests for app.orchestration.survivors -- the pipeline funnel."""
from __future__ import annotations

from app.orchestration import survivors as sv
from app.strategy.library import compute_pipeline_progress


def test_empty_library():
    out = sv.build_funnel([])
    assert out["total"] == 0 and out["ready_verdict_count"] == 0
    assert all(s["count"] == 0 and s["pct_of_total"] == 0.0 for s in out["stages"])
    assert [s["key"] for s in out["stages"]][0] == "create"


def test_counts_match_the_per_strategy_progress_exactly():
    metas = [
        {},
        {"last_run": {"verdict": "READY"}},
        {"last_run": {}, "last_search": {"x": 1}, "last_deploy": {"at": 1}},
        {"last_run": {"verdict": "MARGINAL"}, "last_optimize": {"x": 1}},
    ]
    out = sv.build_funnel(metas)
    by_key = {s["key"]: s["count"] for s in out["stages"]}
    for key in by_key:
        expected = sum(1 for m in metas for st in compute_pipeline_progress(m)["stages"] if st["key"] == key and st["done"])
        assert by_key[key] == expected, key
    assert by_key["create"] == 4 and out["total"] == 4
    assert out["ready_verdict_count"] == 1


def test_percentages_are_of_the_whole_library():
    out = sv.build_funnel([{"last_run": {"verdict": "READY"}}, {}, {}, {}])
    test_stage = next(s for s in out["stages"] if s["key"] == "test")
    assert test_stage["count"] == 1 and test_stage["pct_of_total"] == 25.0


def test_none_metadata_does_not_raise():
    assert sv.build_funnel([None, {}])["total"] == 2


def test_load_library_funnel_reads_the_real_library(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("app.strategy.library.list_saved_strategies",
                        lambda: [SimpleNamespace(metadata={"last_run": {"verdict": "READY"}})])
    out = sv.load_library_funnel()
    assert out["total"] == 1 and out["ready_verdict_count"] == 1
