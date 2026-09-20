import pytest

from app.search.budget_allocator import (
    allocate_search_budget,
    normalize_evolution_record,
    pooled_cross_instrument_leaderboard,
)


def test_allocate_search_budget_splits_evenly_across_jobs():
    plans = allocate_search_budget(1200, ["A", "B", "C"], min_population=10, default_population=60)
    assert len(plans) == 3
    for plan in plans:
        assert plan.evaluation_budget <= 1200 // 3
    total = sum(p.evaluation_budget for p in plans)
    assert total <= 1200


def test_allocate_search_budget_more_jobs_means_smaller_per_job_share():
    two_jobs = allocate_search_budget(1000, ["A", "B"])
    five_jobs = allocate_search_budget(1000, ["A", "B", "C", "D", "E"])
    assert five_jobs[0].evaluation_budget <= two_jobs[0].evaluation_budget


def test_allocate_search_budget_floors_population_even_if_it_overspends():
    plans = allocate_search_budget(10, ["A", "B"], min_population=20)
    for plan in plans:
        assert plan.population_size == 20
        assert plan.max_generations == 1


def test_allocate_search_budget_rejects_bad_input():
    with pytest.raises(ValueError):
        allocate_search_budget(0, ["A"])
    with pytest.raises(ValueError):
        allocate_search_budget(100, [])


def test_normalize_evolution_record_shapes_a_checkpoint_dict():
    checkpoint = {
        "candidate_id": "abc123",
        "spec": {"source_type": "manual", "config": {"name": "test"}},
        "meta": {"family": "rsi_extreme_reversion"},
        "fitness": {"final_score": 0.42},
        "stats": {"net_profit": 100.0},
        "mc_summary": {"evaluation_pass_probability": 0.8},
    }
    rec = normalize_evolution_record(checkpoint, instrument_label="ES/15m")
    assert rec["candidate_id"] == "abc123"
    assert rec["family"] == "rsi_extreme_reversion"
    assert rec["source_type"] == "manual"
    assert rec["config"] == {"name": "test"}
    assert rec["composite_score"] == 0.42
    assert rec["instrument"] == "ES/15m"


def test_pooled_cross_instrument_leaderboard_caps_family_globally():
    job_records = {
        "ES/15m": [
            {"candidate_id": "es_1", "family": "rsi_extreme_reversion", "composite_score": 0.9},
            {"candidate_id": "es_2", "family": "rsi_extreme_reversion", "composite_score": 0.5},
        ],
        "GC/15m": [
            {"candidate_id": "gc_1", "family": "rsi_extreme_reversion", "composite_score": 0.95},
            {"candidate_id": "gc_2", "family": "macd_cross_trend", "composite_score": 0.7},
        ],
    }
    pooled = pooled_cross_instrument_leaderboard(job_records, max_per_family=1, top_n=10)
    ids = {rec["candidate_id"] for rec in pooled}
    assert ids == {"gc_1", "gc_2"}
    assert pooled[0]["candidate_id"] == "gc_1"  # highest score first
    assert all("instrument" in rec for rec in pooled)
