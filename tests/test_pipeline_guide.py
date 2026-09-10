"""Tests for app.orchestration.pipeline_guide -- the shared "what's next"
guidance text used by both the web app and the desktop app at every stage
of create -> promote -> validate -> optimize -> Full Pipeline -> champion."""
from app.orchestration import pipeline_guide as pg


def test_after_evolution_stop_empty_leaderboard():
    msg = pg.after_evolution_stop(0)
    assert "no candidate" in msg.lower()
    assert "promote" not in msg.lower() or "nothing to promote" in msg.lower()


def test_after_evolution_stop_with_leaderboard():
    msg = pg.after_evolution_stop(5)
    assert "PROMOTE" in msg
    assert "5" in msg


def test_after_promote_to_library_mentions_filename_and_full_pipeline():
    msg = pg.after_promote_to_library("my_strategy.json")
    assert "my_strategy.json" in msg
    assert "Full Pipeline" in msg


def test_after_search_complete_with_champion():
    msg = pg.after_search_complete("cand_123", 10)
    assert "cand_123" in msg
    assert "PROMOTE" in msg


def test_after_search_complete_no_champion_but_leaderboard():
    msg = pg.after_search_complete(None, 3)
    assert "3 candidate" in msg


def test_after_search_complete_totally_empty():
    msg = pg.after_search_complete(None, 0)
    assert "Stage 1" in msg


def test_after_full_pipeline_ready():
    msg = pg.after_full_pipeline("READY", saved_to_library=True)
    assert "READY" in msg
    assert "forward-test" in msg
    assert "saved to the Strategy Library" in msg


def test_after_full_pipeline_marginal():
    msg = pg.after_full_pipeline("MARGINAL", saved_to_library=False)
    assert "MARGINAL" in msg
    assert "Quick Optimize" in msg


def test_after_full_pipeline_not_ready():
    msg = pg.after_full_pipeline("NOT READY", saved_to_library=False)
    assert "NOT READY" in msg
    assert "Evolution Lab or Search Lab" in msg


def test_after_full_pipeline_batch_prefers_best_ready():
    outcomes = [
        {"label": "a", "ok": True, "verdict": "MARGINAL", "eval_pass_probability": 40.0},
        {"label": "b", "ok": True, "verdict": "READY", "eval_pass_probability": 61.5},
        {"label": "c", "ok": False, "verdict": None, "eval_pass_probability": 0.0},
    ]
    msg = pg.after_full_pipeline_batch(outcomes)
    assert "'b'" in msg
    assert "61.5%" in msg


def test_after_full_pipeline_batch_no_ready_falls_back_to_marginal():
    outcomes = [{"label": "a", "ok": True, "verdict": "MARGINAL", "eval_pass_probability": 40.0}]
    msg = pg.after_full_pipeline_batch(outcomes)
    assert "MARGINAL" in msg


def test_after_full_pipeline_batch_all_failed():
    outcomes = [{"label": "a", "ok": False, "verdict": None, "eval_pass_probability": 0.0}]
    msg = pg.after_full_pipeline_batch(outcomes)
    assert "Evolution Lab or Search Lab" in msg
