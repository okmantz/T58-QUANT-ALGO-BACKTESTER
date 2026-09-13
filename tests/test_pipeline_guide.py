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


def test_after_first_backtest_zero_trades():
    msg = pg.after_first_backtest({"trade_count": 0})
    assert "zero trades" in msg.lower()


def test_after_first_backtest_too_few_trades():
    msg = pg.after_first_backtest({"trade_count": 5, "profit_factor": 1.5, "max_drawdown_pct": 5})
    assert "5 trade" in msg
    assert "too few" in msg.lower()


def test_after_first_backtest_losing():
    msg = pg.after_first_backtest({"trade_count": 50, "profit_factor": 0.8, "max_drawdown_pct": 5})
    assert "0.80" in msg
    assert "loses money" in msg.lower()


def test_after_first_backtest_severe_drawdown():
    msg = pg.after_first_backtest({"trade_count": 50, "profit_factor": 1.4, "max_drawdown_pct": 55})
    assert "55.0%" in msg
    assert "drawdown" in msg.lower()


def test_after_first_backtest_failed_eval():
    msg = pg.after_first_backtest(
        {"trade_count": 50, "profit_factor": 1.4, "max_drawdown_pct": 12}, passed_evaluation=False,
    )
    assert "failed the prop-firm simulation" in msg


def test_after_first_backtest_looks_good():
    msg = pg.after_first_backtest(
        {"trade_count": 80, "profit_factor": 1.6, "max_drawdown_pct": 10}, passed_evaluation=True,
    )
    assert "pass probability" in msg.lower()
    assert "Full Pipeline" in msg


def test_strategy_creation_recommendation_with_idea():
    msg = pg.strategy_creation_recommendation(True)
    assert "Manual Builder" in msg


def test_strategy_creation_recommendation_without_idea():
    msg = pg.strategy_creation_recommendation(False)
    assert "Search Lab" in msg
    assert "Evolution Lab" in msg


def test_after_forward_test_with_errors():
    msg = pg.after_forward_test({"trade_count": 3, "errors": ["symbol not found"]})
    assert "1 error" in msg


def test_after_forward_test_no_trades():
    msg = pg.after_forward_test({"trade_count": 0, "errors": []})
    assert "no trades" in msg.lower()


def test_after_forward_test_mismatch():
    msg = pg.after_forward_test({"trade_count": 10, "errors": [], "matched_backtest": False})
    assert "diverged" in msg.lower()


def test_after_forward_test_clean():
    msg = pg.after_forward_test({"trade_count": 10, "errors": [], "matched_backtest": True})
    assert "Deploy Live" in msg


def test_after_first_backtest_zero_trades_names_tabs():
    msg = pg.after_first_backtest({"trade_count": 0})
    assert "1 Strategy Configuration" in msg
    assert "2 Market Data" in msg


def test_after_first_backtest_losing_names_specific_actions():
    msg = pg.after_first_backtest({"trade_count": 50, "profit_factor": 0.8, "max_drawdown_pct": 5})
    assert "5 Run & Report" in msg
    assert "win rate" in msg.lower()


def test_after_first_backtest_drawdown_names_tab_and_lever():
    msg = pg.after_first_backtest({"trade_count": 50, "profit_factor": 1.4, "max_drawdown_pct": 55})
    assert "4 Risk & Execution" in msg
    assert "position size" in msg.lower() or "position-size" in msg.lower()


def test_after_evolution_stop_no_pbo_note_for_small_leaderboard():
    msg = pg.after_evolution_stop(1)
    assert "PBO" not in msg


def test_after_evolution_stop_pbo_note_uses_total_evaluated():
    msg = pg.after_evolution_stop(5, total_evaluated=300)
    assert "PBO" in msg
    assert "300 candidates" in msg


def test_after_evolution_stop_pbo_note_falls_back_to_leaderboard_size():
    msg = pg.after_evolution_stop(5)
    assert "PBO" in msg
    assert "5 candidates" in msg


def test_after_search_complete_champion_pbo_note_uses_total_candidates():
    msg = pg.after_search_complete("cand_123", 10, total_candidates=250)
    assert "PBO" in msg
    assert "250 candidates" in msg
    assert "coin flip" in msg.lower()


def test_after_search_complete_no_champion_leaderboard_pbo_note():
    msg = pg.after_search_complete(None, 3, total_candidates=40)
    assert "PBO" in msg


def test_after_search_complete_single_candidate_no_pbo_note():
    msg = pg.after_search_complete("cand_1", 1, total_candidates=1)
    assert "PBO" not in msg


def test_should_show_first_run_welcome_true_when_nothing_exists():
    assert pg.should_show_first_run_welcome(has_stored_datasets=False, has_run_history=False) is True


def test_should_show_first_run_welcome_false_once_data_loaded():
    assert pg.should_show_first_run_welcome(has_stored_datasets=True, has_run_history=False) is False


def test_should_show_first_run_welcome_false_once_a_run_exists():
    assert pg.should_show_first_run_welcome(has_stored_datasets=False, has_run_history=True) is False


def test_first_run_welcome_names_the_first_three_tabs():
    msg = pg.first_run_welcome()
    assert "2 Market Data" in msg
    assert "1 Strategy Configuration" in msg
    assert "5 Run & Report" in msg
    assert "/user-manual" in msg
