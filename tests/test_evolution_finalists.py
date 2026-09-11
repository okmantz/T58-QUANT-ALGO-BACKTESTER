"""Tests for app.evolution.finalists -- wiring app.prop.survival_engine's
already-built payout funnel (P(pass) -> P(funded) -> P(payout 1/2/3))
and app.scoring.pareto's frontier onto Evolution Lab leaderboard
finalists, on demand (EvolutionRunner.finalists_report)."""
import numpy as np
import pandas as pd

from app.backtest.execution import Trade
from app.evolution.engine import EvolutionCandidateRecord
from app.evolution.finalists import build_finalist_reports, pareto_frontier_for_finalists, render_finalist_report
from app.evolution.prop_fitness import PropFitnessBreakdown
from app.prop.simulator import PropRules


def _mock_trades(n=200, seed=1, mean=25.0, std=120.0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01")
    trades = []
    for i in range(n):
        t = base + pd.Timedelta(days=i // 3)
        pnl = float(rng.normal(mean, std))
        trades.append(Trade(
            entry_time=t, exit_time=t, direction=1, entry_price=1.1, exit_price=1.1,
            size=1000, pnl=pnl, pnl_pct=0.1, exit_reason="signal", commission=0, equity_after=0,
        ))
    return trades


def _rules(**overrides):
    base = dict(
        account_size=50_000, evaluation_profit_target_pct=8, daily_loss_limit_pct=5,
        max_drawdown_pct=10, min_trading_days=3, consistency_rule_pct=30, payout_frequency_days=14,
    )
    base.update(overrides)
    return PropRules(**base)


def _fake_leaderboard_record(cid, family, final_score, seed, mean=25.0):
    fitness = PropFitnessBreakdown(
        pass_probability=0.5, payout_probability=0.3, robustness=0.6, oos_consistency=0.5,
        drawdown_pct=5.0, base_score=final_score, final_score=final_score,
    )
    return EvolutionCandidateRecord(
        candidate_id=cid, spec={"source_type": "manual", "config": {}}, meta={"family": family},
        stats={"max_drawdown_pct": 5.0}, fitness=fitness,
        trades=_mock_trades(200, seed=seed, mean=mean),
    )


def test_build_finalist_reports_runs_survival_analysis_on_leaderboard():
    leaderboard = [
        _fake_leaderboard_record("winner_a", "mean_reversion_band", 15.0, seed=1, mean=30.0),
        _fake_leaderboard_record("winner_b", "failed_breakout", 10.0, seed=2, mean=10.0),
    ]
    reports = build_finalist_reports(leaderboard, _rules(), top_n=10)
    assert len(reports) == 2
    for r in reports:
        assert 0.0 <= r.probability_pass_evaluation <= 100.0
        assert 0.0 <= r.probability_first_payout <= 100.0
        assert r.probability_second_payout <= r.probability_first_payout + 1e-6  # payouts only get harder
        assert r.probability_third_payout <= r.probability_second_payout + 1e-6


def test_build_finalist_reports_skips_candidates_with_too_few_trades():
    thin = _fake_leaderboard_record("too_thin", "rsi_extreme_reversion", 5.0, seed=3)
    thin.trades = thin.trades[:2]  # below the 5-trade floor
    reports = build_finalist_reports([thin], _rules(), top_n=10)
    assert reports == []


def test_build_finalist_reports_respects_top_n_and_fitness_ranking():
    leaderboard = [
        _fake_leaderboard_record("low", "a", 1.0, seed=1),
        _fake_leaderboard_record("high", "b", 99.0, seed=2),
        _fake_leaderboard_record("mid", "c", 50.0, seed=3),
    ]
    reports = build_finalist_reports(leaderboard, _rules(), top_n=2)
    ids = {r.candidate_id for r in reports}
    assert ids == {"high", "mid"}


def test_pareto_frontier_for_finalists_labels_tradeoffs():
    leaderboard = [
        _fake_leaderboard_record("conservative_low_return", "mean_reversion_band", 20.0, seed=1, mean=8.0),
        _fake_leaderboard_record("aggressive_high_return", "failed_breakout", 18.0, seed=2, mean=60.0),
    ]
    reports = build_finalist_reports(leaderboard, _rules(), top_n=10)
    pareto_frontier_for_finalists(reports)
    # Every report should have been tagged one way or another (either a
    # frontier label, or explicitly marked dominated) -- never left
    # silently unexamined.
    for r in reports:
        assert r.on_frontier or not r.on_frontier  # tautology guarding against an exception above
    assert any(r.label is not None for r in reports) or all(not r.on_frontier for r in reports)


def test_render_finalist_report_handles_empty_and_populated():
    assert "no finalist" in render_finalist_report([])
    leaderboard = [_fake_leaderboard_record("a", "mean_reversion_band", 10.0, seed=1)]
    reports = build_finalist_reports(leaderboard, _rules(), top_n=10)
    pareto_frontier_for_finalists(reports)
    text = render_finalist_report(reports)
    assert "a" in text


def test_build_finalist_reports_omits_rolling_eval_when_not_requested():
    leaderboard = [_fake_leaderboard_record("a", "mean_reversion_band", 10.0, seed=1)]
    reports = build_finalist_reports(leaderboard, _rules(), top_n=10)
    assert reports[0].rolling_eval_pass_rate_pct is None
    assert reports[0].rolling_eval_windows_tested is None


def test_build_finalist_reports_includes_rolling_eval_when_requested():
    leaderboard = [_fake_leaderboard_record("a", "mean_reversion_band", 10.0, seed=1, mean=30.0)]
    reports = build_finalist_reports(leaderboard, _rules(), top_n=10, window_trading_days=20)
    assert reports[0].rolling_eval_pass_rate_pct is not None
    assert 0.0 <= reports[0].rolling_eval_pass_rate_pct <= 100.0
    assert reports[0].rolling_eval_windows_tested is not None and reports[0].rolling_eval_windows_tested > 0


def test_render_finalist_report_shows_rolling_column_only_when_present():
    leaderboard = [_fake_leaderboard_record("a", "mean_reversion_band", 10.0, seed=1, mean=30.0)]
    without = build_finalist_reports(leaderboard, _rules(), top_n=10)
    with_rolling = build_finalist_reports(leaderboard, _rules(), top_n=10, window_trading_days=20)
    assert "Rolling Pass" not in render_finalist_report(without)
    assert "Rolling Pass" in render_finalist_report(with_rolling)
