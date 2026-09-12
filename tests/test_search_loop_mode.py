"""Tests for Search Lab's new "Loop Mode" (app.orchestration.loop_runner) --
Search Lab's counterpart to Evolution Lab's target_eval_pass_pct (see
tests/test_evolution_loop_mode.py), except Search Lab has no internal
generation loop to teach a stop condition to, so this is a real outer
loop around repeated run_search() calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration.loop_runner import SearchLoopConfig, run_search_loop
from app.prop.simulator import PropRules
from app.search.batch_runner import SearchStageConfig


def _trending_df(n=2500, seed=3, drift=0.00015):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.00003)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00002))
        l = min(o, c) - abs(rng.normal(0, 0.00002))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _flat_df(n=2500, seed=5):
    """Pure noise, no drift -- nothing here should ever clear a real
    target, used for the max_rounds/widening tests."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 1.1000 + np.cumsum(rng.normal(0, 0.00003, n))
    high = close + abs(rng.normal(0, 0.00002, n))
    low = close - abs(rng.normal(0, 0.00002, n))
    openp = close + rng.normal(0, 0.00001, n)
    return pd.DataFrame({
        "timestamp": ts, "open": openp, "high": high, "low": low, "close": close, "volume": 100.0,
    })


def _fast_stage_cfg(**overrides) -> SearchStageConfig:
    base = dict(
        min_trades=3, min_profit_factor=0.5, max_drawdown_buffer_mult=5.0,
        stage1_top_n=6, ga_population=4, ga_generations=1, ga_search_sims=30,
        stage2_top_n=3, full_mc_sims=50, walk_forward_folds=0, robustness_neighbors=0,
        workers=1, random_seed=42,
    )
    base.update(overrides)
    return SearchStageConfig(**base)


@pytest.fixture(autouse=True)
def _hermetic_app_dirs(tmp_path, monkeypatch):
    """Every graveyard write and family-health read/write in these tests
    must stay inside tmp_path -- never touch this machine's real T58 app
    data directory."""
    monkeypatch.setattr("app.search.graveyard.get_app_base_dir", lambda: tmp_path)
    from app.reports import run_history
    monkeypatch.setattr(run_history, "history_path", lambda: tmp_path / "run_history.json")


def test_loop_stops_with_target_reached_on_an_easy_trending_market(tmp_path):
    df = _trending_df()
    loop_cfg = SearchLoopConfig(
        target_eval_pass_pct=0.0,  # trivially easy target -- any passing candidate clears it
        max_rounds=3, stall_rounds_before_widen=1,
        starting_family="trend_breakout", starting_max_candidates=8,
    )
    result = run_search_loop(
        df, RiskConfig(), PropRules(), _fast_stage_cfg(), db_dir=tmp_path / "loop_dbs",
        loop_cfg=loop_cfg, instrument="EURUSD", timeframe="5m",
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert result.stopped_reason == "target_reached"
    assert result.winner_round is not None
    assert result.winner_candidate_id is not None
    assert len(result.rounds) == 1  # found it on round 1 -- never needed to widen


def test_loop_widens_after_stalling_then_hits_max_rounds_on_flat_data(tmp_path):
    df = _flat_df()
    loop_cfg = SearchLoopConfig(
        target_eval_pass_pct=99.9,  # effectively unreachable
        max_rounds=3, stall_rounds_before_widen=1,
        starting_family="trend_breakout", starting_max_candidates=6,
        widened_max_candidates=10,
    )
    result = run_search_loop(
        df, RiskConfig(), PropRules(), _fast_stage_cfg(), db_dir=tmp_path / "loop_dbs",
        loop_cfg=loop_cfg, instrument="XAUUSD", timeframe="5m",
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert result.stopped_reason == "max_rounds"
    assert len(result.rounds) == 3
    assert result.winner_round is None
    assert result.winner_candidate_id is None
    # Round 1 stalled (no target) -> should have widened for round 2 onward.
    assert result.rounds[0].family == "trend_breakout"
    assert result.rounds[1].family is None  # widened to "every family"
    assert result.rounds[1].max_candidates == 10


def test_loop_respects_a_tiny_time_budget(tmp_path):
    df = _flat_df()
    loop_cfg = SearchLoopConfig(
        target_eval_pass_pct=99.9, max_rounds=50, time_budget_seconds=0.0,
        starting_family="trend_breakout", starting_max_candidates=4,
    )
    result = run_search_loop(
        df, RiskConfig(), PropRules(), _fast_stage_cfg(), db_dir=tmp_path / "loop_dbs",
        loop_cfg=loop_cfg, instrument="EURUSD", timeframe="5m",
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert result.stopped_reason == "time_budget"
    assert len(result.rounds) == 0  # the budget check runs BEFORE round 1 even starts


def test_loop_stops_cleanly_when_cancelled_before_first_round(tmp_path):
    import threading
    df = _flat_df()
    cancel_event = threading.Event()
    cancel_event.set()
    loop_cfg = SearchLoopConfig(max_rounds=5, starting_family="trend_breakout", starting_max_candidates=4)
    result = run_search_loop(
        df, RiskConfig(), PropRules(), _fast_stage_cfg(), db_dir=tmp_path / "loop_dbs",
        loop_cfg=loop_cfg, instrument="EURUSD", timeframe="5m", cancel_event=cancel_event,
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert result.stopped_reason == "cancelled"
    assert len(result.rounds) == 0


def test_loop_result_carries_the_graveyard_path_search_lab_wrote_to(tmp_path):
    df = _flat_df()
    loop_cfg = SearchLoopConfig(
        target_eval_pass_pct=99.9, max_rounds=1,
        starting_family="trend_breakout", starting_max_candidates=6,
    )
    result = run_search_loop(
        df, RiskConfig(), PropRules(), _fast_stage_cfg(), db_dir=tmp_path / "loop_dbs",
        loop_cfg=loop_cfg, instrument="EURUSD", timeframe="5m",
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    # Flat/noise data at Stage 3 should produce at least one Stage-3
    # rejection, so the graveyard path should be populated.
    assert result.graveyard_path is not None
    from app.search.graveyard import load_graveyard
    rows = load_graveyard(result.graveyard_path)
    assert len(rows) > 0


def test_loop_calls_on_round_after_every_round(tmp_path):
    df = _flat_df()
    seen = []
    loop_cfg = SearchLoopConfig(
        target_eval_pass_pct=99.9, max_rounds=2,
        starting_family="trend_breakout", starting_max_candidates=4,
    )
    run_search_loop(
        df, RiskConfig(), PropRules(), _fast_stage_cfg(), db_dir=tmp_path / "loop_dbs",
        loop_cfg=loop_cfg, instrument="EURUSD", timeframe="5m", on_round=seen.append,
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert len(seen) == 2
    assert seen[0].round_index == 1 and seen[1].round_index == 2


def test_loop_reports_cancelled_not_error_when_a_round_raises_search_cancelled(tmp_path, monkeypatch):
    """Regression guard: a round can raise SearchCancelled directly (not
    just have the loop's own top-of-round cancel_event check catch it
    first) if cancellation lands mid-round -- this must still surface as
    stopped_reason='cancelled', not fall through to the generic
    stopped_reason='error' path (which is what happened before this fix,
    since SearchCancelled is an Exception subclass)."""
    from app.search.batch_runner import SearchCancelled

    def _cancelled(*a, **k):
        raise SearchCancelled("stopped mid-round")

    monkeypatch.setattr("app.orchestration.loop_runner.run_search", _cancelled)
    df = _flat_df()
    loop_cfg = SearchLoopConfig(max_rounds=3, starting_family="trend_breakout", starting_max_candidates=4)
    result = run_search_loop(
        df, RiskConfig(), PropRules(), _fast_stage_cfg(), db_dir=tmp_path / "loop_dbs",
        loop_cfg=loop_cfg, instrument="EURUSD", timeframe="5m",
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert result.stopped_reason == "cancelled"
    assert result.error is None


def test_loop_never_raises_and_reports_error_reason_when_a_round_blows_up(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("simulated crash mid-round")

    monkeypatch.setattr("app.orchestration.loop_runner.run_search", _boom)
    df = _flat_df()
    loop_cfg = SearchLoopConfig(max_rounds=3, starting_family="trend_breakout", starting_max_candidates=4)
    result = run_search_loop(
        df, RiskConfig(), PropRules(), _fast_stage_cfg(), db_dir=tmp_path / "loop_dbs",
        loop_cfg=loop_cfg, instrument="EURUSD", timeframe="5m",
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert result.stopped_reason == "error"
    assert "simulated crash" in result.error
