"""Tests for Speed Run's new Loop Mode (app.orchestration.loop_runner.run_speed_run_loop).
Speed Run's own run_speed_run already returns a definitive winner-or-not
verdict per call (Full-Pipeline-validated, not just a leaderboard), so
Loop Mode's stop condition here is simply "did this round find a winner" --
see run_speed_run_loop's own docstring for the full policy.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration.loop_runner import SpeedRunLoopConfig, run_speed_run_loop
from app.orchestration.speed_run import SpeedRunConfig
from app.prop.simulator import PropRules


def _trending_df(n=1500, seed=3, drift=0.00015):
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


def _flat_df(n=1500, seed=5):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 1.1000 + np.cumsum(rng.normal(0, 0.00003, n))
    high = close + abs(rng.normal(0, 0.00002, n))
    low = close - abs(rng.normal(0, 0.00002, n))
    openp = close + rng.normal(0, 0.00001, n)
    return pd.DataFrame({
        "timestamp": ts, "open": openp, "high": high, "low": low, "close": close, "volume": 100.0,
    })


def _fast_base_cfg(**overrides) -> SpeedRunConfig:
    base = dict(
        max_per_family_stage1=2, stage1_top_n=4,
        ga_population=4, ga_generations=1, ga_search_sims=20, stage2_top_n=2,
        full_mc_sims=40, walk_forward_folds=0, robustness_neighbors=0,
        discovery_workers=1, max_concurrent_validations=1,
        validation_ga_population=4, validation_ga_generations=1,
        validation_ga_search_mc_sims=20, validation_final_mc_sims=40,
        validation_folds=0, save_winner_to_library=False,
    )
    base.update(overrides)
    return SpeedRunConfig(**base)


@pytest.fixture
def loose_prop_rules() -> PropRules:
    return PropRules(
        account_size=50_000, evaluation_profit_target_pct=4.0, max_drawdown_pct=20.0,
        daily_loss_limit_pct=10.0,
    )


def test_speed_run_loop_uses_base_config_thresholds_every_round(tmp_path, loose_prop_rules):
    """Regression guard: a caller's fast SpeedRunConfig thresholds must
    carry into every round -- only max_candidates/top_k_to_validate/seeds
    are the loop's own to adjust."""
    df = _trending_df()
    loop_cfg = SpeedRunLoopConfig(
        max_rounds=1, starting_max_candidates=30, starting_top_k_to_validate=2,
        base_config=_fast_base_cfg(max_candidates=999, top_k_to_validate=999),
    )
    result = run_speed_run_loop(
        df, RiskConfig(), loose_prop_rules, output_dir=tmp_path / "loop_out",
        loop_cfg=loop_cfg, instrument="TEST",
    )
    assert len(result.rounds) == 1
    assert result.rounds[0].max_candidates == 30  # loop's starting value won, not base_config's 999


def test_speed_run_loop_stops_when_a_round_finds_a_winner(tmp_path, loose_prop_rules):
    df = _trending_df()
    loop_cfg = SpeedRunLoopConfig(
        max_rounds=2, starting_max_candidates=30, starting_top_k_to_validate=2,
        base_config=_fast_base_cfg(),
    )
    result = run_speed_run_loop(
        df, RiskConfig(), loose_prop_rules, output_dir=tmp_path / "loop_out",
        loop_cfg=loop_cfg, instrument="TEST",
    )
    assert result.stopped_reason in {"target_reached", "max_rounds"}
    if result.stopped_reason == "target_reached":
        assert result.winner_round is not None
        assert result.winner_round.found_winner is True


def test_speed_run_loop_widens_after_stalling_with_no_winner(tmp_path):
    """Tight prop rules on flat/noise data -- no round should find a real
    winner, so this exercises the stall -> widen path deterministically."""
    df = _flat_df()
    tight_rules = PropRules(
        account_size=50_000, evaluation_profit_target_pct=15.0, max_drawdown_pct=3.0,
        daily_loss_limit_pct=2.0,
    )
    loop_cfg = SpeedRunLoopConfig(
        max_rounds=2, stall_rounds_before_widen=1,
        starting_max_candidates=10, widened_max_candidates=20,
        starting_top_k_to_validate=1, widened_top_k_to_validate=2,
        base_config=_fast_base_cfg(),
    )
    result = run_speed_run_loop(
        df, RiskConfig(), tight_rules, output_dir=tmp_path / "loop_out",
        loop_cfg=loop_cfg, instrument="TEST",
    )
    assert result.stopped_reason == "max_rounds"
    assert len(result.rounds) == 2
    assert result.rounds[0].max_candidates == 10
    assert result.rounds[1].max_candidates == 20  # widened after round 1 found no winner
    assert result.rounds[0].widened_after_this_round is True


def test_speed_run_loop_respects_a_tiny_time_budget(tmp_path, loose_prop_rules):
    df = _flat_df()
    loop_cfg = SpeedRunLoopConfig(
        max_rounds=50, time_budget_seconds=0.0,
        starting_max_candidates=10, base_config=_fast_base_cfg(),
    )
    result = run_speed_run_loop(
        df, RiskConfig(), loose_prop_rules, output_dir=tmp_path / "loop_out",
        loop_cfg=loop_cfg, instrument="TEST",
    )
    assert result.stopped_reason == "time_budget"
    assert len(result.rounds) == 0


def test_speed_run_loop_stops_cleanly_when_cancelled_before_first_round(tmp_path, loose_prop_rules):
    df = _flat_df()
    cancel_event = threading.Event()
    cancel_event.set()
    loop_cfg = SpeedRunLoopConfig(max_rounds=5, starting_max_candidates=10, base_config=_fast_base_cfg())
    result = run_speed_run_loop(
        df, RiskConfig(), loose_prop_rules, output_dir=tmp_path / "loop_out",
        loop_cfg=loop_cfg, instrument="TEST", cancel_event=cancel_event,
    )
    assert result.stopped_reason == "cancelled"
    assert len(result.rounds) == 0


def test_speed_run_loop_never_raises_and_reports_error_reason_when_a_round_blows_up(tmp_path, loose_prop_rules, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("simulated crash mid-round")

    monkeypatch.setattr("app.orchestration.speed_run.run_speed_run", _boom)
    df = _flat_df()
    loop_cfg = SpeedRunLoopConfig(max_rounds=3, starting_max_candidates=10, base_config=_fast_base_cfg())
    result = run_speed_run_loop(
        df, RiskConfig(), loose_prop_rules, output_dir=tmp_path / "loop_out",
        loop_cfg=loop_cfg, instrument="TEST",
    )
    assert result.stopped_reason == "error"
    assert "simulated crash" in result.error
