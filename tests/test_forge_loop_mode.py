"""Tests for Forge Strategy's new Loop Mode (app.orchestration.loop_runner.run_forge_loop)
-- Forge's counterpart to Search Lab's run_search_loop. Unlike Search Lab,
Forge already searches every family at once each round, so "widening" on
a stall means searching DEEPER (more hypotheses, more survivors carried
forward) instead of broadening family scope -- see run_forge_loop's own
docstring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration.forge import ForgeConfig
from app.orchestration.loop_runner import ForgeLoopConfig, run_forge_loop
from app.prop.simulator import PropRules


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
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 1.1000 + np.cumsum(rng.normal(0, 0.00003, n))
    high = close + abs(rng.normal(0, 0.00002, n))
    low = close - abs(rng.normal(0, 0.00002, n))
    openp = close + rng.normal(0, 0.00001, n)
    return pd.DataFrame({
        "timestamp": ts, "open": openp, "high": high, "low": low, "close": close, "volume": 100.0,
    })


def _tiny_base_config(**overrides) -> ForgeConfig:
    base = dict(
        min_trades=3, min_profit_factor=0.5, stage1_top_n=6,
        ga_population=4, ga_generations=1, stage2_top_n=3,
        stage3_mc_sims=30, walk_forward_folds=0, robustness_neighbors=0,
        cpcv_pool_size=2, cpcv_survivors=2, cpcv_n_groups=4, cpcv_n_test_groups=1,
        final_mc_sims=50, mc_survivors=2, eval_window_days=10, rolling_survivors=2,
        locked_holdout_frac=0.15, workers=1,
        graveyard_min_attempts=3,
    )
    base.update(overrides)
    return ForgeConfig(**base)


@pytest.fixture(autouse=True)
def _hermetic_app_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("app.search.graveyard.get_app_base_dir", lambda: tmp_path)


def test_forge_loop_uses_the_base_config_thresholds_every_round(tmp_path):
    """Regression guard: a caller's fast/tiny ForgeConfig settings (min_trades,
    ga_population, mc sims, etc.) must survive into every round's actual
    ForgeConfig -- only n_hypotheses/stage1_top_n/stage2_top_n/exclude_families/
    seed are the loop's own to adjust. Without base_config, this would run
    against ForgeConfig()'s slow full defaults (min_trades=20, ga_population=12,
    stage3_mc_sims=2000, final_mc_sims=10000...) and take far too long for a test."""
    df = _trending_df()
    loop_cfg = ForgeLoopConfig(
        target_pass_rate_pct=0.0, require_locked_oos_passed=False,  # trivially easy
        max_rounds=1, starting_n_hypotheses=6,
        base_config=_tiny_base_config(n_hypotheses=999),  # n_hypotheses here must be overridden by starting_n_hypotheses
    )
    result = run_forge_loop(
        df, RiskConfig(), PropRules(), db_dir=tmp_path / "forge_loop", loop_cfg=loop_cfg,
        instrument="TEST", timeframe="5m",
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert len(result.rounds) == 1
    assert result.rounds[0].result.funnel[0].n_out == 6  # starting_n_hypotheses won, not base_config's 999


def test_forge_loop_stops_with_target_reached_on_easy_data(tmp_path):
    df = _trending_df()
    loop_cfg = ForgeLoopConfig(
        target_pass_rate_pct=0.0, require_locked_oos_passed=False,
        max_rounds=2, starting_n_hypotheses=6,
        base_config=_tiny_base_config(),
    )
    result = run_forge_loop(
        df, RiskConfig(), PropRules(), db_dir=tmp_path / "forge_loop", loop_cfg=loop_cfg,
        instrument="TEST", timeframe="5m",
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert result.stopped_reason in {"target_reached", "max_rounds"}


def test_forge_loop_widens_to_deeper_search_after_stalling(tmp_path):
    df = _flat_df()
    loop_cfg = ForgeLoopConfig(
        target_pass_rate_pct=99.9, require_locked_oos_passed=False,  # effectively unreachable
        max_rounds=2, stall_rounds_before_widen=1,
        starting_n_hypotheses=6, widened_n_hypotheses=12,
        starting_stage1_top_n=6, widened_stage1_top_n=6,
        starting_stage2_top_n=3, widened_stage2_top_n=3,
        base_config=_tiny_base_config(),
    )
    result = run_forge_loop(
        df, RiskConfig(), PropRules(), db_dir=tmp_path / "forge_loop", loop_cfg=loop_cfg,
        instrument="TEST", timeframe="5m",
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert result.stopped_reason == "max_rounds"
    assert len(result.rounds) == 2
    assert result.rounds[0].n_hypotheses == 6
    assert result.rounds[1].n_hypotheses == 12  # widened after round 1 stalled
    assert result.rounds[0].widened_after_this_round is True


def test_forge_loop_respects_a_tiny_time_budget(tmp_path):
    df = _flat_df()
    loop_cfg = ForgeLoopConfig(
        target_pass_rate_pct=99.9, max_rounds=50, time_budget_seconds=0.0,
        starting_n_hypotheses=6, base_config=_tiny_base_config(),
    )
    result = run_forge_loop(
        df, RiskConfig(), PropRules(), db_dir=tmp_path / "forge_loop", loop_cfg=loop_cfg,
        instrument="TEST", timeframe="5m",
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert result.stopped_reason == "time_budget"
    assert len(result.rounds) == 0


def test_forge_loop_stops_cleanly_when_cancelled_before_first_round(tmp_path):
    import threading
    df = _flat_df()
    cancel_event = threading.Event()
    cancel_event.set()
    loop_cfg = ForgeLoopConfig(max_rounds=5, starting_n_hypotheses=6, base_config=_tiny_base_config())
    result = run_forge_loop(
        df, RiskConfig(), PropRules(), db_dir=tmp_path / "forge_loop", loop_cfg=loop_cfg,
        instrument="TEST", timeframe="5m", cancel_event=cancel_event,
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert result.stopped_reason == "cancelled"
    assert len(result.rounds) == 0


def test_forge_loop_never_raises_and_reports_error_reason_when_a_round_blows_up(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("simulated crash mid-round")

    monkeypatch.setattr("app.orchestration.loop_runner.run_forge", _boom, raising=False)
    import app.orchestration.loop_runner as loop_runner_module

    # run_forge is imported LOCALLY inside run_forge_loop (from app.orchestration.forge
    # import ForgeConfig, run_forge), so patch it at the source module instead.
    monkeypatch.setattr("app.orchestration.forge.run_forge", _boom)
    df = _flat_df()
    loop_cfg = ForgeLoopConfig(max_rounds=3, starting_n_hypotheses=6, base_config=_tiny_base_config())
    result = run_forge_loop(
        df, RiskConfig(), PropRules(), db_dir=tmp_path / "forge_loop", loop_cfg=loop_cfg,
        instrument="TEST", timeframe="5m",
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert result.stopped_reason == "error"
    assert "simulated crash" in result.error


def test_forge_loop_calls_on_round_after_every_round(tmp_path):
    df = _trending_df()
    seen = []
    loop_cfg = ForgeLoopConfig(
        target_pass_rate_pct=99.9, require_locked_oos_passed=False,
        max_rounds=2, starting_n_hypotheses=6,
        base_config=_tiny_base_config(),
    )
    run_forge_loop(
        df, RiskConfig(), PropRules(), db_dir=tmp_path / "forge_loop", loop_cfg=loop_cfg,
        instrument="TEST", timeframe="5m", on_round=seen.append,
        family_health_search_dir=tmp_path / "fh_search", family_health_evolution_dir=tmp_path / "fh_evo",
    )
    assert len(seen) == 2
    assert seen[0].round_index == 1 and seen[1].round_index == 2


def test_require_locked_oos_passed_true_ignores_a_champion_that_failed_holdout(tmp_path):
    """When require_locked_oos_passed is True (the default), a champion
    whose locked_oos_status isn't PASSED must not count toward the target,
    even if it has a great pass_rate_pct -- confirms the flag actually
    changes which value _forge_champion_value returns."""
    from app.orchestration.loop_runner import _forge_champion_value

    class _Row:
        candidate_id = "c1"
        pass_rate_pct = 99.0
        locked_oos_status = "FAILED"

    class _Result:
        champion_candidate_id = "c1"
        leaderboard = [_Row()]

    value, cid = _forge_champion_value(_Result(), require_locked_oos_passed=True)
    assert value is None

    value2, cid2 = _forge_champion_value(_Result(), require_locked_oos_passed=False)
    assert value2 == 99.0
    assert cid2 == "c1"
