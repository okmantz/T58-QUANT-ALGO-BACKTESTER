"""Tests for app.orchestration.forge -- kept deliberately small-scale (few
hypotheses, workers=1, low Monte Carlo sim counts) so this runs in a
reasonable time under CI while still exercising the real pipeline end to
end, same philosophy as tests/test_batch_runner.py.

Primary focus: the Strategy Graveyard feedback loop (skip hypotheses whose
parameter neighborhood a prior run already proved dead) -- this is new
wiring, not previously covered anywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration.forge import ForgeConfig, run_forge
from app.prop.simulator import PropRules
from app.search.graveyard import GraveyardEntry, param_signature, record_rejections
from app.search.strategy_space import generate_search_space


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


def _tiny_config(**overrides) -> ForgeConfig:
    base = dict(
        n_hypotheses=6, seed=1,
        min_trades=3, min_profit_factor=0.5, stage1_top_n=6,
        ga_population=4, ga_generations=1, stage2_top_n=3,
        stage3_mc_sims=30, walk_forward_folds=0, robustness_neighbors=0,
        cpcv_pool_size=2, cpcv_survivors=2, cpcv_n_groups=4, cpcv_n_test_groups=1,
        final_mc_sims=50, mc_survivors=2, eval_window_days=10, rolling_survivors=2,
        locked_holdout_frac=0.15, workers=1, random_seed=1,
        graveyard_min_attempts=3,
    )
    base.update(overrides)
    return ForgeConfig(**base)


def test_run_forge_end_to_end_no_crash(tmp_path):
    df = _trending_df()
    result = run_forge(
        df, RiskConfig(), PropRules(), _tiny_config(),
        db_path=str(tmp_path / "forge.db"), instrument="TEST", timeframe="5m",
        graveyard_path=tmp_path / "graveyard.jsonl",
    )
    assert result.run_id
    assert result.funnel[0].name == "Hypotheses generated"
    assert result.funnel[0].n_out == 6
    assert result.elapsed_seconds > 0


def test_graveyard_skip_removes_known_dead_hypothesis(tmp_path):
    """Pre-populate the graveyard with enough rejections at the exact
    param_signature of one of the actual generated hypotheses, then
    confirm run_forge's Stage 0 drops it (and only it) before Stage 1,
    logs the skip, and records a 'Graveyard pre-filter' funnel stage."""
    grave_path = tmp_path / "graveyard.jsonl"
    space = generate_search_space(mode="family", family="all", max_candidates=6, seed=1)
    target_cid, target_spec = next(iter(space.candidates.items()))
    target_family = space.meta[target_cid]["family"]
    target_sig = param_signature(target_family, target_spec.get("config"))

    record_rejections([
        GraveyardEntry(
            candidate_id=f"prior_{i}", family=target_family, generation=None,
            stage_died="cpcv", reason="always failed CPCV before",
            param_signature=target_sig,
        )
        for i in range(5)  # >= graveyard_min_attempts=3
    ], path=grave_path)

    logs: list[str] = []
    df = _trending_df()
    result = run_forge(
        df, RiskConfig(), PropRules(), _tiny_config(),
        db_path=str(tmp_path / "forge.db"), instrument="TEST", timeframe="5m",
        graveyard_path=grave_path, progress_cb=logs.append,
    )
    graveyard_stage = next((s for s in result.funnel if s.name.startswith("Graveyard pre-filter")), None)
    assert graveyard_stage is not None, "expected a Graveyard pre-filter funnel stage when a hypothesis is skipped"
    assert graveyard_stage.n_out == graveyard_stage.n_in - 1
    assert any("Strategy Graveyard: skipped" in line for line in logs)


def test_graveyard_no_match_skips_nothing(tmp_path):
    """A graveyard with unrelated history (different signatures, or below
    the min_attempts threshold) must not remove anything or claim it did."""
    grave_path = tmp_path / "graveyard.jsonl"
    record_rejections([
        GraveyardEntry(
            candidate_id="prior_1", family="totally_unrelated_family", generation=None,
            stage_died="cpcv", reason="irrelevant", param_signature="totally_unrelated_family|x=1.0",
        )
    ], path=grave_path)

    logs: list[str] = []
    df = _trending_df()
    result = run_forge(
        df, RiskConfig(), PropRules(), _tiny_config(),
        db_path=str(tmp_path / "forge.db"), instrument="TEST", timeframe="5m",
        graveyard_path=grave_path, progress_cb=logs.append,
    )
    assert not any(s.name.startswith("Graveyard pre-filter") for s in result.funnel)
    assert any("none were skipped" in line for line in logs)


def test_graveyard_skip_disabled_by_config(tmp_path):
    grave_path = tmp_path / "graveyard.jsonl"
    space = generate_search_space(mode="family", family="all", max_candidates=6, seed=1)
    target_cid, target_spec = next(iter(space.candidates.items()))
    target_family = space.meta[target_cid]["family"]
    target_sig = param_signature(target_family, target_spec.get("config"))
    record_rejections([
        GraveyardEntry(
            candidate_id=f"prior_{i}", family=target_family, generation=None,
            stage_died="cpcv", reason="always failed CPCV before", param_signature=target_sig,
        )
        for i in range(5)
    ], path=grave_path)

    df = _trending_df()
    result = run_forge(
        df, RiskConfig(), PropRules(), _tiny_config(graveyard_skip_known_dead=False),
        db_path=str(tmp_path / "forge.db"), instrument="TEST", timeframe="5m",
        graveyard_path=grave_path,
    )
    assert not any(s.name.startswith("Graveyard pre-filter") for s in result.funnel)
    assert result.funnel[0].n_out == 6
