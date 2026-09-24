"""Tests for app.search.batch_runner.save_search_candidate_to_library --
the missing "Save to Strategy Library" counterpart to promote_champion.
Fixture shapes deliberately mirror tests/test_batch_runner.py's own (same
SearchStageConfig fields, same trending_df) rather than reinventing them."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.search.batch_runner import SearchStageConfig, run_search, save_search_candidate_to_library
from app.search.results_db import ResultsDB
from app.search.strategy_space import generate_search_space
from app.strategy.library import StrategyAlreadyExists
from app.strategy.python import PythonStrategy


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


def _fast_stage_cfg(**overrides) -> SearchStageConfig:
    base = dict(
        min_trades=3, min_profit_factor=0.5, max_drawdown_buffer_mult=5.0,
        stage1_top_n=6, ga_population=4, ga_generations=1, ga_search_sims=30,
        stage2_top_n=3, full_mc_sims=50, walk_forward_folds=0, robustness_neighbors=0,
        workers=1, random_seed=42,
    )
    base.update(overrides)
    return SearchStageConfig(**base)


@pytest.fixture
def small_family_space():
    return generate_search_space(mode="family", family="trend_breakout", max_candidates=8, seed=1)


def test_raises_for_unknown_candidate(tmp_path):
    with ResultsDB(tmp_path / "search.db") as db:
        db.create_run("run1", "family", "trend_breakout", "x", "y", 1, {})
    with pytest.raises(ValueError):
        save_search_candidate_to_library(str(tmp_path / "search.db"), "run1", "does-not-exist")


def test_saves_a_manual_family_candidate(tmp_path, small_family_space):
    df = _trending_df()
    db_path = tmp_path / "search.db"
    summary = run_search(
        df, RiskConfig(), PropRules(), small_family_space, _fast_stage_cfg(),
        db_path=str(db_path), instrument="TEST", timeframe="5m",
    )
    assert summary.stage3_survivors > 0, "expected at least one candidate to reach Stage 3 with this fixture"
    candidate_id = summary.leaderboard[0]["candidate_id"]

    result = save_search_candidate_to_library(str(db_path), summary.run_id, candidate_id)
    assert result["strategy_type"] == "manual"
    assert result["filename"].startswith(f"searchlab_{summary.run_id[:10]}")
    assert result["path"].exists()
    saved_config = json.loads(result["path"].read_text(encoding="utf-8"))
    assert isinstance(saved_config, dict)


def test_saves_a_python_candidate(tmp_path):
    df = _trending_df()
    path = tmp_path / "strat.py"
    path.write_text(
        'STRATEGY_NAME = "Test"\n'
        "EMA_FAST = 5\nEMA_SLOW = 15\nSTOP_LOSS_PIPS = 20\nTAKE_PROFIT_PIPS = 40\n"
        "def generate_signals(df):\n"
        '    fast = df["close"].ewm(span=EMA_FAST, adjust=False).mean()\n'
        '    slow = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()\n'
        "    return (fast > slow).astype(int) - (fast < slow).astype(int)\n",
        encoding="utf-8",
    )
    strategy = PythonStrategy(path)
    space = generate_search_space(mode="family", strategy=strategy, grid_points_per_gene=2, max_candidates=8, seed=1)
    db_path = tmp_path / "search.db"
    summary = run_search(
        df, RiskConfig(), PropRules(), space, _fast_stage_cfg(),
        db_path=str(db_path), instrument="TEST", timeframe="5m",
    )
    assert summary.stage3_survivors > 0
    candidate_id = summary.leaderboard[0]["candidate_id"]
    assert summary.leaderboard[0]["source_type"] == "python"

    result = save_search_candidate_to_library(str(db_path), summary.run_id, candidate_id)
    assert result["strategy_type"] == "python"
    assert result["filename"].endswith(".py")
    assert result["path"].read_text(encoding="utf-8") != ""


def test_saving_the_same_candidate_twice_raises_already_exists(tmp_path, small_family_space):
    df = _trending_df()
    db_path = tmp_path / "search.db"
    summary = run_search(
        df, RiskConfig(), PropRules(), small_family_space, _fast_stage_cfg(),
        db_path=str(db_path), instrument="TEST", timeframe="5m",
    )
    assert summary.stage3_survivors > 0
    candidate_id = summary.leaderboard[0]["candidate_id"]

    save_search_candidate_to_library(str(db_path), summary.run_id, candidate_id)
    with pytest.raises(StrategyAlreadyExists):
        save_search_candidate_to_library(str(db_path), summary.run_id, candidate_id)


def test_saved_strategy_is_tagged_search_lab(tmp_path, small_family_space):
    df = _trending_df()
    db_path = tmp_path / "search.db"
    summary = run_search(
        df, RiskConfig(), PropRules(), small_family_space, _fast_stage_cfg(),
        db_path=str(db_path), instrument="TEST", timeframe="5m",
    )
    assert summary.stage3_survivors > 0
    candidate_id = summary.leaderboard[0]["candidate_id"]
    result = save_search_candidate_to_library(str(db_path), summary.run_id, candidate_id)

    meta_path = Path(str(result["path"]) + ".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "search-lab" in meta.get("tags", [])
