"""Tests for app.orchestration.multi_instrument_search.run_multi_instrument_search_loop
-- Loop Mode's multi-instrument counterpart, fanning out
app.orchestration.loop_runner.run_search_loop (not a single run_search
call) the same way run_multi_instrument_search fans out run_search."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration.loop_runner import SearchLoopConfig
from app.orchestration.multi_instrument_search import InstrumentJob, run_multi_instrument_search_loop
from app.prop.simulator import PropRules
from app.search.batch_runner import SearchStageConfig


def _trending_csv(path, n=1200, seed=3, drift=0.00015):
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
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.to_csv(path, index=False)
    return path


def _fast_stage_cfg(**overrides) -> SearchStageConfig:
    base = dict(
        min_trades=3, min_profit_factor=0.5, max_drawdown_buffer_mult=5.0,
        stage1_top_n=4, ga_population=4, ga_generations=1, ga_search_sims=30,
        stage2_top_n=2, full_mc_sims=50, walk_forward_folds=0, robustness_neighbors=0,
        workers=1, random_seed=42,
    )
    base.update(overrides)
    return SearchStageConfig(**base)


@pytest.fixture
def two_instrument_jobs(tmp_path):
    csv_a = _trending_csv(tmp_path / "EURUSD5.csv", seed=1)
    csv_b = _trending_csv(tmp_path / "GBPUSD5.csv", seed=2)
    return [
        InstrumentJob(instrument="EURUSD", timeframe="5m", csv_path=csv_a),
        InstrumentJob(instrument="GBPUSD", timeframe="5m", csv_path=csv_b),
    ]


@pytest.fixture(autouse=True)
def _hermetic_app_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("app.search.graveyard.get_app_base_dir", lambda: tmp_path)
    from app.reports import run_history
    monkeypatch.setattr(run_history, "history_path", lambda: tmp_path / "run_history.json")


def test_runs_an_independent_loop_per_instrument(tmp_path, two_instrument_jobs):
    loop_cfg = SearchLoopConfig(
        target_eval_pass_pct=0.0,  # trivially easy -- any passer clears it
        max_rounds=2, stall_rounds_before_widen=1,
        starting_family="trend_breakout", starting_max_candidates=6,
    )
    results = run_multi_instrument_search_loop(
        two_instrument_jobs, RiskConfig(), PropRules(), _fast_stage_cfg(),
        loop_cfg, db_dir=tmp_path / "loop_dbs", max_concurrent_instruments=2,
    )
    assert set(results.keys()) == {"EURUSD/5m", "GBPUSD/5m"}
    for label, result in results.items():
        assert result.error is None
        assert result.loop_result is not None
        assert result.loop_result.stopped_reason in {"target_reached", "max_rounds"}
    # Each instrument's own loop got its own subdirectory of round DBs.
    eurusd_dir = tmp_path / "loop_dbs" / "EURUSD_5m"
    gbpusd_dir = tmp_path / "loop_dbs" / "GBPUSD_5m"
    assert eurusd_dir.exists() and any(eurusd_dir.glob("*.db"))
    assert gbpusd_dir.exists() and any(gbpusd_dir.glob("*.db"))


def test_one_bad_csv_path_does_not_sink_the_other_instruments_loop(tmp_path, two_instrument_jobs):
    jobs = list(two_instrument_jobs)
    jobs[0] = InstrumentJob(instrument="BROKEN", timeframe="5m", csv_path=tmp_path / "does_not_exist.csv")
    loop_cfg = SearchLoopConfig(
        target_eval_pass_pct=0.0, max_rounds=2, stall_rounds_before_widen=1,
        starting_family="trend_breakout", starting_max_candidates=6,
    )
    results = run_multi_instrument_search_loop(
        jobs, RiskConfig(), PropRules(), _fast_stage_cfg(),
        loop_cfg, db_dir=tmp_path / "loop_dbs", max_concurrent_instruments=2,
    )
    assert results["BROKEN/5m"].error is not None
    assert results["BROKEN/5m"].loop_result is None
    assert results["GBPUSD/5m"].error is None
    assert results["GBPUSD/5m"].loop_result is not None
