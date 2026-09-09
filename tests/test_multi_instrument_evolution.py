"""Tests for app.evolution.multi_instrument.MultiInstrumentEvolutionGroup --
manages several independent EvolutionRunner instances (one per instrument)
concurrently, each with fully isolated on-disk state, started/monitored/
stopped together as one unit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.evolution.engine import EvolutionConfig
from app.evolution.multi_instrument import EvolutionInstrumentJob, MultiInstrumentEvolutionGroup
from app.prop.simulator import PropRules


def _trending_df_csv(path, n=500, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.linspace(0, 40, n)
    noise = np.cumsum(rng.normal(0, 0.4, n))
    price = 1900 + drift + noise
    df = pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.3, "low": price - 0.3,
        "close": price, "volume": 100.0,
    })
    df.to_csv(path, index=False)
    return path


def _fast_cfg(**overrides) -> EvolutionConfig:
    base = dict(
        population_size=8, elite_keep=2, max_generations=1,
        min_trades=3, min_profit_factor=0.0, max_drawdown_buffer_mult=20.0,
        mc_sims=20, robustness_neighbors=1, walk_forward_folds=1,
        cpcv_top_n=2, cpcv_max_paths=3, cpcv_n_groups=3,
        save_to_library=False, parallel_workers=1,
    )
    base.update(overrides)
    return EvolutionConfig(**base)


@pytest.fixture
def two_instrument_jobs(tmp_path):
    csv_a = _trending_df_csv(tmp_path / "EURUSD5.csv", seed=1)
    csv_b = _trending_df_csv(tmp_path / "GBPUSD5.csv", seed=2)
    return [
        EvolutionInstrumentJob(instrument="EURUSD", timeframe="5m", csv_path=str(csv_a)),
        EvolutionInstrumentJob(instrument="GBPUSD", timeframe="5m", csv_path=str(csv_b)),
    ]


@pytest.fixture(autouse=True)
def _isolated_family_health_dir(tmp_path, monkeypatch):
    """EvolutionRunner now consults app.search.family_health (default
    dead-end-family exclusion) on construction -- isolate its own
    independent get_app_base_dir binding too, not just
    app.evolution.multi_instrument's, so these tests never depend on
    (or pollute) this machine's real reports/search or data/evolution
    directories."""
    monkeypatch.setattr("app.search.family_health.get_app_base_dir", lambda: tmp_path / "family_health_base")


def test_group_creates_one_isolated_runner_per_job(tmp_path, two_instrument_jobs, monkeypatch):
    monkeypatch.setattr("app.evolution.multi_instrument.get_app_base_dir", lambda: tmp_path)
    group = MultiInstrumentEvolutionGroup(
        "testgroup1", two_instrument_jobs, RiskConfig(), PropRules(), _fast_cfg(),
    )
    assert set(group.runners.keys()) == {"EURUSD/5m", "GBPUSD/5m"}
    assert not group.errors
    # Each runner's checkpoint path must be distinct and namespaced under the group.
    paths = {label: r.cfg.checkpoint_path for label, r in group.runners.items()}
    assert len(set(paths.values())) == 2
    for p in paths.values():
        assert "testgroup1" in p


def test_group_start_and_stop_all(tmp_path, two_instrument_jobs, monkeypatch):
    monkeypatch.setattr("app.evolution.multi_instrument.get_app_base_dir", lambda: tmp_path)
    group = MultiInstrumentEvolutionGroup(
        "testgroup2", two_instrument_jobs, RiskConfig(), PropRules(), _fast_cfg(),
    )
    group.start_all()
    assert group.is_running
    stopped = group.stop_all(timeout=15.0)
    assert stopped
    assert not group.is_running


def test_group_status_reports_per_instrument_state(tmp_path, two_instrument_jobs, monkeypatch):
    monkeypatch.setattr("app.evolution.multi_instrument.get_app_base_dir", lambda: tmp_path)
    group = MultiInstrumentEvolutionGroup(
        "testgroup3", two_instrument_jobs, RiskConfig(), PropRules(), _fast_cfg(),
    )
    group.start_all()
    group.stop_all(timeout=15.0)
    status = group.status()
    assert set(status["labels"]) == {"EURUSD/5m", "GBPUSD/5m"}
    assert status["running"] is False
    for label in status["labels"]:
        inst = status["instruments"][label]
        assert "generation" in inst
        assert "leaderboard" in inst


def test_a_bad_csv_is_recorded_as_an_error_not_a_crash(tmp_path, two_instrument_jobs, monkeypatch):
    monkeypatch.setattr("app.evolution.multi_instrument.get_app_base_dir", lambda: tmp_path)
    jobs = two_instrument_jobs + [
        EvolutionInstrumentJob(instrument="BAD", timeframe="1m", csv_path="/no/such/file.csv")
    ]
    group = MultiInstrumentEvolutionGroup("testgroup4", jobs, RiskConfig(), PropRules(), _fast_cfg())
    assert "BAD/1m" in group.errors
    assert "BAD/1m" not in group.runners
    assert set(group.runners.keys()) == {"EURUSD/5m", "GBPUSD/5m"}
