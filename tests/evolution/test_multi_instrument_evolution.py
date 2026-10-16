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


def test_no_total_budget_keeps_every_job_at_base_cfg_population(tmp_path, two_instrument_jobs, monkeypatch):
    """total_evaluation_budget=None (the default) must reproduce the
    exact prior behavior -- every job gets base_cfg unchanged."""
    monkeypatch.setattr("app.evolution.multi_instrument.get_app_base_dir", lambda: tmp_path)
    group = MultiInstrumentEvolutionGroup(
        "testgroup5", two_instrument_jobs, RiskConfig(), PropRules(), _fast_cfg(population_size=8, max_generations=1),
    )
    for runner in group.runners.values():
        assert runner.cfg.population_size == 8
        assert runner.cfg.max_generations == 1


def test_total_budget_splits_population_across_jobs(tmp_path, two_instrument_jobs, monkeypatch):
    """Widening the search: a fixed total_evaluation_budget should be
    spread across both jobs rather than each job getting the full,
    unmodified base_cfg population/generations."""
    monkeypatch.setattr("app.evolution.multi_instrument.get_app_base_dir", lambda: tmp_path)
    group = MultiInstrumentEvolutionGroup(
        "testgroup6", two_instrument_jobs, RiskConfig(), PropRules(),
        _fast_cfg(population_size=60, max_generations=None),
        total_evaluation_budget=200,
    )
    assert set(group.runners.keys()) == {"EURUSD/5m", "GBPUSD/5m"}
    total_spent = 0
    for runner in group.runners.values():
        # Neither job should get the full, un-split base population.
        assert runner.cfg.population_size <= 60
        assert runner.cfg.max_generations is not None
        total_spent += runner.cfg.population_size * runner.cfg.max_generations
    # Combined, the group should spend roughly the requested total budget
    # (allocate_search_budget rounds down per job, so this is <=, with a
    # small floor-driven allowance).
    assert total_spent <= 220


def test_total_budget_never_raises_an_explicit_finite_max_generations(tmp_path, two_instrument_jobs, monkeypatch):
    """An explicit, finite max_generations on base_cfg is a ceiling --
    a generous total_evaluation_budget must not raise it."""
    monkeypatch.setattr("app.evolution.multi_instrument.get_app_base_dir", lambda: tmp_path)
    group = MultiInstrumentEvolutionGroup(
        "testgroup7", two_instrument_jobs, RiskConfig(), PropRules(),
        _fast_cfg(population_size=20, max_generations=1),
        total_evaluation_budget=100_000,
    )
    for runner in group.runners.values():
        assert runner.cfg.max_generations == 1


def test_worker_pool_is_divided_across_concurrent_instruments(tmp_path, two_instrument_jobs, monkeypatch):
    """FIX (multi-instrument Evolution Lab crash/force-stop): every
    runner in this group starts on its own thread and independently
    sizes its own ProcessPoolExecutor via
    app.orchestration.resource_guard.safe_worker_count(), which has no
    idea sibling runners exist. Before this fix, base_cfg's
    parallel_workers (e.g. None -> os.cpu_count(), or an explicit
    higher count) passed through to EVERY job unchanged, so N
    instruments running together independently claimed N x the
    intended CPU/memory budget -- the exact oversubscription failure
    this fix (_resolved_workers_per_job, mirroring Search Lab's
    identical multi-instrument fix) exists to prevent. Confirms each
    job's resolved parallel_workers is base_cfg's budget divided by
    the number of concurrent jobs, never left at the un-split value."""
    monkeypatch.setattr("app.evolution.multi_instrument.get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr("app.evolution.multi_instrument.os.cpu_count", lambda: 8)
    group = MultiInstrumentEvolutionGroup(
        "testgroup8", two_instrument_jobs, RiskConfig(), PropRules(),
        _fast_cfg(parallel_workers=None),
    )
    assert set(group.runners.keys()) == {"EURUSD/5m", "GBPUSD/5m"}
    for runner in group.runners.values():
        assert runner.cfg.parallel_workers == 4  # 8 cpus // 2 concurrent instruments


def test_worker_pool_division_never_goes_below_one(tmp_path, two_instrument_jobs, monkeypatch):
    monkeypatch.setattr("app.evolution.multi_instrument.get_app_base_dir", lambda: tmp_path)
    group = MultiInstrumentEvolutionGroup(
        "testgroup9", two_instrument_jobs, RiskConfig(), PropRules(),
        _fast_cfg(parallel_workers=1),
    )
    for runner in group.runners.values():
        assert runner.cfg.parallel_workers == 1


def test_pooled_leaderboard_applies_family_cap_across_the_whole_group(tmp_path, two_instrument_jobs, monkeypatch):
    monkeypatch.setattr("app.evolution.multi_instrument.get_app_base_dir", lambda: tmp_path)
    group = MultiInstrumentEvolutionGroup(
        "testgroup8", two_instrument_jobs, RiskConfig(), PropRules(), _fast_cfg(),
    )
    from app.evolution.engine import EvolutionCandidateRecord
    from app.evolution.prop_fitness import PropFitnessBreakdown

    def _fake_record(cid, family, score):
        return EvolutionCandidateRecord(
            candidate_id=cid,
            spec={"source_type": "manual", "config": {"name": cid}},
            meta={"family": family},
            stats=None, mc_summary=None,
            fitness=PropFitnessBreakdown(
                pass_probability=0.5, payout_probability=0.5, robustness=1.0,
                oos_consistency=1.0, drawdown_pct=1.0, base_score=score, final_score=score,
            ),
        )

    eur_runner = group.runners["EURUSD/5m"]
    gbp_runner = group.runners["GBPUSD/5m"]
    eur_runner.leaderboard = [
        _fake_record("eur_1", "rsi_extreme_reversion", 0.9),
        _fake_record("eur_2", "rsi_extreme_reversion", 0.7),
    ]
    gbp_runner.leaderboard = [
        _fake_record("gbp_1", "rsi_extreme_reversion", 0.95),
        _fake_record("gbp_2", "macd_cross_trend", 0.6),
    ]

    pooled = group.pooled_leaderboard(top_n=10, max_per_family=1)
    families = [rec["family"] for rec in pooled]
    # Only the single best rsi_extreme_reversion candidate across BOTH
    # instruments should survive the max_per_family=1 cap -- not one per
    # instrument.
    assert families.count("mean_reversion") <= 1
    ids = {rec["candidate_id"] for rec in pooled}
    assert "gbp_1" in ids  # the highest-scoring mean-reversion candidate, across the whole group
    assert "eur_1" not in ids
    assert "gbp_2" in ids  # the only trend-following candidate, unaffected by the cap
