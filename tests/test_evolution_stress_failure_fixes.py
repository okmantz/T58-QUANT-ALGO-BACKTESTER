"""Tests for the 2026-09-11 Evolution Lab fixes:

  1. max_drawdown_buffer_mult's auto-relax is now capped (was unbounded --
     a real run's log showed it compounding to >10^13x).
  2. Once min_trades/min_profit_factor are floored, repeated empty
     pre-filter generations trigger a full random-immigrant "stagnation
     escape" instead of uselessly continuing to inflate the (capped)
     drawdown buffer forever.
  3. A generation where nothing survives the stress test no longer falls
     back to promoting a stress-test FAILURE as "WINNER" / feeding it
     into next generation's elite breeding pool -- it's routed to the
     Strategy Graveyard instead, and elite-seeding falls back to near-
     miss breeding, exactly like an empty pre-filter already did.
  4. Repeated stress-test failures also trigger the same stagnation
     escape as (2).

These use lightweight monkeypatches of the funnel stages (_prefilter,
_full_eval, _cpcv_and_pbo, _stress_test) with hand-built
EvolutionCandidateRecord fakes, rather than relying on a real stochastic
backtest to happen to reach CPCV/stress -- deterministic and fast.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.evolution.engine import EvolutionCandidateRecord, EvolutionConfig, EvolutionRunner
from app.evolution.prop_fitness import PropFitnessBreakdown
from app.prop.simulator import PropRules
from app.search.graveyard import load_graveyard


def _trending_df(n=500, seed=3):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.linspace(0, 40, n)
    noise = np.cumsum(rng.normal(0, 0.4, n))
    price = 1900 + drift + noise
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.3, "low": price - 0.3,
        "close": price, "volume": 100.0,
    })


def _fake_record(cid: str, family: str, final_score: float, robustness_ratio: float = 0.5,
                  mc_pass: float = 40.0) -> EvolutionCandidateRecord:
    fitness = PropFitnessBreakdown(
        pass_probability=mc_pass / 100.0, payout_probability=0.2, robustness=robustness_ratio,
        oos_consistency=0.5, drawdown_pct=5.0, base_score=final_score, final_score=final_score,
        notes=["High parameter sensitivity -- synthetic test fixture."],
    )
    return EvolutionCandidateRecord(
        candidate_id=cid, spec={"source_type": "manual", "config": {"lookback": 20, "atr_mult": 1.5}},
        meta={"family": family}, stats={"total_trades": 40, "max_drawdown_pct": 4.0, "net_profit": 50.0},
        mc_summary={"evaluation_pass_probability": mc_pass, "first_payout_probability": 20.0},
        robustness={"stability_ratio": robustness_ratio, "is_stable": robustness_ratio >= 0.4},
        walk_forward={"walk_forward_efficiency": -0.1},
        pbo=0.3, cpcv_degradation=1.0, cpcv_oos_eval_pass_probability=max(0.0, mc_pass - 20),
        stressed_ok=False, fitness=fitness, trade_pnls=[10.0] * 40, trades=[],
    )


def _cfg(tmp_path, **overrides) -> EvolutionConfig:
    base = dict(
        population_size=6, elite_keep=3, max_generations=1,
        save_to_library=False, adaptive_family_budget_enabled=False,
        knowledge_graph_path=str(tmp_path / "kg.jsonl"),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        tested_log_path=str(tmp_path / "tested_candidates.jsonl"),
    )
    base.update(overrides)
    return EvolutionConfig(**base)


# ---------------------------------------------------------------------------
# 1 & 2. Drawdown buffer cap + stagnation escape on empty pre-filter
# ---------------------------------------------------------------------------

def test_drawdown_buffer_relax_is_capped(tmp_path):
    cfg = _cfg(tmp_path, max_drawdown_buffer_mult=1.5, max_drawdown_buffer_mult_cap=8.0,
               auto_relax_after_empty_generations=1)
    runner = EvolutionRunner(pd.DataFrame(), RiskConfig(), PropRules(), cfg, progress_cb=None)
    for gen in range(30):
        runner._handle_empty_prefilter(gen, {"unprofitable": 10, "profit_factor": 10})
    assert runner.cfg.max_drawdown_buffer_mult <= 8.0
    # Without the cap this would be 1.5 * 1.25**30 ~= 1200+ -- confirm it
    # actually stopped growing, not just happened to land under the cap.
    assert runner.cfg.max_drawdown_buffer_mult == 8.0


def test_handle_empty_prefilter_floors_thresholds_and_triggers_stagnation_escape(tmp_path):
    cfg = _cfg(tmp_path, min_trades=20, min_profit_factor=1.05,
               max_drawdown_buffer_mult=1.5, auto_relax_after_empty_generations=1)
    runner = EvolutionRunner(pd.DataFrame(), RiskConfig(), PropRules(), cfg, progress_cb=None)
    assert runner._force_full_immigrant_next is False
    for gen in range(10):
        runner._handle_empty_prefilter(gen, {"unprofitable": 10, "profit_factor": 10})
        if runner._force_full_immigrant_next:
            break
    assert runner.cfg.min_trades == 5
    assert runner.cfg.min_profit_factor == 1.0
    assert runner._force_full_immigrant_next is True, (
        "once min_trades/min_profit_factor are floored, continued empty generations "
        "must trigger a stagnation escape instead of silently doing nothing"
    )


def test_handle_empty_prefilter_does_not_escape_before_floored(tmp_path):
    """A single empty generation (or a few) must NOT immediately force a
    full-immigrant reset -- only sustained stagnation AFTER thresholds
    are already floored should."""
    cfg = _cfg(tmp_path, min_trades=20, min_profit_factor=1.05,
               max_drawdown_buffer_mult=1.5, auto_relax_after_empty_generations=1)
    runner = EvolutionRunner(pd.DataFrame(), RiskConfig(), PropRules(), cfg, progress_cb=None)
    runner._handle_empty_prefilter(0, {"unprofitable": 10})
    assert runner._force_full_immigrant_next is False


# ---------------------------------------------------------------------------
# 3. Stress-test failures never become a disguised winner
# ---------------------------------------------------------------------------

def test_stress_failure_does_not_become_disguised_winner_or_seed_elites(tmp_path):
    cfg = _cfg(tmp_path)
    runner = EvolutionRunner(_trending_df(), RiskConfig(), PropRules(), cfg, progress_cb=None)

    fake_evaluated = [_fake_record(f"rsi_extreme_reversion-gen0-{i:04x}", "rsi_extreme_reversion", -27.0 - i)
                       for i in range(3)]
    runner._prefilter = lambda population, gen: (
        [(r.candidate_id, r.spec, r.meta, None) for r in fake_evaluated], {}, [],
    )
    runner._full_eval = lambda stage1: fake_evaluated
    runner._cpcv_and_pbo = lambda evaluated: evaluated
    runner._stress_test = lambda pool: []  # nothing survives -- exactly the uploaded log's pattern

    runner._run_one_generation(0)

    assert runner.leaderboard == [], "a candidate that failed the stress test must never reach the leaderboard"
    assert runner._elites == [], "the GA must not breed next generation's children from a stress-test failure"
    assert runner.journal, "a journal entry should still be written even with no genuine winner"
    assert "none this generation" in runner.journal[-1]
    assert "WINNER: rsi_extreme_reversion" not in runner.journal[-1]
    # Near-miss seeding should still have happened, from the cpcv pool,
    # so the NEXT generation isn't purely random either.
    assert runner._near_miss_seeds, "near-miss seeds should be populated from the failed cpcv_pool"
    assert runner._consecutive_stress_failures == 1


def test_stress_failures_are_written_to_the_graveyard(tmp_path):
    cfg = _cfg(tmp_path)
    runner = EvolutionRunner(_trending_df(), RiskConfig(), PropRules(), cfg, progress_cb=None)

    fake_evaluated = [_fake_record("orb_breakout-gen0-aaaa", "prev_day_range_breakout", -12.0, robustness_ratio=0.12)]
    runner._prefilter = lambda population, gen: (
        [(r.candidate_id, r.spec, r.meta, None) for r in fake_evaluated], {}, [],
    )
    runner._full_eval = lambda stage1: fake_evaluated
    runner._cpcv_and_pbo = lambda evaluated: evaluated
    runner._stress_test = lambda pool: []

    runner._run_one_generation(0)

    graveyard_path = Path(cfg.tested_log_path).with_name("strategy_graveyard.jsonl")
    rows = load_graveyard(graveyard_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_id"] == "orb_breakout-gen0-aaaa"
    assert row["family"] == "prev_day_range_breakout"
    assert row["stage_died"] == "stress"
    assert row["reason"]
    assert row["neighbor_robustness_pct"] == pytest.approx(12.0)

    # graveyard_summary() (the runner-level convenience method) should
    # surface the same row, clustered.
    summary = runner.graveyard_summary()
    assert len(summary) == 1
    assert summary[0]["family"] == "prev_day_range_breakout"


def test_stress_survivors_still_become_a_genuine_winner(tmp_path):
    """The fix must not break the happy path -- a REAL stress survivor
    should still reach the leaderboard and be reported as WINNER."""
    cfg = _cfg(tmp_path)
    runner = EvolutionRunner(_trending_df(), RiskConfig(), PropRules(), cfg, progress_cb=None)

    survivor = _fake_record("mean_reversion_band-gen0-bbbb", "mean_reversion_band", 15.0, robustness_ratio=0.8)
    survivor.trades = []
    fake_evaluated = [survivor]
    runner._prefilter = lambda population, gen: (
        [(r.candidate_id, r.spec, r.meta, None) for r in fake_evaluated], {}, [],
    )
    runner._full_eval = lambda stage1: fake_evaluated
    runner._cpcv_and_pbo = lambda evaluated: evaluated
    runner._stress_test = lambda pool: pool  # everything survives this time

    runner._run_one_generation(0)

    assert len(runner.leaderboard) == 1
    assert runner.leaderboard[0].candidate_id == "mean_reversion_band-gen0-bbbb"
    assert runner._elites, "a genuine survivor should seed next generation's elites"
    assert "WINNER: mean_reversion_band-gen0-bbbb" in runner.journal[-1]
    assert runner._consecutive_stress_failures == 0


def test_repeated_stress_failures_trigger_stagnation_escape(tmp_path):
    cfg = _cfg(tmp_path, stress_failure_stagnation_threshold=2)
    runner = EvolutionRunner(_trending_df(), RiskConfig(), PropRules(), cfg, progress_cb=None)

    fake_evaluated = [_fake_record("x-gen-a", "rsi_extreme_reversion", -5.0)]
    runner._prefilter = lambda population, gen: (
        [(r.candidate_id, r.spec, r.meta, None) for r in fake_evaluated], {}, [],
    )
    runner._full_eval = lambda stage1: fake_evaluated
    runner._cpcv_and_pbo = lambda evaluated: evaluated
    runner._stress_test = lambda pool: []

    runner._run_one_generation(0)
    assert runner._force_full_immigrant_next is False  # only 1 failure so far
    runner._run_one_generation(1)
    assert runner._force_full_immigrant_next is True   # 2nd in a row hits the threshold
    assert runner._consecutive_stress_failures == 0    # reset after triggering


# ---------------------------------------------------------------------------
# 4. _generate_population honors the forced full-immigrant flag
# ---------------------------------------------------------------------------

def test_generate_population_force_full_immigrant_bypasses_elite_mutation(tmp_path):
    cfg = _cfg(tmp_path, population_size=30, elite_keep=5, random_seed=1, min_immigrants_per_family=1)
    runner = EvolutionRunner(_trending_df(), RiskConfig(), PropRules(), cfg, progress_cb=None)
    fake_elites = [({"source_type": "manual", "config": {"lookback": 20}}, {"family": "mean_reversion_band"})]

    runner._force_full_immigrant_next = True
    population = runner._generate_population(1, fake_elites)

    assert runner._force_full_immigrant_next is False, "the flag must be consumed (one-shot), not sticky"
    assert all("mutated_from" not in meta for _cid, _spec, meta in population), (
        "a forced full-immigrant generation must contain zero elite-mutated children"
    )
    assert len(population) > 0


def test_generate_population_normal_path_still_mutates_elites(tmp_path):
    """Sanity check the fix didn't accidentally disable elite mutation
    on the normal (non-forced) path."""
    cfg = _cfg(tmp_path, population_size=30, elite_keep=5, random_seed=1, min_immigrants_per_family=1)
    runner = EvolutionRunner(_trending_df(), RiskConfig(), PropRules(), cfg, progress_cb=None)
    fake_elites = [({"source_type": "manual", "config": {"lookback": 20, "atr_mult": 1.2}}, {"family": "mean_reversion_band"})]

    population = runner._generate_population(1, fake_elites)

    assert any(meta.get("mutated_from") for _cid, _spec, meta in population), (
        "the normal path should still produce elite-mutated children"
    )
