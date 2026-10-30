"""Tests for the auto-shrink-on-low-trades upgrade to / Tests for the GA-searches-what-it's-graded-on upgrade: compute_fitness's"""
from __future__ import annotations

import numpy as np
import pandas as pd
from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.refinement import RefinementConfig
from app.optimize.walkforward_ga import run_walkforward_aware_refinement
from app.prop.simulator import PropRules
from app.strategy.manual import ManualStrategy
from app.backtest.execution import Trade
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.refinement import apply_ga_reliability_adjustments
import math
from dataclasses import replace
from app.monte_carlo.engine import MonteCarloResult
from app.optimize.refinement import RefinementConfig, _apply_ruin_penalty, compute_fitness


# ---- from test_ga_auto_shrink.py ----

def _trending_df(n=2400, seed=3, drift=0.00015):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.00006)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00003))
        l = min(o, c) - abs(rng.normal(0, 0.00003))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config():
    return {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }


def test_auto_shrink_reduces_generations_on_thin_trade_count():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=30)
    # A deliberately absurd min_oos_trades_per_candidate -- guarantees the
    # baseline's real (but finite) chained-OOS trade count falls short of
    # it, so this test doesn't depend on hand-tuning a strategy config to
    # produce some exact, small trade count on this synthetic dataset.
    refine_cfg = RefinementConfig(
        population_size=6, generations=8, search_monte_carlo_sims=20,
        min_oos_trades_per_candidate=100_000, auto_shrink_on_low_trades=True,
    )
    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3, parallel=False,
    )
    assert len(result.generation_history) - 1 < refine_cfg.generations
    assert any("Auto-shrink" in w for w in result.warnings)


def test_auto_shrink_disabled_keeps_full_configured_width():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=30)
    refine_cfg = RefinementConfig(
        population_size=6, generations=3, search_monte_carlo_sims=20,
        min_oos_trades_per_candidate=100_000, auto_shrink_on_low_trades=False,
    )
    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3, parallel=False,
    )
    assert len(result.generation_history) - 1 == refine_cfg.generations
    assert not any("Auto-shrink" in w for w in result.warnings)


def test_normal_trade_count_does_not_shrink():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=30)
    refine_cfg = RefinementConfig(
        population_size=6, generations=3, search_monte_carlo_sims=20,
        min_oos_trades_per_candidate=5,  # low bar -- this strategy trades plenty
    )
    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3, parallel=False,
    )
    assert len(result.generation_history) - 1 == refine_cfg.generations
    assert not any("Auto-shrink" in w for w in result.warnings)


def test_risk_of_ruin_cap_runs_end_to_end_without_error():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    risk = RiskConfig()
    rules = PropRules()
    mc_cfg = MonteCarloConfig(n_simulations=30)
    refine_cfg = RefinementConfig(
        population_size=6, generations=2, search_monte_carlo_sims=20, risk_of_ruin_cap=20.0,
    )
    result = run_walkforward_aware_refinement(
        df, strategy, risk, rules, mc_cfg, refinement_config=refine_cfg, n_folds=3, parallel=False,
    )
    assert result.best is not None
    assert result.total_evaluations > 0


# ---- from test_ga_reliability.py ----

def _trades(n, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2022-01-03 15:00")
    out = []
    for i in range(n):
        d = t0 + pd.Timedelta(days=i // 2, hours=i % 2)
        p = float(rng.choice([750, -500]))
        out.append(Trade(entry_time=d, exit_time=d + pd.Timedelta(minutes=30), direction=1, entry_price=1, exit_price=1,
                         size=1, pnl=p, pnl_pct=0, exit_reason="x", commission=0, equity_after=0, initial_risk=1))
    return out


def test_few_trades_cannot_be_ranked_and_quiet_never_beats_active():
    assert apply_ga_reliability_adjustments(5.0, 14) == float("-inf")
    quiet = apply_ga_reliability_adjustments(1.0, 20)
    active = apply_ga_reliability_adjustments(0.5, 300)
    assert active > quiet


def test_widened_stop_skip_rate_is_penalised_and_negative_untouched():
    clean = apply_ga_reliability_adjustments(1.0, 200, skipped_for_sizing=0)
    widened = apply_ga_reliability_adjustments(1.0, 200, skipped_for_sizing=800)
    assert widened < 0.3 * clean
    assert apply_ga_reliability_adjustments(-1.0, 200, skipped_for_sizing=800) == -1.0


def test_fitness_noise_across_40_seeds_is_bounded_at_the_floor():
    rules = PropRules(account_size=50_000, max_drawdown_pct=4.0, drawdown_type="trailing", daily_loss_limit_pct=2.4,
                      evaluation_profit_target_pct=6.0)
    trades = _trades(120)
    vals = []
    # 2026-10-30: 1000 sims, not 300. The 2026-10-29 calibration moved the default day_block_days from 3 to 1; at 300 sims the
    # seed-to-seed std is then ~11% of the mean (binomial noise at a ~27% pass rate is already ~9.4%, so a 10% bar had no
    # headroom). At 1000 sims it is ~5.5%, so the 10% bound stays meaningful without loosening it or undoing the calibration.
    for seed in range(40):
        r = run_monte_carlo(trades, rules, MonteCarloConfig(n_simulations=1000, random_seed=seed))
        vals.append(r.per_attempt_pass_probability)
    vals = np.array(vals, dtype=float)
    assert (vals > 0).all()
    assert vals.std() < 0.10 * vals.mean()
    assert (vals <= 1.0 + 1e-9).all() or (vals <= 100.0).all()


# ---- from test_ga_ruin_penalty.py ----

def _mc(risk_of_ruin_pct: float, eval_pass=60.0, first_payout=40.0) -> MonteCarloResult:
    return MonteCarloResult(
        n_simulations=1000,
        evaluation_pass_probability=eval_pass,
        first_payout_probability=first_payout,
        failure_before_payout_probability=0.0,
        multiple_payout_probability=0.0,
        median_days_to_pass=None,
        median_days_to_first_payout=None,
        average_days_to_first_payout=None,
        median_return_pct=0.0,
        mean_return_pct=0.0,
        expected_payout=1000.0,
        median_payout=1000.0,
        total_simulated_withdrawals=0.0,
        median_drawdown_pct=5.0,
        p95_drawdown_pct=8.0,
        worst_drawdown_pct=10.0,
        risk_of_ruin_pct=risk_of_ruin_pct,
        median_max_losing_streak=3.0,
        worst_max_losing_streak=5,
        # Fitness ranks on per-attempt (single-account) odds since P0-1; the
        # defaults for these are 0.0, so the fixture must set them.
        per_attempt_pass_probability=eval_pass,
        per_attempt_payout_probability=first_payout,
        n_source_trades=100,
    )


def test_no_cap_means_no_change_to_eval_pass_probability():
    mc = _mc(risk_of_ruin_pct=40.0)
    with_no_cap = compute_fitness({}, None, mc, "eval_pass_probability")
    assert with_no_cap == mc.evaluation_pass_probability


def test_cap_erodes_fitness_when_ruin_present():
    mc = _mc(risk_of_ruin_pct=10.0)
    baseline = compute_fitness({}, None, mc, "eval_pass_probability")
    penalized = compute_fitness({}, None, mc, "eval_pass_probability", risk_of_ruin_cap=20.0)
    assert penalized < baseline
    assert penalized > 0  # gentle below the cap, not zeroed out


def test_penalty_is_much_steeper_once_ruin_exceeds_the_cap():
    cap = 20.0
    at_cap = compute_fitness({}, None, _mc(risk_of_ruin_pct=cap), "eval_pass_probability", risk_of_ruin_cap=cap)
    double_cap = compute_fitness({}, None, _mc(risk_of_ruin_pct=cap * 2), "eval_pass_probability", risk_of_ruin_cap=cap)
    baseline = 60.0
    erosion_at_cap = baseline - at_cap
    erosion_double_cap = baseline - double_cap
    assert erosion_double_cap > erosion_at_cap
    # Above the cap the search should already look near-hopeless, well
    # before the actual hard veto -- the whole point is not wasting the
    # search budget converging on a doomed candidate.
    assert double_cap < baseline * 0.3


def test_ruin_aware_metrics_are_never_double_penalized():
    mc = _mc(risk_of_ruin_pct=50.0)
    for metric in ("composite_prop_score", "prop_guide_score"):
        unpenalized = compute_fitness({"total_trades": 100, "profit_factor": 1.4, "win_rate": 55.0}, None, mc, metric)
        penalized = compute_fitness(
            {"total_trades": 100, "profit_factor": 1.4, "win_rate": 55.0}, None, mc, metric, risk_of_ruin_cap=20.0,
        )
        assert unpenalized == penalized


def test_zero_or_negative_fitness_is_left_alone():
    mc = _mc(risk_of_ruin_pct=90.0)
    assert _apply_ruin_penalty(0.0, mc, 20.0, "eval_pass_probability") == 0.0
    assert _apply_ruin_penalty(-5.0, mc, 20.0, "eval_pass_probability") == -5.0


def test_zero_ruin_is_unpenalized():
    mc = _mc(risk_of_ruin_pct=0.0)
    baseline = compute_fitness({}, None, mc, "eval_pass_probability")
    penalized = compute_fitness({}, None, mc, "eval_pass_probability", risk_of_ruin_cap=20.0)
    assert baseline == penalized


def test_refinement_config_defaults_keep_old_behavior():
    cfg = RefinementConfig()
    assert cfg.risk_of_ruin_cap is None
    assert cfg.auto_shrink_on_low_trades is True
    assert cfg.min_oos_trades_per_candidate == 30
