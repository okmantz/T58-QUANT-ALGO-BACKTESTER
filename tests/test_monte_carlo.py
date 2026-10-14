"""Tests for the regime-aware-throttle-vs-iid-resampling fix: / v5 (2026-10-04) objective-accuracy fixes for the Monte Carlo engine."""
from __future__ import annotations

import pandas as pd
import pytest
from app.backtest.execution import Trade
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules
from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskRule
from app.monte_carlo.engine import default_method_for_adaptive_risk
import math
from types import SimpleNamespace
from app.monte_carlo.engine import (
    MIN_TRADES_FOR_VERDICT,
    MonteCarloConfig,
    _resample_indices,
    _wilson_score_interval,
    eval_pass_probability_for_trades,
    run_monte_carlo,
)
from app.prop.simulator import PropRules, simulate_account
from app.scoring.t58_scorecard import score_from_results


# ---- from test_monte_carlo.py ----

def _mock_trades(n=60, seed_pnls=None):
    trades = []
    base = pd.Timestamp("2024-01-01")
    pnls = seed_pnls or ([150, -80] * (n // 2))
    for i, pnl in enumerate(pnls[:n]):
        t = base + pd.Timedelta(days=i // 3)
        trades.append(Trade(
            entry_time=t, exit_time=t, direction=1, entry_price=1.1, exit_price=1.1,
            size=1000, pnl=pnl, pnl_pct=0.1, exit_reason="signal", commission=0, equity_after=0,
        ))
    return trades


def test_monte_carlo_runs_and_produces_probabilities():
    trades = _mock_trades(90)
    rules = PropRules(account_size=10000, evaluation_profit_target_pct=5, daily_loss_limit_pct=50,
                       max_drawdown_pct=50, min_trading_days=1, consistency_rule_pct=None)
    cfg = MonteCarloConfig(n_simulations=200, method="bootstrap", random_seed=1)
    result = run_monte_carlo(trades, rules, cfg)
    assert 0 <= result.evaluation_pass_probability <= 100
    assert 0 <= result.first_payout_probability <= 100
    assert result.n_simulations == 200


def test_monte_carlo_shuffle_method():
    trades = _mock_trades(60)
    rules = PropRules()
    cfg = MonteCarloConfig(n_simulations=100, method="shuffle", random_seed=2)
    result = run_monte_carlo(trades, rules, cfg)
    assert result.n_simulations == 100


def test_monte_carlo_empty_trades_raises():
    with pytest.raises(ValueError):
        run_monte_carlo([], PropRules(), MonteCarloConfig(n_simulations=10))


def test_reset_on_breach_off_is_byte_identical_to_before():
    """An explicit reset_on_breach=False reproduces the original single-
    attempt-per-path output exactly. (The DEFAULT is now True -- see
    test_reset_on_breach_is_on_by_default -- so this opts out explicitly.)"""
    trades = _mock_trades(90, seed_pnls=[300, -700] * 45)
    rules = PropRules(account_size=10000, evaluation_profit_target_pct=5, daily_loss_limit_pct=50,
                       max_drawdown_pct=50, min_trading_days=1, consistency_rule_pct=None)
    cfg = MonteCarloConfig(n_simulations=150, method="bootstrap", random_seed=7, reset_on_breach=False)
    result = run_monte_carlo(trades, rules, cfg)
    assert result.reset_on_breach is False
    assert result.mean_attempts_per_path == 1.0
    assert result.median_attempts_per_path == 1.0
    assert "reset_on_breach was ON" not in result.methodology_note


def test_reset_on_breach_is_off_by_default():
    """2026-10 audit: a path ends at the first breach by default. Resetting on breach and chaining attempts flatters pass
    odds (a long enough chain almost always clears the bar once); the chained mode stays available with reset_on_breach=True
    and the bankroll simulator still chains attempts."""
    assert MonteCarloConfig().reset_on_breach is False


def test_reset_on_breach_chains_within_each_resampled_path():
    """With reset_on_breach=True, each resampled path is walked through
    simulate_account's chain instead of stopping at the first bust, which
    can only raise (never lower) the any-attempt pass probability versus
    the single-attempt run on the identical resampling."""
    trades = _mock_trades(150, seed_pnls=[400, -900] * 75)
    rules = PropRules(account_size=10000, evaluation_profit_target_pct=5, daily_loss_limit_pct=20,
                       max_drawdown_pct=20, min_trading_days=1, consistency_rule_pct=None)

    single_cfg = MonteCarloConfig(n_simulations=300, method="block_bootstrap", block_size=8,
                                   reset_on_breach=False, random_seed=3)
    single = run_monte_carlo(trades, rules, single_cfg)

    chained_cfg = MonteCarloConfig(n_simulations=300, method="block_bootstrap", block_size=8,
                                    reset_on_breach=True, random_seed=3)
    chained = run_monte_carlo(trades, rules, chained_cfg)

    assert chained.reset_on_breach is True
    assert chained.mean_attempts_per_path >= 1.0
    assert chained.evaluation_pass_probability >= single.evaluation_pass_probability
    assert chained.any_attempt_pass_probability == chained.evaluation_pass_probability
    assert "reset_on_breach was ON" in chained.methodology_note
    assert len(chained.attempts_distribution) == 300


# ---- from test_mc_method_adaptive_risk_default.py ----

def _enabled_adaptive_risk() -> AdaptiveRiskConfig:
    return AdaptiveRiskConfig(
        enabled=True,
        rules=[AdaptiveRiskRule(trigger="consecutive_losses", threshold=2, risk_multiplier=0.5)],
    )


def test_none_adaptive_risk_uses_block_bootstrap_default():
    assert default_method_for_adaptive_risk(None) == "block_bootstrap"


def test_disabled_adaptive_risk_uses_block_bootstrap_default():
    cfg = AdaptiveRiskConfig(enabled=False, rules=[])
    assert default_method_for_adaptive_risk(cfg) == "block_bootstrap"


def test_enabled_adaptive_risk_switches_to_block_bootstrap():
    assert default_method_for_adaptive_risk(_enabled_adaptive_risk()) == "block_bootstrap"


def test_object_without_enabled_attribute_is_safe():
    class NotAdaptiveRisk:
        pass

    assert default_method_for_adaptive_risk(NotAdaptiveRisk()) == "block_bootstrap"


# ---- from test_mc_objective_fixes.py ----

def _trades(pnls, trades_per_day=2, start="2024-01-01"):
    base = pd.Timestamp(start)
    return [
        Trade(
            entry_time=base + pd.Timedelta(days=i // trades_per_day),
            exit_time=base + pd.Timedelta(days=i // trades_per_day),
            direction=1, entry_price=1.1, exit_price=1.1, size=1000, pnl=p,
            pnl_pct=0.1, exit_reason="signal", commission=0, equity_after=0,
        )
        for i, p in enumerate(pnls)
    ]


def _divergence_rules():
    # Single-trade attempts: +1600 passes a 15% ($1500) target, -2100 busts
    # a 20% trailing DD floor. ~50/50 per attempt, many attempts per path.
    return PropRules(account_size=10000, evaluation_profit_target_pct=15,
                     daily_loss_limit_pct=50, max_drawdown_pct=20,
                     min_trading_days=1, consistency_rule_pct=None)


def test_per_attempt_diverges_from_chain_when_reset_on_breach():
    """Reproduction: with reset_on_breach on, the chain-level number reads
    ~100% (a long enough chain almost always clears the bar once) while
    the per-attempt number is ~50% -- the honest 'will ONE account
    succeed' answer."""
    trades = _trades([1600, -2100] * 30)
    cfg = MonteCarloConfig(n_simulations=400, method="bootstrap", random_seed=11, reset_on_breach=True)  # 2026-10: default is now False
    result = run_monte_carlo(trades, _divergence_rules(), cfg)
    assert result.mean_attempts_per_path > 5  # genuinely a multi-attempt regime
    assert result.evaluation_pass_probability > 95.0  # chain-level: inflated
    assert 30.0 < result.per_attempt_pass_probability < 70.0  # per-attempt: honest
    assert result.evaluation_pass_probability != pytest.approx(result.per_attempt_pass_probability)


def test_eval_pass_probability_returns_per_attempt_not_chain():
    """The fold-level scoring primitive must return per_attempt_pass_-
    probability -- the value WF/CPCV/regime/forge callers optimize."""
    trades = _trades([1600, -2100] * 30)
    cfg = MonteCarloConfig(n_simulations=400, method="bootstrap", random_seed=11, reset_on_breach=True)
    expected = run_monte_carlo(trades, _divergence_rules(), cfg).per_attempt_pass_probability
    got = eval_pass_probability_for_trades(trades, _divergence_rules(), cfg)
    assert got == pytest.approx(expected)
    # ...and NOT the chain-level number from the same run
    chain = run_monte_carlo(trades, _divergence_rules(), cfg).evaluation_pass_probability
    assert got != pytest.approx(chain)


def test_eval_per_attempt_matches_chain_when_single_attempt():
    """reset_on_breach=False: one attempt per path, per-attempt and
    chain-level are identical -- no behavior change for that case."""
    trades = _trades([1600, -2100] * 30)
    cfg = MonteCarloConfig(n_simulations=200, method="bootstrap", random_seed=11,
                           reset_on_breach=False)
    result = run_monte_carlo(trades, _divergence_rules(), cfg)
    assert result.per_attempt_pass_probability == pytest.approx(result.evaluation_pass_probability)
    assert eval_pass_probability_for_trades(trades, _divergence_rules(), cfg) == pytest.approx(
        result.per_attempt_pass_probability)


def test_wilson_ci_known_proportion():
    # 700/1000 at 95%: Wilson ~= (67.09, 72.76)
    lo, hi = _wilson_score_interval(700, 1000)
    assert 67.0 < lo < 68.0
    assert 72.0 < hi < 73.0
    assert lo < 70.0 < hi


def test_wilson_ci_edges():
    lo, hi = _wilson_score_interval(0, 100)
    assert lo == 0.0 and hi > 0.0  # stays honest at the boundary, no negative
    lo, hi = _wilson_score_interval(100, 100)
    assert hi == pytest.approx(100.0) and lo < 100.0
    assert _wilson_score_interval(0, 0) == (0.0, 0.0)


def test_wilson_ci_narrower_with_more_sims():
    narrow = _wilson_score_interval(500, 1000)
    wide = _wilson_score_interval(50, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_mc_result_carries_cis():
    trades = _trades([150, -80] * 45)
    rules = PropRules(account_size=10000, evaluation_profit_target_pct=5,
                      daily_loss_limit_pct=50, max_drawdown_pct=50,
                      min_trading_days=1, consistency_rule_pct=None)
    cfg = MonteCarloConfig(n_simulations=200, method="bootstrap", random_seed=1)
    result = run_monte_carlo(trades, rules, cfg)
    lo, hi = result.pass_probability_ci95
    assert lo <= result.evaluation_pass_probability <= hi
    lo, hi = result.payout_probability_ci95
    assert lo <= result.first_payout_probability <= hi


def _fake_mc_result(pass_pt, payout_pt, pass_ci, payout_ci):
    return SimpleNamespace(
        evaluation_pass_probability=pass_pt,
        first_payout_probability=payout_pt,
        pass_probability_ci95=pass_ci,
        payout_probability_ci95=payout_ci,
        risk_of_ruin_pct=5.0,
    )


def test_verdict_gates_on_wilson_lower_bound():
    """Two results with identical point estimates but different CIs must
    score differently -- the acceptance path uses the lower bound."""
    optimistic = _fake_mc_result(80.0, 60.0, (75.0, 85.0), (55.0, 65.0))
    noisy = _fake_mc_result(80.0, 60.0, (40.0, 85.0), (20.0, 65.0))
    s_opt = score_from_results(mc_result=optimistic)
    s_noisy = score_from_results(mc_result=noisy)
    assert s_opt.components["pass_probability"]["value"] == pytest.approx(75.0)
    assert s_noisy.components["pass_probability"]["value"] == pytest.approx(40.0)
    assert s_opt.components["first_payout_probability"]["value"] == pytest.approx(55.0)
    assert s_noisy.components["first_payout_probability"]["value"] == pytest.approx(20.0)
    assert s_opt.score > s_noisy.score


def test_verdict_falls_back_to_point_estimate_without_ci():
    """Results built before the CI fields existed (no ci95 attrs at all)
    keep the old point-estimate behavior."""
    legacy = SimpleNamespace(
        evaluation_pass_probability=80.0,
        first_payout_probability=60.0,
        risk_of_ruin_pct=5.0,
    )
    s = score_from_results(mc_result=legacy)
    assert s.components["pass_probability"]["value"] == pytest.approx(80.0)
    assert s.components["first_payout_probability"]["value"] == pytest.approx(60.0)


def test_ruin_counts_daily_loss_limit_deaths():
    """Every simulated day loses $300 > the $200 daily-loss limit, but the
    3% drawdown never touches the 50% max-DD floor. Old definition: 0%
    ruin. Correct: 100% -- daily-loss death is the dominant death mode."""
    trades = _trades([-300] * 40, trades_per_day=1)
    rules = PropRules(account_size=10000, evaluation_profit_target_pct=5,
                      daily_loss_limit_pct=2, max_drawdown_pct=50,
                      min_trading_days=1, consistency_rule_pct=None)
    # sanity: the simulator really does kill these attempts via daily loss
    single = simulate_account(
        [-300.0] * 40,
        [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(40)],
        rules, reset_on_breach=True,
    )
    assert single.attempts[0].failed
    assert single.attempts[0].failure_reason == "daily_loss_limit"

    cfg = MonteCarloConfig(n_simulations=100, method="bootstrap", random_seed=5)
    result = run_monte_carlo(trades, rules, cfg)
    assert result.worst_drawdown_pct < rules.max_drawdown_pct  # old metric saw nothing
    assert result.risk_of_ruin_pct == pytest.approx(100.0)


def test_ruin_still_counts_max_dd_deaths():
    """Max-drawdown deaths keep counting under the new definition."""
    trades = _trades([-1500] * 40, trades_per_day=1)
    rules = PropRules(account_size=10000, evaluation_profit_target_pct=5,
                      daily_loss_limit_pct=50, max_drawdown_pct=20,
                      min_trading_days=1, consistency_rule_pct=None)
    cfg = MonteCarloConfig(n_simulations=100, method="bootstrap", random_seed=5)
    result = run_monte_carlo(trades, rules, cfg)
    assert result.risk_of_ruin_pct == pytest.approx(100.0)


def test_min_trades_floor_blocks_thin_folds():
    """A fold with fewer than MIN_TRADES_FOR_VERDICT trades contributes no
    verdict (0.0) instead of a noisy MC number."""
    assert MIN_TRADES_FOR_VERDICT == 15
    rules = PropRules(account_size=10000, evaluation_profit_target_pct=5,
                      daily_loss_limit_pct=50, max_drawdown_pct=50,
                      min_trading_days=1, consistency_rule_pct=None)
    thin = _trades([150, -80] * 5)  # 10 trades
    assert len(thin) < MIN_TRADES_FOR_VERDICT
    assert eval_pass_probability_for_trades(thin, rules) == 0.0
    assert eval_pass_probability_for_trades([], rules) == 0.0


def test_min_trades_floor_passes_real_folds():
    """At/above the floor the primitive actually runs the MC (returns the
    per-attempt number, not the floor's 0.0)."""
    rules = PropRules(account_size=10000, evaluation_profit_target_pct=5,
                      daily_loss_limit_pct=50, max_drawdown_pct=50,
                      min_trading_days=1, consistency_rule_pct=None)
    trades = _trades([150, -80] * 10)  # 20 trades
    cfg = MonteCarloConfig(n_simulations=200, method="bootstrap", random_seed=1)
    got = eval_pass_probability_for_trades(trades, rules, cfg)
    expected = run_monte_carlo(trades, rules, cfg).per_attempt_pass_probability
    assert got == pytest.approx(expected)


def test_block_bootstrap_never_wraps():
    """Blocks must be real contiguous slices -- no wrapping the series'
    head onto its tail (the old (start + j) % n behavior)."""
    rng = __import__("numpy").random.default_rng(0)
    for _ in range(300):
        idx = _resample_indices(rng, 23, "block_bootstrap", 5)
        assert len(idx) == 23
        assert idx.min() >= 0 and idx.max() < 23
        for s in range(0, len(idx), 5):
            chunk = idx[s:s + 5]
            for j in range(len(chunk) - 1):
                assert chunk[j + 1] - chunk[j] == 1, f"wrapped block: {chunk}"


def test_block_bootstrap_degenerate_block_size():
    """block_size >= n: the whole series is one block, emitted in order."""
    rng = __import__("numpy").random.default_rng(3)
    idx = _resample_indices(rng, 4, "block_bootstrap", 8)
    assert list(idx) == [0, 1, 2, 3]
