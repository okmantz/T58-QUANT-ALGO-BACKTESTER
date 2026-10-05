"""v6 W2 bundle tests: prop objective & scoring honesty fixes.

Covers: B1 (full preset-field thread-through via the server helper),
B2+D2 (Wilson CIs on the per-attempt number; scorecard prefers the
per-attempt CI lower bound), B3 (loop-stop default + fallback), B4
(_prop_guide_score per-attempt reads + prop_rules-relative DD band),
B11a (multi-preset champion board helper), B12-fitness (headroom bonus),
D4(a) (plateau probe counting -- see tests/test_plateau_robust_selection.py).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.evolution.prop_fitness import compute_prop_fitness
from app.monte_carlo.engine import MonteCarloResult, run_monte_carlo, MonteCarloConfig
from app.backtest.execution import Trade
from app.orchestration.loop_runner import SearchLoopConfig, _best_value_in_leaderboard
from app.optimize.refinement import (
    _headroom_bonus,
    _prop_guide_score,
    compute_fitness,
)
from app.prop.presets import get_preset
from app.prop.simulator import PropRules
from app.scoring.t58_scorecard import score_from_results
from app.web.server import _prop_rules_from_search_evo_form


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_trades(n=60):
    import pandas as pd
    trades = []
    base = pd.Timestamp("2024-01-01")
    pnls = [150, -80] * (n // 2)
    for i, pnl in enumerate(pnls[:n]):
        t = base + pd.Timedelta(days=i // 3)
        trades.append(Trade(
            entry_time=t, exit_time=t, direction=1, entry_price=1.1, exit_price=1.1,
            size=1000, pnl=pnl, pnl_pct=0.1, exit_reason="signal", commission=0, equity_after=0,
        ))
    return trades


def _mc_result(**overrides):
    base = dict(
        n_simulations=1000,
        evaluation_pass_probability=97.0,
        first_payout_probability=90.0,
        failure_before_payout_probability=10.0,
        multiple_payout_probability=5.0,
        median_days_to_pass=10.0,
        median_days_to_first_payout=20.0,
        average_days_to_first_payout=22.0,
        median_return_pct=5.0,
        mean_return_pct=5.5,
        expected_payout=100.0,
        median_payout=90.0,
        total_simulated_withdrawals=1000.0,
        median_drawdown_pct=3.0,
        p95_drawdown_pct=7.0,
        worst_drawdown_pct=9.0,
        risk_of_ruin_pct=5.0,
        median_max_losing_streak=3.0,
        worst_max_losing_streak=8,
    )
    base.update(overrides)
    return MonteCarloResult(**base)


def _form_from_preset(preset):
    """What the updated search/evolution forms submit after the preset-apply
    JS runs: preset values mapped onto the form field names (None
    consistency -> the JS clears the field to an empty string)."""
    return {
        "account_size": str(preset.account_size),
        "profit_target": str(preset.evaluation_profit_target_pct),
        "daily_loss": str(preset.daily_loss_limit_pct),
        "max_dd": str(preset.max_drawdown_pct),
        "dd_type": preset.drawdown_type,
        "dd_check_mode": preset.drawdown_check_mode,
        "consistency": "" if preset.consistency_rule_pct is None else str(preset.consistency_rule_pct),
        "min_days": str(preset.min_trading_days),
    }


# ---------------------------------------------------------------------------
# B2(a): per-attempt Wilson CI fields are populated by run_monte_carlo
# ---------------------------------------------------------------------------

def test_run_monte_carlo_populates_per_attempt_cis():
    rules = PropRules(account_size=10000, evaluation_profit_target_pct=5, daily_loss_limit_pct=50,
                      max_drawdown_pct=50, min_trading_days=1, consistency_rule_pct=None)
    cfg = MonteCarloConfig(n_simulations=100, method="bootstrap", random_seed=1)
    result = run_monte_carlo(_mock_trades(90), rules, cfg)
    lo, hi = result.per_attempt_pass_ci95
    assert 0.0 <= lo <= hi <= 100.0
    lo2, hi2 = result.per_attempt_payout_ci95
    assert 0.0 <= lo2 <= hi2 <= 100.0
    # The CI lower bound sits at/below the point estimate and at/above 0.
    assert lo <= result.per_attempt_pass_probability
    assert lo2 <= result.per_attempt_payout_probability


# ---------------------------------------------------------------------------
# B2(b): scorecard prefers the per-attempt CI lower bound
# ---------------------------------------------------------------------------

def test_scorecard_prefers_per_attempt_ci_lower_bound():
    mc = _mc_result(
        evaluation_pass_probability=97.0,
        first_payout_probability=90.0,
        pass_probability_ci95=(96.0, 98.0),          # inflated chain-level
        payout_probability_ci95=(88.0, 92.0),
        per_attempt_pass_ci95=(33.0, 45.0),          # honest single-account
        per_attempt_payout_ci95=(40.0, 55.0),
        per_attempt_pass_probability=39.0,
        per_attempt_payout_probability=47.0,
    )
    result = score_from_results(mc_result=mc)
    assert result.components["pass_probability"]["value"] == pytest.approx(33.0)
    assert result.components["first_payout_probability"]["value"] == pytest.approx(40.0)


def test_scorecard_falls_back_to_chain_ci_for_legacy_results():
    # A result built before the per-attempt fields existed has no
    # per_attempt_*_ci95 attributes at all -- the chain-level CI still gates.
    mc = SimpleNamespace(
        evaluation_pass_probability=97.0,
        first_payout_probability=90.0,
        risk_of_ruin_pct=5.0,
        pass_probability_ci95=(96.0, 98.0),
        payout_probability_ci95=(88.0, 92.0),
    )
    result = score_from_results(mc_result=mc)
    assert result.components["pass_probability"]["value"] == pytest.approx(96.0)
    assert result.components["first_payout_probability"]["value"] == pytest.approx(88.0)


def test_scorecard_falls_back_to_point_estimate_when_no_ci():
    mc = SimpleNamespace(
        evaluation_pass_probability=97.0,
        first_payout_probability=90.0,
        risk_of_ruin_pct=5.0,
    )
    result = score_from_results(mc_result=mc)
    assert result.components["pass_probability"]["value"] == pytest.approx(97.0)


# ---------------------------------------------------------------------------
# B1: preset-field thread-through via the server helper
# ---------------------------------------------------------------------------

def test_preset_fields_thread_through_server_helper():
    preset = get_preset("apex_50k_eod")
    rules = _prop_rules_from_search_evo_form(_form_from_preset(preset))
    expected = preset.to_prop_rules()
    # All eight fields thread through exactly as the preset defines them.
    assert rules.account_size == expected.account_size
    assert rules.evaluation_profit_target_pct == expected.evaluation_profit_target_pct
    assert rules.daily_loss_limit_pct == expected.daily_loss_limit_pct
    assert rules.max_drawdown_pct == expected.max_drawdown_pct
    assert rules.drawdown_type == expected.drawdown_type
    assert rules.drawdown_check_mode == expected.drawdown_check_mode
    assert rules.consistency_rule_pct == expected.consistency_rule_pct
    assert rules.min_trading_days == expected.min_trading_days
    # And the fields the OLD 4-field construction silently dropped really
    # do differ from the dataclass defaults on this preset.
    defaults = PropRules()
    assert rules.account_size != defaults.account_size
    assert rules.evaluation_profit_target_pct != defaults.evaluation_profit_target_pct
    assert rules.daily_loss_limit_pct != defaults.daily_loss_limit_pct
    assert rules.max_drawdown_pct != defaults.max_drawdown_pct
    assert rules.consistency_rule_pct is None and defaults.consistency_rule_pct == 30.0
    assert rules.min_trading_days == 0 and defaults.min_trading_days == 5


def test_helper_empty_consistency_means_none():
    form = _form_from_preset(get_preset("ftmo_100k"))
    form["consistency"] = ""  # user explicitly cleared the field
    rules = _prop_rules_from_search_evo_form(form)
    assert rules.consistency_rule_pct is None


def test_helper_absent_consistency_keeps_old_default():
    # A template not yet updated with the new fields doesn't send
    # `consistency` at all -- pre-B1 behavior (PropRules default 30.0) is
    # preserved rather than silently turning the gate off.
    form = {k: v for k, v in _form_from_preset(get_preset("ftmo_100k")).items() if k != "consistency"}
    rules = _prop_rules_from_search_evo_form(form)
    assert rules.consistency_rule_pct == PropRules.consistency_rule_pct == 30.0


def test_helper_defaults_for_untouched_form():
    rules = _prop_rules_from_search_evo_form({})
    defaults = PropRules()
    assert rules.drawdown_type == defaults.drawdown_type == "trailing"
    assert rules.drawdown_check_mode == defaults.drawdown_check_mode == "intrabar"
    assert rules.min_trading_days == defaults.min_trading_days == 5


# ---------------------------------------------------------------------------
# B3: loop-stop conditions read per-attempt
# ---------------------------------------------------------------------------

def test_loop_default_metric_is_per_attempt():
    assert SearchLoopConfig().target_metric == "per_attempt_pass_probability"


def test_best_value_falls_back_for_old_leaderboard_dicts():
    old_row = {
        "candidate_id": "a", "passed_stage3_gate": True,
        "mc_summary": {"evaluation_pass_probability": 60.0},
    }
    value, cid = _best_value_in_leaderboard([old_row], "per_attempt_pass_probability", True)
    assert value == pytest.approx(60.0)
    assert cid == "a"


def test_best_value_reads_per_attempt_when_present():
    row = {
        "candidate_id": "a", "passed_stage3_gate": True,
        "mc_summary": {"per_attempt_pass_probability": 33.0, "evaluation_pass_probability": 97.0},
    }
    value, _ = _best_value_in_leaderboard([row], "per_attempt_pass_probability", True)
    assert value == pytest.approx(33.0)


def test_best_value_does_not_replace_zero_per_attempt_with_chain():
    # A genuine 0.0 per-attempt value must not be silently swapped for the
    # inflated chain-level number (an `or`-style fallback would do that).
    row = {
        "candidate_id": "a", "passed_stage3_gate": True,
        "mc_summary": {"per_attempt_pass_probability": 0.0, "evaluation_pass_probability": 97.0},
    }
    value, _ = _best_value_in_leaderboard([row], "per_attempt_pass_probability", True)
    assert value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# B4: _prop_guide_score reads per-attempt + prop_rules-relative DD band
# ---------------------------------------------------------------------------

def test_prop_guide_score_reads_per_attempt():
    stats = {"profit_factor": 1.4, "total_trades": 120, "win_rate": 55.0}
    # Same chain-level numbers, different per-attempt numbers: the score
    # must follow the per-attempt legs.
    mc_chain_inflated = _mc_result(
        evaluation_pass_probability=97.0, first_payout_probability=90.0,
        per_attempt_pass_probability=33.0, per_attempt_payout_probability=40.0,
    )
    mc_honest_chain = _mc_result(
        evaluation_pass_probability=33.0, first_payout_probability=40.0,
        per_attempt_pass_probability=33.0, per_attempt_payout_probability=40.0,
    )
    assert _prop_guide_score(stats, mc_chain_inflated) == pytest.approx(
        _prop_guide_score(stats, mc_honest_chain)
    )


def test_prop_guide_score_dd_band_relative_to_prop_rules():
    stats = {"profit_factor": 1.4, "total_trades": 120, "win_rate": 55.0}
    mc = _mc_result(p95_drawdown_pct=14.0)  # above the old 6-10% band -> dd_score 0
    baseline = _prop_guide_score(stats, mc)
    wide = _prop_guide_score(stats, mc, prop_rules=PropRules(max_drawdown_pct=20.0))
    # 14% P95 is inside (0.6*20, 20) = (12, 20), so the relative band
    # scores it better than the hardcoded 6-10% band did.
    assert wide > baseline


def test_prop_guide_score_none_prop_rules_matches_old_band():
    stats = {"profit_factor": 1.4, "total_trades": 120, "win_rate": 55.0}
    mc = _mc_result()
    assert _prop_guide_score(stats, mc, prop_rules=None) == pytest.approx(_prop_guide_score(stats, mc))


# ---------------------------------------------------------------------------
# B12-fitness: margin-of-safety headroom gradient
# ---------------------------------------------------------------------------

def test_headroom_bonus_zero_when_absent():
    assert _headroom_bonus(None) == 0.0
    assert _headroom_bonus({}) == 0.0
    assert _headroom_bonus({"best_day_profit_pct_of_limit": None}) == 0.0


def test_headroom_bonus_rewards_low_utilization():
    low = {
        "best_day_profit_pct_of_limit": 0.2,
        "worst_daily_loss_pct_of_limit": 0.3,
        "max_dd_pct_of_limit": 0.4,
    }
    high = {
        "best_day_profit_pct_of_limit": 1.0,
        "worst_daily_loss_pct_of_limit": 1.0,
        "max_dd_pct_of_limit": 1.0,
    }
    assert _headroom_bonus(low) == pytest.approx(0.1 * ((0.8 + 0.7 + 0.6) / 3))
    assert _headroom_bonus(high) == pytest.approx(0.0)


def test_compute_prop_fitness_headroom_term():
    headroom = {
        "best_day_profit_pct_of_limit": 0.2,
        "worst_daily_loss_pct_of_limit": 0.2,
        "max_dd_pct_of_limit": 0.2,
    }
    mc_summary = {
        "per_attempt_pass_probability": 70.0,
        "per_attempt_payout_probability": 50.0,
        **headroom,
    }
    stats = {"max_drawdown_pct": 5.0, "total_trades": 100, "net_profit": 1000.0}
    with_headroom = compute_prop_fitness(stats, mc_summary, None, None, [10.0] * 100)
    without = compute_prop_fitness(
        stats,
        {k: v for k, v in mc_summary.items() if "of_limit" not in k},
        None, None, [10.0] * 100,
    )
    assert with_headroom.headroom_bonus == pytest.approx(0.1 * 0.8)
    assert without.headroom_bonus == pytest.approx(0.0)
    assert with_headroom.final_score > without.final_score


def test_compute_prop_fitness_headroom_weight_zero_disables():
    mc_summary = {
        "per_attempt_pass_probability": 70.0,
        "per_attempt_payout_probability": 50.0,
        "best_day_profit_pct_of_limit": 0.2,
    }
    stats = {"max_drawdown_pct": 5.0, "total_trades": 100}
    b = compute_prop_fitness(stats, mc_summary, None, None, [10.0] * 100, weights={"headroom_bonus": 0.0})
    assert b.headroom_bonus == pytest.approx(0.0)


def test_refinement_composite_headroom_bonus():
    mc = _mc_result(
        per_attempt_pass_probability=70.0,
        per_attempt_payout_probability=50.0,
        risk_of_ruin_pct=10.0,
    )
    prop_summary = {
        "best_day_profit_pct_of_limit": 0.2,
        "worst_daily_loss_pct_of_limit": 0.2,
        "max_dd_pct_of_limit": 0.2,
    }
    with_headroom = compute_fitness({}, prop_summary, mc, "composite_prop_score")
    without = compute_fitness({}, None, mc, "composite_prop_score")
    # 0.1 * mean(0.8, 0.8, 0.8) = 0.08 additive on the composite scale.
    assert with_headroom - without == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# B11a: multi-preset champion board row helper
# ---------------------------------------------------------------------------

def test_multi_preset_champion_row_prefers_champion_candidate():
    from app.web.server import _multi_preset_champion_row

    class FakeSummary:
        champion_candidate_id = "cand-2"
        leaderboard = [
            {"candidate_id": "cand-1", "composite_score": 99.0,
             "mc_summary": {"per_attempt_pass_probability": 10.0},
             "statistics": {"net_profit": 1.0}},
            {"candidate_id": "cand-2", "composite_score": 50.0, "family": "ema",
             "mc_summary": {"per_attempt_pass_probability": 33.0,
                            "per_attempt_payout_probability": 20.0,
                            "evaluation_pass_probability": 90.0},
             "statistics": {"net_profit": 500.0, "profit_factor": 1.4},
             "passed_stage3_gate": True},
        ]

    row = _multi_preset_champion_row(FakeSummary())
    assert row["candidate_id"] == "cand-2"
    assert row["per_attempt_pass_probability"] == pytest.approx(33.0)
    assert row["family"] == "ema"
    assert row["passed_stage3_gate"] is True


def test_multi_preset_champion_row_empty_leaderboard():
    from app.web.server import _multi_preset_champion_row

    class FakeSummary:
        champion_candidate_id = None
        leaderboard = []

    assert _multi_preset_champion_row(FakeSummary()) is None
