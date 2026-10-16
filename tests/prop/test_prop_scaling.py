import pandas as pd

from app.backtest.execution import Trade
from app.monte_carlo.engine import MonteCarloConfig
from app.prop.scaling import ScalingPlan, run_scaling_stress_test, simulate_account_with_scaling
from app.prop.simulator import PropRules


def _make_trades(pnls, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(pnls), freq="D")
    trades = []
    for d, pnl in zip(dates, pnls):
        trades.append(Trade(
            entry_time=d, exit_time=d, direction=1, entry_price=100.0, exit_price=100.0 + pnl,
            size=1.0, pnl=pnl, pnl_pct=pnl / 10000.0, exit_reason="signal", commission=0.0,
            equity_after=10000.0 + pnl,
        ))
    return trades


def _dates_and_pnls(pnls, start="2024-01-01"):
    dates = list(pd.date_range(start, periods=len(pnls), freq="D"))
    return pnls, dates


def test_scaling_plan_clamps_invalid_values():
    plan = ScalingPlan(payouts_per_scale=0, scale_multiplier=0.5, max_scale_multiple=0.1)
    assert plan.payouts_per_scale >= 1
    assert plan.scale_multiplier >= 1.0
    assert plan.max_scale_multiple >= 1.0


def test_simulate_with_scaling_no_payouts_matches_base_result():
    rules = PropRules(account_size=10_000, evaluation_profit_target_pct=50, daily_loss_limit_pct=100, max_drawdown_pct=100)
    pnls, dates = _dates_and_pnls([10, -5, 8, -3] * 3)
    plan = ScalingPlan(payouts_per_scale=1, scale_multiplier=1.25, max_scale_multiple=4.0)
    result = simulate_account_with_scaling(pnls, dates, rules, plan)
    assert result.total_scale_ups == 0
    assert result.final_account_size == rules.account_size


def test_simulate_with_scaling_triggers_scale_up():
    rules = PropRules(
        account_size=10_000, evaluation_profit_target_pct=2, daily_loss_limit_pct=100,
        max_drawdown_pct=100, consistency_rule_pct=None, min_trading_days=1,
        payout_threshold_pct=1, payout_frequency_days=1, required_buffer_pct=0,
    )
    # Big early profit passes eval and immediately queues up several payouts.
    pnls, dates = _dates_and_pnls([500] + [300] * 20)
    plan = ScalingPlan(payouts_per_scale=2, scale_multiplier=1.5, max_scale_multiple=3.0, reset_payout_count_on_scale=True)
    result = simulate_account_with_scaling(pnls, dates, rules, plan)
    assert result.total_scale_ups >= 1
    assert result.final_account_size > rules.account_size
    assert result.final_account_size <= rules.account_size * plan.max_scale_multiple


def test_simulate_with_scaling_never_exceeds_cap():
    rules = PropRules(
        account_size=10_000, evaluation_profit_target_pct=2, daily_loss_limit_pct=100,
        max_drawdown_pct=100, consistency_rule_pct=None, min_trading_days=1,
        payout_threshold_pct=1, payout_frequency_days=1, required_buffer_pct=0,
    )
    pnls, dates = _dates_and_pnls([500] + [300] * 60)
    plan = ScalingPlan(payouts_per_scale=1, scale_multiplier=2.0, max_scale_multiple=2.0)
    result = simulate_account_with_scaling(pnls, dates, rules, plan)
    assert result.final_account_size <= rules.account_size * 2.0 + 1e-6


def test_run_scaling_stress_test_raises_on_no_trades():
    rules = PropRules(account_size=10_000)
    plan = ScalingPlan()
    try:
        run_scaling_stress_test([], rules, plan)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_scaling_stress_test_produces_uplift_metrics():
    trades = _make_trades([500] + [200, -50, 150, -30] * 15)
    rules = PropRules(
        account_size=10_000, evaluation_profit_target_pct=2, daily_loss_limit_pct=100,
        max_drawdown_pct=100, consistency_rule_pct=None, min_trading_days=1,
        payout_threshold_pct=1, payout_frequency_days=1, required_buffer_pct=0,
    )
    plan = ScalingPlan(payouts_per_scale=2, scale_multiplier=1.25, max_scale_multiple=4.0)
    result = run_scaling_stress_test(trades, rules, plan, mc_cfg=MonteCarloConfig(n_simulations=50, random_seed=1))
    assert result.n_simulations == 50
    assert 0.0 <= result.probability_any_scale_up <= 100.0
    assert result.expected_total_payout_scaled >= result.expected_total_payout_unscaled - 1e-6
