"""
Payout-cadence account-scaling stress test.

Many modern prop firms scale a FUNDED account's size up after the trader
collects a run of consecutive payouts (a common shape: +25% account size
every 4 consecutive profitable payout cycles, up to some cap such as 4x
or a fixed dollar ceiling). Nothing in this app's existing payout
tooling models that -- app.prop.survival_engine.run_prop_survival_analysis
and app.monte_carlo.engine.run_monte_carlo both run the funded stage
against ONE fixed account_size for the account's entire simulated life.
This module adds an optional, separate scaling-aware stress test that
sits ALONGSIDE those (it does not replace or modify either), so a trader
can see how a scaling plan changes the expected payout trajectory
without touching the existing (non-scaling) numbers everywhere else in
the app.

Modeling approach -- stated plainly, because this is an approximation:
when the account scales from size S to size S' = S * scale_multiplier,
this implementation assumes the strategy's position sizing (and
therefore every subsequent trade's dollar P&L) scales by the same
factor S'/S -- i.e. the strategy keeps risking the same % of equity per
trade, just against a bigger base, which is exactly the assumption this
app's own % based position sizing (app.backtest.risk.RiskConfig) already
makes everywhere else. This is NOT a claim that a real broker fills an
order 4x the size with zero additional slippage or liquidity impact --
stress-test that separately via the existing slippage_stress_pct knob
on MonteCarloConfig / PropSurvivalConfig.

Every dollar figure here still comes from the same
app.prop.simulator.simulate_account() rules engine used everywhere
else in the app; this module only handles re-basing account_size and
re-scaling subsequent trade P&Ls across a chain of "segments" split at
each scale-up event. No prop-firm rule logic is reimplemented.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from app.backtest.execution import Trade
from app.monte_carlo.engine import MonteCarloConfig, _apply_slippage_stress, _resample_pnls
from app.prop.simulator import (
    AccountSimResult, PayoutEvent, PropRules, precompute_day_structure, simulate_account,
)


@dataclass
class ScalingPlan:
    payouts_per_scale: int = 4              # consecutive payouts required to trigger the next scale-up
    scale_multiplier: float = 1.25          # new_size = current_size * scale_multiplier
    max_scale_multiple: float = 4.0         # cap: never scale beyond base_account_size * this multiple
    reset_payout_count_on_scale: bool = True  # most real plans restart the counter after each scale-up

    def __post_init__(self):
        self.payouts_per_scale = max(int(self.payouts_per_scale), 1)
        self.scale_multiplier = max(float(self.scale_multiplier), 1.0)
        self.max_scale_multiple = max(float(self.max_scale_multiple), 1.0)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ScalingSimResult:
    base_result: AccountSimResult      # what happened at the ORIGINAL (unscaled) account size, for comparison
    scaled_result: AccountSimResult    # what happened WITH the scaling plan applied
    scale_events: list                  # list of dicts: {payout_count, day_index, old_size, new_size}
    final_account_size: float
    total_scale_ups: int

    def to_dict(self) -> dict:
        return {
            "base_final_balance": self.base_result.final_balance,
            "base_total_payouts": sum(p.amount for p in self.base_result.payouts),
            "scaled_final_balance": self.scaled_result.final_balance,
            "scaled_total_payouts": sum(p.amount for p in self.scaled_result.payouts),
            "scale_events": list(self.scale_events),
            "final_account_size": self.final_account_size,
            "total_scale_ups": self.total_scale_ups,
        }


def _rules_with_size(rules: PropRules, size: float) -> PropRules:
    """A copy of `rules` re-based to a new account_size for the next
    segment. Only account_size changes -- every % rule (drawdown, daily
    loss, profit target, payout threshold, etc.) stays the same
    percentage, which is exactly what 'scaling' means in a real funded
    plan: the % rules don't change, only what they're a % OF."""
    return dataclasses.replace(rules, account_size=size)


def simulate_account_with_scaling(
    trade_pnls: list[float],
    trade_dates: list,
    rules: PropRules,
    plan: ScalingPlan,
) -> ScalingSimResult:
    """Runs the SAME trade sequence twice: once through the plain (no
    scaling) simulate_account() for comparison, and once through a
    scaling-aware segment loop that re-invokes simulate_account() fresh
    for each segment between scale-up events."""
    base_result = simulate_account(trade_pnls, trade_dates, rules)

    base_size = rules.account_size
    current_size = base_size
    consecutive_payouts = 0
    scale_events: list[dict] = []

    remaining_pnls = list(trade_pnls)
    remaining_dates = list(trade_dates)
    all_payouts: list[PayoutEvent] = []
    day_offset = 0
    final_result: AccountSimResult | None = None

    while remaining_pnls:
        cur_rules = _rules_with_size(rules, current_size)
        segment = simulate_account(remaining_pnls, remaining_dates, cur_rules)
        for p in segment.payouts:
            all_payouts.append(PayoutEvent(
                day_index=p.day_index + day_offset, date=p.date,
                amount=p.amount, balance_after=p.balance_after,
            ))
        final_result = segment

        if not segment.payouts or current_size >= base_size * plan.max_scale_multiple:
            break

        # Walk this segment's payouts to find the first one that
        # completes a full payouts_per_scale batch -- scaling happens
        # only AFTER a full batch of consecutive payouts, never before.
        trigger = None
        for p in segment.payouts:
            consecutive_payouts += 1
            if consecutive_payouts >= plan.payouts_per_scale:
                trigger = p
                break
        if trigger is None:
            break  # never completed another full batch this segment -- done scaling

        old_size = current_size
        current_size = min(current_size * plan.scale_multiplier, base_size * plan.max_scale_multiple)
        if plan.reset_payout_count_on_scale:
            consecutive_payouts = 0
        scale_events.append({
            "payout_count": len(all_payouts),
            "day_index": trigger.day_index + day_offset,
            "old_size": old_size,
            "new_size": current_size,
        })

        # Cut the remaining trade sequence at the trigger day and rescale
        # everything after it by the size ratio.
        ds = precompute_day_structure(remaining_dates)
        cut_trade_idx = len(remaining_pnls)
        for i, d in enumerate(ds.day_index_per_trade):
            if d > trigger.day_index:
                cut_trade_idx = i
                break
        factor = current_size / old_size if old_size else 1.0
        day_offset += trigger.day_index + 1
        remaining_pnls = [p * factor for p in remaining_pnls[cut_trade_idx:]]
        remaining_dates = remaining_dates[cut_trade_idx:]
        if not remaining_pnls:
            break

    scaled_final = final_result if final_result is not None else base_result
    scaled_result = AccountSimResult(
        passed_evaluation=scaled_final.passed_evaluation or base_result.passed_evaluation,
        failed=scaled_final.failed,
        failure_reason=scaled_final.failure_reason,
        failure_day_index=(
            scaled_final.failure_day_index + day_offset if scaled_final.failure_day_index is not None else None
        ),
        days_to_pass=base_result.days_to_pass,
        first_payout_day_index=all_payouts[0].day_index if all_payouts else None,
        first_payout_amount=all_payouts[0].amount if all_payouts else None,
        payouts=all_payouts,
        final_balance=scaled_final.final_balance,
        max_drawdown_pct_reached=max(base_result.max_drawdown_pct_reached, scaled_final.max_drawdown_pct_reached),
        trading_days_count=day_offset + scaled_final.trading_days_count,
    )

    return ScalingSimResult(
        base_result=base_result, scaled_result=scaled_result,
        scale_events=scale_events, final_account_size=current_size,
        total_scale_ups=len(scale_events),
    )


@dataclass
class ScalingStressResult:
    n_simulations: int
    probability_any_scale_up: float
    median_total_scale_ups: float
    median_final_account_size: float
    expected_total_payout_unscaled: float   # mean total $ withdrawn WITHOUT the scaling plan, same resamples
    expected_total_payout_scaled: float     # mean total $ withdrawn WITH the scaling plan, same resamples
    payout_uplift_pct: float                # % increase in expected total payout the scaling plan adds
    plan: ScalingPlan

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["plan"] = self.plan.to_dict()
        return d


def run_scaling_stress_test(
    trades: list[Trade],
    rules: PropRules,
    plan: ScalingPlan,
    mc_cfg: MonteCarloConfig | None = None,
) -> ScalingStressResult:
    """Monte-Carlo version of simulate_account_with_scaling(): resamples
    the trade sequence the same way app.monte_carlo.engine.run_monte_carlo
    does (same _resample_pnls / _apply_slippage_stress helpers, so this
    stays consistent with every other Monte Carlo number in the app),
    runs BOTH the unscaled and scaled simulation on each resample, and
    aggregates the difference -- this is the number that answers "does a
    real scaling plan make the expected-payout estimate meaningfully
    more realistic for firms that offer one."
    """
    if not trades:
        raise ValueError("No trades to simulate -- run a backtest first.")

    cfg = mc_cfg or MonteCarloConfig(n_simulations=1000)
    pnls = np.array([t.pnl for t in trades], dtype=float)
    dates = [t.entry_time for t in trades]
    rng = np.random.default_rng(cfg.random_seed)

    scale_up_flags: list[bool] = []
    scale_up_counts: list[int] = []
    final_sizes: list[float] = []
    unscaled_payouts: list[float] = []
    scaled_payouts: list[float] = []

    for _ in range(cfg.n_simulations):
        resampled = _resample_pnls(rng, pnls, cfg)
        resampled = _apply_slippage_stress(resampled, cfg.slippage_stress_pct)
        result = simulate_account_with_scaling(list(resampled), dates, rules, plan)
        scale_up_flags.append(result.total_scale_ups > 0)
        scale_up_counts.append(result.total_scale_ups)
        final_sizes.append(result.final_account_size)
        unscaled_payouts.append(sum(p.amount for p in result.base_result.payouts))
        scaled_payouts.append(sum(p.amount for p in result.scaled_result.payouts))

    expected_unscaled = float(np.mean(unscaled_payouts)) if unscaled_payouts else 0.0
    expected_scaled = float(np.mean(scaled_payouts)) if scaled_payouts else 0.0
    uplift = (
        (expected_scaled - expected_unscaled) / expected_unscaled * 100.0
        if expected_unscaled > 0 else 0.0
    )

    return ScalingStressResult(
        n_simulations=cfg.n_simulations,
        probability_any_scale_up=float(np.mean(scale_up_flags) * 100.0) if scale_up_flags else 0.0,
        median_total_scale_ups=float(np.median(scale_up_counts)) if scale_up_counts else 0.0,
        median_final_account_size=float(np.median(final_sizes)) if final_sizes else rules.account_size,
        expected_total_payout_unscaled=expected_unscaled,
        expected_total_payout_scaled=expected_scaled,
        payout_uplift_pct=round(uplift, 2),
        plan=plan,
    )
