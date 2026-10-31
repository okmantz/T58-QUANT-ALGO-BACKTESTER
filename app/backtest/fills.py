"""
Shared fill, cost and entry-planning logic.

Single-instrument execution (``app.backtest.execution.run_execution``) and the unified portfolio engine
(``app.backtest.portfolio_engine``) both need to answer the same instrument-level questions:

  * what price does an order actually fill at (spread / slippage, gap-through stops)?
  * what does a closed position cost (round-turn spread/slippage, flat + per-contract commission)?
  * how big is a new position, where are its stop / target / trailing distance, and is it allowed at all
    (``RiskConfig.size_for_stop``, whole-contract flooring, fit-stop, micro fallback, adaptive risk)?
  * how is an exit classified (stop / breakeven / trailing / take-profit / ...)?

Those answers are pure functions of one instrument's ``RiskConfig`` plus one bar's prices, so they live here
ONCE and both engines call them -- there is deliberately no second copy to drift. Everything here was lifted
out of ``run_execution`` verbatim (same arithmetic, same order of operations); ``run_execution`` now calls it.

Nothing in this module knows about accounts, equity curves, prop rules or the bar loop -- that is what the
engines own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.backtest.risk import (
    SKIP_COSTS_EXCEED_BUDGET, SKIP_NO_MICRO, SKIP_PLANNED_RISK, SKIP_STOP_CAP_TOO_TIGHT,
    SKIP_STOP_TOO_WIDE, SKIP_TOO_WIDE_FOR_MICRO,
)

# Sizing skip reasons that mean "the budget cannot afford this signal" (as opposed to a degenerate stop / no
# money) -- they feed the sizing-halt diagnostic and the preflight's skip-rate gate.
SIZING_SKIP_REASONS = frozenset({
    SKIP_STOP_TOO_WIDE, SKIP_COSTS_EXCEED_BUDGET, SKIP_STOP_CAP_TOO_TIGHT,
    SKIP_TOO_WIDE_FOR_MICRO, SKIP_NO_MICRO, SKIP_PLANNED_RISK,
})

DEFAULT_STOP_PCT_OF_PRICE = 0.01  # 1% of entry price, used only when a strategy defines no stop at all

# C6 (market-impact guardrail): entries above this many whole contracts fill at the quoted price with zero
# market impact assumed -- the engines warn (RuntimeWarning) instead of silently pretending a 30+ lot sweep
# doesn't move the book.
MAX_CONTRACTS_BEFORE_IMPACT_WARN = 30


def sizing_mode_is_contract_skip(risk) -> bool:
    """The bounded dead-lock guard applies only to the plain 'skip' mode;
    fit_stop / micro_fallback already keep a small account trading."""
    return getattr(risk, "sizing_mode", "skip") == "skip"


# --------------------------------------------------------------------------------------------------------------
# Fill prices and costs
# --------------------------------------------------------------------------------------------------------------
def side_cost(risk) -> float:
    """Spread + slippage, in price units, charged on EACH side of a round turn."""
    return risk.spread_pips * risk.pip_size + risk.slippage_pips * risk.pip_size


def entry_fill_price(risk, raw_price: float, direction: int) -> float:
    return raw_price + side_cost(risk) * direction


def exit_fill_price(risk, raw_exit_price: float, direction: int) -> float:
    return raw_exit_price - side_cost(risk) * direction


def resolve_intrabar_exit(direction_: int, stop, take, low: float, high: float, open_: float):
    """Pure fill-resolution logic: does this bar's high/low trigger the resting stop or target, and if so, at
    what honest price? A resting stop that the bar gapped straight through does NOT fill at the stop level --
    it fills at the open, which is worse. Stop beats target on a bar that spans both."""
    if direction_ == 1:
        if stop is not None and low <= stop:
            return min(stop, open_), "stop_loss"
        if take is not None and high >= take:
            return take, "take_profit"
    else:
        if stop is not None and high >= stop:
            return max(stop, open_), "stop_loss"
        if take is not None and low <= take:
            return take, "take_profit"
    return None, None


def exit_cause_for(reason: str, open_pos: dict) -> str:
    """Maps the engine-internal exit `reason` to the machine-verified exit_cause taxonomy stored on
    Trade.exit_cause. A stop resting AT the entry price is a "breakeven", a stop the trailing ratchet moved is a
    "trailing_stop" -- lumping those in with plain stop losses is the "entries fine, exits broken" blindness
    this taxonomy closes."""
    reason = reason.removesuffix("_invalid_pnl_skipped")
    if reason == "stop_loss":
        stop_now = open_pos.get("stop_price")
        entry = open_pos.get("entry_price")
        direction_ = open_pos.get("direction")
        init_risk = open_pos.get("initial_risk")
        if (
            stop_now is not None and entry is not None
            and math.isclose(stop_now, entry, rel_tol=1e-9, abs_tol=1e-12)
        ):
            return "breakeven"
        if init_risk and direction_ and stop_now is not None and entry is not None:
            orig_stop = entry - direction_ * init_risk
            if (
                open_pos.get("trailing_distance")
                and not math.isclose(stop_now, orig_stop, rel_tol=1e-9, abs_tol=1e-12)
            ):
                return "trailing_stop"
        return "stop_loss"
    return {
        "take_profit": "take_profit",
        "signal": "signal_exit",
        "time_stop": "time_stop",
        "daily_loss_limit_forced_close": "daily_loss_close",
        "weekend_hold_forced_close": "session_close",
        "session_flatten_forced_close": "session_close",
        "account_blown_forced_close": "session_close",
        "partial_take_profit": "take_profit",
        "end_of_data": "end_of_data",
    }.get(reason, reason)


def clamp_trade_loss(risk, pnl_value: float, equity_at_entry: float) -> tuple[float, bool]:
    """A single trade's loss can never exceed what the account actually has to lose (negative-balance
    protection) or a configured hard per-trade ceiling (RiskConfig.max_trade_loss). Wins are never touched.
    Returns (pnl, was_clamped)."""
    if pnl_value >= 0 or not math.isfinite(pnl_value):
        return pnl_value, False
    cap = risk.max_trade_loss(equity_at_entry)
    if cap and -pnl_value > cap:
        return -cap, True
    return pnl_value, False


@dataclass
class SettleMath:
    filled_exit_price: float
    pnl: float                  # after costs, BEFORE the per-trade loss clamp
    commission: float
    reason: str                 # possibly suffixed "_invalid_pnl_skipped"
    gap_loss: bool              # a stop fill several times worse than the trade's sized risk


def settle_math(risk, open_pos: dict, raw_exit_price: float, reason: str, direction_: int) -> SettleMath:
    """Round-turn costs for closing ALL of ``open_pos``: spread/slippage on the exit side, flat + per-contract
    commission (B2-4), and the non-finite-pnl guard."""
    filled = exit_fill_price(risk, raw_exit_price, direction_)
    pnl = (filled - open_pos["entry_price"]) * open_pos["size"] * direction_
    pos_cs = open_pos.get("contract_size", risk.contract_size)
    pos_cpc = open_pos.get("commission_per_contract", risk.commission_per_contract)
    contracts_closed = open_pos["size"] / pos_cs if pos_cs else 0.0
    commission = risk.commission_per_trade + pos_cpc * contracts_closed
    pnl -= commission
    if not math.isfinite(pnl):
        # Guard against a runaway/degenerate trade (e.g. an entry sized off a near-zero ATR-based stop
        # distance) ever corrupting the equity curve with NaN/inf.
        pnl = 0.0
        reason = f"{reason}_invalid_pnl_skipped"
    gap = bool(
        reason == "stop_loss"
        and open_pos["initial_risk"]
        and abs(pnl) > 3 * open_pos["initial_risk"] * open_pos["size"]
    )
    return SettleMath(filled, pnl, commission, reason, gap)


@dataclass
class PartialSettleMath:
    partial_size: float
    filled_exit_price: float
    pnl: float                  # BEFORE the loss clamp
    commission: float


def partial_settle_math(risk, open_pos: dict, fill_price: float, direction_: int, fraction: float) -> PartialSettleMath:
    """Closes `fraction` of the ORIGINAL size: same spread/slippage/commission treatment as a full settle,
    commission pro-rata to the fraction closed."""
    partial_size = open_pos["initial_size"] * fraction
    filled = exit_fill_price(risk, fill_price, direction_)
    pnl = (filled - open_pos["entry_price"]) * partial_size * direction_
    pos_cs = open_pos.get("contract_size", risk.contract_size)
    pos_cpc = open_pos.get("commission_per_contract", risk.commission_per_contract)
    contracts_total = open_pos["initial_size"] / pos_cs if pos_cs else 0.0
    commission = (risk.commission_per_trade + pos_cpc * contracts_total) * fraction
    pnl -= commission
    if not math.isfinite(pnl):
        pnl = 0.0
    return PartialSettleMath(partial_size, filled, pnl, commission)


# --------------------------------------------------------------------------------------------------------------
# Entry planning: size, stop, target, trailing distance
# --------------------------------------------------------------------------------------------------------------
@dataclass
class EntryPlan:
    """Everything the engines need to open (or count a skip of) one entry.

    ``position`` is the open-position dict (without entry_time / entry_bar_idx, which the caller owns) or None
    when the entry was skipped. The boolean/float fields are the bookkeeping the caller turns into its own
    counters and warnings -- exactly the counters ``run_execution`` has always kept."""
    position: dict | None
    skipped: bool = False
    skip_reason: str | None = None            # SizingDecision.skip_reason when skipped
    skip_is_floor: bool = False               # skipped AND counts toward the "zero whole contracts" tally
    adaptive_blocked: bool = False            # skipped while the adaptive multiplier was 0
    min_contract_rescue: bool = False
    adaptive_probe: bool = False
    used_fallback_stop: bool = False
    stop_capped: bool = False
    used_micro: bool = False
    pip_mismatch_ratio: float | None = None
    atr_mismatch_ratio: float | None = None
    impact_contracts: float | None = None
    intended_risk_dollars: float | None = None


def plan_entry(
    *,
    risk,
    equity: float,
    direction: int,
    raw_price: float,
    bar_sl_distance: float | None,
    bar_tp_distance: float | None,
    bar_trail_distance: float | None,
    stop_loss_pips: float | None,
    take_profit_pips: float | None,
    adaptive_risk=None,
    adaptive_state=None,
    vol_pct: float | None = None,
    atr_now: float | None = None,
) -> EntryPlan:
    """The entry block of ``run_execution``, verbatim, as a function of one bar's inputs.

    ``equity`` is the balance the position is sized (and its loss clamped) against. ``vol_pct`` is the current
    volatility percentile for adaptive-risk volatility triggers (None when unused / NaN). ``atr_now`` is the
    14-bar ATR yardstick for the fixed-pips scale-mismatch check."""
    spread_slip = side_cost(risk)
    entry_price = raw_price + spread_slip * direction

    used_fallback_stop = False
    if not bar_sl_distance and not stop_loss_pips:
        # The strategy defined no stop loss at all (no ATR-based distance and no fixed pips). Fall back to a stop
        # sized as a percentage of the entry price -- scales correctly regardless of instrument or pip size.
        bar_sl_distance = abs(raw_price) * DEFAULT_STOP_PCT_OF_PRICE
        used_fallback_stop = True

    if bar_sl_distance:
        sl_price_dist = bar_sl_distance
    else:
        sl_price_dist = (stop_loss_pips or 0) * risk.pip_size
    intended_risk_dollars = risk.risk_amount(equity)
    decision = risk.size_for_stop(equity, sl_price_dist, tp_distance=bar_tp_distance)
    size = decision.units
    pos_contract_size = decision.contract_size if decision.contract_size else risk.contract_size
    pos_commission_pc = decision.commission_per_contract
    rescue_above_risk_target = bool(decision.above_budget)

    # Dead-lock guard (bounded): a config that a FRESH account can afford (>= 1 whole contract at
    # initial_balance) must not stall forever just because a small loss pulled equity a hair under the
    # one-contract threshold.
    deadlock_guard = False
    if size <= 0 and risk.contract_size and equity > 0 and decision.skip_reason is not None \
            and sizing_mode_is_contract_skip(risk):
        fresh = risk.size_for_stop(risk.initial_balance, sl_price_dist, tp_distance=bar_tp_distance)
        if fresh.units > 0 and fresh.contracts >= 1:
            one = risk.worst_case_loss(
                float(risk.contract_size), sl_price_dist, risk.contract_size, risk.commission_per_contract,
            )
            deadlock_guard = one <= 1.5 * max(intended_risk_dollars, 1e-9)
    min_contract_rescue = False
    if size <= 0 and (risk.allow_single_contract_minimum or deadlock_guard) and risk.contract_size and equity > 0:
        size = float(risk.contract_size)
        pos_contract_size = risk.contract_size
        pos_commission_pc = risk.commission_per_contract
        min_contract_rescue = True
        rescue_above_risk_target = True

    adaptive_multiplier = 1.0
    adaptive_rules_active: list[str] = []
    adaptive_probe = False
    if adaptive_risk is not None and adaptive_risk.enabled:
        adaptive_multiplier = adaptive_state.active_multiplier(adaptive_risk, vol_pct)
        adaptive_rules_active = adaptive_state.active_rule_labels(adaptive_risk, vol_pct)
        size_before_adaptive = size
        size *= adaptive_multiplier
        # Adaptive throttling may shrink size TOWARD the 1-contract minimum, never past it: re-floor to whole
        # contracts, and if flooring would erase a trade the UNTHROTTLED sizing could afford (>= 1 contract),
        # open at the 1-contract minimum instead. A multiplier of exactly 0.0 still blocks outright.
        if pos_contract_size and size > 0 and adaptive_multiplier > 0:
            lots = math.floor(size / pos_contract_size + 1e-9)
            if lots <= 0 and size_before_adaptive >= pos_contract_size:
                lots = 1
                adaptive_probe = True
            size = lots * pos_contract_size

    if not math.isfinite(size) or size <= 0:
        # Degenerate sizing (e.g. an ATR-based stop distance that rounds to ~0 for this bar) -- skip this
        # entry rather than opening a trade with an invalid size.
        is_floor = bool(risk.contract_size and decision.skip_reason in SIZING_SKIP_REASONS)
        return EntryPlan(
            position=None, skipped=True, skip_reason=decision.skip_reason, skip_is_floor=is_floor,
            adaptive_blocked=adaptive_multiplier <= 0, min_contract_rescue=min_contract_rescue,
            adaptive_probe=adaptive_probe, intended_risk_dollars=intended_risk_dollars,
        )

    stop_price = None
    take_price = None
    pip_ratio = None
    atr_ratio = None
    if bar_sl_distance:
        stop_price = entry_price - direction * bar_sl_distance
    elif stop_loss_pips:
        fixed_stop_distance = stop_loss_pips * risk.pip_size
        stop_price = entry_price - direction * fixed_stop_distance
        # Sanity-check the fixed pip stop against this bar's actual price level (see run_execution history): a
        # STOP_LOSS_PIPS is only meaningful if risk.pip_size matches the instrument. Never changes behaviour --
        # it only reports the ratios so the caller can raise one clear warning.
        if raw_price:
            distance_ratio = abs(fixed_stop_distance) / abs(raw_price)
            if distance_ratio < 0.0002 or distance_ratio > 0.25:
                pip_ratio = distance_ratio
        # Second, independent check against the instrument's own recent ATR.
        if atr_now and atr_now > 0:
            ratio = abs(fixed_stop_distance) / atr_now
            if ratio < 0.15:
                atr_ratio = ratio
    if decision.stop_capped:
        # fit_stop: 1 contract, stop shrunk to the distance whose worst case equals the budget. The target
        # follows per risk.fit_stop_target (scale / keep / fixed_r); a trailing distance shrinks with the stop
        # so the strategy's shape is preserved.
        stop_price = entry_price - direction * decision.stop_distance
        base_tp = bar_tp_distance if bar_tp_distance else (
            take_profit_pips * risk.pip_size if take_profit_pips else None
        )
        if risk.fit_stop_target == "fixed_r":
            bar_tp_distance = risk.fit_stop_target_r * decision.stop_distance
        elif base_tp and risk.fit_stop_target == "scale":
            bar_tp_distance = base_tp * decision.stop_scale
        elif base_tp:
            bar_tp_distance = base_tp
        if bar_trail_distance:
            bar_trail_distance = bar_trail_distance * decision.stop_scale
    if bar_tp_distance:
        take_price = entry_price + direction * bar_tp_distance
    elif take_profit_pips:
        take_price = entry_price + direction * take_profit_pips * risk.pip_size

    initial_risk = abs(entry_price - stop_price) if stop_price is not None else None

    impact_contracts = None
    if pos_contract_size:
        whole = size / pos_contract_size
        if whole > MAX_CONTRACTS_BEFORE_IMPACT_WARN:
            impact_contracts = whole

    position = {
        "direction": direction,
        "entry_price": entry_price,
        "size": size,
        "initial_size": size,
        "stop_price": stop_price,
        "take_price": take_price,
        "equity_at_entry": equity,
        "best_price": entry_price,
        "worst_price": entry_price,
        "mfe_price": entry_price,
        "initial_risk": initial_risk,
        "intended_risk_dollars": intended_risk_dollars,
        # 2026-10-31 audit (section 3): under sizing_mode="rr_planned" the sizer plans anchor/RR dollars of risk, which is
        # BELOW the configured cap (intended_risk_dollars). Record the plan so reconciliation compares against it, and
        # flag trades whose stop was the 1%-of-price placeholder (the strategy defined none) so they can be excluded.
        "planned_risk_dollars": decision.planned_risk,
        "used_fallback_stop": bool(used_fallback_stop),
        "breakeven_done": False,
        "partial_taken": False,
        "trailing_distance": bar_trail_distance,
        "adaptive_multiplier": adaptive_multiplier,
        "adaptive_rules_active": adaptive_rules_active,
        "sized_above_risk_target": rescue_above_risk_target,
        "contract_size": pos_contract_size,
        "commission_per_contract": pos_commission_pc,
        "risk_at_stop_dollars": risk.worst_case_loss(
            size, abs(entry_price - stop_price) if stop_price is not None else sl_price_dist,
            pos_contract_size, pos_commission_pc,
        ),
        "stop_capped": bool(decision.stop_capped),
        "sizing_mode": risk.sizing_mode,
        "used_micro": bool(decision.used_micro),
        "contracts": (size / pos_contract_size) if pos_contract_size else None,
        "fill_resolution": "bar",
    }
    return EntryPlan(
        position=position, skipped=False, skip_reason=decision.skip_reason,
        min_contract_rescue=min_contract_rescue, adaptive_probe=adaptive_probe,
        used_fallback_stop=used_fallback_stop, stop_capped=bool(decision.stop_capped),
        used_micro=bool(decision.used_micro), pip_mismatch_ratio=pip_ratio, atr_mismatch_ratio=atr_ratio,
        impact_contracts=impact_contracts, intended_risk_dollars=intended_risk_dollars,
    )
