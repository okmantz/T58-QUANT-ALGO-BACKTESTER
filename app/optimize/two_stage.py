"""Two-stage optimizer (report 2026-10-20, section 4 / 6).

The old objective mixed edge discovery with prop-outcome sizing: a GA that optimises eval-pass probability is partly choosing
risk-per-trade luck and window length, and the sizing it implicitly picked is never reported. The two stages separate them.

Stage A -- EDGE DISCOVERY at fixed unit risk, measured in R
    fitness = lower confidence bound of mean R per trade (day-block bootstrap) on purged CV folds, run with COSTS STRESSED x2,
    with a trade floor and a year-stability weight. ``edge_fitness_from_trades`` is the metric ("edge_lb_r" in refinement);
    ``stress_risk`` builds the stressed RiskConfig; ``stage_a_summary`` reports survivors.

Stage B -- SIZING per survivor (or per portfolio) once the edge exists
    scan risk-per-trade 0.25%..1.5% x eligible firms; for each cell re-scale the SAME trades to that risk, run the per-attempt
    Monte Carlo and the bankroll survival, compute EV (ev_verdict) and keep the cell with the best EV subject to
    bankroll ruin <= cap and funded breach <= cap. ``stage_b_scan`` returns every cell so the choice is auditable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from app.validation.edge_stats import day_block_bootstrap_mean, edge_by_year

DEFAULT_RISK_GRID_PCT = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def trade_r_and_days(trades) -> tuple[list, list]:
    from app.search.robustness import _r_and_days
    return _r_and_days(trades)


def edge_fitness_from_trades(trades, min_trades: int = 30, conf: float = 0.95, n_boot: int = 400) -> float:
    """Stage A fitness: day-block-bootstrap lower bound of mean R, weighted by the share of positive calendar years.

    Returns -inf when there are too few measurable trades (cannot be ranked). A negative bound is returned as is, so a losing
    candidate never improves by trading less. n_boot is modest because the GA calls this thousands of times; the final gate
    re-runs the full 2,000-draw bootstrap."""
    rs, days = trade_r_and_days(trades)
    if len(rs) < min_trades:
        return float("-inf")
    bs = day_block_bootstrap_mean(rs, days, conf=conf, n_boot=n_boot, seed=11)
    lb = bs.get("lb")
    if lb is None or not math.isfinite(lb):
        return float("-inf")
    years = edge_by_year(rs, days)
    if len(years) >= 2 and lb > 0:
        frac = sum(1 for y in years if y["mean_r"] > 0) / len(years)
        lb *= 0.5 + 0.5 * frac
    return float(lb)


def stress_risk(risk, multiplier: float = 2.0):
    """RiskConfig with spread and slippage (the tick-denominated components) multiplied; posted commission is not scaled."""
    return replace(risk, spread_pips=risk.spread_pips * multiplier, slippage_pips=risk.slippage_pips * multiplier)


@dataclass
class StageASummary:
    n_trades: int
    mean_r: float | None
    lb_r: float | None
    ub_r: float | None
    years: list = field(default_factory=list)
    survives: bool = False
    reasons: list = field(default_factory=list)


def stage_a_summary(trades, min_trades: int = 150, min_positive_year_frac: float = 0.5) -> StageASummary:
    rs, days = trade_r_and_days(trades)
    if len(rs) < 10:
        return StageASummary(len(rs), None, None, None, reasons=[f"only {len(rs)} measurable trades"])
    bs = day_block_bootstrap_mean(rs, days, n_boot=2000, seed=11)
    years = edge_by_year(rs, days)
    frac = (sum(1 for y in years if y["mean_r"] > 0) / len(years)) if years else 0.0
    out = StageASummary(len(rs), bs.get("mean"), bs.get("lb"), bs.get("ub"), years=years)
    if len(rs) < min_trades:
        out.reasons.append(f"{len(rs)} trades < Stage-A floor {min_trades}")
    if out.lb_r is None or out.lb_r <= 0:
        out.reasons.append(f"lower bound of mean R is {out.lb_r if out.lb_r is None else round(out.lb_r, 3)} (not above zero)")
    if years and frac < min_positive_year_frac:
        out.reasons.append(f"only {frac * 100:.0f}% of calendar years are positive")
    out.survives = not out.reasons
    return out


def rescale_trades_to_risk(trades, account_size: float, risk_pct: float) -> list:
    """Same trades, re-sized so each risks ``risk_pct`` % of ``account_size`` at its stop (price paths unchanged)."""
    out = []
    target = float(account_size) * float(risk_pct) / 100.0
    for t in trades or []:
        ir, size = getattr(t, "initial_risk", None), getattr(t, "size", None)
        if not ir or ir <= 0 or not size:
            continue
        k = target / (float(ir) * float(size))
        out.append(replace(t, size=float(size) * k, pnl=float(t.pnl) * k, commission=float(t.commission) * k))
    return out


@dataclass
class SizingCell:
    firm: str
    risk_pct: float
    pass_pct: float | None = None
    ruin_pct: float | None = None
    bankroll_ruin_pct: float | None = None
    ev: float | None = None
    ev_label: str = ""
    feasible: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def stage_b_scan(trades, firms: dict, *, risk_grid_pct=DEFAULT_RISK_GRID_PCT, ruin_cap_pct: float = 20.0,
                 funded_breach_cap_pct: float = 50.0, bankroll: float = 1000.0, sims: int = 1500, seed: int = 7,
                 bankroll_sims: int = 500, margin_multiple: float = 1.5, margin_points: float = 0.05) -> dict:
    """Scan risk % x firm. ``firms`` maps preset_key -> PropRules (built from the preset). Returns {"cells": [...], "best": cell|None}.

    Uses the trades' OWN R-multiples, so Stage B can never improve the edge, only choose how hard to lean on it."""
    from app.monte_carlo.bankroll import BankrollConfig, simulate_bankroll_survival
    from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
    from app.orchestration.ev_verdict import evaluate_ev
    from app.prop.presets import get_fee_info
    from app.prop.survival_engine import ResetEconomics

    cells: list[SizingCell] = []
    for key, rules in firms.items():
        fee = get_fee_info(key)
        for rp in risk_grid_pct:
            cell = SizingCell(firm=key, risk_pct=float(rp))
            try:
                scaled = rescale_trades_to_risk(trades, rules.account_size, rp)
                if len(scaled) < 30:
                    cell.note = "fewer than 30 measurable trades"
                    cells.append(cell)
                    continue
                mc = run_monte_carlo(scaled, rules, MonteCarloConfig(n_simulations=sims, random_seed=seed, reset_on_breach=False),
                                     selection_bias_caveat=False)
                cell.pass_pct = float(mc.per_attempt_pass_probability)
                cell.ruin_pct = float(mc.risk_of_ruin_pct)
                ruin = None
                if fee is not None:
                    eco = ResetEconomics(evaluation_fee=fee.challenge_price or 0.0, reset_fee=fee.reset_price,
                                         profit_split_pct=fee.profit_split_pct, max_attempts=5)
                    br = simulate_bankroll_survival(scaled, rules, BankrollConfig(
                        starting_bankroll=bankroll, reset_economics=eco, n_simulations=bankroll_sims, method="block_bootstrap",
                        stop_after_first_payout=True))
                    ruin = float(br.probability_ruin_before_first_payout)
                    cell.bankroll_ruin_pct = ruin
                    target = float(rules.account_size) * float(rules.evaluation_profit_target_pct) / 100.0
                    fp = float(mc.first_payout_probability or 0.0) / 100.0
                    gross = (float(mc.expected_payout) / fp) if fp > 0 else None
                    ev = evaluate_ev(
                        legacy_verdict="MARGINAL", holdout_pass_rate=cell.pass_pct / 100.0,
                        window_payout_rate=float(mc.per_attempt_payout_probability or 0.0) / 100.0,
                        payout_value=min(gross, target) if gross else target, challenge_price=fee.challenge_price,
                        reset_price=fee.reset_price, profit_split_pct=fee.profit_split_pct, activation_fee=fee.activation_fee,
                        bankroll_ruin_pct=ruin, ruin_cap_pct=ruin_cap_pct, margin_multiple=margin_multiple, margin_points=margin_points,
                        monthly_subscription=bool(fee.monthly_subscription))
                    cell.ev, cell.ev_label = ev.ev, ev.label
                else:
                    cell.note = "no fee row for this preset: EV not computed"
                breach_ok = cell.ruin_pct is None or cell.ruin_pct <= funded_breach_cap_pct
                ruin_ok = ruin is not None and ruin <= ruin_cap_pct
                cell.feasible = bool(breach_ok and ruin_ok and cell.ev is not None)
            except Exception as exc:  # noqa: BLE001 -- one bad cell never kills the scan
                cell.note = f"failed: {exc}"
            cells.append(cell)
    feas = [c for c in cells if c.feasible and c.ev is not None]
    best = max(feas, key=lambda c: c.ev) if feas else None
    return {"cells": [c.to_dict() for c in cells], "best": best.to_dict() if best else None,
            "ruin_cap_pct": ruin_cap_pct, "funded_breach_cap_pct": funded_breach_cap_pct}
