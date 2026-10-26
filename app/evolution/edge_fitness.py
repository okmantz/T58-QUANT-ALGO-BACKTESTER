"""Evolution Lab's edge-based fitness (Strategy Discovery Engine objective).

Replaces ``compute_prop_fitness`` as the thing the GA maximises. ``final_score`` is in HUNDREDTHS OF R (so +3.0 means a +0.03R
lower bound after the complexity charge), built from the same ``edge_lb_r`` Search Lab ranks on:

    final_score = 100 * (edge_lb_r - complexity_penalty - graveyard_penalty)  -  stability penalties

Stability penalties (all in score points): parameter-neighbourhood fragility, probability of backtest overfitting above 0.5, and
walk-forward / CPCV degradation. Prop-pass probability is NOT in the score; it is carried on the breakdown (``pass_probability``,
``payout_probability``) purely as the last-stage SIZING information and for display.

It returns the same ``PropFitnessBreakdown`` the rest of the engine already consumes, so ``.final_score`` ordering, journals, the
leaderboard and the UI keep working unchanged.
"""
from __future__ import annotations

import math

from app.evolution.prop_fitness import PropFitnessBreakdown, _trade_concentration

SCORE_PER_R = 100.0


def compute_edge_fitness(
    stats: dict, trades, mc_summary: dict | None, robustness_dict: dict | None, walk_forward_dict: dict | None, trade_pnls: list,
    *, spec: dict | None = None, pbo: float | None = None, cpcv_degradation: float | None = None, min_trades_target: int = 100,
    complexity_weight: float = 0.002, graveyard_penalty_r: float = 0.0,
) -> PropFitnessBreakdown:
    from app.discovery_engine.objective import complexity_penalty, edge_lb_r

    mc_summary = mc_summary or {}
    lb = edge_lb_r(trades) if trades else float("-inf")
    cplx = complexity_penalty(spec, complexity_weight)
    notes: list[str] = []
    pass_p = max(0.0, min(1.0, float(mc_summary.get("per_attempt_pass_probability", mc_summary.get("evaluation_pass_probability", 0.0)) or 0.0) / 100.0))
    pay_p = max(0.0, min(1.0, float(mc_summary.get("per_attempt_payout_probability", mc_summary.get("first_payout_probability", 0.0)) or 0.0) / 100.0))
    robustness = max(0.0, min(1.0, float(robustness_dict["stability_ratio"]))) if robustness_dict else 0.5
    oos = 0.5
    if walk_forward_dict is not None:
        oos = max(0.0, min(1.0, float(walk_forward_dict.get("walk_forward_efficiency", 0.5))))
    dd = max(float(stats.get("max_drawdown_pct", 0.0) or 0.0), 1.0)

    if not math.isfinite(lb):
        n = int(stats.get("total_trades", 0) or 0)
        notes.append(f"Edge not measurable ({n} trades) -- cannot be ranked.")
        out = PropFitnessBreakdown(pass_probability=pass_p, payout_probability=pay_p, robustness=robustness, oos_consistency=oos,
                                   drawdown_pct=dd, base_score=-1e6, notes=notes)
        out.final_score = -1e6
        return out

    base = SCORE_PER_R * (lb - cplx - float(graveyard_penalty_r or 0.0))
    if cplx:
        notes.append(f"Complexity charge -{cplx:.4f}R (prefers the simpler of two equal edges).")
    if graveyard_penalty_r:
        notes.append(f"Graveyard neighbourhood charge -{graveyard_penalty_r:.4f}R (this parameter region keeps dying).")

    n_trades = int(stats.get("total_trades", 0) or 0)
    pen_few = 0.0
    if n_trades < min_trades_target:
        pen_few = (min_trades_target - n_trades) / max(min_trades_target, 1) * 5.0
        notes.append(f"Only {n_trades} trades (target {min_trades_target}+).")
    pen_param = (1.0 - robustness) * 2.0
    if robustness_dict is not None and not robustness_dict.get("is_stable", True):
        notes.append(f"Parameter-neighbourhood stability {robustness:.2f}: nearby values behave differently (fit to noise?).")
    pen_pbo = 0.0
    if pbo is not None:
        p = max(0.0, min(1.0, pbo))
        if p > 0.5:
            pen_pbo = (p - 0.5) * 8.0
            notes.append(f"Probability of Backtest Overfitting {p:.0%} against its peers.")
    pen_deg = 0.0
    if cpcv_degradation is not None and cpcv_degradation > 0:
        pen_deg = min(float(cpcv_degradation), 0.5) * SCORE_PER_R * 0.5       # mean-R degradation, in R, charged at half weight
        notes.append(f"CPCV in-sample vs out-of-sample degradation {cpcv_degradation:.3f}R.")
    if walk_forward_dict is not None and not walk_forward_dict.get("is_stable", True):
        pen_deg += 3.0
        notes.append("Walk-forward folds are not stable on mean R.")
    conc = _trade_concentration(trade_pnls)
    pen_conc = (conc - 0.2) * 6.0 if conc > 0.2 else 0.0
    if pen_conc:
        notes.append(f"Single best trade is {conc:.0%} of gross profit.")

    out = PropFitnessBreakdown(
        pass_probability=pass_p, payout_probability=pay_p, robustness=robustness, oos_consistency=oos, drawdown_pct=dd, base_score=base,
        penalty_too_few_trades=pen_few, penalty_param_sensitivity=pen_param, penalty_high_pbo=pen_pbo,
        penalty_is_oos_degradation=pen_deg, penalty_concentration=pen_conc, notes=notes,
    )
    out.final_score = base - out.total_penalty()
    return out
