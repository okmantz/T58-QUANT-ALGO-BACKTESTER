"""Shared validation-check dispatcher.

This used to live inside app/web/server.py, so only the web app could run
the Validate page's checks. It is now Flask-free and shared: the web
Validate job AND the desktop app's Validate tab call the same function, so
both run identical checks with identical settings and produce identical
results (and feed the same plain-language report in
app.validation.simple_report).

``form`` is any object with a ``.get(key, default)`` method (a Flask
MultiDict, or a plain dict on the desktop side).
"""
from __future__ import annotations

import random

from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.optimize.code_parameter_space import apply_code_genome, discover_code_genes
from app.optimize.parameter_space import apply_genome, extract_genome
from app.optimize.refinement import RefinementConfig
from app.optimize.walkforward_ga import WalkforwardGACancelled, run_walkforward_aware_refinement
from app.search.strategy_space import spec_from_strategy
from app.validation.cpcv import compute_pbo, run_cpcv
from app.validation.parameter_robustness import compute_parameter_robustness
from app.validation.regime_matrix import run_regime_matrix
from app.validation.sensitivity import compute_1d_sensitivity
from app.validation.walk_forward_opt import run_walk_forward_optimization


def perturbed_variant_specs(strategy, n_variants: int, seed: int) -> list[dict]:
    """N random perturbations of `strategy`'s own tunable numeric
    parameters, +/-30% of each gene's own range around its base value --
    identical perturbation logic to app.main.py's run_pbo_cli, generalized
    from manual-only to any of the 4 source types via the same gene
    discovery/apply machinery Iterative Refinement's GA already uses."""
    if n_variants <= 0:
        return []
    rng = random.Random(seed)
    out = []
    if strategy.source_type == "manual":
        genes = extract_genome(strategy.config)
        if not genes:
            return []
        for _ in range(n_variants):
            genome = [max(min(g.base_value + rng.uniform(-0.3, 0.3) * (g.hi - g.lo), g.hi), g.lo) for g in genes]
            out.append({"source_type": "manual", "config": apply_genome(strategy.config, genes, genome)})
    else:
        genes = discover_code_genes(strategy)
        if not genes:
            return []
        base_spec = spec_from_strategy(strategy)
        for _ in range(n_variants):
            genome = [max(min(g.base_value + rng.uniform(-0.3, 0.3) * (g.hi - g.lo), g.hi), g.lo) for g in genes]
            code_text = apply_code_genome(base_spec["code_text"], genes, genome)
            out.append({"source_type": strategy.source_type, "code_text": code_text, "code_extension": base_spec["code_extension"]})
    return out


def dispatch_validation_checks(df, strategy, risk, rules, checks, form, baseline_trades=None,
                                on_start=None, on_done=None, cancel_event=None, log=None,
                                on_fraction=None):
    """Run the ticked checks against ONE hardened baseline. Every check is
    isolated: a failure is reported per-check, never fatal to the others.
    Returns a list of {check, ok, summary, error} with honest numbers.

    Optional hooks (all best-effort, used by the Validate job for its live
    percent bar and to save each result onto the strategy the moment it
    lands): on_start(check), on_done(check, result_dict),
    on_fraction(fraction_0_to_1) for checks that can report their own
    progress, log(message), and cancel_event (stops before the next check).
    """
    out: list[dict] = []
    mc_sims = min(int(form.get("mc_sims", 1000) or 1000), 10_000)
    metric = form.get("validation_metric", "profit_factor") or "profit_factor"
    _log = log or (lambda _m: None)
    for check in checks:
        if cancel_event is not None and cancel_event.is_set():
            break
        if on_start is not None:
            try:
                on_start(check)
            except Exception:  # noqa: BLE001 -- a progress hook must never break a check
                pass
        try:
            if check == "monte_carlo":
                if not baseline_trades:
                    raise ValueError("Baseline backtest produced no trades to resample.")
                mc = run_monte_carlo(baseline_trades, rules, MonteCarloConfig(n_simulations=mc_sims))
                summary = {
                    "evaluation_pass_probability": mc.headline_evaluation_pass_probability,
                    "first_payout_probability": mc.headline_first_payout_probability,
                    "any_attempt_evaluation_pass_probability": mc.evaluation_pass_probability,
                    "any_attempt_first_payout_probability": mc.first_payout_probability,
                    "risk_of_ruin_pct": mc.risk_of_ruin_pct,
                    "expected_payout": mc.expected_payout,
                    "n_simulations": mc.n_simulations,
                }
            elif check == "cpcv":
                res = run_cpcv(
                    df, lambda s=strategy: s, risk,
                    n_groups=int(form.get("cpcv_groups", 4) or 4),
                    n_test_groups=int(form.get("cpcv_test_groups", 2) or 2),
                    metric=metric, prop_rules=rules,
                    mc_cfg=MonteCarloConfig(n_simulations=min(mc_sims, 500)),
                    max_paths=int(form.get("cpcv_max_paths", 15) or 15),
                )
                summary = {
                    "n_paths": res.n_paths, "metric": res.metric,
                    "mean_oos_metric": res.mean_oos_metric, "median_oos_metric": res.median_oos_metric,
                    "mean_is_metric": res.mean_is_metric, "is_robust": bool(res.is_robust),
                }
            elif check == "pbo":
                specs = [spec_from_strategy(strategy)] + perturbed_variant_specs(
                    strategy, int(form.get("pbo_variants", 4) or 4), 42)
                if len(specs) < 2:
                    raise ValueError("PBO needs at least 2 candidates; this strategy has no tunable parameters to perturb.")
                res = compute_pbo(
                    df, specs, risk,
                    n_groups=int(form.get("cpcv_groups", 4) or 4),
                    n_test_groups=int(form.get("cpcv_test_groups", 2) or 2),
                    metric=metric, prop_rules=rules, max_paths=int(form.get("cpcv_max_paths", 15) or 15),
                )
                summary = {"pbo": res.pbo, "n_candidates": res.n_candidates, "n_paths": res.n_paths, "note": res.note}
            elif check == "walk_forward":
                res = run_walk_forward_optimization(
                    df, strategy, risk, rules,
                    MonteCarloConfig(n_simulations=min(mc_sims, 500)),
                    n_folds=int(form.get("wf_folds", 3) or 3),
                    window_mode=form.get("wf_window_mode", "rolling") or "rolling",
                    progress_cb=_log,
                )
                summary = res.to_summary_dict() if hasattr(res, "to_summary_dict") else {
                    "n_folds_completed": len(getattr(res, "folds", []) or []),
                    "out_of_sample_efficiency": getattr(res, "out_of_sample_efficiency", None),
                }
            elif check == "wfga":
                ga_res = run_walkforward_aware_refinement(
                    df, strategy, risk, rules,
                    MonteCarloConfig(n_simulations=min(mc_sims, 500)),
                    RefinementConfig(
                        population_size=int(form.get("wfga_population", 8) or 8),
                        generations=int(form.get("wfga_generations", 4) or 4),
                        search_monte_carlo_sims=200,
                        fitness_metric=form.get("fitness_metric", "eval_pass_probability") or "eval_pass_probability",
                    ),
                    n_folds=int(form.get("wf_folds", 3) or 3),
                    window_mode=form.get("wf_window_mode", "rolling") or "rolling",
                    progress_cb=_log, cancel_event=cancel_event,
                    batch_progress_cb=(lambda done, total: on_fraction(done / total) if (on_fraction and total) else None),
                )
                summary = {
                    "overfitting_gap": ga_res.overfitting_gap,
                    "in_sample_fitness": ga_res.best.in_sample_fitness,
                    "oos_fitness": ga_res.best.fitness,
                    "oos_trade_count": ga_res.best.oos_trade_count,
                    "n_folds": ga_res.n_folds,
                }
            elif check == "sensitivity":
                res_list = compute_1d_sensitivity(
                    df, strategy, risk, rules, MonteCarloConfig(n_simulations=min(mc_sims, 500)),
                    metric=metric, pct_range=float(form.get("pct_range", 0.5) or 0.5),
                    n_steps=int(form.get("n_steps", 5) or 5), max_params=int(form.get("max_params", 4) or 4),
                    progress_cb=_log,
                )
                summary = {
                    "n_parameters": len(res_list),
                    "n_cliffs": sum(1 for r in res_list if r.cliff_detected),
                    "parameters": [
                        {"label": r.gene_label, "base_value": r.base_value, "base_metric": r.base_metric,
                         "cliff_detected": bool(r.cliff_detected), "max_pct_drop": r.max_pct_drop_between_adjacent_steps}
                        for r in res_list
                    ],
                }
            elif check == "robustness":
                res = compute_parameter_robustness(
                    df, strategy, risk, rules, MonteCarloConfig(n_simulations=min(mc_sims, 500)),
                    metric=form.get("robustness_metric", "eval_pass_probability") or "eval_pass_probability",
                    pct_range=float(form.get("pct_range", 0.5) or 0.5),
                    n_steps_1d=int(form.get("n_steps", 5) or 5), max_params=int(form.get("max_params", 4) or 4),
                    n_heatmap_pairs=1,
                )
                summary = {
                    "parameter_robustness_score": res.parameter_robustness_score,
                    "n_parameters_checked": res.n_parameters_checked,
                    "n_cliffs_detected": res.n_cliffs_detected,
                }
            elif check == "regime":
                res = run_regime_matrix(
                    df, strategy, risk,
                    dimensions=((form.get("regime_dim_a", "volatility") or "volatility"),
                                (form.get("regime_dim_b", "environment") or "environment")),
                )
                if res is None:
                    raise ValueError("Not enough bars to classify regimes reliably.")
                summary = {
                    "n_cells": len(res.cells),
                    "n_disable_recommended": len(res.disable_regimes()),
                    "notes": list(res.notes or []),
                }
            else:
                continue
            result = {"check": check, "ok": True, "summary": summary, "error": None}
        except WalkforwardGACancelled:
            break
        except Exception as exc:  # noqa: BLE001 -- per-check isolation
            result = {"check": check, "ok": False, "summary": None, "error": str(exc)}
        out.append(result)
        if on_done is not None:
            try:
                on_done(check, result)
            except Exception:  # noqa: BLE001 -- saving/reporting must never sink the rest
                pass
    return out
