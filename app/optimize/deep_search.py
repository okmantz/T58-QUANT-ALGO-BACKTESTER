"""Deep search (v9.8 item 12): thousands of strategy variants across
parameter settings x timeframes x market datasets, ranked for prop-firm
use by ONE explicit formula -- PropFit -- then verified before anything
is called a winner.

Two stages, on purpose:

* Stage 1 (screen): one backtest per variant; PropFit is computed from
  that backtest's trade list through the app's per-attempt Monte Carlo
  prop simulation (cheap, no re-backtest). With large budgets a quick
  60%-slice screen prunes first; the ranking that matters is always on
  full data.
* Stage 2 (verify): the top finalists are re-scored with
  walk-forward/CPCV, Monte Carlo seed dispersion and a true 2x cost
  stress. A variant whose edge evaporates scores 0 verified -- the
  table shows the evaporation instead of crowning a curve-fit.

PropFit is a RANKING score only. It never overrides the app's 70%
per-attempt READY gate; the results page says which bar each row
actually clears.

The scoring/decision core is pure functions (unit-tested); the caller
(web layer) injects `evaluate` / `verify` callables that run the real
backtests, so the two-stage behavior is testable without an engine.
"""
from __future__ import annotations

import random

# --- PropFit: the weights live here and nowhere else -------------------
PROPFIT_WEIGHTS = {
    "eval_pass": 45.0,      # x per-attempt eval-pass probability (0..1)
    "payout": 30.0,         # x first-payout probability (0..1)
    "profit_factor": 15.0,  # x min(PF, 3) / 3
    "sufficiency": 10.0,    # x min(trades, 100) / 100
}
PROPFIT_PF_CAP = 3.0
SUFFICIENCY_TRADES = 100
DD_NEAR_LIMIT_FRAC = 0.90        # max DD within 10% of the preset limit
PENALTY_DRAWDOWN_NEAR_LIMIT = 10.0
PENALTY_COST_STRESS = 30.0
FINALIST_COUNT = 20
PRUNE_ABOVE = 400                # budgets above this get a slice screen
SLICE_FRAC = 0.60


def formula_text() -> str:
    w = PROPFIT_WEIGHTS
    return (
        f"PropFit = {w['eval_pass']:g} x eval-pass probability (per attempt) "
        f"+ {w['payout']:g} x first-payout probability "
        f"+ {w['profit_factor']:g} x min(profit factor, {PROPFIT_PF_CAP:g}) / {PROPFIT_PF_CAP:g} "
        f"+ {w['sufficiency']:g} x trade sufficiency (1.0 at >={SUFFICIENCY_TRADES} trades, scaled below) "
        f"- {PENALTY_DRAWDOWN_NEAR_LIMIT:g} if max drawdown is within 10% of the prop limit "
        f"- {PENALTY_COST_STRESS:g} if the 2x-cost-stressed attempt simulation is not profitable. "
        "Non-positive expectancy scores 0. PropFit ranks variants only; it never "
        "overrides the 70% per-attempt READY gate."
    )


def propfit_from_metrics(*, eval_pass: float, payout: float, profit_factor: float,
                         n_trades: int, expectancy: float, max_dd_pct: float,
                         dd_limit_pct: float, stressed_net: float):
    """(score, components). Pure: a known trade list's metrics must give
    a known score (unit-tested)."""
    w = PROPFIT_WEIGHTS
    suff = min(float(n_trades), float(SUFFICIENCY_TRADES)) / float(SUFFICIENCY_TRADES)
    comps = {
        "eval_pass": float(eval_pass), "payout": float(payout),
        "profit_factor": float(profit_factor), "n_trades": int(n_trades),
        "expectancy": float(expectancy), "max_dd_pct": float(max_dd_pct),
        "dd_limit_pct": float(dd_limit_pct), "stressed_net": float(stressed_net),
        "sufficiency": suff, "penalties": 0.0, "floored": False,
    }
    if n_trades <= 0 or expectancy <= 0:
        comps["floored"] = True
        return 0.0, comps
    score = (
        w["eval_pass"] * float(eval_pass)
        + w["payout"] * float(payout)
        + w["profit_factor"] * min(float(profit_factor), PROPFIT_PF_CAP) / PROPFIT_PF_CAP
        + w["sufficiency"] * suff
    )
    if dd_limit_pct and float(max_dd_pct) >= DD_NEAR_LIMIT_FRAC * float(dd_limit_pct):
        score -= PENALTY_DRAWDOWN_NEAR_LIMIT
        comps["penalties"] += PENALTY_DRAWDOWN_NEAR_LIMIT
    if stressed_net <= 0:
        score -= PENALTY_COST_STRESS
        comps["penalties"] += PENALTY_COST_STRESS
    return max(0.0, score), comps


def verified_score(screen_score: float, *, cpcv_oos_pf, stress_net: float,
                   seed_std_pts: float) -> float:
    """Stage-2 re-score. An edge that evaporates out-of-sample is worth
    exactly 0, however pretty the screen looked."""
    if cpcv_oos_pf is None or float(cpcv_oos_pf) < 1.0:
        return 0.0
    score = float(screen_score)
    if stress_net <= 0:
        score -= PENALTY_COST_STRESS
    score -= min(20.0, float(seed_std_pts))
    return max(0.0, score)


def is_above_water(row: dict) -> bool:
    """Above water means: verified score positive, the edge survives
    CPCV (OOS PF >= 1), the 2x cost stress stays profitable, AND the
    verified per-attempt eval-pass probability is actually above zero
    -- a "winner that passes" cannot be a variant no simulated attempt
    ever passes with. (The 70% READY gate stays separate and higher.)"""
    return (
        row.get("verified") is not None
        and float(row["verified"]) > 0
        and row.get("cpcv_oos_pf") is not None
        and float(row["cpcv_oos_pf"]) >= 1.0
        and float(row.get("stress_net", 0.0)) > 0
        and float(row.get("eval_pass_verified") or 0.0) > 0
    )


def decide(finalists: list[dict], n_variants: int):
    """(ranked, winner_or_None, message). Ranked by verified score;
    the winner must be above water, else the honest no-winner text."""
    ranked = sorted(finalists, key=lambda r: float(r.get("verified") or 0.0), reverse=True)
    if ranked and is_above_water(ranked[0]):
        w = ranked[0]
        return ranked, w, (
            f"Winner: {w.get('label', 'variant')} — verified PropFit "
            f"{float(w['verified']):.1f} (screen {float(w.get('propfit') or 0):.1f})."
        )
    return ranked, None, (
        f"No robust winner found in {n_variants} variants — nothing survived "
        "walk-forward/CPCV verification, seed dispersion and the 2x cost stress. "
        "The least-bad rows are shown below, uncrowned."
    )


def _sample_gene(rng: random.Random, gene) -> float:
    v = rng.uniform(float(gene.lo), float(gene.hi))
    if gene.is_int:
        v = float(int(round(v)))
    return min(max(v, float(gene.lo)), float(gene.hi))


def sample_genomes(genes, count: int, seed: int = 42) -> list[list[float]]:
    """`count` genome vectors within the genes' bounds; the base genome
    (the strategy as-is) is always first so the original is ranked too."""
    if not genes:
        return [[]]
    rng = random.Random(seed)
    out: list[list[float]] = [[float(g.base_value) for g in genes]]
    while len(out) < max(1, int(count)):
        out.append([_sample_gene(rng, g) for g in genes])
    return out


def run_search(variants, evaluate, verify=None, *, evaluate_many=None,
               finalist_count: int = FINALIST_COUNT,
               prune_above: int = PRUNE_ABOVE, log=lambda msg: None) -> dict:
    """Two-stage search over `variants` (opaque dicts).

    evaluate(variant, phase) -> row dict | None; phase is "slice" for
    the quick screen, "full" for the real ranking (full data only).
    evaluate_many(variants, phase) -> [row|None] may replace evaluate
    for batch/parallel evaluation. Rows must carry: variant, label,
    dataset, timeframe, n_trades, eval_pass, payout, profit_factor,
    propfit. verify(row) -> {"cpcv_oos_pf", "stress_net", "seed_std_pts"}.
    """
    def _eval_all(vs, phase):
        if evaluate_many is not None:
            return [r for r in evaluate_many(list(vs), phase) if r]
        return [r for r in (evaluate(v, phase) for v in vs) if r]

    n = len(variants)
    rows: list[dict] = []
    if n > prune_above:
        log(f"Stage 1 screen: {n} variants on a {int(SLICE_FRAC * 100)}% data slice...")
        sliced = _eval_all(variants, "slice")
        sliced.sort(key=lambda r: float(r.get("propfit") or 0.0), reverse=True)
        keep = sliced[: max(60, n // 4)]
        log(f"Screen kept {len(keep)} of {n}; scoring survivors on full data...")
        rows = _eval_all([r["variant"] for r in keep], "full")
    else:
        rows = _eval_all(variants, "full")
    rows.sort(key=lambda r: float(r.get("propfit") or 0.0), reverse=True)
    finalists = [dict(r) for r in rows[:finalist_count]]
    if verify is not None:
        log(f"Stage 2: verifying the top {len(finalists)} finalists "
            "(walk-forward/CPCV + seed dispersion + 2x cost stress)...")
        for r in finalists:
            try:
                v = verify(r)
            except Exception as exc:  # noqa: BLE001 -- one finalist must not sink the rest
                v = {"cpcv_oos_pf": None, "stress_net": 0.0, "seed_std_pts": 0.0,
                     "verify_error": str(exc)}
            r.update(v)
            r["verified"] = verified_score(float(r.get("propfit") or 0.0), **{
                k: v.get(k) for k in ("cpcv_oos_pf", "stress_net", "seed_std_pts")})
    ranked, winner, message = decide(finalists, n)
    log(message)
    return {"n_variants": n, "rows": ranked, "winner": winner, "message": message,
            "n_screened_full": len(rows)}
