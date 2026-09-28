"""
Cross-Instrument Strategy Search -- "pick several instruments, find the ONE
strategy that works across all of them."

Where this fits next to what already exists in the app:

  * Search Lab / Evolution Lab multi-instrument modes run the SAME search
    independently per instrument. You get "the best ES strategy" and "the
    best NQ strategy" side by side (optionally pooled into one leaderboard
    afterward, see app.search.budget_allocator) -- but no candidate was ever
    REQUIRED to work on more than one market.
  * app.optimize.multi_market tunes the parameters of ONE strategy you
    already picked so it scores well across several markets.

This module is the discovery counterpart: it generates candidates from the
strategy families (same generate_search_space() Search Lab uses), scores
EVERY candidate against EVERY selected instrument's own data, aggregates the
per-instrument scores into one robustness number (mean / worst-case /
mean-minus-dispersion -- same aggregation app.optimize.multi_market uses),
and ranks candidates by that. A candidate that dies on any one instrument
(zero trades / errors) is disqualified outright, never averaged away.

Pipeline
--------
  1. Generate candidates from the chosen families (balanced sampling).
  2. Screen: score each on every instrument at cheap Monte Carlo fidelity
     (stops early the moment a candidate is dead on an instrument).
  3. Pick finalists (with a per-family cap so the shortlist is diverse).
  4. Re-score finalists at full Monte Carlo fidelity.
  5. Optional: refine the winner's parameters with the multi-market GA.
  6. Holdout check: the last `holdout_frac` of every instrument's data is
     never used for steps 2-5; finalists are scored on it afterwards.

Honesty note: choosing the best of N candidates on the same data is
selection-biased. The holdout column is the guard against that, and a
winner should still go through Full Pipeline / CPCV before being trusted.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.multi_market import AGGREGATION_METHODS, _aggregate, run_multi_market_search
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import FITNESS_METRICS, RefinementConfig, _evaluate
from app.prop.simulator import PropRules
from app.search.strategy_space import FAMILIES, StrategySpaceError, generate_search_space
from app.strategy.manual import ManualStrategy

ProgressCallback = Callable[[str], None]
CancelCheck = Callable[[], bool]

MIN_HOLDOUT_BARS = 200
MIN_DEV_BARS = 300


class CrossInstrumentCancelled(Exception):
    """Raised when the caller's cancel check fires mid-search."""


@dataclass
class MarketScore:
    market: str
    fitness: float
    trade_count: int
    evaluated: bool = True  # False when an early exit skipped this market


@dataclass
class CrossInstrumentCandidate:
    candidate_id: str
    family: str
    params: dict
    config: dict
    per_market: list  # list[MarketScore]
    mean_fitness: float
    worst_case_fitness: float
    dispersion: float
    robustness_score: float
    holdout_per_market: list | None = None  # list[MarketScore] | None
    holdout_robustness: float | None = None
    holdout_verdict: str = ""
    refined: bool = False

    @property
    def is_viable(self) -> bool:
        return math.isfinite(self.robustness_score)


@dataclass
class CrossInstrumentResult:
    markets: list
    aggregation: str
    fitness_metric: str
    families: list
    candidates_generated: int
    candidates_screened: int
    candidates_viable: int
    total_backtests: int
    holdout_frac: float
    leaderboard: list  # finalists at full fidelity, best first
    best: CrossInstrumentCandidate | None
    elapsed_seconds: float
    warnings: list = field(default_factory=list)


def _score_across_markets(
    config: dict, dfs: dict, risk: RiskConfig, rules: PropRules, mc_cfg: MonteCarloConfig,
    metric: str, aggregation: str, adaptive_risk=None, early_exit: bool = True,
    backtest_counter: list | None = None, error_log: list | None = None,
    risk_by_market: dict | None = None, check_cancel=None,
) -> tuple[list, float, float, float, float]:
    """Backtests ONE candidate config against every market. A fresh strategy
    instance is built per market (same statefulness discipline as
    app.optimize.multi_market.evaluate_multi_market). Returns
    (per_market, mean, worst_case, dispersion, robustness)."""
    per_market: list[MarketScore] = []
    dead = False
    for label, df in dfs.items():
        if check_cancel is not None:
            check_cancel()   # per market, not just per candidate -- a finalist/holdout pass is many long backtests
        market_risk = (risk_by_market or {}).get(label, risk)
        if dead and early_exit:
            per_market.append(MarketScore(label, float("-inf"), 0, evaluated=False))
            continue
        try:
            strategy = ManualStrategy(copy.deepcopy(config))
            fitness, stats, _p, _m, bt_result, _mc, _single = _evaluate(
                df, strategy, market_risk, rules, mc_cfg, metric, keep_full=False, adaptive_risk=adaptive_risk,
            )
            trades = len(bt_result.trades) if bt_result is not None else int((stats or {}).get("total_trades", 0) or 0)
        except Exception as exc:  # noqa: BLE001 -- one bad candidate/market must not kill the whole search
            fitness, trades = float("-inf"), 0
            if error_log is not None and len(error_log) < 5:
                error_log.append(f"{label}: {type(exc).__name__}: {exc}")
        if backtest_counter is not None:
            backtest_counter[0] += 1
        if not math.isfinite(fitness):
            dead = True
        per_market.append(MarketScore(label, fitness, trades))
    mean, worst, dispersion, robustness = _aggregate(per_market, aggregation)
    return per_market, mean, worst, dispersion, robustness


def _split_dev_holdout(dfs: dict, holdout_frac: float, warnings: list) -> tuple[dict, dict]:
    """Chronological split of every market. Returns (dev_dfs, holdout_dfs);
    holdout_dfs is empty when holdout is disabled or any market is too short."""
    if holdout_frac <= 0:
        return dict(dfs), {}
    dev, hold = {}, {}
    for label, df in dfs.items():
        cut = int(len(df) * (1.0 - holdout_frac))
        if cut < MIN_DEV_BARS or (len(df) - cut) < MIN_HOLDOUT_BARS:
            warnings.append(
                f"Holdout disabled: {label} has too few bars ({len(df)}) to reserve "
                f"{holdout_frac:.0%} for holdout. All data was used for the search."
            )
            return dict(dfs), {}
        dev[label] = df.iloc[:cut]
        hold[label] = df.iloc[cut:]
    return dev, hold


def _pick_finalists(scored: list, n: int, max_per_family: int) -> list:
    """Greedy by robustness with a per-family cap; tops up past the cap only
    if there aren't enough distinct families to fill `n`."""
    viable = sorted((c for c in scored if c.is_viable), key=lambda c: c.robustness_score, reverse=True)
    picked, per_family = [], {}
    for c in viable:
        if len(picked) >= n:
            break
        if per_family.get(c.family, 0) < max_per_family:
            picked.append(c)
            per_family[c.family] = per_family.get(c.family, 0) + 1
    for c in viable:
        if len(picked) >= n:
            break
        if c not in picked:
            picked.append(c)
    return picked


def _holdout_verdict(in_sample: CrossInstrumentCandidate, holdout: list, robustness: float) -> str:
    if not holdout:
        return "no holdout"
    if not math.isfinite(robustness):
        return "FAILED holdout (dead on at least one instrument)"
    if any(m.trade_count == 0 for m in holdout):
        return "FAILED holdout (no trades on at least one instrument)"
    if robustness <= 0:
        return "weak holdout (non-positive aggregate score)"
    if math.isfinite(in_sample.robustness_score) and in_sample.robustness_score > 0:
        ratio = robustness / in_sample.robustness_score
        if ratio >= 0.6:
            return f"held up ({ratio:.0%} of in-sample score)"
        return f"degraded ({ratio:.0%} of in-sample score)"
    return "held up"


def run_cross_instrument_search(
    dfs: dict,
    risk: RiskConfig,
    prop_rules: PropRules,
    mc_config: MonteCarloConfig,
    families: list | None = None,
    fitness_metric: str = "eval_pass_probability",
    aggregation: str = "mean_minus_dispersion",
    max_candidates: int = 60,
    screen_mc_sims: int = 150,
    finalists: int = 5,
    max_per_family: int = 2,
    holdout_frac: float = 0.2,
    refine_winner: bool = False,
    refine_config: RefinementConfig | None = None,
    seed: int = 42,
    adaptive_risk=None,
    progress_cb: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    risk_by_market: dict | None = None,
) -> CrossInstrumentResult:
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    def check_cancel() -> None:
        if cancel_check and cancel_check():
            raise CrossInstrumentCancelled("Cross-instrument search cancelled.")

    check_cancel_fn = check_cancel

    if len(dfs) < 2:
        raise RefinementError("Cross-instrument search needs at least 2 instruments selected.")
    if aggregation not in AGGREGATION_METHODS:
        raise RefinementError(f"Unknown aggregation method '{aggregation}'. Supported: {list(AGGREGATION_METHODS)}.")
    if fitness_metric not in FITNESS_METRICS:
        raise RefinementError(f"Unknown fitness metric '{fitness_metric}'.")

    t0 = time.time()
    warnings: list[str] = []
    markets = list(dfs.keys())
    backtests = [0]
    errors: list[str] = []

    selected = [f for f in (families or []) if f]
    unknown = [f for f in selected if f not in FAMILIES]
    if unknown:
        raise RefinementError(f"Unknown strategy famil{'y' if len(unknown) == 1 else 'ies'}: {', '.join(unknown)}.")
    family_arg = "all"
    exclude = None
    if selected:
        exclude = set(FAMILIES) - set(selected)
    try:
        space = generate_search_space(
            mode="family", family=family_arg, max_candidates=max(1, int(max_candidates)), seed=seed,
            exclude_families=exclude,
        )
    except StrategySpaceError as exc:
        raise RefinementError(str(exc)) from exc
    if selected:
        skipped = [f for f in selected if f not in {m["family"] for m in space.meta.values()}]
        if skipped:
            warnings.append(
                "Skipped families that need extra data merged in (pair or economic-calendar): "
                + ", ".join(skipped) + "."
            )
    families_used = sorted({m["family"] for m in space.meta.values()})
    log(f"Generated {len(space.candidates)} candidate(s) from {len(families_used)} famil"
        f"{'y' if len(families_used) == 1 else 'ies'}"
        f"{' (sampled from ' + str(space.total_generated) + ' possible)' if space.sampled else ''}.")

    dev_dfs, holdout_dfs = _split_dev_holdout(dfs, holdout_frac, warnings)
    if holdout_dfs:
        log(f"Reserved the last {holdout_frac:.0%} of every instrument as an untouched holdout.")

    screen_cfg = MonteCarloConfig(
        n_simulations=int(screen_mc_sims), random_seed=mc_config.random_seed,
        reset_on_breach=mc_config.reset_on_breach,
    )

    # -- Stage 1: screen every candidate on every instrument --------------
    log(f"Screening across {len(markets)} instrument(s): {', '.join(markets)}.")
    scored: list[CrossInstrumentCandidate] = []
    total = len(space.candidates)
    for i, (cid, spec) in enumerate(space.candidates.items(), start=1):
        check_cancel()
        config = spec["config"]
        per_market, mean, worst, disp, robust = _score_across_markets(
            config, dev_dfs, risk, prop_rules, screen_cfg, fitness_metric, aggregation,
            adaptive_risk=adaptive_risk, early_exit=True, backtest_counter=backtests, error_log=errors,
            risk_by_market=risk_by_market, check_cancel=check_cancel_fn,
        )
        meta = space.meta.get(cid, {})
        scored.append(CrossInstrumentCandidate(
            candidate_id=cid, family=meta.get("family", "unknown"), params=meta.get("params", {}),
            config=config, per_market=per_market, mean_fitness=mean, worst_case_fitness=worst,
            dispersion=disp, robustness_score=robust,
        ))
        if i % 10 == 0 or i == total:
            viable = sum(1 for c in scored if c.is_viable)
            best = max((c.robustness_score for c in scored if c.is_viable), default=float("-inf"))
            log(f"Screened {i}/{total} candidates -- {viable} traded on every instrument"
                + (f", best robustness so far {best:.3f}." if math.isfinite(best) else "."))

    viable_count = sum(1 for c in scored if c.is_viable)
    if errors:
        warnings.append("Some backtests raised errors and were scored as dead: " + " | ".join(errors))
    if viable_count == 0:
        warnings.append(
            "No candidate produced trades on EVERY selected instrument. Try more families, a larger candidate "
            "budget, or check that pip size / contract settings suit all the selected instruments."
        )
        return CrossInstrumentResult(
            markets=markets, aggregation=aggregation, fitness_metric=fitness_metric, families=families_used,
            candidates_generated=total, candidates_screened=total, candidates_viable=0,
            total_backtests=backtests[0], holdout_frac=holdout_frac if holdout_dfs else 0.0,
            leaderboard=[], best=None, elapsed_seconds=time.time() - t0, warnings=warnings,
        )

    # -- Stage 2: finalists at full Monte Carlo fidelity -------------------
    shortlist = _pick_finalists(scored, max(1, int(finalists)), max(1, int(max_per_family)))
    log(f"Re-scoring {len(shortlist)} finalist(s) at full Monte Carlo fidelity...")
    final: list[CrossInstrumentCandidate] = []
    for c in shortlist:
        check_cancel()
        per_market, mean, worst, disp, robust = _score_across_markets(
            c.config, dev_dfs, risk, prop_rules, mc_config, fitness_metric, aggregation,
            adaptive_risk=adaptive_risk, early_exit=False, backtest_counter=backtests, error_log=errors,
            risk_by_market=risk_by_market, check_cancel=check_cancel_fn,
        )
        final.append(CrossInstrumentCandidate(
            candidate_id=c.candidate_id, family=c.family, params=c.params, config=c.config,
            per_market=per_market, mean_fitness=mean, worst_case_fitness=worst, dispersion=disp,
            robustness_score=robust,
        ))
    final.sort(key=lambda c: c.robustness_score, reverse=True)

    # -- Stage 3 (optional): tune the winner's parameters across markets ---
    per_market_risk_differs = bool(risk_by_market) and len({
        (r.pip_size, r.contract_size, r.commission_per_trade) for r in risk_by_market.values()
    }) > 1
    if refine_winner and per_market_risk_differs:
        warnings.append(
            "Skipped the optional refinement step: it scores every market with ONE shared risk setup, but these "
            "instruments each need their own pip size / contract size. Re-run the winner through Full Pipeline "
            "per instrument to tune it."
        )
    elif refine_winner and final and final[0].is_viable:
        check_cancel()
        winner = final[0]
        log(f"Refining the top candidate ({winner.family}) with the multi-market optimizer...")
        try:
            refined = run_multi_market_search(
                dev_dfs, ManualStrategy(copy.deepcopy(winner.config)), risk, prop_rules, mc_config,
                refine_config or RefinementConfig(
                    enabled=True, fitness_metric=fitness_metric, population_size=10, generations=4,
                    elite_count=2, search_monte_carlo_sims=int(screen_mc_sims), random_seed=seed,
                ),
                aggregation=aggregation, progress_cb=progress_cb,
            )
            backtests[0] += refined.total_evaluations * len(markets)
            if refined.best.config and refined.best.robustness_score > winner.robustness_score:
                rc = refined.best
                final.append(CrossInstrumentCandidate(
                    candidate_id=f"{winner.candidate_id}-refined", family=f"{winner.family} (refined)",
                    params=winner.params, config=rc.config,
                    per_market=[MarketScore(p.market, p.fitness, p.trade_count) for p in rc.per_market],
                    mean_fitness=rc.mean_fitness, worst_case_fitness=rc.worst_case_fitness,
                    dispersion=rc.dispersion, robustness_score=rc.robustness_score, refined=True,
                ))
                final.sort(key=lambda c: c.robustness_score, reverse=True)
                log("Refinement improved the aggregate score.")
            else:
                log("Refinement did not beat the unrefined candidate; keeping the original.")
        except CrossInstrumentCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 -- refinement is a bonus, never fatal
            warnings.append(f"Refinement step failed and was skipped: {exc}")

    # -- Stage 4: untouched holdout ---------------------------------------
    if holdout_dfs:
        log("Scoring finalists on the untouched holdout data...")
        for c in final:
            check_cancel()
            per_market, _mean, _worst, _disp, robust = _score_across_markets(
                c.config, holdout_dfs, risk, prop_rules, mc_config, fitness_metric, aggregation,
                adaptive_risk=adaptive_risk, early_exit=False, backtest_counter=backtests, error_log=errors,
                risk_by_market=risk_by_market, check_cancel=check_cancel_fn,
            )
            c.holdout_per_market = per_market
            c.holdout_robustness = robust
            c.holdout_verdict = _holdout_verdict(c, per_market, robust)

    warnings.append(
        "Best-of-N selection is optimistic: treat the winner as a lead, not a verdict. "
        "Send it through Full Pipeline / CPCV before trusting it."
    )
    best = final[0] if final and final[0].is_viable else None
    elapsed = time.time() - t0
    log(f"Cross-instrument search complete in {elapsed:.1f}s ({backtests[0]} backtests).")
    return CrossInstrumentResult(
        markets=markets, aggregation=aggregation, fitness_metric=fitness_metric, families=families_used,
        candidates_generated=total, candidates_screened=total, candidates_viable=viable_count,
        total_backtests=backtests[0], holdout_frac=holdout_frac if holdout_dfs else 0.0,
        leaderboard=final, best=best, elapsed_seconds=elapsed, warnings=warnings,
    )
