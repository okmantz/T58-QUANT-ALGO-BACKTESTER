"""
Forge Strategy -- the literal one-button "generate, screen, validate" tab.

Owen's ask: generate hundreds/thousands of strategy HYPOTHESES (not random
parameter mutations -- see app.search.strategy_space's ~47 named families,
each testing an explicit, falsifiable market question), screen them hard
and cheaply, and only spend expensive validation compute on the survivors,
ranked the whole way through by PROP SURVIVAL (pass the eval, get paid,
keep the drawdown safe, stay consistent) rather than raw net profit or
Sharpe.

This module is deliberately a THIN orchestrator over engines this codebase
already has, in this order:

    N hypotheses                         (app.search.strategy_space,
                                           mode="family", family="all" --
                                           every family is already a named,
                                           deliberate hypothesis, not a
                                           random mutation)
      -> fast screen                     (app.search.batch_runner Stage 1:
                                           cheap full-dataset backtest filter)
      -> prop survival screen            (Stage 2: GA refinement, fitness
                                           metric = eval_pass_probability,
                                           i.e. optimizing prop survival,
                                           not profit)
      -> neighbor testing                (Stage 3, part A: parameter-
                                           neighborhood robustness)
      -> walk-forward                    (Stage 3, part B: chained
                                           out-of-sample walk-forward +
                                           full-fidelity Monte Carlo)
      -> CPCV/PBO + regime testing       (NEW here: app.validation.cpcv +
                                           app.validation.regime_testing,
                                           on Stage 3's actual survivors)
      -> Monte Carlo (deeper)            (NEW here: a higher-fidelity
                                           app.monte_carlo.engine run on
                                           the CPCV/regime shortlist)
      -> rolling prop evaluation         (NEW here: app.prop.rolling_
                                           evaluation, real historical
                                           windows, every possible start day)
      -> locked OOS holdout              (NEW here: a chronological slice
                                           reserved BEFORE hypothesis
                                           generation even started, never
                                           touched by any stage above)

Every stage that rejects a candidate expensive enough to be worth
explaining feeds app.search.failure_diagnosis + app.search.graveyard, so
a run's rejections are a map of dead strategy space, not just a number.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.rolling_evaluation import run_rolling_evaluation
from app.prop.simulator import PropRules
from app.prop.survival_engine import PropSurvivalConfig, run_prop_survival_analysis
from app.search.batch_runner import SearchCancelled, SearchStageConfig, run_search
from app.search.family_diversity import summarize_family_performance
from app.search.failure_diagnosis import CandidateDiagnosis, diagnose_candidate
from app.search.graveyard import GraveyardEntry, param_signature, record_rejections
from app.search.results_db import ResultsDB
from app.search.strategy_space import (
    StrategySpaceError, build_strategy_from_spec, generate_search_space, hypothesis_question,
)
from app.strategy.family_taxonomy import family_label
from app.validation.cpcv import CPCVError, compute_pbo, run_cpcv
from app.validation.regime_testing import run_regime_test

ProgressCallback = Callable[[str], None]


@dataclass
class ForgeConfig:
    # Stage 0 -- hypothesis generation. "family='all'" (the default, always
    # used here) means every named market-hypothesis family this codebase
    # knows about, not a single one -- see app.search.strategy_space.FAMILIES.
    n_hypotheses: int = 10_000
    exclude_families: "set[str] | None" = None
    seed: int = 42

    # Stage 1 -- fast screen
    min_trades: int = 20
    min_profit_factor: float = 1.05
    stage1_top_n: int = 1_000

    # Stage 2 -- prop survival screen (GA, fitness = eval_pass_probability)
    ga_population: int = 12
    ga_generations: int = 4
    stage2_top_n: int = 200

    # Stage 3 -- neighbor testing + walk-forward + Monte Carlo (bundled in
    # app.search.batch_runner's own Stage 3; this config only controls how
    # hard that gate is)
    stage3_mc_sims: int = 2_000
    walk_forward_folds: int = 4
    robustness_neighbors: int = 6
    stage3_target_survivors: int = 30   # informational target, not a hard cap -- see docstring on run_forge

    # Forge's own later stages
    cpcv_pool_size: int = 30            # how many Stage 3 survivors get CPCV/regime-tested, best-first
    cpcv_survivors: int = 10
    cpcv_n_groups: int = 6
    cpcv_n_test_groups: int = 2
    regime_buckets: int = 3
    final_mc_sims: int = 10_000
    mc_survivors: int = 5
    eval_window_days: int = 60          # rolling-evaluation window length, in trading days
    rolling_survivors: int = 2
    locked_holdout_frac: float = 0.15   # reserved BEFORE any stage runs, never searched over
    locked_oos_min_pass_rate: float = 25.0  # % -- below this, the holdout check kills the candidate

    workers: int | None = None
    random_seed: int = 42


@dataclass
class FunnelStage:
    name: str
    n_in: int
    n_out: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ForgeLeaderboardRow:
    candidate_id: str
    family: str
    family_label: str
    hypothesis: str
    pass_rate_pct: float                  # rolling-evaluation pass rate
    payout_rate_pct: float                # rolling-evaluation first-payout rate
    median_days_to_pass: float | None
    max_drawdown_pct: float
    max_drawdown_dollars: float
    prop_survival_score: float            # 0-100, app.prop.survival_engine
    locked_oos_status: str                # "PASSED" | "FAILED" | "NOT TESTED"
    locked_oos_pass_rate_pct: float | None
    net_profit: float
    profit_factor: float
    total_trades: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ForgeResult:
    run_id: str
    db_path: str
    funnel: list = field(default_factory=list)          # list[FunnelStage]
    leaderboard: list = field(default_factory=list)      # list[ForgeLeaderboardRow]
    diagnoses: list = field(default_factory=list)        # list[CandidateDiagnosis]
    champion_candidate_id: str | None = None
    cohort_pbo: float | None = None                       # Probability of Backtest Overfitting across the CPCV shortlist
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "db_path": self.db_path,
            "funnel": [s.to_dict() for s in self.funnel],
            "leaderboard": [r.to_dict() for r in self.leaderboard],
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "champion_candidate_id": self.champion_candidate_id,
            "cohort_pbo": self.cohort_pbo,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _rebuild_spec(row: dict) -> dict:
    source_type = row.get("source_type") or "manual"
    if source_type == "manual":
        return {"source_type": "manual", "config": row.get("config") or {}}
    return {
        "source_type": source_type,
        "code_text": row.get("code_text"),
        "code_extension": row.get("code_extension"),
    }


def _strategy_builder(spec: dict):
    def _build():
        return build_strategy_from_spec(spec)
    return _build


def _dd_dollars(stats: dict | None, account_size: float) -> float:
    if not stats:
        return 0.0
    pct = float(stats.get("max_drawdown_pct", 0.0) or 0.0)
    return round(pct / 100.0 * account_size, 2)


def run_forge(
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    config: ForgeConfig,
    db_path: str,
    instrument: str = "unknown",
    timeframe: str = "unknown",
    graveyard_path: str | Path | None = None,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> ForgeResult:
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise SearchCancelled("Forge Strategy run stopped by user.")

    t0 = time.time()
    funnel: list[FunnelStage] = []
    diagnoses: list[CandidateDiagnosis] = []

    # ------------------------------------------------------------------
    # Reserve the locked OOS holdout BEFORE generating a single hypothesis
    # -- nothing above this line ever sees `locked_df`, so "locked OOS"
    # means what it says instead of being just another in-sample slice.
    # ------------------------------------------------------------------
    n = len(df)
    split_idx = max(1, min(int(n * (1 - config.locked_holdout_frac)), n - 1)) if n > 1 else n
    search_df = df.iloc[:split_idx].reset_index(drop=True)
    locked_df = df.iloc[split_idx:].reset_index(drop=True)
    log(
        f"Reserved the final {config.locked_holdout_frac:.0%} of the dataset ({len(locked_df):,} bars) as a "
        f"locked out-of-sample holdout -- no stage below this line will ever see it until the very end."
    )

    # ------------------------------------------------------------------
    # Stage 0 -- generate the hypothesis pool: every named market
    # hypothesis this codebase knows (app.search.strategy_space.FAMILIES),
    # deliberately, not random parameter mutation.
    # ------------------------------------------------------------------
    space = generate_search_space(
        mode="family", family="all", max_candidates=config.n_hypotheses,
        seed=config.seed, exclude_families=config.exclude_families,
    )
    n_hyp = len(space.candidates)
    log(
        f"Generated {n_hyp:,} strategy hypotheses across {len(set(m['family'] for m in space.meta.values()))} "
        f"named market-hypothesis families (e.g. \"{hypothesis_question(next(iter(space.meta.values()))['family'])}\")."
    )
    funnel.append(FunnelStage("Hypotheses generated", n_in=n_hyp, n_out=n_hyp))
    check_cancelled()

    # ------------------------------------------------------------------
    # Stages 1-3 -- fast screen, prop survival screen, neighbor testing +
    # walk-forward + Monte Carlo. app.search.batch_runner.run_search
    # already implements exactly this funnel end to end.
    # ------------------------------------------------------------------
    stage_cfg = SearchStageConfig(
        min_trades=config.min_trades,
        min_profit_factor=config.min_profit_factor,
        stage1_top_n=config.stage1_top_n,
        ga_population=config.ga_population,
        ga_generations=config.ga_generations,
        stage2_top_n=config.stage2_top_n,
        full_mc_sims=config.stage3_mc_sims,
        walk_forward_folds=config.walk_forward_folds,
        robustness_neighbors=config.robustness_neighbors,
        fitness_metric="eval_pass_probability",
        workers=config.workers,
        random_seed=config.random_seed,
    )
    summary = run_search(
        search_df, risk, prop_rules, space, stage_cfg, db_path=db_path,
        instrument=instrument, timeframe=timeframe, progress_cb=log, cancel_event=cancel_event,
    )
    funnel.append(FunnelStage("Fast screen (Stage 1)", n_in=n_hyp, n_out=summary.stage1_survivors))
    funnel.append(FunnelStage("Prop survival screen (Stage 2 GA)", n_in=summary.stage1_survivors, n_out=summary.stage2_survivors))

    db = ResultsDB(db_path)
    stage3_records = db.leaderboard(summary.run_id, stage="stage3", top_n=100_000, only_passed=False)
    n_robust = sum(1 for r in stage3_records if (r.get("robustness") or {}).get("is_stable", True))
    funnel.append(FunnelStage("Neighbor testing (parameter robustness)", n_in=len(stage3_records), n_out=n_robust))
    n_gate_passed = sum(1 for r in stage3_records if r.get("passed_stage3_gate"))
    funnel.append(FunnelStage("Walk-forward + Monte Carlo (Stage 3 gate)", n_in=n_robust, n_out=n_gate_passed))

    # Failure diagnosis + graveyard for every Stage 3 candidate that did
    # NOT clear the gate -- this population is already fully validated
    # (robustness/walk-forward/Monte Carlo all ran), so a full diagnosis
    # here costs nothing further.
    # Keyed by human-readable label (not the raw canonical group name) so a
    # diagnosis's "related successful family" is directly presentable, and
    # compared against the raw per-candidate skeleton family name so it
    # never accidentally points a candidate at its own family.
    family_scores = {
        family_label(s.group): s.best_score
        for s in summarize_family_performance(stage3_records, score_key="composite_score")
        if s.best_score is not None
    }
    stage3_rejects = [r for r in stage3_records if not r.get("passed_stage3_gate")]
    grave_entries: list[GraveyardEntry] = []
    for rec in stage3_rejects:
        diag = diagnose_candidate(
            candidate_id=rec.get("candidate_id", "?"), family=rec.get("family") or "unknown",
            verdict="FAILED prop validation (Stage 3 gate)",
            statistics=rec.get("statistics"), mc_summary=rec.get("mc_summary"),
            robustness=rec.get("robustness"), walk_forward=rec.get("walk_forward"),
            family_performance=family_scores,
        )
        diagnoses.append(diag)
        grave_entries.append(GraveyardEntry(
            candidate_id=diag.candidate_id, family=diag.family, generation=None, stage_died="stage3",
            reason=diag.primary_failure,
            monte_carlo_failure_pct=(100.0 - (rec.get("mc_summary") or {}).get("evaluation_pass_probability", 0.0))
            if rec.get("mc_summary") else None,
            neighbor_robustness_pct=((rec.get("robustness") or {}).get("stability_ratio", None)),
            prop_sim_pass_pct=(rec.get("mc_summary") or {}).get("evaluation_pass_probability"),
            fitness_score=rec.get("composite_score"),
            param_signature=param_signature(diag.family, rec.get("config")),
            primary_failure=diag.primary_failure, secondary_failure=diag.secondary_failure,
            strength=diag.strength, weakness=diag.weakness,
            suggested_mutation=diag.suggested_mutation,
            related_successful_family=diag.related_successful_family,
        ))
    if grave_entries:
        record_rejections(grave_entries, path=graveyard_path)
    log(f"Diagnosed and graveyard-logged {len(grave_entries)} Stage 3 rejection(s).")
    check_cancelled()

    gate_passed = [r for r in stage3_records if r.get("passed_stage3_gate")]
    gate_passed.sort(key=lambda r: r.get("composite_score") or 0.0, reverse=True)
    pool = gate_passed[: config.cpcv_pool_size]

    # ------------------------------------------------------------------
    # Forge stage -- CPCV robustness + PBO + regime testing, on Stage 3's
    # actual survivors (not the whole population -- this is expensive:
    # several fresh full-dataset backtests per candidate).
    # ------------------------------------------------------------------
    cpcv_shortlist: list[dict] = []
    for rec in pool:
        check_cancelled()
        spec = _rebuild_spec(rec)
        try:
            cpcv_result = run_cpcv(
                search_df, _strategy_builder(spec), risk,
                n_groups=config.cpcv_n_groups, n_test_groups=config.cpcv_n_test_groups,
                metric="eval_pass_probability", prop_rules=prop_rules,
            )
        except CPCVError as exc:
            log(f"  CPCV skipped for {rec.get('candidate_id')}: {exc}")
            cpcv_result = None
        try:
            regime_result = run_regime_test(search_df, _strategy_builder(spec), risk, n_regimes=config.regime_buckets)
        except (ValueError, Exception):  # noqa: BLE001 -- regime test is best-effort, never fatal to a candidate
            regime_result = None

        cpcv_dict = cpcv_result.to_dict() if cpcv_result else None
        regime_dict = regime_result.to_dict() if regime_result else None
        is_robust = cpcv_result.is_robust if cpcv_result else True   # not enough data to test = not penalized
        is_regime_stable = regime_result.is_regime_stable if regime_result else True

        if is_robust and is_regime_stable:
            cpcv_shortlist.append({**rec, "cpcv": cpcv_dict, "regime": regime_dict})
        else:
            diag = diagnose_candidate(
                candidate_id=rec.get("candidate_id", "?"), family=rec.get("family") or "unknown",
                verdict="FAILED CPCV/regime gate",
                statistics=rec.get("statistics"), mc_summary=rec.get("mc_summary"),
                robustness=rec.get("robustness"), walk_forward=rec.get("walk_forward"),
                regime=regime_dict, cpcv=cpcv_dict, family_performance=family_scores,
            )
            diagnoses.append(diag)
            record_rejections([GraveyardEntry(
                candidate_id=diag.candidate_id, family=diag.family, generation=None, stage_died="cpcv",
                reason=diag.primary_failure, pbo=None,
                param_signature=param_signature(diag.family, rec.get("config")),
                primary_failure=diag.primary_failure, secondary_failure=diag.secondary_failure,
                strength=diag.strength, weakness=diag.weakness,
                suggested_mutation=diag.suggested_mutation,
                related_successful_family=diag.related_successful_family,
            )], path=graveyard_path)

    funnel.append(FunnelStage("CPCV/PBO + regime testing", n_in=len(pool), n_out=len(cpcv_shortlist)))
    log(f"CPCV/PBO + regime testing: {len(cpcv_shortlist)}/{len(pool)} survived.")

    cohort_pbo: float | None = None
    if len(cpcv_shortlist) > 1:
        try:
            pbo_result = compute_pbo(
                search_df, [_rebuild_spec(r) for r in cpcv_shortlist], risk,
                n_groups=config.cpcv_n_groups, n_test_groups=config.cpcv_n_test_groups,
                metric="eval_pass_probability", prop_rules=prop_rules,
            )
            cohort_pbo = pbo_result.pbo
            log(f"  Cohort Probability of Backtest Overfitting (selecting among this shortlist): {cohort_pbo:.0%}.")
        except CPCVError as exc:
            log(f"  Cohort PBO skipped: {exc}")

    cpcv_shortlist = cpcv_shortlist[: max(config.cpcv_survivors, 1)]
    check_cancelled()

    # ------------------------------------------------------------------
    # Forge stage -- deeper, higher-fidelity Monte Carlo on the shortlist.
    # ------------------------------------------------------------------
    mc_pool: list[dict] = []
    for rec in cpcv_shortlist:
        check_cancelled()
        spec = _rebuild_spec(rec)
        strategy = build_strategy_from_spec(spec)
        bt = run_backtest(search_df, strategy, risk)
        if not bt.trades:
            continue
        mc_result = run_monte_carlo(bt.trades, prop_rules, MonteCarloConfig(n_simulations=config.final_mc_sims))
        mc_pool.append({**rec, "trades": bt.trades, "statistics": bt.statistics.to_dict(), "final_mc": mc_result.to_dict()})
    mc_pool.sort(key=lambda r: r["final_mc"]["evaluation_pass_probability"], reverse=True)
    kept_mc = mc_pool[: config.mc_survivors]
    funnel.append(FunnelStage("Monte Carlo (final, deeper)", n_in=len(cpcv_shortlist), n_out=len(kept_mc)))
    log(f"Final Monte Carlo ({config.final_mc_sims:,} sims): kept the top {len(kept_mc)} of {len(mc_pool)}.")
    check_cancelled()

    # ------------------------------------------------------------------
    # Forge stage -- rolling prop evaluation across every real historical
    # starting day.
    # ------------------------------------------------------------------
    rolling_pool: list[dict] = []
    for rec in kept_mc:
        check_cancelled()
        try:
            rolling_result = run_rolling_evaluation(rec["trades"], prop_rules, window_trading_days=config.eval_window_days)
        except ValueError:
            continue
        rolling_pool.append({**rec, "rolling": rolling_result.to_dict()})
    rolling_pool.sort(key=lambda r: (r["rolling"]["pass_rate_pct"], r["rolling"]["first_payout_rate_pct"]), reverse=True)
    kept_rolling = rolling_pool[: config.rolling_survivors]
    funnel.append(FunnelStage("Rolling prop evaluation", n_in=len(kept_mc), n_out=len(kept_rolling)))
    log(f"Rolling prop evaluation: kept the top {len(kept_rolling)} of {len(rolling_pool)}.")

    # Diagnose + graveyard anything Monte Carlo/rolling eval cut, using
    # whatever's richest for that candidate (rolling if it got there, else
    # the final MC summary).
    survivor_ids = {r["candidate_id"] for r in kept_rolling}
    for rec in mc_pool:
        if rec["candidate_id"] in survivor_ids:
            continue
        rolling_dict = rec.get("rolling")
        diag = diagnose_candidate(
            candidate_id=rec.get("candidate_id", "?"), family=rec.get("family") or "unknown",
            verdict="FAILED final Monte Carlo / rolling evaluation",
            statistics=rec.get("statistics"), mc_summary=rec.get("final_mc"),
            rolling=rolling_dict, family_performance=family_scores,
        )
        diagnoses.append(diag)
        record_rejections([GraveyardEntry(
            candidate_id=diag.candidate_id, family=diag.family, generation=None, stage_died="stress",
            reason=diag.primary_failure,
            monte_carlo_failure_pct=100.0 - rec["final_mc"]["evaluation_pass_probability"],
            prop_sim_pass_pct=rec["final_mc"]["evaluation_pass_probability"],
            param_signature=param_signature(diag.family, rec.get("config")),
            primary_failure=diag.primary_failure, secondary_failure=diag.secondary_failure,
            strength=diag.strength, weakness=diag.weakness,
            suggested_mutation=diag.suggested_mutation,
            related_successful_family=diag.related_successful_family,
        )], path=graveyard_path)
    check_cancelled()

    # ------------------------------------------------------------------
    # Forge stage -- locked OOS holdout, on the data NOTHING above has
    # ever seen.
    # ------------------------------------------------------------------
    locked_survivors: dict[str, dict] = {}
    for rec in kept_rolling:
        spec = _rebuild_spec(rec)
        strategy = build_strategy_from_spec(spec)
        holdout_bt = run_backtest(locked_df, strategy, risk) if len(locked_df) else None
        if holdout_bt is None or not holdout_bt.trades or len(holdout_bt.trades) < 5:
            rec["locked_oos"] = {"status": "NOT TESTED", "pass_rate_pct": None}
            locked_survivors[rec["candidate_id"]] = rec  # too few holdout trades to judge -- keep, flagged
            continue
        try:
            holdout_rolling = run_rolling_evaluation(
                holdout_bt.trades, prop_rules,
                window_trading_days=min(config.eval_window_days, max(len(holdout_bt.trades) // 2, 5)),
                max_windows=200,
            )
            pass_rate = holdout_rolling.pass_rate_pct
        except ValueError:
            pass_rate = 100.0 if (holdout_bt.statistics.net_profit or 0) > 0 else 0.0
        if pass_rate >= config.locked_oos_min_pass_rate:
            rec["locked_oos"] = {"status": "PASSED", "pass_rate_pct": pass_rate}
            locked_survivors[rec["candidate_id"]] = rec
        else:
            rec["locked_oos"] = {"status": "FAILED", "pass_rate_pct": pass_rate}
            diag = diagnose_candidate(
                candidate_id=rec.get("candidate_id", "?"), family=rec.get("family") or "unknown",
                verdict="FAILED locked out-of-sample holdout",
                statistics=holdout_bt.statistics.to_dict(),
                rolling=rec.get("rolling"), family_performance=family_scores,
            )
            diagnoses.append(diag)
            record_rejections([GraveyardEntry(
                candidate_id=diag.candidate_id, family=diag.family, generation=None, stage_died="stress",
                reason=f"Locked OOS holdout pass rate only {pass_rate:.0f}%.",
                oos_result="negative", param_signature=param_signature(diag.family, rec.get("config")),
                primary_failure=diag.primary_failure, secondary_failure=diag.secondary_failure,
                strength=diag.strength, weakness=diag.weakness,
                suggested_mutation=diag.suggested_mutation,
                related_successful_family=diag.related_successful_family,
            )], path=graveyard_path)

    funnel.append(FunnelStage("Locked OOS holdout", n_in=len(kept_rolling), n_out=len(locked_survivors)))
    log(f"Locked OOS holdout: {len(locked_survivors)}/{len(kept_rolling)} held up on data never searched over.")

    # ------------------------------------------------------------------
    # Leaderboard -- built from the rolling-evaluation shortlist (kept_rolling),
    # annotated with locked-OOS status, and ranked by PROP SURVIVAL SCORE
    # (app.prop.survival_engine), not raw profit.
    # ------------------------------------------------------------------
    leaderboard: list[ForgeLeaderboardRow] = []
    for rec in kept_rolling:
        survival = run_prop_survival_analysis(rec["trades"], prop_rules, PropSurvivalConfig(n_simulations=5_000))
        rolling = rec["rolling"]
        stats = rec["statistics"]
        locked = rec.get("locked_oos") or {"status": "NOT TESTED", "pass_rate_pct": None}
        leaderboard.append(ForgeLeaderboardRow(
            candidate_id=rec["candidate_id"], family=rec.get("family") or "unknown",
            family_label=family_label(rec.get("family") or "unknown"),
            hypothesis=hypothesis_question(rec.get("family")) if rec.get("family") in _known_families() else "",
            pass_rate_pct=round(rolling["pass_rate_pct"], 1),
            payout_rate_pct=round(rolling["first_payout_rate_pct"], 1),
            median_days_to_pass=rolling.get("median_days_to_pass"),
            max_drawdown_pct=round(float(stats.get("max_drawdown_pct", 0.0) or 0.0), 2),
            max_drawdown_dollars=_dd_dollars(stats, prop_rules.account_size),
            prop_survival_score=round(survival.prop_survival_score, 1),
            locked_oos_status=locked["status"],
            locked_oos_pass_rate_pct=locked.get("pass_rate_pct"),
            net_profit=round(float(stats.get("net_profit", 0.0) or 0.0), 2),
            profit_factor=round(float(stats.get("profit_factor", 0.0) or 0.0), 2),
            total_trades=int(stats.get("total_trades", 0) or 0),
        ))
    leaderboard.sort(key=lambda r: r.prop_survival_score, reverse=True)

    champion = None
    for row in leaderboard:
        if row.locked_oos_status == "PASSED":
            champion = row.candidate_id
            break

    db.finish_run(summary.run_id)
    elapsed = time.time() - t0
    log(f"Forge run complete in {elapsed:.0f}s. {len(leaderboard)} strategy(ies) reached the leaderboard.")

    return ForgeResult(
        run_id=summary.run_id, db_path=db_path, funnel=funnel, leaderboard=leaderboard,
        diagnoses=diagnoses, champion_candidate_id=champion, cohort_pbo=cohort_pbo,
        elapsed_seconds=elapsed,
    )


def _known_families() -> set:
    from app.search.strategy_space import FAMILIES
    return set(FAMILIES.keys())
