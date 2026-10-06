"""
Search Lab batch runner -- Stages 1 through 5.

    Stage 0 (app.search.strategy_space)  generates the candidate pool.
    Stage 1  Cheap filter        -- one fast backtest per candidate, no Monte
                                     Carlo, run in parallel across CPU cores.
                                     Kills the vast majority of candidates in
                                     minutes, not hours.
    Stage 2  GA refinement       -- the app's EXISTING Iterative Refinement
                                     genetic algorithm (app.optimize.refinement,
                                     completely unmodified), applied to each
                                     Stage 1 survivor in parallel instead of
                                     to one hand-picked strategy at a time.
    Stage 3  Validation gate     -- full-fidelity Monte Carlo, multi-fold
                                     walk-forward (no re-tuning), the generic
                                     lookahead-bias detector, a cost-ladder
                                     stress test, and parameter-neighborhood
                                     robustness. A candidate must clear every
                                     gate to pass -- this is deliberately
                                     strict, because Stage 1/2 alone WILL
                                     surface noise at this scale.
    Stage 4  Leaderboard         -- every candidate at every stage is
                                     persisted to SQLite (app.search.results_db);
                                     Stage 3 survivors are ranked by a
                                     composite score that includes the
                                     Deflated Sharpe Ratio, which corrects
                                     for how many candidates were tried.
    Stage 5  Champion promotion  -- app.search.batch_runner.promote_champion()
                                     re-runs one chosen candidate through the
                                     app's existing, trusted
                                     generate_full_report() pipeline, so it
                                     "graduates" into the exact same report
                                     format every other strategy in this app
                                     produces.

Runs in "single" mode (one user-supplied strategy, re-validated through the
exact same funnel) or "family" mode (a generated grid) -- see
app.search.strategy_space. Both use this one pipeline; single-strategy mode
is not a separate code path.

Multiprocessing note: worker processes load the market data ONCE at pool
startup (via ProcessPoolExecutor's `initializer`) rather than having it
pickled through IPC on every one of potentially thousands of per-candidate
tasks. Workers only ever return small, JSON-safe dicts -- never DataFrames
or Trade objects -- back to the main process, which is the only process
that writes to the results database.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig, with_prop_safety_defaults
from app.backtest.statistics import compute_cost_ladder, compute_statistics
from app.backtest.vectorized_fastpath import VectorizedCandidate, is_vectorizable, run_vectorized_batch
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.orchestration.resource_guard import safe_worker_count
from app.optimize.parameter_space import RefinementError
from app.optimize.refinement import RefinementConfig, compute_fitness, run_iterative_refinement
from app.prop.simulator import PropRules, simulate_account, summarize_single_run
from app.reports.generator import generate_full_report
from app.search.failure_triage import aggregate_failure_reasons
from app.search.family_diversity import enforce_family_diversity
from app.search.graveyard import GraveyardEntry, graveyard_path_for, param_signature, record_rejections
from app.search.instrument_risk import resolve_leg_risk
from app.search.results_db import ResultsDB
from app.search.robustness import (
    deflated_sharpe_ratio, parameter_neighborhood_robustness, run_walk_forward,
)
from app.search.strategy_space import SearchSpace, build_strategy_from_spec
from app.strategy.library import StrategyAlreadyExists, save_strategy_metadata, save_strategy_text, set_strategy_status
from app.strategy.lookahead_check import check_for_lookahead

ProgressCallback = Callable[[str], None]


def _record_search_candidates_to_dashboard(
    stage3_records: list[dict], instrument: str, timeframe: str, family: str | None,
) -> None:
    """Every Stage 3 candidate went through a real backtest + full Monte
    Carlo run -- exactly the same kind of result a single Run & Report (or
    Bulk Backtest) run produces -- so it's recorded into run_history the
    same way, purely so Search Lab activity shows up on the Dashboard
    instead of only Bulk Backtest ever doing so. Deliberately scoped to
    Stage 3 only (typically a handful to a few dozen candidates per run,
    never the hundreds/thousands seen at Stage 1) so this can't flood the
    history file. Best-effort: a recording failure here must never affect
    the search result itself."""
    try:
        from app.reports import run_history
    except Exception:
        return
    for rec in stage3_records:
        stats = rec.get("statistics") or {}
        if not stats or not stats.get("total_trades"):
            continue
        try:
            label = f"[Search Lab] {family or rec.get('family') or 'candidate'} {rec['candidate_id'][:8]}"
            report = {
                "strategy": {
                    "name": label, "source_type": rec.get("source_type", "manual"),
                    "instrument": instrument, "timeframe": timeframe,
                },
                "historical_backtest": {"statistics": stats},
                "prop_firm_single_run": rec.get("prop_summary") or {},
                "monte_carlo": rec.get("mc_summary") or {},
            }
            run_history.record_run(report, {"html": ""})
        except Exception:  # noqa: BLE001 -- dashboard visibility is a convenience, never core output
            continue


def _write_search_graveyard_entries(
    stage3_records: list[dict], instrument: str, timeframe: str,
) -> Path | None:
    """Search Lab's counterpart to app.evolution.engine's
    _write_graveyard_entries -- Evolution Lab has recorded expensive-stage
    rejections to the strategy graveyard for a while; Search Lab's Stage 3
    (the same kind of full Monte Carlo / walk-forward / robustness gate)
    never did, so a candidate this run's Stage 3 already ruled out could
    still get re-generated and re-tested from scratch by a LATER search
    or by Forge Strategy against the same instrument+timeframe. Scoped to
    Stage 3 only (not Stage 1/2) -- see app.search.graveyard's own module
    docstring for why the cheap-filter rejection log is deliberately kept
    separate. Uses the same per-instrument path convention as Forge
    (app.search.graveyard.graveyard_path_for) rather than evolution
    engine's flat tested_log_path-relative file, so a Search Lab run and a
    Forge run against the same instrument share and accumulate into one
    history; returns that path (or None if nothing was written / logging
    itself failed) purely for surfacing in SearchSummary/the UI -- never
    allowed to affect the search result.
    """
    entries = []
    for rec in stage3_records:
        if rec.get("passed_stage3_gate"):
            continue
        fam = rec.get("family") or "?"
        config = rec.get("config")
        robustness = rec.get("robustness") or {}
        stability_pct = None
        if robustness.get("stability_ratio") is not None:
            stability_pct = max(0.0, min(1.0, float(robustness["stability_ratio"]))) * 100.0
        wf = rec.get("walk_forward") or {}
        oos_result = None
        if wf.get("walk_forward_efficiency") is not None:
            oos_result = "negative" if wf["walk_forward_efficiency"] < 0 else "positive"
        mc = rec.get("mc_summary") or {}
        raw_pass_pct = mc.get("evaluation_pass_probability")
        mc_failure_pct = (100.0 - raw_pass_pct) if raw_pass_pct is not None else None

        reason = rec.get("gate_notes") or rec.get("error") or (
            "Failed Stage 3's validation gate (Monte Carlo, walk-forward, lookahead, or "
            "parameter-neighborhood robustness) after clearing Stages 1 and 2."
        )
        entries.append(GraveyardEntry(
            candidate_id=rec["candidate_id"], family=fam, generation=None, stage_died="search_lab_stage3",
            reason=reason, oos_result=oos_result,
            neighbor_robustness_pct=stability_pct,
            monte_carlo_failure_pct=mc_failure_pct,
            prop_sim_pass_pct=raw_pass_pct,
            cpcv_oos_pass_pct=None,
            pbo=None,
            fitness_score=rec.get("fitness"),
            param_signature=param_signature(fam, config),
            notes=[reason] if reason else [],
        ))
    if not entries:
        return None
    path = graveyard_path_for(instrument, timeframe)
    try:
        record_rejections(entries, path)
    except Exception:  # noqa: BLE001 -- graveyard logging is diagnostic, never allowed to break a run
        return None
    return path


def _record_fields_from_spec(spec: dict) -> dict:
    """The subset of a candidate spec that gets persisted to / propagated
    through the results DB record for a given stage's output -- separated
    out so every stage task builds this the same way regardless of source
    type, rather than each one hand-rolling which of config/code_text/
    code_extension applies."""
    source_type = spec.get("source_type", "manual")
    if source_type == "manual":
        return {"source_type": "manual", "config": spec.get("config")}
    return {
        "source_type": source_type,
        "code_text": spec.get("code_text"),
        "code_extension": spec.get("code_extension"),
    }


def _spec_from_record(record: dict) -> dict:
    """Reconstructs a candidate spec dict from a stage's output record (or
    a row read back from ResultsDB) -- the inverse of
    _record_fields_from_spec, used to feed one stage's survivors into the
    next stage's task, and to rebuild a strategy for champion promotion."""
    return _record_fields_from_spec(record)


# ---------------------------------------------------------------------------
# Configuration & result containers
# ---------------------------------------------------------------------------

@dataclass
class SearchStageConfig:
    # Stage 1 -- cheap filter
    min_trades: int = 20
    min_profit_factor: float = 1.05
    max_drawdown_buffer_mult: float = 1.5     # candidate's own DD must be <= prop max_drawdown_pct * this
    stage1_top_n: int = 40                    # survivors that advance to Stage 2

    # Stage 2 -- GA refinement (delegates to the existing RefinementConfig)
    # v7 (2026-10-05, worker B, fix #9): raised defaults 10x4 -> 40x15.
    # The old budget was a weekend of random sampling for the space being
    # searched, not a serious search. Callers passing explicit values are
    # unaffected; the web Search Lab form defaults moved with these.
    ga_population: int = 40
    ga_generations: int = 15
    ga_search_sims: int = 300
    stage2_top_n: int = 10                    # survivors that advance to Stage 3

    # ----- v5 B1-1 grammar candidate hook: config (BEGIN) -----
    # Stage 2 candidate source. "templates" (default) keeps the exact old
    # behavior: Stage 2 refines only the Stage 1 skeleton survivors.
    # "grammar" ADDITIONALLY draws fresh structurally-novel candidates
    # straight from app.search.grammar.generate_random() and runs them
    # through the same Stage 2 GA refinement -- this is how Search Lab
    # invents structure instead of only tuning frozen templates.
    candidate_source: str = "templates"       # "templates" | "grammar"
    # Extra grammar-drawn candidates refined per Stage-1 survivor when
    # candidate_source == "grammar".
    grammar_candidates_per_survivor: int = 2
    # ----- v5 B1-1 grammar candidate hook: config (END) -----

    # Stage 2 cost-stress: see RefinementConfig.cost_stress_* -- exposed here
    # so a family-wide search can bias its whole GA toward candidates that
    # survive worse execution, not just candidates that look best under the
    # default cost assumptions.
    cost_stress_enabled: bool = True
    cost_stress_multiplier: float = 2.0
    cost_stress_penalty_weight: float = 0.35

    # Stage 3 -- validation gate
    #
    # Hard floor, checked BEFORE the expensive part of Stage 3 (full-fidelity
    # Monte Carlo, the lookahead detector, walk-forward, and parameter-
    # neighborhood robustness all run once each per candidate).
    #
    # FIX (2026-09-04): this used to default to the exact same bar as Stage
    # 1 (min_trades=20, profit_factor>=1.05, net_profit>0 required) and was
    # applied to a PLAIN full-dataset backtest. But Stage 2's GA doesn't
    # optimize for full-dataset raw profit factor -- it optimizes chained
    # out-of-sample fitness (stage_cfg.fitness_metric, "eval_pass_probability"
    # by default: a Monte-Carlo/prop-rules-aware score computed on folded
    # OOS slices). A genome the GA legitimately selected for a strong
    # chained-OOS eval-pass score can easily land under 1.05 profit factor,
    # or even net-negative, on a single plain full-dataset backtest -- those
    # are two different numbers computed two different ways, not a
    # regression. Re-applying Stage 1's exact bar here silently discarded
    # candidates Stage 2 had already found real signal in, before Monte
    # Carlo -- the stage that actually scores the metric being optimized for
    # -- ever got to see them. Loosened to a genuine "protect against
    # wasting a full Monte Carlo run on something catastrophically broken"
    # floor: net profit is no longer required pre-MC (a candidate with a
    # small full-dataset loss but a real out-of-sample edge deserves its
    # Monte Carlo run), and the profit-factor floor is low enough to only
    # catch configs that are actually broken, not merely below Stage 1's bar.
    stage3_min_trades: int = 20
    stage3_min_profit_factor: float = 0.85
    stage3_max_drawdown_buffer_mult: float = 1.5
    stage3_require_positive_net: bool = False

    # P1-4 acceptance floor on the thing being optimized: Stage 3's gate
    # is stability, not level -- without these, a candidate with an
    # arbitrarily low pass probability can pass and become champion.
    # Scales are MonteCarloResult's own 0-100 fields: 70.0 means 70%,
    # 50.0 means 50%.
    min_eval_pass_probability: float = 70.0
    # B1-2 (w4-forge): payout acceptance floor on the 0-100 Monte Carlo
    # scale -- 50.0 means "at least a 50% per-attempt chance of reaching
    # a first payout". (Was 0.5, a units bug: a 0.5% payout gate, i.e.
    # effectively no gate at all.)
    min_first_payout_probability: float = 50.0

    # P1-4 locked OOS holdout (Forge pattern -- see
    # app.orchestration.forge): the last locked_holdout_frac of `df` is
    # reserved at run_search() start and never touched by Stages 0-3;
    # only promote_champion evaluates on it, first, before the report.
    locked_holdout_frac: float = 0.2

    full_mc_sims: int = 3000
    walk_forward_folds: int = 4
    walk_forward_metric: str = "eval_pass_probability"
    walk_forward_min_efficiency: float = 0.4
    robustness_neighbors: int = 6
    robustness_perturbation_frac: float = 0.15
    robustness_min_stability: float = 0.4

    fitness_metric: str = "eval_pass_probability"
    workers: int | None = None                # None = os.cpu_count()
    random_seed: int = 42

    # UPGRADE (per-family optimizer choice): which search algorithm Stage
    # 2's GA refinement itself uses per candidate -- see
    # app.optimize.refinement.RefinementConfig.optimizer_mode /
    # OPTIMIZER_MODES for the supported values (genetic/tpe/cma_es) and
    # what each does. Exposed here (rather than only on QuickOptimizeConfig/
    # FullPipelineConfig) so Search Lab/Evolution Lab's own "Optimizer
    # mode" dropdown actually reaches Stage 2 instead of always running
    # the genetic default regardless of what was selected.
    optimizer_mode: str = "genetic"

    # UPGRADE (prop-firm reset-on-breach as the search basis): when True,
    # every Monte Carlo pass this stage runs (Stage 2's GA inner-loop
    # scoring AND Stage 3's full-fidelity validation MC), plus each
    # candidate's single-run prop_summary, treats a blown account the way
    # a real prop trader actually would -- buy a new eval and keep going
    # -- instead of scoring that path as a dead loss the moment one
    # account busts. See app.prop.simulator.simulate_account's own
    # reset_on_breach docstring for the mechanics. False (default) is
    # byte-identical to every run before this field existed -- the web/
    # desktop routes that build this config default their own checkbox to
    # CHECKED, so a normal user gets reset-on-breach scoring unless they
    # explicitly turn it off; this field only stays False here so a config
    # built directly in code/tests without going through that form is
    # unaffected.
    reset_on_breach: bool = False

    # B1-2 (w4-forge): which balance the prop simulator's drawdown checks
    # trail on for every simulate_account / run_monte_carlo scoring path
    # in this search. "realized" (default): today's behavior, byte-
    # identical -- checks run against realized trade-close balance only.
    # "adverse": degrade the max-drawdown check with each trade's initial
    # risk as a floating-drawdown proxy (see
    # app.prop.simulator.PropRules.floating_drawdown_mode). The "adverse"
    # simulator path itself is implemented by a sibling worker; this field
    # only plumbs the value through the search configs to the simulator.
    floating_drawdown_mode: str = "realized"

    # Strategy Family Diversity -- caps how many Stage 1 survivors from
    # the SAME classified family (app.strategy.family_taxonomy) can
    # advance into Stage 2, so an "all families" or wide-grid search
    # can't let one family's sheer combinatorial size (e.g. a 10,000-
    # candidate parameter grid for one hypothesis) crowd out every other
    # family before the expensive stages even see them. None (default)
    # disables the cap -- exactly today's behavior, unchanged.
    max_per_family_stage1: int | None = None

    # FIX (2026-09-27, "CI/Search Lab suddenly takes forever"): _stage2_task
    # builds a RefinementConfig for every Stage 1 survivor but never passed
    # these two through, so every Stage 2 refinement silently ran with
    # RefinementConfig's own defaults (plateau_robust_selection=True,
    # plateau_finalist_pool=5) NO MATTER what ga_population/ga_generations/
    # ga_search_sims this config was tuned to. Plateau-robust selection's
    # cost is `plateau_finalist_pool * 2 * len(genes)` FULL extra
    # evaluations on top of the population/generations search itself (see
    # app.optimize.refinement._select_plateau_robust) -- for a strategy
    # with a handful of tunable genes that's dozens of extra full
    # backtest+Monte-Carlo evaluations per Stage 1 survivor, regardless of
    # how cheap the caller tried to make the actual GA. This is exactly
    # why a Search Lab config deliberately scaled down for a fast run (or
    # this file's own small/fast test fixtures) still took 10-50x longer
    # than its population/generations/sims would suggest, and why CI's
    # test suite -- which runs several such small Search Lab runs end to
    # end -- started intermittently blowing well past its 30-minute test
    # timeout. Defaults here match RefinementConfig's own defaults exactly
    # (byte-identical behavior for anyone not explicitly overriding these),
    # but now a caller that wants a genuinely fast/cheap run (this file's
    # own tests included) has an actual knob to turn it down instead of
    # silently paying full plateau-robustness cost every single time.
    plateau_robust_selection: bool = True
    plateau_finalist_pool: int = 5

    def __post_init__(self):
        self.min_trades = max(int(self.min_trades), 1)
        self.min_profit_factor = max(float(self.min_profit_factor), 0.0)
        self.stage1_top_n = max(int(self.stage1_top_n), 1)
        self.stage3_min_trades = max(int(self.stage3_min_trades), 1)
        self.stage3_min_profit_factor = max(float(self.stage3_min_profit_factor), 0.0)
        self.stage3_max_drawdown_buffer_mult = max(float(self.stage3_max_drawdown_buffer_mult), 0.1)
        self.ga_population = max(int(self.ga_population), 4)
        self.ga_generations = max(int(self.ga_generations), 1)
        self.stage2_top_n = max(int(self.stage2_top_n), 1)
        self.full_mc_sims = max(int(self.full_mc_sims), 100)
        self.walk_forward_folds = max(int(self.walk_forward_folds), 0)
        self.robustness_neighbors = max(int(self.robustness_neighbors), 0)
        # B1-2 (w4-forge): fail fast on a typo'd drawdown mode rather than
        # silently scoring as "realized".
        if self.floating_drawdown_mode not in ("realized", "adverse"):
            raise ValueError(
                f"SearchStageConfig.floating_drawdown_mode must be 'realized' or 'adverse', "
                f"got {self.floating_drawdown_mode!r}."
            )


@dataclass
class SearchSummary:
    run_id: str
    mode: str
    family: str | None
    total_candidates: int
    stage1_survivors: int
    stage2_survivors: int
    stage3_survivors: int
    champion_candidate_id: str | None
    elapsed_seconds: float
    db_path: str
    leaderboard: list = field(default_factory=list)
    graveyard_path: str | None = None
    # v9: batches killed by the stall watchdog (240s with no completion) in
    # Stage 2 / Stage 3. The recovery engine uses these to distinguish
    # "too slow for this box" from "genuinely weak candidates".
    stage2_stalled_skipped: int = 0
    stage3_stalled_skipped: int = 0
    # P1-4: how the input df was split for this run -- Stages 0-3 ran on
    # the first (1 - locked_holdout_frac) of bars; the last
    # locked_holdout_frac was locked for promote_champion. Recorded so a
    # later promote_champion call can re-derive the identical locked slice
    # from the same full df even though run_search never returns it.
    locked_holdout_frac: float = 0.2
    dev_bars: int = 0
    locked_bars: int = 0


# ---------------------------------------------------------------------------
# Worker process state & tasks (module-level so ProcessPoolExecutor can
# pickle/import them; state is per-process, populated once by _init_worker)
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(df_pickle_path: str, risk_kwargs: dict, prop_kwargs: dict, tmp_dir_path: str) -> None:
    global _WORKER
    _WORKER["df"] = pd.read_pickle(df_pickle_path)
    _WORKER["risk"] = RiskConfig(**risk_kwargs)
    _WORKER["prop_rules"] = PropRules(**prop_kwargs)
    # Shared scratch directory for materializing Python-strategy candidates
    # to disk (PythonStrategy only accepts a file path). Every write uses a
    # uuid4 filename, so concurrent workers sharing this directory is safe.
    _WORKER["tmp_dir"] = Path(tmp_dir_path)


def _worker_ready_ping() -> bool:
    """No-op task with one job: a ProcessPoolExecutor never runs a
    worker's initializer eagerly at spawn time -- it only runs
    _init_worker (which unpickles the FULL dataset, one copy per worker;
    see _make_pool) lazily, the first time that worker is actually handed
    a task. Submitting this trivial ping to every worker right after
    (re)spawning a pool -- see _warm_up_pool -- and waiting on it forces
    that one-time initialization to happen and complete before any real
    candidate is dispatched, so its cost is measured and logged on its
    own instead of silently eating into the first batch's stall-timeout
    budget (see _warm_up_pool's docstring for the bug this fixes)."""
    return True


def _passes_stage1_filters(stats: dict, min_trades: int, min_profit_factor: float,
                            max_dd_limit: float, require_positive_net: bool = True) -> bool:
    """The Stage 1 pass/fail test, factored out so it can be re-applied
    against already-computed per-candidate statistics with progressively
    looser thresholds (see run_search's auto-relax step) without
    re-running any backtests."""
    if not stats:
        return False
    pf = stats.get("profit_factor", 0.0)
    pf_val = 10.0 if pf == float("inf") else float(pf or 0.0)
    n_trades = stats.get("total_trades", 0)
    max_dd = stats.get("max_drawdown_pct", 0.0) or 0.0
    net_ok = (stats.get("net_profit", 0.0) or 0.0) > 0 if require_positive_net else True
    return bool(n_trades >= min_trades and pf_val >= min_profit_factor and max_dd <= max_dd_limit and net_ok)


def _stage1_score(
    base: dict, stats: dict, has_trades: bool, filters: dict, prop_rules: PropRules,
    scale_mismatch: bool = False,
) -> dict:
    """The Stage 1 pass/fail + quick_score definition, factored out so
    the scalar path (_stage1_task) and the vectorized fast path
    (_stage1_task_batch / app.backtest.vectorized_fastpath) can never
    silently drift apart on what counts as a Stage 1 survivor -- both
    call this one function instead of each keeping its own copy of the
    scoring logic."""
    if not has_trades:
        return {
            **base, "statistics": stats, "error": "no trades generated on this data",
            "passed_stage1": False, "scale_mismatch_warning": scale_mismatch,
        }

    pf = stats.get("profit_factor", 0.0)
    pf_val = 10.0 if pf == float("inf") else float(pf or 0.0)
    n_trades = stats.get("total_trades", 0)
    max_dd = stats.get("max_drawdown_pct", 0.0) or 0.0
    sharpe = stats.get("sharpe_ratio", 0.0) or 0.0

    passed = (
        n_trades >= filters["min_trades"]
        and pf_val >= filters["min_profit_factor"]
        and max_dd <= prop_rules.max_drawdown_pct * filters["max_drawdown_buffer_mult"]
        and stats.get("net_profit", 0.0) > 0
    )
    # Rewards edge (profit factor) AND sample size together, so a 3-trade
    # profit-factor-of-8 fluke doesn't outrank a 200-trade profit-factor-of-1.3
    # strategy that's actually been tested by the data.
    quick_score = pf_val * math.log(n_trades + 1)

    return {
        **base, "statistics": stats,
        "quick_score": quick_score, "sharpe": sharpe, "passed_stage1": bool(passed),
        "scale_mismatch_warning": scale_mismatch,
    }


# ===========================================================================
# w10-astra B3 EDIT -- SEARCH PRE-SCREEN: skip unfundable (strategy, risk%)
# candidates. MERGE POINT: a sibling worker also edits this file; this
# block (helper + two hook call-sites + one driver log line, all tagged
# "w10-astra") is self-contained -- keep it together when merging.
# ---------------------------------------------------------------------------
# Part C root cause operating inside the search: nothing pre-filtered
# (strategy, risk%) pairs that can't afford 1 whole contract, so the search
# spent full Stage-1 backtests on candidates the engine then floors to
# zero contracts on every signal (0 trades, wasted compute, and a
# misleading "no winners" verdict). This pre-screen runs each candidate's
# median stop distance through the SAME RiskConfig.position_size() the
# engine uses; if that sizes to less than one whole contract, the candidate
# is skipped with a counted reason instead of being backtested.
# ===========================================================================
def _candidate_median_stop_pips(spec: dict, risk: RiskConfig, df) -> float | None:
    """Median stop distance in pips for a candidate spec, or None when the
    spec's stop can't be determined without running the strategy (non-manual
    sources, or no stop configured at all -- those pass through)."""
    if spec.get("source_type", "manual") != "manual":
        return None
    cfg = spec.get("config", {}) or {}
    rm = cfg.get("risk_management", {}) or {}
    stop_type = str(rm.get("stop_type", "")).lower()
    if stop_type == "fixed" and rm.get("stop_value") not in (None, ""):
        try:
            return float(rm["stop_value"])
        except (TypeError, ValueError):
            return None
    if cfg.get("stop_loss_pips") not in (None, ""):
        try:
            return float(cfg["stop_loss_pips"])
        except (TypeError, ValueError):
            return None
    if stop_type == "atr" and rm.get("stop_value") not in (None, ""):
        # ATR stop: median stop ~= mult x median(ATR) over the search data,
        # converted to pips. Uses the same ATR definition the engine sizes
        # against (app.strategy.indicators.atr, via manual's _atr_series).
        try:
            mult = float(rm["stop_value"])
            period = max(int(rm.get("stop_atr_period", 14) or 14), 1)
            from app.strategy.indicators import atr as _atr
            atr_med = float(_atr(df, period).median())
            if atr_med > 0 and math.isfinite(atr_med) and risk.pip_size:
                return (mult * atr_med) / risk.pip_size
        except (TypeError, ValueError):
            return None
        return None
    return None


def _prescreen_funding(base: dict, spec: dict, risk: RiskConfig, df) -> dict | None:
    """w10-astra B3: returns a skip-result dict when the candidate cannot
    afford 1 whole contract at its median stop, else None (proceed to the
    Stage 1 backtest). The skip mirrors the engine's own whole-contract
    flooring exactly: RiskConfig.position_size() < contract_size."""
    if not getattr(risk, "contract_size", None):
        return None  # no whole-contract flooring in play -- nothing to pre-screen
    stop_pips = _candidate_median_stop_pips(spec, risk, df)
    if stop_pips is None or not math.isfinite(stop_pips) or stop_pips <= 0:
        return None
    units = risk.position_size(risk.initial_balance, stop_pips)
    if units >= risk.contract_size - 1e-9:
        return None
    detail = (
        f"risk_value={risk.risk_value}% of ${risk.initial_balance:,.0f} cannot afford "
        f"1 contract (contract_size={risk.contract_size:g}) at this candidate's median "
        f"stop of {stop_pips:.1f} pips -- every signal would floor to 0 contracts"
    )
    return {
        **base,
        "passed_stage1": False,
        "prescreen_skipped": True,
        "prescreen_skip_reason": "unfundable",
        # "error" (not a crash -- a deliberate skip) feeds the existing
        # failure-triage counting so the skip shows up in the Stage 1
        # triage summary with its reason.
        "error": f"pre-screen: unfundable ({detail})",
        "prescreen_detail": detail,
    }
# ============================ end w10-astra B3 block ===========================


def _stage1_task(candidate_id: str, spec: dict, filters: dict) -> dict:
    """Stage 1: one fast backtest, no Monte Carlo. Runs in a worker
    process. This is the scalar fallback path -- always correct for
    every strategy shape, used directly for anything the vectorized
    fast path can't handle (see app.backtest.vectorized_fastpath) and
    as the safety-net if a vectorized batch itself raises."""
    df, risk, prop_rules = _WORKER["df"], _WORKER["risk"], _WORKER["prop_rules"]
    tmp_dir = _WORKER.get("tmp_dir")
    base = {"candidate_id": candidate_id, **_record_fields_from_spec(spec)}
    # w10-astra B3 hook: skip candidates that can't afford 1 contract at
    # their median stop before spending a backtest on them.
    prescreen = _prescreen_funding(base, spec, risk, df)
    if prescreen is not None:
        return prescreen
    try:
        strategy = build_strategy_from_spec(spec, tmp_dir)
        bt = run_backtest(df, strategy, risk)
    except Exception as exc:  # noqa: BLE001 -- a bad generated config must not kill the whole search
        return {**base, "error": str(exc), "passed_stage1": False}

    stats = bt.statistics.to_dict()
    # bt.warnings (from app.backtest.engine.run_backtest's own
    # warnings.catch_warnings capture) carries execution.py's
    # pip_scale_mismatch / atr_scale_mismatch / fallback_stop warnings.
    # Those are the single most useful diagnostic when an ENTIRE search
    # comes back with 0 survivors -- e.g. an FX-default pip_size (0.0001)
    # applied to a whole-dollar stock or an index makes every fixed-pip
    # stop nonsensically tiny, so literally every family and parameter
    # combination fails the same way and it looks like "nothing works"
    # rather than "one setting is wrong." Before this, these warnings
    # were raised with plain warnings.warn() *inside a Stage 1 worker
    # subprocess* and never made it back to the run's log at all --
    # invisible to the very diagnosis that most needed them.
    scale_mismatch = any(
        "pip_scale_mismatch" in w or "pip size" in w.lower() or "atr_scale_mismatch" in w
        or ("instrument" in w.lower() and "scale" in w.lower())
        for w in (bt.warnings or [])
    )
    return _stage1_score(base, stats, bool(bt.trades), filters, prop_rules, scale_mismatch)


def _stage1_task_batch(items: list[tuple[str, dict]], filters: dict) -> list[dict]:
    """Stage 1 for a CHUNK of candidates at once (the vectorized fast
    path -- see app.backtest.vectorized_fastpath for why this exists and
    exactly what it does and doesn't simulate). Builds every candidate's
    strategy and signals, routes anything eligible (fixed-pips stop/
    target only) through ONE vectorized pass over the bars, and falls
    back to the exact same per-candidate scalar path (_stage1_task) for
    everything else -- including, defensively, every candidate in this
    chunk if the vectorized batch call itself raises for any reason.
    Runs in a worker process, same as _stage1_task."""
    df, risk, prop_rules = _WORKER["df"], _WORKER["risk"], _WORKER["prop_rules"]
    tmp_dir = _WORKER.get("tmp_dir")

    results: list[dict] = []
    vec_batch: list[VectorizedCandidate] = []
    vec_bases: dict[str, dict] = {}
    spec_by_id: dict[str, dict] = {}

    for candidate_id, spec in items:
        spec_by_id[candidate_id] = spec
        base = {"candidate_id": candidate_id, **_record_fields_from_spec(spec)}
        # w10-astra B3 hook: same pre-screen as the scalar path, before the
        # strategy is even built (vectorized or not, an unfundable candidate
        # is an unfundable candidate).
        prescreen = _prescreen_funding(base, spec, risk, df)
        if prescreen is not None:
            results.append(prescreen)
            continue
        try:
            strategy = build_strategy_from_spec(spec, tmp_dir)
            strat_result = strategy.generate(df)
        except Exception as exc:  # noqa: BLE001 -- a bad generated config must not kill the whole search
            results.append({**base, "error": str(exc), "passed_stage1": False})
            continue

        if is_vectorizable(strat_result):
            # v5: per-bar stop/target distance arrays (e.g. ATR stops)
            # now vectorize too -- pass them through so the fast path
            # prices and sizes off the same distances the scalar engine
            # would use. Length is validated inside run_vectorized_batch
            # (mismatch -> exception -> scalar fallback below).
            sl_dist = strat_result.stop_loss_distance
            tp_dist = strat_result.take_profit_distance
            vec_batch.append(VectorizedCandidate(
                candidate_id=candidate_id,
                signals=strat_result.signals.to_numpy(),
                stop_loss_pips=strat_result.stop_loss_pips,
                take_profit_pips=strat_result.take_profit_pips,
                stop_distances=np.asarray(sl_dist, dtype=float) if sl_dist is not None else None,
                take_distances=np.asarray(tp_dist, dtype=float) if tp_dist is not None else None,
            ))
            vec_bases[candidate_id] = base
        else:
            # Dynamic stop/target, trailing, or breakeven -- not
            # something the fast path can honestly approximate. Exact
            # same treatment this candidate would have gotten before
            # this function existed.
            results.append(_stage1_task(candidate_id, spec, filters))

    if vec_batch:
        try:
            outcomes = run_vectorized_batch(df, vec_batch, risk)
        except Exception:  # noqa: BLE001 -- a batch-level bug must never silently drop candidates; fall back to the trusted scalar path for all of them
            for cand in vec_batch:
                results.append(_stage1_task(cand.candidate_id, spec_by_id[cand.candidate_id], filters))
        else:
            for cand in vec_batch:
                base = vec_bases[cand.candidate_id]
                outcome = outcomes.get(cand.candidate_id)
                trades = outcome.trades if outcome else []
                stats = (
                    compute_statistics(trades, outcome.equity_curve, risk.initial_balance).to_dict()
                    if trades else {}
                )
                mismatch = bool(outcome.scale_mismatch) if outcome else False
                results.append(_stage1_score(base, stats, bool(trades), filters, prop_rules, scale_mismatch=mismatch))

    return results


def _stage2_task(
    candidate_id: str, spec: dict, refine_kwargs: dict, mc_search_sims: int,
    fitness_metric: str, seed: int,
) -> dict:
    """Stage 2: the app's existing GA refinement, run on one Stage 1 survivor."""
    df, risk, prop_rules = _WORKER["df"], _WORKER["risk"], _WORKER["prop_rules"]
    tmp_dir = _WORKER.get("tmp_dir")
    strategy = build_strategy_from_spec(spec, tmp_dir)
    reset_on_breach = refine_kwargs.get("reset_on_breach", False)
    mc_cfg = MonteCarloConfig(n_simulations=mc_search_sims, random_seed=seed, reset_on_breach=reset_on_breach)
    refine_cfg = RefinementConfig(
        fitness_metric=fitness_metric,
        population_size=refine_kwargs["population"],
        generations=refine_kwargs["generations"],
        search_monte_carlo_sims=mc_search_sims,
        random_seed=seed,
        cost_stress_enabled=refine_kwargs.get("cost_stress_enabled", True),
        cost_stress_multiplier=refine_kwargs.get("cost_stress_multiplier", 2.0),
        cost_stress_penalty_weight=refine_kwargs.get("cost_stress_penalty_weight", 0.35),
        optimizer_mode=refine_kwargs.get("optimizer_mode", "genetic"),
        plateau_robust_selection=refine_kwargs.get("plateau_robust_selection", True),
        plateau_finalist_pool=refine_kwargs.get("plateau_finalist_pool", 5),
    )
    try:
        result = run_iterative_refinement(df, strategy, risk, prop_rules, mc_cfg, refine_cfg, progress_cb=None)
    except RefinementError as exc:
        # No tunable numeric parameters (rare for a generated skeleton;
        # possible for a hand-written single_config/strategy in "single"
        # mode) -- fall through with a plain backtest so it can still reach
        # Stage 3 rather than being silently dropped.
        base = {"candidate_id": candidate_id, **_record_fields_from_spec(spec)}
        bt = run_backtest(df, strategy, risk)
        stats = bt.statistics.to_dict()
        if not bt.trades:
            return {**base, "error": str(exc), "passed_stage2": False}
        pnls = [t.pnl for t in bt.trades]
        dates = [t.entry_time for t in bt.trades]
        single_run = simulate_account(pnls, dates, prop_rules, reset_on_breach=reset_on_breach)
        mc = run_monte_carlo(bt.trades, prop_rules, mc_cfg)
        prop_summary = summarize_single_run(single_run)
        fitness = compute_fitness(stats, prop_summary, mc, fitness_metric)
        return {
            **base, "statistics": stats,
            "prop_summary": prop_summary, "fitness": fitness,
            "passed_stage2": math.isfinite(fitness), "ga_skipped_reason": str(exc),
            # D4(c): no GA inner loop ran on this path -- the single
            # backtest+MC above is the baseline trial (counted by the
            # caller), not a GA evaluation.
            "ga_total_evaluations": 0,
        }

    best = result.best
    out_spec_fields = {"source_type": best.source_type}
    if best.source_type == "manual":
        out_spec_fields["config"] = best.config if best.config is not None else spec.get("config")
    else:
        out_spec_fields["code_text"] = best.code_text
        out_spec_fields["code_extension"] = best.code_extension
    return {
        "candidate_id": candidate_id, **out_spec_fields,
        "statistics": best.statistics,
        "prop_summary": best.prop_summary,
        "mc_summary": best.mc_summary,
        "fitness": best.fitness,
        "baseline_fitness": result.baseline.fitness,
        "genes_count": len(result.genes),
        "passed_stage2": math.isfinite(best.fitness),
        # D4(c): the ACTUALLY-RAN inner-loop evaluation count (see
        # RefinementResult.total_evaluations) -- the honest per-candidate
        # trial count the pipeline DSR deflates against. A configured
        # population*(generations+1) budget would under/over-count:
        # auto-shrink-on-low-trades and AI-assist move the real number.
        "ga_total_evaluations": int(getattr(result, "total_evaluations", 0) or 0),
    }


def _stage3_task(candidate_id: str, spec: dict, cfg: dict) -> dict:
    """Stage 3: the strict validation gate. Runs in a worker process."""
    df, risk, prop_rules = _WORKER["df"], _WORKER["risk"], _WORKER["prop_rules"]
    tmp_dir = _WORKER.get("tmp_dir")
    base = {"candidate_id": candidate_id, **_record_fields_from_spec(spec)}
    notes: list[str] = []

    try:
        strategy = build_strategy_from_spec(spec, tmp_dir)
        bt = run_backtest(df, strategy, risk)
    except Exception as exc:  # noqa: BLE001
        return {**base, "error": str(exc), "passed_stage3_gate": False}

    stats = bt.statistics.to_dict()
    if not bt.trades:
        return {
            **base, "statistics": stats,
            "error": "no trades on full dataset", "passed_stage3_gate": False,
        }

    # Early-kill floor -- BEFORE the expensive Monte Carlo / lookahead /
    # walk-forward / robustness work below. See SearchStageConfig's
    # stage3_min_trades/stage3_min_profit_factor/stage3_max_drawdown_buffer_mult
    # docstring for why this exists.
    if not _passes_stage1_filters(
        stats,
        min_trades=cfg.get("stage3_min_trades", 20),
        min_profit_factor=cfg.get("stage3_min_profit_factor", 0.85),
        max_dd_limit=prop_rules.max_drawdown_pct * cfg.get("stage3_max_drawdown_buffer_mult", 1.5),
        require_positive_net=cfg.get("stage3_require_positive_net", False),
    ):
        return {
            **base, "statistics": stats,
            "error": (
                "failed Stage 3 early-kill floor (min_trades="
                f"{cfg.get('stage3_min_trades', 20)}, min_profit_factor="
                f"{cfg.get('stage3_min_profit_factor', 0.85):.2f}, "
                f"max_drawdown<={prop_rules.max_drawdown_pct * cfg.get('stage3_max_drawdown_buffer_mult', 1.5):.1f}%)"
                " -- skipped before Monte Carlo/walk-forward/robustness."
            ),
            "passed_stage3_gate": False,
        }

    trade_pnls = [t.pnl for t in bt.trades]
    trade_dates = [t.entry_time for t in bt.trades]
    reset_on_breach = cfg.get("reset_on_breach", False)
    single_run = simulate_account(trade_pnls, trade_dates, prop_rules, reset_on_breach=reset_on_breach)
    prop_summary = summarize_single_run(single_run)

    mc_cfg = MonteCarloConfig(n_simulations=cfg["full_mc_sims"], random_seed=cfg["random_seed"], reset_on_breach=reset_on_breach)
    mc_result = run_monte_carlo(bt.trades, prop_rules, mc_cfg)
    mc_summary = {
        "evaluation_pass_probability": mc_result.evaluation_pass_probability,
        "first_payout_probability": mc_result.first_payout_probability,
        # P0-1: per-attempt (single-account) odds carried alongside the
        # chain-level reporting fields, so Stage 4's composite_score and
        # the acceptance floor below rank on the honest number.
        "per_attempt_pass_probability": mc_result.per_attempt_pass_probability,
        "per_attempt_payout_probability": mc_result.per_attempt_payout_probability,
        "expected_payout": mc_result.expected_payout,
        "risk_of_ruin_pct": mc_result.risk_of_ruin_pct,
        "median_drawdown_pct": mc_result.median_drawdown_pct,
        "n_simulations": mc_result.n_simulations,
    }
    fitness = compute_fitness(stats, prop_summary, mc_result, cfg["fitness_metric"])

    cost_ladder = compute_cost_ladder(bt.trades)

    # A fresh strategy instance for the lookahead check, walk-forward, and
    # robustness passes below -- consistent with this app's existing
    # "always build fresh per use" convention for strategy instances (see
    # app.search.robustness.run_walk_forward's own docstring) rather than
    # reusing the one already run above.
    lookahead = check_for_lookahead(build_strategy_from_spec(spec, tmp_dir), df)
    lookahead_dict = {
        "checked": lookahead.checked, "bug_detected": lookahead.bug_detected,
        "skip_reason": lookahead.skip_reason,
    }
    if lookahead.bug_detected:
        notes.append("LOOKAHEAD BUG DETECTED -- excluded regardless of every other score.")

    wf_result = None
    wf_dict = None
    if cfg["walk_forward_folds"] >= 2:
        wf_result = run_walk_forward(
            df, lambda: build_strategy_from_spec(spec, tmp_dir), risk,
            n_folds=cfg["walk_forward_folds"], metric=cfg["walk_forward_metric"],
            stability_threshold=cfg["walk_forward_min_efficiency"],
            prop_rules=prop_rules, mc_cfg=mc_cfg,
        )
        if wf_result is not None:
            wf_dict = {
                "n_folds": wf_result.n_folds, "metric": wf_result.metric,
                "mean_train_metric": wf_result.mean_train_metric,
                "mean_test_metric": wf_result.mean_test_metric,
                "walk_forward_efficiency": wf_result.walk_forward_efficiency,
                "is_stable": wf_result.is_stable,
            }
            if not wf_result.is_stable:
                notes.append(
                    f"Walk-forward efficiency {wf_result.walk_forward_efficiency:.2f} below "
                    f"{cfg['walk_forward_min_efficiency']:.2f} threshold."
                )
        else:
            notes.append("Not enough data to walk-forward test -- treated as unproven, not failed.")

    robustness = None
    robustness_dict = None
    if cfg["robustness_neighbors"] > 0:
        robustness = parameter_neighborhood_robustness(
            spec, df, risk, prop_rules, mc_cfg,
            fitness_metric=cfg["fitness_metric"],
            perturbation_frac=cfg["robustness_perturbation_frac"],
            n_neighbors=cfg["robustness_neighbors"],
            seed=cfg["random_seed"],
            stability_threshold=cfg["robustness_min_stability"],
            tmp_dir=tmp_dir,
        )
        if robustness is not None:
            robustness_dict = {
                "n_neighbors_tested": robustness.n_neighbors_tested,
                "stability_ratio": robustness.stability_ratio,
                "mean_neighbor_fitness": robustness.mean_neighbor_fitness,
                "min_neighbor_fitness": robustness.min_neighbor_fitness,
                "is_stable": robustness.is_stable,
            }
            if not robustness.is_stable:
                notes.append(
                    f"Parameter-neighborhood stability ratio {robustness.stability_ratio:.2f} below "
                    f"{cfg['robustness_min_stability']:.2f} -- may be fit to noise in this window."
                )

    # P1-4 acceptance floor: rank on per-attempt (single-account) odds and
    # require the candidate to clear BOTH floors -- a candidate below
    # either fails Stage 3 no matter how stable/robust it looks. Scales
    # are MonteCarloResult's 0-100 fields; cfg defaults come from
    # SearchStageConfig.min_eval_pass_probability /
    # min_first_payout_probability.
    per_attempt_pass = mc_summary.get(
        "per_attempt_pass_probability", mc_summary.get("evaluation_pass_probability", 0.0))
    per_attempt_payout = mc_summary.get(
        "per_attempt_payout_probability", mc_summary.get("first_payout_probability", 0.0))
    min_eval_floor = cfg.get("min_eval_pass_probability", 70.0)
    min_payout_floor = cfg.get("min_first_payout_probability", 50.0)  # B1-2 (w4-forge): 0-100 MC scale
    if per_attempt_pass < min_eval_floor:
        notes.append(
            f"Per-attempt eval pass probability {per_attempt_pass:.1f}% below "
            f"acceptance floor {min_eval_floor:.1f}% -- failed Stage 3 on level, not stability."
        )
    if per_attempt_payout < min_payout_floor:
        notes.append(
            f"Per-attempt first-payout probability {per_attempt_payout:.1f}% below "
            f"acceptance floor {min_payout_floor:.1f}% -- failed Stage 3 on level, not stability."
        )

    passed = (
        not lookahead.bug_detected
        and (wf_result is None or wf_result.is_stable)
        and (robustness is None or robustness.is_stable)
        and math.isfinite(fitness)
        and per_attempt_pass >= min_eval_floor
        and per_attempt_payout >= min_payout_floor
    )

    return {
        **base, "statistics": stats,
        "prop_summary": prop_summary, "mc_summary": mc_summary, "fitness": fitness,
        "sharpe": stats.get("sharpe_ratio", 0.0), "cost_ladder": cost_ladder,
        "lookahead": lookahead_dict, "walk_forward": wf_dict, "robustness": robustness_dict,
        "passed_stage3_gate": bool(passed), "gate_notes": "; ".join(notes),
    }


class SearchCancelled(Exception):
    """Raised when the caller sets ``cancel_event`` mid-run. Not an error --
    the UI catches this to report a clean user-requested stop rather than a
    crash. Any candidates already scored before the cancel point are still
    written to the results DB, so a stopped run isn't a wasted one."""


def _warm_up_pool(pool, workers: int, n_bars: int, log) -> None:
    """Forces every worker's one-time initializer (_init_worker -- which
    unpickles a FULL COPY of the dataset in each worker process) to run
    and complete before this pool is handed any real candidates, and
    times/logs how long that took.

    This fixes a real, previously-unexplained failure mode: a
    ProcessPoolExecutor does not run its `initializer` eagerly when the
    pool is created -- it only runs lazily, in each worker, the first
    time that worker is actually handed a task. On a small dataset that
    cost is a few milliseconds and genuinely was negligible. On a large
    one (millions of bars -- the exact "loaded 2,353,209 bars" scale
    this app now regularly runs against) unpickling that dataframe
    `workers` times over, all starting at once and competing for the
    same disk/CPU, can easily take minutes. Since that unpickling
    happened to count as "time since the first real candidate future was
    submitted" with zero completions, _drain_futures' stall-timeout
    (see its docstring) could not tell that apart from a genuinely
    wedged worker -- it would fire, terminate the "stuck" workers, spawn
    a fresh pool, and immediately pay the exact same unpickling cost
    again, over and over, every batch getting marked skipped without a
    single candidate ever actually running. That is precisely the
    "candidates: 200, Stage 1: 0, ... 0/0 survived" shape this addresses:
    the strategies were never the problem, nothing was ever scored.

    Warming up here moves that one-time cost into its own clearly-logged
    step, outside the stall-detection window entirely, for every pool
    this run creates -- both the initial one and any stall-recovery
    respawn (`_make_pool` is used as both, see run_search).
    """
    t0 = time.monotonic()
    futures = [pool.submit(_worker_ready_ping) for _ in range(max(1, workers))]
    try:
        done, pending = futures_wait(set(futures), timeout=max(60.0, n_bars / 2000.0))
    except Exception:
        return
    elapsed = time.monotonic() - t0
    if pending:
        log(
            f"  ** Worker pool warm-up: {len(pending)}/{len(futures)} worker(s) still hadn't finished "
            f"loading this {n_bars:,}-bar dataset after {elapsed:.0f}s. Continuing anyway -- if Stage 1 "
            f"immediately reports every batch as stalled/skipped, this dataset is too large for this "
            f"machine to hold {workers} full in-memory copies of comfortably; try fewer parallel workers "
            f"or a smaller/downsampled dataset."
        )
    elif elapsed > 5.0:
        log(f"  Worker pool ready ({workers} worker(s) loaded {n_bars:,} bars in {elapsed:.0f}s).")


class StageStalled(Exception):
    """Internal signal from _drain_futures: raised only when no
    pool_factory was supplied to recover from a stall, so the caller can
    decide how to handle it. When a pool_factory IS supplied, _drain_futures
    recovers on its own (terminates the wedged worker(s), respawns a fresh
    pool, marks the stuck batch(es) skipped) and never raises this."""


def _drain_futures(
    pool_box: list, futures: dict, cancel_event: threading.Event | None, on_result, log,
    pool_factory=None, stall_timeout: float = 240.0,
) -> None:
    """Consumes a {future: label} dict as futures complete, calling
    on_result(label, future) for each one -- used by every Search Lab
    stage INSTEAD of Python's `for f in as_completed(futures)`.

    as_completed() with no timeout blocks until the NEXT future completes,
    however long that takes -- and a cancel check placed only BETWEEN loop
    iterations never gets a chance to run until a future actually
    completes. One hung/wedged candidate (a pathological generated
    config, a stuck worker process, anything) therefore made the whole
    stage -- and the STOP button along with it -- block indefinitely. This
    is the exact same bug class already fixed once in the Evolution Lab
    (see app.evolution.engine.EvolutionRunner._drain_futures); Search Lab
    had its own separate copy of the pattern in 3 places (Stage 1, 2, 3)
    that never got the same fix, which is why it can still be reported as
    freshly "stalling with no output" even after that fix shipped. This
    polls with a short timeout instead, so cancel_event gets checked
    roughly once a second regardless of how long any individual candidate
    takes, and abandons the remaining futures immediately (rather than
    waiting on them) the moment a stop is requested.

    That earlier fix only made the STOP BUTTON responsive during a stall
    -- it did nothing for a stall nobody notices in time to click Stop for
    (an overnight/unattended run), which is exactly the "stalled again at
    stage 1, N/M batches done, never any further" report this second fix
    targets. `pool_box` is a mutable [pool] (not a bare pool) so this
    function can swap in a freshly-spawned pool mid-stage: if
    `stall_timeout` seconds pass with ZERO futures completing while at
    least one is still pending, the remaining pending futures are assumed
    wedged (a worker stuck on one pathological candidate, or a worker
    process that silently died and will never report back). Every
    still-alive worker process backing this pool is terminated outright
    (cancel_futures=True alone only drops futures that hadn't STARTED yet
    -- it does not stop a worker already inside a hung call), a fresh pool
    is spawned via `pool_factory()` for the caller to keep using, and the
    stuck batch(es) are reported to `on_result` with `fut=None` (skipped,
    not scored) so the stage can finish with everything else instead of
    hanging forever. Only engages when `pool_factory` is supplied --
    without one this behaves exactly as before (a genuine stall with no
    caller-provided recovery path still surfaces, rather than silently
    swallowing a real bug during testing/debugging).
    """
    pending = set(futures.keys())
    last_progress = time.monotonic()
    while pending:
        pool = pool_box[0]
        if cancel_event is not None and cancel_event.is_set():
            log("\nStop requested -- cancelling remaining candidates and shutting down workers...")
            for fut in pending:
                fut.cancel()  # only frees futures that hadn't started yet
            pool.shutdown(wait=False, cancel_futures=True)
            raise SearchCancelled("Search Lab run stopped by user.")
        done, pending = futures_wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
        if done:
            last_progress = time.monotonic()
            for fut in done:
                on_result(futures[fut], fut)
            continue
        if not pending:
            break
        stalled_for = time.monotonic() - last_progress
        if pool_factory is None or stalled_for < stall_timeout:
            continue
        stuck_labels = [futures[f] for f in pending]
        log(
            f"\nNo progress for {int(stalled_for)}s -- {len(pending)} batch(es) appear stuck "
            f"(a worker likely hung on one pathological candidate, or its process died silently): "
            f"{stuck_labels}. Terminating the stuck worker process(es), marking those batch(es) as "
            f"skipped, and continuing with a freshly-spawned worker pool instead of hanging forever."
        )
        for fut in list(pending):
            fut.cancel()
        try:
            for proc in list(getattr(pool, "_processes", {}).values()):
                if proc.is_alive():
                    proc.terminate()
        except Exception as exc:
            log(f"  (couldn't terminate a stuck worker process cleanly: {exc})")
        pool.shutdown(wait=False, cancel_futures=True)
        pool_box[0] = pool_factory()
        for fut in pending:
            on_result(futures[fut], None)  # None fut == skipped, never scored
        pending = set()



# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# B1-2 (w4-forge): stamp the search's floating_drawdown_mode onto the
# PropRules every simulator/MC scoring path in run_search uses. Default
# "realized" is byte-identical to today's behavior (PropRules' own
# default); "adverse" is implemented inside the simulator by a sibling
# worker. Factored out (rather than inline) so the threading is directly
# unit-testable.
def _search_prop_rules(prop_rules: PropRules, stage_cfg: "SearchStageConfig") -> PropRules:
    return replace(prop_rules, floating_drawdown_mode=stage_cfg.floating_drawdown_mode)


def _stage_eval_counts(stage1_records: list, stage2_records: list, stage3_records: list) -> dict:
    """D4(c): honest per-stage trial counts for the champion metadata.

    Every backtested configuration counts as a trial, not just the
    survivors that reached Stage 3 -- this is the multiple-testing
    burden a downstream pipeline DSR (see count_all_trials in
    app.search.robustness) must deflate against:
      stage1_backtests: one cheap evaluation per Stage 1 candidate (a
        pre-screen skip still counts -- the search examined that
        configuration and made a selection decision on it);
      stage2_ga_evaluations: the ACTUALLY-RAN GA inner-loop trials
        summed across every refined candidate (see ga_total_evaluations
        on the Stage 2 records), not the configured
        population*(generations+1) budget;
      stage3_backtests: one full validation backtest per Stage 3
        candidate;
      total_evaluations: the sum, for convenience.
    Factored out (rather than inline in run_search) so the assembly is
    directly unit-testable.
    """
    counts = {
        "stage1_backtests": len(stage1_records),
        "stage2_ga_evaluations": sum(int(r.get("ga_total_evaluations", 0) or 0) for r in stage2_records),
        "stage3_backtests": len(stage3_records),
    }
    counts["total_evaluations"] = (
        counts["stage1_backtests"] + counts["stage2_ga_evaluations"] + counts["stage3_backtests"]
    )
    return counts


def run_search(
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    space: SearchSpace,
    stage_cfg: SearchStageConfig,
    db_path: str,
    instrument: str = "unknown",
    timeframe: str = "unknown",
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> SearchSummary:
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    def check_cancelled(pool) -> None:
        """Call between/within stage loops. On a stop request, cancels every
        not-yet-started future and drops out of the pool without waiting for
        the whole remaining batch, then raises SearchCancelled so the caller
        can stop cleanly instead of surfacing this as a crash."""
        if cancel_event is not None and cancel_event.is_set():
            log("\nStop requested -- cancelling remaining candidates and shutting down workers...")
            pool.shutdown(wait=False, cancel_futures=True)
            raise SearchCancelled("Search Lab run stopped by user.")

    def drain_futures(pool_box: list, futures: dict, on_result, pool_factory=None) -> None:
        """Thin wrapper binding this run's cancel_event/log into the
        module-level _drain_futures (see its docstring for why this
        exists instead of `for f in as_completed(futures)`)."""
        _drain_futures(pool_box, futures, cancel_event, on_result, log, pool_factory=pool_factory)

    # FIX (audit, Sep 2026): every other heavy-job orchestrator (Evolution
    # Lab, Quick Optimize, Full Pipeline, Speed Run) hardens its RiskConfig
    # against the active PropRules before running a single backtest, so the
    # account-blown circuit breaker and daily-loss circuit breaker (see
    # app.backtest.execution) actually fire during the RAW backtest every
    # stage below scores fitness from -- Search Lab was the one heavy job
    # that never did this, so its Stage 1/2/3 backtests could keep opening
    # new trades straight through a blown account or a breached daily-loss
    # limit, something no real prop account could ever do. That silently
    # let Search Lab rank candidates on stats no live account could
    # reproduce. See app.backtest.risk.with_prop_safety_defaults' own
    # docstring for exactly what this fills in (and never overrides an
    # explicit value the caller already set).
    risk = with_prop_safety_defaults(risk, prop_rules)
    # FIX (2026-09-18): see RiskConfig.reset_on_breach's docstring --
    # stage_cfg.reset_on_breach was already threaded into the post-hoc
    # simulate_account/MonteCarloConfig scoring layer below but never into
    # the RiskConfig Stage 1/2/3 actually backtest every candidate against.
    risk = replace(risk, reset_on_breach=stage_cfg.reset_on_breach)

    # v7 (2026-10-05, worker B, fix #7): pip_size backstop -- WARN loudly
    # if the untouched FX default disagrees with the data. We do NOT
    # silently fix it here; the fix belongs client-side (auto-fill with
    # user confirmation), and Stage 1's "** LIKELY ROOT CAUSE **" line
    # must still fire. (Multi-instrument legs ARE fixed per-leg in
    # cross_instrument.py, where there is no per-leg UI.)
    from app.search.instrument_risk import resolve_leg_risk
    _, _v7_risk_notes = resolve_leg_risk(risk, df, instrument)
    for _v7_note in _v7_risk_notes:
        log(_v7_note)

    # B1-2 (w4-forge): thread floating_drawdown_mode through to every
    # simulator call below -- simulate_account and run_monte_carlo both
    # read it off PropRules. Stamping it here covers the in-process
    # simulate_account calls AND the worker processes (prop_kwargs =
    # asdict(prop_rules) is built from this object below).
    prop_rules = _search_prop_rules(prop_rules, stage_cfg)

    # P1-4: locked OOS holdout, reserved BEFORE any stage runs (Forge
    # pattern -- see app.orchestration.forge). Stages 0-3 below only ever
    # see `dev_df`; the last locked_holdout_frac of bars is locked away
    # until promote_champion's first evaluation. The split is
    # chronological (tail = most recent bars = the honest OOS slice).
    n_bars = len(df)
    split_idx = max(1, min(int(n_bars * (1 - stage_cfg.locked_holdout_frac)), n_bars - 1)) if n_bars > 1 else n_bars
    dev_df = df.iloc[:split_idx].reset_index(drop=True)
    locked_df = df.iloc[split_idx:].reset_index(drop=True)
    log(
        f"Reserved the final {stage_cfg.locked_holdout_frac:.0%} of the dataset ({len(locked_df):,} bars) as a "
        f"locked out-of-sample holdout -- Stages 0-3 below run on the first {len(dev_df):,} bars only."
    )

    run_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    workers = stage_cfg.workers or max(os.cpu_count() or 2, 1)
    workers = max(1, min(workers, len(space.candidates)))
    # Each worker process below loads its OWN full copy of `dev_df` (see
    # _init_worker) -- on a large dataset (e.g. years of 1-minute bars),
    # os.cpu_count() workers each holding a full copy can exhaust system
    # memory well before it exhausts CPU, especially if another heavy job
    # (Evolution Lab, Full Pipeline) is ALSO running at the same time. Cap
    # to what's actually safe given this dataset's size and currently
    # available memory rather than trusting the caller's/CPU count blindly.
    # See app.orchestration.resource_guard for the full rationale.
    safe_workers = safe_worker_count(dev_df, requested=workers, max_candidates_in_flight=len(space.candidates))
    if safe_workers < workers:
        log(
            f"Reducing worker processes from {workers} to {safe_workers} -- {len(dev_df):,} bars is "
            f"large enough that {workers} full copies of it (one per worker) would risk exhausting "
            f"available memory. Install 'psutil' for a more precise estimate; for now this uses a "
            f"conservative fallback."
        )
    workers = safe_workers

    db = ResultsDB(db_path)
    db.create_run(run_id, space.mode, space.family, instrument, timeframe, len(space.candidates), asdict(stage_cfg))

    tmp_dir = Path(tempfile.mkdtemp(prefix="t58_search_"))
    df_path = tmp_dir / "data.pkl"
    dev_df.to_pickle(df_path)
    risk_kwargs = asdict(risk)
    prop_kwargs = asdict(prop_rules)

    filters = {
        "min_trades": stage_cfg.min_trades,
        "min_profit_factor": stage_cfg.min_profit_factor,
        "max_drawdown_buffer_mult": stage_cfg.max_drawdown_buffer_mult,
    }

    log(
        f"Search space ready: {len(space.candidates)} candidate(s) "
        f"({space.mode}{', family=' + space.family if space.family else ''}"
        f"{', sampled from ' + str(space.total_generated) if space.sampled else ''})."
    )

    stage1_records: list[dict] = []
    survivors1: list[dict] = []
    stage2_records: list[dict] = []
    survivors2: list[dict] = []
    stage3_records: list[dict] = []

    def _make_pool():
        pool = ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker,
            initargs=(str(df_path), risk_kwargs, prop_kwargs, str(tmp_dir)),
            # Explicit "spawn" rather than the platform default (fork on
            # Linux/macOS): run_search() is routinely called from a
            # background thread rather than a process's main thread -- the
            # desktop GUI's Search Lab tab and the web app's search job
            # both do this so the UI/HTTP response isn't blocked for
            # minutes. forking a multi-threaded process is documented as
            # unsafe (can deadlock if another thread held a lock at fork
            # time) and Python 3.12+ warns about exactly this. spawn avoids
            # the hazard entirely -- see _warm_up_pool below for why its
            # startup cost is NOT "negligible" on a large dataset, contrary
            # to what this comment used to say.
            mp_context=multiprocessing.get_context("spawn"),
        )
        _warm_up_pool(pool, workers, len(dev_df), log)
        return pool

    pool_box = [_make_pool()]
    try:
        # NOTE: this used to be `with ProcessPoolExecutor(...) as pool:`.
        # It's now a plain owned-and-shut-down-in-finally pool, held in a
        # 1-element list (`pool_box`) rather than a bare local, because
        # _drain_futures can replace it mid-stage (kills a wedged pool and
        # spawns a fresh one) when it detects a real stall -- see
        # _drain_futures' docstring. The `if True:` below is just keeping
        # the original with-block's indentation so this diff stays
        # reviewable; it has no effect on control flow.
        if True:
            # ---------------- Stage 1: cheap filter ----------------
            # Candidates are submitted in chunks, each evaluated by
            # _stage1_task_batch: eligible ones (fixed-pips stop/target
            # only) run through one vectorized pass over the bars per
            # chunk instead of one full Python bar-loop per candidate --
            # see app.backtest.vectorized_fastpath. Anything ineligible
            # (or any chunk that itself errors) falls back to the exact
            # same per-candidate scalar path this used before. Chunk size
            # is capped so the vectorized pass's per-chunk memory
            # footprint (roughly bars x chunk_size) stays bounded even on
            # very large candidate pools.
            items = list(space.candidates.items())
            chunk_size = max(1, min(40, -(-len(items) // max(workers * 4, 1))))
            chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
            log(
                f"Stage 1/5: cheap filter across {len(items)} candidate(s) "
                f"({len(chunks)} batch(es) of up to {chunk_size}) on {workers} worker(s)..."
            )
            futures = {pool_box[0].submit(_stage1_task_batch, chunk, filters): i for i, chunk in enumerate(chunks)}
            done_batches = 0
            log_every = max(1, len(futures) // 10)

            def _on_stage1_done(_label, fut):
                nonlocal done_batches
                if fut is None:
                    # Stall recovery skipped this whole batch -- see
                    # _drain_futures. These candidates are simply left out
                    # of stage1_records (not scored, not passed) rather
                    # than crashing the run; the log line the recovery
                    # itself emits already explains why.
                    skipped_ids = [cid for cid, _spec in chunks[_label]]
                    log(f"  Stage 1: batch {_label} skipped ({len(skipped_ids)} candidate(s) not scored).")
                    done_batches += 1
                    return
                recs = fut.result()
                for rec in recs:
                    rec["family"] = space.meta.get(rec["candidate_id"], {}).get("family", space.family or "single")
                    stage1_records.append(rec)
                    db.insert_candidate(run_id, rec["candidate_id"], "stage1", rec)
                done_batches += 1
                if done_batches % log_every == 0 or done_batches == len(futures):
                    log(f"  Stage 1: {done_batches}/{len(futures)} batch(es) evaluated ({len(stage1_records)}/{len(items)} candidates)...")

            drain_futures(pool_box, futures, _on_stage1_done, pool_factory=_make_pool)
            check_cancelled(pool_box[0])

            passed_stage1 = [r for r in stage1_records if r.get("passed_stage1")]
            diversity_dropped = 0
            if stage_cfg.max_per_family_stage1:
                passed_stage1, _dropped = enforce_family_diversity(
                    passed_stage1, stage_cfg.max_per_family_stage1, score_key="quick_score",
                )
                diversity_dropped = len(_dropped)
            survivors1 = sorted(
                passed_stage1,
                key=lambda r: r.get("quick_score", 0.0), reverse=True,
            )[: stage_cfg.stage1_top_n]
            log(
                f"Stage 1 complete: {len(survivors1)}/{len(stage1_records)} candidate(s) survived "
                f"the cheap filter and advance to Stage 2 (GA refinement)."
                + (f" ({diversity_dropped} further dropped by the family-diversity cap.)" if diversity_dropped else "")
            )
            stage1_triage = aggregate_failure_reasons(
                stage1_records, "Stage 1", "passed_stage1",
                min_trades=stage_cfg.min_trades, min_profit_factor=stage_cfg.min_profit_factor,
            )
            for line in stage1_triage.format_log_lines():
                log(line)
            # w10-astra B3: counted pre-screen skips (kept next to the triage
            # so "0 survivors" is never confused with "0 candidates scored").
            prescreen_skipped = sum(1 for r in stage1_records if r.get("prescreen_skipped"))
            if prescreen_skipped:
                log(
                    f"  Stage 1: {prescreen_skipped} candidate(s) skipped by the funding "
                    f"pre-screen (risk_value too small to afford 1 contract at their "
                    f"median stop) -- no backtest was run for them."
                )
            mismatch_count = sum(1 for r in stage1_records if r.get("scale_mismatch_warning"))
            if stage1_records and mismatch_count / len(stage1_records) >= 0.25:
                log(
                    f"  ** LIKELY ROOT CAUSE: {mismatch_count}/{len(stage1_records)} candidate(s) "
                    f"({100.0 * mismatch_count / len(stage1_records):.0f}%) triggered a pip-size / "
                    f"instrument-scale mismatch warning during execution (configured pip_size = "
                    f"{risk.pip_size}). "
                    "This almost always means every family failed for the SAME underlying reason -- "
                    "not that no strategy has an edge. Go to the Data tab and click \"detect pip size "
                    "from data\" (or set it manually: ~0.01 for stocks/indices/JPY pairs/gold, 0.0001 "
                    "for most other FX pairs) and run Speed Run again before trusting a 'no winner' "
                    "result on this instrument."
                )
            if not survivors1:
                # Auto-relax: the original filters found nothing to work
                # with at all, which throws away the whole search rather
                # than giving Stage 2's GA a weaker starting point to try
                # to improve. Re-apply progressively looser thresholds
                # against the SAME already-computed Stage 1 statistics --
                # no re-backtesting -- so a strict default doesn't turn an
                # otherwise-viable search into a dead end. Stage 3's
                # validation gate is unaffected and stays exactly as
                # strict, so a candidate that only got through on relaxed
                # Stage 1 filters still has to earn its way past Stage 3
                # on its own merits.
                relax_rounds = [
                    {
                        "min_trades": max(5, stage_cfg.min_trades // 2),
                        "min_profit_factor": max(0.9, stage_cfg.min_profit_factor * 0.85),
                        "max_drawdown_buffer_mult": stage_cfg.max_drawdown_buffer_mult * 1.5,
                        "require_positive_net": True,
                    },
                    {
                        "min_trades": max(3, stage_cfg.min_trades // 4),
                        "min_profit_factor": 0.7,
                        "max_drawdown_buffer_mult": stage_cfg.max_drawdown_buffer_mult * 2.5,
                        "require_positive_net": False,
                    },
                ]
                relaxed_note = None
                for round_idx, relaxed in enumerate(relax_rounds, start=1):
                    max_dd_limit = prop_rules.max_drawdown_pct * relaxed["max_drawdown_buffer_mult"]
                    candidates_now = [
                        r for r in stage1_records
                        if _passes_stage1_filters(
                            r.get("statistics") or {}, relaxed["min_trades"], relaxed["min_profit_factor"],
                            max_dd_limit, relaxed["require_positive_net"],
                        )
                    ]
                    if candidates_now:
                        if stage_cfg.max_per_family_stage1:
                            candidates_now, _ = enforce_family_diversity(
                                candidates_now, stage_cfg.max_per_family_stage1, score_key="quick_score",
                            )
                        survivors1 = sorted(
                            candidates_now, key=lambda r: r.get("quick_score", 0.0), reverse=True,
                        )[: stage_cfg.stage1_top_n]
                        relaxed_note = (
                            f"Stage 1's original filters (min {stage_cfg.min_trades} trades, "
                            f"profit factor >= {stage_cfg.min_profit_factor:.2f}) found nothing, so it "
                            f"automatically loosened to min {relaxed['min_trades']} trades, profit factor "
                            f">= {relaxed['min_profit_factor']:.2f}"
                            + (", net profit not required" if not relaxed["require_positive_net"] else "")
                            + f" (auto-relax round {round_idx}/{len(relax_rounds)}) and found "
                            f"{len(survivors1)} candidate(s) to refine. These start weaker than the "
                            f"original filters wanted -- Stage 2's GA will try to tune them into "
                            f"something real, and Stage 3's validation gate is unchanged and still strict."
                        )
                        log(f"  Auto-relax round {round_idx}: loosened Stage 1 filters -> "
                            f"{len(survivors1)} candidate(s) now advance.")
                        break
                    log(f"  Auto-relax round {round_idx}: still 0 candidates even with loosened filters.")
                if relaxed_note:
                    log(relaxed_note)
            if not survivors1:
                n_scored = len(stage1_records)
                n_skipped = len(items) - n_scored
                if n_scored == 0:
                    # Every single candidate was skipped before scoring --
                    # see the same distinction now made in
                    # app.search.failure_triage.FailureTriageSummary. This
                    # is a worker-pool problem (a stall/crash that ate every
                    # batch), not evidence about the strategy space, and
                    # must never be reported in the same breath as "the
                    # search space has no edge".
                    log(
                        f"STOPPED: 0 of {len(items):,} candidate(s) were actually scored in Stage 1 -- "
                        "every batch was skipped before it could run. This means the worker pool never "
                        "successfully evaluated a single candidate (look above for 'batch ... skipped' "
                        "or stall/terminate messages), most likely because this dataset is large enough "
                        "that a worker stalled or ran out of memory before finishing even one batch. "
                        "This is NOT evidence that no strategy can pass a prop eval on this data -- the "
                        "search never actually ran. Next steps: (1) re-run with fewer parallel workers "
                        "(Search settings -> Workers) so each one has more memory headroom, (2) re-run "
                        "on a shorter slice of the data first to confirm scoring works at all, then scale "
                        "back up, or (3) check Task Manager/Activity Monitor during the run for memory "
                        "pressure or a worker process disappearing."
                    )
                    db.finish_run(run_id, status="worker_failure")
                    return SearchSummary(
                        run_id, space.mode, space.family, len(space.candidates), 0, 0, 0,
                        None, time.time() - t0, str(db_path), [],
                        locked_holdout_frac=stage_cfg.locked_holdout_frac,
                        dev_bars=len(dev_df), locked_bars=len(locked_df),
                    )
                log(
                    f"No candidates survived Stage 1 even after automatically loosening the filters "
                    f"twice -- nothing to refine or validate. {n_scored:,} candidate(s) WERE actually "
                    f"scored (this is a real result about the search space, not a worker problem)"
                    + (f", {n_skipped:,} were skipped due to worker-pool issues (see above)" if n_skipped else "")
                    + ". This means the search space itself (the strategy family, or the strategy you "
                    "provided) doesn't produce a workable number of trades on this data, not just a "
                    "strictness setting. Concrete next steps: (1) try a different family in Strategy "
                    "Space (some families assume a session/volatility regime this instrument/timeframe "
                    "doesn't have), (2) confirm pip_size and timeframe match this data (see the pip-size "
                    "mismatch check above if it fired), (3) widen n_hypotheses / stage1_top_n so more of "
                    "the space gets a chance, or (4) check the Strategy Graveyard for this instrument -- "
                    "if every past run also died here, the data itself (not the search settings) is "
                    "the more likely cause."
                )
                db.finish_run(run_id, status="no_survivors")
                return SearchSummary(
                    run_id, space.mode, space.family, len(space.candidates), 0, 0, 0,
                    None, time.time() - t0, str(db_path), [],
                    locked_holdout_frac=stage_cfg.locked_holdout_frac,
                    dev_bars=len(dev_df), locked_bars=len(locked_df),
                )

            # ---------------- Stage 2: GA refinement ----------------
            log(f"Stage 2/5: genetic-algorithm refinement on {len(survivors1)} surviving skeleton(s)...")
            refine_kwargs = {
                "population": stage_cfg.ga_population, "generations": stage_cfg.ga_generations,
                "cost_stress_enabled": stage_cfg.cost_stress_enabled,
                "cost_stress_multiplier": stage_cfg.cost_stress_multiplier,
                "cost_stress_penalty_weight": stage_cfg.cost_stress_penalty_weight,
                "reset_on_breach": stage_cfg.reset_on_breach,
                "optimizer_mode": stage_cfg.optimizer_mode,
                "plateau_robust_selection": stage_cfg.plateau_robust_selection,
                "plateau_finalist_pool": stage_cfg.plateau_finalist_pool,
                # ----- v5 B1-1 grammar candidate hook: passthrough (BEGIN) -----
                # Carried so _stage2_task's worker-side code (and any future
                # grammar-aware refinement) can see which source mode this
                # run uses. The actual grammar draw happens in run_search's
                # Stage 2 section below, not in the worker.
                "candidate_source": stage_cfg.candidate_source,
                "grammar_candidates_per_survivor": stage_cfg.grammar_candidates_per_survivor,
                # ----- v5 B1-1 grammar candidate hook: passthrough (END) -----
            }
            futures = {
                pool_box[0].submit(
                    _stage2_task, r["candidate_id"], _spec_from_record(r), refine_kwargs,
                    stage_cfg.ga_search_sims, stage_cfg.fitness_metric, stage_cfg.random_seed,
                ): r["candidate_id"]
                for r in survivors1
            }
            # ----- v5 B1-1 grammar candidate hook: draw (BEGIN) -----
            # When candidate_source == "grammar", draw fresh candidates from
            # the compositional grammar and refine each through the SAME
            # Stage 2 GA task as the template survivors. Additive only:
            # every template survivor above is still refined exactly as
            # before; candidate_source == "templates" (the default) skips
            # this block entirely. Drawn in the parent process (grammar
            # generation is cheap); the expensive GA refinement still runs
            # in the worker pool via _stage2_task.
            if stage_cfg.candidate_source == "grammar":
                import random as _random

                from app.search.grammar import generate_random as _grammar_generate_random
                from app.search.grammar import building_block_pool as _grammar_building_block_pool

                # A3 (v6 W1): seed the grammar draws from the template
                # building-block pool (proven ingredients), not pure
                # random terminals -- built once per run, not per draw.
                try:
                    _grammar_block_pool = _grammar_building_block_pool()
                except Exception:  # noqa: BLE001 -- a pool that can't be built is a miss, not a run failure
                    _grammar_block_pool = None

                _grammar_rng = _random.Random(stage_cfg.random_seed + 0x6A4D4D41)
                _n_grammar_each = max(0, int(stage_cfg.grammar_candidates_per_survivor))
                for _s in survivors1:
                    for _k in range(_n_grammar_each):
                        try:
                            _g_config = _grammar_generate_random(rng=_grammar_rng, block_pool=_grammar_block_pool)
                        except Exception:
                            continue  # a failed draw is a miss, not a run failure
                        _g_cid = f"grammar-s2-{_s['candidate_id']}-{_k}"
                        _g_spec = {"source_type": "manual", "config": _g_config}
                        futures[pool_box[0].submit(
                            _stage2_task, _g_cid, _g_spec, refine_kwargs,
                            stage_cfg.ga_search_sims, stage_cfg.fitness_metric, stage_cfg.random_seed,
                        )] = _g_cid
                if _n_grammar_each:
                    log(f"  Stage 2: +{len(survivors1) * _n_grammar_each} grammar-drawn candidate(s) "
                        f"queued for GA refinement (candidate_source='grammar').")
            # ----- v5 B1-1 grammar candidate hook: draw (END) -----
            done = 0
            stage2_stalled = 0

            def _on_stage2_done(_label, fut):
                nonlocal done, stage2_stalled
                if fut is None:
                    log(f"  Stage 2: candidate {_label} skipped (stall recovery -- see log above).")
                    done += 1
                    stage2_stalled += 1
                    return
                rec = fut.result()
                rec["family"] = space.meta.get(rec["candidate_id"], {}).get("family", space.family or "single")
                stage2_records.append(rec)
                db.insert_candidate(run_id, rec["candidate_id"], "stage2", rec)
                done += 1
                log(f"  Stage 2: {done}/{len(survivors1)} skeleton(s) refined...")

            drain_futures(pool_box, futures, _on_stage2_done, pool_factory=_make_pool)
            check_cancelled(pool_box[0])

            survivors2 = sorted(
                (r for r in stage2_records if r.get("passed_stage2") and math.isfinite(r.get("fitness", float("-inf")))),
                key=lambda r: r.get("fitness", float("-inf")), reverse=True,
            )[: stage_cfg.stage2_top_n]
            log(f"Stage 2 complete: {len(survivors2)} candidate(s) advance to the Stage 3 validation gate.")
            if not survivors2:
                log("No candidates survived Stage 2 refinement.")
                db.finish_run(run_id, status="no_survivors")
                return SearchSummary(
                    run_id, space.mode, space.family, len(space.candidates), len(survivors1), 0, 0,
                    None, time.time() - t0, str(db_path), [],
                    locked_holdout_frac=stage_cfg.locked_holdout_frac,
                    dev_bars=len(dev_df), locked_bars=len(locked_df),
                )

            # ---------------- Stage 3: validation gate ----------------
            log(
                f"Stage 3/5: validation gate (full Monte Carlo, walk-forward, lookahead check, "
                f"cost-ladder stress, parameter-neighborhood robustness) on {len(survivors2)} candidate(s)..."
            )
            stage3_cfg = {
                "full_mc_sims": stage_cfg.full_mc_sims, "random_seed": stage_cfg.random_seed,
                "fitness_metric": stage_cfg.fitness_metric,
                "stage3_min_trades": stage_cfg.stage3_min_trades,
                "stage3_min_profit_factor": stage_cfg.stage3_min_profit_factor,
                "stage3_max_drawdown_buffer_mult": stage_cfg.stage3_max_drawdown_buffer_mult,
                "stage3_require_positive_net": stage_cfg.stage3_require_positive_net,
                # P1-4 acceptance floor (enforced in _stage3_task's `passed`).
                "min_eval_pass_probability": stage_cfg.min_eval_pass_probability,
                "min_first_payout_probability": stage_cfg.min_first_payout_probability,
                "walk_forward_folds": stage_cfg.walk_forward_folds,
                "walk_forward_metric": stage_cfg.walk_forward_metric,
                "walk_forward_min_efficiency": stage_cfg.walk_forward_min_efficiency,
                "robustness_neighbors": stage_cfg.robustness_neighbors,
                "robustness_perturbation_frac": stage_cfg.robustness_perturbation_frac,
                "robustness_min_stability": stage_cfg.robustness_min_stability,
                "reset_on_breach": stage_cfg.reset_on_breach,
            }
            futures = {
                pool_box[0].submit(_stage3_task, r["candidate_id"], _spec_from_record(r), stage3_cfg): r["candidate_id"]
                for r in survivors2
            }
            done = 0

            stage3_stalled = 0

            def _on_stage3_done(_label, fut):
                nonlocal done, stage3_stalled
                if fut is None:
                    log(f"  Stage 3: candidate {_label} skipped (stall recovery -- see log above).")
                    done += 1
                    stage3_stalled += 1
                    return
                rec = fut.result()
                rec["family"] = space.meta.get(rec["candidate_id"], {}).get("family", space.family or "single")
                stage3_records.append(rec)
                done += 1
                log(f"  Stage 3: {done}/{len(survivors2)} candidate(s) validated...")

            drain_futures(pool_box, futures, _on_stage3_done, pool_factory=_make_pool)
            check_cancelled(pool_box[0])
            stage3_triage = aggregate_failure_reasons(
                stage3_records, "Stage 3", "passed_stage3_gate",
                min_trades=stage_cfg.stage3_min_trades, min_profit_factor=stage_cfg.stage3_min_profit_factor,
            )
            for line in stage3_triage.format_log_lines():
                log(line)
    except SearchCancelled:
        db.finish_run(run_id, status="cancelled")
        db.close()
        raise
    finally:
        try:
            pool_box[0].shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---------------- Stage 4: deflate, rank, persist ----------------
    log("Stage 4/5: computing deflated Sharpe ratios and ranking the leaderboard...")
    trial_sharpes = [r.get("sharpe", 0.0) for r in stage1_records if isinstance(r.get("sharpe"), (int, float))]
    n_trials = len(stage1_records)

    # D4(c): honest per-stage trial counts on every validated (Stage 3)
    # record, so a downstream pipeline DSR can deflate against the
    # search's REAL multiple-testing burden via
    # app.search.robustness.count_all_trials -- every backtested
    # configuration counts as a trial, not just the survivors that
    # reached this stage. See _stage_eval_counts() for the field
    # semantics.
    _eval_counts = _stage_eval_counts(stage1_records, stage2_records, stage3_records)

    for rec in stage3_records:
        rec["stage_eval_counts"] = dict(_eval_counts)
        sharpe = rec.get("sharpe", 0.0) or 0.0
        n_trade_returns = (rec.get("statistics") or {}).get("total_trades", 0)
        dsr = deflated_sharpe_ratio(sharpe, trial_sharpes, n_trials, n_trade_returns)
        rec["deflated_sharpe"] = {
            "observed_sharpe": dsr.observed_sharpe, "benchmark_sharpe": dsr.benchmark_sharpe,
            "probabilistic_sharpe": dsr.probabilistic_sharpe, "is_significant": dsr.is_significant,
            "n_trials": dsr.n_trials, "note": dsr.note,
        }
        mc = rec.get("mc_summary") or {}
        if rec.get("passed_stage3_gate"):
            # P0-1: rank on per-attempt (single-account) odds, not the
            # chain-level "did >=1 rebuy attempt ever pass" number -- the
            # chain-level fields stay in mc_summary for reporting only.
            rec["composite_score"] = (
                mc.get("per_attempt_pass_probability", mc.get("evaluation_pass_probability", 0.0)) * 0.35
                + mc.get("per_attempt_payout_probability", mc.get("first_payout_probability", 0.0)) * 0.25
                - mc.get("risk_of_ruin_pct", 0.0) * 0.15
                + dsr.probabilistic_sharpe * 100 * 0.25
            )
        else:
            rec["composite_score"] = -1.0
        db.insert_candidate(run_id, rec["candidate_id"], "stage3", rec)

    leaderboard = db.leaderboard(run_id, stage="stage3", top_n=25, only_passed=False)
    champion = next((r for r in leaderboard if r.get("passed_stage3_gate")), None)
    db.finish_run(run_id, status="completed")
    db.close()

    _record_search_candidates_to_dashboard(stage3_records, instrument, timeframe, space.family)
    graveyard_path = _write_search_graveyard_entries(stage3_records, instrument, timeframe)

    elapsed = time.time() - t0
    n_passed = sum(1 for r in stage3_records if r.get("passed_stage3_gate"))
    log(
        f"Stage 5/5: search complete in {elapsed:.1f}s. "
        f"{n_passed}/{len(stage3_records)} candidate(s) passed every Stage 3 gate."
    )
    if champion:
        log(
            f"Champion candidate: {champion['candidate_id']} "
            f"(composite score {champion.get('composite_score', 0):.2f}, "
            f"PSR {champion.get('deflated_sharpe', {}).get('probabilistic_sharpe', 0):.2f})."
        )
    else:
        log(
            "No candidate passed every Stage 3 gate. That is an honest, useful result -- "
            "'nothing in this search beats the deflated chance benchmark' is a real finding, "
            "not a failed run. See the leaderboard for the closest calls and why each one failed."
        )

    return SearchSummary(
        run_id=run_id, mode=space.mode, family=space.family, total_candidates=len(space.candidates),
        stage1_survivors=len(survivors1), stage2_survivors=len(survivors2), stage3_survivors=len(stage3_records),
        champion_candidate_id=champion["candidate_id"] if champion else None,
        elapsed_seconds=elapsed, db_path=str(db_path), leaderboard=leaderboard,
        graveyard_path=str(graveyard_path) if graveyard_path else None,
        locked_holdout_frac=stage_cfg.locked_holdout_frac,
        dev_bars=len(dev_df), locked_bars=len(locked_df),
        stage2_stalled_skipped=stage2_stalled, stage3_stalled_skipped=stage3_stalled,
    )


# ---------------------------------------------------------------------------
# Stage 5: champion promotion
# ---------------------------------------------------------------------------

def promote_champion(
    db_path: str, run_id: str, candidate_id: str, df: pd.DataFrame,
    risk: RiskConfig, prop_rules: PropRules, output_dir: str, mc_sims: int = 10000,
    reset_on_breach: bool = False,
    locked_df: pd.DataFrame | None = None,
    locked_holdout_frac: float = 0.2,
) -> dict:
    """
    Re-runs one chosen Stage 3 survivor through the app's EXISTING,
    trusted single-strategy report pipeline (the identical
    generate_full_report() call the normal Run & Report tab uses), plus a
    fresh full-dataset holdout check -- so the champion graduates into the
    exact same report format every other strategy in this app already
    produces, rather than a search-specific artifact nobody's used to
    reading yet.

    P1-4: BEFORE the report, the champion is first evaluated on the
    locked out-of-sample holdout slice -- the last `locked_holdout_frac`
    of bars, which run_search()'s Stages 0-3 never saw -- and those
    holdout stats are returned alongside as `locked_holdout`. Pass
    `locked_df` explicitly when you have run_search()'s exact reserved
    slice; otherwise it is re-derived as the tail `locked_holdout_frac`
    of `df` (same chronological split run_search used).
    """
    with ResultsDB(db_path) as db:
        record = db.get_candidate(candidate_id, run_id=run_id, stage="stage3")
    if record is None:
        raise ValueError(f"Candidate '{candidate_id}' not found in run '{run_id}' at stage3.")

    spec = _spec_from_record(record)
    source_type = spec.get("source_type", "manual")
    if source_type == "manual" and not spec.get("config"):
        raise ValueError(f"Candidate '{candidate_id}' has no stored configuration to promote.")
    if source_type != "manual" and not spec.get("code_text"):
        raise ValueError(f"Candidate '{candidate_id}' has no stored source code to promote.")

    # FIX (2026-09-18): this re-validation backtest never hardened `risk`
    # against `prop_rules` at all (unlike run_search above), so a promoted
    # champion's final report could run with no account-blown/daily-loss
    # circuit breaker, or with one but no reset_on_breach even though the
    # candidate was searched/scored with it on -- a different account than
    # the one Stage 1-3 actually found this candidate under. See
    # RiskConfig.reset_on_breach's own docstring.
    risk = with_prop_safety_defaults(risk, prop_rules)
    risk = replace(risk, reset_on_breach=reset_on_breach)

    promote_tmp_dir = Path(tempfile.mkdtemp(prefix="t58_promote_"))
    try:
        strategy = build_strategy_from_spec(spec, promote_tmp_dir)

        # P1-4: locked-holdout FIRST evaluation -- data Stages 0-3 never
        # saw. Re-derive the tail slice when the caller didn't pass the
        # exact reserved one (same chronological split as run_search).
        if locked_df is None:
            _n = len(df)
            _split = max(1, min(int(_n * (1 - locked_holdout_frac)), _n - 1)) if _n > 1 else _n
            locked_df = df.iloc[_split:].reset_index(drop=True)
        locked_holdout: dict | None = None
        if len(locked_df):
            try:
                locked_bt = run_backtest(
                    locked_df, build_strategy_from_spec(spec, promote_tmp_dir), risk)
                locked_trades = locked_bt.trades
                locked_single = simulate_account(
                    [t.pnl for t in locked_trades], [t.entry_time for t in locked_trades],
                    prop_rules, reset_on_breach=reset_on_breach,
                )
                locked_mc = run_monte_carlo(
                    locked_trades, prop_rules,
                    MonteCarloConfig(n_simulations=mc_sims, reset_on_breach=reset_on_breach),
                )
                locked_holdout = {
                    "n_bars": len(locked_df),
                    "n_trades": len(locked_trades),
                    "statistics": locked_bt.statistics.to_dict(),
                    "single_run": summarize_single_run(locked_single),
                    "evaluation_pass_probability": locked_mc.evaluation_pass_probability,
                    "per_attempt_pass_probability": locked_mc.per_attempt_pass_probability,
                    "first_payout_probability": locked_mc.first_payout_probability,
                    "per_attempt_payout_probability": locked_mc.per_attempt_payout_probability,
                    "risk_of_ruin_pct": locked_mc.risk_of_ruin_pct,
                    "n_simulations": locked_mc.n_simulations,
                    "methodology_note": locked_mc.methodology_note,
                }
            except Exception:  # noqa: BLE001 -- same policy as the holdout check below
                locked_holdout = None

        bt_result = run_backtest(df, strategy, risk)
        trade_pnls = [t.pnl for t in bt_result.trades]
        trade_dates = [t.entry_time for t in bt_result.trades]
        single_run = simulate_account(trade_pnls, trade_dates, prop_rules, reset_on_breach=reset_on_breach)
        mc_cfg = MonteCarloConfig(n_simulations=mc_sims, reset_on_breach=reset_on_breach)
        mc_result = run_monte_carlo(bt_result.trades, prop_rules, mc_cfg)
        try:
            holdout = run_holdout_comparison(
                df, build_strategy_from_spec(spec, promote_tmp_dir), risk, holdout_frac=0.2,
            )
        except Exception:  # noqa: BLE001 -- a holdout that can't run isn't a reason to block promotion
            holdout = None

        if source_type == "manual":
            strategy_name = spec["config"].get("name", f"Search Champion {candidate_id}")
        else:
            strategy_name = f"Search Champion {candidate_id} ({source_type})"

        period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
        paths = generate_full_report(
            output_dir=output_dir,
            strategy_name=strategy_name,
            strategy_source_type=source_type,
            instrument="search-lab",
            timeframe="unknown",
            backtest_period=period,
            backtest_result=bt_result,
            prop_rules=prop_rules,
            prop_single_run=single_run,
            monte_carlo_result=mc_result,
            holdout_comparison=holdout,
            risk_config=risk,
            price_df=df,
        )
        return {
            "candidate_id": candidate_id, "spec": spec, "config": spec.get("config"),
            "report_paths": paths, "locked_holdout": locked_holdout,
        }
    finally:
        shutil.rmtree(promote_tmp_dir, ignore_errors=True)


def save_search_candidate_to_library(
    db_path: str, run_id: str, candidate_id: str, library_status: str = "draft",
    filename_prefix: str = "searchlab",
) -> dict:
    """UPGRADE (search-lab-has-no-save-to-library): the missing
    counterpart to promote_champion above -- that function re-validates a
    Stage 3 survivor and writes a REPORT, but was never wired to actually
    save the candidate's own strategy definition anywhere a person could
    find and reuse it afterward (Search Lab's run/report files, unlike
    Evolution Lab's, aren't the Strategy Library). Mirrors app.evolution.
    engine.EvolutionRunner._maybe_save_to_library's exact shape (a manual
    config as pretty-printed JSON, or python/pinescript/mql5 source as-is;
    same "search-lab" tag convention as that function's "evolution-lab"
    tag) so a candidate saved through either path is found and filtered
    the same way in the Strategy Library / Dashboard afterward.

    Works identically for a single-instrument or multi-instrument Search
    Lab run -- both write through the SAME ResultsDB/stage3 record shape
    (see app.search.results_db.ResultsDB.get_candidate and
    _spec_from_record above), so this function needs no separate
    multi-instrument variant; a multi-instrument run's per-instrument
    `db_path` is passed exactly as a single-instrument run's is.

    Raises ValueError for a candidate not found at stage3, or with no
    stored config/source to save (same conditions promote_champion
    already checks) -- and re-raises StrategyAlreadyExists (from
    app.strategy.library) rather than silently overwriting an existing
    file, since a candidate ID is only unique within its own run, not
    globally across every run's saves.

    Returns {"path": <saved file path>, "filename": ..., "strategy_type": ...}."""
    with ResultsDB(db_path) as db:
        record = db.get_candidate(candidate_id, run_id=run_id, stage="stage3")
    if record is None:
        raise ValueError(f"Candidate '{candidate_id}' not found in run '{run_id}' at stage3.")

    spec = _spec_from_record(record)
    source_type = spec.get("source_type", "manual")
    family = record.get("family") or "strategy"
    base_name = f"{filename_prefix}_{run_id[:10]}_{family}_{candidate_id[-6:]}"

    if source_type == "manual":
        config = spec.get("config")
        if not config:
            raise ValueError(f"Candidate '{candidate_id}' has no stored configuration to save.")
        text = json.dumps(config, indent=2)
        filename = f"{base_name}.json"
        strategy_type = "manual"
    else:
        code_text = spec.get("code_text")
        if not code_text:
            raise ValueError(f"Candidate '{candidate_id}' has no stored source code to save.")
        text = code_text
        extension = spec.get("code_extension") or {"python": "py", "pinescript": "pine", "mql5": "mq5"}.get(source_type, "txt")
        filename = f"{base_name}.{extension.lstrip('.')}"
        strategy_type = source_type

    saved_path = save_strategy_text(text, filename, strategy_type, overwrite=False)
    set_strategy_status(strategy_type, filename, library_status)
    save_strategy_metadata(
        strategy_type, filename,
        {
            "tags": ["search-lab"],
            "description": f"Search Lab run {run_id}, family '{family}', candidate {candidate_id}",
        },
        merge=True,
    )
    return {"path": saved_path, "filename": filename, "strategy_type": strategy_type}
