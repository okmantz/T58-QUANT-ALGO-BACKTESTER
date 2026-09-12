"""Search Lab "Loop Mode" -- Owen's ask: run Search Lab in a loop (like
Evolution Lab already effectively does across generations) until a
candidate actually clears his prop-firm target, or a round/time budget
runs out, rather than requiring him to manually re-launch Search Lab
after every single-shot run and eyeball the leaderboard himself.

Evolution Lab's own version of this idea is EvolutionConfig.target_eval_pass_pct
(see app.evolution.engine) -- it already loops generations internally, so
its "loop mode" is just teaching that internal loop to stop itself once a
leaderboard candidate clears the target. Search Lab has no internal
loop at all (app.search.batch_runner.run_search is a single, one-shot
Stage 1-5 pass), so its loop mode has to be a real outer loop: repeat
run_search() calls, and between rounds, decide whether to keep going as-is
(still making progress), widen the search (stalled -- broaden from one
family to every registered family, drop any family app.search.family_health
has flagged as a dead end, and raise the candidate cap), or stop (target
reached, round/time budget exhausted, or the caller cancelled).

Each round shares one accumulating strategy-graveyard file (via
run_search's own instrument+timeframe-scoped graveyard write-back --
see app.search.batch_runner._write_search_graveyard_entries) so a later
round in the SAME loop, a plain Search Lab run, or a Forge Strategy run
against the same instrument all benefit from what earlier rounds already
ruled out, instead of every round re-discovering the same dead ends from
a cold start.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.search.batch_runner import SearchCancelled, SearchStageConfig, SearchSummary, run_search
from app.search.family_health import apply_family_exclusions
from app.search.strategy_space import generate_search_space

@dataclass
class SearchLoopConfig:
    # Stop condition -- a leaderboard candidate's `target_metric` value
    # must reach this before the loop declares a winner and stops.
    target_eval_pass_pct: float = 60.0
    # "evaluation_pass_probability" is Search Lab's own Stage 3 Monte
    # Carlo pass estimate (see batch_runner's mc_summary) -- the same
    # raw, in-sample-biased number Evolution Lab's target_metric warns
    # about (see EvolutionCandidateRecord's own field comment); Search
    # Lab's Stage 3 doesn't run CPCV the way Evolution Lab does, so
    # there is no honest held-out equivalent to target here yet. Only
    # candidates that already cleared every OTHER Stage 3 gate (walk-
    # forward stability, lookahead, parameter-neighborhood robustness --
    # see require_passed_gate below) are ever considered, which is the
    # best honesty check currently available at this layer.
    target_metric: str = "evaluation_pass_probability"
    # Only count a candidate that passed EVERY Stage 3 gate (walk-forward
    # stability, no lookahead bug, parameter-neighborhood robustness) --
    # not just one that happens to have a high raw Monte Carlo number
    # despite failing everything else. Almost never a good reason to set
    # this False; kept as an escape hatch, not a recommended mode.
    require_passed_gate: bool = True

    max_rounds: int = 20
    # None = no wall-clock limit (only max_rounds / cancel_event stop it).
    # Set this to Owen's actual deadline (e.g. 3 days = 259200 seconds
    # minus whatever's already been spent) to make the loop respect a
    # real-world cutoff on its own.
    time_budget_seconds: float | None = None

    # Widen (broaden family scope + raise the candidate cap) after this
    # many CONSECUTIVE rounds with no improvement over the best value
    # seen so far in this loop.
    stall_rounds_before_widen: int = 2

    # Starting scope -- None means start already searching every family
    # (skip straight to the same scope a stall would widen to).
    starting_family: str | None = None
    starting_max_candidates: int = 200
    widened_max_candidates: int = 800
    # Below this many samples, app.search.family_health has no real
    # evidence yet either way -- passed straight through to
    # apply_family_exclusions.
    family_health_min_samples: int = 30

    seed: int = 42


@dataclass
class SearchLoopRoundResult:
    round_index: int
    family: str | None                 # None = "every family" this round
    max_candidates: int
    excluded_families: list            # dead-end families excluded this round, if any
    summary: SearchSummary
    best_value: float | None
    best_candidate_id: str | None
    widened_after_this_round: bool
    elapsed_seconds: float
    # The SearchSpace this round actually generated/searched -- kept so a
    # caller can hand this round's (summary, space) pair straight to
    # app.search.search_report.generate_search_report for the winning (or
    # most recent) round, the same way a single-shot run_search() call's
    # caller already does. Not used by the loop's own stop/widen logic.
    space: object = None


@dataclass
class SearchLoopResult:
    stopped_reason: str                # "target_reached" | "max_rounds" | "time_budget" | "cancelled" | "error"
    rounds: list = field(default_factory=list)   # list[SearchLoopRoundResult]
    winner_round: SearchLoopRoundResult | None = None
    winner_candidate_id: str | None = None
    total_elapsed_seconds: float = 0.0
    graveyard_path: str | None = None
    error: str | None = None


def _best_value_in_leaderboard(
    leaderboard: list[dict], metric: str, require_passed_gate: bool,
) -> tuple[float | None, str | None]:
    """The best (candidate_id, value) pair for `metric` across a Stage 3
    leaderboard, restricted to gate-passing rows unless told otherwise --
    see SearchLoopConfig.require_passed_gate's own docstring for why that
    default matters. Returns (None, None) if nothing qualifies."""
    best_value: float | None = None
    best_id: str | None = None
    for row in leaderboard:
        if require_passed_gate and not row.get("passed_stage3_gate"):
            continue
        if metric == "composite_score":
            value = row.get("composite_score")
        else:
            value = (row.get("mc_summary") or {}).get(metric)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_id = row.get("candidate_id")
    return best_value, best_id


def run_search_loop(
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    stage_cfg: SearchStageConfig,
    db_dir: str | Path,
    loop_cfg: SearchLoopConfig,
    instrument: str = "unknown",
    timeframe: str = "unknown",
    progress_cb=None,
    cancel_event=None,
    family_health_search_dir=None,
    family_health_evolution_dir=None,
    on_round=None,
) -> SearchLoopResult:
    """Repeats app.search.batch_runner.run_search, widening the search
    scope on stall, until a candidate's target_metric clears
    target_eval_pass_pct, max_rounds/time_budget_seconds runs out, or
    cancel_event is set. See this module's own docstring for the full
    round-to-round policy. `on_round`, if given, is called with each
    SearchLoopRoundResult right after it's appended to `rounds` -- e.g.
    so a caller driving a live status page can show the latest round's
    leaderboard without waiting for the whole loop to finish (a loop can
    run for many rounds/a long time, unlike a single run_search() call).
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    family = loop_cfg.starting_family
    max_candidates = loop_cfg.starting_max_candidates
    stall_count = 0
    best_ever: float | None = None
    winner_round: SearchLoopRoundResult | None = None
    rounds: list[SearchLoopRoundResult] = []
    graveyard_path: str | None = None
    stopped_reason = "max_rounds"

    try:
        for round_index in range(1, loop_cfg.max_rounds + 1):
            if cancel_event is not None and cancel_event.is_set():
                stopped_reason = "cancelled"
                break
            if (
                loop_cfg.time_budget_seconds is not None
                and (time.time() - t0) >= loop_cfg.time_budget_seconds
            ):
                stopped_reason = "time_budget"
                break

            excluded: list[str] = []
            if family in (None, "all"):
                families_to_search, excluded_list = apply_family_exclusions(
                    search_dir=family_health_search_dir, evolution_base_dir=family_health_evolution_dir,
                    min_samples=loop_cfg.family_health_min_samples,
                )
                # families_to_search is None both when nothing is dead-end
                # yet AND when excluding everything flagged would be unsafe
                # (see apply_family_exclusions' own docstring) -- either
                # way that means "don't actually exclude anything this
                # round," even though excluded_list may still be non-empty
                # (flagged-but-unsafe-to-drop).
                excluded = excluded_list if families_to_search is not None else []

            log(
                f"Loop round {round_index}/{loop_cfg.max_rounds}: "
                f"searching {'every family' if family in (None, 'all') else family} "
                f"(cap {max_candidates} candidates)..."
            )
            round_t0 = time.time()
            space = generate_search_space(
                mode="family", family=family, max_candidates=max_candidates,
                seed=loop_cfg.seed + round_index,
                exclude_families=set(excluded) if excluded else None,
            )
            db_path = str(db_dir / f"loop_round_{round_index:03d}.db")
            summary = run_search(
                df, risk, prop_rules, space, stage_cfg, db_path=db_path,
                instrument=instrument, timeframe=timeframe,
                progress_cb=progress_cb, cancel_event=cancel_event,
            )
            if summary.graveyard_path:
                graveyard_path = summary.graveyard_path

            best_value, best_id = _best_value_in_leaderboard(
                summary.leaderboard, loop_cfg.target_metric, loop_cfg.require_passed_gate,
            )

            improved = best_value is not None and (best_ever is None or best_value > best_ever)
            if improved:
                best_ever = best_value
                stall_count = 0
            else:
                stall_count += 1

            widened_this_round = False
            target_reached = best_value is not None and best_value >= loop_cfg.target_eval_pass_pct

            round_result = SearchLoopRoundResult(
                round_index=round_index, family=family, max_candidates=max_candidates,
                excluded_families=excluded, summary=summary,
                best_value=best_value, best_candidate_id=best_id,
                widened_after_this_round=widened_this_round,
                elapsed_seconds=time.time() - round_t0,
                space=space,
            )
            rounds.append(round_result)
            if on_round:
                on_round(round_result)

            if target_reached:
                log(
                    f"Loop round {round_index}: candidate {best_id} cleared "
                    f"{loop_cfg.target_eval_pass_pct:.0f}% on {loop_cfg.target_metric} -- stopping."
                )
                winner_round = round_result
                stopped_reason = "target_reached"
                break

            if stall_count >= loop_cfg.stall_rounds_before_widen and (
                family not in (None, "all") or max_candidates < loop_cfg.widened_max_candidates
            ):
                log(
                    f"Loop round {round_index}: no improvement for {stall_count} round(s) -- "
                    "widening to every family and raising the candidate cap for the next round."
                )
                family = None
                max_candidates = loop_cfg.widened_max_candidates
                round_result.widened_after_this_round = True
                stall_count = 0
    except SearchCancelled:
        # A round's own run_search() can raise this directly (rather than
        # this loop's own top-of-round cancel_event check catching it
        # first) if cancellation lands WHILE a round is already in flight
        # -- a deliberate stop, not a crash, so it must produce the same
        # stopped_reason="cancelled" the top-of-round check does, not fall
        # through to the generic error path below.
        stopped_reason = "cancelled"
    except Exception as exc:  # noqa: BLE001 -- surface as a clean stopped-with-error result, not a raised crash
        return SearchLoopResult(
            stopped_reason="error", rounds=rounds, winner_round=winner_round,
            winner_candidate_id=(winner_round.best_candidate_id if winner_round else None),
            total_elapsed_seconds=time.time() - t0, graveyard_path=graveyard_path, error=str(exc),
        )

    return SearchLoopResult(
        stopped_reason=stopped_reason, rounds=rounds, winner_round=winner_round,
        winner_candidate_id=(winner_round.best_candidate_id if winner_round else None),
        total_elapsed_seconds=time.time() - t0, graveyard_path=graveyard_path,
    )


# ---------------------------------------------------------------------------
# Forge Strategy "Loop Mode" -- Forge (app.orchestration.forge.run_forge) is
# already a much deeper single funnel than Search Lab's run_search (10,000
# hypotheses by default, searched across EVERY family at once, with its own
# CPCV/regime/rolling-evaluation/locked-holdout stages on top) -- so unlike
# Search Lab, "widening" on a stall isn't about broadening from one family
# to every family (Forge already searches every family every round). It's
# about searching DEEPER: more hypotheses, more Stage 1/2 survivors carried
# forward, so a stalled run explores more of the same already-full space
# instead of re-running an identical attempt and expecting a different
# result. Forge already accumulates its own strategy-graveyard file
# (run_forge's own `graveyard_path` param, via
# app.search.graveyard.is_known_dead_neighborhood) across calls given the
# same path, so passing the SAME path every round here is what makes later
# rounds actually benefit from earlier ones' dead ends, the same role
# run_search_loop's shared graveyard plays for Search Lab.
# ---------------------------------------------------------------------------

@dataclass
class ForgeLoopConfig:
    # Forge's own headline number is pass_rate_pct -- the ROLLING-EVALUATION
    # pass rate (repeated simulated eval attempts over the historical
    # window), already a more realistic estimate than a single Monte Carlo
    # pass probability. require_locked_oos_passed additionally requires the
    # champion to have cleared Forge's own locked holdout slice (data
    # reserved before any stage ran, never searched over) -- the closest
    # thing this app has to genuinely unseen data.
    target_pass_rate_pct: float = 60.0
    require_locked_oos_passed: bool = True

    max_rounds: int = 10
    time_budget_seconds: float | None = None
    stall_rounds_before_widen: int = 2

    starting_n_hypotheses: int = 10_000
    widened_n_hypotheses: int = 30_000
    starting_stage1_top_n: int = 1_000
    widened_stage1_top_n: int = 2_500
    starting_stage2_top_n: int = 200
    widened_stage2_top_n: int = 400

    family_health_min_samples: int = 30
    seed: int = 42
    # Every OTHER ForgeConfig field (Stage 1/2/3 thresholds, CPCV/rolling/
    # holdout settings, worker count, ...) is taken from this base config
    # each round, via dataclasses.replace -- only n_hypotheses,
    # stage1_top_n, stage2_top_n, exclude_families, and seed/random_seed
    # are ever overridden by the loop itself. None uses a fresh, all-
    # defaults ForgeConfig() as the base.
    base_config: object = None


@dataclass
class ForgeLoopRoundResult:
    round_index: int
    n_hypotheses: int
    excluded_families: list
    result: object                     # app.orchestration.forge.ForgeResult
    champion_pass_rate_pct: float | None
    champion_candidate_id: str | None
    widened_after_this_round: bool
    elapsed_seconds: float


@dataclass
class ForgeLoopResult:
    stopped_reason: str                # "target_reached" | "max_rounds" | "time_budget" | "cancelled" | "error"
    rounds: list = field(default_factory=list)   # list[ForgeLoopRoundResult]
    winner_round: ForgeLoopRoundResult | None = None
    winner_candidate_id: str | None = None
    total_elapsed_seconds: float = 0.0
    error: str | None = None


def _forge_champion_value(result, require_locked_oos_passed: bool) -> tuple[float | None, str | None]:
    """The champion row's pass_rate_pct -- None if Forge found no champion
    at all, or (when require_locked_oos_passed) the champion exists but its
    own locked_oos_status isn't \"PASSED\"."""
    if not result.champion_candidate_id:
        return None, None
    row = next((r for r in result.leaderboard if r.candidate_id == result.champion_candidate_id), None)
    if row is None:
        return None, None
    if require_locked_oos_passed and row.locked_oos_status != "PASSED":
        return None, row.candidate_id
    return row.pass_rate_pct, row.candidate_id


def run_forge_loop(
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    db_dir: str | Path,
    loop_cfg: ForgeLoopConfig,
    instrument: str = "unknown",
    timeframe: str = "unknown",
    progress_cb=None,
    cancel_event=None,
    family_health_search_dir=None,
    family_health_evolution_dir=None,
    on_round=None,
) -> ForgeLoopResult:
    """Repeats app.orchestration.forge.run_forge, searching deeper
    (more hypotheses, more survivors carried forward) on a stall, until a
    champion's pass_rate_pct clears target_pass_rate_pct, max_rounds/
    time_budget_seconds runs out, or cancel_event is set. See this
    module's own docstring above for the full round-to-round policy.
    `on_round`, if given, is called with each ForgeLoopRoundResult right
    after it's appended to `rounds` -- same purpose as run_search_loop's
    own `on_round` (a live status page can show the latest round without
    waiting for the whole loop, which for Forge can run for a long time).
    """
    from app.orchestration.forge import ForgeConfig, run_forge
    from app.search.graveyard import graveyard_path_for
    from dataclasses import replace as _replace

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    graveyard_path = graveyard_path_for(instrument, timeframe)
    base_config = loop_cfg.base_config if loop_cfg.base_config is not None else ForgeConfig()

    t0 = time.time()
    n_hypotheses = loop_cfg.starting_n_hypotheses
    stage1_top_n = loop_cfg.starting_stage1_top_n
    stage2_top_n = loop_cfg.starting_stage2_top_n
    stall_count = 0
    best_ever: float | None = None
    winner_round: ForgeLoopRoundResult | None = None
    rounds: list[ForgeLoopRoundResult] = []
    stopped_reason = "max_rounds"

    try:
        for round_index in range(1, loop_cfg.max_rounds + 1):
            if cancel_event is not None and cancel_event.is_set():
                stopped_reason = "cancelled"
                break
            if (
                loop_cfg.time_budget_seconds is not None
                and (time.time() - t0) >= loop_cfg.time_budget_seconds
            ):
                stopped_reason = "time_budget"
                break

            _, excluded_list = apply_family_exclusions(
                search_dir=family_health_search_dir, evolution_base_dir=family_health_evolution_dir,
                min_samples=loop_cfg.family_health_min_samples,
            )

            log(
                f"Forge Loop round {round_index}/{loop_cfg.max_rounds}: "
                f"{n_hypotheses} hypotheses, stage1_top_n={stage1_top_n}, stage2_top_n={stage2_top_n}..."
            )
            round_t0 = time.time()
            config = _replace(
                base_config,
                n_hypotheses=n_hypotheses, exclude_families=set(excluded_list) or None,
                stage1_top_n=stage1_top_n, stage2_top_n=stage2_top_n,
                seed=loop_cfg.seed + round_index, random_seed=loop_cfg.seed + round_index,
            )
            db_path = str(db_dir / f"forge_loop_round_{round_index:03d}.db")
            result = run_forge(
                df, risk, prop_rules, config, db_path=db_path,
                instrument=instrument, timeframe=timeframe, graveyard_path=graveyard_path,
                progress_cb=progress_cb, cancel_event=cancel_event,
            )

            champion_value, champion_id = _forge_champion_value(result, loop_cfg.require_locked_oos_passed)
            improved = champion_value is not None and (best_ever is None or champion_value > best_ever)
            if improved:
                best_ever = champion_value
                stall_count = 0
            else:
                stall_count += 1

            target_reached = champion_value is not None and champion_value >= loop_cfg.target_pass_rate_pct
            widened_this_round = False
            round_result = ForgeLoopRoundResult(
                round_index=round_index, n_hypotheses=n_hypotheses, excluded_families=excluded_list,
                result=result, champion_pass_rate_pct=champion_value, champion_candidate_id=champion_id,
                widened_after_this_round=widened_this_round, elapsed_seconds=time.time() - round_t0,
            )
            rounds.append(round_result)
            if on_round:
                on_round(round_result)

            if target_reached:
                log(
                    f"Forge Loop round {round_index}: candidate {champion_id} cleared "
                    f"{loop_cfg.target_pass_rate_pct:.0f}% -- stopping."
                )
                winner_round = round_result
                stopped_reason = "target_reached"
                break

            if stall_count >= loop_cfg.stall_rounds_before_widen and (
                n_hypotheses < loop_cfg.widened_n_hypotheses
            ):
                log(
                    f"Forge Loop round {round_index}: no improvement for {stall_count} round(s) -- "
                    "searching deeper (more hypotheses, more survivors carried forward) next round."
                )
                n_hypotheses = loop_cfg.widened_n_hypotheses
                stage1_top_n = loop_cfg.widened_stage1_top_n
                stage2_top_n = loop_cfg.widened_stage2_top_n
                round_result.widened_after_this_round = True
                stall_count = 0
    except SearchCancelled:
        stopped_reason = "cancelled"
    except Exception as exc:  # noqa: BLE001 -- surface as a clean stopped-with-error result, not a raised crash
        return ForgeLoopResult(
            stopped_reason="error", rounds=rounds, winner_round=winner_round,
            winner_candidate_id=(winner_round.champion_candidate_id if winner_round else None),
            total_elapsed_seconds=time.time() - t0, error=str(exc),
        )

    return ForgeLoopResult(
        stopped_reason=stopped_reason, rounds=rounds, winner_round=winner_round,
        winner_candidate_id=(winner_round.champion_candidate_id if winner_round else None),
        total_elapsed_seconds=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Speed Run "Loop Mode" -- app.orchestration.speed_run.run_speed_run is
# already a full discover-then-validate funnel (Phase 1: wide discovery
# across every family; Phase 2/3: validate the top K discoveries through
# Full Pipeline) that returns a definitive winner-or-not verdict in one
# call, unlike Search Lab's single run_search which only ever produces a
# LEADERBOARD (no "was Full Pipeline itself satisfied" check). So Speed
# Run's stop condition is simpler than either Search Lab's or Forge's:
# `result.winner is not None` -- a validated, Full-Pipeline-approved
# candidate already exists, nothing further to decide. On a stall (no
# winner found), widening means raising the candidate cap and how many
# discoveries get validated, and rotating the discovery random seed so a
# repeated round doesn't just re-search the exact same random sample of
# an unchanged grid.
# ---------------------------------------------------------------------------

@dataclass
class SpeedRunLoopConfig:
    max_rounds: int = 10
    time_budget_seconds: float | None = None
    stall_rounds_before_widen: int = 2

    starting_max_candidates: int = 1_200
    widened_max_candidates: int = 3_000
    starting_top_k_to_validate: int = 3
    widened_top_k_to_validate: int = 6

    seed: int = 42
    # Every OTHER SpeedRunConfig field (Stage 1-3 thresholds, validation
    # settings, fitness metric, ...) is taken from this base config each
    # round, via dataclasses.replace -- only max_candidates,
    # top_k_to_validate, and discovery_random_seed/random_seed are ever
    # overridden by the loop itself. None uses a fresh, all-defaults
    # SpeedRunConfig() as the base.
    base_config: object = None


@dataclass
class SpeedRunLoopRoundResult:
    round_index: int
    max_candidates: int
    top_k_to_validate: int
    result: object                     # app.orchestration.speed_run.SpeedRunResult
    found_winner: bool
    widened_after_this_round: bool
    elapsed_seconds: float


@dataclass
class SpeedRunLoopResult:
    stopped_reason: str                # "target_reached" | "max_rounds" | "time_budget" | "cancelled" | "error"
    rounds: list = field(default_factory=list)   # list[SpeedRunLoopRoundResult]
    winner_round: SpeedRunLoopRoundResult | None = None
    total_elapsed_seconds: float = 0.0
    error: str | None = None


def run_speed_run_loop(
    df: pd.DataFrame,
    risk: RiskConfig,
    prop_rules: PropRules,
    output_dir: str | Path,
    loop_cfg: SpeedRunLoopConfig,
    instrument: str = "unknown",
    progress_cb=None,
    cancel_event=None,
) -> SpeedRunLoopResult:
    """Repeats app.orchestration.speed_run.run_speed_run, raising the
    candidate cap and validation width on a stall (no winner found), until
    a round actually produces a winner, max_rounds/time_budget_seconds
    runs out, or cancel_event is set. See this module's own docstring
    above for the full round-to-round policy.

    Unlike run_search_loop/run_forge_loop, run_speed_run itself never
    raises SearchCancelled -- it already returns a clean, winner=None
    SpeedRunResult when cancelled (see its own try/except around Phase 1's
    run_search call) -- so this loop only needs its own top-of-round and
    after-round cancel_event checks, no SearchCancelled handling.
    """
    from dataclasses import replace as _replace
    from app.orchestration.speed_run import SpeedRunConfig, run_speed_run

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = loop_cfg.base_config if loop_cfg.base_config is not None else SpeedRunConfig()

    t0 = time.time()
    max_candidates = loop_cfg.starting_max_candidates
    top_k_to_validate = loop_cfg.starting_top_k_to_validate
    stall_count = 0
    winner_round: SpeedRunLoopRoundResult | None = None
    rounds: list[SpeedRunLoopRoundResult] = []
    stopped_reason = "max_rounds"

    try:
        for round_index in range(1, loop_cfg.max_rounds + 1):
            if cancel_event is not None and cancel_event.is_set():
                stopped_reason = "cancelled"
                break
            if (
                loop_cfg.time_budget_seconds is not None
                and (time.time() - t0) >= loop_cfg.time_budget_seconds
            ):
                stopped_reason = "time_budget"
                break

            log(
                f"Speed Run Loop round {round_index}/{loop_cfg.max_rounds}: "
                f"max_candidates={max_candidates}, top_k_to_validate={top_k_to_validate}..."
            )
            round_t0 = time.time()
            config = _replace(
                base_config,
                max_candidates=max_candidates, top_k_to_validate=top_k_to_validate,
                discovery_random_seed=loop_cfg.seed + round_index, random_seed=loop_cfg.seed + round_index,
            )
            round_output_dir = output_dir / f"round_{round_index:03d}"
            result = run_speed_run(
                df, risk, prop_rules, round_output_dir, config,
                progress_cb=progress_cb, instrument=instrument, cancel_event=cancel_event,
            )

            found_winner = result.winner is not None
            widened_this_round = False
            round_result = SpeedRunLoopRoundResult(
                round_index=round_index, max_candidates=max_candidates, top_k_to_validate=top_k_to_validate,
                result=result, found_winner=found_winner, widened_after_this_round=widened_this_round,
                elapsed_seconds=time.time() - round_t0,
            )
            rounds.append(round_result)

            if cancel_event is not None and cancel_event.is_set():
                stopped_reason = "cancelled"
                break

            if found_winner:
                log(f"Speed Run Loop round {round_index}: winner found -- stopping.")
                winner_round = round_result
                stopped_reason = "target_reached"
                break

            stall_count += 1
            if stall_count >= loop_cfg.stall_rounds_before_widen and (
                max_candidates < loop_cfg.widened_max_candidates
                or top_k_to_validate < loop_cfg.widened_top_k_to_validate
            ):
                log(
                    f"Speed Run Loop round {round_index}: no winner for {stall_count} round(s) -- "
                    "raising the candidate cap and validating more discoveries next round."
                )
                max_candidates = loop_cfg.widened_max_candidates
                top_k_to_validate = loop_cfg.widened_top_k_to_validate
                round_result.widened_after_this_round = True
                stall_count = 0
    except Exception as exc:  # noqa: BLE001 -- surface as a clean stopped-with-error result, not a raised crash
        return SpeedRunLoopResult(
            stopped_reason="error", rounds=rounds, winner_round=winner_round,
            total_elapsed_seconds=time.time() - t0, error=str(exc),
        )

    return SpeedRunLoopResult(
        stopped_reason=stopped_reason, rounds=rounds, winner_round=winner_round,
        total_elapsed_seconds=time.time() - t0,
    )
