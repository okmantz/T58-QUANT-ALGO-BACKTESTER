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
