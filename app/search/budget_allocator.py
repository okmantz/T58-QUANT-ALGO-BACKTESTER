"""
Search Budget Allocator -- "spread the SAME trial budget across more
instruments/timeframes instead of deepening the grid on one" (Owen's own
framing, in the Sept 2026 session that added this module).

Two problems this fixes, both about app.evolution.multi_instrument.
MultiInstrumentEvolutionGroup:

  1. Budget allocation. Before this module, starting a multi-instrument
     Evolution Lab session gave EVERY instrument/timeframe job the SAME
     population_size/max_generations as a single-instrument run -- i.e.
     running against 5 instruments cost 5x the compute of running against
     1, rather than spending the SAME total compute more broadly. That
     is the opposite of what actually helps: app.search.robustness's
     Probabilistic Sharpe Ratio / deflated Sharpe machinery penalizes a
     candidate for how many OTHER candidates were tried on the SAME
     series before it (n_trials in app.search.batch_runner is scoped per
     run) -- burying one instrument in ever-deeper parameter grids just
     multiplies how much chance-driven "best of N" noise you have to
     explain away on that one series, whereas spreading a FIXED budget
     across more series gives the search more independent chances to
     find a real, instrument-specific edge instead of over-mining one.

     allocate_search_budget() below turns one total_evaluation_budget
     (roughly population_size * max_generations -- the total number of
     candidate evaluations a run will spend) into a per-job
     population_size/max_generations pair, so N jobs together spend
     ~total_evaluation_budget regardless of how many jobs there are.

  2. Cross-instrument family diversity. app.search.family_diversity.
     enforce_family_diversity already caps how many candidates from the
     same classified family survive WITHIN one run -- but
     MultiInstrumentEvolutionGroup never applied it ACROSS the group, so
     "we searched 5 instruments" could still hand back 5 near-identical
     RSI-reversion winners, one per instrument, with no signal that
     they're all really the same idea. pooled_cross_instrument_
     leaderboard() below pools every job's own leaderboard, tags each
     record with which instrument/timeframe it came from, and applies
     the SAME family cap globally -- so the final shortlist reflects
     genuinely diverse hypotheses, not just diverse instruments.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.search.family_diversity import enforce_family_diversity

DEFAULT_MIN_POPULATION = 20
DEFAULT_MAX_POPULATION = 120
DEFAULT_POPULATION = 60


@dataclass(frozen=True)
class JobBudget:
    label: str
    population_size: int
    max_generations: int
    evaluation_budget: int  # population_size * max_generations -- what this job will actually spend


def allocate_search_budget(
    total_evaluation_budget: int,
    job_labels: list[str],
    min_population: int = DEFAULT_MIN_POPULATION,
    max_population: int = DEFAULT_MAX_POPULATION,
    default_population: int = DEFAULT_POPULATION,
) -> list[JobBudget]:
    """Splits `total_evaluation_budget` (an approximate total candidate-
    evaluation count -- population_size * max_generations, the same unit
    Evolution Lab already reports run-over-run) evenly across
    `job_labels`. Each job's population_size is pulled down toward its
    own share of the budget when the split is tight (favors running more
    generations of a smaller population over one single-generation pass
    of an oversized one, since a GA needs several generations of
    selection pressure to be worth anything), but never below
    `min_population`, and never above `max_population` or
    `default_population`. Every job's population_size *
    max_generations is <= its exact even share of the total budget,
    rounded down -- the SUM across all jobs is always <=
    total_evaluation_budget, never more, except that a job whose exact
    share is smaller than min_population still gets min_population (a
    GA can't usefully run with fewer than that many genomes per
    generation) at max_generations=1, which is the one case this
    allocator will slightly overspend the requested total rather than
    hand a job a population size too small to function."""
    if total_evaluation_budget <= 0:
        raise ValueError("total_evaluation_budget must be positive.")
    if not job_labels:
        raise ValueError("job_labels must be non-empty.")

    n = len(job_labels)
    per_job_budget = total_evaluation_budget // n
    plans = []
    for label in job_labels:
        if per_job_budget < min_population:
            population = min_population
            generations = 1
        else:
            population = max(min_population, min(default_population, per_job_budget, max_population))
            generations = max(per_job_budget // population, 1)
        plans.append(JobBudget(
            label=label, population_size=population, max_generations=generations,
            evaluation_budget=population * generations,
        ))
    return plans


def normalize_evolution_record(checkpoint_dict: dict, instrument_label: str | None = None) -> dict:
    """Adapts one app.evolution.engine.EvolutionCandidateRecord.
    to_checkpoint_dict() entry into the same flat record shape
    app.search.results_db's candidate rows already use
    (candidate_id/family/source_type/config/code_text/composite_score/
    statistics/mc_summary) -- the shape app.strategy.family_taxonomy.
    classify_record and app.ensemble.auto_builder.strategy_from_record
    both expect. `spec` (the genome) already carries source_type +
    config/code_text directly (see app.search.strategy_space.
    build_strategy_from_spec, which every EvolutionRunner candidate is
    built from the same way), so this is a re-labeling, not a
    reconstruction."""
    spec = checkpoint_dict.get("spec") or {}
    meta = checkpoint_dict.get("meta") or {}
    fitness = checkpoint_dict.get("fitness") or {}
    out = {
        "candidate_id": checkpoint_dict.get("candidate_id", ""),
        "family": meta.get("family"),
        "source_type": spec.get("source_type", "manual"),
        "config": spec.get("config"),
        "code_text": spec.get("code_text"),
        "composite_score": fitness.get("final_score") if isinstance(fitness, dict) else None,
        "statistics": checkpoint_dict.get("stats"),
        "mc_summary": checkpoint_dict.get("mc_summary"),
    }
    if instrument_label is not None:
        out["instrument"] = instrument_label
    return out


def pooled_cross_instrument_leaderboard(
    job_records: dict[str, list[dict]],
    max_per_family: int = 2,
    top_n: int = 25,
    score_key: str = "composite_score",
) -> list[dict]:
    """Pools every job's own (already-normalized, results_db-shaped)
    leaderboard records into one list, tags each with which instrument/
    timeframe it came from (unless already tagged), applies
    app.search.family_diversity.enforce_family_diversity ACROSS the
    whole pooled set (not per-instrument -- that's the point), and
    returns the top `top_n` by `score_key`. See module docstring, problem
    2, for why this is the piece MultiInstrumentEvolutionGroup was
    missing."""
    pooled: list[dict] = []
    for label, records in job_records.items():
        for rec in records:
            merged = dict(rec)
            merged.setdefault("instrument", label)
            pooled.append(merged)

    kept, _dropped = enforce_family_diversity(pooled, max_per_family=max_per_family, score_key=score_key)

    def _score(rec: dict) -> float:
        val = rec.get(score_key)
        return float(val) if isinstance(val, (int, float)) else float("-inf")

    kept.sort(key=_score, reverse=True)
    return kept[:top_n]
