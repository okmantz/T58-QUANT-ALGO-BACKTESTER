"""
Multi-instrument Evolution Lab.

Same reasoning as app.orchestration.multi_instrument_search and
app.orchestration.multi_instrument_speed_run (see either module's own
docstring): a real edge is often instrument/timeframe-dependent, so
running the same search CONCURRENTLY against several instruments covers
more ground per unit wall-clock time than one overnight run on a single
instrument. Evolution Lab's own EvolutionRunner is stateful and
checkpointed (unlike a single Search Lab/Speed Run call, which runs once
and returns), so this isn't a thin "fan out one function call" wrapper
like those two modules -- it's a small MANAGER that owns a group of
independent EvolutionRunner instances, one per instrument/timeframe, each
running on its own background thread with its own fully-isolated on-disk
checkpoint/tested-log/knowledge-graph files (never shared -- giving two
runners the same checkpoint path would have them silently corrupt each
other's resume state).

This does NOT change how any one instrument's generation loop works --
every runner in the group is a completely ordinary EvolutionRunner,
running the exact same generate/filter/validate/mutate loop it always
does. The manager's only job is starting, monitoring, and stopping the
whole group together as one unit for the web UI.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

from app.backtest.risk import RiskConfig
from app.data.importer import import_csv
from app.data.storage import get_app_base_dir
from app.evolution.engine import EvolutionConfig, EvolutionRunner
from app.prop.simulator import PropRules
from app.search.budget_allocator import (
    allocate_search_budget,
    normalize_evolution_record,
    pooled_cross_instrument_leaderboard,
)


@dataclass(frozen=True)
class EvolutionInstrumentJob:
    instrument: str
    timeframe: str
    csv_path: str

    @property
    def label(self) -> str:
        return f"{self.instrument}/{self.timeframe}"

    @property
    def slug(self) -> str:
        return f"{self.instrument}_{self.timeframe}".replace("/", "_").replace(" ", "_")


class MultiInstrumentEvolutionGroup:
    """Owns one 'multi-instrument Evolution Lab' session: a named group of
    independent EvolutionRunner instances, one per instrument/timeframe.
    A CSV that fails to load is recorded in `.errors` rather than raising
    -- one bad file must not prevent starting Evolution Lab on the rest of
    the group."""

    def __init__(
        self, group_id: str, jobs: list[EvolutionInstrumentJob], risk: RiskConfig,
        prop_rules: PropRules, base_cfg: EvolutionConfig,
        total_evaluation_budget: int | None = None,
    ):
        """
        total_evaluation_budget: optional. None (the default) is the
        EXACT prior behavior -- every job gets base_cfg's own
        population_size/max_generations unchanged, so N instruments cost
        N times the compute of one. When given, the SAME total budget
        (roughly population_size * max_generations) is split across all
        `jobs` instead (see app.search.budget_allocator.
        allocate_search_budget) -- "widen the search instead of
        deepening the grid," spending the same compute on more
        instruments/timeframes rather than multiplying it. Per-job
        max_generations is only overridden when the caller's base_cfg
        left it as None (unbounded/loop mode) or when a budget was
        given; an EXPLICIT finite max_generations on base_cfg is treated
        as a hard ceiling and never raised, only ever lowered to fit a
        job's allocated share.
        """
        self.group_id = group_id
        self.jobs = jobs
        self.total_evaluation_budget = total_evaluation_budget
        self._lock = threading.Lock()
        self.runners: dict[str, EvolutionRunner] = {}
        self.logs: dict[str, list[str]] = {}
        self.errors: dict[str, str] = {}

        budget_by_label = {}
        if total_evaluation_budget is not None:
            for plan in allocate_search_budget(total_evaluation_budget, [job.label for job in jobs]):
                budget_by_label[plan.label] = plan

        base_dir = get_app_base_dir() / "data" / "evolution" / "multi_instrument" / group_id
        for job in jobs:
            job_dir = base_dir / job.slug
            job_dir.mkdir(parents=True, exist_ok=True)
            try:
                import_result = import_csv(job.csv_path)
                if not import_result.is_valid:
                    raise ValueError("; ".join(import_result.errors) or "invalid or unreadable CSV")
                df = import_result.dataframe
            except Exception as exc:  # noqa: BLE001 -- one bad file must not sink the group
                self.errors[job.label] = f"Could not load {job.csv_path}: {exc}"
                continue
            overrides: dict = {
                "instrument": job.label,
                "checkpoint_path": str(job_dir / "checkpoint.json"),
                "tested_log_path": str(job_dir / "tested_candidates.jsonl"),
                "knowledge_graph_path": str(job_dir / "knowledge_graph.jsonl"),
            }
            plan = budget_by_label.get(job.label)
            if plan is not None:
                overrides["population_size"] = plan.population_size
                overrides["max_generations"] = min(plan.max_generations, base_cfg.max_generations) \
                    if base_cfg.max_generations is not None else plan.max_generations
            cfg = replace(base_cfg, **overrides)
            log_line = f"Loaded {len(df)} bars from {job.csv_path}."
            if plan is not None:
                log_line += (
                    f" Allocated budget: population={plan.population_size}, "
                    f"max_generations={cfg.max_generations} (~{plan.population_size * cfg.max_generations:,} "
                    f"evaluations of this group's {total_evaluation_budget:,}-total budget)."
                )
            self.logs[job.label] = [log_line]
            self.runners[job.label] = EvolutionRunner(
                df, risk, prop_rules, cfg,
                progress_cb=lambda msg, label=job.label: self._log(label, msg),
            )

    def _log(self, label: str, msg: str) -> None:
        with self._lock:
            log = self.logs.setdefault(label, [])
            log.append(msg)
            del log[:-500]

    def start_all(self) -> None:
        for runner in self.runners.values():
            runner.start()

    def stop_all(self, timeout: float = 8.0) -> bool:
        """Signals stop to every runner and waits (up to `timeout` seconds
        TOTAL across the whole group, not per-runner) for them all to
        exit. Returns True once every runner has genuinely stopped."""
        for runner in self.runners.values():
            runner.stop()
        deadline = time.time() + timeout
        for runner in self.runners.values():
            remaining = max(0.0, deadline - time.time())
            if runner._thread is not None:
                runner._thread.join(timeout=remaining)
        return not self.is_running

    @property
    def is_running(self) -> bool:
        return any(r.is_running for r in self.runners.values())

    def promote(self, label: str, candidate_id: str):
        """Delegates to the named instrument's own runner -- see
        app.web.server's single-instrument /evolution/promote route for
        the exact same logic this mirrors (leaderboard-then-checkpoint
        fallback, manual-config-only)."""
        runner = self.runners.get(label)
        if runner is None:
            return None
        for r in runner.leaderboard:
            if r.candidate_id == candidate_id:
                return r.to_checkpoint_dict()
        return None

    def pooled_leaderboard(self, top_n: int = 25, max_per_family: int = 2) -> list[dict]:
        """Every runner's own leaderboard, pooled and re-ranked with
        app.search.family_diversity's family cap applied GLOBALLY across
        the whole group instead of per-instrument -- see
        app.search.budget_allocator's module docstring, problem 2. Each
        returned record carries an 'instrument' key naming which job it
        came from. Safe to call at any time, including mid-run (reads
        each runner's current in-memory leaderboard, not a final
        result)."""
        job_records = {
            label: [normalize_evolution_record(r.to_checkpoint_dict(), instrument_label=label) for r in runner.leaderboard]
            for label, runner in self.runners.items()
        }
        return pooled_cross_instrument_leaderboard(job_records, max_per_family=max_per_family, top_n=top_n)

    def status(self) -> dict:
        with self._lock:
            logs_copy = {label: list(v) for label, v in self.logs.items()}
        instruments = {}
        for label, runner in self.runners.items():
            runner_status = runner.status()
            instruments[label] = {
                "running": runner.is_running,
                "generation": runner.generation,
                "leaderboard_size": len(runner.leaderboard),
                "leaderboard": [r.to_checkpoint_dict() for r in runner.leaderboard[:10]],
                "log": logs_copy.get(label, [])[-100:],
                # Loop mode -- see EvolutionConfig.target_eval_pass_pct. Every
                # runner in the group shares base_cfg's target settings (they
                # only differ in checkpoint/tested-log/knowledge-graph paths),
                # so each stops itself independently the moment ITS OWN
                # leaderboard clears the shared target -- one instrument
                # finishing early does not stop the others.
                "target_eval_pass_pct": runner_status.get("target_eval_pass_pct"),
                "target_reached": runner_status.get("target_reached", False),
                "target_reached_candidate_id": runner_status.get("target_reached_candidate_id"),
            }
        for label, err in self.errors.items():
            instruments[label] = {"running": False, "error": err}
        return {
            "labels": [j.label for j in self.jobs],
            "running": self.is_running,
            "instruments": instruments,
        }
