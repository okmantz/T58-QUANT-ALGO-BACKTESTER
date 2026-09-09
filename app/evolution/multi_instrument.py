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
    ):
        self.group_id = group_id
        self.jobs = jobs
        self._lock = threading.Lock()
        self.runners: dict[str, EvolutionRunner] = {}
        self.logs: dict[str, list[str]] = {}
        self.errors: dict[str, str] = {}

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
            cfg = replace(
                base_cfg,
                checkpoint_path=str(job_dir / "checkpoint.json"),
                tested_log_path=str(job_dir / "tested_candidates.jsonl"),
                knowledge_graph_path=str(job_dir / "knowledge_graph.jsonl"),
            )
            self.logs[job.label] = [f"Loaded {len(df)} bars from {job.csv_path}."]
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

    def status(self) -> dict:
        with self._lock:
            logs_copy = {label: list(v) for label, v in self.logs.items()}
        instruments = {}
        for label, runner in self.runners.items():
            instruments[label] = {
                "running": runner.is_running,
                "generation": runner.generation,
                "leaderboard_size": len(runner.leaderboard),
                "leaderboard": [r.to_checkpoint_dict() for r in runner.leaderboard[:10]],
                "log": logs_copy.get(label, [])[-100:],
            }
        for label, err in self.errors.items():
            instruments[label] = {"running": False, "error": err}
        return {
            "labels": [j.label for j in self.jobs],
            "running": self.is_running,
            "instruments": instruments,
        }
