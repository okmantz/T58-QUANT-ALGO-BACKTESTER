"""
Multi-instrument Speed Run orchestration.

The exact same reasoning as app.orchestration.multi_instrument_search
(see that module's own docstring), applied to Speed Run instead of plain
Search Lab: Speed Run's own real-world track record so far (per Owen's
own notes) is a winner found overnight on Bitcoin, no winner across two
overnight runs on AAPL -- a strong signal that which INSTRUMENT you point
Speed Run at matters as much as anything else about the run itself. This
module runs the SAME SpeedRunConfig CONCURRENTLY against several
instrument/timeframe CSVs so a single overnight session covers several
instruments at once instead of guessing which one to try next.

Deliberately a thin orchestration layer on top of the EXISTING, already-
tested app.orchestration.speed_run.run_speed_run -- it does not
reimplement any discovery/validation logic, just fans it out the same way
app.orchestration.multi_instrument_search fans out run_search:

  1. Loads each instrument/timeframe's CSV once.
  2. Divides the worker/concurrency budget across the number of
     CONCURRENT instrument runs (each run_speed_run call still spins up
     its own Search Lab worker pool AND its own concurrent Full Pipeline
     validations internally), so running N instruments at once doesn't
     oversubscribe the machine N times over.
  3. Gives each instrument/timeframe its OWN output directory -- Speed
     Run's Phase 2 report-writing was built assuming one run owns its
     output_dir exclusively, so (same reasoning as results-db isolation
     in multi_instrument_search) the simplest way to make that true under
     concurrency is to actually give each job its own directory rather
     than teach report-writing to be collision-safe.
  4. Runs jobs on a ThreadPoolExecutor (not another process pool) capped
     at `max_concurrent_instruments` -- real parallelism ends up as
     (concurrent instruments) x (workers per instrument's own internal
     pools), bounded to stay within one shared CPU budget.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from app.backtest.risk import RiskConfig
from app.data.importer import import_csv
from app.orchestration.multi_instrument_search import InstrumentJob
from app.orchestration.speed_run import SpeedRunConfig, SpeedRunResult, run_speed_run
from app.prop.simulator import PropRules

ProgressCallback = Callable[[str, str], None]   # (job_label, message)


@dataclass
class MultiInstrumentSpeedRunResult:
    job: InstrumentJob
    result: SpeedRunResult | None
    error: str | None = None

    @property
    def label(self) -> str:
        return f"{self.job.instrument}/{self.job.timeframe}"

    @property
    def has_winner(self) -> bool:
        return self.result is not None and self.result.winner is not None


def _resolved_concurrency_per_job(cfg: SpeedRunConfig, n_concurrent: int) -> tuple[int | None, int]:
    """Splits Speed Run's own two worker-ish knobs (Phase 1's discovery_workers
    and Phase 2's max_concurrent_validations) across however many instrument
    jobs will actually run at once -- same reasoning as
    multi_instrument_search._resolved_workers_per_job. Never goes below 1
    for either."""
    total_workers = cfg.discovery_workers or (os.cpu_count() or 2)
    per_job_workers = max(1, total_workers // max(1, n_concurrent))
    per_job_validations = max(1, cfg.max_concurrent_validations // max(1, n_concurrent))
    return per_job_workers, per_job_validations


def run_multi_instrument_speed_run(
    jobs: list[InstrumentJob],
    risk: RiskConfig,
    prop_rules: PropRules,
    cfg: SpeedRunConfig,
    output_dir: str | Path,
    max_concurrent_instruments: int = 2,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, MultiInstrumentSpeedRunResult]:
    """Runs `cfg` (the SAME Speed Run configuration -- discovery settings,
    validation settings, fitness metric, everything) against every job's
    own market data, up to `max_concurrent_instruments` at once. Returns
    {"INSTRUMENT/TIMEFRAME": MultiInstrumentSpeedRunResult}.

    `output_dir`: a directory (created if missing) -- each job gets its
    own `<output_dir>/<instrument>_<timeframe>/` subdirectory (see this
    module's own docstring for why they're never shared).
    """
    if not jobs:
        raise ValueError("run_multi_instrument_speed_run requires at least one InstrumentJob.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_concurrent = max(1, min(int(max_concurrent_instruments), len(jobs)))
    per_job_workers, per_job_validations = _resolved_concurrency_per_job(cfg, n_concurrent)
    per_job_cfg = replace(
        cfg, discovery_workers=per_job_workers, max_concurrent_validations=per_job_validations,
    )

    def log(job: InstrumentJob, msg: str) -> None:
        if progress_cb:
            progress_cb(f"{job.instrument}/{job.timeframe}", msg)

    def run_one(job: InstrumentJob) -> MultiInstrumentSpeedRunResult:
        try:
            log(job, f"Loading {job.csv_path}...")
            import_result = import_csv(job.csv_path)
            if not import_result.is_valid:
                raise ValueError(
                    f"Could not import {job.csv_path}: " + "; ".join(import_result.errors)
                )
            df = import_result.dataframe
            job_output_dir = output_dir / f"{job.instrument}_{job.timeframe}"
            log(job, f"Starting Speed Run ({per_job_workers} discovery worker(s), "
                     f"{per_job_validations} concurrent validation(s))...")
            result = run_speed_run(
                df, risk, prop_rules, job_output_dir, per_job_cfg,
                progress_cb=lambda m: log(job, m), instrument=job.instrument,
            )
            verdict = "a winner" if result.winner is not None else "no winner"
            log(job, f"Complete: {verdict} ({result.elapsed_seconds:.1f}s).")
            return MultiInstrumentSpeedRunResult(job=job, result=result)
        except Exception as exc:  # noqa: BLE001 -- one instrument's failure must not sink the others
            log(job, f"FAILED: {exc}")
            return MultiInstrumentSpeedRunResult(
                job=job, result=None, error=f"{exc}\n{traceback.format_exc()}",
            )

    results: dict[str, MultiInstrumentSpeedRunResult] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            results[res.label] = res

    elapsed = time.time() - t0
    n_winners = sum(1 for r in results.values() if r.has_winner)
    if progress_cb:
        progress_cb(
            "multi-instrument",
            f"All {len(jobs)} instrument/timeframe job(s) complete in {elapsed:.1f}s "
            f"({n_winners} produced a winner).",
        )
    return results


def best_speed_run_across_instruments(
    results: dict[str, MultiInstrumentSpeedRunResult],
) -> MultiInstrumentSpeedRunResult | None:
    """Convenience: which instrument/timeframe actually produced the best
    Speed Run winner, ranked the same way a single Speed Run ranks its own
    candidates (app.orchestration.speed_run._rank_key: READY beats
    MARGINAL, then by eval_pass_probability). Returns None if nothing
    anywhere produced a winner."""
    from app.orchestration.speed_run import _rank_key

    with_winner = [r for r in results.values() if r.has_winner]
    if not with_winner:
        return None
    return min(with_winner, key=lambda r: _rank_key(r.result.winner))
