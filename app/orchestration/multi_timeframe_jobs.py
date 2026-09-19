"""
Builds Search Lab / Evolution Lab job lists (InstrumentJob /
EvolutionInstrumentJob) from ONE base dataset expanded across several
timeframes via app.data.timeframe_sweep, so Multi-Instrument Search and
Multi-Instrument Evolution -- built for "N separately-sourced files, one
job each" -- can just as well run "N timeframes resampled from ONE file,
one job each" with NO changes to either engine: a job is a job, whether
its csv_path came from a person's own upload or from this module writing
a resampled variant into data/raw/ moments earlier.

This is the one shared chokepoint every "Timeframes to test" control in
the app (Search Lab, Evolution Lab, and the Multi-Instrument versions of
both) funnels through, so a person can pick 1 dataset + several
timeframes, or several datasets + several timeframes (every dataset times
every timeframe), from any of those four surfaces and get the same,
consistently-labeled job list either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.data.timeframe_sweep import SweepSkipped, TimeframeSweepPlan, build_timeframe_sweep, write_sweep_targets_to_raw


@dataclass
class TimeframeExpansionResult:
    instrument: str
    plan: TimeframeSweepPlan
    paths: list[Path] = field(default_factory=list)  # same order as plan.targets

    @property
    def skipped(self) -> list[SweepSkipped]:
        return self.plan.skipped


def expand_dataset_across_timeframes(df: pd.DataFrame, instrument: str, requested_labels: list[str]) -> TimeframeExpansionResult:
    """Resamples `df` into every timeframe in `requested_labels` (see
    app.data.timeframe_sweep.build_timeframe_sweep for exactly what gets
    skipped and why) and writes each result to data/raw/ so it becomes an
    ordinary, reusable dataset file. `instrument` is used only as the base
    filename/label -- it does not need to match any existing dataset
    name."""
    plan = build_timeframe_sweep(df, requested_labels)
    paths = write_sweep_targets_to_raw(plan.targets, instrument) if plan.targets else []
    return TimeframeExpansionResult(instrument=instrument, plan=plan, paths=paths)


def search_jobs_from_expansion(expansion: TimeframeExpansionResult):
    """One InstrumentJob (app.orchestration.multi_instrument_search) per
    successfully-produced timeframe. Imported lazily to avoid a hard
    dependency on the search stack for callers (e.g. Evolution-only code)
    that never need it."""
    from app.orchestration.multi_instrument_search import InstrumentJob

    return [
        InstrumentJob(instrument=expansion.instrument, timeframe=target.label, csv_path=str(path))
        for target, path in zip(expansion.plan.targets, expansion.paths)
    ]


def evolution_jobs_from_expansion(expansion: TimeframeExpansionResult):
    """One EvolutionInstrumentJob (app.evolution.multi_instrument) per
    successfully-produced timeframe."""
    from app.evolution.multi_instrument import EvolutionInstrumentJob

    return [
        EvolutionInstrumentJob(instrument=expansion.instrument, timeframe=target.label, csv_path=str(path))
        for target, path in zip(expansion.plan.targets, expansion.paths)
    ]


def describe_skipped(expansion: TimeframeExpansionResult, dataset_label: str) -> list[str]:
    """Plain-English lines for a job log / warnings list, one per skipped
    requested timeframe -- e.g. for a Search Lab / Evolution Lab job log
    that already prints one line per accepted target and wants the same
    treatment for anything that didn't make it in."""
    return [f"{dataset_label}: {s.requested_label} -- {s.reason}" for s in expansion.skipped]
