"""
Timeframe sweep -- pre-resamples ONE loaded dataset into several coarser
timeframes so a single downloaded file (typically the finest one
available, e.g. 1-minute) can be searched/evolved/optimized against
multiple bar sizes automatically, instead of the person needing to
separately source and upload a 5m/15m/1h/4h file for each one.

This is deliberately just data preparation, not a new search/evolution/
optimization algorithm: it reuses app.data.timeframe_resample.
resample_ohlcv (the exact same resampling this app already trusts for a
strategy's own declared execution timeframe -- see that module's
docstring) once per requested timeframe, and hands back the results as a
list of ordinary (label, DataFrame) pairs. Everything downstream --
Search Lab, Evolution Lab, Multi-Instrument Search/Evolution, Quick
Optimize, Multi-Objective, and any GA refinement they call into -- is
completely unaware this happened; each one just sees N ordinary in-memory
DataFrames (or, once written out via write_sweep_targets_to_raw, N
ordinary CSV files it can load exactly like any other dataset).

Nothing here changes what a strategy that already declares its own
`"timeframe"` (see app.data.timeframe_resample) does -- a sweep and a
strategy's own declaration are two independent things. A sweep changes
what DATA a search/evolution/optimization run is handed; a strategy
declaration changes what a given STRATEGY resamples that data to before
trading it. Running a sweep target through a strategy that also declares
its own timeframe just means that strategy's own declaration wins for
that one run, same as always.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.data.timeframe_resample import (
    TimeframeError,
    native_bar_minutes,
    normalize_timeframe_label,
    parse_timeframe_label,
    resample_ohlcv,
)

# The set every "Timeframes to test" control across the app offers as a
# starting point -- deliberately excludes anything at or below common raw-
# data granularity (1m) since sweeping a 1-minute file against a "1m"
# target would just be the native data again, under a second name.
DEFAULT_SWEEP_TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "30m", "1h", "4h")

# A sweep target with fewer bars than this is skipped rather than handed
# to a search/evolution/optimization tool that would just fail on it (or
# worse, "succeed" on statistically meaningless sample size) -- 30 is a
# deliberately low floor (not itself a claim of statistical significance,
# just a sanity floor beneath which nothing downstream can do anything
# useful at all).
_MIN_USABLE_BARS = 30


@dataclass
class SweepTarget:
    """One timeframe's resampled dataset, ready to hand to any tool that
    already accepts a plain OHLCV DataFrame."""
    label: str          # normalized display label, e.g. "1h"
    minutes: float       # bar spacing in minutes, e.g. 60.0
    dataframe: pd.DataFrame
    bar_count: int


@dataclass
class SweepSkipped:
    """A requested timeframe that could NOT be produced, and why -- never
    raised, always reported back to the caller so one bad/duplicate/too-
    fine entry in a typed list doesn't kill the other timeframes' runs."""
    requested_label: str
    reason: str


@dataclass
class TimeframeSweepPlan:
    targets: list[SweepTarget] = field(default_factory=list)
    skipped: list[SweepSkipped] = field(default_factory=list)
    native_minutes: float = 0.0

    @property
    def labels(self) -> list[str]:
        return [t.label for t in self.targets]


def parse_sweep_timeframes(raw: str | list[str] | None) -> list[str]:
    """Splits a comma/semicolon/whitespace-separated string (or accepts an
    already-split list) into a clean list of non-empty label strings,
    preserving the order given and de-duplicating case/spacing variants of
    the SAME typed label ("1H" and "1h " collapse to one request) --
    actual timeframe normalization/validation happens in
    build_timeframe_sweep, this just cleans up what the person typed."""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in re.split(r"[,;\s]+", raw) if p.strip()]
    else:
        parts = [str(p).strip() for p in raw if str(p).strip()]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def build_timeframe_sweep(raw_df: pd.DataFrame, requested_labels: list[str]) -> TimeframeSweepPlan:
    """Resamples `raw_df` once per label in `requested_labels`. Never
    raises: an unparseable label, a label finer than the loaded data's own
    native bar spacing, a label that normalizes to the SAME timeframe as
    one already produced earlier in the list (e.g. "60m" after "1h"), or a
    label that would leave too few bars to be useful is recorded in
    `.skipped` with a plain-English reason instead of stopping the sweep.
    An empty `requested_labels` returns an empty plan (`.targets == []`)
    -- the caller decides what that means (typically: fall back to
    running once against the native data, exactly as before this sweep
    existed).
    """
    native_minutes = native_bar_minutes(raw_df)
    targets: list[SweepTarget] = []
    skipped: list[SweepSkipped] = []
    produced_minutes: set[float] = set()

    for requested in requested_labels:
        try:
            label = normalize_timeframe_label(requested)
            minutes = parse_timeframe_label(label)
        except TimeframeError as exc:
            skipped.append(SweepSkipped(requested, str(exc)))
            continue
        if minutes < native_minutes - 1e-6:
            skipped.append(SweepSkipped(
                requested,
                f"'{label}' is FINER than the loaded data's native ~{native_minutes:.0f}-minute bars -- "
                "cannot resample finer than the source data.",
            ))
            continue
        if minutes in produced_minutes:
            skipped.append(SweepSkipped(
                requested, f"'{label}' is the same bar size as one already in this sweep -- skipped as a duplicate.",
            ))
            continue
        df = raw_df.reset_index(drop=True) if abs(minutes - native_minutes) < 1e-6 else resample_ohlcv(raw_df, label)
        if len(df) < _MIN_USABLE_BARS:
            skipped.append(SweepSkipped(
                requested,
                f"'{label}' would leave only {len(df)} bar(s) from this dataset -- too few to search/evolve/"
                "optimize against, skipped.",
            ))
            continue
        produced_minutes.add(minutes)
        targets.append(SweepTarget(label=label, minutes=minutes, dataframe=df, bar_count=len(df)))

    return TimeframeSweepPlan(targets=targets, skipped=skipped, native_minutes=native_minutes)


def _unique_raw_destination(raw_dir: Path, filename: str) -> Path:
    dest = raw_dir / filename
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 2
    while (raw_dir / f"{stem} ({i}){suffix}").exists():
        i += 1
    return raw_dir / f"{stem} ({i}){suffix}"


def write_sweep_targets_to_raw(targets: list[SweepTarget], base_label: str) -> list[Path]:
    """Writes each SweepTarget out as a standalone CSV under data/raw/, so
    Multi-Instrument Search/Evolution (and anything else that only knows
    how to load a plain `csv_path`) can treat each timeframe exactly like
    any other separately-sourced dataset file -- selectable and re-usable
    from the normal "Available Datasets" list from then on, not a
    throwaway temp file that vanishes after one run. Returns the written
    paths in the SAME order as `targets`.
    """
    from app.data.storage import get_raw_data_dir

    raw_dir = get_raw_data_dir()
    safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_label).strip("_") or "dataset"
    paths: list[Path] = []
    for target in targets:
        dest = _unique_raw_destination(raw_dir, f"{safe_base}__{target.label}.csv")
        target.dataframe.to_csv(dest, index=False)
        paths.append(dest)
    return paths
