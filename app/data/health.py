"""Data Center: per-file and per-instrument health analysis for everything
stored under data/raw/ -- timeframe availability, coverage gaps,
duplicate timestamps, timezone, session-hour coverage, and bar counts.

Reuses app.data.importer.import_csv (the same parser every upload/fetch
path already goes through) rather than re-implementing CSV/parquet
parsing, so a file's health report is always consistent with how the
backtest engine would actually read it.

Deliberately read-only and best-effort: a single corrupt/locked file must
never blank out the health report for every other dataset on disk (same
defensive posture as app.data.storage.list_datasets_by_instrument, which
this module is the deeper-diagnostic sibling of).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.data.importer import import_csv
from app.data.storage import get_raw_data_dir, list_stored_datasets
from app.data.timeframe_resample import infer_timeframe_label


@dataclass
class FileHealth:
    name: str            # path relative to data/raw/, e.g. "ES/ES1_5m.csv"
    instrument: str
    size_bytes: int
    ok: bool
    error: Optional[str] = None
    timeframe: Optional[str] = None
    bar_count: int = 0
    start: Optional[str] = None
    end: Optional[str] = None
    timezone: Optional[str] = None
    duplicate_count: int = 0
    gap_count: int = 0
    largest_gap: Optional[str] = None
    session_hours: list[int] = field(default_factory=list)
    session_coverage_pct: float = 0.0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "instrument": self.instrument, "size_bytes": self.size_bytes,
            "ok": self.ok, "error": self.error, "timeframe": self.timeframe,
            "bar_count": self.bar_count, "start": self.start, "end": self.end,
            "timezone": self.timezone, "duplicate_count": self.duplicate_count,
            "gap_count": self.gap_count, "largest_gap": self.largest_gap,
            "session_hours": self.session_hours, "session_coverage_pct": self.session_coverage_pct,
            "issues": self.issues,
            "healthy": self.ok and self.duplicate_count == 0 and self.gap_count == 0 and not self.issues,
        }


# A gap wider than this multiple of the file's own median bar spacing is
# flagged -- 3x tolerates the ordinary weekend/holiday gap every intraday
# futures/forex file has without flooding the report with expected,
# harmless breaks; only a genuinely unusual hole in the data trips this.
GAP_MULTIPLE = 3.0
# Below this many bars, gap statistics are too noisy to mean anything
# (e.g. a 5-row smoke-test file where every diff looks like an "outlier").
MIN_BARS_FOR_GAP_CHECK = 20


def compute_file_health(path: Path, relative_name: str) -> FileHealth:
    instrument = relative_name.split("/")[0] if "/" in relative_name else "(ungrouped)"
    size_bytes = path.stat().st_size if path.exists() else 0
    fh = FileHealth(name=relative_name, instrument=instrument, size_bytes=size_bytes, ok=False)

    try:
        result = import_csv(path)
    except Exception as exc:  # noqa: BLE001 -- never let one bad file break the report
        fh.error = f"Could not read file: {exc}"
        return fh

    if not result.is_valid or result.dataframe is None:
        fh.error = "; ".join(result.errors) or "Could not parse as OHLCV data."
        return fh

    df = result.dataframe
    fh.ok = True
    fh.bar_count = len(df)

    # NOTE: import_csv() already drops duplicate timestamps from
    # `df` itself (sort + drop_duplicates, keep first) as part of its own
    # cleaning, and records how many it removed as a "Removed N duplicate
    # timestamp row(s)" warning -- so duplicate detection has to read that
    # warning rather than re-counting df.duplicated(), which is always 0
    # by the time we see it here.
    for w in result.warnings:
        m = re.search(r"Removed (\d+) duplicate", w)
        if m:
            fh.duplicate_count = int(m.group(1))
            fh.issues.append(f"{fh.duplicate_count} duplicate timestamp(s) (kept first, duplicates dropped on read)")
            break

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    ts = ts.dropna().sort_values()
    if ts.empty:
        fh.error = "No valid timestamps after parsing."
        fh.ok = False
        return fh

    fh.start = str(ts.iloc[0])
    fh.end = str(ts.iloc[-1])
    tz = getattr(ts.dt, "tz", None)
    fh.timezone = str(tz) if tz is not None else "naive (no timezone info in file)"

    try:
        fh.timeframe = infer_timeframe_label(df)
    except Exception:
        fh.timeframe = None

    hours = sorted(set(int(h) for h in ts.dt.hour))
    fh.session_hours = hours
    fh.session_coverage_pct = round(100.0 * len(hours) / 24.0, 1)

    if len(ts) >= MIN_BARS_FOR_GAP_CHECK:
        diffs = ts.diff().dropna()
        positive = diffs[diffs > pd.Timedelta(0)]
        if not positive.empty:
            median_diff = positive.median()
            if median_diff > pd.Timedelta(0):
                gaps = positive[positive > median_diff * GAP_MULTIPLE]
                fh.gap_count = int(len(gaps))
                if fh.gap_count:
                    fh.largest_gap = str(gaps.max())
                    fh.issues.append(f"{fh.gap_count} coverage gap(s) (largest {fh.largest_gap})")

    if size_bytes <= 32:
        fh.issues.append("File is empty or a placeholder (<=32 bytes)")

    return fh


def compute_data_center() -> dict[str, Any]:
    """Full Data Center report: every stored dataset's health, grouped by
    instrument, plus per-instrument timeframe-availability and an overall
    summary. Read-only -- never modifies anything on disk."""
    raw_dir = get_raw_data_dir()
    files_by_health: list[FileHealth] = []
    for ds in list_stored_datasets():
        try:
            files_by_health.append(compute_file_health(ds.path, ds.name))
        except Exception as exc:  # noqa: BLE001 -- one bad file must not blank the whole report
            files_by_health.append(FileHealth(
                name=ds.name, instrument=ds.name.split("/")[0] if "/" in ds.name else "(ungrouped)",
                size_bytes=ds.size_bytes, ok=False, error=f"Unexpected error: {exc}",
            ))

    groups: dict[str, list[FileHealth]] = {}
    for fh in files_by_health:
        groups.setdefault(fh.instrument, []).append(fh)

    instruments = []
    for instrument in sorted(groups.keys()):
        group_files = groups[instrument]
        timeframes = sorted({f.timeframe for f in group_files if f.timeframe})
        total_bars = sum(f.bar_count for f in group_files)
        unhealthy = [f for f in group_files if not f.to_dict()["healthy"]]
        instruments.append({
            "instrument": instrument,
            "files": [f.to_dict() for f in sorted(group_files, key=lambda x: x.name)],
            "file_count": len(group_files),
            "timeframes_available": timeframes,
            "total_bars": total_bars,
            "unhealthy_count": len(unhealthy),
        })

    return {
        "instruments": instruments,
        "total_files": len(files_by_health),
        "total_unhealthy": sum(1 for f in files_by_health if not f.to_dict()["healthy"]),
        "total_bars": sum(f.bar_count for f in files_by_health),
        "raw_dir": str(raw_dir),
    }
