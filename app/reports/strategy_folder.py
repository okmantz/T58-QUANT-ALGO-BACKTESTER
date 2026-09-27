"""
Consolidated per-strategy report folder (item 4: "I want all of the
tests and reports for one strategy to be in its individual folder").

Every validation tool (CPCV, Walk-Forward Opt, Sensitivity, Walk-Forward
GA) and the main Run & Report / Full Pipeline backtest each write their
own HTML/JSON report into their OWN separate folder under reports/
(reports/cpcv/, reports/walk_forward_opt/, reports/sensitivity/, ...),
named only by a job-id or timestamp. Before this module, there was no
single place to go answer "what's been tested on strategy X, and why did
it pass or fail" -- you had to already know which tool produced which
file and dig through each tool's own folder separately.

This module gives every strategy ONE folder --
reports/by_strategy/<slug>/ -- containing a COPY of every report
generated for it (so the tool-specific folders are untouched and every
other part of the app that already links to those original paths keeps
working exactly as before), plus an index.json summarizing what ran,
when, and the pass/fail verdict for each entry.

Identity: prefers the strategy's stable Strategy Library filename
(library_type/library_filename) when available -- exactly the same
identity app.reports.strategy_state now uses for the validation
checklist -- so a strategy's folder doesn't fragment across the several
different display names it accumulates as it moves through Full
Pipeline/Quick Optimize/etc. Falls back to a slug of the display name +
instrument for one-off strategies never saved to the library.

Recording is always best-effort: a failure here must never break the
validation job or backtest run that produced the report.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Optional

_INDEX_FILENAME = "index.json"
MAX_ENTRIES_PER_STRATEGY = 500


def _reports_root() -> Path:
    from app.data.storage import get_app_base_dir

    d = get_app_base_dir() / "reports" / "by_strategy"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reports_root() -> Path:
    """Public accessor for the reports/by_strategy/ root -- used by
    app.web.server to serve a copied report file directly out of a
    strategy's folder (see serve_strategy_report_file)."""
    return _reports_root()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", (text or "").strip()).strip("_").lower()
    return slug or "strategy"


def strategy_slug(strategy_name: str, instrument: str, *, library_type: str = "", library_filename: str = "") -> str:
    """The folder name for one strategy's consolidated reports. Prefers
    the Strategy Library filename (stable across a strategy's lifetime)
    over the display name (which Full Pipeline/Quick Optimize/etc. all
    decorate differently, e.g. "-- Full Pipeline optimized (...)") --
    two runs of the SAME saved strategy always land in the same folder
    even if their display names differ.

    When a library identity is available, the slug is keyed on THAT
    alone (no instrument suffix): "one strategy's folder" means the
    saved file, however many different instruments it's been tested
    against, and it also keeps the slug reproducible for a caller that
    only has the library_ref and not the instrument on hand (e.g. this
    page being linked from the Strategy Library, which doesn't track a
    "last instrument" separately from "last_run"). Only a one-off
    strategy with no library identity falls back to name+instrument,
    since name alone isn't a safe folder key across unrelated strategies
    that happen to share a display name."""
    if library_type and library_filename:
        return f"{_slugify(library_type)}__{_slugify(Path(library_filename).stem)}"
    base = _slugify(strategy_name)
    inst = _slugify(instrument) if instrument else ""
    return f"{base}__{inst}" if inst else base


def strategy_folder_path(strategy_name: str, instrument: str, *, library_type: str = "", library_filename: str = "") -> Path:
    """The actual directory for this strategy's consolidated reports,
    creating it if needed."""
    slug = strategy_slug(strategy_name, instrument, library_type=library_type, library_filename=library_filename)
    d = _reports_root() / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_index(folder: Path) -> list[dict[str, Any]]:
    path = folder / _INDEX_FILENAME
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_index(folder: Path, entries: list[dict[str, Any]]) -> None:
    path = folder / _INDEX_FILENAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries[-MAX_ENTRIES_PER_STRATEGY:], indent=2), encoding="utf-8")
    tmp.replace(path)


def record_report(
    strategy_name: str,
    instrument: str,
    tool: str,
    *,
    library_type: str = "",
    library_filename: str = "",
    passed: Optional[bool] = None,
    summary: str = "",
    source_files: Optional[dict[str, Any]] = None,
) -> None:
    """Copies every file in ``source_files`` (e.g.
    {"html": "/abs/path/to/report.html", "json": "/abs/path/to/report.json"})
    into this strategy's consolidated folder, prefixed with
    ``<tool>_<timestamp>_`` so multiple runs of the same tool (or the
    same tool re-run later) never collide, then appends one summary
    entry to that folder's index.json describing what ran, when, and
    whether it passed. Best-effort throughout: a copy failure for one
    file, or the whole call failing, never raises -- this is a
    convenience mirror, not the source of truth (the JSON checklist in
    app.reports.strategy_state is)."""
    try:
        folder = strategy_folder_path(strategy_name, instrument, library_type=library_type, library_filename=library_filename)
        ts = time.time()
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(ts))
        copied: dict[str, str] = {}
        for kind, src in (source_files or {}).items():
            if not src:
                continue
            try:
                src_path = Path(src)
                if not src_path.is_file():
                    continue
                dest_name = f"{_slugify(tool)}_{stamp}_{src_path.name}"
                dest_path = folder / dest_name
                shutil.copy2(src_path, dest_path)
                copied[kind] = dest_path.name
            except Exception:
                continue

        entries = _load_index(folder)
        entries.append({
            "tool": tool,
            "strategy_name": strategy_name,
            "instrument": instrument,
            "passed": passed,
            "summary": summary,
            "recorded_at": ts,
            "files": copied,
        })
        _save_index(folder, entries)
    except Exception:
        pass


def list_reports(strategy_name: str, instrument: str, *, library_type: str = "", library_filename: str = "") -> list[dict[str, Any]]:
    """Every recorded report entry for this strategy, oldest first (as
    stored) -- used by the "Strategy reports" page to list what's in its
    folder without the caller needing to know the folder path."""
    folder = strategy_folder_path(strategy_name, instrument, library_type=library_type, library_filename=library_filename)
    return _load_index(folder)
