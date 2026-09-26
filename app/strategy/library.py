"""
Persistent strategy library.

Lets a strategy (Python / PineScript / MQL5) be saved *inside* the app's own
data folder — the same persistent, writable, frozen-exe-aware location that
app.data.storage already uses for market-data CSVs — instead of only ever
being pulled from wherever it happens to live on a particular computer or
phone. Once saved, a strategy shows up in the library on every future run,
no re-browsing required. Uploading straight from a device is still fully
supported and is in fact how strategies get into the library in the first
place (save_strategy_path / save_strategy_bytes / save_strategy_text).

Folder layout (created on demand):

    <app base dir>/strategies/python/*.py            (+ *.py.meta.json sidecars)
    <app base dir>/strategies/pinescript/*.pine       (+ *.pine.meta.json sidecars)
    <app base dir>/strategies/mql5/*.mq5              (+ *.mq5.meta.json sidecars)

Mirrors app.data.storage.py's StoredDataset / list_stored_datasets /
store_csv_path / store_csv_bytes naming and behavior on purpose, so the two
subsystems stay easy to reason about side by side.

IMPORTANT — packaged .exe vs. the git repo: get_app_base_dir() (see
app/data/storage.py) resolves next to the running .exe for a frozen/
packaged build, but resolves to the repo root during normal development.
That means a strategy saved from a *built .exe* lands next to that .exe,
not inside your git repo, so it won't show up on GitHub until you copy it
over yourself (or use export_library_zip() below and unzip it into your
repo's strategies/ folder). Saving from the app run out of the repo
(`python -m app.web.server` / `python run_app.py`) writes directly into
the repo's strategies/ folder and needs no extra step.

Metadata sidecars (<file>.meta.json) hold everything besides the raw
source: description, market, timeframe, tags (list[str]), status (one of
STRATEGY_STATUSES -- the draft -> tested (failed/passed) -> validated ->
ready-for-demo -> ready-for-live lifecycle), and results other features
stamp onto a saved strategy after they run against it: last_run
(record_backtest_result), lookahead (record_lookahead_result), last_search
(record_search_result). None of that is required -- an entry with no
metadata at all is just a file with defaults (status "draft").
"""
from __future__ import annotations

import io
import json
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.data.storage import get_app_base_dir

# "manual" was added alongside the three code types so a Manual Strategy
# Builder config (the same JSON shape app.strategy.manual.ManualStrategy
# consumes, and the same shape the Evolution Lab's leaderboard PROMOTE
# button produces) is a first-class library citizen -- listed, viewable,
# loadable, batch-queueable, and deletable exactly like a saved .py/.pine/
# .mq5 file, just stored as .json. Before this, PROMOTE TO STRATEGY
# LIBRARY called save_strategy_text(..., "manual", ...) against a type
# _normalize_type() didn't recognize, which always raised
# "Unknown strategy type 'manual'" -- promoting any Evolution Lab leader
# (every leader IS a manual-builder config; see engine.py's documented
# scope limit) failed every time.
STRATEGY_TYPES = ("python", "pinescript", "mql5", "manual")

_EXTENSIONS = {
    "python": ".py",
    "pinescript": ".pine",
    "mql5": ".mq5",
    "manual": ".json",
}

_META_SUFFIX = ".meta.json"

# The strategy lifecycle this library tracks, cheapest-to-riskiest:
#   draft            -- default; being built/edited, not yet trusted
#   tested_failed    -- run through a backtest/pipeline and it failed
#                       (didn't pass prop rules, negative expectancy, etc.)
#   tested_passed    -- ran clean on a first pass, but hasn't been through
#                       deeper out-of-sample validation yet
#   validated        -- passed deeper validation (walk-forward, CPCV,
#                       lookahead/falsification checks, holdout, etc.)
#   ready_for_demo   -- validated and cleared to run on a demo account
#   ready_for_live   -- proven on demo and cleared to trade real money
# Nothing in this module enforces the *order* of transitions; it's a status
# label the person (or Full Pipeline's verdict, see
# app.orchestration.full_pipeline) sets deliberately, not a state machine
# that blocks you -- you can always drop a strategy back to "draft" or
# jump straight to "ready_for_live" if that's genuinely where it belongs.
STRATEGY_STATUSES = (
    "draft",
    "tested_failed",
    "tested_passed",
    "validated",
    "ready_for_demo",
    "ready_for_live",
)
DEFAULT_STATUS = "draft"

# Human-readable labels for the UI (listbox prefixes, dropdowns, badges).
# Falls back to status.upper() for anything not listed here, so an unknown
# / future status never breaks display.
STATUS_LABELS: dict[str, str] = {
    "draft": "DRAFT",
    "tested_failed": "TESTED / FAILED",
    "tested_passed": "TESTED / PASSED",
    "validated": "VALIDATED",
    "ready_for_demo": "READY FOR DEMO",
    "ready_for_live": "READY FOR LIVE",
}

# Old status values from before this lifecycle was expanded -- kept so a
# strategy tagged under the old 3-stage scheme ("draft"/"validated"/"live")
# still normalizes and displays sensibly instead of erroring, without
# needing to rewrite every existing .meta.json sidecar.
_LEGACY_STATUS_ALIASES: dict[str, str] = {
    "live": "ready_for_live",
}

# Pipeline reorg plan section 45 ("status taxonomy") maps almost exactly
# onto the STRATEGY_STATUSES lifecycle this library already had -- rather
# than introduce a second, parallel set of status strings (which would
# mean every existing caller that filters/sets status has to learn a new
# vocabulary, and every already-saved .meta.json sidecar needs migrating),
# this is a purely additive DISPLAY layer: the doc's six-stage names,
# shown alongside the existing slug wherever a caller wants the more
# research-pipeline-flavored label instead of (or next to) "TESTED /
# PASSED" etc. Nothing reads or writes PIPELINE_STAGE_LABELS as the
# actual stored status -- `status` on disk is unchanged.
PIPELINE_STAGE_LABELS: dict[str, str] = {
    "draft": "EXPLORING / FILTER",
    "tested_failed": "REJECTED",
    "tested_passed": "ROBUSTNESS / PROP REVIEW",
    "validated": "QUALIFIED (FINAL SELECTION)",
    "ready_for_demo": "FORWARD TESTING",
    "ready_for_live": "LIVE CANDIDATE",
}


def pipeline_stage_label(status: str) -> str:
    """The pipeline reorg plan's six-stage-funnel name for a strategy's
    CURRENT saved status, e.g. 'validated' -> 'QUALIFIED (FINAL
    SELECTION)'. Purely cosmetic -- see PIPELINE_STAGE_LABELS above.
    Falls back to the ordinary status_label() for anything unrecognized,
    so this never raises."""
    s = _LEGACY_STATUS_ALIASES.get((status or "").strip().lower(), (status or "").strip().lower())
    return PIPELINE_STAGE_LABELS.get(s, status_label(s))


def status_label(status: str) -> str:
    """Human-readable form of a status slug, for display -- e.g.
    'tested_failed' -> 'TESTED / FAILED'. Never raises."""
    s = _LEGACY_STATUS_ALIASES.get((status or "").strip().lower(), (status or "").strip().lower())
    return STATUS_LABELS.get(s, s.upper() if s else STATUS_LABELS[DEFAULT_STATUS])


class StrategyAlreadyExists(Exception):
    """Raised by the non-overwriting save/rename calls when the destination
    filename is already taken, so callers (UI code) can ask the user
    "overwrite or save as a new file?" instead of silently renaming."""

    def __init__(self, strategy_type: str, filename: str):
        self.strategy_type = strategy_type
        self.filename = filename
        super().__init__(f"A saved {strategy_type} strategy named '{filename}' already exists.")


def _normalize_type(strategy_type: str) -> str:
    t = (strategy_type or "").strip().lower()
    if t not in STRATEGY_TYPES:
        raise ValueError(
            f"Unknown strategy type '{strategy_type}'. Expected one of {STRATEGY_TYPES}."
        )
    return t


def _normalize_status(status: str) -> str:
    s = (status or "").strip().lower()
    s = _LEGACY_STATUS_ALIASES.get(s, s)
    if s not in STRATEGY_STATUSES:
        raise ValueError(f"Unknown status '{status}'. Expected one of {STRATEGY_STATUSES}.")
    return s


def _ensure_extension(filename: str, strategy_type: str) -> str:
    ext = _EXTENSIONS[strategy_type]
    name = Path(filename).name.strip() or f"strategy{ext}"
    if not name.lower().endswith(ext):
        name += ext
    return name


def safe_filename_stem(display_name: str, fallback: str = "strategy") -> str:
    """Turns an arbitrary human-readable strategy name into a safe base
    filename component (no extension).

    FIX (2026-09-16): callers used to do `Path(display_name).stem` to
    strip a would-be extension and get a filesystem-safe base name, but
    Path() treats "/" (and, on Windows, "\\") in the string as directory
    separators -- a display name like "VWAP Trend Continuation (ema
    50/200, rsi7)" (the "50/200" being a perfectly normal way to write
    "EMA 50 over EMA 200") got silently truncated to whatever followed
    the LAST separator, producing a nonsense saved filename like
    "200,_rsi7)_optimized.json" instead of the strategy's actual name.
    This strips/replaces filesystem-unsafe characters directly instead
    of routing through Path(), so every character of the intended name
    survives (as an underscore where it can't be used literally).
    """
    name = (display_name or "").strip()
    # Replace anything that isn't alphanumeric, space, hyphen, underscore,
    # or parens with an underscore -- this covers path separators (both
    # slash directions), colons, and any other character a filesystem
    # could misinterpret, without depending on pathlib to parse it.
    name = re.sub(r"[^A-Za-z0-9 _().-]+", "_", name)
    name = name.replace(" ", "_")
    name = re.sub(r"_+", "_", name).strip("_.")
    return name or fallback


def provenance_stamped_name(base_display_name: str, *, origin: str, seed: int | None) -> str:
    """Appends a short, honest provenance stamp to a strategy's display
    name -- "<base name> [origin, seed=N, YYYY-MM-DD]" -- for a config
    that a GA has actually MUTATED away from `base_display_name`'s own
    parameters.

    FIX (2026-09-17 Quick-Optimize-vs-Full-Pipeline naming-drift bug):
    Quick Optimize and Full Pipeline both used to save a GA winner's
    filename (and, for manual configs, its JSON "name" field) using the
    ORIGINAL strategy's display name verbatim -- e.g. a manual config
    literally named "Pure RSI Extreme Reversion (rsi21, 20/80)" could
    have every one of its actual RSI periods/thresholds mutated by the
    GA into something unrecognizable, and still get saved back out
    under that same stale "(rsi21, 20/80)" name. Two independently-run
    optimizations of the same starting file then produced two
    completely different parameter sets filed under near-identical
    names, which is exactly what made the Quick-Optimize-vs-Full-
    Pipeline discrepancy look like the same strategy giving
    inconsistent results, when it was actually two different mutated
    strategies sharing a label neither of them still matched.

    This does NOT attempt to auto-summarize the winning parameters into
    the name -- an arbitrary manual/code config has no generic, always-
    readable way to render that compactly, and a wrong or truncated
    summary would just trade one misleading name for another. Instead
    it stamps WHERE this exact file came from (which tool, which
    reproducible seed, which day) so two mutated results are never
    filed as if they were the same thing, and the saved file is always
    traceable back to a specific run. Callers use this for the file's
    base name (safe_filename_stem is applied afterward) and, for manual
    configs, also overwrite the saved JSON's own "name" field with it,
    so the two never drift apart again.

    Only called when the config was actually mutated (see each caller's
    own "did the GA produce a real winner" check) -- an unmutated
    baseline correctly keeps its original name.
    """
    seed_part = f"seed={seed}" if seed is not None else "seed=random"
    date_part = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{base_display_name} [{origin}, {seed_part}, {date_part}]"


def _seed_bundled_strategies(base: Path) -> None:
    """Copy strategy files embedded by PyInstaller into the persistent
    strategies/ folder, once, on first run of a packaged .exe.

    Mirrors app.data.storage._seed_bundled_raw_data exactly: a frozen build
    only ever has whatever PyInstaller was told to --add-data into the
    bundle (see the build-exe.yml / build-web-exe.yml workflows -- both
    must pass "strategies;strategies" for this to have anything to copy).
    Without this, get_strategy_library_dir() still creates the three
    per-language subfolders (so the app doesn't crash), but they come up
    empty in every packaged build -- the folders exist, nothing's in them.
    Never overwrites a file the user already saved/edited locally.
    """
    if not getattr(sys, "frozen", False):
        return
    bundle_root = getattr(sys, "_MEIPASS", None)
    if not bundle_root:
        return
    bundled_strategies = Path(bundle_root) / "strategies"
    if not bundled_strategies.exists():
        return
    for t in STRATEGY_TYPES:
        bundled_dir = bundled_strategies / t
        if not bundled_dir.exists():
            continue
        dest_dir = base / t
        for source in bundled_dir.iterdir():
            if not source.is_file():
                continue
            destination = dest_dir / source.name
            if not destination.exists():
                try:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                except OSError:
                    continue


def get_strategy_library_dir(strategy_type: str | None = None) -> Path:
    """Return the persistent strategies/ folder, creating it (and its three
    per-language subfolders) as needed, and seeding it from the bundled
    strategies the first time a packaged .exe runs. Pass a strategy_type to
    get that one subfolder directly."""
    base = get_app_base_dir() / "strategies"
    for t in STRATEGY_TYPES:
        (base / t).mkdir(parents=True, exist_ok=True)
    _seed_bundled_strategies(base)
    if strategy_type is None:
        return base
    return base / _normalize_type(strategy_type)


def _metadata_path(directory: Path, filename: str) -> Path:
    return directory / f"{filename}{_META_SUFFIX}"


@dataclass
class StoredStrategy:
    name: str                       # filename only, e.g. "ny_liquidity_fvg.py"
    strategy_type: str              # "python" | "pinescript" | "mql5"
    path: Path
    size_bytes: int
    modified: float                 # unix timestamp — newest-first sort key
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        raw = (self.metadata.get("status") or DEFAULT_STATUS).strip().lower()
        return _LEGACY_STATUS_ALIASES.get(raw, raw)

    @property
    def status_display(self) -> str:
        """Human-readable status, e.g. 'TESTED / FAILED' -- what the UI
        should actually show; `.status` stays the raw slug used for
        filtering/storage."""
        return status_label(self.status)

    @property
    def tags(self) -> list[str]:
        return list(self.metadata.get("tags") or [])

    @property
    def pipeline_progress(self) -> dict[str, Any]:
        """Create -> Test -> Optimize -> Validate -> Champion Check ->
        Ready progress for this strategy, derived from its metadata -- see
        compute_pipeline_progress() below."""
        return compute_pipeline_progress(self.metadata)


def list_misplaced_files(strategy_type: str) -> list[str]:
    """Filenames sitting inside strategy_type's own folder that don't have
    that type's expected extension -- almost always means someone dropped
    a file into the wrong one of the three subfolders (e.g. a .pine file
    inside strategies/python/ instead of strategies/pinescript/), which
    silently never shows up for any mode's REFRESH LIBRARY since the
    listing only ever globs for its own extension. Surfacing this by name
    turns "my file just isn't showing up, with no error" into something
    the person can actually fix themselves."""
    t = _normalize_type(strategy_type)
    d = get_strategy_library_dir(t)
    expected_ext = _EXTENSIONS[t]
    out = []
    for f in sorted(d.iterdir()):
        if not f.is_file():
            continue
        if f.name.endswith(_META_SUFFIX):
            continue
        if not f.name.lower().endswith(expected_ext):
            out.append(f.name)
    return out


def strategy_exists(strategy_type: str, filename: str) -> bool:
    """Whether a saved strategy with this exact filename already exists —
    check this before saving/renaming so the caller can ask "overwrite or
    save as a new file?" instead of silently getting a " (2)" duplicate."""
    t = _normalize_type(strategy_type)
    name = _ensure_extension(filename, t)
    return (get_strategy_library_dir(t) / name).exists()


def list_saved_strategies(
    strategy_type: str | None = None,
    query: str = "",
    tag: str | None = None,
    market: str | None = None,
    status: str | None = None,
) -> list[StoredStrategy]:
    """Return saved strategies, newest first.

    - strategy_type filters to one language; omit for all three.
    - query is a free-text, case-insensitive substring match against the
      filename, description, market, and tags (an OR-style search box).
    - tag/market/status are exact-match filters (case-insensitive) for
      *browsing* -- e.g. every strategy tagged "mean-reversion", or every
      XAUUSD strategy, or everything still in "draft" -- as opposed to
      query's free-text search. All provided filters combine with AND.
    """
    types = (_normalize_type(strategy_type),) if strategy_type else STRATEGY_TYPES
    q = query.strip().lower()
    tag_q = tag.strip().lower() if tag else None
    market_q = market.strip().lower() if market else None
    status_q = _normalize_status(status) if status else None

    out: list[StoredStrategy] = []
    for t in types:
        d = get_strategy_library_dir(t)
        for f in sorted(d.glob(f"*{_EXTENSIONS[t]}")):
            if not f.is_file():
                continue
            # "manual"'s own extension (.json) is a suffix of every
            # metadata sidecar's name (<file>.meta.json), so for manual
            # strategies (and only manual -- .py/.pine/.mq5 sidecars
            # never collide with their own type's glob) the glob above
            # also matches sidecars themselves. Without this check every
            # manual strategy appeared twice: once as the real strategy,
            # once as its own metadata file misidentified as a strategy.
            if f.name.endswith(_META_SUFFIX):
                continue
            stat = f.stat()
            meta = _read_metadata_file(_metadata_path(d, f.name))
            item = StoredStrategy(
                name=f.name, strategy_type=t, path=f,
                size_bytes=stat.st_size, modified=stat.st_mtime, metadata=meta,
            )
            if q and not _matches_query(item, q):
                continue
            if tag_q and tag_q not in [x.lower() for x in item.tags]:
                continue
            if market_q and market_q != str(item.metadata.get("market", "")).strip().lower():
                continue
            if status_q and status_q != item.status:
                continue
            out.append(item)
    out.sort(key=lambda s: s.modified, reverse=True)
    return out


def list_all_tags(strategy_type: str | None = None) -> list[str]:
    """Every distinct tag in use, sorted, for populating a "browse by tag"
    filter control."""
    tags: set[str] = set()
    for item in list_saved_strategies(strategy_type):
        tags.update(item.tags)
    return sorted(tags, key=str.lower)


def list_all_markets(strategy_type: str | None = None) -> list[str]:
    """Every distinct non-empty market value in use, sorted, for populating
    a "browse by market" filter control."""
    markets = {
        str(item.metadata.get("market", "")).strip()
        for item in list_saved_strategies(strategy_type)
        if str(item.metadata.get("market", "")).strip()
    }
    return sorted(markets, key=str.lower)


def _matches_query(item: StoredStrategy, q: str) -> bool:
    haystacks = [
        item.name,
        str(item.metadata.get("description", "")),
        str(item.metadata.get("market", "")),
        " ".join(item.tags),
    ]
    return any(q in h.lower() for h in haystacks)


def _unique_destination(directory: Path, filename: str) -> Path:
    dest = directory / filename
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 2
    while (directory / f"{stem} ({i}){suffix}").exists():
        i += 1
    return directory / f"{stem} ({i}){suffix}"


def save_strategy_path(source_path: str | Path, strategy_type: str, overwrite: bool = False) -> Path:
    """Copy an external strategy file (from BROWSE STRATEGY FILE, a phone's
    share sheet, anywhere on disk) into the persistent library.

    By default (overwrite=False) a name collision raises StrategyAlreadyExists
    rather than silently creating a " (2)" duplicate -- check strategy_exists()
    first and ask the user, or pass overwrite=True once they've confirmed.
    Already-in-place files (re-selecting a file that's already the stored
    copy) are always a no-op, never an error."""
    t = _normalize_type(strategy_type)
    source_path = Path(source_path)
    d = get_strategy_library_dir(t)
    try:
        if source_path.resolve().parent == d.resolve():
            return source_path
    except OSError:
        pass
    dest = d / source_path.name
    if dest.exists() and not overwrite:
        raise StrategyAlreadyExists(t, source_path.name)
    shutil.copyfile(source_path, dest)
    return dest


def save_strategy_bytes(content: bytes, filename: str, strategy_type: str, overwrite: bool = False) -> Path:
    """Write uploaded strategy-file bytes into the persistent library.
    Mirrors app.data.storage.store_csv_bytes. See save_strategy_path() for
    the overwrite/duplicate behavior."""
    t = _normalize_type(strategy_type)
    d = get_strategy_library_dir(t)
    name = Path(filename).name
    dest = d / name
    if dest.exists() and not overwrite:
        raise StrategyAlreadyExists(t, name)
    dest.write_bytes(content)
    return dest


def save_strategy_replacing_version(
    text: str, strategy_type: str, filename: str,
) -> Path:
    """Overwrite an EXISTING saved strategy's content in place, archiving
    the version being replaced first (into a `.versions/` sibling
    directory, tracked in metadata as "version_history") instead of
    silently discarding it -- and, critically, WITHOUT touching any of
    that file's existing pipeline-progress metadata (last_run/
    last_optimize/last_validation/last_champion_check/last_forward_test/
    last_deploy).

    This is the "replace" side of the Quick Optimize / Full Pipeline /
    etc. "replace this strategy or save as a new copy?" choice: every one
    of those tools used to ALWAYS write a new, provenance-stamped filename
    for anything it produced (see provenance_stamped_name), so running a
    strategy through Optimize, then Full Pipeline, then a validation tool
    left three-plus near-duplicate files in the library, each starting
    its own pipeline-progress tracking from zero -- exactly why the
    Dashboard/Strategy Library's stage progress could look "stuck":
    the tool that just ran had recorded its result onto a DIFFERENT file
    than the one being looked at. Calling this instead of
    save_strategy_text(..., overwrite=True) keeps everything the person
    has done to a strategy attached to the one file they think of as
    "the strategy," while still never losing a prior version outright.

    Raises FileNotFoundError if `filename` doesn't already exist -- this
    function is only for REPLACING something that exists; use
    save_strategy_text for a brand new file.
    """
    t = _normalize_type(strategy_type)
    name = _ensure_extension(filename, t)
    d = get_strategy_library_dir(t)
    dest = d / name
    if not dest.exists():
        raise FileNotFoundError(f"No saved {t} strategy named '{name}' to replace.")

    versions_dir = d / ".versions" / name
    versions_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    archived_path = versions_dir / f"{ts}{Path(name).suffix}"
    # Extremely unlikely on a single-file replace, but two replaces within
    # the same wall-clock second must never collide and silently drop the
    # first version's archive.
    n = 1
    while archived_path.exists():
        archived_path = versions_dir / f"{ts}-{n}{Path(name).suffix}"
        n += 1
    archived_path.write_text(dest.read_text(encoding="utf-8"), encoding="utf-8")

    dest.write_text(text, encoding="utf-8")

    existing_meta = load_strategy_metadata(t, name)
    history = list(existing_meta.get("version_history") or [])
    history.append({"path": str(archived_path.relative_to(d)), "replaced_at": ts})
    save_strategy_metadata(t, name, {"version_history": history}, merge=True)
    return dest


def list_strategy_versions(strategy_type: str, filename: str) -> list[dict[str, Any]]:
    """The archived-version history save_strategy_replacing_version has
    built up for this strategy, oldest first. Each entry's "path" is
    relative to this strategy_type's library directory (pass it to
    load_strategy_version to read that snapshot's content back)."""
    t = _normalize_type(strategy_type)
    name = _ensure_extension(filename, t)
    meta = load_strategy_metadata(t, name)
    return list(meta.get("version_history") or [])


def load_strategy_version(strategy_type: str, relative_path: str) -> str:
    """Read back one archived version's content by the relative "path"
    list_strategy_versions returned for it."""
    t = _normalize_type(strategy_type)
    base = get_strategy_library_dir(t)
    candidate = (base / relative_path).resolve()
    if base.resolve() not in candidate.parents:
        raise FileNotFoundError("Invalid version path.")
    if not candidate.is_file():
        raise FileNotFoundError("That archived version no longer exists.")
    return candidate.read_text(encoding="utf-8")


def save_strategy_text(text: str, filename: str, strategy_type: str, overwrite: bool = False) -> Path:
    """Write pasted/edited strategy source text into the persistent library
    (used by the web app, where a strategy may arrive as pasted text rather
    than an uploaded file). See save_strategy_path() for the
    overwrite/duplicate behavior."""
    t = _normalize_type(strategy_type)
    name = _ensure_extension(filename, t)
    d = get_strategy_library_dir(t)
    dest = d / name
    if dest.exists() and not overwrite:
        raise StrategyAlreadyExists(t, name)
    dest.write_text(text, encoding="utf-8")
    return dest


def resolve_saved_strategy_path(strategy_type: str, filename: str) -> Path:
    """Resolve `filename` to a path inside strategies/<type>/. Only the
    filename's basename is used (any directory components are stripped),
    which also rules out path-traversal via '../'."""
    t = _normalize_type(strategy_type)
    d = get_strategy_library_dir(t)
    candidate = d / Path(filename).name
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"No saved {t} strategy named '{filename}'.")
    return candidate


def load_strategy_text(strategy_type: str, filename: str) -> str:
    """Read back a saved strategy's source text by type + filename."""
    return resolve_saved_strategy_path(strategy_type, filename).read_text(encoding="utf-8")


def delete_saved_strategy(strategy_type: str, filename: str) -> None:
    path = resolve_saved_strategy_path(strategy_type, filename)
    path.unlink()
    meta_path = _metadata_path(path.parent, path.name)
    meta_path.unlink(missing_ok=True)


def delete_many(items: Iterable[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """Bulk delete. `items` is an iterable of (strategy_type, filename)
    pairs. Returns (deleted, failed) -- `deleted` is a list of "type/name"
    strings that were removed, `failed` is a list of "type/name: reason"
    strings for ones that couldn't be. One bad item never aborts the rest."""
    deleted, failed = [], []
    for strategy_type, filename in items:
        label = f"{strategy_type}/{filename}"
        try:
            delete_saved_strategy(strategy_type, filename)
            deleted.append(label)
        except (ValueError, FileNotFoundError) as exc:
            failed.append(f"{label}: {exc}")
    return deleted, failed


def rename_saved_strategy(
    strategy_type: str, old_filename: str, new_filename: str, overwrite: bool = False
) -> Path:
    """Rename a saved strategy (and its metadata sidecar, if any) in place.
    Raises StrategyAlreadyExists if new_filename is already taken and
    overwrite is False."""
    t = _normalize_type(strategy_type)
    old_path = resolve_saved_strategy_path(t, old_filename)
    new_name = _ensure_extension(new_filename, t)
    new_path = old_path.parent / new_name

    if new_path == old_path:
        return old_path
    if new_path.exists() and not overwrite:
        raise StrategyAlreadyExists(t, new_name)

    old_meta_path = _metadata_path(old_path.parent, old_path.name)
    new_meta_path = _metadata_path(new_path.parent, new_path.name)

    old_path.rename(new_path)
    if old_meta_path.exists():
        old_meta_path.rename(new_meta_path)
    return new_path


# ---------------------------------------------------------------------------
# Metadata sidecars — market/timeframe/description/tags/status/results
# ---------------------------------------------------------------------------

def _read_metadata_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_strategy_metadata(strategy_type: str, filename: str) -> dict[str, Any]:
    """Return the saved metadata dict for a strategy (description, market,
    tags, status, last_run, lookahead, last_search, etc.), or {} if none
    has been saved yet."""
    t = _normalize_type(strategy_type)
    d = get_strategy_library_dir(t)
    name = Path(filename).name
    return _read_metadata_file(_metadata_path(d, name))


def save_strategy_metadata(strategy_type: str, filename: str, metadata: dict[str, Any], merge: bool = True) -> Path:
    """Write (or merge into) a strategy's metadata sidecar. The strategy
    file itself does not need to exist yet -- this can be called right
    after save_strategy_* with the same filename. `merge=True` (default)
    updates only the keys provided, keeping the rest of any existing
    metadata (e.g. updating last_run without clobbering description)."""
    t = _normalize_type(strategy_type)
    d = get_strategy_library_dir(t)
    name = Path(filename).name
    path = _metadata_path(d, name)

    existing = _read_metadata_file(path) if merge else {}
    existing.update(metadata)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    return path


def set_strategy_tags(strategy_type: str, filename: str, tags: list[str]) -> Path:
    """Replace a strategy's tag list wholesale (dedupes, strips, drops
    blanks, case-preserving)."""
    seen: dict[str, str] = {}
    for tag in tags:
        clean = tag.strip()
        if clean and clean.lower() not in seen:
            seen[clean.lower()] = clean
    return save_strategy_metadata(strategy_type, filename, {"tags": list(seen.values())})


def set_strategy_status(strategy_type: str, filename: str, status: str) -> Path:
    """Set a strategy's lifecycle status -- one of STRATEGY_STATUSES
    ('draft', 'tested_failed', 'tested_passed', 'validated',
    'ready_for_demo', 'ready_for_live'). This is a plain label the person
    (or Full Pipeline's verdict) sets deliberately -- nothing here
    enforces moving through the stages in order."""
    return save_strategy_metadata(strategy_type, filename, {"status": _normalize_status(status)})


def record_backtest_result(strategy_type: str, filename: str, stats: dict[str, Any]) -> Path:
    """Convenience wrapper for saving a "last_run" block into a strategy's
    metadata right after a backtest/report finishes, e.g.:

        record_backtest_result("python", "fvg_v1.py", {
            "trades": 173, "net_profit": 34973.31, "win_rate": 51.7,
            "max_dd": 8.4, "report_html": "/reports/fvg_v1_20260827.html",
        })
    """
    return save_strategy_metadata(strategy_type, filename, {"last_run": stats}, merge=True)


def record_lookahead_result(strategy_type: str, filename: str, result: dict[str, Any]) -> Path:
    """Convenience wrapper for stamping the lookahead checker's verdict onto
    a saved strategy, e.g.:

        record_lookahead_result("python", "fvg_v1.py", {
            "clean": False, "summary": "Lookahead detected at 2 checkpoint(s)",
        })

    This is what makes the library list itself a trust signal instead of
    just a list of filenames -- a strategy that's never been checked, one
    that's clean, and one with a known leak all look different at a glance.
    """
    return save_strategy_metadata(strategy_type, filename, {"lookahead": result}, merge=True)


def record_search_result(strategy_type: str, filename: str, result: dict[str, Any]) -> Path:
    """Convenience wrapper for stamping a Search Lab run's outcome onto the
    base strategy it searched around (single/family_grid modes), e.g.:

        record_search_result("python", "fvg_v1.py", {
            "candidates_tested": 200, "best_fitness": 1.42,
            "fitness_metric": "composite_prop_score",
        })
    """
    return save_strategy_metadata(strategy_type, filename, {"last_search": result}, merge=True)


def record_optimize_result(strategy_type: str, filename: str, result: dict[str, Any]) -> Path:
    """Convenience wrapper for stamping a Quick Optimize (or Full
    Pipeline's own GA re-optimization step) outcome onto a saved strategy
    as "last_optimize", e.g.:

        record_optimize_result("python", "fvg_v1.py", {
            "improved": True, "net_profit": 34973.31, "win_rate": 51.7,
            "eval_pass_probability": 71.2,
        })

    This is what the Strategy Library's pipeline-progress tracker (see
    compute_pipeline_progress below) checks to mark the "Optimize" stage
    complete -- before this existed, Quick Optimize saved a strategy's
    code/status but stamped no metadata at all, so the dashboard had no
    way to tell a Quick-Optimized strategy apart from one that had never
    left Manual Builder."""
    return save_strategy_metadata(strategy_type, filename, {"last_optimize": result}, merge=True)


def record_validation_result(strategy_type: str, filename: str, result: dict[str, Any]) -> Path:
    """Convenience wrapper for stamping a deeper-validation outcome (CPCV/
    PBO, walk-forward-optimization out-of-sample check, or Full
    Pipeline's own Step 5 OOS/CPCV step) onto a saved strategy as
    "last_validation", e.g.:

        record_validation_result("python", "fvg_v1.py", {
            "method": "cpcv", "pbo": 34.2, "efficiency": 61.5,
        })

    Checked by compute_pipeline_progress below to mark the "Validate"
    stage complete."""
    return save_strategy_metadata(strategy_type, filename, {"last_validation": result}, merge=True)


def record_champion_check_result(strategy_type: str, filename: str, result: dict[str, Any]) -> Path:
    """Convenience wrapper for stamping the final READY/MARGINAL/NOT READY
    champion-check verdict (Full Pipeline's own _make_verdict, or a
    dedicated Champion Check action) onto a saved strategy as
    "last_champion_check", e.g.:

        record_champion_check_result("python", "fvg_v1.py", {
            "verdict": "READY", "t58_score": 78.4, "t58_tier": "Strong",
        })

    Checked by compute_pipeline_progress below to mark the "Champion
    Check" stage complete, and (only when verdict == "READY") the final
    "Ready" stage too -- this is the one stage compute_pipeline_progress
    treats as conditional rather than just "did this step run", since a
    strategy can be champion-checked and still come back NOT READY."""
    return save_strategy_metadata(strategy_type, filename, {"last_champion_check": result}, merge=True)


def record_forward_test_result(strategy_type: str, filename: str, result: dict[str, Any]) -> Path:
    """Convenience wrapper for stamping a Forward Test (paper/live-adjacent
    dry run, e.g. the MT5 bridge) outcome onto a saved strategy as
    "last_forward_test", e.g.:

        record_forward_test_result("python", "fvg_v1.py", {
            "platform": "mt5", "days_run": 14, "net_profit": 412.30,
        })

    Checked by compute_pipeline_progress below to mark the "Forward Test"
    stage complete."""
    return save_strategy_metadata(strategy_type, filename, {"last_forward_test": result}, merge=True)


def record_deploy_result(strategy_type: str, filename: str, result: dict[str, Any]) -> Path:
    """Convenience wrapper for stamping a Deploy Live action onto a saved
    strategy as "last_deploy", e.g.:

        record_deploy_result("python", "fvg_v1.py", {
            "broker": "MT5 demo", "account_size": 100000,
        })

    Checked by compute_pipeline_progress below to mark the final "Deploy"
    stage complete."""
    return save_strategy_metadata(strategy_type, filename, {"last_deploy": result}, merge=True)


# ---------------------------------------------------------------------------
# Pipeline-progress tracker -- Create -> Test -> Optimize -> Validate ->
# Champion Check -> Ready, the six stages Owen asked the Strategy Library
# dashboard to visibly track per strategy (distinct from, and finer-
# grained than, the six-stage STRATEGY_STATUSES/PIPELINE_STAGE_LABELS
# funnel above, which is a single status a person sets deliberately --
# this instead derives directly from which tools have actually TOUCHED
# this strategy, so it can never drift out of sync with reality the way a
# manually-set status can).
#
# Purely derived from whatever metadata is already on the sidecar --
# last_run (Run & Report / Quick Optimize / Full Pipeline / Forge all
# already call record_backtest_result), last_optimize (Quick Optimize's
# and Full Pipeline's GA step, see record_optimize_result), last_search
# (Search Lab, an alternate route into the same "explored parameter
# space" stage), last_validation (CPCV/PBO or a primary walk-forward/OOS
# check, see record_validation_result), and last_champion_check (Full
# Pipeline's final verdict, see record_champion_check_result). A
# strategy with none of these is simply at "Create" -- exactly a fresh
# Manual Builder draft or upload, matching STRATEGY_STATUSES' "draft".
# ---------------------------------------------------------------------------

PIPELINE_STAGE_NAMES = ("create", "test", "optimize", "validate", "champion_check", "forward_test", "deploy")
PIPELINE_STAGE_TITLES: dict[str, str] = {
    "create": "Create",
    "test": "Test",
    "optimize": "Optimize",
    "validate": "Validate",
    "champion_check": "Champion",
    "forward_test": "Forward Test",
    "deploy": "Deploy",
}
# Where the dashboard's "what should I do next" button should point for
# each not-yet-done stage. One primary route per stage (the tool named
# first in Owen's own spec for that stage); a strategy can still reach
# "done" for a stage via any of the other tools compute_pipeline_progress
# below checks (e.g. "validate" also completes via CPCV/PBO/sensitivity/
# regime survival/Full Pipeline, not just Walk-Forward Opt) -- this is
# only which link the button shows, never a restriction on which tool
# counts.
PIPELINE_STAGE_NEXT_HREF: dict[str, str] = {
    "test": "/",                      # Run & Report (Full Pipeline also completes this stage)
    "optimize": "/quick-optimize",
    "validate": "/cpcv",
    "champion_check": "/family-diversity",
    "forward_test": "/forward-test",
    "deploy": "/deploy-live",
}


def compute_pipeline_progress(metadata: dict[str, Any]) -> dict[str, Any]:
    """Given a strategy's metadata dict (StoredStrategy.metadata, or any
    dict shaped like a .meta.json sidecar), return:

        {
            "stages": [{"key", "title", "done"}, ...],   # in stage order
            "current_stage": "optimize",                  # furthest stage reached
            "next_stage": "validate",                     # first not-yet-done stage, or None if Deploy is done
            "next_href": "/cpcv",                          # where the dashboard's next-step button should point
            "progress_pct": 33.3,                          # % of the 6 non-"create" stages completed
            "verdict": "NOT READY" | "MARGINAL" | "READY" | None,
        }

    Never raises -- a strategy with no metadata at all comes back as
    everything False except "create", same as a fresh draft."""
    last_run = metadata.get("last_run") or {}
    last_optimize = metadata.get("last_optimize") or {}
    last_search = metadata.get("last_search") or {}
    last_validation = metadata.get("last_validation") or {}
    last_champion_check = metadata.get("last_champion_check") or {}
    last_forward_test = metadata.get("last_forward_test") or {}
    last_deploy = metadata.get("last_deploy") or {}

    tested = bool(last_run)
    optimized = bool(last_optimize) or bool(last_search)
    validated = bool(last_validation)
    champion_checked = bool(last_champion_check)
    forward_tested = bool(last_forward_test)
    deployed = bool(last_deploy)
    # Full Pipeline's record_backtest_result already stamps "verdict" onto
    # last_run too (see run_full_pipeline) -- fall back to that when a
    # dedicated champion-check record isn't present, so a strategy run
    # only through Full Pipeline (not a separate Champion Check action)
    # still shows READY/MARGINAL/NOT READY correctly.
    verdict = str(last_champion_check.get("verdict") or last_run.get("verdict") or "").upper() or None

    stages = [
        {"key": "create", "title": PIPELINE_STAGE_TITLES["create"], "done": True},
        {"key": "test", "title": PIPELINE_STAGE_TITLES["test"], "done": tested},
        {"key": "optimize", "title": PIPELINE_STAGE_TITLES["optimize"], "done": optimized},
        {"key": "validate", "title": PIPELINE_STAGE_TITLES["validate"], "done": validated},
        {"key": "champion_check", "title": PIPELINE_STAGE_TITLES["champion_check"], "done": champion_checked},
        {"key": "forward_test", "title": PIPELINE_STAGE_TITLES["forward_test"], "done": forward_tested},
        {"key": "deploy", "title": PIPELINE_STAGE_TITLES["deploy"], "done": deployed},
    ]

    current_stage = "create"
    for stage in stages:
        if stage["done"]:
            current_stage = stage["key"]

    next_stage = next((s["key"] for s in stages if not s["done"]), None)
    non_create = stages[1:]
    progress_pct = round(100.0 * sum(1 for s in non_create if s["done"]) / len(non_create), 1)

    return {
        "stages": stages,
        "current_stage": current_stage,
        "next_stage": next_stage,
        "next_href": PIPELINE_STAGE_NEXT_HREF.get(next_stage) if next_stage else None,
        "progress_pct": progress_pct,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Backup / export
# ---------------------------------------------------------------------------

def _iter_export_files(selection: Iterable[tuple[str, str]] | None):
    """Yield (real_path, arcname) pairs for either the whole library
    (selection=None) or just the given (strategy_type, filename) pairs,
    each strategy file paired with its metadata sidecar if it has one."""
    base = get_strategy_library_dir()
    if selection is None:
        for t in STRATEGY_TYPES:
            d = get_strategy_library_dir(t)
            for f in sorted(d.iterdir()):
                if f.is_file():
                    yield f, str(f.relative_to(base.parent))
        return

    for strategy_type, filename in selection:
        path = resolve_saved_strategy_path(strategy_type, filename)
        yield path, str(path.relative_to(base.parent))
        meta_path = _metadata_path(path.parent, path.name)
        if meta_path.exists():
            yield meta_path, str(meta_path.relative_to(base.parent))


def export_library_zip_bytes(selection: Iterable[tuple[str, str]] | None = None) -> bytes:
    """Zip either the entire strategy library (selection=None) or just the
    given (strategy_type, filename) pairs (bulk-export a selection), and
    return the zip content as bytes -- for streaming a download (the web
    app) without writing a temp file. The zip's internal paths are rooted
    at "strategies/<type>/<file>" so unzipping it directly into a repo
    checkout drops files in the right place -- this is also the fix for a
    packaged .exe's library (which lives next to the .exe, not in the git
    repo): export, unzip into the repo's strategies/ folder, commit."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for real_path, arcname in _iter_export_files(selection):
            zf.write(real_path, arcname=arcname)
    return buffer.getvalue()


def export_library_zip(destination_path: str | Path, selection: Iterable[tuple[str, str]] | None = None) -> Path:
    """Same as export_library_zip_bytes but writes straight to
    `destination_path` on disk (the desktop app's EXPORT LIBRARY button).
    Returns the path written."""
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(export_library_zip_bytes(selection))
    return destination_path
