"""Persistent local market-data storage for development and packaged builds."""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from app.data.importer import SUPPORTED_ARCHIVE_EXTENSIONS

# werkzeug ships with Flask (a hard dependency of the web edition), so
# this import needs no requirements-file change -- the same sanitize
# step runs identically on the desktop path. (Dependency pinning is a
# separate audit item; requirements files are left untouched here.)
from werkzeug.utils import secure_filename

# Every extension the importer (app.data.importer.SUPPORTED_DATA_EXTENSIONS,
# plus SUPPORTED_ARCHIVE_EXTENSIONS) can actually read as tabular market
# data. list_stored_datasets() and the bundled-data seeder below used to
# hardcode "*.csv" only, which meant a .parquet (or .tsv/.txt) file sitting
# right there in data/raw was invisible to the desktop app's stored-dataset
# list AND the web app's dropdown even though selecting/importing it would
# have worked fine -- kept in sync with the importer's own sets rather than
# duplicated as separate literals so the two can't drift apart again.
#
# UPGRADE (2026-09-03): .zip and .7z were missing from this tuple even
# though app.data.importer has fully supported reading both for a while
# (_read_zip/_read_7z, py7zr already in config/requirements.txt) -- an
# archive sitting in data/raw/ was importable if you somehow got its path
# into the app, but never showed up in the "Available Datasets" list or the
# web dropdown in the first place, so in practice it was invisible. Fixed
# by importing SUPPORTED_ARCHIVE_EXTENSIONS from the importer instead of
# re-guessing it here -- exactly the kind of drift this tuple's own comment
# already warned about.
RAW_DATA_EXTENSIONS = (".csv", ".tsv", ".txt", ".parquet") + tuple(SUPPORTED_ARCHIVE_EXTENSIONS)


def get_app_base_dir() -> Path:
    """Return a writable persistent application-data root."""
    if getattr(sys, "frozen", False):
        preferred = Path(sys.executable).resolve().parent
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            probe = preferred / ".t58_write_test"
            probe.touch(exist_ok=True)
            probe.unlink(missing_ok=True)
            return preferred
        except OSError:
            return user_data_dir()
    return Path(__file__).resolve().parents[2]


def user_data_dir() -> Path:
    """Per-user writable data folder, using each OS's own convention (used when
    the folder next to the executable is read-only -- e.g. inside a macOS .app
    bundle, /usr/local/bin, or Program Files):
      Windows: %LOCALAPPDATA%\\T58 Prop Algo Backtester   (unchanged from before)
      macOS:   ~/Library/Application Support/T58 Prop Algo Backtester
      Linux:   $XDG_DATA_HOME or ~/.local/share/T58 Prop Algo Backtester
    """
    name = "T58 Prop Algo Backtester"
    if sys.platform.startswith("win"):
        local_app_data = os.environ.get("LOCALAPPDATA")
        return (Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local") / name
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / name
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local" / "share") / name


def _seed_bundled_raw_data(raw_dir: Path) -> None:
    """Copy CSVs embedded by PyInstaller into the persistent raw-data folder,
    preserving any instrument subfolders in the bundle rather than flattening
    them (mirrors list_stored_datasets()'s recursive scan below)."""
    if not getattr(sys, "frozen", False):
        return
    bundle_root = getattr(sys, "_MEIPASS", None)
    if not bundle_root:
        return
    bundled_raw = Path(bundle_root) / "data" / "raw"
    if not bundled_raw.exists():
        return
    for ext in RAW_DATA_EXTENSIONS:
        for source in bundled_raw.rglob(f"*{ext}"):
            destination = raw_dir / source.relative_to(bundled_raw)
            if not destination.exists():
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                except OSError:
                    continue


def get_raw_data_dir() -> Path:
    """Return the persistent data/raw directory, creating and seeding it."""
    raw_dir = get_app_base_dir() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _seed_bundled_raw_data(raw_dir)
    return raw_dir


@dataclass
class StoredDataset:
    name: str   # path relative to data/raw/, POSIX-style (e.g. "EURUSD/EURUSD5.csv")
                # -- so instrument subfolders show up as "EURUSD/EURUSD5.csv"
                # rather than colliding on/hiding behind the bare filename.
    path: Path
    size_bytes: int


def list_stored_datasets() -> list[StoredDataset]:
    """
    Return every importable market-data file under data/raw/ (.csv, .tsv,
    .txt, .parquet -- see RAW_DATA_EXTENSIONS), newest first -- including
    files organized into subfolders (e.g. data/raw/EURUSD/EURUSD5.csv), not
    just ones directly in data/raw/ itself. `name` is the POSIX-style path
    relative to data/raw/, which both the desktop app's stored-dataset list
    and the web app's dropdown display as-is and (for the web app) submit
    back as the selection value -- `get_raw_data_dir() / name` resolves a
    subfolder entry correctly either way, since Path() accepts "/"
    separators on every platform including Windows.
    """
    raw_dir = get_raw_data_dir()
    files: list[Path] = []
    for ext in RAW_DATA_EXTENSIONS:
        files.extend(raw_dir.rglob(f"*{ext}"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        StoredDataset(name=f.relative_to(raw_dir).as_posix(), path=f, size_bytes=f.stat().st_size)
        for f in files
    ]


# A CSV under this many bytes cannot contain a header plus even one data
# row -- it's a placeholder/failed-export artifact, not real market data.
EMPTY_DATASET_BYTES = 32


# ---------------------------------------------------------------------------
# Row-count cache (PERF, Oct 2026). _quick_row_count used to re-read every
# byte of every dataset (hundreds of MB of CSV) each time it was called -- and
# the desktop app called it ~25 times during startup (once per tab's dataset
# list, plus the Data Center and Dashboard), which is what made launching take
# over a minute. A file's row count can only change if the file itself
# changes, so it is cached by (size, mtime) -- in memory for the session and
# in data/config/row_count_cache.json across launches -- and only recounted
# when the file actually changes.
# ---------------------------------------------------------------------------
_ROW_COUNT_CACHE: dict[str, list[int]] | None = None
_ROW_COUNT_CACHE_DIRTY = False
_ROW_COUNT_LOCK = threading.Lock()
ROWS_NOT_COUNTED = -2  # list views that skip counting report this instead of a number


def _row_cache_path() -> Path:
    return get_app_base_dir() / "data" / "config" / "row_count_cache.json"


def _load_row_cache() -> dict[str, list[int]]:
    global _ROW_COUNT_CACHE
    if _ROW_COUNT_CACHE is None:
        cache: dict[str, list[int]] = {}
        try:
            raw = json.loads(_row_cache_path().read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                cache = {k: v for k, v in raw.items() if isinstance(v, list) and len(v) == 3}
        except Exception:
            cache = {}  # missing/corrupt cache just means a one-time recount
        _ROW_COUNT_CACHE = cache
    return _ROW_COUNT_CACHE


def flush_row_count_cache() -> None:
    """Persist newly counted rows. Best-effort: never raises."""
    global _ROW_COUNT_CACHE_DIRTY
    with _ROW_COUNT_LOCK:
        if not _ROW_COUNT_CACHE_DIRTY or _ROW_COUNT_CACHE is None:
            return
        try:
            path = _row_cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(_ROW_COUNT_CACHE), encoding="utf-8")
            tmp.replace(path)
            _ROW_COUNT_CACHE_DIRTY = False
        except Exception:
            pass


def _cached_row_count(path: Path) -> int | None:
    """Cached count if the file is unchanged since it was counted, else None."""
    try:
        st = path.stat()
    except OSError:
        return None
    with _ROW_COUNT_LOCK:
        hit = _load_row_cache().get(str(path))
    if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
        return hit[2]
    return None


def _count_newlines(path: Path) -> int:
    """Newline count via large block reads -- an order of magnitude faster
    than iterating a file line by line in Python."""
    count = 0
    last = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            count += block.count(b"\n")
            last = block[-1:]
    if last and last != b"\n":
        count += 1  # final line with no trailing newline still counts as a line
    return count


def _quick_row_count(path: Path) -> int:
    """Cheap row count, without loading the file through pandas. Cached by
    (size, mtime) -- see the cache note above.

    .parquet is a binary columnar format, not newline-delimited text --
    pyarrow reads its row count straight from the footer metadata without
    decoding any column data.

    .zip/.7z archives are binary too, with no cheap metadata row count --
    returns -1 (a "count unknown, don't decompress just to render a number"
    sentinel) rather than a meaningless newline count over compressed bytes.
    """
    global _ROW_COUNT_CACHE_DIRTY
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_ARCHIVE_EXTENSIONS:
        return -1
    cached = _cached_row_count(path)
    if cached is not None:
        return cached
    try:
        st = path.stat()
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
                rows = pq.ParquetFile(path).metadata.num_rows
            except Exception:
                return 0
        else:
            rows = max(_count_newlines(path) - 1, 0)
    except OSError:
        return 0
    with _ROW_COUNT_LOCK:
        _load_row_cache()[str(path)] = [st.st_size, st.st_mtime_ns, rows]
        _ROW_COUNT_CACHE_DIRTY = True
    return rows


def list_datasets_by_instrument(count_rows: bool = True) -> list[dict]:
    """Groups list_stored_datasets() by its top-level data/raw/ subfolder
    (the instrument), for the Dashboard's "Market Data Library" card --
    this is what actually makes the data Owen already has on disk visible
    in the app, independent of whether any backtest has been run yet.

    `count_rows=False` (PERF) is for plain dataset PICKERS that only need
    names and an "empty" flag: it uses an already-cached count when there is
    one and otherwise reports ROWS_NOT_COUNTED instead of reading the file.
    Only a file small enough to hold no data at all is flagged empty without
    counting (size check), so nothing is hidden or mislabeled."""
    raw_dir = get_raw_data_dir()
    groups: dict[str, list[dict]] = {}
    for ds in list_stored_datasets():
        try:
            parts = ds.name.split("/")
            instrument = parts[0] if len(parts) > 1 else "(ungrouped)"
            try:
                if count_rows:
                    rows = _quick_row_count(ds.path)
                else:
                    cached = _cached_row_count(ds.path)
                    rows = cached if cached is not None else (
                        -1 if ds.path.suffix.lower() in SUPPORTED_ARCHIVE_EXTENSIONS else ROWS_NOT_COUNTED
                    )
            except Exception:
                # A single unreadable/locked/mid-write file must never
                # blank out every other dataset in the list -- every page
                # that renders "Use a previously stored dataset" depends
                # on this function returning something for every file
                # list_stored_datasets() already found on disk.
                rows = -1
            groups.setdefault(instrument, []).append({
                "name": parts[-1],
                "full_name": ds.name,
                "size_bytes": ds.size_bytes,
                "rows": rows,
                # rows == -1 is the "unknown, it's an archive" sentinel from
                # _quick_row_count -- never treat that as empty just because
                # it's not a positive count. A genuinely tiny/placeholder file
                # (archive or not) is still caught by the size_bytes check.
                "empty": ds.size_bytes <= EMPTY_DATASET_BYTES or rows == 0,
            })
        except Exception:
            # Same defensive reasoning one level up: whatever went wrong
            # with this one dataset entry, every other already-discovered
            # dataset must still show up.
            continue

    if count_rows:
        flush_row_count_cache()
    result = []
    for instrument in sorted(groups.keys()):
        files = sorted(groups[instrument], key=lambda x: x["name"])
        result.append({
            "instrument": instrument,
            "files": files,
            "file_count": len(files),
            "empty_count": sum(1 for f in files if f["empty"]),
            # -1 is _quick_row_count's "unknown, it's an archive" sentinel
            # (see its docstring) -- summed in naively, one archive in a
            # folder would silently knock 1 off the whole group's displayed
            # total. Excluded here so archives just don't contribute to the
            # row count instead of corrupting it.
            "total_rows": sum(f["rows"] for f in files if f["rows"] >= 0),
        })
    return result


def sanitize_stored_filename(filename: str) -> str:
    """Sanitize a user- (or browser-) supplied filename before it is ever
    joined onto the data directory. Strips path components ("../",
    absolute paths, Windows drive letters) via werkzeug's
    secure_filename. Raises ValueError if nothing usable remains (e.g.
    a filename that was pure traversal like "../../..")."""
    cleaned = secure_filename((filename or "").strip().replace("\\", "/").split("/")[-1])
    if not cleaned or cleaned in (".", ".."):
        raise ValueError(f"Rejected unsafe upload filename: {filename!r}")
    return cleaned


def _contained_within(directory: Path, candidate: Path) -> bool:
    """True iff candidate, fully resolved (symlinks and all), still lives
    inside directory. The second line of defense after sanitize: catches
    any symlink trickery that survives filename sanitizing."""
    try:
        base = directory.resolve()
        target = candidate.resolve()
    except OSError:
        return False
    return target == base or base in target.parents


def _unique_destination(raw_dir: Path, filename: str) -> Path:
    # Containment check on the final path: even a fully-sanitized name
    # must never resolve outside raw_dir (defense against symlink games
    # on raw_dir itself). Rejects with ValueError instead of writing.
    dest = raw_dir / filename
    if not dest.exists():
        if not _contained_within(raw_dir, dest):
            raise ValueError(f"Refusing to write outside the data directory: {filename!r}")
        return dest
    stem, suffix = dest.stem, dest.suffix
    i = 2
    while (raw_dir / f"{stem} ({i}){suffix}").exists():
        i += 1
    dest = raw_dir / f"{stem} ({i}){suffix}"
    if not _contained_within(raw_dir, dest):
        raise ValueError(f"Refusing to write outside the data directory: {filename!r}")
    return dest


def resolve_stored_dataset(name: str) -> Path | None:
    """Resolve a user-supplied stored-dataset name (e.g. the web edition's
    "existing_dataset" form field) to a real file inside data/raw/, or
    return None if the name is unsafe or doesn't exist. Callers should
    treat None as "no such dataset" (400/404), never as a path to open.
    This is the read-side companion to store_csv_bytes' write-side
    sanitizing: both funnel through sanitize_stored_filename +
    resolve-and-contain so a traversal payload can't read or write
    outside data/raw/."""
    try:
        cleaned = sanitize_stored_filename(name)
    except ValueError:
        return None
    candidate = get_raw_data_dir() / cleaned
    if not _contained_within(get_raw_data_dir(), candidate):
        return None
    return candidate if candidate.exists() else None


def store_csv_path(source_path: str | Path) -> Path:
    """Copy an external CSV into persistent data/raw/ unless already there."""
    source_path = Path(source_path)
    raw_dir = get_raw_data_dir()
    try:
        if source_path.resolve().parent == raw_dir.resolve():
            return source_path
    except OSError:
        pass
    dest = _unique_destination(raw_dir, source_path.name)
    shutil.copyfile(source_path, dest)
    return dest


def store_csv_bytes(content: bytes, filename: str) -> Path:
    """Write uploaded CSV bytes into persistent data/raw/."""
    raw_dir = get_raw_data_dir()
    # Write-side path-traversal fix (P0-7): sanitize the browser-supplied
    # filename BEFORE joining it onto the data directory; a traversal
    # payload like "../../app.py" becomes a harmless flat name or raises.
    dest = _unique_destination(raw_dir, sanitize_stored_filename(filename))
    dest.write_bytes(content)
    return dest
