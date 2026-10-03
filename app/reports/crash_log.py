"""
Process-wide crash logging for T58's long-running background jobs.

Evolution Lab, Full Pipeline, and Speed Run all run on background
threads, sometimes for hours, unattended (overnight). Before this
module existed, an unhandled exception in one of those threads -- or
the whole process dying (OOM kill, a worker crash taking a
ProcessPoolExecutor's parent down with it, etc.) -- left NOTHING on
disk to explain what happened: the only record was whatever had
scrolled past in the Tkinter log widget or the Flask job's in-memory
log, which is gone the moment the process exits. This writes every
crash straight to a plain text file the moment it happens,
independent of the GUI event loop or any in-memory log, so "the app
crashed overnight and I have no idea why" has an actual answer next
time: check data/logs/crash_log.txt (next to the .exe, or under your
user AppData folder -- see app.data.storage.get_app_base_dir).
"""
from __future__ import annotations

import logging
import logging.handlers
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.data.storage import get_app_base_dir

# P2-8 (Oct 2026): everything that used to be print() or an unrotated
# append-only file now goes through standard logging with rotation:
# 5 MB per file, 3 backups. crash_log.txt keeps its name and location
# (data/logs/crash_log.txt) so old runbooks still point at the right
# file -- it just rotates now instead of growing forever.
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


def logs_dir() -> Path:
    p = get_app_base_dir() / "data" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def crash_log_path() -> Path:
    return logs_dir() / "crash_log.txt"


def setup_rotating_logging(name: str = "t58") -> logging.Logger:
    """Configure process-wide logging once: a RotatingFileHandler
    (LOG_MAX_BYTES x LOG_BACKUP_COUNT) on data/logs/<name>.log plus a
    plain stderr handler, idempotent across repeated calls. Entry
    points (run_app.py / run_web.py / cli.py) call this at startup;
    every other module just does logging.getLogger(__name__)."""
    logger = logging.getLogger(name)
    if getattr(logger, "_t58_rotating_configured", False):
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir() / f"{name}.log",
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger._t58_rotating_configured = True  # noqa: SLF001 -- same-module idempotency flag
    return logger


_crash_logger: logging.Logger | None = None
_crash_logger_lock = threading.Lock()


def _get_crash_logger() -> logging.Logger:
    """The crash log as a rotating logger on the SAME crash_log.txt path
    the old append-only writer used -- public behavior (function names,
    file location) is unchanged; the file now rotates."""
    global _crash_logger
    with _crash_logger_lock:
        if _crash_logger is None:
            _crash_logger = logging.getLogger("t58.crash")
            _crash_logger.setLevel(logging.ERROR)
            handler = logging.handlers.RotatingFileHandler(
                crash_log_path(),
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            _crash_logger.addHandler(handler)
            _crash_logger.propagate = False
        return _crash_logger


_WRITE_LOCK = threading.Lock()


def log_crash(component: str, exc: Optional[BaseException] = None, extra: Optional[str] = None) -> Path:
    """Appends a timestamped crash record to the crash log and returns
    its path. Safe to call from any thread; never raises -- a failure
    to log a crash must never itself crash the caller.

    exc: pass the caught exception directly when you have it (keeps
        its own traceback rather than whatever is on the stack at the
        point log_crash is called). If omitted, falls back to
        traceback.format_exc() -- only meaningful when called from
        inside an `except:` block.
    """
    # P2-8 (Oct 2026): the record now goes through the rotating crash
    # logger instead of a raw append to crash_log.txt -- same file, same
    # content shape, but it rotates (5 MB x 3) instead of growing forever.
    try:
        ts = datetime.now(timezone.utc).isoformat()
        if exc is not None:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        else:
            tb = traceback.format_exc()
        lines = [f"\n{'=' * 70}\n[{ts}] {component}\n{'=' * 70}\n"]
        if extra:
            lines.append(extra.rstrip() + "\n")
        lines.append(tb)
        _get_crash_logger().error("".join(lines))
    except Exception:
        pass  # logging a crash must never itself raise
    return crash_log_path()


_HOOK_INSTALLED = False
_HOOK_LOCK = threading.Lock()


def install_thread_excepthook() -> None:
    """Installs a process-wide threading.excepthook that logs any
    exception which kills a background thread (daemon or not) to the
    crash log before Python's default hook runs (which, for a frozen
    .exe with no console attached, otherwise silently swallows it).
    Call this once, as early as possible, from each entry point
    (run_app.py / run_web.py) -- idempotent, safe to call more than
    once."""
    global _HOOK_INSTALLED
    with _HOOK_LOCK:
        if _HOOK_INSTALLED:
            return
        default_hook = threading.excepthook

        def _hook(args) -> None:
            try:
                log_crash(f"Unhandled exception in thread {args.thread.name!r}", exc=args.exc_value)
            except Exception:
                pass
            try:
                default_hook(args)
            except Exception:
                pass

        threading.excepthook = _hook
        _HOOK_INSTALLED = True
