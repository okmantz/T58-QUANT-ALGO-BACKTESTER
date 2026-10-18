"""
Shared process-wide guardrails for T58's heavy, multi-worker-process
jobs (Search Lab, Evolution Lab, Full Pipeline batch, Speed Run).

Why this exists
----------------
Every one of those jobs independently spins up its own
ProcessPoolExecutor sized to (roughly) os.cpu_count() workers, and
each worker process loads its OWN full copy of the active market
data DataFrame (see app.search.batch_runner._init_worker and
app.evolution.engine._evo_init_worker). That's fine in isolation.

It is NOT fine if two or three of these jobs are started at the same
time in the same running app -- e.g. Evolution Lab left running
overnight, Full Pipeline batch started on top of it, then Speed Run
clicked on top of THAT. Each one independently claims the whole
machine's CPU count, so N simultaneous jobs try to hold N x
os.cpu_count() worker processes at once, each with its own copy of a
potentially large (months/years of 1-minute bars) DataFrame in
memory. On a real machine that is exactly the failure mode that looks
like "the whole desktop froze and then shut down" -- not a bug in any
one job, but the complete absence of any coordination between jobs
that each assume they own the whole machine.

This module provides two independent guardrails against that:

  HEAVY_JOB_GUARD    -- a simple named-slot lock. Only one heavy job
                        (by name) may be "active" at a time in this
                        process. Both the desktop GUI
                        (app.ui.main_window) and the web server
                        (app.web.server) import the same singleton,
                        so this protects whichever of the two entry
                        points is running -- it does NOT protect
                        against running the .exe AND the web server
                        at the same time from two separate OS
                        processes; there's no cheap way to guard
                        across process boundaries, so the UI-level
                        warning this enables is a strong hint, not an
                        absolute guarantee. If you run both, still
                        avoid stacking heavy jobs across them
                        yourself.

  safe_worker_count  -- caps a ProcessPoolExecutor's worker count so
                        that (workers x one worker's DataFrame memory
                        footprint) stays within a conservative budget
                        of AVAILABLE system memory, instead of
                        blindly using os.cpu_count() regardless of
                        how big the loaded dataset is. Falls back to
                        a conservative fixed assumption if psutil
                        isn't installed (it's an optional dependency
                        here, same as pywebview / MetaTrader5
                        elsewhere in this repo's requirements).
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Optional

import pandas as pd

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

# Conservative assumption used ONLY when psutil isn't installed and we
# therefore have no way to ask the OS how much RAM is actually free
# right now. Deliberately low -- better to under-parallelize on a
# beefy machine than to repeat the freeze this module exists to
# prevent.
_FALLBACK_AVAILABLE_BYTES = 3 * 1024 ** 3  # 3 GB

# Never plan to use more than this fraction of *currently available*
# memory for one job's worker pool -- leaves headroom for the GUI /
# Flask process itself, the OS, and (since this can't see other heavy
# jobs running in a different OS process) some slack for those too.
_MAX_MEMORY_FRACTION = 0.5

# Per-worker overhead multiplier on top of the raw DataFrame size --
# indicator columns, intermediate arrays, and the backtest engine's
# own bookkeeping cost.
#
# 2026-10-15 review fix: this used to be 1.6x the frame with no fixed term,
# which UNDER-counted a real worker. Measured on a 2.16M-row 1-minute NQ
# frame (123 MB): a worker also carries ~110-135 MB of interpreter + imported
# modules (numpy/pandas/the app's own packages), unpickles its own full copy
# of the data, and builds indicator/resampled copies on top. The review's
# working budget is therefore ~150 MB base + ~3x the frame per worker.
_WORKER_OVERHEAD_MULTIPLIER = 3.0
_WORKER_BASE_BYTES = 150 * 1024 ** 2

# Memory watchdog floor: when free RAM falls under this fraction of total
# RAM, long-running loops should stop launching new heavy work (see
# memory_pressure_ok / shrink_pool_for_memory_pressure below).
_MIN_FREE_MEMORY_FRACTION = 0.15


def _available_memory_bytes() -> int:
    if _HAS_PSUTIL:
        try:
            return int(psutil.virtual_memory().available)
        except Exception:
            pass
    return _FALLBACK_AVAILABLE_BYTES


def safe_worker_count(
    df: Optional[pd.DataFrame],
    requested: Optional[int] = None,
    max_candidates_in_flight: Optional[int] = None,
) -> int:
    """Returns a worker-process count that is safe to hand to a
    ProcessPoolExecutor given the size of `df` (the dataset every
    worker will load its own full copy of) and the memory actually
    available right now.

    requested: the caller's own preferred worker count (e.g.
        stage_cfg.workers, or None to mean "os.cpu_count()"). This is
        the UPPER bound -- safe_worker_count only ever returns this
        value or something SMALLER, never more workers than the
        caller wanted.
    max_candidates_in_flight: if the caller also can't usefully use
        more workers than it has work items for (e.g. don't spin up
        16 workers to process 3 candidates), pass that count here to
        additionally cap on it.

    Always returns at least 1.
    """
    if df is None or len(df) == 0:
        one_copy_bytes = 0.0
    else:
        try:
            one_copy_bytes = float(df.memory_usage(deep=True).sum())
        except Exception:
            one_copy_bytes = 0.0
    return safe_worker_count_for_bytes(
        one_copy_bytes, requested=requested, max_candidates_in_flight=max_candidates_in_flight,
    )


def safe_worker_count_for_bytes(
    one_worker_payload_bytes: float,
    requested: Optional[int] = None,
    max_candidates_in_flight: Optional[int] = None,
) -> int:
    """Same guardrail as safe_worker_count, but for callers that don't
    have a single DataFrame to measure -- e.g. app.optimize.walkforward_ga,
    whose worker initializer loads EVERY walk-forward fold's slice into
    each worker process, not one shared DataFrame. Pass the raw byte size
    of whatever one worker will hold in memory (sum across however many
    pieces that is); this applies the same available-memory budget and
    per-worker overhead multiplier safe_worker_count does.

    Always returns at least 1.
    """
    cpu_cap = requested if requested is not None else (os.cpu_count() or 2)
    cpu_cap = max(1, int(cpu_cap))
    if max_candidates_in_flight is not None:
        cpu_cap = max(1, min(cpu_cap, int(max_candidates_in_flight)))

    if one_worker_payload_bytes <= 0:
        return cpu_cap

    # Per-worker cost = fixed interpreter/module base + a multiple of the data it holds.
    one_copy_bytes = _WORKER_BASE_BYTES + float(one_worker_payload_bytes) * _WORKER_OVERHEAD_MULTIPLIER
    budget = _available_memory_bytes() * _MAX_MEMORY_FRACTION
    memory_cap = max(1, int(budget // one_copy_bytes))

    # Watchdog: when free RAM is already under the floor, size the pool down
    # again rather than adding load to a machine that is about to thrash.
    return shrink_pool_for_memory_pressure(max(1, min(cpu_cap, memory_cap)))


def _total_memory_bytes() -> int:
    if _HAS_PSUTIL:
        try:
            return int(psutil.virtual_memory().total)
        except Exception:
            pass
    return 0


def memory_pressure_ok(min_free_fraction: float = _MIN_FREE_MEMORY_FRACTION) -> bool:
    """Low-memory watchdog check: True while free RAM is at or above
    `min_free_fraction` of total RAM AND the machine is not leaning on swap
    (heavy swap use is what freezes a desktop long before free RAM hits zero).
    Returns True (assume fine) when psutil is unavailable. Callers that loop
    over heavy work should call this between units of work and stop
    submitting new work -- or shrink their pool -- when it returns False."""
    total = _total_memory_bytes()
    if total <= 0:
        return True
    if _available_memory_bytes() < total * float(min_free_fraction):
        return False
    try:
        sw = psutil.swap_memory()
        if sw.total > 0 and sw.used > sw.total * 0.5 and sw.used > 1024 ** 3:
            return False
    except Exception:
        pass
    return True


def shrink_pool_for_memory_pressure(current_workers: int, min_free_fraction: float = _MIN_FREE_MEMORY_FRACTION) -> int:
    """Watchdog helper: returns `current_workers` unchanged while memory is
    healthy, and half of it (never below 1) when free RAM is under the floor.
    Use it to size the NEXT batch of work, not to kill workers already running."""
    n = max(1, int(current_workers))
    if memory_pressure_ok(min_free_fraction):
        return n
    return max(1, n // 2)


class HeavyJobGuard:
    """Named-slot mutex: only one heavy multi-process job may be
    'active' at a time per process. Not a queue -- a second job that
    tries to acquire while one is active is simply refused (the
    caller decides what to tell the user)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_name: Optional[str] = None
        self._health_checks: dict[str, Callable[[], bool]] = {}

    def register_health_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        """Registers a zero-arg callable that reports whether job `name`
        is genuinely still active right now. try_acquire() consults this
        to auto-clear a stale slot if the job holding it stopped without
        ever calling release() itself. This exists because at least one
        job (Evolution Lab) only releases its slot when its own web
        status endpoint gets polled and notices the job has stopped --
        if nothing polls it (a backgrounded/closed browser tab, or the
        job's own thread getting wedged so it never even reaches
        "stopped"), every OTHER heavy job was refused indefinitely with
        no way to recover short of restarting the server. A name with no
        registered check behaves exactly as before (first-come lock,
        cleared only by an explicit release())."""
        with self._lock:
            self._health_checks[name] = check_fn

    def try_acquire(self, name: str) -> bool:
        with self._lock:
            if self._active_name is not None:
                check = self._health_checks.get(self._active_name)
                if check is not None:
                    try:
                        still_active = bool(check())
                    except Exception:
                        still_active = True  # a broken health check must never falsely free the slot
                    if not still_active:
                        self._active_name = None
            if self._active_name is not None:
                return False
            self._active_name = name
            return True

    def release(self, name: str) -> None:
        released = False
        with self._lock:
            if self._active_name == name:
                self._active_name = None
                released = True
        if released:
            # A heavy job just finished: hand its memory back (gc + heap trim) on a side thread so the caller isn't
            # delayed. This single hook covers Full Pipeline, Search Lab, Evolution Lab, WFO, Speed Run, etc.
            def _cleanup() -> None:
                try:
                    from app.orchestration.memory_hygiene import release_memory

                    release_memory(f"after:{name}")
                except Exception:  # noqa: BLE001
                    pass

            threading.Thread(target=_cleanup, name="t58-post-job-cleanup", daemon=True).start()

    @property
    def active_name(self) -> Optional[str]:
        with self._lock:
            return self._active_name


# Shared singleton -- imported by both app.ui.main_window and
# app.web.server so the two entry points respect the same guard
# whenever both happen to be running inside the same Python process.
# (The web server does not normally run inside the same OS process as
# the desktop .exe -- see module docstring for that limit.)
HEAVY_JOB_GUARD = HeavyJobGuard()

# Human-readable names for the guard's slot, shared by both UIs so the
# "X is already running" message always names the job consistently.
JOB_SEARCH_LAB = "Search Lab"
JOB_MULTI_INSTRUMENT_SEARCH = "Multi-Instrument Search"
JOB_EVOLUTION_LAB = "Evolution Lab"
JOB_MULTI_INSTRUMENT_EVOLUTION = "Multi-Instrument Evolution Lab"
JOB_FULL_PIPELINE = "Full Pipeline"
JOB_SPEED_RUN = "Speed Run"
JOB_MULTI_INSTRUMENT_SPEED_RUN = "Multi-Instrument Speed Run"
JOB_FORGE = "Forge Strategy"
# v9 (2026-10-06): this slot was missing entirely -- app/web/server.py's
# /quick-optimize/start route (and the v9 /recovery/quick-optimize route)
# referenced JOB_QUICK_OPTIMIZE, which was never defined, so EVERY Quick
# Optimize start raised NameError. The web test suite never caught it
# because no test POSTs to /quick-optimize/start.
JOB_QUICK_OPTIMIZE = "Quick Optimize"

# UPGRADE (Sep 2026 UI pass, round 2): these five don't spin up their own
# ProcessPoolExecutor the way the four above do, but Walk-Forward Opt/GA in
# particular re-run a full parameter search per fold (their own internal
# multi-generation population search, repeated n_folds times) -- easily as
# expensive as one of the four "big" jobs, just single-process. None of
# them were guarded before, so nothing stopped e.g. WFO and CPCV and
# Sensitivity all running at once on top of each other (or on top of a
# Search Lab run) and adding up to the exact same "the whole machine froze"
# failure mode this module was written to prevent -- they just didn't
# multiply by cpu_count() while doing it. They share the SAME
# HEAVY_JOB_GUARD slot as the four above (not a separate guard), so any one
# of these nine job types blocks any other from starting concurrently in
# this process.
JOB_WFO = "Walk-Forward Optimization"
JOB_WFGA = "Walk-Forward GA"
JOB_CPCV = "CPCV"
JOB_SENSITIVITY = "Sensitivity"
JOB_MULTI_OBJECTIVE = "Multi-Objective Optimization"
# UPGRADE (multi-market aggregate scoring): app.optimize.multi_market's
# search loop is single-process (no ProcessPoolExecutor of its own yet --
# see that module's docstring), but it runs len(markets) full backtests
# PER candidate, so it belongs in the same shared slot as the nine above,
# not left unguarded.
JOB_MULTI_MARKET = "Multi-Market Aggregate Search"
JOB_REGIME_MATRIX = "Regime Survival Matrix"
JOB_PARAMETER_ROBUSTNESS = "Parameter Stability / Robustness Map"
# UI pass (indicators/families/UI-parity round): PBO's own candidate-pool
# picker shares this same guard slot -- it's exactly as expensive as CPCV
# (same combinatorial-path machinery), just evaluating several candidates
# per path instead of one.
JOB_PBO = "Probability of Backtest Overfitting"
