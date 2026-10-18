"""One persistent spawn pool for the search stages of a Full Pipeline run (2026-10-17).

GA (step 2), embargoed walk-forward (step 4) and the DSR trials (step 6c) used to build their own
``ProcessPoolExecutor``, pay the spawn + app-import cost in every worker, and pickle the whole price frame into each
worker again. This module keeps ONE pool alive and gives each stage a cheap *handle*:

    handle = acquire_stage(workers, _my_worker_init, (df, risk, ...))   # None -> caller runs serially
    fut = handle.submit(_my_task, arg)                                   # same call shape as ProcessPoolExecutor
    handle.shutdown()                                                    # releases the stage; the pool stays alive

* A stage's initializer runs LAZILY inside a worker, the first time that worker sees a task of the new stage
  (the stage key travels with every task), so a worker process is spawned and imports the app once, not per stage.
* DataFrames in ``init_args`` are spilled ONCE to a temp pickle and loaded by each worker from disk, instead of
  being pickled into every task / every worker spawn.
* ``handle.shutdown(cancel_futures=True)`` (the user-Stop path) discards the underlying pool so no orphan workers
  keep running; the next ``acquire_stage`` builds a fresh one.
* An idle reaper and ``atexit`` close the pool, like ``app.monte_carlo.shared_pool``.

Any failure to build or use the pool is the CALLER's cue to run the same math serially (callers already do).
"""
from __future__ import annotations

import atexit
import multiprocessing
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

IDLE_SECONDS = 120.0

_lock = threading.RLock()
_pool: Optional[ProcessPoolExecutor] = None
_pool_workers = 0
_last_used = 0.0
_reaper: Optional[threading.Timer] = None


class _Spilled:
    """A DataFrame written to disk by the parent; workers load it by path."""

    def __init__(self, path: str) -> None:
        self.path = path


# ---- worker side ------------------------------------------------------------------------------------------------
_WORKER_STAGE: dict = {"key": None}


def _resolve(arg: Any) -> Any:
    if isinstance(arg, _Spilled):
        import pandas as pd

        return pd.read_pickle(arg.path)
    return arg


def _stage_call(key: str, init_fn: Callable, init_args: tuple, fn: Callable, args: tuple):
    """Runs in a worker: (re)initialise for this stage if the worker last served another one, then run the task."""
    if _WORKER_STAGE.get("key") != key:
        _WORKER_STAGE["key"] = None            # a failed init must not leave a half-initialised stage marked ready
        init_fn(*[_resolve(a) for a in init_args])
        _WORKER_STAGE["key"] = key
    return fn(*args)


# ---- parent side ------------------------------------------------------------------------------------------------
def _is_broken(pool) -> bool:
    return bool(getattr(pool, "_broken", False)) or bool(getattr(pool, "_shutdown_thread", False))


def _shutdown_locked(wait: bool = False, terminate: bool = False) -> None:
    global _pool, _pool_workers, _reaper
    if _reaper is not None:
        _reaper.cancel()
        _reaper = None
    pool = _pool
    _pool, _pool_workers = None, 0
    if pool is None:
        return
    try:
        pool.shutdown(wait=wait, cancel_futures=True)
    except Exception:  # noqa: BLE001
        pass
    if terminate:
        try:
            procs = list(getattr(pool, "_processes", {}).values())
        except Exception:  # noqa: BLE001
            procs = []
        for proc in procs:
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        for proc in procs:
            try:
                proc.join(timeout=5)
            except Exception:  # noqa: BLE001
                pass


def shutdown_stage_pool(wait: bool = False) -> None:
    with _lock:
        _shutdown_locked(wait)


def _arm_reaper(delay: float) -> None:
    global _reaper
    t = threading.Timer(delay, _reap)
    t.daemon = True
    t.start()
    _reaper = t


def _reap() -> None:
    global _reaper
    with _lock:
        _reaper = None
        if _pool is None:
            return
        idle = time.monotonic() - _last_used
        if idle >= IDLE_SECONDS:
            _shutdown_locked(False)
        else:
            _arm_reaper(IDLE_SECONDS - idle + 1.0)


def _get_pool(workers: int) -> Optional[ProcessPoolExecutor]:
    """The shared pool. Reused when its size is no larger than requested (and at least half of it); a pool larger
    than the memory guard allows, or far too small, is rebuilt."""
    global _pool, _pool_workers, _last_used
    with _lock:
        if _pool is not None and (_is_broken(_pool) or _pool_workers > workers or _pool_workers * 2 < workers):
            _shutdown_locked(False)
        if _pool is None:
            try:
                _pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
                _pool_workers = workers
            except Exception:  # noqa: BLE001
                _pool, _pool_workers = None, 0
                return None
            _arm_reaper(IDLE_SECONDS + 1.0)
        _last_used = time.monotonic()
        return _pool


class StageHandle:
    """ProcessPoolExecutor-shaped view of the shared pool for one stage."""

    def __init__(self, key: str, init_fn: Callable, init_args: tuple, spill_dir: Optional[Path], workers: int = 1) -> None:
        self._workers = workers
        self._key, self._init_fn, self._init_args, self._spill_dir = key, init_fn, init_args, spill_dir
        self._released = False

    def submit(self, fn: Callable, *args):
        pool = _get_pool(self._workers)           # also refreshes the idle clock; rebuilds a reaped/broken pool
        if pool is None:
            raise RuntimeError("shared stage pool unavailable")
        return pool.submit(_stage_call, self._key, self._init_fn, self._init_args, fn, args)

    def map(self, fn: Callable, iterable):
        futs = [self.submit(fn, a) for a in iterable]
        return (f.result() for f in futs)

    @property
    def _processes(self) -> dict:
        return dict(getattr(_pool, "_processes", {}) or {})

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """Normal end of stage: just release it (the pool stays warm). ``cancel_futures=True`` is the Stop path: the
        shared pool is terminated and rebuilt on the next request."""
        if cancel_futures:
            with _lock:
                _shutdown_locked(False, terminate=True)
        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._spill_dir is not None:
            shutil.rmtree(self._spill_dir, ignore_errors=True)

    def __enter__(self) -> "StageHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


def acquire_stage(workers: Optional[int], init_fn: Callable, init_args: tuple) -> Optional[StageHandle]:
    """A handle on the shared pool for one stage, or None when ``workers <= 1`` or the pool cannot be built."""
    if workers is None or int(workers) <= 1:
        return None
    workers = int(workers)
    spill_dir: Optional[Path] = None
    try:
        import pandas as pd

        packed = []
        for a in init_args:
            if isinstance(a, pd.DataFrame):
                if spill_dir is None:
                    spill_dir = Path(tempfile.mkdtemp(prefix="t58_stage_"))
                p = spill_dir / f"arg_{len(packed)}.pkl"
                a.to_pickle(p)
                packed.append(_Spilled(str(p)))
            else:
                packed.append(a)
        if _get_pool(workers) is None:
            raise RuntimeError("no pool")
        return StageHandle(uuid.uuid4().hex, init_fn, tuple(packed), spill_dir, workers)
    except Exception:  # noqa: BLE001 -- caller runs serially
        if spill_dir is not None:
            shutil.rmtree(spill_dir, ignore_errors=True)
        return None


atexit.register(shutdown_stage_pool)
