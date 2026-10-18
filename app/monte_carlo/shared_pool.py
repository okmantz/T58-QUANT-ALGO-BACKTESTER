"""One persistent spawn process pool for Monte Carlo (2026-10-15 audit).

Full Pipeline used to create a pool for the baseline Monte Carlo, shut it down, create another for the
final one, shut it down, and again for the seed-stability runs. Each spawn re-imports the whole app in every
worker (seconds per worker, much longer on Windows). This module keeps ONE pool alive for the process:

    pool = get_shared_mc_pool(workers)      # None when workers <= 1 or a pool cannot be made
    run_monte_carlo(..., pool=pool)         # stateless chunk tasks, bit-identical results

* the pool is created lazily (ProcessPoolExecutor starts its workers on first submit, so nothing is spawned
  at all when the vectorized kernel handles every Monte Carlo call);
* it is reused across calls AND across pipeline runs, recreated when the requested worker count changes or
  the pool is broken, and shut down at interpreter exit;
* an idle reaper shuts it down after ``IDLE_SECONDS`` without a request, so idle workers never hold RAM
  against the optimizer's own worker pool during the long search steps.

Callers never shut the pool down themselves.
"""
from __future__ import annotations

import atexit
import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

IDLE_SECONDS = 120.0

_lock = threading.RLock()
_pool: Optional[ProcessPoolExecutor] = None
_pool_workers = 0
_last_used = 0.0
_reaper: Optional[threading.Timer] = None


def _is_broken(pool) -> bool:
    return bool(getattr(pool, "_broken", False)) or bool(getattr(pool, "_shutdown_thread", False))


def _shutdown_locked(wait: bool = False) -> None:
    global _pool, _pool_workers, _reaper
    if _reaper is not None:
        _reaper.cancel()
        _reaper = None
    if _pool is not None:
        try:
            _pool.shutdown(wait=wait, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
    _pool, _pool_workers = None, 0


def shutdown_shared_mc_pool(wait: bool = False) -> None:
    with _lock:
        _shutdown_locked(wait)


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


def _arm_reaper(delay: float) -> None:
    global _reaper
    t = threading.Timer(delay, _reap)
    t.daemon = True
    t.start()
    _reaper = t


def get_shared_mc_pool(workers: Optional[int]):
    """The process-wide pool sized ``workers``, or None when parallel MC is pointless/unavailable."""
    global _pool, _pool_workers, _last_used
    if workers is None or int(workers) <= 1:
        return None
    workers = int(workers)
    with _lock:
        if _pool is not None and (_pool_workers != workers or _is_broken(_pool)):
            _shutdown_locked(False)
        if _pool is None:
            try:
                _pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
                _pool_workers = workers
            except Exception:  # noqa: BLE001 -- serial / vectorized MC is always available
                _pool, _pool_workers = None, 0
                return None
            _arm_reaper(IDLE_SECONDS + 1.0)
        _last_used = time.monotonic()
        return _pool


atexit.register(shutdown_shared_mc_pool)
