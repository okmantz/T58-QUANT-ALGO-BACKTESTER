"""Memory hygiene for long-running T58 sessions (Full Pipeline, batch runs, labs).

Why: a Full Pipeline run holds one DataFrame per instrument in the main process, spawns worker processes that each
hold their own copy, keeps finished job results/logs in memory, and (separately) Ollama keeps a model resident. On a
machine with little RAM the OS starts swapping and everything slows to a crawl, then the OOM killer ends the app.

What this module does (all best-effort, never raises):

* ``release_memory(reason)``  -- gc + give freed heap pages back to the OS (glibc ``malloc_trim``; Windows working-set trim).
* ``memory_status()``         -- RSS of this process (+ children), system available RAM, swap use, Ollama's resident models.
* ``memory_pressure()``       -- True when available RAM is low OR swap is being used heavily (swap use is what makes a
                                 PC unusable long before RAM hits zero).
* ``unload_ollama()``         -- asks a local Ollama to unload its models (keep_alive=0) so its multi-GB model stops
                                 consuming RAM/swap; it reloads automatically on the next AI request.
* ``start_idle_watchdog()``   -- daemon thread: when no heavy job is running it trims memory, closes the idle worker
                                 pool, prunes finished jobs, and records an RSS history so a genuine leak (RSS growing
                                 while idle) is visible at ``/api/memory`` instead of guessed at.
"""
from __future__ import annotations

import gc
import os
import sys
import threading
import time
from collections import deque
from typing import Any, Optional

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

_LOW_AVAILABLE_FRACTION = 0.20      # under 20% of RAM free -> pressure
_SWAP_USED_FRACTION = 0.35          # over 35% of swap in use  -> pressure
_SWAP_USED_BYTES = 2 * 1024 ** 3    # or more than 2 GB of swap in use
_history: deque = deque(maxlen=240)  # (ts, rss_bytes, idle)
_watchdog: Optional[threading.Thread] = None
_watchdog_stop = threading.Event()
_lock = threading.Lock()


def _trim_heap() -> None:
    try:
        if sys.platform.startswith("linux"):
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        elif sys.platform == "win32" and psutil is not None:
            import ctypes

            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def release_memory(reason: str = "") -> dict[str, Any]:
    """Collect garbage and return freed pages to the OS. Returns before/after RSS (bytes) for logging."""
    before = rss_bytes()
    try:
        for _ in range(2):
            gc.collect()
        try:
            import matplotlib.pyplot as plt  # only if already imported by something else

            if "matplotlib.pyplot" in sys.modules:
                plt.close("all")
        except Exception:  # noqa: BLE001
            pass
        _trim_heap()
    except Exception:  # noqa: BLE001
        pass
    return {"reason": reason, "rss_before": before, "rss_after": rss_bytes()}


def rss_bytes(include_children: bool = False) -> int:
    if psutil is None:
        return 0
    try:
        p = psutil.Process(os.getpid())
        total = p.memory_info().rss
        if include_children:
            for c in p.children(recursive=True):
                try:
                    total += c.memory_info().rss
                except Exception:  # noqa: BLE001
                    pass
        return int(total)
    except Exception:  # noqa: BLE001
        return 0


def ollama_status(timeout: float = 1.5) -> list[dict]:
    """Models a local Ollama currently holds in memory (empty when Ollama is not running)."""
    try:
        import json
        import urllib.request

        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        if not host.startswith("http"):
            host = "http://" + host
        with urllib.request.urlopen(host.rstrip("/") + "/api/ps", timeout=timeout) as r:  # noqa: S310 - local
            data = json.loads(r.read().decode("utf-8", "replace"))
        return [{"name": m.get("name"), "size_bytes": int(m.get("size", 0))} for m in data.get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def unload_ollama(timeout: float = 5.0) -> int:
    """Ask Ollama to unload every resident model (keep_alive=0). Returns how many were unloaded."""
    n = 0
    try:
        import json
        import urllib.request

        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        if not host.startswith("http"):
            host = "http://" + host
        for m in ollama_status():
            body = json.dumps({"model": m["name"], "keep_alive": 0}).encode()
            req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body, headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=timeout).read()  # noqa: S310 - local
                n += 1
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return n


def memory_status() -> dict[str, Any]:
    out: dict[str, Any] = {"psutil": psutil is not None, "pid": os.getpid()}
    if psutil is None:
        return out
    try:
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        out.update(rss_bytes=rss_bytes(), rss_with_children_bytes=rss_bytes(True), total_bytes=int(vm.total),
                   available_bytes=int(vm.available), available_percent=round(100.0 * vm.available / max(vm.total, 1), 1),
                   swap_total_bytes=int(sw.total), swap_used_bytes=int(sw.used), pressure=memory_pressure())
    except Exception:  # noqa: BLE001
        pass
    out["ollama_models"] = ollama_status()
    hist = list(_history)
    out["history"] = [{"t": t, "rss_mb": round(r / 1048576, 1), "idle": idle} for t, r, idle in hist[-60:]]
    idle_pts = [(t, r) for t, r, idle in hist if idle]
    if len(idle_pts) >= 6:
        first, last = idle_pts[0], idle_pts[-1]
        mins = max((last[0] - first[0]) / 60.0, 1e-6)
        growth = (last[1] - first[1]) / 1048576 / mins
        out["idle_growth_mb_per_min"] = round(growth, 2)
        out["possible_leak"] = bool(growth > 5.0 and (last[1] - first[1]) > 200 * 1048576)
    return out


def memory_pressure() -> bool:
    """True when the machine is low on RAM or leaning on swap. Swap use is the early-warning sign: a PC with 40% swap
    in use is already crawling even though 'free RAM' still looks fine."""
    if psutil is None:
        return False
    try:
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        if vm.available < vm.total * _LOW_AVAILABLE_FRACTION:
            return True
        if sw.total > 0 and (sw.used > sw.total * _SWAP_USED_FRACTION and sw.used > 512 * 1024 ** 2):
            return True
        return sw.used > _SWAP_USED_BYTES
    except Exception:  # noqa: BLE001
        return False


def _heavy_job_active() -> bool:
    try:
        from app.orchestration.resource_guard import HEAVY_JOB_GUARD

        return HEAVY_JOB_GUARD.active_name is not None
    except Exception:  # noqa: BLE001
        return False


def idle_cleanup(*, idle_unload_ollama: bool = False) -> dict[str, Any]:
    """One idle sweep (also callable on demand): worker pools closed, finished jobs pruned, heap trimmed."""
    done: dict[str, Any] = {}
    try:
        from app.orchestration import stage_pool

        stage_pool.shutdown_stage_pool(wait=False)
        done["stage_pool"] = "closed"
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.monte_carlo import shared_pool

        fn = getattr(shared_pool, "shutdown_shared_mc_pool", None)
        if callable(fn):
            fn()
            done["mc_pool"] = "closed"
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.web.job_manager import JOB_MANAGER

        done["jobs_pruned"] = JOB_MANAGER.prune(30 * 60)
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.data.timeframe_resample import clear_resample_cache

        clear_resample_cache()
        done["resample_cache"] = "cleared"
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.web import live_market

        live_market._replay_cache.clear()
        done["replay_cache"] = "cleared"
    except Exception:  # noqa: BLE001
        pass
    if idle_unload_ollama:
        done["ollama_unloaded"] = unload_ollama()
    done.update(release_memory("idle"))
    return done


def start_idle_watchdog(interval: float = 60.0, idle_after: float = 180.0, ollama_idle_after: float = 900.0) -> None:
    """Start (once) the daemon that keeps an idle app small. Safe to call repeatedly."""
    global _watchdog
    with _lock:
        if _watchdog is not None and _watchdog.is_alive():
            return
        _watchdog_stop.clear()

        def loop() -> None:
            idle_since: Optional[float] = None
            cleaned_at = 0.0
            ollama_done = False
            while not _watchdog_stop.wait(interval):
                try:
                    busy = _heavy_job_active()
                    now = time.time()
                    _history.append((now, rss_bytes(), not busy))
                    if busy:
                        idle_since, ollama_done = None, False
                        if memory_pressure():
                            release_memory("pressure-while-busy")
                            try:
                                from app.data.timeframe_resample import clear_resample_cache

                                clear_resample_cache()
                            except Exception:  # noqa: BLE001
                                pass
                        continue
                    idle_since = idle_since or now
                    idle_for = now - idle_since
                    if idle_for >= idle_after and now - cleaned_at >= 600:
                        idle_cleanup()
                        cleaned_at = now
                    if idle_for >= ollama_idle_after and not ollama_done:
                        unload_ollama()
                        ollama_done = True
                except Exception:  # noqa: BLE001
                    pass

        _watchdog = threading.Thread(target=loop, name="t58-memory-watchdog", daemon=True)
        _watchdog.start()


def stop_idle_watchdog() -> None:
    _watchdog_stop.set()
