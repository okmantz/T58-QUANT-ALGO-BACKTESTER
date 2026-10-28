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


def _child_count() -> int:
    if psutil is None:
        return 0
    try:
        return len(psutil.Process(os.getpid()).children(recursive=True))
    except Exception:  # noqa: BLE001
        return 0


def _clear_pure_caches() -> list[str]:
    """Drop in-memory caches that are rebuilt on demand. Touches NOTHING on disk: no strategies, reports, datasets, job history or search
    memory. Returns the names of the caches that were actually cleared."""
    cleared: list[str] = []

    def attempt(name, fn):
        try:
            fn()
            cleared.append(name)
        except Exception:  # noqa: BLE001
            pass

    def _resample():
        from app.data.timeframe_resample import clear_resample_cache
        clear_resample_cache()

    def _replay():
        from app.web import live_market
        live_market._replay_cache.clear()

    def _indicators():
        from app.strategy import indicator_cache
        indicator_cache.clear()

    def _v10():
        from app.strategy import indicators_v10
        indicators_v10._CACHE.clear()

    def _modules():
        from app.strategy.python import clear_module_cache
        clear_module_cache()

    def _macro():
        from app.strategy import macro_filter
        macro_filter._CACHE.clear()

    def _market_intel():
        from app.ai import market_intelligence
        market_intelligence._LOCAL_DATASET_CACHE.clear()

    attempt("resampled bars", _resample)
    attempt("replay buffers", _replay)
    attempt("indicator results", _indicators)
    attempt("extra indicators", _v10)
    attempt("compiled strategy modules", _modules)
    attempt("macro filter data", _macro)
    attempt("market-intelligence data", _market_intel)
    return cleared


def idle_cleanup(*, idle_unload_ollama: bool = False, min_job_age: float = 1800.0) -> dict[str, Any]:
    """One idle sweep (also callable on demand): worker pools closed, finished jobs compacted, caches dropped, heap trimmed.

    Safe by construction: it only releases memory the app can rebuild. A finished run whose result exists ONLY in memory (Discovery
    winners not yet saved, Optimize summaries, Validate results) is kept; only runs already saved to disk are evicted, and only after
    ``min_job_age`` seconds. Nothing on disk is ever deleted."""
    done: dict[str, Any] = {}
    done["workers_before"] = _child_count()
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

        comp = JOB_MANAGER.compact(min_job_age)
        done["jobs_pruned"] = comp["evicted"]
        done["jobs_slimmed"] = comp["slimmed"]
        done["jobs_protected"] = comp["protected"]
    except Exception:  # noqa: BLE001
        pass
    done["caches_cleared"] = _clear_pure_caches()
    if idle_unload_ollama:
        done["ollama_unloaded"] = unload_ollama()
    done.update(release_memory("idle"))
    time.sleep(0.4)  # let closed worker processes actually exit before the caller measures again
    done["workers_after"] = _child_count()
    return done


def _mb(b) -> str:
    b = float(b or 0)
    return f"{b / 1073741824:.1f} GB" if b >= 1073741824 else f"{b / 1048576:.0f} MB"


def free_memory(*, unload_ollama_model: bool = True, min_job_age: float = 1800.0) -> dict[str, Any]:
    """The Runs page 'Free memory now' button. Runs one safe cleanup and returns a plain-language report of what changed:

        {"headline": "Freed 412 MB ...", "freed_bytes": int, "before": {...}, "after": {...}, "steps": [str], "kept": [str], "advice": str|None}

    ``before``/``after`` are the numbers the Runs page shows (app memory incl. workers, free RAM, swap)."""
    def snap() -> dict[str, Any]:
        out = {"app": rss_bytes(), "app_with_workers": rss_bytes(True), "available": None, "swap_used": None}
        if psutil is not None:
            try:
                out["available"] = int(psutil.virtual_memory().available)
                out["swap_used"] = int(psutil.swap_memory().used)
            except Exception:  # noqa: BLE001
                pass
        return out

    before_ollama = ollama_status() if unload_ollama_model else []
    before = snap()
    done = idle_cleanup(idle_unload_ollama=unload_ollama_model, min_job_age=min_job_age)
    after = snap()
    freed = max(0, int(before["app_with_workers"]) - int(after["app_with_workers"]))
    ollama_freed = 0
    if unload_ollama_model and done.get("ollama_unloaded"):
        ollama_freed = sum(int(m.get("size_bytes") or 0) for m in (before_ollama or []))
    steps: list[str] = []
    wb, wa = int(done.get("workers_before") or 0), int(done.get("workers_after") or 0)
    if wb > wa:
        steps.append(f"Closed {wb - wa} idle worker process{'es' if wb - wa != 1 else ''}.")
    if done.get("jobs_pruned"):
        steps.append(f"Cleared {done['jobs_pruned']} finished run{'s' if done['jobs_pruned'] != 1 else ''} from memory (their reports are saved and still open from the Runs list).")
    if done.get("jobs_slimmed"):
        steps.append(f"Trimmed bulky detail from {done['jobs_slimmed']} finished run{'s' if done['jobs_slimmed'] != 1 else ''} (results kept).")
    if done.get("caches_cleared"):
        steps.append("Dropped temporary caches: " + ", ".join(done["caches_cleared"]) + " (they rebuild on demand).")
    if ollama_freed:
        names = ", ".join(str(m.get("name")) for m in before_ollama)
        steps.append(f"Unloaded the local AI model ({names}, about {_mb(ollama_freed)}); it reloads automatically the next time you use the assistant.")
    steps.append("Returned unused memory to the operating system.")
    kept = ["Your strategies, datasets, saved reports, run history and the Strategy Graveyard's search memory are never touched."]
    prot = done.get("jobs_protected") or []
    if prot:
        kept.append(f"Kept {len(prot)} finished run{'s' if len(prot) != 1 else ''} whose results are only in memory (not saved yet): "
                    + ", ".join(f"{p['tool']}" + (f" ({p['instrument']})" if p.get("instrument") else "") for p in prot[:5])
                    + (" ..." if len(prot) > 5 else "") + ". Open and save what you want from them; they are released on the next sweep after you do.")
    total_freed = freed + ollama_freed
    if psutil is None:
        headline, advice = "Cleanup ran, but memory figures are unavailable (install psutil to see them).", None
    elif total_freed >= 20 * 1048576:
        headline = f"Freed about {_mb(total_freed)}. The app now uses {_mb(after['app_with_workers'])}."
        advice = None
    else:
        headline = f"Already lean: the app is using {_mb(after['app_with_workers'])}, so there was little to free."
        advice = None
        if after.get("available") is not None and before.get("available") is not None:
            try:
                vm = psutil.virtual_memory()
                if vm.available < vm.total * _LOW_AVAILABLE_FRACTION:
                    advice = (f"Your computer is still low on free RAM ({_mb(vm.available)} of {_mb(vm.total)}), but the app is not the main user: "
                              "other programs (browser tabs, other apps) are. Closing some of them will help more than this button.")
            except Exception:  # noqa: BLE001
                pass
    return {"headline": headline, "freed_bytes": total_freed, "before": before, "after": after, "steps": steps, "kept": kept,
            "advice": advice, "detail": {k: v for k, v in done.items() if k != "jobs_protected"}}


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
                        idle_cleanup(min_job_age=7200.0)
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
