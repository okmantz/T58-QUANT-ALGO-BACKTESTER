"""Run background worker processes at a lower OS priority so the UI never has to compete with them.

Search Lab, Evolution Lab, Full Pipeline, Speed Run, CPCV/PBO and the Monte Carlo pools each start
roughly os.cpu_count() worker processes. At normal priority those workers fight the app window (the
Tk desktop UI, the Flask server and the browser tab) for every core, which is exactly when the app
"lags" -- pages stop painting and buttons feel stuck while a job runs. Dropping the WORKERS to a
below-normal priority changes nothing when the machine is otherwise idle (a job still gets every core,
so run times are the same) but lets the operating system hand the UI the CPU first whenever there is
contention.

Applied automatically in every multiprocessing child (see app/__init__.py), so each existing pool is
covered without editing its call site, and any future pool is too. Never raises.
"""
from __future__ import annotations

import os
import sys

_applied = False

# Windows BELOW_NORMAL_PRIORITY_CLASS; POSIX niceness added to the inherited value.
_WIN_BELOW_NORMAL = 0x00004000
_POSIX_NICE_INCREMENT = 5


def lower_current_process_priority() -> bool:
    """Lower THIS process's CPU priority. Returns True when it was changed; idempotent."""
    global _applied
    if _applied or os.environ.get("T58_KEEP_WORKER_PRIORITY") == "1":
        return False
    try:
        if sys.platform.startswith("win"):
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            ok = bool(kernel32.SetPriorityClass(ctypes.c_void_p(kernel32.GetCurrentProcess()), _WIN_BELOW_NORMAL))
        else:
            os.nice(_POSIX_NICE_INCREMENT)
            ok = True
    except Exception:  # noqa: BLE001 -- priority is a nicety; never break a worker over it
        ok = False
    _applied = ok or _applied
    return ok


def lower_priority_if_worker() -> bool:
    """Lower priority only inside a multiprocessing worker (never in the main app process)."""
    try:
        import multiprocessing

        if multiprocessing.current_process().name == "MainProcess":
            return False
    except Exception:  # noqa: BLE001
        return False
    return lower_current_process_priority()
