"""T58 Quant Algo Backtester."""
# Responsiveness (2026-10-17 perf pass) -- both pieces are best-effort and can never break an import:
#
# 1. Background worker processes (search / evolution / pipeline / Monte Carlo pools) run at a lower OS
#    priority, so the app window stays smooth while a job uses every core. See app/_worker_priority.py.
#    Spawn-started workers (Windows, macOS, the stage pool) hit this on import; POSIX fork-started
#    workers via the at-fork hook. Set T58_KEEP_WORKER_PRIORITY=1 to turn it off.
#
# 2. In the MAIN process (the Tk window / the Flask server) the interpreter's thread switch interval is
#    cut from 5 ms to 2 ms. Backtests that run on background threads otherwise hold the GIL for up to
#    5 ms at a time, and every UI event or page request needs it several times -- that is the "it
#    freezes for a moment while something runs" feeling. Override with T58_SWITCH_INTERVAL (seconds).
try:
    import os as _os
    import sys as _sys

    from app._worker_priority import lower_current_process_priority as _lower_priority
    from app._worker_priority import lower_priority_if_worker as _lower_priority_if_worker

    if not _lower_priority_if_worker():
        _interval = float(_os.environ.get("T58_SWITCH_INTERVAL", "0.002"))
        if 0.0005 <= _interval < _sys.getswitchinterval():
            _sys.setswitchinterval(_interval)

    if hasattr(_os, "register_at_fork"):
        _os.register_at_fork(after_in_child=_lower_priority)
except Exception:  # noqa: BLE001 -- never let an optimisation break import of the package
    pass
