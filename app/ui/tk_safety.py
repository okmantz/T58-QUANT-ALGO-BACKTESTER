"""
Desktop stability layer (Oct 2026).

Three independent safety nets, installed once from main_window.launch():

1. THREAD-SAFE WIDGET UPDATES. The app starts ~70 background worker threads
   (Full Pipeline, Search Lab, Evolution Lab, Speed Run, ...). Many of them
   update widgets directly (progress text, labels, status variables). Tk is
   single-threaded: a Tcl call made from a worker thread is either marshalled
   to the main thread with the worker BLOCKED until the main thread answers
   (can deadlock the moment the main thread waits on anything), or -- on some
   builds / when a Tk variable is touched off-thread -- aborts the whole
   process ("Tcl_AsyncDelete: async handler deleted by the wrong thread").
   The wrappers below turn the common *mutating* calls (configure, Text
   insert/delete/see, Listbox insert/delete, Variable.set, progressbar
   start/stop) made from a non-main thread into fire-and-forget entries on a
   queue that the main thread drains every ~25ms. Main-thread calls are
   untouched (zero overhead), and query calls (get/cget/...) are never
   deferred, so nothing that needs a return value changes behaviour.

2. BOUNDED LOG WIDGETS. A multi-hour run appends every progress line to a
   Text widget forever; the widget (and Tk's memory) grows without limit and
   the UI gets slower the longer a job runs. Text.insert now trims the oldest
   lines once a widget passes MAX_TEXT_LINES.

3. CRASH CAPTURE. Tk callback exceptions, uncaught exceptions on the main
   thread, and -- via faulthandler -- hard native crashes (segfault / abort)
   are all written to data/logs/crash_log.txt, so "the app crashed" always
   leaves an answer behind.
"""
from __future__ import annotations

import functools
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

MAX_TEXT_LINES = 6000       # trim a log Text once it grows past this many lines
TRIM_TO_LINES = 4500        # ...down to this many
_TRIM_CHECK_EVERY = 40      # check line count every N inserts (it is a Tcl call)
_DRAIN_INTERVAL_MS = 25
_DRAIN_BUDGET_S = 0.015     # never spend more than ~15ms of a tick on the queue

_pending: "queue.Queue[tuple]" = queue.Queue()
_installed = False
_fault_file = None  # keep the faulthandler log file object alive for the process lifetime


def _log(component: str, exc: BaseException | None = None, extra: str | None = None) -> None:
    try:
        from app.reports.crash_log import log_crash
        log_crash(component, exc, extra)
    except Exception:
        pass


def _on_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


# ---------------------------------------------------------------- 1. marshalling
def _defer(orig):
    """Wrap a mutating widget/variable method so a call from a worker thread is
    queued for the main thread instead of touching Tcl directly."""
    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        if _on_main_thread():
            return orig(self, *args, **kwargs)
        _pending.put((orig, self, args, kwargs))
        return None
    wrapper._t58_wrapped = True
    return wrapper


def _defer_configure(orig):
    """configure()/config() both sets options and (with no/str args) QUERIES
    them. Only the setting form is deferred; a query needs its return value."""
    @functools.wraps(orig)
    def wrapper(self, cnf=None, **kw):
        if _on_main_thread():
            return orig(self, cnf, **kw)
        is_set = bool(kw) or isinstance(cnf, dict)
        if not is_set:
            return orig(self, cnf, **kw)
        _pending.put((orig, self, (cnf,), kw))
        return None
    wrapper._t58_wrapped = True
    return wrapper


def call_soon(fn) -> None:
    """Run `fn()` on the Tk main thread as soon as possible. Safe to call from
    any thread (a worker thread never touches Tcl -- it only enqueues). Use
    this instead of `root.after(0, fn)` in worker threads."""
    if _on_main_thread():
        fn()
    else:
        _pending.put((lambda _w, f=fn: f(), None, (), {}))


def _drain(root) -> None:
    deadline = time.monotonic() + _DRAIN_BUDGET_S
    try:
        while time.monotonic() < deadline:
            try:
                fn, widget, args, kwargs = _pending.get_nowait()
            except queue.Empty:
                break
            try:
                fn(widget, *args, **kwargs)
            except tk.TclError:
                pass  # widget was destroyed before its queued update ran -- harmless
            except Exception as exc:  # noqa: BLE001 -- one bad update must not stop the pump
                _log("tk_safety: queued widget update failed", exc)
    finally:
        try:
            # Backlog? Come back immediately; otherwise idle at the normal interval.
            root.after(1 if not _pending.empty() else _DRAIN_INTERVAL_MS, lambda: _drain(root))
        except Exception:
            pass  # root destroyed -- app is exiting


def _install_safe_variable_finalizer() -> None:
    """A Tk variable's __del__ makes Tcl calls. If its last reference is dropped
    on a worker thread, Python runs that finalizer THERE -- the classic cause of
    "Tcl_AsyncDelete: async handler deleted by the wrong thread" aborts. Run it
    on the main thread instead."""
    orig = tk.Variable.__dict__["__del__"]
    if getattr(orig, "_t58_wrapped", False):
        return

    def _run(variable):
        try:
            orig(variable)
        except Exception:
            pass  # interpreter already gone / variable already unset

    def safe_del(self):
        if _on_main_thread():
            _run(self)
        else:
            _pending.put((lambda _w, v=self: _run(v), None, (), {}))

    safe_del._t58_wrapped = True
    tk.Variable.__del__ = safe_del


def _install_marshalling() -> None:
    _install_safe_variable_finalizer()
    for owner, name, wrap in (
        (tk.Misc, "configure", _defer_configure),
        (tk.Misc, "config", _defer_configure),
        (tk.Text, "delete", _defer),
        (tk.Text, "see", _defer),
        (tk.Listbox, "insert", _defer),
        (tk.Listbox, "delete", _defer),
        (tk.Variable, "set", _defer),
        (tk.BooleanVar, "set", _defer),
        (ttk.Progressbar, "start", _defer),
        (ttk.Progressbar, "stop", _defer),
    ):
        current = owner.__dict__.get(name)
        if current is not None and not getattr(current, "_t58_wrapped", False):
            setattr(owner, name, wrap(current))


# ---------------------------------------------------------------- 2. bounded Text
def _install_bounded_text() -> None:
    orig_insert = tk.Text.__dict__["insert"]
    if getattr(orig_insert, "_t58_wrapped", False):
        return
    counter = {"n": 0}

    @functools.wraps(orig_insert)
    def insert(self, index, chars, *args):
        if not _on_main_thread():
            _pending.put((insert, self, (index, chars) + args, {}))
            return None
        result = orig_insert(self, index, chars, *args)
        counter["n"] += 1
        if counter["n"] % _TRIM_CHECK_EVERY == 0:
            try:
                if str(self.cget("state")) == "normal":
                    lines = int(self.index("end-1c").split(".")[0])
                    if lines > MAX_TEXT_LINES:
                        self.delete("1.0", f"{lines - TRIM_TO_LINES}.0")
            except tk.TclError:
                pass
        return result

    insert._t58_wrapped = True
    tk.Text.insert = insert


# ---------------------------------------------------------------- 3. crash capture
def _tk_callback_exception(exc_type, exc, tb):
    _log("Tk callback exception", exc)


def _install_crash_capture() -> None:
    global _fault_file
    try:
        import faulthandler
        from app.reports.crash_log import crash_log_path
        # Opened in append mode and kept open for the whole process: faulthandler
        # needs a live file descriptor at the moment of a native crash.
        _fault_file = open(crash_log_path(), "a", buffering=1, encoding="utf-8")
        _fault_file.write("\n--- session start (native crash tracebacks are written below this line) ---\n")
        faulthandler.enable(file=_fault_file, all_threads=True)
    except Exception:
        pass

    previous = sys.excepthook

    def _excepthook(exc_type, exc, tb):
        _log("Uncaught exception", exc)
        try:
            previous(exc_type, exc, tb)
        except Exception:
            pass

    sys.excepthook = _excepthook


_drain_root = None


def install(root) -> None:
    """Install all three safety nets. Safe to call more than once; never raises.

    The patches themselves are process-wide and applied once. The queue-drain
    loop and the Tk callback-exception handler are per ROOT window, so calling
    install() with a new root (e.g. after the previous one was destroyed) binds
    them to that root -- otherwise queued updates would never be applied."""
    global _installed, _drain_root
    try:
        if not _installed:
            _install_bounded_text()
            _install_marshalling()
            _install_crash_capture()
            _installed = True
        root.report_callback_exception = _tk_callback_exception
        if root is not _drain_root:
            _drain_root = root
            root.after(_DRAIN_INTERVAL_MS, lambda: _drain(root))
    except Exception as exc:  # noqa: BLE001 -- stability aids must never stop the app starting
        _log("tk_safety.install failed", exc)
