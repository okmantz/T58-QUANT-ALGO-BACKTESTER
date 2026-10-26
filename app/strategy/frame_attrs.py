"""Side channel for the non-scalar ``DataFrame.attrs`` entries indicators read (companion frames, Ext series, macro dicts).

pandas deep-copies ``attrs`` whenever two objects are combined (``pd.concat`` / arithmetic) and then compares the copies with ``==``;
a DataFrame inside ``attrs`` makes that comparison raise "truth value of a DataFrame is ambiguous". The v13 builders therefore run on
a shallow copy of the frame whose ``attrs`` hold only scalars, and read the heavy entries through ``frame_attr`` (set for the duration
of the call by ``indicators_v13_core.safe``). The context is thread-local and nests (a builder calling another builder)."""
from __future__ import annotations

import threading
from contextlib import contextmanager

import pandas as pd

_LOCAL = threading.local()
_HEAVY = (pd.DataFrame, pd.Series, dict, list, tuple, set)


def scalar_attrs(attrs: dict | None) -> dict:
    return {k: v for k, v in (attrs or {}).items() if not isinstance(v, _HEAVY)}


@contextmanager
def attrs_context(attrs: dict | None):
    prev = getattr(_LOCAL, "attrs", None)
    _LOCAL.attrs = attrs if attrs is not None else (prev or {})
    try:
        yield
    finally:
        _LOCAL.attrs = prev


def frame_attr(frame, key: str, default=None):
    """``attrs[key]`` for ``frame``: the active context first (the original attrs of the frame a builder was called with), else the frame's own."""
    ctx = getattr(_LOCAL, "attrs", None)
    if ctx and key in ctx:
        return ctx[key]
    return (getattr(frame, "attrs", {}) or {}).get(key, default)
