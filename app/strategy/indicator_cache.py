"""
Persistent indicator/feature cache -- keyed by (data, indicator, params).

Search Lab, Evolution Lab, and Iterative Refinement all re-backtest the
SAME underlying market data hundreds to tens of thousands of times per run
(one Stage 1 candidate, one GA/surrogate generation member, one walk-forward
fold, one robustness neighbor...). A huge fraction of those candidates
share the exact same indicator + period + column combination -- e.g. every
`mtf_pullback` candidate in a parameter grid still computes EMA(20) the
same way, and a GA/surrogate search re-proposes similar gene values
generation after generation. Recomputing EMA/RSI/ATR/etc. from scratch for
every single one of those is pure waste.

This module is a small process-local cache (module-level dict) sitting in
front of app.strategy.indicators.build_indicator_series. It is
DELIBERATELY process-local, not cross-process/shared/disk-backed:

  - Search Lab / Evolution Lab workers already load the market data ONCE
    per worker process (see app.search.batch_runner._init_worker /
    app.evolution.engine._evo_init_worker) and keep it fixed for the
    worker's entire lifetime -- so a process-local cache already covers
    every candidate that worker will ever backtest in this run.
  - Avoids any inter-process serialization cost (pickling indicator
    Series across a process pool would likely cost MORE than just
    recomputing them).
  - Avoids any staleness risk from a shared/disk cache outliving the
    dataset it was computed from.

Cache key includes a cheap fingerprint of the DataFrame (id() + length +
first/last timestamp) in addition to the indicator/period/column/lookback,
so if a worker process is ever reused across two different datasets (a
different Python object happening to receive the same `id()` after the
first one is garbage collected -- rare but possible), a shape/timestamp
mismatch still forces a fresh computation instead of silently returning
another dataset's indicator values.
"""
from __future__ import annotations

import os
import threading
import weakref
from typing import Callable

import pandas as pd

# Entry-count safety cap, kept as a belt-and-suspenders bound. This is NOT
# the primary guardrail any more -- see _MAX_BYTES below. 20,000 entries
# was sized assuming small series; on a multi-year 1-minute dataset (the
# normal case for Search Lab / Evolution Lab / Full Pipeline / Quick
# Optimize, all of which hit this cache), a single cached Series is
# roughly rows * 8 bytes -- ~2.35M rows is ~18-19MB PER ENTRY. 20,000 such
# entries is ~370GB before this cache ever cleared itself, which is the
# root cause behind the "unable to allocate N MiB" MemoryErrors and the
# "process was terminated abruptly" (OOM-killed) crashes seen in Search
# Lab, Quick Optimize, Full Pipeline, and Forge: each worker process's own
# resident memory grows roughly unbounded over the course of one run as
# more distinct indicator/period combinations get cached, on top of the
# market data copy safe_worker_count already budgets for. On cap, the
# whole cache is cleared rather than doing LRU bookkeeping -- cheap, and a
# cleared cache just means the next few candidates recompute once, not a
# correctness issue.
_MAX_ENTRIES = 20_000

# Primary guardrail: total cache size in bytes, per worker process. Sized
# conservatively (well under one worker's share of safe_worker_count's own
# memory budget) since this cache sits ON TOP of that budget, not instead
# of it -- safe_worker_count only accounts for each worker's copy of the
# raw market DataFrame, not the indicator series this module accumulates
# on top of it over a run.
_MAX_BYTES = 256 * 1024 * 1024  # 256 MB

_CACHE: dict[tuple, pd.Series] = {}
_CACHE_BYTES = 0
_LOCK = threading.Lock()
_HITS = 0
_MISSES = 0

# S6 (2026-10-19) slice reuse -- OPT-IN, OFF by default (2026-10-19 audit fix).
#
# The idea: when a frame is a prefix of a frame whose indicator Series is already
# cached (walk-forward train folds, in-sample slices), return that cached Series
# sliced instead of recomputing. It shipped ON and keyed only on TIMESTAMPS, which
# is unsound in two ways, both of which turned the CI test suite red:
#   1. Different datasets that share a timestamp grid (every synthetic test frame,
#      and e.g. two instruments on the same session) silently received another
#      dataset's indicator values.
#   2. app.strategy.lookahead_check.check_for_lookahead feeds `df.iloc[:cut]`
#      prefixes of a cached full frame through this cache. Slicing hands back the
#      FULL frame's values for the prefix, so an indicator that peeks at future
#      bars would look perfectly causal and the detector could never fire.
# It is therefore disabled unless T58_INDICATOR_CACHE_SLICING=1 is set explicitly.
# When enabled it is now conservative: PREFIX slices only (position 0, where the
# values are exactly what a cold computation yields for a causal indicator), and
# only after the actual price/volume data is verified identical to the still-alive
# parent frame (held via weakref, so no extra memory is retained).
_SLICE_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
_SLICE_INDEX: dict[tuple, list[tuple[tuple, "weakref.ReferenceType"]]] = {}


def slicing_enabled() -> bool:
    return os.environ.get("T58_INDICATOR_CACHE_SLICING", "0").strip().lower() in ("1", "true", "yes", "on")


def _series_nbytes(s: pd.Series) -> int:
    try:
        return int(s.memory_usage(deep=True))
    except Exception:  # noqa: BLE001 -- never let a sizing failure break caching
        try:
            return int(s.nbytes)
        except Exception:  # noqa: BLE001
            return 0


def _frame_fingerprint(frame: pd.DataFrame) -> tuple:
    try:
        n = len(frame)
        if n and "timestamp" in frame.columns:
            first_ts = frame["timestamp"].iloc[0]
            last_ts = frame["timestamp"].iloc[-1]
        else:
            first_ts = last_ts = None
        # UPGRADE (multi-market aggregate scoring bug fix): id() is only
        # guaranteed unique among objects alive AT THE SAME TIME -- once a
        # short-lived DataFrame is garbage-collected, Python can and does
        # reuse its memory address for an unrelated new object. Two
        # DataFrames covering the same session/bar-count range (e.g. two
        # CME futures like ES and GC on the same trading day, or just two
        # candidates' temporary frames created and freed in sequence) can
        # therefore share (id, n, first_ts, last_ts) despite having
        # completely different price data -- confirmed directly: running
        # the exact same strategy across several same-length, same-range
        # markets back-to-back (app.optimize.multi_market's whole reason
        # to exist) returned another market's cached indicator values for
        # one of them often enough to make every optimizer mode's
        # "reproducible given the same seed" guarantee fail. A cheap hash
        # of the actual first/last close price closes the gap: it costs
        # two scalar reads, not a full-column hash, and two genuinely
        # different price series sharing it too is astronomically
        # unlikely compared to sharing shape+timestamps alone.
        if n and "close" in frame.columns:
            price_fingerprint = (float(frame["close"].iloc[0]), float(frame["close"].iloc[-1]))
        else:
            price_fingerprint = None
        return (id(frame), n, str(first_ts), str(last_ts), price_fingerprint)
    except Exception:  # noqa: BLE001 -- fingerprinting must never crash a backtest
        return (id(frame), len(frame) if frame is not None else 0, None, None, None)


def get_or_compute(
    frame: pd.DataFrame,
    kind: str,
    period: int,
    column: str,
    lookback: int | None,
    compute_fn: Callable[[], pd.Series],
) -> pd.Series:
    """Returns compute_fn()'s result, from cache if this exact
    (frame, kind, period, column, lookback) combination was already
    computed once by this process. Always returns a fresh `.copy()` so a
    caller mutating the returned Series in place can never corrupt what
    other candidates read from the cache.

    Optional prefix-slice reuse (T58_INDICATOR_CACHE_SLICING=1, default OFF):
    see the comment above ``_SLICE_INDEX`` for why it is opt-in.
    """
    key = (_frame_fingerprint(frame), kind, int(period), column, lookback)
    slice_key = (kind, int(period), column, lookback)

    global _HITS, _MISSES, _CACHE_BYTES
    with _LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        with _LOCK:
            _HITS += 1
        return cached.copy()

    use_slicing = slicing_enabled()
    if use_slicing:
        sliced = _try_slice_from_parent(frame, slice_key, column)
        if sliced is not None:
            sliced_bytes = _series_nbytes(sliced)
            with _LOCK:
                _HITS += 1
                if len(_CACHE) >= _MAX_ENTRIES or (_CACHE_BYTES + sliced_bytes) > _MAX_BYTES:
                    _CACHE.clear()
                    _CACHE_BYTES = 0
                    _SLICE_INDEX.clear()
                _CACHE[key] = sliced
                _CACHE_BYTES += sliced_bytes
            return sliced.copy()

    result = compute_fn()
    result_bytes = _series_nbytes(result)

    with _LOCK:
        _MISSES += 1
        if len(_CACHE) >= _MAX_ENTRIES or (_CACHE_BYTES + result_bytes) > _MAX_BYTES:
            _CACHE.clear()
            _CACHE_BYTES = 0
            _SLICE_INDEX.clear()
        _CACHE[key] = result
        _CACHE_BYTES += result_bytes
        if use_slicing:
            try:
                _SLICE_INDEX.setdefault(slice_key, []).append((key, weakref.ref(frame)))
            except TypeError:  # object not weak-referenceable -> simply never a slice parent
                pass
    return result.copy()


def _same_values(a, b) -> bool:
    import numpy as np
    try:
        a_arr, b_arr = a.to_numpy(), b.to_numpy()
        if a_arr.shape != b_arr.shape:
            return False
        if a_arr.dtype.kind == "f" or b_arr.dtype.kind == "f":
            return bool(np.array_equal(a_arr.astype("float64"), b_arr.astype("float64"), equal_nan=True))
        return bool(np.array_equal(a_arr, b_arr))
    except Exception:  # noqa: BLE001 -- any doubt means "not identical"
        return False


def _try_slice_from_parent(frame: pd.DataFrame, slice_key: tuple, column: str) -> pd.Series | None:
    """Returns the cached Series of a still-alive parent frame sliced to ``frame``'s
    length IF ``frame`` is verifiably a PREFIX of that parent (identical timestamps,
    OHLCV and indicator source column over the first len(frame) rows); else None."""
    n = len(frame)
    if n == 0:
        return None
    with _LOCK:
        candidates = list(_SLICE_INDEX.get(slice_key, []))
    cols = [c for c in dict.fromkeys((*_SLICE_COLUMNS, column)) if c in frame.columns]
    if not cols:
        return None
    for parent_key, parent_ref in candidates:
        parent = parent_ref()
        if parent is None or len(parent) < n:
            continue
        try:
            if any(c not in parent.columns for c in cols):
                continue
            if not all(_same_values(parent[c].iloc[:n], frame[c]) for c in cols):
                continue
            with _LOCK:
                parent_series = _CACHE.get(parent_key)
            if parent_series is None or len(parent_series) < n:
                continue
            out = parent_series.iloc[:n].copy()
            out.index = frame.index          # same labels a cold computation on `frame` would carry
            return out
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return None


def clear() -> None:
    """Drops every cached series. Call when switching to a genuinely new
    dataset within the same long-lived process (e.g. multi-instrument
    search re-using a worker pool across instruments)."""
    global _CACHE_BYTES, _HITS, _MISSES
    with _LOCK:
        _CACHE.clear()
        _SLICE_INDEX.clear()
        _CACHE_BYTES = 0
        _HITS = 0
        _MISSES = 0


def stats() -> dict:
    """Hit/miss counters -- surfaced in Search Lab / Evolution Lab progress
    logs so Owen can see the cache is actually doing something, not just
    trust that it is."""
    with _LOCK:
        total = _HITS + _MISSES
        hit_rate = (_HITS / total) if total else 0.0
        return {
            "entries": len(_CACHE), "hits": _HITS, "misses": _MISSES, "hit_rate": hit_rate,
            "bytes": _CACHE_BYTES, "max_bytes": _MAX_BYTES,
        }
