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

import threading
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
        return (id(frame), n, str(first_ts), str(last_ts))
    except Exception:  # noqa: BLE001 -- fingerprinting must never crash a backtest
        return (id(frame), len(frame) if frame is not None else 0, None, None)


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
    other candidates read from the cache."""
    key = (_frame_fingerprint(frame), kind, int(period), column, lookback)

    global _HITS, _MISSES
    with _LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        with _LOCK:
            _HITS += 1
        return cached.copy()

    result = compute_fn()
    result_bytes = _series_nbytes(result)

    with _LOCK:
        global _CACHE_BYTES
        _MISSES += 1
        if len(_CACHE) >= _MAX_ENTRIES or (_CACHE_BYTES + result_bytes) > _MAX_BYTES:
            _CACHE.clear()
            _CACHE_BYTES = 0
        _CACHE[key] = result
        _CACHE_BYTES += result_bytes
    return result.copy()


def clear() -> None:
    """Drops every cached series. Call when switching to a genuinely new
    dataset within the same long-lived process (e.g. multi-instrument
    search re-using a worker pool across instruments)."""
    global _HITS, _MISSES, _CACHE_BYTES
    with _LOCK:
        _CACHE.clear()
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
