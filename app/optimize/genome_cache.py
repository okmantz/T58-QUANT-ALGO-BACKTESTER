"""Backtest memoization keys (2026-10-17).

Two genomes that differ only below an integer gene's rounding (``12.2`` vs ``11.8`` bars) build the SAME strategy, so they
must share one backtest. ``genome_key`` is that canonical, hashable key; ``MemoTable`` is a thread-safe memo with hit/miss
counters used by the GA and by the per-fold walk-forward optimizer.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Sequence


def genome_key(genes: Sequence[Any] | None, genome: Sequence[float]) -> tuple:
    """Canonical key: integer genes are rounded the way ``apply_genome`` / the code materializer will round them."""
    if not genes or len(genes) != len(genome):
        return tuple(float(x) for x in genome)
    return tuple(int(round(float(v))) if getattr(g, "is_int", False) else float(v) for g, v in zip(genes, genome))


class MemoTable:
    """key -> value memo, safe to share across threads. ``get_or_compute`` may compute twice under a race but never
    stores two different values for a key (first write wins)."""

    def __init__(self) -> None:
        self._d: dict = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: Any, fn: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._d:
                self.hits += 1
                return self._d[key]
        value = fn()
        with self._lock:
            self.misses += 1
            return self._d.setdefault(key, value)

    def __len__(self) -> int:
        return len(self._d)
