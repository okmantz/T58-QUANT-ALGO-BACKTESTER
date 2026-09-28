"""
Equity-curve swarm: every candidate's equity curve on one chart, with a
vertical divider where the holdout (out-of-sample) segment begins.

The chart answers one question at a glance -- "does the crowd of candidates
keep working on data after the divider, or does it fan out and die?" -- which
a single leaderboard row can't show.

How it works (deliberately no changes to the search engines):
  * Candidate specs are rebuilt from the results DB / evolution leaderboard
    with the same ``build_strategy_from_spec`` the engines use.
  * Each candidate gets ONE full-history ``run_backtest``; the per-bar equity
    curve is downsampled to ``points`` samples (default 80) and normalised to
    percent return so different account sizes/instruments share one axis.
  * The divider sits at ``1 - holdout_frac`` of the bars (default 0.2, the same
    convention app.backtest.engine.run_holdout_comparison uses).
  * Capped at ``max_curves`` (default 200) and a wall-clock ``time_budget`` so
    it stays responsive on a phone: whatever finished inside the budget is
    returned, with ``computed``/``requested`` telling the UI how much of the
    swarm it's looking at.

Honest limitation, surfaced in the UI: Search Lab's Stage 1 scores candidates
on the FULL history, so for a Search Lab swarm the segment after the divider
is "the holdout the validation stages treat as unseen", not proof that no
candidate ever saw it. Speed Run / Full Pipeline validation is what confirms
holdout behaviour; the swarm is the visual summary of it.

Pure computation lives in ``downsample_equity`` / ``summarize_curve`` (unit
tested with no backtest engine); ``build_swarm`` wires them to the engine.
"""
from __future__ import annotations

import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

MAX_CURVES = 200
POINTS_PER_CURVE = 80
HOLDOUT_FRAC = 0.2
DEFAULT_TIME_BUDGET_SECONDS = 25.0


@dataclass
class SwarmCurve:
    candidate_id: str
    family: str
    values: list[float]           # cumulative return in %, one per sample point
    total_return_pct: float
    is_return_pct: float          # in-sample segment (before the divider)
    oos_return_pct: float         # holdout segment (after the divider)
    holds_up: bool                # profitable after the divider

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id, "family": self.family,
            "values": self.values, "total_return_pct": self.total_return_pct,
            "is_return_pct": self.is_return_pct, "oos_return_pct": self.oos_return_pct,
            "holds_up": self.holds_up,
        }


@dataclass
class SwarmResult:
    curves: list[SwarmCurve] = field(default_factory=list)
    points: int = POINTS_PER_CURVE
    split_index: int = 0          # first sample index that belongs to the holdout
    holdout_frac: float = HOLDOUT_FRAC
    requested: int = 0
    computed: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0
    timed_out: bool = False

    @property
    def holds_up_count(self) -> int:
        return sum(1 for c in self.curves if c.holds_up)

    def to_dict(self) -> dict[str, Any]:
        n = len(self.curves)
        return {
            "curves": [c.to_dict() for c in self.curves], "points": self.points,
            "split_index": self.split_index, "holdout_frac": self.holdout_frac,
            "requested": self.requested, "computed": self.computed, "failed": self.failed,
            "elapsed_seconds": round(self.elapsed_seconds, 1), "timed_out": self.timed_out,
            "holds_up_count": self.holds_up_count,
            "holds_up_pct": round(100.0 * self.holds_up_count / n, 1) if n else None,
        }


def downsample_equity(equity: Any, points: int = POINTS_PER_CURVE) -> np.ndarray:
    """Evenly-spaced samples of an equity array, always including the first
    and last bar. Shorter inputs are returned whole (never upsampled), so the
    result has ``min(points, len(equity))`` entries. Empty input -> empty."""
    arr = np.asarray(equity, dtype=float)
    if arr.size == 0:
        return arr
    points = max(2, int(points))
    if arr.size <= points:
        return arr
    idx = np.linspace(0, arr.size - 1, points).round().astype(int)
    return arr[idx]


def split_index_for(n_samples: int, holdout_frac: float = HOLDOUT_FRAC) -> int:
    """First sample index inside the holdout, clamped so both segments are
    non-empty whenever there are at least 2 samples."""
    if n_samples < 2:
        return 0
    idx = int(round(n_samples * (1.0 - holdout_frac)))
    return min(max(idx, 1), n_samples - 1)


def summarize_curve(
    candidate_id: str, family: str, equity: Any, initial_balance: float,
    points: int = POINTS_PER_CURVE, holdout_frac: float = HOLDOUT_FRAC,
) -> Optional[SwarmCurve]:
    """Downsample + normalise one equity array into a SwarmCurve, or None if
    it can't be drawn (empty, non-finite, or a non-positive starting balance)."""
    if not initial_balance or initial_balance <= 0:
        return None
    sampled = downsample_equity(equity, points)
    if sampled.size < 2 or not np.all(np.isfinite(sampled)):
        return None
    returns = (sampled / initial_balance - 1.0) * 100.0
    split = split_index_for(sampled.size, holdout_frac)
    is_ret = float(returns[split - 1])
    # Holdout return = growth of equity across the segment, relative to where
    # the segment STARTED (not to the initial balance), so it reads as "what
    # this candidate did on the unseen data".
    base = sampled[split - 1]
    oos_ret = float((sampled[-1] / base - 1.0) * 100.0) if base > 0 else float("nan")
    if not np.isfinite(oos_ret):
        oos_ret = 0.0
    return SwarmCurve(
        candidate_id=candidate_id, family=family or "unknown",
        values=[round(float(v), 3) for v in returns],
        total_return_pct=round(float(returns[-1]), 3),
        is_return_pct=round(is_ret, 3), oos_return_pct=round(oos_ret, 3),
        holds_up=oos_ret > 0,
    )


def build_swarm(
    df: Any, risk: Any, candidates: list[dict],
    max_curves: int = MAX_CURVES, points: int = POINTS_PER_CURVE,
    holdout_frac: float = HOLDOUT_FRAC, time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    cancel_event: Optional[threading.Event] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    clock: Callable[[], float] = time.time,
) -> SwarmResult:
    """Backtests up to ``max_curves`` candidates and returns their swarm.

    ``candidates``: dicts with ``candidate_id``, ``family`` and whatever
    app.search.batch_runner._spec_from_record needs (``source_type`` plus
    ``config`` or ``code_text``/``code_extension``) -- i.e. rows straight from
    ResultsDB, or an evolution record's ``spec`` merged with its id/family.
    A candidate that fails to build or backtest is counted in ``failed`` and
    skipped; it never aborts the swarm."""
    # Imported lazily: these pull in the whole backtest/strategy stack, and the
    # pure helpers above must stay importable (and testable) without it.
    from app.backtest.engine import run_backtest
    from app.search.batch_runner import _spec_from_record
    from app.search.strategy_space import build_strategy_from_spec

    chosen = candidates[: max(0, int(max_curves))]
    result = SwarmResult(points=points, holdout_frac=holdout_frac, requested=len(chosen))
    t0 = clock()
    with tempfile.TemporaryDirectory(prefix="t58_swarm_") as tmp_dir:
        for i, cand in enumerate(chosen):
            if cancel_event is not None and cancel_event.is_set():
                break
            if clock() - t0 > time_budget_seconds:
                result.timed_out = True
                break
            try:
                strategy = build_strategy_from_spec(_spec_from_record(cand), tmp_dir)
                bt = run_backtest(df, strategy, risk)
                curve = summarize_curve(
                    str(cand.get("candidate_id", f"cand-{i}")), str(cand.get("family") or ""),
                    bt.equity_curve["equity"].to_numpy(), bt.initial_balance, points, holdout_frac,
                )
            except Exception:  # noqa: BLE001 -- one bad candidate must never sink the swarm
                curve = None
            if curve is None:
                result.failed += 1
            else:
                result.curves.append(curve)
                result.computed += 1
            if progress_cb is not None:
                try:
                    progress_cb(i + 1, len(chosen))
                except Exception:  # noqa: BLE001
                    pass
    if result.curves:
        result.split_index = split_index_for(min(points, len(result.curves[0].values)), holdout_frac)
        # every curve is sampled from an equity array of the same length, but
        # be safe: trim to the shortest so the chart's x-axis is consistent
        n = min(len(c.values) for c in result.curves)
        for c in result.curves:
            c.values = c.values[:n]
        result.points = n
        result.split_index = split_index_for(n, holdout_frac)
    result.elapsed_seconds = clock() - t0
    return result


class SwarmCache:
    """Small in-memory cache + background runner so the web route can return
    immediately with ``status: running`` and the chart polls until ready.
    Bounded (oldest evicted) so a long-uptime install can't grow it forever."""

    def __init__(self, max_entries: int = 16) -> None:
        self._entries: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_entries

    def get(self, key: str) -> Optional[dict[str, Any]]:
        with self._lock:
            entry = self._entries.get(key)
            return dict(entry) if entry is not None else None

    def start(self, key: str, compute: Callable[[Callable[[int, int], None]], SwarmResult], force: bool = False) -> dict[str, Any]:
        """Starts ``compute(progress_cb)`` on a daemon thread unless this key
        is already running/ready (``force`` recomputes a finished one).
        Returns the entry as it stands right now."""
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None and (existing["status"] == "running" or (existing["status"] == "ready" and not force)):
                return dict(existing)
            entry = {"status": "running", "done": 0, "total": 0, "result": None, "error": None}
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

        def _progress(done: int, total: int) -> None:
            with self._lock:
                e = self._entries.get(key)
                if e is not None:
                    e["done"], e["total"] = done, total

        def _run() -> None:
            try:
                res = compute(_progress)
                with self._lock:
                    e = self._entries.get(key)
                    if e is not None:
                        e["status"], e["result"] = "ready", res.to_dict()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    e = self._entries.get(key)
                    if e is not None:
                        e["status"], e["error"] = "error", f"{type(exc).__name__}: {exc}"

        threading.Thread(target=_run, daemon=True).start()
        return dict(entry)


SWARM_CACHE = SwarmCache()
