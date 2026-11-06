"""Convergence-driven search budget (2026-11-05 audit).

A fixed candidate count either stops while the search is still finding new things or keeps spending after it has stopped. This module measures
the one thing that says which: how many NEW, DISTINCT signal sets each batch of evaluations produced, per 1,000 evaluations. When a batch adds
fewer than ``min_new_per_1000`` the grid is exhausted for practical purposes and the search stops; otherwise it draws another batch (up to
``max_rounds``). Two candidates are "the same signal set" when they trade the same number of times with the same net result -- a parameter
nudge that changes nothing is not a new idea.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MIN_NEW_PER_1000 = 5.0
DEFAULT_MAX_ROUNDS = 4
_FINGERPRINT_COLUMNS = ("total_trades", "trades", "net_profit", "net_pnl")


def signal_fingerprint(row: dict) -> tuple | None:
    """Hashable identity of what a candidate DID (trade count + net result), or None if the row carries neither."""
    stats = row.get("statistics") if isinstance(row.get("statistics"), dict) else {}
    vals = []
    for col in _FINGERPRINT_COLUMNS:
        v = row.get(col, stats.get(col))
        if isinstance(v, (int, float)) and v == v:
            vals.append((col, round(float(v), 2)))
    if len(vals) >= 2:
        return tuple(vals)
    q = row.get("quick_score")
    if isinstance(q, (int, float)) and q == q:           # screen-only rows: the quick score at 4 dp stands in for the trade sequence
        return (("quick_score", round(float(q), 4)),) + tuple(vals)
    return None


@dataclass
class ConvergenceTracker:
    min_new_per_1000: float = DEFAULT_MIN_NEW_PER_1000
    max_rounds: int = DEFAULT_MAX_ROUNDS
    seen: set = field(default_factory=set)
    history: list = field(default_factory=list)   # [{"round", "evaluations", "new_distinct", "per_1000"}]

    def add_round(self, rows, evaluations: int) -> dict:
        """Record a finished batch. ``rows`` are its candidate rows (dicts); returns the round's summary."""
        new = 0
        for r in rows or []:
            fp = signal_fingerprint(r)
            if fp is not None and fp not in self.seen:
                self.seen.add(fp)
                new += 1
        ev = max(int(evaluations or 0), 1)
        item = {"round": len(self.history) + 1, "evaluations": int(evaluations or 0), "new_distinct": new, "per_1000": new * 1000.0 / ev}
        self.history.append(item)
        return item

    def should_continue(self) -> bool:
        if not self.history or len(self.history) >= self.max_rounds:
            return False
        return self.history[-1]["per_1000"] >= self.min_new_per_1000

    def note(self) -> str:
        if not self.history:
            return ""
        last = self.history[-1]
        total_ev = sum(h["evaluations"] for h in self.history)
        base = (f"Convergence: {len(self.history)} round(s), {total_ev:,} evaluations, {len(self.seen):,} distinct signal sets; "
                f"the last round added {last['new_distinct']} new ({last['per_1000']:.1f} per 1,000 evaluations)")
        if len(self.history) >= self.max_rounds and last["per_1000"] >= self.min_new_per_1000:
            return base + f" -- stopped at the {self.max_rounds}-round cap while still finding new ones; run again to keep going."
        return base + f" -- stopped: below {self.min_new_per_1000:g} per 1,000, so more of the same grid is unlikely to add ideas."
