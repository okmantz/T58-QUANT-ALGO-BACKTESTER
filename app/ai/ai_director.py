"""T58 AI Director -- the portfolio-level orchestration layer above
AI Assistant (one-shot market chat), AI Research (agentic single-strategy
investigation), and AI Strategy Creator (drafts one new strategy from an
idea).

None of those three ever look across the whole Strategy Library at once:
the Assistant answers whatever's asked, the Research Agent investigates
the ONE strategy bound to it, the Research Loop iterates on ONE
hypothesis, and Strategy Generator drafts ONE new file. Nobody was
surveying every strategy already sitting in the library and asking "of
everything Owen has going right now, what's the highest-value thing to
do today?" -- that gap is this module's whole job.

Same discipline as everywhere else in app.ai ("Ollama interprets, the
application calculates"): the priority ranking itself is 100%
deterministic, computed straight from each strategy's own metadata via
app.strategy.library.compute_pipeline_progress, its own status and
last-modified timestamp, and (optionally) today's market rankings from
app.ai.market_scanner and the aggregate research-memory counts from
app.ai.experiment_memory. Ollama is only ever handed that already-
computed, already-sorted list (see build_director_prompt /
app.ai.trading_assistant.DIRECTOR_SYSTEM_PROMPT) and asked to write it up
as a short briefing in Owen's voice -- it never re-ranks anything, never
invents a strategy that isn't in the list, and is never the thing that
decides what's actually good or bad (that's still Champion Check /
Monte Carlo / the prop simulator, same as every other AI touchpoint in
this app).

Zero I/O in this module by design: compute_directives() takes plain
already-loaded strategy records (app.strategy.library.StoredStrategy
objects, or any dict/object exposing the same four fields) so it stays a
pure, fast, fully unit-testable function. The caller (a Flask route or
the desktop AI Assistant tab) is responsible for actually listing the
library and fetching today's rankings.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_TOP_N = 12

# A strategy sitting untouched this long picks up a small "don't let this
# go stale forever" priority bump -- separate knobs for the softer
# NOT READY/draft cases (where a little staleness is normal -- Owen has
# 40+ strategies and can't touch all of them daily) vs. the currently
# unused STALE_DAYS_URGENT reserved for a future "flag as abandoned"
# feature.
STALE_DAYS_URGENT = 14.0

# Bonus added when a strategy's own `market` metadata matches a symbol
# that today's Best Markets scan rated at/above MARKET_ALIGN_SCORE_FLOOR.
MARKET_ALIGN_BONUS = 15.0
MARKET_ALIGN_SCORE_FLOOR = 70

# Statuses that mean "already acted on" -- a READY strategy already in one
# of these should never come back as a PROMOTE directive again.
_LIVE_STATUSES = {"live", "deployed", "forward_testing", "forward-testing"}

# Human-readable labels for the deterministic text builder -- the Ollama
# prompt gets the raw action codes (see DIRECTOR_SYSTEM_PROMPT, which
# already spells out how to phrase each one); this dict is only for the
# always-available fallback text.
_ACTION_LABELS = {
    "PROMOTE": "ready to review for deployment",
    "RUN_OPTIMIZE": "send through Quick Optimize / Search Lab next",
    "VALIDATE_FURTHER": "push through Validation Lab",
    "TEST_OR_ARCHIVE": "decide: test it for real, or archive",
    "ITERATE_OR_ARCHIVE": "decide: rework, or archive",
    "MARKET_ALIGNED": "today's market conditions favor this one",
}


@dataclass
class Directive:
    """One deterministically-scored recommendation for one strategy."""

    strategy_name: str
    strategy_type: str
    action: str
    priority: float
    reason: str
    stage: str
    verdict: str | None
    status: str
    market: str | None
    days_idle: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "action": self.action,
            "priority": round(self.priority, 1),
            "reason": self.reason,
            "stage": self.stage,
            "verdict": self.verdict,
            "status": self.status,
            "market": self.market,
            "days_idle": round(self.days_idle, 1),
        }


def _days_idle(modified_ts: float | None, now: float) -> float:
    if not modified_ts:
        return 0.0
    return max(0.0, (now - modified_ts) / 86400.0)


def _hot_markets(rankings: list[dict] | None) -> dict[str, dict]:
    """symbol -> ranking dict, restricted to markets scoring at/above
    MARKET_ALIGN_SCORE_FLOOR today. None/empty rankings -> {} (no bonus
    applied anywhere) -- never raises on a malformed entry, that entry is
    just skipped."""
    out: dict[str, dict] = {}
    for r in rankings or []:
        try:
            score = int(r.get("score", 0))
            symbol = str(r.get("symbol", "")).strip().upper()
        except (TypeError, ValueError, AttributeError):
            continue
        if symbol and score >= MARKET_ALIGN_SCORE_FLOOR:
            out[symbol] = r
    return out


def score_one(
    *,
    name: str,
    strategy_type: str,
    status: str,
    metadata: dict[str, Any],
    modified_ts: float,
    now: float,
    hot_markets: dict[str, dict],
    pipeline_progress_fn: Callable[[dict], dict],
) -> Directive:
    """Pure scoring for one strategy. `pipeline_progress_fn` is injected
    (normally app.strategy.library.compute_pipeline_progress) purely so
    this stays testable with a fake and this module never has to import
    the filesystem-touching library module at call time."""
    metadata = metadata or {}
    progress = pipeline_progress_fn(metadata)
    stage = progress.get("current_stage", "create")
    verdict = progress.get("verdict")
    next_stage = progress.get("next_stage")
    idle = _days_idle(modified_ts, now)
    market = str(metadata.get("market") or "").strip().upper() or None
    status_norm = (status or "draft").strip().lower()

    market_hit = hot_markets.get(market) if market else None
    market_note = ""
    if market_hit:
        market_note = (
            f" {market} is today's top-ranked market (T58 score "
            f"{market_hit.get('score')}, {market_hit.get('status', '')})."
        )

    if verdict == "READY" and status_norm not in _LIVE_STATUSES:
        base = 90.0
        reason = f"Champion-checked READY and not yet live/forward-testing.{market_note}"
        action = "PROMOTE"
    elif verdict == "MARGINAL":
        base = 60.0 - min(idle, 10.0)
        reason = f"MARGINAL verdict -- one more optimize/validate pass could push it to READY.{market_note}"
        action = "RUN_OPTIMIZE"
    elif next_stage == "optimize" and stage == "test":
        base = 50.0
        reason = f"Backtested but never run through Quick Optimize/Search Lab.{market_note}"
        action = "RUN_OPTIMIZE"
    elif stage == "optimize" and next_stage in ("validate", "champion_check"):
        base = 45.0
        reason = f"Optimized but not yet through Validation Lab/Champion Check.{market_note}"
        action = "VALIDATE_FURTHER"
    elif verdict == "NOT READY":
        base = 25.0 + min(idle / 2.0, 15.0)
        reason = f"NOT READY verdict, idle {idle:.0f}d -- decide whether to rework or archive.{market_note}"
        action = "ITERATE_OR_ARCHIVE"
    elif stage == "create":
        base = 10.0 + min(idle, 15.0)
        reason = f"Draft never backtested, idle {idle:.0f}d.{market_note}"
        action = "TEST_OR_ARCHIVE"
    else:
        base = 0.0
        reason = f"No pending pipeline action detected.{market_note}"
        action = "NONE"

    if market_hit and action == "NONE":
        base += MARKET_ALIGN_BONUS
        action = "MARKET_ALIGNED"
        reason = f"No pending pipeline action, but{market_note.strip()}"
    elif market_hit and action != "PROMOTE":
        base += MARKET_ALIGN_BONUS

    return Directive(
        strategy_name=name, strategy_type=strategy_type, action=action,
        priority=base, reason=reason.strip(), stage=stage, verdict=verdict,
        status=status_norm, market=market, days_idle=idle,
    )


def compute_directives(
    strategies: list,
    rankings: list[dict] | None = None,
    now: float | None = None,
    top_n: int = DEFAULT_TOP_N,
    pipeline_progress_fn: Callable[[dict], dict] | None = None,
) -> list[Directive]:
    """`strategies` is a list of objects/dicts each exposing name/
    strategy_type/status/metadata/modified -- app.strategy.library's
    StoredStrategy already has exactly this shape, so callers normally
    just pass app.strategy.library.list_saved_strategies()'s result
    straight through. `rankings` is app.ai.market_scanner's list of
    ranking dicts (see ranking_to_dict), optional -- pass None/[] to skip
    market-alignment scoring entirely.

    Never raises: any single malformed entry is skipped rather than
    aborting the whole scan, and an empty/None `strategies` list simply
    returns []."""
    if pipeline_progress_fn is None:
        from app.strategy.library import compute_pipeline_progress as pipeline_progress_fn
    now = now if now is not None else time.time()
    hot = _hot_markets(rankings)
    out: list[Directive] = []
    for s in strategies or []:
        try:
            if isinstance(s, dict):
                name = s.get("name")
                strategy_type = s.get("strategy_type", "")
                status = s.get("status", "draft")
                metadata = s.get("metadata", {})
                modified_ts = s.get("modified", 0.0)
            else:
                name = getattr(s, "name")
                strategy_type = getattr(s, "strategy_type", "")
                status = getattr(s, "status", "draft")
                metadata = getattr(s, "metadata", {})
                modified_ts = getattr(s, "modified", 0.0)
            if not name:
                continue
            d = score_one(
                name=name, strategy_type=strategy_type, status=status, metadata=metadata,
                modified_ts=modified_ts, now=now, hot_markets=hot,
                pipeline_progress_fn=pipeline_progress_fn,
            )
            if d.action != "NONE":
                out.append(d)
        except Exception:
            continue
    out.sort(key=lambda d: d.priority, reverse=True)
    return out[:top_n]


def build_deterministic_briefing(directives: list[Directive], memory_counts: dict | None = None) -> str:
    """Always-available text, no Ollama required -- same
    "deterministic-first, AI narrative appended on top" convention as
    app.ai.trading_assistant.build_deterministic_outlook, and what the
    AI Director panel shows immediately (before/without Ollama)."""
    lines = ["T58 AI DIRECTOR -- Deterministic Priority List", ""]
    if memory_counts and memory_counts.get("total"):
        by_verdict = memory_counts.get("by_verdict") or {}
        breakdown = ", ".join(f"{v}: {n}" for v, n in sorted(by_verdict.items(), key=lambda kv: -kv[1]))
        lines.append(f"Research memory: {memory_counts['total']} experiments recorded overall ({breakdown}).")
        lines.append("")
    if not directives:
        lines.append(
            "Nothing in the Strategy Library needs action right now -- either the library is empty, "
            "or everything in it is already Ready/live with no stale drafts or unfinished pipelines. "
            "Consider starting a new idea via Strategy Generator, Search Lab, or the Research Loop."
        )
        return "\n".join(lines)
    for i, d in enumerate(directives, start=1):
        label = _ACTION_LABELS.get(d.action, d.action)
        lines.append(f"{i}. {d.strategy_name} ({d.strategy_type}) -- {label} [priority {d.priority:.0f}]")
        lines.append(f"   {d.reason}")
    return "\n".join(lines)


def build_director_prompt(directives: list[Directive], memory_counts: dict | None = None) -> str:
    """User-message text handed to
    app.ai.trading_assistant.TradingAssistantClient.director_briefing()
    alongside DIRECTOR_SYSTEM_PROMPT -- the deterministic list plus the
    research-memory aggregate, nothing else. DIRECTOR_SYSTEM_PROMPT
    instructs the model never to add, remove, or reorder items."""
    payload = {
        "priority_list": [d.to_dict() for d in directives],
        "research_memory_summary": memory_counts or {},
    }
    return (
        "Deterministically-computed priority list (all facts below were computed by the app, not "
        "by you -- do not add, remove, or reorder items; only explain and summarize):\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Write today's T58 AI Director briefing now in the exact format specified."
    )
