"""Plain-English explanations for READY / MARGINAL / NOT READY verdicts.

Feeds the small "why?" popup that opens when a verdict pill is clicked on the
dashboard and the Champion page. The pipeline stores its raw verdict reasons
(long, technical sentences) in ``last_champion_check["reasons"]``; this module
turns them into a short headline plus a handful of everyday-language bullets.

Pure functions, no I/O, so it is trivially testable and safe to call while
rendering a page. Anything unrecognised falls back to the first sentence of
the raw reason, so a new gate can never make the popup blank or wrong.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

# (prefix found in the raw reason, plain-language sentence). First match wins.
_PLAIN: tuple[tuple[str, str], ...] = (
    ("HARD SAFETY GATE FAILED: the lookahead",
     "The strategy appears to peek at prices that had not happened yet, so its backtest can't be trusted."),
    ("HARD SAFETY GATE FAILED: Monte Carlo risk of ruin",
     "Too many simulated accounts blew up. The chance of breaking a prop-firm rule is above the allowed limit."),
    ("WALK-FORWARD", "It did not hold up on data it was not tuned on (walk-forward test)."),
    ("HOLDOUT", "It lost money on the final slice of data that was kept hidden during tuning."),
    ("CPCV", "It failed the cross-validation test, which usually means the results were partly luck."),
    ("DSR", "After allowing for how many variations were tried, the edge is not statistically convincing."),
    ("PBO", "There is a high chance the winning settings were just the luckiest of many tries (overfitting)."),
    ("RUIN WINDOW NOT COVERED",
     "The price history is too short to measure the risk of blowing up over a full evaluation window."),
    ("PASS-AND-PAID TOO LOW",
     "Not enough simulated accounts both passed the evaluation and reached a first payout."),
    ("SAMPLE FLOOR", "There are too few trades or trading days to trust the result yet."),
    ("REPLAY DISAGREES",
     "Re-running the strategy from real start dates gave very different pass odds than the simulation did."),
    ("RANDOM-ENTRY NULL NOT BEATEN",
     "Random entries with the same exits did about as well, so the entry signal may be no better than chance."),
    ("COST STRESS",
     "When trading costs (spread, slippage, commission) are doubled, too little of the profit survives."),
    ("30-DAY PAYOUT", "Too few fresh accounts reached a first payout within 30 days."),
    ("SEED INSTABILITY",
     "The headline pass rate changes a lot between simulation runs, so it is partly luck of the draw."),
    ("CAPPED AT MARGINAL: the T58 Score alone would have called this READY, but the final",
     "Scores like a winner, but the backtest has too few trades to call it READY."),
    ("CAPPED AT MARGINAL: the T58 Score alone would have called this READY, but core",
     "Scores like a winner, but some key safety checks were not measured, so it can't be READY yet."),
    ("CAPPED AT MARGINAL: the T58 Score alone would have called this READY, but only",
     "Scores like a winner, but too many checks were not measured, so it can't be READY yet."),
    ("CAPPED AT MARGINAL: neither the walk-forward",
     "No out-of-sample check produced a result, so a READY call would rest on in-sample results only."),
    ("CAPPED AT MARGINAL: a strategy is not called READY while its holdout",
     "The holdout check had too few trades to count, so it can't be READY yet."),
    ("EV ", None),  # handled specially below (informational)
)

# Reasons that are context, not causes.
_SKIP_PREFIXES = (
    "For reference,", "Evidence (", "This verdict does NOT lock", "Supporting diagnostic",
    "  ", "T58 Score", "ADVISORY",
)

_SUMMARY = {
    "READY": "Every T58 safety gate passed and the score is 75 or more (85+ is labelled Elite). This is the best verdict, not a guarantee.",
    "MARGINAL": "Promising, but something held it back from READY. Fix the points below, then re-run the check.",
    "NOT READY": "It failed a hard safety gate or scored too weakly to trust with real money.",
}


def _first_sentence(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    m = re.search(r"(?<=[.!?])\s", text)
    out = text[: m.start()] if m else text
    return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"


def _plain_for(reason: str) -> Optional[str]:
    for prefix, plain in _PLAIN:
        if reason.startswith(prefix):
            return plain
    return None


def explain_verdict(
    verdict: Optional[str],
    reasons: Optional[Iterable[str]] = None,
    *,
    score: Optional[float] = None,
    tier: Optional[str] = None,
    fallback: Optional[str] = None,
) -> dict[str, Any]:
    """Return ``{"verdict", "title", "summary", "points", "score"}``.

    ``points`` is a list of short plain-English bullets (max 5). When no stored
    reasons exist (older runs), ``fallback`` (e.g. the board's one-line
    diagnosis) is used so the popup still says something specific.
    """
    v = (verdict or "").upper().strip() or "UNTESTED"
    points: list[str] = []
    seen: set[str] = set()
    for raw in reasons or []:
        raw = str(raw)
        if any(raw.startswith(p) for p in _SKIP_PREFIXES):
            continue
        plain = _plain_for(raw)
        if plain is None and raw.startswith("EV "):
            continue
        text = plain or _first_sentence(raw)
        if text and text not in seen:
            seen.add(text)
            points.append(text)
        if len(points) >= 5:
            break
    if not points and fallback:
        points.append(fallback)
    if v == "READY" and not points:
        points.append("No open issues were found in the last check.")
    if v == "UNTESTED":
        summary = "This strategy has not been through a Champion Check yet."
    else:
        summary = _SUMMARY.get(v, "")
    if score is not None:
        tier_txt = f" ({tier})" if tier else ""
        summary = f"{summary} T58 Score: {score:.0f}/100{tier_txt}."
    title = {"READY": "Why it is READY", "MARGINAL": "Why it is MARGINAL",
             "NOT READY": "Why it is NOT READY"}.get(v, "No verdict yet")
    return {"verdict": v, "title": title, "summary": summary, "points": points,
            "score": None if score is None else round(float(score), 1)}
