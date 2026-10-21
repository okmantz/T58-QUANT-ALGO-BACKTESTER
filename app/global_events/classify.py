"""Headline classification: impact (0..3), category, instruments, de-duplication and source-tier weighting.

All keyword lists here are this project's own. Impact is a *keyword* heuristic -- it is shown as a tag, never as
a signal. Lower-tier sources (3-4: aggregators, commentary) are capped at impact 1 until a second publisher
corroborates the same story, which is the cheap, free stand-in for human confirmation.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from .theaters import has_any, theaters_for_text

IMPACT_3 = ("rate decision", "emergency cut", "emergency rate", "circuit breaker", "trading halt", "default on", "sovereign default",
            "invasion", "strait closure", "closes strait", "declares war", "bank run", "bank failure", "nuclear")
IMPACT_2 = ("fomc", "cpi", "consumer price", "nonfarm", "non-farm", "payrolls", "tariff", "opec", "sanction", "rate hike", "rate cut",
            "powell", "ecb", "bank of japan", "boj", "bank of england", "gdp", "pce", "ceasefire", "airstrike", "missile", "blockade",
            "shutdown", "downgrade", "recession")
IMPACT_1 = ("fed ", "federal reserve", "earnings", "guidance", "pmi", "jobless claims", "retail sales", "oil", "crude", "treasury",
            "yields", "inflation", "jobs report", "unemployment", "ppi", "ism", "housing starts", "bitcoin", "etf")
PAST_MARKERS = ("anniversary", "history of", "years ago", "remembering", "in the 1970s", "in the 1980s", "in the 1990s", "retrospective", "looking back")

CATEGORY_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fed", ("fomc", "federal reserve", "powell", "fed ", "ecb", "boj", "bank of england", "rate cut", "rate hike", "rate decision")),
    ("inflation", ("cpi", "inflation", "ppi", "pce", "consumer price")),
    ("jobs", ("nonfarm", "non-farm", "payrolls", "jobless", "unemployment", "jobs report", "labor market")),
    ("energy", ("oil", "crude", "opec", "natural gas", "lng", "refinery", "pipeline", "brent", "hormuz")),
    ("tariff", ("tariff", "trade war", "sanction", "export ban", "export controls", "section 301")),
    ("earnings", ("earnings", "guidance", "revenue", "quarterly results")),
    ("crypto", ("bitcoin", "crypto", "ethereum", "stablecoin", "etf approval")),
    ("geo", ("war", "strike", "missile", "invasion", "ceasefire", "blockade", "troops", "military", "attack")),
)
CATEGORY_INSTRUMENTS = {
    "fed": ("ES", "NQ", "YM", "ZN", "DXY", "GC"),
    "inflation": ("ES", "NQ", "ZN", "DXY", "GC"),
    "jobs": ("ES", "NQ", "YM", "ZN", "DXY"),
    "energy": ("CL", "BZ", "NG"),
    "tariff": ("ES", "NQ", "DXY", "USD/CNH"),
    "earnings": ("ES", "NQ"),
    "crypto": ("BTC", "ETH"),
    "geo": ("CL", "GC", "ES"),
    "other": (),
}

_STOP = frozenset("a an the and or of to in on for at by with from as is are was were be been it its this that these those after before over under says say said new us u.s".split())
_NON_WORD = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    t = _NON_WORD.sub(" ", (title or "").lower())
    return _WS.sub(" ", t).strip()


def tokens(title: str) -> frozenset[str]:
    return frozenset(w for w in normalize_title(title).split() if w not in _STOP and len(w) > 1)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def classify(title: str) -> dict:
    """Return {'impact_raw', 'category', 'instruments', 'theaters'} for one headline (no source-tier logic here)."""
    low = normalize_title(title)
    impact = 0
    for level, words in ((3, IMPACT_3), (2, IMPACT_2), (1, IMPACT_1)):
        if has_any(low, words):
            impact = level
            break
    if impact and has_any(low, PAST_MARKERS):
        impact -= 1
    category = "other"
    for cat, words in CATEGORY_WORDS:
        if has_any(low, words):
            category = cat
            break
    theaters = theaters_for_text(title)
    if impact == 0 and theaters and not has_any(low, PAST_MARKERS):
        impact = 1   # a keyword hit inside a monitored theater is worth a low tag even without a macro keyword
    instruments = list(CATEGORY_INSTRUMENTS.get(category, ()))
    if "taiwan" in theaters:
        instruments += [i for i in ("NQ", "SOX") if i not in instruments]
    if "mideast" in theaters or "redsea" in theaters:
        instruments += [i for i in ("CL", "BZ") if i not in instruments]
    return {"impact_raw": impact, "category": category, "instruments": instruments[:8], "theaters": theaters}


def effective_impact(impact_raw: int, tier: int | None, corroborated: bool) -> int:
    """Tier 1-2 keep their impact; tier 3-4 (or undeclared) cap at 1 unless a second publisher carries the story."""
    if tier is not None and tier <= 2:
        return impact_raw
    return impact_raw if corroborated else min(impact_raw, 1)


def cluster_into(existing: list[tuple[int, frozenset[str], str, datetime]], new_tokens: frozenset[str], publisher: str,
                 when: datetime, window_h: float = 6.0, threshold: float = 0.6) -> tuple[int | None, bool]:
    """Find an existing cluster for a new headline. Returns (cluster_id or None, corroborated_by_other_publisher).

    ``existing`` rows are (cluster_id, tokens, publisher, published_at)."""
    lo = when - timedelta(hours=window_h)
    hi = when + timedelta(hours=window_h)
    for cid, tk, pub, w in existing:
        if lo <= w <= hi and jaccard(new_tokens, tk) >= threshold:
            return cid, (pub != publisher)
    return None, False


def parse_when(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
