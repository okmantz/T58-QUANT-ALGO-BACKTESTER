"""
IDEA COMPILER -- free text -> validated Hypothesis.

Two routes, same output contract:
  1. a language model (app.ai.llm_client.LLMClient) asked for ONE JSON object
     matching the whitelisted schema; its text is parsed and run through
     rule_spec.validate_spec -- a hallucinated kind/param is rejected, never
     coerced;
  2. a deterministic keyword compiler (always available, no model needed).

If the model is missing, errors, or returns something invalid, the keyword
route is used and the reason is recorded on the hypothesis (`warnings`), so the
trader always knows which route produced the rules they are about to test.
The compiler never invents performance claims: `expected_edge` states the
direction of the claim only, and every hypothesis carries explicit falsifiers.
"""
from __future__ import annotations

import json
import re

from app.ai.llm_client import LLMClient, LLMUnavailable
from app.discovery.hypothesis import Hypothesis
from app.discovery.rule_spec import SPEC_SCHEMA, SpecError, validate_spec

_KEYWORDS = [
    ("fvg_retest", ["fair value gap", "fvg", "imbalance", "gap fill", "retest of the gap", "limit order at the gap"]),
    ("donchian_breakout", ["breakout", "break out", "new high", "new low", "donchian", "range break", "channel break"]),
    ("rsi_reversion", ["rsi", "oversold", "overbought"]),
    ("zscore_reversion", ["mean reversion", "mean-reversion", "revert", "z-score", "zscore", "stretched", "fade the move", "snap back"]),
    ("ma_cross", ["moving average", "ma cross", "crossover", "golden cross", "ema cross", "sma cross"]),
    ("momentum", ["momentum", "trend following", "trend-following", "keeps going", "continuation"]),
]
_MECHANISM = {
    "fvg_retest": "Fast one-sided moves leave unfilled imbalances; resting liquidity gets re-tested before the move continues.",
    "donchian_breakout": "Range expansions attract trend-following flow and stop orders that extend the move.",
    "rsi_reversion": "Short-horizon overshoots are partly liquidity-driven and partially revert.",
    "zscore_reversion": "Prices stretched far from their recent mean tend to be pulled back by inventory and value buyers/sellers.",
    "ma_cross": "A persistent shift in the average price indicates a change in the dominant flow.",
    "momentum": "Returns are positively autocorrelated at medium horizons (under-reaction, herding).",
}
_FALSIFIERS = [
    "Net expectancy after costs is not above the random-entry distribution (p >= 0.05)",
    "Edge disappears (or flips sign) on the held-out final 40% of history",
    "Edge only exists in one market / one timeframe and fails the others",
    "Edge vanishes at 1.5x costs",
    "Fewer than 2 of 3 volatility regimes are profitable",
]

SYSTEM_PROMPT = (
    "You convert a trader's idea into ONE JSON object and nothing else. Schema: "
    '{"kind": one of %s, "params": {...numbers only...}, "direction": "both"|"long"|"short", '
    '"mechanism": "why it should work, one sentence"}. Allowed params per kind: %s. '
    "Never output code. Never invent parameters."
) % (sorted(SPEC_SCHEMA), {k: sorted(v) for k, v in SPEC_SCHEMA.items()})


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise SpecError("model reply contained no JSON object")
    try:
        return json.loads(m.group(0))
    except ValueError as exc:
        raise SpecError(f"model reply was not valid JSON: {exc}") from exc


def _keyword_kind(text: str) -> str | None:
    t = text.lower()
    for kind, words in _KEYWORDS:
        if any(w in t for w in words):
            return kind
    return None


def _numbers_hint(text: str, kind: str) -> dict:
    """Pulls an explicit lookback/period such as '40-bar' / '20 period' from the text."""
    m = re.search(r"(\d{1,3})\s*[- ]?\s*(?:bar|period|day|candle)", text.lower())
    if not m:
        return {}
    v = int(m.group(1))
    key = {"donchian_breakout": "lookback", "zscore_reversion": "lookback", "momentum": "lookback",
           "rsi_reversion": "period"}.get(kind)
    return {key: v} if key else {}


def compile_idea(
    idea: str,
    *,
    llm: LLMClient | None = None,
    markets: list | None = None,
    timeframes: list | None = None,
) -> Hypothesis:
    idea = (idea or "").strip()
    if not idea:
        raise SpecError("empty idea")
    warnings: list[str] = []
    spec = None
    mechanism = ""
    source = "keyword"
    if llm is not None:
        try:
            raw = llm.complete(f"Idea: {idea}", system=SYSTEM_PROMPT)
            obj = _extract_json(raw)
            mechanism = str(obj.pop("mechanism", "") or "")
            spec = validate_spec(obj)
            source = "llm"
        except LLMUnavailable as exc:
            warnings.append(f"language model unavailable ({exc}); used the keyword compiler.")
        except SpecError as exc:
            warnings.append(f"language model output rejected ({exc}); used the keyword compiler.")
    if spec is None:
        kind = _keyword_kind(idea)
        if kind is None:
            raise SpecError(
                "could not map the idea to a supported rule family "
                f"{sorted(SPEC_SCHEMA)}; rephrase it (e.g. 'breakout of the 40-bar high') or enter the rule spec directly."
            )
        direction = "long" if re.search(r"\b(long only|buy only|longs only)\b", idea.lower()) else (
            "short" if re.search(r"\b(short only|sell only|shorts only)\b", idea.lower()) else "both")
        spec = validate_spec({"kind": kind, "params": _numbers_hint(idea, kind), "direction": direction})
        mechanism = _MECHANISM[kind]
    return Hypothesis(
        idea=idea, spec=spec, mechanism=mechanism or _MECHANISM[spec["kind"]],
        expected_edge="Positive net expectancy after realistic costs, distinguishable from random entries.",
        markets=list(markets or []), timeframes=list(timeframes or []),
        falsifiers=list(_FALSIFIERS), warnings=warnings, source=source,
    )
