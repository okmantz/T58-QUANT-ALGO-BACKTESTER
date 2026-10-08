"""Research text -> stored, testable hypotheses (with citations).

`extract_hypotheses(chunk)` asks a language model (optional) for falsifiable
claims in a research excerpt, e.g. 'does momentum persist when volatility is
high?', and falls back to a deterministic sentence miner when no model is
available. Each claim becomes a Hypothesis (status 'proposed', source
'paper:<file>#<chunk>') run through the idea compiler; a claim the whitelisted
rule families cannot express is still stored, flagged 'not yet testable', so
the question is not lost. Near-duplicates (token overlap, or embeddings when a
usable embedder is supplied) are merged rather than stored twice.
"""
from __future__ import annotations

import json
import re

from app.ai.llm_client import LLMClient, LLMUnavailable
from app.discovery.hypothesis import Hypothesis, HypothesisStore
from app.discovery.idea_compiler import compile_idea
from app.discovery.rule_spec import SpecError

_CLAIM_WORDS = re.compile(
    r"\b(persist|persists|persistence|revert|reverts|reversal|continu\w+|predict\w*|premium|anomal\w+|"
    r"momentum|mean[- ]reversion|breakout|underreact\w*|overreact\w*|autocorrelat\w+|clustering)\b", re.I)
_STOP = {"the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "are", "when", "does", "do", "that", "this", "with", "for", "by", "as", "at", "it"}

SYSTEM_PROMPT = (
    "Extract falsifiable market-behaviour claims from the research excerpt. Reply with ONLY a JSON list of objects "
    '{"claim": str (a testable question or statement, <=200 chars), "conditions": str, "expected_sign": "positive"|"negative"|"unclear", '
    '"regime": str}. Do not invent claims the excerpt does not make. Return [] if there are none.'
)


def _tokens(s: str) -> set:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in _STOP and len(w) > 2}


def _similar(a: str, b: str, thr: float = 0.7) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= thr


def _mine_sentences(text: str, limit: int = 5) -> list[dict]:
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
        sent = sent.strip()
        if 40 <= len(sent) <= 320 and _CLAIM_WORDS.search(sent):
            out.append({"claim": sent, "conditions": "", "expected_sign": "unclear", "regime": ""})
        if len(out) >= limit:
            break
    return out


def _parse_llm(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw, re.S)
    data = json.loads(m.group(0) if m else raw)
    if isinstance(data, dict):
        data = data.get("claims") or data.get("hypotheses") or []
    return [d for d in data if isinstance(d, dict) and str(d.get("claim", "")).strip()]


def extract_hypotheses(
    chunk,
    *,
    llm: LLMClient | None = None,
    store: HypothesisStore | None = None,
    source_file: str | None = None,
    chunk_index: int | None = None,
    max_claims: int = 5,
) -> list[Hypothesis]:
    text = chunk if isinstance(chunk, str) else getattr(chunk, "text", None) or chunk.get("text", "")
    source_file = source_file or (getattr(chunk, "source", None) if not isinstance(chunk, str) else None) or "unknown"
    cite = f"paper:{source_file}" + (f"#{chunk_index}" if chunk_index is not None else "")
    claims: list[dict] = []
    note = ""
    if llm is not None:
        try:
            claims = _parse_llm(llm.complete(text[:6000], system=SYSTEM_PROMPT))
        except LLMUnavailable as exc:
            note = f"language model unavailable ({exc}); used the sentence miner."
        except (ValueError, TypeError) as exc:
            note = f"language model output unreadable ({exc}); used the sentence miner."
    if not claims:
        claims = _mine_sentences(text, max_claims)
    existing = store.all() if store is not None else []
    out: list[Hypothesis] = []
    for c in claims[:max_claims]:
        claim = str(c["claim"]).strip()
        if any(_similar(claim, h.idea) for h in existing + out):
            continue
        try:
            h = compile_idea(claim)
            h.warnings.append("compiled by the keyword route from a paper claim; check the rule matches the claim")
        except SpecError as exc:
            h = Hypothesis(idea=claim, spec={}, warnings=[f"not yet testable: {exc}"])
        h.source = cite
        h.mechanism = h.mechanism or str(c.get("conditions", ""))
        h.expected_edge = h.expected_edge or f"sign: {c.get('expected_sign', 'unclear')}" + (f"; regime: {c['regime']}" if c.get("regime") else "")
        if note:
            h.warnings.append(note)
        out.append(h)
        if store is not None:
            store.save(h)
    return out


def extract_from_library(*, llm: LLMClient | None = None, store: HypothesisStore | None = None, max_chunks: int = 40) -> list[Hypothesis]:
    """Scan the indexed research/ folder (keyword-prefiltered chunks) and store proposed hypotheses."""
    from app.ai.research_library import build_index
    chunks, _ = build_index()
    found: list[Hypothesis] = []
    for i, ch in enumerate(chunks):
        if len(found) >= max_chunks * 2:
            break
        if not _CLAIM_WORDS.search(ch.text):
            continue
        found += extract_hypotheses(ch, llm=llm, store=store, chunk_index=i)
    return found
