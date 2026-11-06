"""The ONE set of verdict names users see (2026-11-06).

    REJECT -> LEAD -> FORWARD-TEST CANDIDATE -> EVALUATE -> FUNDED

READY / MARGINAL / NOT READY are retired as user-facing words. The pipeline still computes them internally (they drive the gate
vector, the caps and champion routing, and many callers compare against them), but nothing a person reads should show them: every page,
report, console line and card goes through ``headline`` / ``relabel_text`` here. A backtest can never produce EVALUATE or FUNDED; those come
only from recorded forward-test facts (see verdict_ladder).
"""
from __future__ import annotations

import re

from app.orchestration.verdict_ladder import HARD_REASON_MARKERS, STAGES

LEGACY = ("READY", "MARGINAL", "NOT READY")
_TONE = {"REJECT": "bad", "LEAD": "warn", "FORWARD-TEST CANDIDATE": "good", "EVALUATE": "good", "FUNDED": "good"}


def headline(verdict, stored_stage=None, reasons=None) -> str:
    """The stage to show for a run. A stored ladder stage always wins; otherwise the internal verdict maps conservatively
    (READY never implies more than FORWARD-TEST CANDIDATE; NOT READY with a hard-failure reason is REJECT, otherwise LEAD)."""
    if stored_stage and str(stored_stage).upper() in STAGES:
        return str(stored_stage).upper()
    v = str(verdict or "").upper().strip()
    if v in STAGES:
        return v
    if v == "READY":
        return "FORWARD-TEST CANDIDATE"
    if v == "MARGINAL":
        return "LEAD"
    if v == "NOT READY":
        if any(m in str(r) for r in (reasons or []) for m in HARD_REASON_MARKERS):
            return "REJECT"
        return "LEAD"
    return v or "UNTESTED"


def tone(stage) -> str:
    """good / warn / bad / grey for a stage (or for a legacy verdict, mapped first)."""
    s = headline(stage) if str(stage or "").upper() in LEGACY else str(stage or "").upper()
    return _TONE.get(s, "grey")


# Order matters: longest / most specific phrases first. Only the legacy WORDS change; markers other code searches for
# ("CAPPED", "NOT PROVEN", "HARD SAFETY GATE FAILED", "EV TEST FAILED") are left alone.
_SUBS = (
    (re.compile(r"CAPPED AT MARGINAL"), "CAPPED AT LEAD"),
    (re.compile(r"capped at MARGINAL", re.I), "capped at LEAD"),
    (re.compile(r"caps at MARGINAL", re.I), "caps it at LEAD"),
    (re.compile(r"READY/MARGINAL/NOT READY"), "REJECT / LEAD / FORWARD-TEST CANDIDATE"),
    (re.compile(r"READY\s*/\s*MARGINAL\s*/\s*NOT READY"), "REJECT / LEAD / FORWARD-TEST CANDIDATE"),
    (re.compile(r"READY or MARGINAL"), "FORWARD-TEST CANDIDATE or LEAD"),
    (re.compile(r"READY/MARGINAL"), "FORWARD-TEST CANDIDATE or LEAD"),
    (re.compile(r"\b(is|are|was) NOT READY\b"), r"\1 a REJECT"),
    (re.compile(r"\bNOT READY\b"), "REJECT"),
    (re.compile(r"\bMARGINAL\b"), "LEAD"),
    (re.compile(r"\bREADY\b"), "FORWARD-TEST CANDIDATE"),
)


def relabel_text(text):
    """Rewrite legacy verdict words in free text (reasons, log lines, docs) to the ladder names. Non-strings pass through untouched."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pat, rep in _SUBS:
        out = pat.sub(rep, out)
    return out


def relabel_list(items):
    return [relabel_text(x) for x in (items or [])]
