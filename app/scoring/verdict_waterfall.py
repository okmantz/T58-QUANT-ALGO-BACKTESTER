"""One-screen explanation of a Full Pipeline verdict: from the raw score to the final word.

The pipeline decides a verdict in layers -- a 0-100 T58 Score, then hard gates that force NOT READY, then caps that hold a READY at
MARGINAL. Until now the layers were visible only as a long list of technical sentences. ``build_waterfall`` turns the same stored
reasons (nothing is recomputed, so it can never disagree with the verdict) into ordered steps:

    1. the raw score and how much of the evidence was actually measured
    2. every gate / cap that fired, in the order it bites, with the measured numbers pulled from the sentence, and
    3. the SINGLE change that would clear the most important blocker.
    4. the final verdict.

Pure function, no I/O. An unrecognised reason is still shown (first sentence) and never hidden: a new gate cannot make the waterfall lie.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

READY_AT = 75.0                 # T58 Score at which a verdict can be READY (see t58_scorecard._TIERS)

# (reason prefix, kind, short label, the one change that clears it). kind: gate = forces NOT READY, cap = holds READY at MARGINAL,
# info = context only. First match wins; order matters (more specific first).
_RULES: tuple[tuple[str, str, str, str], ...] = (
    ("HARD SAFETY GATE FAILED: the lookahead", "gate", "Peeks at future prices",
     "Fix the rule that reads data from after the bar it trades on (usually a higher-timeframe or whole-day value) and re-run."),
    ("HARD SAFETY GATE FAILED: Monte Carlo risk of ruin", "gate", "Too many simulated accounts blow up",
     "Lower risk per trade or tighten the stop; the Optimizer's risk sweep finds the largest risk that stays under the ruin limit."),
    ("UPPER 95% bound", "gate", "Edge disappears once costs are realistic",
     "Find a signal with more edge per trade (fewer, better entries) or a market where costs are smaller next to the stop."),
    ("EXPECTANCY NEGATIVE AFTER COSTS", "gate", "Loses money per trade after doubled costs",
     "Raise the average win relative to cost: wider target, better entries, or a larger bar size so costs are small next to the stop."),
    ("WALK-FORWARD", "gate", "Fails on data it was not tuned on",
     "Simplify the strategy (fewer settings) or re-optimize with the walk-forward-aware GA so settings are chosen out-of-sample."),
    ("HOLDOUT", "gate", "Loses money on the hidden final slice",
     "Do not tune on the last slice; re-run the optimizer and keep the holdout untouched."),
    ("CPCV", "gate", "Fails cross-validation", "Reduce parameters or widen the plateau; cross-validation says the result depended on the exact split."),
    ("DSR", "gate", "Edge not convincing after counting every idea tried",
     "Collect more trades/years of data, or test fewer variants: the Deflated Sharpe corrects for the total trial count."),
    ("PBO", "gate", "Probably the luckiest of many tries (overfit)", "Cut the search space or pick the centre of a stable parameter plateau, not the peak."),
    ("FUNDED ACCOUNTS DIE", "gate", "Funded accounts breach too often", "Lower risk per trade once funded, or add a daily loss stop."),
    ("PASS-AND-PAID TOO LOW", "gate", "Too few accounts pass AND get paid", "Raise the pass odds (see the next lines) or reduce risk so fewer accounts breach before the first payout."),
    ("RANDOM-ENTRY NULL NOT BEATEN", "gate", "Random entries do about as well",
     "The exits may be doing the work: strengthen the entry signal or test the exits on their own."),
    ("COST STRESS", "gate", "Profit does not survive doubled costs", "Trade a larger bar size or a more liquid market; costs matter less when the stop is wide."),
    ("REPLAY DISAGREES", "gate", "Simulation and real start dates disagree", "Re-run with more history; the resampled odds are not reproducing on real calendars."),
    ("SEED INSTABILITY", "gate", "Pass odds swing between runs", "Increase the Monte Carlo simulations and the trade count; the estimate is still noisy."),
    ("NO CROSS-INSTRUMENT TRANSFER", "cap", "Does not transfer to a sibling market", "Check the same rules on the sibling instrument; a real edge usually shows up there weakly."),
    ("RUIN WINDOW NOT COVERED", "cap", "History too short for the risk estimate", "Load more history so a full evaluation window can be measured."),
    ("SAMPLE FLOOR", "cap", "Too few trades or days", "Run on more history or a smaller bar size until the sample floor is met."),
    ("UNPROVEN", "cap", "Edge positive but not proven", "More out-of-sample trades: the 95% lower bound of average R per trade must clear zero."),
    ("EXPECTANCY NOT PROVEN", "cap", "Edge positive but not proven", "More out-of-sample trades: the 95% lower bound of average R per trade must clear zero."),
    ("NOT HIGH CONFIDENCE", "cap", "Confidence label below 'high'", "Add independent evidence (walk-forward, cross-instrument) so READY does not rest on one Monte Carlo."),
    ("CAPPED AT MARGINAL: the T58 Score alone would have called this READY, but the final", "cap", "Too few trades for READY",
     "Add trades: more history, a smaller bar size, or looser (but still tested) entry filters."),
    ("CAPPED AT MARGINAL: the T58 Score alone would have called this READY, but core", "cap", "Key safety checks were not measured",
     "Turn on the missing checks (pass odds, payout, ruin) in the pipeline settings and re-run."),
    ("CAPPED AT MARGINAL: the T58 Score alone would have called this READY, but only", "cap", "Too many checks not measured",
     "Turn on the unmeasured checks and re-run."),
    ("CAPPED AT MARGINAL: neither the walk-forward", "cap", "No out-of-sample check produced a result",
     "Enable walk-forward or CPCV (needs enough history) so READY is not in-sample only."),
    ("CAPPED AT MARGINAL: a strategy is not called READY while its holdout", "cap", "Holdout had too few trades", "Use a longer holdout or more history."),
    ("CAPPED AT MARGINAL: the holdout", "cap", "Holdout lost money, not provably", "Re-check on fresh data; the holdout did not confirm the edge."),
    ("CAPPED AT MARGINAL: READY needs evidence", "cap", "Needs evidence beyond one Monte Carlo",
     "Add walk-forward, CPCV or a cross-instrument check so READY does not rest on one resampled simulation."),
    ("CAPPED AT MARGINAL", "cap", "Held at MARGINAL", "See the sentence for the specific missing proof."),
    ("PER-ATTEMPT", "cap", "Per-attempt pass odds below the bar", "Raise pass odds with the Optimizer (pass-gradient fitness) or lower risk per trade."),
)
# Gates whose sentence does not begin with their own name (e.g. "HARD VALIDATION GATE FAILED: REJECTED by the Deflated Sharpe gate ...").
# Matched anywhere in the first 240 characters, upper-cased. Checked after _RULES, so every older, more specific rule still wins.
_CONTAINS_RULES: tuple[tuple[str, str, str, str], ...] = (
    ("DEFLATED SHARPE GATE", "gate", "Edge not convincing after counting every idea tried",
     "Collect more trades/years of data, or test fewer variants: the Deflated Sharpe corrects for the total trial count."),
    ("POOLED PURGED OUT-OF-SAMPLE", "gate", "Fails on data it was not tuned on",
     "Simplify the strategy (fewer settings) or re-optimize with the walk-forward-aware GA so settings are chosen out-of-sample."),
    ("PBO GATE", "gate", "Probably the luckiest of many tries (overfit)", "Cut the search space or pick the centre of a stable parameter plateau, not the peak."),
)
_CONTEXT = ("For reference,", "Evidence (", "This verdict does NOT lock", "Supporting diagnostic", "  ", "T58 Score", "ICIR /")
_NUM = re.compile(r"(?<![A-Za-z\d.])[-+]?\d+(?:\.\d+)?\s?(?:%|R|x|trades|days)?")


_TAG_PREFIX = re.compile(r"^\s*(?:HARD (?:VALIDATION|SAFETY) GATE FAILED|CAPPED AT MARGINAL|ADVISORY ONLY[^:]*)\s*:\s*", re.I)


def _first_sentence(text: str, limit: int = 4000) -> str:
    """The first sentence of ``text``, whole. The limit only exists to stop a runaway string and is far above any real reason, because a
    cut sentence ("... Sharpe of...") hid the numbers the reader needed (2026-10-31)."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    m = re.search(r"(?<=[.!?])\s", text)
    out = text[: m.start()] if m else text
    return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"


def _unknown_label(raw: str) -> str:
    """A short heading for a reason no rule recognises. The full sentence always appears underneath as the detail, so the heading is only a
    title: the tag-stripped text up to its first colon / dash / full stop when that is short, else a generic name for the kind of finding."""
    body = _TAG_PREFIX.sub("", re.sub(r"\s+", " ", str(raw)).strip())
    head = re.split(r":| -- |(?<=[.!?])\s", body, maxsplit=1)[0].strip().rstrip(".")
    if 3 <= len(head) <= 80:
        return head[0].upper() + head[1:]
    return "Validation gate failed" if "GATE FAILED" in str(raw)[:60].upper() else "Additional finding"


def _measured(text: str) -> str:
    """The first few numbers in the sentence, as the 'by how much' line (e.g. '62%, 70%'). Empty when there are none."""
    nums = [n.strip() for n in _NUM.findall(re.sub(r"\s+", " ", str(text)))[:3] if n.strip() and n.strip() not in ("0", "1", "2", "3")]
    return ", ".join(nums[:3])


def _get(sc: Any, key: str, default=None):
    if sc is None:
        return default
    return sc.get(key, default) if isinstance(sc, dict) else getattr(sc, key, default)


def build_waterfall(verdict: str | None, reasons: Iterable[str] | None, scorecard: Any = None) -> dict:
    """``{"verdict", "score", "tier", "coverage", "steps": [...], "single_change", "headline"}``.

    Each step: ``{"kind": score|gate|cap|advisory|final, "label", "detail", "measured", "fix", "effect"}``.
    ``effect`` is the verdict that step on its own would allow ("NOT READY", "MARGINAL" or "") so the screen reads top to bottom.
    """
    v = (verdict or "").upper().strip() or "UNTESTED"
    score, tier = _get(scorecard, "score"), _get(scorecard, "tier")
    cov = _get(scorecard, "weight_coverage")
    steps: list[dict] = []
    if score is not None:
        notes = []
        if cov is not None and cov < 0.999:
            notes.append(f"{cov * 100:.0f}% of the evidence was measured")
        if _get(scorecard, "missing_core"):
            notes.append("not measured: " + ", ".join(map(str, _get(scorecard, "missing_core"))))
        steps.append({"kind": "score", "label": f"Raw T58 Score {float(score):.0f}/100" + (f" ({tier})" if tier else ""),
                      "detail": ("Needs %.0f+ to be READY. " % READY_AT) + ("; ".join(notes) + "." if notes else ""),
                      "measured": f"{float(score):.0f}/100", "fix": "", "effect": "READY" if float(score) >= READY_AT else "MARGINAL"})
    seen: set[str] = set()
    for raw in reasons or []:
        raw = str(raw)
        if any(raw.startswith(p) for p in _CONTEXT):
            continue
        if raw.startswith("ADVISORY"):
            steps.append({"kind": "advisory", "label": "Advisory (does not change the verdict)", "detail": _first_sentence(raw[8:].lstrip(" :-")),
                          "measured": _measured(raw), "fix": "", "effect": ""})
            continue
        kind, label, fix = "info", _unknown_label(raw), ""
        matched = False
        for prefix, k, lab, f in _RULES:
            if raw.startswith(prefix) or (prefix.isupper() and prefix in raw[:60].upper()):
                kind, label, fix = k, lab, f
                matched = True
                break
        if not matched:
            head = raw[:240].upper()
            for needle, k, lab, f in _CONTAINS_RULES:
                if needle in head:
                    kind, label, fix = k, lab, f
                    matched = True
                    break
        if not matched and raw.upper().startswith("HARD VALIDATION GATE FAILED"):
            kind = "gate"                       # a validation gate that rejected the run is a gate even when no rule names it
        key = label if matched else raw             # one step per recognised gate; an unrecognised reason is never merged away
        if key in seen:
            continue
        seen.add(key)
        steps.append({"kind": kind, "label": label, "detail": _first_sentence(raw), "measured": _measured(raw), "fix": fix,
                      "effect": "NOT READY" if kind == "gate" else ("MARGINAL" if kind == "cap" else "")})
    blockers = [s for s in steps if s["kind"] in ("gate", "cap")]
    blockers.sort(key=lambda s: (s["kind"] != "gate",))              # a hard gate outranks a cap
    single = None
    if v != "READY" and blockers:
        b = blockers[0]
        single = {"label": b["label"], "fix": b["fix"] or b["detail"],
                  "note": ("This is the hard gate; clearing it moves the verdict to at best MARGINAL." if b["kind"] == "gate"
                           else "Clearing this lets the verdict rise to READY if nothing else blocks it.") + (
                      f" {len(blockers) - 1} other blocker(s) remain after it." if len(blockers) > 1 else "")}
    final_detail = {"READY": "Every gate passed and the score is high enough.",
                    "MARGINAL": "Promising, but at least one cap holds it below READY.",
                    "NOT READY": "At least one hard gate failed, or the score is too low.",
                    "UNTESTED": "No Full Pipeline result yet."}.get(v, "")
    steps.append({"kind": "final", "label": v, "detail": final_detail, "measured": "", "fix": "", "effect": v})
    gates, caps = sum(s["kind"] == "gate" for s in steps), sum(s["kind"] == "cap" for s in steps)
    if v == "READY":
        headline = "Nothing blocked this strategy."
    elif v == "UNTESTED":
        headline = "Not tested yet."
    else:
        bits = []
        if gates:
            bits.append(f"{gates} hard gate{'s' if gates != 1 else ''}")
        if caps:
            bits.append(f"{caps} cap{'s' if caps != 1 else ''}")
        headline = ("Held back by " + " and ".join(bits) + ".") if bits else "Held back by a low score."
    return {"verdict": v, "score": None if score is None else round(float(score), 1), "tier": tier,
            "coverage": cov, "steps": steps, "single_change": single, "headline": headline}
