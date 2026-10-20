"""Verdict ladder (report 2026-10-20, section 4): READY is no longer a binary purchase signal.

    REJECT -> LEAD -> FORWARD-TEST CANDIDATE -> EVALUATE -> FUNDED

* REJECT: a hard gate failed (lookahead, ruin, null test, pooled OOS FAIL, holdout loss, EV says NOT WORTH A FEE).
* LEAD: no hard failure, but the edge is not yet proven (pooled OOS lower bound <= 0, thin trades, capped at MARGINAL).
* FORWARD-TEST CANDIDATE: clears the pre-registered backtest gates AND the EV test. Permission to paper-trade, nothing more.
* EVALUATE: the forward test accepted H1 (SPRT) -- buy the cheapest evaluation.
* FUNDED: evaluation passed and a payout was received. Only set from recorded live facts, never from a backtest.

The last two stages can only be reached through `forward_state` (written by app.forward_test.sprt_monitor).
"""
from __future__ import annotations

from dataclasses import dataclass, field

STAGES = ("REJECT", "LEAD", "FORWARD-TEST CANDIDATE", "EVALUATE", "FUNDED")


@dataclass
class LadderResult:
    stage: str
    reasons: list = field(default_factory=list)
    legacy_verdict: str = ""
    ev_label: str = ""
    high_confidence: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def render(self) -> str:
        hc = " (HIGH CONFIDENCE)" if self.high_confidence else ""
        return f"LADDER: {self.stage}{hc}  |  verdict {self.legacy_verdict}  |  EV {self.ev_label or 'n/a'}"


def ladder_stage(verdict: str, ev_label: str | None, *, pooled_status: str | None = None, hard_failed: bool = False,
                 forward_state: str | None = None, per_attempt_pass_lb_pct: float | None = None,
                 high_confidence_pct: float = 70.0) -> LadderResult:
    """Map the legacy verdict + EV label (+ pooled OOS status, + recorded forward-test state) onto the ladder."""
    v = str(verdict or "").upper()
    ev = str(ev_label or "").upper()
    res = LadderResult(stage="LEAD", legacy_verdict=v, ev_label=ev)
    res.high_confidence = bool(ev == "HIGH CONFIDENCE" or (per_attempt_pass_lb_pct is not None and per_attempt_pass_lb_pct >= high_confidence_pct and v == "READY"))
    fs = str(forward_state or "").upper()
    if fs == "FUNDED":
        res.stage = "FUNDED"
        res.reasons.append("Evaluation passed and a payout was recorded.")
        return res
    if fs == "EVALUATE":
        res.stage = "EVALUATE"
        res.reasons.append("Forward-test SPRT accepted H1; buy the cheapest evaluation.")
        return res
    if hard_failed or (pooled_status or "").upper() == "FAIL" or ev == "NOT WORTH A FEE":
        res.stage = "REJECT"
        res.reasons.append("A hard gate failed or the EV test says the fee is not worth it.")
        return res
    if v in ("READY", "MARGINAL") and ev in ("HIGH CONFIDENCE", "FORWARD-TEST CANDIDATE") and (pooled_status or "PASS").upper() in ("PASS", "UNPROVEN", ""):
        res.stage = "FORWARD-TEST CANDIDATE"
        res.reasons.append("Backtest gates and EV test cleared: paper-trade it under the SPRT monitor before buying anything.")
        return res
    if v == "NOT READY":
        res.reasons.append("Not rejected outright, but not ready: treat as a lead to refine or to collect more trades for.")
    else:
        res.reasons.append("Evidence is positive but not proven (pooled lower bound, EV evidence or trade count still short).")
    return res
