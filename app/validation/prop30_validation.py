"""Validation gate for the 30-day prop objective (Phase 6). Preserves, never loosens, the existing hard gates: this module can only
ADD ways to fail. It works on ``TradeLedger`` objects produced by a FROZEN strategy configuration.

What it refuses to do:
  * call a window "untouched" if it overlaps any range that was previously observed (``observed_ranges``) or starts before the
    selection cut-off plus an embargo -- a holdout read before is development data (``independence_audit``);
  * report eventual / chain Monte Carlo pass as a 30-day pass (it never consumes one);
  * declare success from simulation alone: the best verdict a simulation-only run can reach is ``ESTABLISHED_ON_HOLDOUT_PENDING_FORWARD``;
    ``CONFIRMED`` additionally needs forward results recorded in ``ForwardRegistry``.
Statistics come from ``app.prop.thirty_day`` (dependence-aware intervals, calendar-day clock from purchase).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from app.prop.simulator import PropRules
from app.prop.thirty_day import ThirtyDayConfig, ThirtyDayResult, TradeLedger, claim_target_achieved, evaluate_thirty_day

DEFAULT_EMBARGO_DAYS = 30


@dataclass(frozen=True)
class ObservedRange:
    start: str
    end: str
    label: str


@dataclass
class Partitions:
    discovery: tuple              # (start, end)   where entries/exits were searched
    development: tuple            # (start, end)   where the survivor was tuned / selected
    holdout: tuple | None         # (start, end)   claimed untouched; None = none exists
    selection_cutoff: str | None = None    # last date any selection decision could have seen

    def to_dict(self) -> dict:
        return {"discovery": self.discovery, "development": self.development, "holdout": self.holdout, "selection_cutoff": self.selection_cutoff}


@dataclass
class IndependenceAudit:
    holdout_independent: bool
    problems: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"holdout_independent": self.holdout_independent, "problems": self.problems}


def independence_audit(parts: Partitions, observed_ranges: Sequence[ObservedRange] = (), embargo_days: int = DEFAULT_EMBARGO_DAYS) -> IndependenceAudit:
    probs: list = []
    if parts.holdout is None:
        return IndependenceAudit(False, ["No holdout window was supplied: nothing independent exists to validate on."])
    hs, he = pd.Timestamp(parts.holdout[0]), pd.Timestamp(parts.holdout[1])
    if he <= hs:
        probs.append("The holdout window is empty or reversed.")
    cutoff = pd.Timestamp(parts.selection_cutoff) if parts.selection_cutoff else pd.Timestamp(parts.development[1])
    if hs < cutoff + pd.Timedelta(days=embargo_days):
        probs.append(f"The holdout starts {hs.date()}, inside the {embargo_days}-day embargo after the selection cut-off {cutoff.date()}: "
                     "an attempt purchased then overlaps data the selection saw.")
    for r in observed_ranges:
        rs, re_ = pd.Timestamp(r.start), pd.Timestamp(r.end)
        if hs < re_ and rs < he:
            probs.append(f"The holdout overlaps a range already observed ({r.label}: {r.start}..{r.end}). A window that was looked at before "
                         "cannot be reused as independent validation.")
    for name, w in (("discovery", parts.discovery), ("development", parts.development)):
        if pd.Timestamp(w[1]) > hs:
            probs.append(f"The {name} window runs to {pd.Timestamp(w[1]).date()}, past the holdout start {hs.date()}.")
    return IndependenceAudit(not probs, probs)


# ------------------------------------------------------------------------------------------------------------------ forward test
class ForwardRegistry:
    """Append-only JSON log of forward (paper/live) 30-day attempts for a frozen configuration."""

    def __init__(self, path):
        self.path = Path(path)

    def _load(self) -> list:
        try:
            return json.loads(self.path.read_text()) if self.path.exists() else []
        except (OSError, ValueError):
            return []

    def record(self, config_id: str, start: str, outcome: str, days: float | None = None) -> None:
        if outcome not in ("passed", "failed", "open"):
            raise ValueError("outcome must be passed / failed / open")
        rows = self._load()
        rows.append({"config_id": config_id, "start": start, "outcome": outcome, "days": days})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(rows, indent=2))

    def summary(self, config_id: str, horizon_days: int = 30) -> dict:
        rows = [r for r in self._load() if r["config_id"] == config_id and r["outcome"] != "open"]
        n = len(rows)
        k = sum(1 for r in rows if r["outcome"] == "passed" and (r.get("days") is None or r["days"] <= horizon_days))
        from app.prop.thirty_day import wilson_interval
        lo, hi = wilson_interval(k / n, n) if n else (0.0, 1.0)
        return {"n_resolved": n, "n_passed": k, "p_pass": (k / n) if n else None, "lower": lo, "upper": hi}


# ------------------------------------------------------------------------------------------------------------------ validation
@dataclass
class ValidationReport:
    config_id: str
    verdict: str
    reasons: list
    partitions: dict
    independence: dict
    discovery: dict | None
    development: dict | None
    walk_forward: list
    block_robustness: dict
    holdout: dict | None
    forward: dict | None
    target_probability: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _brief(res: ThirtyDayResult | None) -> dict | None:
    if res is None:
        return None
    return {"n_windows": res.n_windows, "p_pass_30": res.p_pass_30, "lower_bound": res.lower_bound, "p_bust_30": res.p_bust_30,
            "p_pass_60": res.p_pass_60, "median_days": res.median_days_to_pass, "n_trades": res.n_trades,
            "n_independent": res.interval.get("n_independent"), "meets_target_lower_bound": res.meets_target_lower_bound,
            "cost_1.5x_p_pass_30": (res.cost_sensitivity.get(1.5) or {}).get("p_pass_30")}


def _eval(led: TradeLedger, rules, span, cfg) -> ThirtyDayResult | None:
    sub = led.between(span[0], span[1])
    if not len(sub):
        return None
    return evaluate_thirty_day(sub, rules, data_start=pd.Timestamp(span[0]), data_end=pd.Timestamp(span[1]), cfg=cfg, with_sensitivity=True)


def walk_forward_spans(start, end, test_days: int = 120, step_days: int | None = None) -> list:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    step = step_days or test_days
    spans, cur = [], s
    while cur + pd.Timedelta(days=test_days) <= e:
        spans.append((cur, cur + pd.Timedelta(days=test_days)))
        cur += pd.Timedelta(days=step)
    return spans


def validate_prop30(ledger: TradeLedger, rules: PropRules, parts: Partitions, *, config_id: str = "candidate",
                    observed_ranges: Sequence[ObservedRange] = (), cfg: ThirtyDayConfig | None = None,
                    forward: ForwardRegistry | None = None, sealed_ledger=None, sealed_fp: str | None = None,
                    min_forward_attempts: int = 6, wf_test_days: int = 120) -> ValidationReport:
    cfg = cfg or ThirtyDayConfig()
    target = cfg.target_probability
    reasons: list = []
    aud = independence_audit(parts, observed_ranges)
    disc = _eval(ledger, rules, parts.discovery, cfg)
    dev = _eval(ledger, rules, parts.development, cfg)

    wf = []
    for sp in walk_forward_spans(parts.development[0], parts.development[1], test_days=wf_test_days):
        r = _eval(ledger, rules, sp, cfg)
        wf.append({"span": [str(sp[0].date()), str(sp[1].date())], **(_brief(r) or {"n_windows": 0, "p_pass_30": 0.0, "lower_bound": 0.0})})
    wf_ok = [w for w in wf if w.get("n_windows", 0) > 0]
    wf_pass_share = (sum(1 for w in wf_ok if w["p_pass_30"] >= 0.5 * target) / len(wf_ok)) if wf_ok else 0.0

    # block robustness (NOT full CPCV): leave-one-block-out dispersion of the 30-day pass rate across contiguous blocks
    blocks = walk_forward_spans(parts.development[0], parts.development[1], test_days=max(90, wf_test_days // 2))
    bp = [(_eval(ledger, rules, b, cfg) or None) for b in blocks]
    vals = np.array([r.p_pass_30 for r in bp if r is not None])
    rob = {"n_blocks": int(len(vals)), "min": float(vals.min()) if len(vals) else None, "median": float(np.median(vals)) if len(vals) else None,
           "share_blocks_ge_half_target": float((vals >= 0.5 * target).mean()) if len(vals) else 0.0,
           "note": "contiguous-block dispersion, not a full CPCV purge/embargo combinatorial run"}

    hold = None
    hold_res = None
    if aud.holdout_independent and parts.holdout is not None:
        if sealed_ledger is not None and sealed_fp:
            sealed_ledger.spend(sealed_fp)                    # raises HoldoutExhausted when the lifetime budget is gone
        hold_res = _eval(ledger, rules, parts.holdout, cfg)
        hold = _brief(hold_res)

    fwd = forward.summary(config_id, cfg.horizon_days) if forward is not None else None

    # ---- gate
    if dev is None or dev.p_pass_30 < 0.5 * target:
        reasons.append(f"Development P(pass<=30d) is {0.0 if dev is None else dev.p_pass_30 * 100:.1f}%: below half the {target * 100:.0f}% target, "
                       "so there is nothing worth validating.")
    if not aud.holdout_independent:
        reasons.extend(aud.problems)
        reasons.append("No independent holdout: the 30-day claim cannot be established from this data. A forward test (or newer, unseen "
                       "data) is required.")
    elif hold_res is None:
        reasons.append("The holdout window contains no trades or no complete purchase windows.")
    else:
        if not claim_target_achieved(hold_res):
            reasons.append(f"Holdout: 95% lower bound {hold_res.lower_bound * 100:.1f}% (point {hold_res.p_pass_30 * 100:.1f}%) does not reach "
                           f"{target * 100:.0f}% on {hold_res.interval.get('n_independent', 0):.0f} independent windows, or fails at 1.5x costs.")
    if wf_ok and wf_pass_share < 0.6:
        reasons.append(f"Walk-forward: only {wf_pass_share * 100:.0f}% of test spans reach half the target pass rate (need >= 60%).")
    if rob["share_blocks_ge_half_target"] < 0.6:
        reasons.append("Block robustness: the pass rate is concentrated in a minority of blocks, i.e. one regime carries the result.")

    if reasons:
        verdict = "NOT_ESTABLISHED"
    else:
        fwd_ok = bool(fwd and fwd["n_resolved"] >= min_forward_attempts and fwd["lower"] >= target)
        verdict = "CONFIRMED" if fwd_ok else "ESTABLISHED_ON_HOLDOUT_PENDING_FORWARD"
        if not fwd_ok:
            reasons.append(f"Simulation alone cannot confirm success: needs >= {min_forward_attempts} resolved forward attempts with a lower bound "
                           f">= {target * 100:.0f}% (have {0 if not fwd else fwd['n_resolved']}).")
    return ValidationReport(config_id, verdict, reasons, parts.to_dict(), aud.to_dict(), _brief(disc), _brief(dev), wf, rob, hold, fwd, target)
