"""Champion Board: a candidate board with dimensions (Eval / Payout / OOS),
plus an explicit, gated promotion pipeline -- replacing the dashboard's old
"whichever strategy has the highest Sharpe ratio is the champion" logic.

Why this exists
----------------
Owen's own framing: "A strategy shouldn't automatically become the champion
simply because it has a high raw metric." Sharpe (or any single raw metric)
says nothing about whether a strategy has actually been through the app's
own validation tools -- a strategy that has never been CPCV/walk-forward
checked can still post a great single-run Sharpe on lucky data. This module
answers a different question: "which strategies have EARNED trust through
this app's own pipeline, and how far has each one actually gotten."

Data source
-----------
Nothing new is tracked here. Every number this module reports already lives
on a saved strategy's metadata sidecar (see app.strategy.library):
  - last_run:            eval_pass_probability, first_payout_probability
  - last_validation:     efficiency (CPCV mean-OOS-metric or walk-forward
                          efficiency), is_robust, pbo
  - last_champion_check:  verdict (READY/MARGINAL/NOT READY), t58_score
  - promotion (new, this module's own field): stage + a timestamped history
    of every promotion, so "how long has this been in Forward Testing"
    is answerable without a separate tracker file.

Promotion is a manual, explicit action (promote_strategy), never automatic
-- exactly Owen's requirement #7. Calling it only succeeds when every
requirement for the NEXT stage is already met; otherwise it returns the
specific unmet requirement(s) so the UI can show exactly what's missing
instead of a bare "can't promote" message.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.strategy.library import (
    STRATEGY_TYPES,
    list_saved_strategies,
    save_strategy_metadata,
)

# ---------------------------------------------------------------------------
# Promotion stages
# ---------------------------------------------------------------------------
# Deliberately distinct from library.PIPELINE_STAGE_NAMES (Create/Test/
# Optimize/Validate/Champion Check/Ready), which tracks "which TOOLS have
# touched this strategy." This tracks "how much do we actually trust it
# enough to risk something on it" -- a strategy can be Champion-Checked
# (tooled through) and still sit at "candidate" here forever if nobody has
# explicitly promoted it.

PROMOTION_STAGES = ("candidate", "validated", "champion_candidate", "forward_testing", "production_ready")

PROMOTION_STAGE_TITLES: dict[str, str] = {
    "candidate": "Candidate",
    "validated": "Validated",
    "champion_candidate": "Champion Candidate",
    "forward_testing": "Forward Testing",
    "production_ready": "Production Ready",
}

# Minimum wall-clock time a strategy must sit in Forward Testing before it's
# eligible for Production Ready. This app doesn't yet track live/forward
# fill-by-fill correlation against the backtest (see app.forward_test), so
# rather than fabricate a pass/fail signal from data that doesn't exist,
# this stage gate is honestly time-based: enough calendar days must have
# actually elapsed since promotion to Forward Testing for a live sample to
# mean anything at all. Owen can tighten/loosen this once real forward-test
# result tracking lands.
MIN_FORWARD_TEST_DAYS = 14.0

# Thresholds for the Validated -> Champion Candidate gate. Chosen to match
# the same eval/payout bar Full Pipeline's own scorecard treats as
# meaningfully strong (see app.scoring.t58_scorecard), not arbitrary.
CHAMPION_CANDIDATE_MIN_EVAL_PCT = 60.0
CHAMPION_CANDIDATE_MIN_PAYOUT_PCT = 55.0
CHAMPION_CANDIDATE_MIN_OOS_PCT = 50.0

# Champion Candidate -> Forward Testing gate: this is the point where real
# money-shaped risk starts, so the bar is the app's own top verdict, not
# just "better than the previous gate."
FORWARD_TEST_MIN_EVAL_PCT = 70.0


def _num(d: dict, *keys: str) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def oos_pct(last_validation: dict) -> Optional[float]:
    """Best-effort OOS-quality percentage from whichever validation method
    actually ran. Never fabricates a number: returns None (displayed as
    "--", never as 0%) when nothing usable is on record.

    - CPCV/walk-forward record efficiency as a 0..1 fraction (see
      full_pipeline.py's record_validation_result calls) -- scaled to a
      percentage.
    - A bare PBO (Probability of Backtest Overfitting) is inverted: a low
      PBO means the result is NOT likely overfit, i.e. good OOS quality.
    """
    if not last_validation:
        return None
    efficiency = _num(last_validation, "efficiency", "oos_efficiency", "walk_forward_efficiency", "oos_pct")
    if efficiency is not None:
        # efficiency is a 0..1 fraction in every current caller; guard
        # against a caller that already scaled to 0..100.
        return round(efficiency * 100.0, 1) if efficiency <= 1.0 else round(efficiency, 1)
    pbo = _num(last_validation, "pbo")
    if pbo is not None:
        return round(max(0.0, 100.0 - pbo), 1)
    return None


def status_from_verdict(verdict: Optional[str]) -> str:
    """DEVELOPING / MARGINAL / READY board-status label -- what the
    Champion Board's "Status" column shows. Distinct from promotion stage
    (a strategy can be verdict=READY and still sit at promotion
    stage=candidate until someone actually promotes it)."""
    v = (verdict or "").upper()
    if v == "READY":
        return "READY"
    if v == "MARGINAL":
        return "MARGINAL"
    if v == "NOT READY":
        return "NOT READY"
    return "DEVELOPING"


@dataclass
class PromotionRequirement:
    key: str
    label: str
    met: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "met": self.met, "detail": self.detail}


@dataclass
class PromotionState:
    stage: str
    stage_title: str
    next_stage: Optional[str]
    next_stage_title: Optional[str]
    can_promote: bool
    requirements: list[PromotionRequirement] = field(default_factory=list)
    promoted_at: dict[str, float] = field(default_factory=dict)  # stage -> unix ts

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "stage_title": self.stage_title,
            "stage_index": PROMOTION_STAGES.index(self.stage),
            "stage_pct": round(100.0 * PROMOTION_STAGES.index(self.stage) / (len(PROMOTION_STAGES) - 1), 1),
            "next_stage": self.next_stage,
            "next_stage_title": self.next_stage_title,
            "can_promote": self.can_promote,
            "requirements": [r.to_dict() for r in self.requirements],
            "promoted_at": self.promoted_at,
        }


def _requirements_for_next_stage(metadata: dict[str, Any], current_stage: str) -> list[PromotionRequirement]:
    """Explicit, named requirements to advance FROM current_stage to the
    next one. Every transition has its own concrete checklist -- nothing
    here is "just a higher number is better," each item is a pass/fail
    gate that either holds or doesn't."""
    last_run = metadata.get("last_run") or {}
    last_validation = metadata.get("last_validation") or {}
    last_champion_check = metadata.get("last_champion_check") or {}
    lookahead = metadata.get("lookahead") or {}
    promotion = metadata.get("promotion") or {}

    eval_pct = _num(last_run, "eval_pass_probability")
    payout_pct = _num(last_run, "first_payout_probability", "eval_pass_probability")
    oos = oos_pct(last_validation)
    verdict = str(last_champion_check.get("verdict") or last_run.get("verdict") or "").upper()

    if current_stage == "candidate":
        return [
            PromotionRequirement(
                "backtested", "Has at least one recorded backtest",
                bool(last_run), "Run this strategy through Run & Report or Full Pipeline first." if not last_run else "",
            ),
            PromotionRequirement(
                "validated_run", "A deeper validation check has been run (CPCV / PBO / Walk-Forward)",
                bool(last_validation),
                "" if last_validation else "Run CPCV or Walk-Forward Opt against this strategy.",
            ),
            PromotionRequirement(
                "lookahead_clean", "No confirmed lookahead-bias leak",
                lookahead.get("clean") is not False,
                "" if lookahead.get("clean") is not False else "Fix the confirmed lookahead leak before validating further.",
            ),
        ]

    if current_stage == "validated":
        return [
            PromotionRequirement(
                "eval_threshold", f"Eval pass probability >= {CHAMPION_CANDIDATE_MIN_EVAL_PCT:.0f}%",
                eval_pct is not None and eval_pct >= CHAMPION_CANDIDATE_MIN_EVAL_PCT,
                f"Currently {eval_pct:.1f}%." if eval_pct is not None else "No eval pass probability on record yet.",
            ),
            PromotionRequirement(
                "payout_threshold", f"Payout probability >= {CHAMPION_CANDIDATE_MIN_PAYOUT_PCT:.0f}%",
                payout_pct is not None and payout_pct >= CHAMPION_CANDIDATE_MIN_PAYOUT_PCT,
                f"Currently {payout_pct:.1f}%." if payout_pct is not None else "No payout probability on record yet.",
            ),
            PromotionRequirement(
                "oos_threshold", f"OOS validation quality >= {CHAMPION_CANDIDATE_MIN_OOS_PCT:.0f}%",
                oos is not None and oos >= CHAMPION_CANDIDATE_MIN_OOS_PCT,
                f"Currently {oos:.1f}%." if oos is not None else "No OOS efficiency on record yet.",
            ),
            PromotionRequirement(
                "not_rejected", "Champion Check has not returned NOT READY",
                verdict != "NOT READY",
                "" if verdict != "NOT READY" else "Last Champion Check verdict was NOT READY.",
            ),
        ]

    if current_stage == "champion_candidate":
        return [
            PromotionRequirement(
                "champion_check_ready", "Champion Check verdict is READY",
                verdict == "READY",
                f"Currently {verdict or 'not checked'}." if verdict != "READY" else "",
            ),
            PromotionRequirement(
                "eval_threshold", f"Eval pass probability >= {FORWARD_TEST_MIN_EVAL_PCT:.0f}%",
                eval_pct is not None and eval_pct >= FORWARD_TEST_MIN_EVAL_PCT,
                f"Currently {eval_pct:.1f}%." if eval_pct is not None else "No eval pass probability on record yet.",
            ),
        ]

    if current_stage == "forward_testing":
        entered_at = promotion.get("history", {}).get("forward_testing")
        elapsed_days = (time.time() - entered_at) / 86400.0 if entered_at else 0.0
        return [
            PromotionRequirement(
                "min_duration", f"At least {MIN_FORWARD_TEST_DAYS:.0f} days in Forward Testing",
                elapsed_days >= MIN_FORWARD_TEST_DAYS,
                f"{elapsed_days:.1f} of {MIN_FORWARD_TEST_DAYS:.0f} days elapsed." if entered_at else "Not yet promoted to Forward Testing.",
            ),
        ]

    return []  # production_ready is the final stage -- nothing further to require


def evaluate_promotion(metadata: dict[str, Any]) -> PromotionState:
    """The full promotion picture for one strategy's metadata dict: which
    stage it's actually at (never inferred past what promote_strategy has
    explicitly recorded), what the next stage is, and exactly what's
    missing to get there."""
    promotion = metadata.get("promotion") or {}
    stage = promotion.get("stage") or "candidate"
    if stage not in PROMOTION_STAGES:
        stage = "candidate"
    idx = PROMOTION_STAGES.index(stage)
    next_stage = PROMOTION_STAGES[idx + 1] if idx + 1 < len(PROMOTION_STAGES) else None

    requirements = _requirements_for_next_stage(metadata, stage)
    can_promote = bool(next_stage) and all(r.met for r in requirements)

    return PromotionState(
        stage=stage,
        stage_title=PROMOTION_STAGE_TITLES[stage],
        next_stage=next_stage,
        next_stage_title=PROMOTION_STAGE_TITLES.get(next_stage) if next_stage else None,
        can_promote=can_promote,
        requirements=requirements,
        promoted_at=dict(promotion.get("history") or {}),
    )


def promote_strategy(strategy_type: str, filename: str, metadata: Optional[dict[str, Any]] = None) -> tuple[bool, str, Optional[str]]:
    """Attempts to advance one saved strategy to the next promotion stage.
    Never automatic -- this is only ever called from an explicit user
    action (a Promote button). Returns (ok, message, new_stage). On
    failure, message names exactly which requirement(s) are unmet.

    metadata can be passed in (e.g. already loaded for a Champion Board
    row) to avoid a redundant disk read; otherwise it's loaded fresh."""
    from app.strategy.library import load_strategy_metadata

    if metadata is None:
        metadata = load_strategy_metadata(strategy_type, filename)

    state = evaluate_promotion(metadata)
    if state.next_stage is None:
        return False, f"'{filename}' is already at the final stage ({state.stage_title}).", None

    unmet = [r for r in state.requirements if not r.met]
    if unmet:
        lines = "; ".join(f"{r.label} -- {r.detail or 'not met'}" for r in unmet)
        return False, f"Can't promote to {state.next_stage_title} yet: {lines}", None

    history = dict((metadata.get("promotion") or {}).get("history") or {})
    history[state.next_stage] = time.time()
    save_strategy_metadata(strategy_type, filename, {
        "promotion": {"stage": state.next_stage, "history": history},
    }, merge=True)
    return True, f"Promoted '{filename}' to {PROMOTION_STAGE_TITLES[state.next_stage]}.", state.next_stage


def demote_strategy(strategy_type: str, filename: str, to_stage: str = "candidate") -> tuple[bool, str]:
    """Manual downgrade -- e.g. a strategy that regressed during Forward
    Testing and should be pulled back rather than left showing a stage it
    no longer deserves. Always allowed (no gate on stepping backward)."""
    if to_stage not in PROMOTION_STAGES:
        return False, f"Unknown stage '{to_stage}'."
    from app.strategy.library import load_strategy_metadata

    metadata = load_strategy_metadata(strategy_type, filename)
    history = dict((metadata.get("promotion") or {}).get("history") or {})
    history[f"demoted_from_{(metadata.get('promotion') or {}).get('stage', 'candidate')}"] = time.time()
    save_strategy_metadata(strategy_type, filename, {
        "promotion": {"stage": to_stage, "history": history},
    }, merge=True)
    return True, f"'{filename}' moved back to {PROMOTION_STAGE_TITLES[to_stage]}."


def candidate_dimensions(metadata: dict[str, Any]) -> dict[str, Any]:
    """The board's per-strategy dimension readout: Status, Eval%, Payout%,
    OOS%. Any dimension with no data on record reports None (rendered as
    "--"), never a fabricated 0."""
    last_run = metadata.get("last_run") or {}
    last_validation = metadata.get("last_validation") or {}
    last_champion_check = metadata.get("last_champion_check") or {}
    verdict = str(last_champion_check.get("verdict") or last_run.get("verdict") or "").upper() or None
    return {
        "status": status_from_verdict(verdict),
        "verdict": verdict,
        "eval_pct": _num(last_run, "eval_pass_probability"),
        "payout_pct": _num(last_run, "first_payout_probability", "eval_pass_probability"),
        "oos_pct": oos_pct(last_validation),
        "t58_score": _num(last_champion_check, "t58_score"),
    }


def _composite_strength(dims: dict[str, Any]) -> Optional[float]:
    """Weighted composite of the three board dimensions, used ONLY to
    order/select among strategies that have already earned a validated
    promotion stage -- never as a substitute for actually being validated.
    Missing dimensions are excluded from the weighted average (not
    treated as 0) so a strategy that simply hasn't run one particular
    check isn't penalized as if it had failed it."""
    weights = {"eval_pct": 0.40, "payout_pct": 0.35, "oos_pct": 0.25}
    total_w, total_v = 0.0, 0.0
    for key, w in weights.items():
        v = dims.get(key)
        if v is not None:
            total_v += w * v
            total_w += w
    if total_w == 0:
        return None
    return round(total_v / total_w, 2)


def list_board(strategy_type: Optional[str] = None) -> list[dict[str, Any]]:
    """One row per saved strategy across the given type (or all three),
    combining its board dimensions and promotion state. This is the
    Champion Board's actual data source."""
    types = [strategy_type] if strategy_type else list(STRATEGY_TYPES)
    rows: list[dict[str, Any]] = []
    for t in types:
        for item in list_saved_strategies(t):
            dims = candidate_dimensions(item.metadata)
            promo = evaluate_promotion(item.metadata)
            rows.append({
                "strategy_type": t,
                "filename": item.name,
                "display_name": Path(item.name).stem,
                **dims,
                "strength": _composite_strength(dims),
                "promotion": promo.to_dict(),
                "modified": item.modified,
            })
    rows.sort(key=lambda r: r["modified"], reverse=True)
    return rows


def strongest_validated_candidate(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The board's actual "champion" pick: the strongest candidate strategy
    that has already earned at least "validated" promotion stage (i.e. has
    genuinely been through this app's own validation tools, not just
    whichever one happens to post the highest raw metric). Among eligible
    rows, ranks by promotion-stage depth first (further along the pipeline
    beats a higher raw score at an earlier stage), then by composite
    strength as the tiebreaker. Returns None -- not a fallback guess --
    when nothing has reached "validated" yet, so the caller can say so
    honestly instead of crowning an unvalidated strategy."""
    eligible = [r for r in rows if r["promotion"]["stage"] != "candidate"]
    if not eligible:
        return None

    def sort_key(r):
        stage_rank = PROMOTION_STAGES.index(r["promotion"]["stage"])
        strength = r["strength"] if r["strength"] is not None else -1.0
        return (stage_rank, strength)

    return max(eligible, key=sort_key)


# ---------------------------------------------------------------------------
# The dashboard's "five questions" -- see app.web.server's /dashboard route
# and app.ui.main_window's dashboard tab, both of which call this with the
# person's currently-tracked strategy (app.reports.strategy_state) matched
# against the Champion Board, falling back to the board's own strongest
# candidate when nothing is explicitly tracked.
# ---------------------------------------------------------------------------

def _diagnose_why(row: dict[str, Any]) -> str:
    """A specific, numbers-backed reason for the current state -- never a
    generic "it's fine" or "it's broken" with nothing underneath it."""
    verdict = row.get("verdict")
    eval_pct, payout_pct, oos = row.get("eval_pct"), row.get("payout_pct"), row.get("oos_pct")

    if verdict is None:
        return "Hasn't been through a Champion Check yet -- no verdict on record."
    if verdict == "NOT READY":
        return "Failed its last Champion Check (a hard safety gate, or a very weak scorecard)."
    if verdict == "READY":
        return "Passed its last Champion Check with no open issues."

    # MARGINAL / DEVELOPING: point at the specific weakest dimension.
    dims = [("eval_pct", "Eval pass rate", eval_pct), ("payout_pct", "Payout probability", payout_pct), ("oos_pct", "OOS validation", oos)]
    known = [(label, v) for _key, label, v in dims if v is not None]
    if not known:
        return "Not enough data recorded yet to diagnose -- run a deeper validation check."
    label, value = min(known, key=lambda kv: kv[1])
    if label == "OOS validation":
        return f"OOS degradation -- out-of-sample validation quality is only {value:.0f}%."
    return f"{label} is the weakest area at {value:.0f}%."


def _next_action(row: dict[str, Any]) -> str:
    promo = row["promotion"]
    if promo["can_promote"] and promo["next_stage_title"]:
        return f"Promote to {promo['next_stage_title']} -- every requirement is already met."
    unmet = [r for r in promo["requirements"] if not r["met"]]
    if unmet:
        return unmet[0]["label"] + (f" ({unmet[0]['detail']})" if unmet[0]["detail"] else "")
    if row.get("verdict") == "NOT READY":
        return "Go back to Search Lab or Evolution Lab for a different candidate."
    return "Run a Champion Check (Full Pipeline) to get a verdict."


def five_question_snapshot(current: Optional[dict], rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Answers the five questions the dashboard's first screen should
    always answer: what am I working on, where is it in the process, is
    it working, why, and what should I do next. `current` is
    strategy_state.get_current_strategy()'s dict (or None); when set, this
    prefers the matching board row so the dashboard reflects whatever the
    person is actually focused on, falling back to the board's own
    strongest validated candidate when nothing is tracked. Returns None
    only when there is genuinely nothing to show (empty library)."""
    row = None
    if current:
        name = (current.get("strategy_name") or "").strip().lower()
        for r in rows:
            if Path(r["filename"]).stem.strip().lower() == name or r["filename"].strip().lower() == name:
                row = r
                break
    if row is None:
        row = strongest_validated_candidate(rows) or (rows[0] if rows else None)
    if row is None:
        return None

    return {
        "what": Path(row["filename"]).stem,
        "where": row["promotion"]["stage_title"],
        "where_pct": row["promotion"]["stage_pct"],
        "is_it_working": row.get("status", "DEVELOPING"),
        "why": _diagnose_why(row),
        "next_action": _next_action(row),
        "row": row,
    }
