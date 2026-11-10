"""
Automated Payout Optimization: how a strategy should size its risk as the account moves through the evaluation and funded
lifecycle.

The goal of the account changes at every stage, so the risk it is allowed to take changes with it:

    evaluation          Pass without excessive breach risk
    newly_funded        Preserve account survival
    approaching_payout  Satisfy payout conditions
    post_first_payout   Preserve the remaining cushion
    scaling             Maximize sustainable withdrawal potential

Built on the Rulebook v2 sizing in ``app.prop.lifecycle_v2`` (cushion-proportional risk, the 3R / 60%-of-soft-limit day stop).
The user picks a preset -- Conservative, Balanced or Accelerated -- which scales the base policy and moves the Guardian's
warn / reduce / block thresholds. ``HARD LIMITS ARE NEVER OVERRIDDEN``: whatever the preset, the final risk is clamped so one
trade can never use more than a fixed share of the remaining daily-loss allowance or of the drawdown cushion, and can never exceed
the firm's contract cap. ``risk_plan`` reports every clamp it applied, so the page can say why a number is smaller than the preset
alone would give.

Pure functions, no I/O and no Flask, so the Deploy pages, the Guardian and the tests share one source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.prop import lifecycle_v2 as lv2

STAGES = ("evaluation", "newly_funded", "approaching_payout", "post_first_payout", "scaling")

STAGE_INFO = {
    "evaluation": {"label": "Evaluation", "objective": "Pass without excessive breach risk"},
    "newly_funded": {"label": "Newly funded", "objective": "Preserve account survival"},
    "approaching_payout": {"label": "Approaching payout", "objective": "Satisfy payout conditions"},
    "post_first_payout": {"label": "After first payout", "objective": "Preserve the remaining cushion"},
    "scaling": {"label": "Scaling", "objective": "Maximize sustainable withdrawal potential"},
}

# Multiplier on the Rulebook v2 base risk for each stage (before the preset). Evaluation uses the v2 evaluation policy as is; the
# funded stages use the v2 funded policy, tightened while survival or the payout conditions are what matter most.
STAGE_RISK_MULT = {"evaluation": 1.0, "newly_funded": 0.8, "approaching_payout": 0.6, "post_first_payout": 0.7, "scaling": 1.0}

PRESETS = {
    "conservative": {"label": "Conservative", "risk_scale": 0.6, "warn": 0.50, "reduce": 0.70, "block": 0.85, "day_stop_frac": 0.50,
                     "blurb": "Smaller size, earlier warnings. Slower, but built to keep the account alive."},
    "balanced": {"label": "Balanced", "risk_scale": 1.0, "warn": 0.60, "reduce": 0.80, "block": 0.92, "day_stop_frac": 0.60,
                 "blurb": "The Rulebook v2 policy as written."},
    "accelerated": {"label": "Accelerated", "risk_scale": 1.4, "warn": 0.70, "reduce": 0.88, "block": 0.96, "day_stop_frac": 0.75,
                    "blurb": "Bigger size and later brakes, to reach targets faster. Still stopped by the hard limits below."},
}
DEFAULT_PRESET = "balanced"

# Hard limits: no preset, and no user setting, can exceed these. They are shares of what is LEFT, not of the account.
HARD_MAX_SHARE_OF_REMAINING_DAILY = 0.25      # one trade never risks more than 25% of the remaining daily-loss allowance
HARD_MAX_SHARE_OF_CUSHION = 0.20              # ... nor more than 20% of the drawdown cushion
PAYOUT_APPROACH_FRACTION = 0.70               # at 70% of the payout threshold the account counts as "approaching payout"


def preset(name: str | None) -> dict:
    return PRESETS.get(str(name or "").lower(), PRESETS[DEFAULT_PRESET])


def preset_name(name: str | None) -> str:
    n = str(name or "").lower()
    return n if n in PRESETS else DEFAULT_PRESET


def stage_for(account: dict, *, payout_progress: float | None = None) -> str:
    """Lifecycle stage for a Prop Desk account.

    ``payout_progress`` is profit earned toward the first payout as a fraction of what it needs (1.0 = eligible). When it is None the
    account is treated as "newly funded" until it has a payout on record."""
    if account.get("stage") != "funded":
        return "evaluation"
    n = len(account.get("payouts") or [])
    if n >= 2:
        return "scaling"
    if n == 1:
        return "post_first_payout"
    if payout_progress is not None and payout_progress >= PAYOUT_APPROACH_FRACTION:
        return "approaching_payout"
    return "newly_funded"


@dataclass
class RiskPlan:
    stage: str
    preset: str
    objective: str
    risk_per_trade: float               # dollars, AFTER every clamp
    preset_risk_per_trade: float        # dollars the policy and preset alone would give
    day_stop: float                     # stop trading for the day after losing this much
    max_contracts: int | None
    cushion: float
    clamps: list = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def risk_plan(
    stage: str,
    preset_name_: str | None,
    rules,
    *,
    balance: float | None = None,
    floor: float | None = None,
    daily_pnl: float = 0.0,
    max_contracts: int | None = None,
) -> RiskPlan:
    """Per-trade risk and day stop for ``stage`` under ``preset_name_``.

    ``rules`` is a ``PropRules``, a ``V2Rules`` or anything with the same fields. ``balance`` / ``floor`` default to a fresh account
    (balance = account size, floor = size - drawdown). ``daily_pnl`` is today's realized P&L so far (negative on a losing day).
    ``max_contracts`` is the firm's contract cap if there is one."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}")
    p = preset(preset_name_)
    v2 = rules if isinstance(rules, lv2.V2Rules) else lv2.V2Rules.from_prop_rules(rules)
    size, limit, soft = v2.account_size, v2.loss_limit, v2.soft_daily_limit
    bal = size if balance is None else float(balance)
    flr = (size - limit) if floor is None else float(floor)
    cushion = max(0.0, bal - flr)

    if stage == "evaluation":
        base = lv2.eval_risk(cushion, limit, v2.micro_risk)
    else:
        base = lv2.funded_risk(cushion, limit, v2.micro_risk)
    policy_risk = base * STAGE_RISK_MULT[stage] * p["risk_scale"]

    clamps: list[str] = []
    risk = policy_risk
    daily_left = max(0.0, soft + min(0.0, daily_pnl))
    cap_daily = HARD_MAX_SHARE_OF_REMAINING_DAILY * daily_left
    if risk > cap_daily:
        clamps.append(f"Capped at {HARD_MAX_SHARE_OF_REMAINING_DAILY:.0%} of the ${daily_left:,.0f} daily-loss allowance left today (hard limit).")
        risk = cap_daily
    cap_cushion = HARD_MAX_SHARE_OF_CUSHION * cushion
    if risk > cap_cushion:
        clamps.append(f"Capped at {HARD_MAX_SHARE_OF_CUSHION:.0%} of the ${cushion:,.0f} drawdown cushion (hard limit).")
        risk = cap_cushion
    risk = max(0.0, risk)

    day_stop = min(lv2.day_stop_dollars(max(risk, 0.0), soft), p["day_stop_frac"] * soft)
    notes = []
    if cushion <= 0:
        notes.append("No drawdown cushion left: no new risk is allowed.")
    if max_contracts:
        notes.append(f"The firm caps position size at {int(max_contracts)} contract(s); no setting can exceed it.")
    return RiskPlan(stage=stage, preset=preset_name(preset_name_), objective=STAGE_INFO[stage]["objective"],
                    risk_per_trade=round(risk, 2), preset_risk_per_trade=round(policy_risk, 2), day_stop=round(day_stop, 2),
                    max_contracts=int(max_contracts) if max_contracts else None, cushion=round(cushion, 2), clamps=clamps,
                    thresholds={k: p[k] for k in ("warn", "reduce", "block")}, notes=notes)


def clamp_to_hard_limits(requested_risk: float, rules, *, balance: float | None = None, floor: float | None = None,
                         daily_pnl: float = 0.0) -> tuple[float, list]:
    """The last word on any risk figure (preset, user override or strategy default): returns (allowed_risk, reasons it was cut)."""
    v2 = rules if isinstance(rules, lv2.V2Rules) else lv2.V2Rules.from_prop_rules(rules)
    bal = v2.account_size if balance is None else float(balance)
    flr = (v2.account_size - v2.loss_limit) if floor is None else float(floor)
    cushion = max(0.0, bal - flr)
    daily_left = max(0.0, v2.soft_daily_limit + min(0.0, daily_pnl))
    allowed = max(0.0, float(requested_risk))
    why = []
    if allowed > HARD_MAX_SHARE_OF_REMAINING_DAILY * daily_left:
        allowed = HARD_MAX_SHARE_OF_REMAINING_DAILY * daily_left
        why.append("daily loss allowance")
    if allowed > HARD_MAX_SHARE_OF_CUSHION * cushion:
        allowed = HARD_MAX_SHARE_OF_CUSHION * cushion
        why.append("drawdown cushion")
    return round(allowed, 2), why


def payout_progress(account: dict, rules, *, balance: float | None = None) -> float | None:
    """Share of the first-payout profit requirement earned so far (1.0 = eligible), or None when it cannot be told.

    Uses the rules' payout threshold (extra profit above the account size) when it is set, else the Rulebook v2 default cumulative
    minimum. Returns None without a balance: it never guesses."""
    if balance is None:
        return None
    v2 = rules if isinstance(rules, lv2.V2Rules) else lv2.V2Rules.from_prop_rules(rules)
    need = float(getattr(rules, "payout_threshold_pct", 0.0) or 0.0) / 100.0 * v2.account_size or v2.payout.cumulative_min
    if need <= 0:
        return None
    return max(0.0, (float(balance) - v2.account_size) / need)


def all_stage_plans(rules, preset_name_: str | None = None, **kw) -> list[dict]:
    """One row per lifecycle stage, for the 'how your risk changes' table on the Deploy page."""
    out = []
    for st in STAGES:
        plan = risk_plan(st, preset_name_, rules, **kw)
        out.append({"stage": st, "label": STAGE_INFO[st]["label"], "objective": STAGE_INFO[st]["objective"],
                    "risk_per_trade": plan.risk_per_trade, "day_stop": plan.day_stop, "clamps": plan.clamps})
    return out
