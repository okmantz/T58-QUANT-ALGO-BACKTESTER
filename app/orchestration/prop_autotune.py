"""Prop-parameter auto-tuning -- Owen's ask: input your prop-firm
parameters and get sensible search/risk settings tuned to them, instead
of always starting from generic defaults regardless of how strict or
loose the actual eval rules are.

This is a HEURISTIC ADVISOR, not a search or optimizer -- it never
touches market data, never runs a backtest, and makes no claim about
what will actually pass. It reads the PropRules numbers already entered
in Prop Rules / Risk & Execution and returns a plain, deterministic
suggestion for: per-trade risk, Loop Mode's target %, how many rounds to
tolerate before widening/searching deeper, and which strategy-family
canonical groups (see app.strategy.family_taxonomy) to start with. Every
suggestion is a REASONABLE STARTING POINT for a human to review and
override, not an authoritative answer -- there is no way to know what
will actually pass a given prop firm's eval without running the search.

The three formulas below are all simple, monotonic, and bounded on
purpose: each one moves in the obvious direction as its input rule gets
stricter or looser, and never leaves this app's own sane ranges, rather
than chasing a "more sophisticated" model that would be harder to reason
about and no more trustworthy (this is a starting point, not a fitted
model -- there is no training data for "which suggestion led to a pass").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.prop.simulator import PropRules


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class PropAutotuneSuggestion:
    risk_value_pct: float
    target_eval_pass_pct: float
    stall_rounds_before_widen: int
    preferred_family_groups: list          # list[str] -- canonical groups, see app.strategy.family_taxonomy
    preferred_families: list               # list[str] -- actual FAMILIES keys in those groups
    tightness_label: str                   # "tight" | "moderate" | "loose" -- for display only
    rationale: list = field(default_factory=list)  # short human-readable reasons, for display

    def to_dict(self) -> dict:
        return {
            "risk_value_pct": self.risk_value_pct,
            "target_eval_pass_pct": self.target_eval_pass_pct,
            "stall_rounds_before_widen": self.stall_rounds_before_widen,
            "preferred_family_groups": self.preferred_family_groups,
            "preferred_families": self.preferred_families,
            "tightness_label": self.tightness_label,
            "rationale": self.rationale,
        }


def _risk_value_pct(prop_rules: PropRules) -> float:
    """Risk per trade as roughly 1/5 of the daily loss limit -- lets a
    strategy absorb about 5 losing trades in a row before the daily-loss
    circuit breaker would trip, a conventional risk-of-ruin buffer, not a
    formula unique to any one prop firm. At this app's own defaults
    (5% daily loss limit) this lands on 1.0% -- the app's own existing
    RiskConfig default -- by construction, not coincidence: that default
    was itself picked as a sane number for a 5% daily loss limit.
    Clamped to [0.1%, 2.0%] so an unusually loose or tight daily limit
    never produces an unreasonable per-trade risk suggestion."""
    return round(_clamp(prop_rules.daily_loss_limit_pct / 5.0, 0.1, 2.0), 2)


def _tightness_score(prop_rules: PropRules) -> float:
    """0.0 (loose) to 1.0 (tight) -- blends how much profit target is
    demanded relative to how much drawdown room is allowed. A firm asking
    for a lot of profit with very little drawdown room to get there is a
    tight eval; a modest profit target with generous drawdown room is
    loose. `ratio` of 0.5 (e.g. 5% target / 10% drawdown, or 8%/16%) is
    treated as the "moderate" midpoint."""
    ratio = prop_rules.evaluation_profit_target_pct / max(prop_rules.max_drawdown_pct, 0.1)
    return _clamp(ratio / 1.0, 0.0, 1.0)


def _target_eval_pass_pct(tightness: float) -> float:
    """Loop Mode's own target % -- how good a candidate's eval-pass
    probability must be before the loop declares a winner and stops. A
    tighter eval (per _tightness_score) makes a genuinely high pass
    probability rarer to find, so the SUGGESTED target is lower for a
    tight eval (accept a more modest edge as "done searching") and higher
    for a loose one (worth continuing to search for something better).
    Bounded to [35, 75], a range centered close to this app's own Loop
    Mode default of 60."""
    return round(_clamp(75.0 - tightness * 40.0, 35.0, 75.0), 1)


def _stall_rounds_before_widen(tightness: float) -> int:
    """A tight eval is a harder search -- give it more rounds at the
    starting scope before concluding it's stalled and widening/searching
    deeper, rather than bailing to a broader (slower per-round) search
    prematurely. Bounded to [1, 4]."""
    return int(round(_clamp(1.0 + tightness * 3.0, 1.0, 4.0)))


def _preferred_family_groups(tightness: float) -> list[str]:
    """Canonical family groups (see app.strategy.family_taxonomy) to
    START a search with -- not an exclusion of anything else, just a
    starting_family suggestion for the loop config. Tighter evals favor
    lower-variance mechanisms (mean reversion, pullbacks, VWAP reversion)
    that tend to produce a smoother equity curve; looser evals can afford
    the bigger, choppier swings trend-following/breakout/momentum
    mechanisms often carry, in exchange for larger moves."""
    if tightness >= 0.66:
        return ["mean_reversion", "pullback", "vwap"]
    if tightness <= 0.33:
        return ["trend_following", "breakout", "momentum"]
    return ["pullback", "breakout", "vwap"]


def _families_in_groups(groups: list) -> list:
    from app.strategy.family_taxonomy import _SKELETON_TO_GROUP
    group_set = set(groups)
    # Preserve FAMILIES' own dict order (insertion order, i.e. the order
    # each expansion round registered them) rather than dict-comprehension
    # order across group_set, so results are stable and match the app's
    # own family listing order elsewhere.
    return [name for name, group in _SKELETON_TO_GROUP.items() if group in group_set]


def suggest_from_prop_rules(prop_rules: PropRules) -> PropAutotuneSuggestion:
    """The single entry point -- everything else in this module is a
    private helper. Deterministic: the same PropRules always produces the
    same suggestion (no randomness, no data dependence)."""
    tightness = _tightness_score(prop_rules)
    label = "tight" if tightness >= 0.66 else ("loose" if tightness <= 0.33 else "moderate")
    groups = _preferred_family_groups(tightness)
    rationale = [
        f"Daily loss limit {prop_rules.daily_loss_limit_pct:g}% -> suggested risk per trade "
        f"{_risk_value_pct(prop_rules):g}% (roughly 1/5, so ~5 losing trades before the daily "
        "circuit breaker would trip).",
        f"Profit target {prop_rules.evaluation_profit_target_pct:g}% vs. max drawdown "
        f"{prop_rules.max_drawdown_pct:g}% -> this eval reads as {label} "
        f"(tightness {tightness:.2f} on a 0=loose/1=tight scale).",
        f"Starting family groups for a {label} eval: {', '.join(groups)}.",
    ]
    return PropAutotuneSuggestion(
        risk_value_pct=_risk_value_pct(prop_rules),
        target_eval_pass_pct=_target_eval_pass_pct(tightness),
        stall_rounds_before_widen=_stall_rounds_before_widen(tightness),
        preferred_family_groups=groups,
        preferred_families=_families_in_groups(groups),
        tightness_label=label,
        rationale=rationale,
    )
