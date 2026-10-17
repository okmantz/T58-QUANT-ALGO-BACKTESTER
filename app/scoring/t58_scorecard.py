"""
T58 Strategy Scorecard.

Implements the T58 Quant Trading Masterclass material's Part XI scorecard
verbatim: a single 0-100 number built from already-computed, already-
trusted signals this app produces elsewhere -- never a new metric
invented here, only a documented way of weighting and combining what
already exists:

    Metric                      Weight   Source
    Pass probability              25     app.monte_carlo.engine.MonteCarloResult.per_attempt_pass_ci95
                                          LOWER bound (Wilson 95% on per-attempt, single-account odds --
                                          the chain-level pass_probability_ci95 is inflated by mechanical
                                          rebuys and is only a fallback for results built before the
                                          per-attempt fields existed). Gates on the pessimistic end of
                                          MC noise, not the point estimate
    First payout probability      20     ...per_attempt_paid_ci95 lower bound: P(pass AND get paid) per
                                          purchased attempt -- the PRODUCT of pass x payout, which is
                                          what the README question ("pass the evaluation and reach my
                                          first payout") actually asks. (Until 2026-10-15 this slot used
                                          the CONDITIONAL rate -- payouts per FUNDED account -- which
                                          flatters a strategy that rarely passes but pays when it does.
                                          The conditional rate is still reported, and is the fallback
                                          for results built before the paid-product fields existed)
    Risk of ruin                  20     100 minus MonteCarloResult.risk_of_ruin_pct (inverted so that
                                          higher is better). Since 2026-10-15 that percentage is
                                          TIME-BOUNDED: share of purchased accounts that breach within
                                          the first ruin_horizon_days (default 90) calendar days, so a
                                          longer backtest no longer raises it
    Walk-forward stability        15     app.search.robustness.WalkForwardResult.walk_forward_efficiency
                                          (or CPCV, if that's the run's chosen primary generalization
                                          test -- see app.orchestration.full_pipeline's
                                          primary_robustness_method config; whichever one is primary
                                          for a given run fills THIS slot)
    Monte Carlo robustness        10     seed-to-seed dispersion of the per-attempt pass rate (see
                                          _mc_robustness_score). REPLACED 2026-10-15: the old
                                          "25-75 percentile spread / median return" ratio was
                                          degenerate -- payouts reset the balance, so the median return
                                          sits near 0 and the ratio exploded (or the component was
                                          dropped at exactly 0, which RAISED the score). Missing when
                                          no dispersion measurement exists -- never derived from the
                                          median-return ratio
    Parameter stability            5     app.validation.parameter_robustness.ParameterRobustnessResult
                                          (or the cheaper app.search.robustness.RobustnessResult, if
                                          that's the one already computed -- both are 0-100-like already)
    Expectancy                     5     app.backtest.statistics.BacktestStatistics.average_r, rescaled
                                          2026-10-15 so a realistic good edge scores 50+ (0R -> 0,
                                          0.3R -> 50, 0.6R or better -> 100)
    Drawdown                      10     app.backtest.statistics.BacktestStatistics.max_drawdown_pct
    Parsimony                      5     app.scoring.parsimony.ParsimonyResult.score -- reward for
                                          reaching the same result with fewer unnecessary degrees of
                                          freedom (see that module). A modest tiebreaker weight, not a
                                          dominant factor -- pipeline reorg plan section 24.
    CPCV / PBO (supporting)        5     app.validation.cpcv.CPCVResult -- only populated when CPCV was
                                          run as a SECOND, supporting diagnostic alongside a different
                                          primary generalization test (walk-forward), never when CPCV
                                          IS the primary test (that would double-count the same
                                          evidence in two components -- see section 40/41 of the
                                          pipeline reorg plan on avoiding exactly this).
    (T58 conformance -- informational only, NOT scored since 2026-10-15; see below)
                                         t58_conformance_score() below -- (P2-9/M8, 2026-10-03):
                                          fraction of the strategy's entry conditions that reference
                                          Market Structure / Liquidity / Supply-Demand primitives
                                          (resolved via app.strategy.dna tags) plus a premium/
                                          discount alignment sub-check (concept from
                                          app.ai.t58_strategy_engine). See "WHAT THE SCORE MEASURES"
                                          below -- this component deliberately measures something
                                          DIFFERENT from every row above it.

    The weights sum to 120 (25+20+20+15+10+5+5+10+5+5), NOT 90 or 100: a component's
    share of the score is weight/120, e.g. expectancy is 5/120 = 4.2% of the total. (Older
    text called expectancy "5/90ths" -- that predates the parsimony, CPCV-supporting and
    conformance components and is wrong.) The score itself is
    sum(weight x value) / sum(weight of the components PRESENT).

WHAT THE SCORE MEASURES -- READ BEFORE REINTERPRETING IT

The ten pre-existing components above (pass probability, first payout
probability, risk of ruin, walk-forward stability, Monte Carlo robustness,
parameter stability, expectancy, drawdown, parsimony, CPCV-supporting)
measure PROP-SURVIVAL: can this strategy plausibly pass a prop-firm
evaluation and get paid without breaching the firm's rules. They say
nothing about whether the strategy trades the way Owen's T58 framework
(Market Structure, Liquidity, Supply & Demand) says to trade -- a pure
RSI-mean-reversion bot can score Elite here without a single structural
idea in it.

`t58_conformance` (informational badge, weight 0 -- removed from the score on 2026-10-15) measures FRAMEWORK CONFORMANCE instead:
how much of the strategy's entry logic is expressed in Market Structure
/ Liquidity / Supply-Demand primitives, and whether its long/short
entries align with premium/discount location (longs want discount,
shorts want premium -- the dealing-range rule from
app.ai.t58_strategy_engine). A high total score with low conformance is
a working strategy that does NOT trade the T58 way; a high conformance
with a low total score is a T58-faithful strategy the prop-survival
evidence rejects. The two readings complement each other; neither
redefines the other, and the old score's meaning is unchanged -- when
`t58_conformance` is supplied it is only echoed on the result as a badge; it never
changes the score, the weight coverage or the READY verdict.

Tiers (recalibrated 2026-10-17 so the score matches the gates; the Masterclass cuts of 92/85/75/65 made READY
mean "excellent on everything"):

    85+  Elite       75+  Strong       65+  Promising      <65  Reject

READY = every hard gate passed AND score >= 75 (Strong or Elite). Elite is a label, not the READY bar.

Several of these inputs (Monte Carlo robustness, parameter stability,
expectancy, drawdown, parsimony, CPCV-as-supporting) aren't already
expressed on a 0-100 "higher is better" scale in this app, so this module
documents exactly how each is mapped onto one -- see the docstring on
each _*_score helper. Every mapping is a heuristic; none of them invents
a new backtest metric, they only rescale ones that already exist.
`score_from_results()` does that mapping for you from the objects this
app's own pipeline already produces; `T58ScorecardInputs` is there for a
caller that wants to supply the 0-100 numbers itself (e.g. from records
already persisted in app.search.results_db, where a fresh Monte Carlo/
robustness object may not be sitting in memory anymore).

IMPORTANT -- missing-aware, but missing evidence can never lift a verdict:
compute_t58_score() re-normalizes over whichever components are actually
present (see its own docstring), so a strategy that hasn't been walk-forward
tested yet, or whose parsimony couldn't be counted, is not scored as if it
had FAILED that check. Re-normalizing alone, though, lets a strategy that
omits its weakest components score higher than it should (omitting the five
weakest moved one audit example from 70.9 to 83.2). So every result also
carries `weight_coverage` (the share of the 120 total weight that was
actually measured) and `missing_core` (any of pass / payout / ruin absent),
and app.orchestration.full_pipeline._make_verdict_core caps READY at
MARGINAL when coverage is too thin or a core survival component is missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_WEIGHTS: dict[str, float] = {
    "pass_probability": 25.0,
    "first_payout_probability": 20.0,
    "risk_of_ruin": -20.0,
    "walk_forward_stability": 15.0,
    "monte_carlo_robustness": 10.0,
    "parameter_stability": 5.0,
    "expectancy": 5.0,
    "drawdown": 10.0,
    "parsimony": 5.0,
    "cpcv_supporting": 5.0,
    # t58_conformance is NO LONGER scored (2026-10-15 audit): it measures how T58-styled the entry logic is,
    # not whether the strategy survives a prop evaluation. It is reported beside the score as an
    # informational badge (T58ScorecardResult.t58_conformance) and never enters the score, the weight
    # coverage, or the READY verdict.
}

# Tier meaning (2026-10-17 recalibration): READY = every hard gate passed AND score >= READY_MIN_SCORE (75, the
# "Strong" tier -- roughly 85% pass LB / 65% paid LB / 8% ruin with good secondary components). "Elite" (>= 85) is a
# label for excellence on everything, NOT the READY bar. "Promising" (65-75) is MARGINAL; below 65 is Reject.
READY_MIN_SCORE = 75.0
ELITE_MIN_SCORE = 85.0
_TIERS: list[tuple[float, str]] = [
    (85.0, "Elite"), (75.0, "Strong"), (65.0, "Promising"),
]


# The three survival components every READY verdict must have measured, and the minimum
# share of total scorecard weight that must have been measured for READY. Used by
# app.orchestration.full_pipeline._make_verdict_core.
_CORE_COMPONENTS = ("pass_probability", "first_payout_probability", "risk_of_ruin")
MIN_WEIGHT_COVERAGE_FOR_READY = 0.70


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


@dataclass
class T58ScorecardInputs:
    """All components, already expressed 0-100 ('higher is always
    better', including risk_of_ruin and drawdown -- see score_from_results
    for how those two get inverted onto this scale before they arrive
    here). None for a component means 'not computed' -- the final score
    re-normalizes over only the weights actually present rather than
    silently treating a missing check as a zero, so a strategy that
    simply hasn't been walk-forward tested yet isn't penalized as if it
    had FAILED walk-forward testing."""
    pass_probability: float | None = None
    first_payout_probability: float | None = None
    risk_of_ruin: float | None = None
    walk_forward_stability: float | None = None
    monte_carlo_robustness: float | None = None
    parameter_stability: float | None = None
    expectancy: float | None = None
    drawdown: float | None = None
    parsimony: float | None = None
    cpcv_supporting: float | None = None
    # NEW (P2-9/M8): 0-100 T58 framework-conformance value, e.g. from
    # t58_conformance_score() below. None (the default) keeps the
    # pre-existing prop-survival-only score byte-identical.
    t58_conformance: float | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class T58ScorecardResult:
    score: float                 # 0-100
    tier: str                    # "Elite" | "Strong" | "Promising" | "Reject"
    components: dict             # component name -> {"value": float|None, "weight": float, "contribution": float|None}
    n_components_used: int
    n_components_total: int
    notes: list = field(default_factory=list)
    # 2026-10-15: share (0-1) of the total scorecard weight that was actually measured, and which
    # of the three core survival components (pass / payout / ruin) were absent. Missing evidence
    # must cap a verdict, not lift a score -- see the module docstring.
    weight_coverage: float = 1.0
    missing_core: list = field(default_factory=list)
    # Informational only (0-100 or None): how T58-styled the entry logic is. NOT part of the score.
    t58_conformance: float | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def render_line(self) -> str:
        used = f"{self.n_components_used}/{self.n_components_total} checks available"
        return f"T58 Score: {self.score:.1f}/100 -- {self.tier} ({used})"


def tier_for_score(score: float) -> str:
    for threshold, name in _TIERS:
        if score >= threshold:
            return name
    return "Reject"


def compute_t58_score(inputs: T58ScorecardInputs) -> T58ScorecardResult:
    """Weighted sum over whatever components are present, RE-NORMALIZED
    to the weight actually available (see T58ScorecardInputs docstring
    for why a missing check isn't scored as a failing one). Every component
    present and maxed would score exactly 100; every component present
    and floored (0 everywhere, risk_of_ruin term also at its floor of 0)
    would score exactly 0. Positive and negative weights are handled
    the same way: a component's "value" is always 0-100 'higher is
    better' by this point (risk_of_ruin=0 means NO ruin risk, i.e. the
    best possible outcome), so `weight * value` is added for every
    component including the nominally-negative-weighted risk_of_ruin --
    the sign only matters for how score_from_results() maps the RAW
    risk-of-ruin percentage onto this inverted scale."""
    values = inputs.to_dict()
    components: dict = {}
    total_weight_available = 0.0
    weighted_sum = 0.0
    notes: list[str] = []
    _c = values.get("t58_conformance")
    _conf = None if _c is None else _clip(float(_c))

    for name, weight in _WEIGHTS.items():
        val = values.get(name)
        abs_weight = abs(weight)
        if val is None:
            components[name] = {"value": None, "weight": weight, "contribution": None}
            continue
        val_clipped = _clip(float(val))
        contribution = abs_weight * val_clipped
        components[name] = {"value": val_clipped, "weight": weight, "contribution": contribution}
        weighted_sum += contribution
        total_weight_available += abs_weight * 100.0

    n_total = len(_WEIGHTS)
    n_used = sum(1 for c in components.values() if c["value"] is not None)
    total_weight_all = sum(abs(w) for w in _WEIGHTS.values()) * 100.0
    coverage = (total_weight_available / total_weight_all) if total_weight_all > 0 else 0.0
    missing_core = [n for n in _CORE_COMPONENTS if components[n]["value"] is None]
    if total_weight_available <= 0:
        notes.append("No scorecard components were available -- score is 0 by default, not a real evaluation.")
        return T58ScorecardResult(score=0.0, tier="Reject", components=components,
                                   n_components_used=0, n_components_total=n_total, notes=notes,
                                   weight_coverage=0.0, missing_core=missing_core,
                                   t58_conformance=_conf)

    score = _clip(weighted_sum / total_weight_available * 100.0)
    if n_used < n_total:
        missing = [name for name, c in components.items() if c["value"] is None]
        notes.append(
            f"{n_total - n_used} of {n_total} checks weren't available ({', '.join(missing)}) -- "
            "score is re-normalized over what WAS checked, not penalized for what wasn't; "
            f"{coverage * 100:.0f}% of the scorecard weight was measured, and a verdict cannot reach "
            "READY on thin coverage."
        )
    return T58ScorecardResult(
        score=score, tier=tier_for_score(score), components=components,
        n_components_used=n_used, n_components_total=n_total, notes=notes,
        weight_coverage=coverage, missing_core=missing_core, t58_conformance=_conf,
    )


# ---------------------------------------------------------------------------
# Mapping real pipeline objects onto the 0-100 'higher is better' scale
# ---------------------------------------------------------------------------

def _walk_forward_score(walk_forward_efficiency: float) -> float:
    """walk_forward_efficiency is mean_test_metric/mean_train_metric,
    already clipped to [-5, 5] by app.search.robustness -- 1.0 means
    the test period did exactly as well as train (ideal), > 1 is even
    better (rare), < 1 means some degradation out of sample. Maps
    linearly: 1.0+ -> 100, 0.0 (test performance vanished) -> 0,
    negative (test period LOST when train WON) -> 0 floor."""
    return _clip(walk_forward_efficiency * 100.0)


# Seed-to-seed swing (in percentage points, standard deviation of the per-attempt pass rate across
# independent Monte Carlo seeds) at which the robustness component reaches 0. A swing of 0 scores 100;
# the pipeline's own seed-instability cap is 10 points, which scores 50 here.
_MC_DISPERSION_ZERO_AT_PTS = 20.0


def _mc_robustness_score(mc_result=None, dispersion_pts: float | None = None) -> float | None:
    """'Monte Carlo robustness' = how STABLE the headline pass rate is when the Monte Carlo is
    re-run with different random seeds: 0 points of seed-to-seed standard deviation scores 100,
    `_MC_DISPERSION_ZERO_AT_PTS` (20) points or more scores 0, linear between.

    REPLACED 2026-10-15. The previous definition divided the 25th-75th percentile return spread
    by the MEDIAN return. In a prop simulation payouts reset the account balance, so the median
    return of a healthy strategy sits near zero and that ratio exploded: the component scored 0
    for a strong edge (70% win rate scored LOWER than 58% in the audit's edge ladder) and when
    the median was exactly 0 the component was silently DROPPED, which re-normalized the score
    upward. A stronger edge could therefore score lower, and missing data could score higher.

    Returns None (missing -- UNPROVEN, not 0) when no dispersion measurement was supplied. It
    is never derived from the median-return ratio any more. `mc_result` is accepted only so
    existing callers keep working; a result object may carry an optional
    `seed_dispersion_pts` attribute, which is used when `dispersion_pts` is not given."""
    d = dispersion_pts
    if d is None:
        d = getattr(mc_result, "seed_dispersion_pts", None)
    if d is None:
        return None
    try:
        d = float(d)
    except (TypeError, ValueError):
        return None
    if d != d or d < 0:  # NaN or nonsensical
        return None
    return _clip(100.0 * (1.0 - d / _MC_DISPERSION_ZERO_AT_PTS))


# Average R per trade at which the expectancy component reaches 100 (0R scores 0; the midpoint,
# 0.3R, scores 50).
_EXPECTANCY_FULL_SCORE_R = 0.6


def _expectancy_score(average_r: float | None) -> float | None:
    """average_r is the strategy's mean R-multiple per trade (see
    app.backtest.statistics). 0R or worse scores 0; 0.6R or better scores 100, linear between,
    so a realistic good edge (0.3R) scores 50.

    RESCALED 2026-10-15: it used to be 0R -> 0 and 2R -> 100, which scored a genuinely good 0.3R
    edge at 15/100 and a 0.44R edge (the Masterclass Lesson 1 example) at 22/100. Combined with
    the other components that put the maximum reachable score at ~84.6 even with everything else
    perfect -- just under the 85 'Elite' cut. This component is 5/120 = 4.2% of the total score."""
    if average_r is None:
        return None
    return _clip(average_r / _EXPECTANCY_FULL_SCORE_R * 100.0)


def _drawdown_score(max_drawdown_pct: float | None, prop_max_drawdown_pct: float | None) -> float | None:
    """Inverted and scaled against the account's OWN max-drawdown rule
    (PropRules.max_drawdown_pct) rather than an arbitrary fixed number,
    since 'how much drawdown is acceptable' is defined by the firm's own
    rules, not a universal constant: using exactly the full allowance
    scores 0 (no safety margin left), using none of it scores 100."""
    if max_drawdown_pct is None or not prop_max_drawdown_pct:
        return None
    return _clip(100.0 * (1.0 - abs(max_drawdown_pct) / abs(prop_max_drawdown_pct)))


def _risk_of_ruin_score(risk_of_ruin_pct: float | None) -> float | None:
    """Simple inversion onto the 'higher is better' scale every other
    component uses: 0% risk of ruin -> 100, 100% risk of ruin -> 0."""
    if risk_of_ruin_pct is None:
        return None
    return _clip(100.0 - risk_of_ruin_pct)


def _cpcv_score(cpcv_result) -> float | None:
    """Maps app.validation.cpcv.CPCVResult onto the same kind of
    'generalization efficiency' scale _walk_forward_score uses: how much
    of the in-sample metric survived out-of-sample, averaged across every
    CPCV path (mean_oos_metric / mean_is_metric), clipped to [0, 1] then
    scaled x100. Scores 0 when the in-sample metric was already <= 0 (no
    edge to retain -- this used to score 100, which rewarded strategies that
    lose equally in and out of sample) and 0 when the out-of-sample paths
    lost money."""
    mean_is = getattr(cpcv_result, "mean_is_metric", None)
    mean_oos = getattr(cpcv_result, "mean_oos_metric", None)
    if mean_is is None or mean_oos is None:
        return None
    # 2026-10-15 (audit item 10): a strategy that did not make money out of sample has nothing
    # to "generalize", however its in/out-of-sample metrics compare.
    if getattr(cpcv_result, "oos_profitable", None) is False:
        return 0.0
    if mean_is <= 0:
        # No in-sample edge to retain. This used to score 100 whenever OOS "didn't get worse" --
        # i.e. a strategy that lost equally in and out of sample (profit factors 0.85-0.95 in the
        # audit) scored as perfectly generalizing. Nothing to retain scores 0.
        return 0.0
    return _clip((mean_oos / mean_is) * 100.0)


# ---------------------------------------------------------------------------
# t58_conformance -- NEW component (P2-9 / M8 fix, 2026-10-03).
#
# Every other component on this scorecard measures PROP-SURVIVAL (see
# "WHAT THE SCORE MEASURES" in the module docstring). This one measures
# FRAMEWORK CONFORMANCE: how much of the strategy's entry logic is
# expressed in T58 primitives -- Market Structure, Liquidity, and
# Supply/Demand -- rather than in generic momentum/volatility terms.
#
# Primitive resolution goes through app.strategy.dna's EXISTING entry
# genes (no vocabulary changes in dna.py were needed):
#   - entry.market_structure -- its keyword list already covers market
#     structure ("break of structure", "bos", "choch", "swing high/low",
#     "higher_high", ...) AND supply/demand ("supply zone", "demand
#     zone", "order block").
#   - entry.liquidity -- covers liquidity ("liquidity sweep",
#     "stop hunt", "equal highs/lows", "inducement", ...) and, via the
#     same gene, fair value gaps.
# Together the two tags span all three T58 pillars.
#
# Premium/discount alignment sub-check: a real premium/discount concept
# EXISTS in this codebase -- app.ai.t58_strategy_engine's dealing-range
# LocationContext (zone in {"premium", "discount", "equilibrium"}) plus
# its alignment rule "longs prefer discount, shorts prefer premium"
# (t58_strategy_engine.py, T58Assessment logic). This sub-check mirrors
# that rule at the scorecard level (a documented keyword mapping here,
# NOT an import of the AI layer, keeping app.scoring dependency-free of
# app.ai): each entry side scores 100 when its conditions reference the
# favored zone without the opposing one, 0 when they reference the
# opposing zone without the favored one, and 50 (neutral) otherwise.
# ---------------------------------------------------------------------------

# dna.py tags (as "section.gene") that count as T58 primitives.
_T58_PRIMITIVE_TAGS = ("entry.market_structure", "entry.liquidity")

# Minimal scorecard-local supplement (NOT added to dna.py): dna.py's
# keyword vocabulary predates the swing_bos/swing_choch indicator kinds
# added by the P2-9 fix, so its word-boundary matcher doesn't fire on
# them ("swing_bos" doesn't match the "bos" keyword). Those two kinds are
# thin wrappers around app.quant_lab.market_structure's real
# fractal-swing detectors, so a condition referencing them IS
# referencing a Market Structure primitive. This mapping lives here, in
# the scorecard module, leaving dna.py's shared vocabulary untouched.
_T58_KIND_SUPPLEMENT = ("swing_bos", "swing_choch")

# Premium/discount location vocabulary for the alignment sub-check.
_PREMIUM_KEYWORDS = ("premium", "dealing range", "dealing_range")
_DISCOUNT_KEYWORDS = ("discount",)

# 80/20 split between the two sub-measures: primitive coverage is the
# component's core; premium/discount alignment is a small modifier.
_CONFORMANCE_PRIMITIVE_WEIGHT = 0.8
_CONFORMANCE_PD_WEIGHT = 0.2


def _dna_references_t58_primitives(text: str) -> bool:
    """True when app.strategy.dna tags `text` with a Market Structure /
    Liquidity / Supply-Demand primitive gene (see _T58_PRIMITIVE_TAGS),
    or when it references one of the scorecard-local supplementary kind
    names in _T58_KIND_SUPPLEMENT."""
    # Lazy import keeps app.scoring import-light; dna.py itself only
    # needs the stdlib, so there is no import cycle either way.
    from app.strategy.dna import extract_dna_from_text

    text = text or ""
    if any(kind in text.lower() for kind in _T58_KIND_SUPPLEMENT):
        return True
    dna = extract_dna_from_text(text)
    return any(dna.entry.get(tag.split(".", 1)[1], False) for tag in _T58_PRIMITIVE_TAGS)


def _premium_discount_side_score(text: str, favored: tuple[str, ...], opposed: tuple[str, ...]) -> float:
    """100/0/50 alignment of one entry side's text against the
    premium/discount rule: favored zone mentioned without the opposed
    zone -> 100; opposed without favored -> 0; both/neither -> 50."""
    t = (text or "").lower()
    has_favored = any(k in t for k in favored)
    has_opposed = any(k in t for k in opposed)
    if has_favored and not has_opposed:
        return 100.0
    if has_opposed and not has_favored:
        return 0.0
    return 50.0


def _premium_discount_alignment(long_text: str, short_text: str) -> float:
    """Premium/discount ALIGNMENT sub-check (0-100), mirroring
    app.ai.t58_strategy_engine's rule: longs want discount, shorts want
    premium. A side that never mentions premium/discount location at all
    scores neutral 50 -- not mentioning location isn't misalignment, it
    just isn't evidence of alignment."""
    long_score = _premium_discount_side_score(long_text, _DISCOUNT_KEYWORDS, _PREMIUM_KEYWORDS)
    short_score = _premium_discount_side_score(short_text, _PREMIUM_KEYWORDS, _DISCOUNT_KEYWORDS)
    return (long_score + short_score) / 2.0


def t58_conformance_score(
    strategy_text: str | None = None,
    *,
    strategy_type: str = "manual",
    long_entry_texts: list[str] | None = None,
    short_entry_texts: list[str] | None = None,
) -> float | None:
    """0-100 T58 framework-conformance for one strategy, or None when no
    strategy text was supplied at all (missing-aware -- the scorecard
    re-normalizes without this component rather than scoring it 0).

    Two modes:
      per-condition (preferred) -- pass `long_entry_texts` /
      `short_entry_texts` (one text per Manual entry condition, e.g.
      json.dumps of each condition dict): the primary measure is the
      FRACTION of entry conditions referencing T58 primitives, and the
      premium/discount alignment sub-check is scored per side.
      whole-text (fallback) -- pass `strategy_text` (Python/PineScript/
      MQL5 source, or a Manual config dict dumped to JSON): primitive
      coverage is binary (100 if any primitive gene fires on the whole
      text, else 0) and the alignment sub-check is neutral 50, because
      directional alignment needs per-side conditions to judge.
      `strategy_type` is accepted for symmetry with
      app.strategy.dna.extract_dna; only "manual" enables the
      per-condition path, but the per-condition lists work regardless
      of type when supplied.

    Final: 100 * (0.8 * primitive_fraction + 0.2 * pd_alignment/100)."""
    long_texts = [t for t in (long_entry_texts or []) if t]
    short_texts = [t for t in (short_entry_texts or []) if t]

    if long_texts or short_texts:
        all_texts = long_texts + short_texts
        primitive_fraction = (
            sum(1 for t in all_texts if _dna_references_t58_primitives(t)) / len(all_texts)
        )
        pd_alignment = _premium_discount_alignment(" ".join(long_texts), " ".join(short_texts)) / 100.0
    elif strategy_text:
        primitive_fraction = 1.0 if _dna_references_t58_primitives(strategy_text) else 0.0
        # Whole-text mode can't judge directional alignment -- neutral.
        pd_alignment = 0.5
    else:
        return None

    return _clip(
        100.0
        * (
            _CONFORMANCE_PRIMITIVE_WEIGHT * primitive_fraction
            + _CONFORMANCE_PD_WEIGHT * pd_alignment
        )
    )


def t58_conformance_for_manual_config(config: dict) -> float | None:
    """Convenience: score a Manual Strategy Builder config dict by
    extracting each entry condition's own text (long + short sides) and
    running the per-condition path of t58_conformance_score(). Returns
    None when the config has no entry conditions to judge."""
    import json

    entries = (config or {}).get("entry_conditions", {}) or {}
    long_conds = entries.get("long", []) or []
    short_conds = entries.get("short", []) or []
    long_texts = [json.dumps(c, default=str) for c in long_conds]
    short_texts = [json.dumps(c, default=str) for c in short_conds]
    if not long_texts and not short_texts:
        return None
    return t58_conformance_score(long_entry_texts=long_texts, short_entry_texts=short_texts)


def _nonzero_ci(ci) -> bool:
    """True when a (lo, hi) interval is present and non-zero -- a present
    (0.0, 0.0) interval means "unknown" (a result built before the field
    existed), not a genuine all-fail run."""
    if ci is None:
        return False
    try:
        return not (float(ci[0]) == 0.0 and float(ci[1]) == 0.0)
    except (TypeError, ValueError, IndexError):
        return False


def _preferred_pass_ci(mc_result):
    """v6 (2026-10-04): the CI the acceptance verdict gates on -- the
    PER-ATTEMPT pass interval when present, else the legacy chain-level
    interval, else None (caller falls back to the point estimate).
    Returns the (lo, hi) tuple, never the point estimate."""
    ci = getattr(mc_result, "per_attempt_pass_ci95", None)
    if _nonzero_ci(ci):
        return ci
    ci = getattr(mc_result, "pass_probability_ci95", None)
    if _nonzero_ci(ci):
        return ci
    return None


def _preferred_payout_ci(mc_result):
    """v6 (2026-10-04): payout-leg twin of _preferred_pass_ci -- the
    PER-ATTEMPT first-payout interval when present, else the legacy
    chain-level interval, else None."""
    ci = getattr(mc_result, "per_attempt_payout_ci95", None)
    if _nonzero_ci(ci):
        return ci
    ci = getattr(mc_result, "payout_probability_ci95", None)
    if _nonzero_ci(ci):
        return ci
    return None


def score_from_results(
    mc_result=None,
    walk_forward_result=None,
    robustness_result=None,
    statistics=None,
    prop_max_drawdown_pct: float | None = None,
    parsimony_result=None,
    cpcv_supporting_result=None,
    t58_conformance: float | None = None,
    manual_config: dict | None = None,
    mc_dispersion_pts: float | None = None,
) -> T58ScorecardResult:
    """Convenience entry point: pass whichever of this app's own result
    objects you already have in hand (any/all may be None -- a partial
    scorecard, correctly re-normalized, beats refusing to score at
    all). `robustness_result` accepts either
    app.validation.parameter_robustness.ParameterRobustnessResult
    (reads .parameter_robustness_score directly) or
    app.search.robustness.RobustnessResult (reads .stability_ratio,
    scaled x100 and clipped -- it's a ratio centered near 1.0, not
    already a 0-100 score). `parsimony_result` accepts an
    app.scoring.parsimony.ParsimonyResult. `cpcv_supporting_result`
    accepts an app.validation.cpcv.CPCVResult -- pass this ONLY when
    CPCV ran as a supporting diagnostic alongside a different primary
    generalization test; if CPCV itself is this run's primary test,
    pass its efficiency through `walk_forward_result`-shaped scoring
    instead (see app.orchestration.full_pipeline) so it isn't counted
    twice. `t58_conformance` accepts a precomputed 0-100 framework-
    conformance value (see t58_conformance_score()); `manual_config`
    accepts a Manual Strategy Builder config dict from which the
    per-condition conformance is derived when `t58_conformance` isn't
    given. Omit both and the component is simply absent -- the pre-
    existing prop-survival-only score is unchanged. `mc_dispersion_pts` is the seed-to-seed
    standard deviation (percentage points) of the per-attempt pass rate; it feeds the Monte Carlo
    robustness component, which is missing when it is not supplied."""
    # v5 (2026-10-04): gate on the Wilson 95% LOWER bound, not the point
    # estimate. A point estimate vs a hard acceptance bar flips inside MC
    # noise (69.2% vs 70% is the same measurement), so the verdict only
    # treats the bar as cleared when the PESSIMISTIC end of the sampling
    # noise clears it too.
    # v6 (2026-10-04): gate on the PER-ATTEMPT CI lower bound -- the
    # chain-level pass_probability_ci95 interval is inflated almost by
    # construction once reset_on_breach is on (a long enough rebuy chain
    # eventually clears a low bar), so preferring it would let the
    # verdict answer "did a mechanical rebuyer eventually pass" instead
    # of "would ONE account attempt pass." Falls back to the chain-level
    # interval, and then to the point estimate, for results built before
    # the per-attempt fields existed -- but a present (0.0, 0.0) CI means
    # "unknown", so only a non-zero interval is trusted; a genuine
    # all-fail run reports its lower bound as 0.0 anyway, which the point
    # estimate would give too.
    pass_probability = getattr(mc_result, "evaluation_pass_probability", None)
    first_payout_probability = getattr(mc_result, "first_payout_probability", None)
    risk_of_ruin_pct = getattr(mc_result, "risk_of_ruin_pct", None)
    _pass_ci = _preferred_pass_ci(mc_result)
    if _pass_ci is not None:
        pass_probability = float(_pass_ci[0])
    _payout_ci = _preferred_payout_ci(mc_result)
    if _payout_ci is not None:
        first_payout_probability = float(_payout_ci[0])
    # 2026-10-15: prefer the pass-AND-paid PRODUCT (per purchased attempt) over the conditional
    # payout rate (per funded account) -- the README question is "pass AND get paid". Falls back
    # to the conditional interval/point estimate for results built before the product existed.
    _paid_ci = getattr(mc_result, "per_attempt_paid_ci95", None)
    if _nonzero_ci(_paid_ci):
        first_payout_probability = float(_paid_ci[0])

    walk_forward_stability = None
    if walk_forward_result is not None:
        walk_forward_stability = _walk_forward_score(walk_forward_result.walk_forward_efficiency)
        if getattr(walk_forward_result, "oos_profitable", None) is False:
            walk_forward_stability = 0.0   # lost money out of sample: nothing generalized

    parameter_stability = None
    if robustness_result is not None:
        if hasattr(robustness_result, "parameter_robustness_score"):
            parameter_stability = _clip(robustness_result.parameter_robustness_score)
        elif hasattr(robustness_result, "stability_ratio"):
            parameter_stability = _clip(robustness_result.stability_ratio * 100.0)

    average_r = getattr(statistics, "average_r", None) if statistics is not None else None
    max_drawdown_pct = getattr(statistics, "max_drawdown_pct", None) if statistics is not None else None

    parsimony_score = getattr(parsimony_result, "score", None) if parsimony_result is not None else None
    cpcv_supporting_score = _cpcv_score(cpcv_supporting_result) if cpcv_supporting_result is not None else None

    # NEW (P2-9/M8): T58 framework-conformance. An explicit 0-100 value
    # wins; otherwise derive it per-condition from a Manual config; when
    # neither is supplied the component stays None and the score
    # re-normalizes over the pre-existing prop-survival components alone
    # (old behavior byte-identical).
    if t58_conformance is None and manual_config is not None:
        t58_conformance = t58_conformance_for_manual_config(manual_config)

    inputs = T58ScorecardInputs(
        pass_probability=pass_probability,
        first_payout_probability=first_payout_probability,
        risk_of_ruin=_risk_of_ruin_score(risk_of_ruin_pct),
        walk_forward_stability=walk_forward_stability,
        monte_carlo_robustness=_mc_robustness_score(mc_result, mc_dispersion_pts) if (mc_result is not None or mc_dispersion_pts is not None) else None,
        parameter_stability=parameter_stability,
        expectancy=_expectancy_score(average_r),
        drawdown=_drawdown_score(max_drawdown_pct, prop_max_drawdown_pct),
        parsimony=parsimony_score,
        cpcv_supporting=cpcv_supporting_score,
        t58_conformance=t58_conformance,
    )
    return compute_t58_score(inputs)
