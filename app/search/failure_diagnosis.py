"""
"Why Did This Strategy Fail?" engine.

Owen's ask: when Forge Strategy (app.orchestration.forge) rejects a
candidate that got far enough to be expensive -- it survived the cheap
Stage 1/2 filters and reached Stage 3 validation, CPCV/regime testing, or
later -- a bare "FAILED" tells the generator nothing it can act on. This
module turns everything that stage's validation already computed
(statistics, Monte Carlo summary, parameter-neighborhood robustness,
walk-forward efficiency, regime-test buckets, rolling-evaluation window
outcomes) into one small, structured verdict:

    Strategy #4832
    FAILED prop validation
    Primary failure:   daily loss
    Secondary failure: excessive losing streak
    Strength:          excellent target achievement
    Weakness:          losses cluster during high-volatility periods
    Potential mutation: tighten per-trade risk or add a volatility filter
    Related successful family: Failed Breakout Reversal

so a person (or a future generation of Forge Strategy) reading a wall of
rejections can act on a pattern ("this family keeps reaching the profit
target but violating daily loss -- explore tighter risk / a volatility
filter") instead of re-discovering it by hand.

This module is deliberately a pure function of already-computed data --
it never runs a new backtest, Monte Carlo, or simulation of its own. Every
input here is something app.search.batch_runner's Stage 3, or Forge
Strategy's own later stages, already produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Canonical, human-readable labels for app.prop.simulator's failure_reason /
# app.prop.rolling_evaluation's failure_category strings -- both modules use
# their own short machine-friendly tokens; this is the one place they're
# translated into the phrasing a person reads in a diagnosis.
_FAILURE_LABELS: dict[str, str] = {
    "daily_loss_limit": "daily loss",
    "max_drawdown": "max drawdown",
    "max_drawdown (trailing)": "max drawdown (trailing)",
    "max_drawdown (static)": "max drawdown (static)",
    "target_not_reached": "target rarely reached",
    "too_many_days": "too slow to reach the target",
    "zero_trades": "no trades generated",
}

# Primary-failure token -> a concrete, actionable mutation suggestion. Kept
# short and structural (a search-space knob to try, not a fully-specified
# fix) since the point is to point the NEXT generation of hypotheses in a
# promising direction, not to hand-tune this one candidate.
_MUTATION_SUGGESTIONS: dict[str, str] = {
    "daily loss": "tighten per-trade risk sizing or add a volatility filter so single days can't run away",
    "max drawdown": "add an equity-curve risk throttle (reduce size after a losing stretch) or widen the stop",
    "max drawdown (trailing)": "add an equity-curve risk throttle (reduce size after a losing stretch) or widen the stop",
    "max drawdown (static)": "add an equity-curve risk throttle (reduce size after a losing stretch) or widen the stop",
    "target rarely reached": "widen the profit target or add a trend/momentum confirmation to let winners run further",
    "too slow to reach the target": "tighten entry criteria to fewer, higher-quality signals, or add a confirmation filter",
    "no trades generated": "loosen the entry condition or check the condition against this instrument's actual data range",
    "excessive losing streak": "add a losing-streak cooldown, a session filter, or a volatility-regime gate",
    "parameter overfitting": "reduce the number of tunable parameters, or prefer coarser grid steps for this family",
    "walk-forward instability": "add a volatility or session filter -- the edge may be real but regime-specific",
    "regime-dependent": "add an explicit volatility-regime filter so the strategy only trades its favorable regime",
    "backtest-overfit vs. peers": "this family's whole neighborhood may be curve-fit -- try a materially different parameter region",
}


@dataclass
class CandidateDiagnosis:
    candidate_id: str
    family: str
    verdict: str                                  # e.g. "FAILED prop validation", "FAILED CPCV/regime gate"
    primary_failure: str
    secondary_failure: str | None = None
    strength: str | None = None
    weakness: str | None = None
    suggested_mutation: str | None = None
    related_successful_family: str | None = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def render(self) -> str:
        lines = [f"Strategy {self.candidate_id}", self.verdict, ""]
        lines.append(f"Primary failure: {self.primary_failure}")
        if self.secondary_failure:
            lines.append(f"Secondary failure: {self.secondary_failure}")
        if self.strength:
            lines.append(f"Strength: {self.strength}")
        if self.weakness:
            lines.append(f"Weakness: {self.weakness}")
        if self.suggested_mutation:
            lines.append(f"Potential mutation: {self.suggested_mutation}")
        if self.related_successful_family:
            lines.append(f"Related successful family: {self.related_successful_family}")
        return "\n".join(lines)


def _label(token: str | None) -> str | None:
    if not token:
        return None
    return _FAILURE_LABELS.get(token, token.replace("_", " "))


def _failure_from_rolling(rolling: dict | None) -> tuple[str | None, str | None]:
    """Primary/secondary failure from a RollingEvaluationResult dict's
    `failure_breakdown` -- the single richest signal available (real
    historical windows, not resampled), so this is checked first."""
    if not rolling:
        return None, None
    breakdown = rolling.get("failure_breakdown") or {}
    if not breakdown:
        return None, None
    ranked = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)
    primary = _label(ranked[0][0])
    secondary = _label(ranked[1][0]) if len(ranked) > 1 else None
    return primary, secondary


def _failure_from_stats(statistics: dict | None) -> str | None:
    """Fallback: the single historical run's own failure_reason, if the
    caller attached one (app.prop.simulator.AccountSimResult.failure_reason,
    commonly copied onto a candidate's stats/prop_summary dict)."""
    if not statistics:
        return None
    return _label(statistics.get("failure_reason"))


def _streak_is_excessive(statistics: dict | None) -> bool:
    if not statistics:
        return False
    n_trades = int(statistics.get("total_trades", 0) or 0)
    streak = int(statistics.get("max_losing_streak", 0) or 0)
    return n_trades > 0 and (streak / n_trades) > 0.15


def _identify_strength(statistics: dict | None, mc_summary: dict | None) -> str | None:
    statistics = statistics or {}
    mc_summary = mc_summary or {}
    win_rate = statistics.get("win_rate")
    profit_factor = statistics.get("profit_factor")
    reach_target_pct = mc_summary.get("evaluation_pass_probability")
    if reach_target_pct is not None and reach_target_pct >= 55 and mc_summary.get("first_payout_probability", 0) < reach_target_pct * 0.6:
        return "excellent target achievement -- reaches the profit target reliably, but rarely converts that into a funded payout"
    if isinstance(win_rate, (int, float)) and win_rate >= 60:
        return f"high win rate ({win_rate:.0f}%)"
    if isinstance(profit_factor, (int, float)) and profit_factor >= 1.8:
        return f"strong per-trade edge (profit factor {profit_factor:.2f})"
    return None


def _identify_weakness(
    statistics: dict | None, robustness: dict | None, regime: dict | None,
) -> str | None:
    statistics = statistics or {}
    if _streak_is_excessive(statistics):
        n_trades = int(statistics.get("total_trades", 0) or 0)
        streak = int(statistics.get("max_losing_streak", 0) or 0)
        return f"prone to long losing streaks ({streak} of {n_trades} trades)"
    if regime and not regime.get("is_regime_stable", True):
        buckets = regime.get("buckets") or []
        losing = [b for b in buckets if not b.get("is_profitable", True)]
        if losing:
            names = ", ".join(b.get("label", "?") for b in losing)
            return f"losses cluster in the {names} regime(s) -- not profitable across all market conditions"
    if robustness is not None and not robustness.get("is_stable", True):
        return "results are sensitive to small parameter changes -- may be fit to this specific configuration"
    return None


def _primary_from_gates(
    robustness: dict | None, walk_forward: dict | None, regime: dict | None,
    cpcv: dict | None,
) -> str | None:
    """Structural failure reasons that aren't about a specific prop-rule
    breach -- used when neither rolling evaluation nor a historical
    failure_reason is available (e.g. the candidate died at the CPCV/
    regime gate, before any prop simulation ran)."""
    if cpcv is not None and not cpcv.get("is_robust", True):
        return "backtest-overfit vs. peers"
    if walk_forward is not None and not walk_forward.get("is_stable", True):
        return "walk-forward instability"
    if regime is not None and not regime.get("is_regime_stable", True):
        return "regime-dependent"
    if robustness is not None and not robustness.get("is_stable", True):
        return "parameter overfitting"
    return None


def diagnose_candidate(
    candidate_id: str,
    family: str,
    verdict: str,
    statistics: dict | None = None,
    mc_summary: dict | None = None,
    robustness: dict | None = None,
    walk_forward: dict | None = None,
    regime: dict | None = None,
    rolling: dict | None = None,
    cpcv: dict | None = None,
    family_performance: dict[str, float] | None = None,
) -> CandidateDiagnosis:
    """Builds one structured diagnosis from whatever this candidate's
    stage(s) already computed. Every argument is optional and independent
    -- pass whichever of these this candidate actually has; the function
    degrades gracefully (falls through to the next-richest signal) rather
    than requiring the full validation stack.

    family_performance: {family_name: best_score_seen_this_run}, used only
    to name a `related_successful_family` different from this candidate's
    own family -- omit it (or leave this candidate's family as the only
    entry) to get None back, rather than a meaningless self-reference.
    """
    primary, secondary = _failure_from_rolling(rolling)
    if primary is None:
        primary = _failure_from_stats(statistics)
    if primary is None:
        primary = _primary_from_gates(robustness, walk_forward, regime, cpcv)
    if primary is None:
        primary = "failed validation thresholds"

    if secondary is None and _streak_is_excessive(statistics):
        secondary = "excessive losing streak"
    if secondary is None:
        structural = _primary_from_gates(robustness, walk_forward, regime, cpcv)
        if structural and structural != primary:
            secondary = structural

    strength = _identify_strength(statistics, mc_summary)
    weakness = _identify_weakness(statistics, robustness, regime)

    suggested_mutation = _MUTATION_SUGGESTIONS.get(primary)
    if suggested_mutation is None and secondary:
        suggested_mutation = _MUTATION_SUGGESTIONS.get(secondary)

    related_successful_family = None
    if family_performance:
        others = {f: s for f, s in family_performance.items() if f and f != family}
        if others:
            best_family = max(others, key=others.get)
            if others[best_family] > 0:
                related_successful_family = best_family

    return CandidateDiagnosis(
        candidate_id=candidate_id, family=family, verdict=verdict,
        primary_failure=primary, secondary_failure=secondary,
        strength=strength, weakness=weakness,
        suggested_mutation=suggested_mutation,
        related_successful_family=related_successful_family,
    )
