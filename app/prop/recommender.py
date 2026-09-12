"""
Prop-Firm Recommender -- takes a strategy's own trade sequence (from a
backtest that's already run) and scores it against every preset in
app.prop.presets (or a caller-supplied list of PropRules), instead of
the app's existing question ("given ONE firm's rules, what's my pass
probability"). This is the reverse: "given my strategy, which firm's
rules does it actually fit best."

This is deliberately a thin layer: the exact same
app.monte_carlo.engine.run_monte_carlo() used everywhere else in the app
is run once per candidate rule set, so a recommendation is always
directly comparable to (and reproducible from) the plain Payout
Probability / Speed Run / Full Pipeline numbers for that same firm's
rules -- nothing about prop-rule evaluation is reimplemented here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.backtest.execution import Trade
from app.monte_carlo.engine import MonteCarloConfig, MonteCarloResult, run_monte_carlo
from app.prop.presets import PropFirmPreset, list_presets
from app.prop.simulator import PropRules


@dataclass
class FirmRecommendation:
    preset: PropFirmPreset | None      # None when scored against a caller-supplied raw PropRules, not a catalog preset
    label: str                          # display label (preset.label, or a caller-supplied name)
    rules: PropRules
    mc_result: MonteCarloResult
    composite_score: float              # 0-100, see _composite_score

    def to_dict(self) -> dict:
        return {
            "preset_key": self.preset.key if self.preset else None,
            "label": self.label,
            "evaluation_pass_probability": self.mc_result.evaluation_pass_probability,
            "first_payout_probability": self.mc_result.first_payout_probability,
            "multiple_payout_probability": self.mc_result.multiple_payout_probability,
            "risk_of_ruin_pct": self.mc_result.risk_of_ruin_pct,
            "median_days_to_pass": self.mc_result.median_days_to_pass,
            "expected_payout": self.mc_result.expected_payout,
            "composite_score": self.composite_score,
        }


def _composite_score(mc: MonteCarloResult) -> float:
    """A single 0-100 ranking number, blending the three things that
    actually matter for 'which firm should I take this eval with':
    passing at all (50% weight -- if you can't pass, nothing else
    matters), reaching a first payout (30%), and not blowing up on the
    way there (20%, via risk_of_ruin_pct). Deliberately simple and
    documented rather than a fitted model -- same philosophy as
    app.orchestration.prop_autotune's suggestion formulas: monotonic,
    bounded, and easy to second-guess by reading the three inputs
    directly on the results table this score sits next to.
    """
    pass_component = max(0.0, min(100.0, mc.evaluation_pass_probability))
    payout_component = max(0.0, min(100.0, mc.first_payout_probability))
    survival_component = max(0.0, 100.0 - max(0.0, min(100.0, mc.risk_of_ruin_pct)))
    return round(0.5 * pass_component + 0.3 * payout_component + 0.2 * survival_component, 2)


def recommend_prop_firms(
    trades: list[Trade],
    candidates: list[PropFirmPreset] | list[tuple[str, PropRules]] | None = None,
    mc_cfg: MonteCarloConfig | None = None,
    progress_cb=None,
) -> list[FirmRecommendation]:
    """
    trades: the strategy's own closed trades from ONE backtest run (same
        input run_monte_carlo() always takes) -- run the backtest once,
        against whatever account size your OWN sizing used; this
        function only varies the RULES being checked against, not the
        trade sequence itself. A strategy sized very differently at
        $10k vs $200k should really be backtested once per size class
        first; this is a ranking tool over rule shapes, not a position-
        sizing optimizer.
    candidates: defaults to every preset in app.prop.presets.list_presets().
        Can also be a list of (label, PropRules) pairs to score custom /
        not-yet-cataloged firm rules alongside or instead of the catalog.
    mc_cfg: shared Monte Carlo settings applied identically to every
        candidate, so the comparison between firms isn't confounded by
        different simulation counts/methods -- callers wanting a fast
        first pass should just lower n_simulations here rather than
        vary it per candidate.

    Returns recommendations sorted best-first by composite_score.
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    if not trades:
        return []

    cfg = mc_cfg or MonteCarloConfig(n_simulations=2000)
    items = candidates if candidates is not None else list_presets()

    results: list[FirmRecommendation] = []
    for i, item in enumerate(items):
        if isinstance(item, PropFirmPreset):
            preset, label, rules = item, item.label, item.to_prop_rules()
        else:
            label, rules = item
            preset = None
        log(f"Scoring against {label} ({i + 1}/{len(items)})...")
        try:
            mc = run_monte_carlo(trades, rules, cfg)
        except ValueError as exc:
            log(f"  Skipped {label}: {exc}")
            continue
        results.append(FirmRecommendation(
            preset=preset, label=label, rules=rules, mc_result=mc,
            composite_score=_composite_score(mc),
        ))

    results.sort(key=lambda r: r.composite_score, reverse=True)
    return results


def render_recommendation_table(recommendations: list[FirmRecommendation]) -> str:
    """Plain-text ranked table, in the same spirit as
    PayoutFunnelStats.render_table -- suitable for a console log, a
    report, or an experiment-memory entry."""
    if not recommendations:
        return "No candidates scored."
    lines = [
        f"{'Firm / Preset':<38}{'Score':>7}{'Pass %':>9}{'Payout %':>10}{'RoR %':>8}",
        "-" * 72,
    ]
    for r in recommendations:
        lines.append(
            f"{r.label:<38}{r.composite_score:>7.1f}"
            f"{r.mc_result.evaluation_pass_probability:>9.1f}"
            f"{r.mc_result.first_payout_probability:>10.1f}"
            f"{r.mc_result.risk_of_ruin_pct:>8.1f}"
        )
    return "\n".join(lines)
