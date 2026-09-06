"""
Auto Regime Selector -- promotes app.strategy.regime_router.RegimeRouterStrategy
from "you hand-pick which sub-strategy runs in which regime" into a
first-class engine feature: given a pool of candidate strategies
(typically the validated Strategy Library), this module runs each
candidate ONCE over the full dataset, attributes its trades to whichever
regime (trend / volatility / session / environment -- the same four
classification dimensions app.validation.regime_matrix already fits) was
active at each trade's entry, scores every (strategy, regime) pair, and
assembles the single best-suited strategy per regime into a
RegimeRouterStrategy automatically -- so a validated strategy trades only
the conditions it's actually good at, and a different validated strategy
takes over the moment the regime changes, rather than one strategy
running blind through everything.

Scoring: for a (strategy, regime) pair with at least `min_trades_per_cell`
trades in that regime, the score is eval_pass_probability_for_trades()
run on JUST that regime-isolated trade subset -- "if this strategy only
ever traded this regime, what's its prop-eval pass probability" -- the
same eval_pass_probability metric every other optimizer in this app
(Iterative Refinement, Quick Optimize, Full Pipeline, Evolution Lab,
Search Lab, CPCV/PBO, the walk-forward-aware GA) already scores against,
per app.monte_carlo.engine.eval_pass_probability_for_trades's own
docstring. Below that trade threshold a pair is scored by plain per-trade
expectancy instead purely for visibility in the results table -- it is
NEVER eligible to be selected as a regime's winner, since a promising
expectancy on a handful of trades is exactly the kind of unproven result
this app's own philosophy (see app.search.batch_runner's early-kill
floor, app.validation.walk_forward_opt) already treats with suspicion. A
regime with no candidate clearing the threshold is left UNASSIGNED --
RegimeRouterStrategy already forces unassigned-regime bars flat (see its
own docstring), which is the strictly safer statement "no strategy has
been vetted for this condition" rather than silently falling back to
some other regime's winner.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig, eval_pass_probability_for_trades
from app.prop.simulator import PropRules
from app.strategy.base import Strategy
from app.strategy.regime_router import RegimeRouterStrategy, _VALID_DIMENSIONS
from app.validation.regime_matrix import RegimeThresholds, label_regimes


class RegimeSelectorError(Exception):
    """Raised when auto regime selection cannot proceed at all (bad
    dimension name, no candidates supplied)."""


@dataclass
class RegimeCandidateScore:
    strategy_name: str
    regime_label: str
    n_trades: int
    eval_pass_probability: float | None   # None when below min_trades_per_cell -- see scored_by
    expectancy: float                     # mean pnl per trade in this regime, for context either way
    scored_by: str                        # "monte_carlo" | "expectancy_only" | "no_trades"


@dataclass
class AutoRegimeSelectionResult:
    regime_dimension: str
    router: RegimeRouterStrategy | None
    assignments: dict                     # {regime_label: strategy_name} -- winners only
    unassigned_regimes: list              # regimes present in the data with no qualifying candidate
    scores: list                          # list[RegimeCandidateScore], every (candidate, regime) pair evaluated
    thresholds: RegimeThresholds
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "regime_dimension": self.regime_dimension,
            "assignments": dict(self.assignments),
            "unassigned_regimes": list(self.unassigned_regimes),
            "scores": [
                {
                    "strategy_name": s.strategy_name, "regime_label": s.regime_label, "n_trades": s.n_trades,
                    "eval_pass_probability": s.eval_pass_probability, "expectancy": s.expectancy,
                    "scored_by": s.scored_by,
                }
                for s in self.scores
            ],
            "warnings": list(self.warnings),
        }

    def render_table(self) -> str:
        dim = self.regime_dimension.title()
        lines = [f"Auto Regime Selection ({dim})", ""]
        header = f"{'Strategy':<28}{'Regime':<16}{'Trades':>8}{'Pass Prob':>11}{'Expectancy':>12}  Basis"
        lines.append(header)
        lines.append("-" * len(header))
        for s in sorted(self.scores, key=lambda s: (s.regime_label, -(s.eval_pass_probability or -1))):
            pp = f"{s.eval_pass_probability:.1f}%" if s.eval_pass_probability is not None else "n/a"
            lines.append(f"{s.strategy_name:<28}{s.regime_label:<16}{s.n_trades:>8}{pp:>11}{s.expectancy:>+12.2f}  {s.scored_by}")
        lines.append("")
        if self.assignments:
            lines.append("Winning assignment: " + ", ".join(f"{r} -> {n}" for r, n in self.assignments.items()))
        if self.unassigned_regimes:
            lines.append("Unassigned (forced flat): " + ", ".join(self.unassigned_regimes))
        return "\n".join(lines)


def _expectancy(pnls: list[float]) -> float:
    return sum(pnls) / len(pnls) if pnls else 0.0


def _trades_by_regime(trades: list, regime_series: pd.Series, df: pd.DataFrame) -> dict[str, list]:
    """Maps each trade to the regime label active at its entry bar, using
    an as-of (never look-ahead) lookup: the same searchsorted-then-step-
    back-one-bar convention app.validation.regime_matrix._attribute_trades
    already uses for identical reasons."""
    ts = pd.to_datetime(df["timestamp"])
    out: dict[str, list] = {}
    for t in trades:
        idx = ts.searchsorted(pd.Timestamp(t.entry_time), side="right") - 1
        if idx < 0 or idx >= len(regime_series):
            continue
        label = regime_series.iloc[idx]
        if pd.isna(label):
            continue
        out.setdefault(label, []).append(t)
    return out


def select_regime_strategies(
    df: pd.DataFrame,
    candidates: dict[str, Strategy],
    regime_dimension: str,
    risk: RiskConfig | None = None,
    pip_size: float | None = None,
    prop_rules: PropRules | None = None,
    mc_config: MonteCarloConfig | None = None,
    min_trades_per_cell: int = 20,
    thresholds: RegimeThresholds | None = None,
) -> AutoRegimeSelectionResult:
    """
    df: full OHLCV history to fit regimes on and backtest every candidate
        against (same data every candidate should already be validated on).
    candidates: {display_name: Strategy} -- e.g. built from
        app.strategy.library_loader.load_validated_candidates().
    regime_dimension: one of "trend" | "volatility" | "session" | "environment".
    risk: RiskConfig every candidate is backtested with. Defaults to
        RiskConfig() if not supplied -- pass the SAME risk settings you'd
        use for a real run, since sizing affects which trades survive to
        be attributed to a regime.
    pip_size: passed through to the resulting RegimeRouterStrategy for its
        own scalar-pips-to-distance conversion (see that class's
        docstring). Defaults to risk.pip_size when not given.
    thresholds: reuse previously-fit RegimeThresholds instead of refitting
        fresh quantiles on `df` -- important when calling this again on a
        forward/live window so "high volatility" keeps meaning the same
        thing it meant when the candidates were originally validated.
    """
    if regime_dimension not in _VALID_DIMENSIONS:
        raise RegimeSelectorError(f"Unknown regime dimension '{regime_dimension}'. Must be one of {sorted(_VALID_DIMENSIONS)}.")
    if not candidates:
        raise RegimeSelectorError("select_regime_strategies needs at least one named candidate strategy.")

    risk_cfg = risk or RiskConfig()
    rules = prop_rules or PropRules()
    mc_cfg = mc_config or MonteCarloConfig(n_simulations=1000)
    resolved_pip_size = pip_size if pip_size is not None else risk_cfg.pip_size

    labels, fit_thresholds = label_regimes(df, thresholds=thresholds)
    regime_series = labels[regime_dimension]
    present_regimes = sorted(str(r) for r in regime_series.dropna().unique())

    warnings: list[str] = []
    scores: list[RegimeCandidateScore] = []
    best_for_regime: dict[str, tuple[float, str]] = {}

    for name, strategy in candidates.items():
        try:
            bt = run_backtest(df, strategy, risk_cfg)
        except Exception as exc:  # noqa: BLE001 -- one bad candidate must not stop the rest
            warnings.append(f"'{name}' failed to backtest and was excluded: {exc}")
            continue
        if not bt.trades:
            warnings.append(f"'{name}' produced zero trades on this data and was excluded.")
            continue

        by_regime = _trades_by_regime(bt.trades, regime_series, df)
        for regime_label in present_regimes:
            subset = by_regime.get(regime_label, [])
            pnls = [t.pnl for t in subset]
            if len(subset) >= min_trades_per_cell:
                score = eval_pass_probability_for_trades(subset, rules, mc_cfg)
                scores.append(RegimeCandidateScore(name, regime_label, len(subset), score, _expectancy(pnls), "monte_carlo"))
                current = best_for_regime.get(regime_label)
                if current is None or score > current[0]:
                    best_for_regime[regime_label] = (score, name)
            elif subset:
                scores.append(RegimeCandidateScore(name, regime_label, len(subset), None, _expectancy(pnls), "expectancy_only"))
            else:
                scores.append(RegimeCandidateScore(name, regime_label, 0, None, 0.0, "no_trades"))

    assignments = {regime: name for regime, (_, name) in best_for_regime.items()}
    unassigned = [r for r in present_regimes if r not in assignments]
    if unassigned:
        warnings.append(
            f"No candidate cleared {min_trades_per_cell}+ trades in regime(s): {', '.join(unassigned)}. "
            "These bars will be forced flat by the router rather than assigned an unproven strategy."
        )

    router = None
    if assignments:
        router = RegimeRouterStrategy(
            regime_dimension=regime_dimension,
            strategies_by_regime={r: candidates[name] for r, name in assignments.items()},
            pip_size=resolved_pip_size,
            thresholds=fit_thresholds,
            name=f"Auto Regime Router ({regime_dimension}: " + ", ".join(f"{r}->{n}" for r, n in assignments.items()) + ")",
        )
    else:
        warnings.append("No regime had a candidate clear the minimum trade threshold -- no router was built.")

    return AutoRegimeSelectionResult(
        regime_dimension=regime_dimension, router=router, assignments=assignments,
        unassigned_regimes=unassigned, scores=scores, thresholds=fit_thresholds, warnings=warnings,
    )
