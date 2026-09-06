"""
Automated Portfolio Composer -- app.portfolio.portfolio.run_portfolio_backtest
already does the hard part (correlation-aware re-weighting, a combined
shared-account trade sequence, and -- when PortfolioConfig.prop_rules/
mc_config are supplied -- the portfolio's OWN Monte Carlo eval-pass
probability), but it requires the CALLER to already know which legs to
combine. This module is the layer on top: given a pool of candidate legs
(typically every validated Strategy Library entry, each paired with the
market data it was validated on), it SEARCHES for the N-strategy
combination that maximizes the combined portfolio's eval_pass_probability,
rather than only ever simulating a combo a human picked by hand.

Search strategy: exhaustive when the candidate pool and leg-count range
are small enough to fit within `max_evaluations` real portfolio backtests
(each of which re-runs every leg's backtest at least twice -- see
run_portfolio_backtest's own two-pass docstring -- so this is genuinely
expensive per combo, unlike a cheap parameter sweep); otherwise a greedy
forward-selection (start from the single best-performing solo leg, then
repeatedly add whichever remaining candidate most improves the combined
score, stopping at max_legs or the first non-improving addition). Greedy
forward selection is not guaranteed to find the global optimum, but it IS
guaranteed to only ever grow the portfolio when doing so provably helps
its own combined Monte Carlo score -- which is what "searches for the
best combination" needs to mean here given real, unbounded search costs,
rather than silently downgrading to something that looks exhaustive but
isn't.

Scoring: PortfolioConfig.prop_rules + mc_config must both be supplied
for the composer to use the portfolio's own combined eval_pass_probability
as its objective (the number the module's own docstring says this
optimizes for). Without them, it falls back to the combined trade
sequence's profit factor purely so the module still runs end-to-end for
exploratory use -- this fallback is clearly labeled in the result, since
optimizing profit factor is a materially different (and, per this app's
own established preference -- see app.monte_carlo.engine.
eval_pass_probability_for_trades's docstring -- weaker) objective than
prop-eval pass probability.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from app.monte_carlo.engine import MonteCarloConfig
from app.portfolio.portfolio import (
    InstrumentLeg,
    PortfolioConfig,
    PortfolioError,
    PortfolioResult,
    run_portfolio_backtest,
)
from app.prop.simulator import PropRules


class PortfolioComposerError(Exception):
    """Raised when a search cannot proceed at all (too few candidate legs,
    or every candidate combination failed to backtest)."""


@dataclass
class ComposerCandidateResult:
    combo_names: tuple
    score: float
    score_metric: str          # "eval_pass_probability" | "profit_factor"
    net_profit: float
    diversification_ratio: float | None


@dataclass
class PortfolioComposerResult:
    best_combo_names: tuple
    best_result: PortfolioResult | None
    score_metric: str
    search_mode: str            # "exhaustive" | "greedy"
    leaderboard: list           # list[ComposerCandidateResult], best-first
    n_candidates_considered: int
    n_combinations_evaluated: int
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_combo_names": list(self.best_combo_names),
            "score_metric": self.score_metric,
            "search_mode": self.search_mode,
            "leaderboard": [
                {
                    "combo_names": list(c.combo_names), "score": c.score, "score_metric": c.score_metric,
                    "net_profit": c.net_profit, "diversification_ratio": c.diversification_ratio,
                }
                for c in self.leaderboard
            ],
            "n_candidates_considered": self.n_candidates_considered,
            "n_combinations_evaluated": self.n_combinations_evaluated,
            "warnings": list(self.warnings),
        }

    def render_table(self) -> str:
        lines = [f"Portfolio Composer -- search mode: {self.search_mode}, objective: {self.score_metric}", ""]
        header = f"{'Rank':<6}{'Legs':>6}  Combination"
        lines.append(header)
        lines.append("-" * 70)
        for i, c in enumerate(self.leaderboard, start=1):
            lines.append(f"{i:<6}{len(c.combo_names):>6}  {', '.join(c.combo_names)}  (score={c.score:.4f})")
        if self.best_combo_names:
            lines.append("")
            lines.append("Best combination: " + ", ".join(self.best_combo_names))
        return "\n".join(lines)


def _score_result(result: PortfolioResult, prop_rules: PropRules | None, mc_config: MonteCarloConfig | None) -> tuple[float, str]:
    if prop_rules is not None and mc_config is not None and result.mc_result is not None:
        return float(result.mc_result.evaluation_pass_probability), "eval_pass_probability"
    stats = result.combined_statistics
    pf = getattr(stats, "profit_factor", None)
    return (float(pf) if pf is not None and math.isfinite(pf) else 0.0), "profit_factor"


def _evaluate_combo(
    combo: tuple[InstrumentLeg, ...],
    base_config: PortfolioConfig,
) -> tuple[PortfolioResult | None, float, str, str | None]:
    try:
        result = run_portfolio_backtest(list(combo), base_config)
    except PortfolioError as exc:
        return None, float("-inf"), "eval_pass_probability", str(exc)
    except Exception as exc:  # noqa: BLE001 -- one bad combo must not stop the search
        return None, float("-inf"), "eval_pass_probability", str(exc)
    score, metric = _score_result(result, base_config.prop_rules, base_config.mc_config)
    return result, score, metric, None


def compose_portfolio(
    candidates: list[InstrumentLeg],
    min_legs: int = 2,
    max_legs: int = 4,
    max_evaluations: int = 60,
    portfolio_config: PortfolioConfig | None = None,
    top_k: int = 10,
) -> PortfolioComposerResult:
    """
    candidates: every leg worth considering -- typically one InstrumentLeg
        per validated Strategy Library entry, each already paired with the
        market data it was validated on (build these with
        app.strategy.library_loader.load_validated_candidates() plus your
        own per-instrument DataFrames and RiskConfigs).
    min_legs/max_legs: the combination sizes to search over. min_legs is
        floored at 2 -- run_portfolio_backtest itself requires at least 2
        legs (a "portfolio" of one strategy is just that strategy).
    max_evaluations: the hard cap on real portfolio backtests this search
        will run. When the full combinatorial space (sum of
        C(len(candidates), k) for k in [min_legs, max_legs]) exceeds this,
        the search automatically falls back to greedy forward selection
        instead of silently truncating an exhaustive search partway
        through (which would bias toward whichever combinations happened
        to be generated first).
    portfolio_config: the PortfolioConfig template applied to EVERY
        evaluated combination (correlation settings, initial_balance, and
        -- for the real objective -- prop_rules + mc_config). A fresh
        default PortfolioConfig() is used if not supplied, which means no
        Monte Carlo is run and the search falls back to scoring by profit
        factor (see module docstring).
    top_k: how many entries the returned leaderboard keeps.
    """
    min_legs = max(int(min_legs), 2)
    max_legs = max(int(max_legs), min_legs)
    cfg = portfolio_config or PortfolioConfig()
    if len(candidates) < min_legs:
        raise PortfolioComposerError(
            f"Need at least {min_legs} candidate legs to search combinations of that size; got {len(candidates)}."
        )

    warnings: list[str] = []
    names = [leg.name for leg in candidates]
    if len(set(names)) != len(names):
        raise PortfolioComposerError("Candidate leg names must be unique.")
    by_name = {leg.name: leg for leg in candidates}

    total_combos = sum(math.comb(len(candidates), k) for k in range(min_legs, max_legs + 1))
    use_exhaustive = total_combos <= max_evaluations

    leaderboard: list[ComposerCandidateResult] = []
    best_combo: tuple[str, ...] = ()
    best_result: PortfolioResult | None = None
    best_score = float("-inf")
    score_metric = "eval_pass_probability" if (cfg.prop_rules and cfg.mc_config) else "profit_factor"
    n_evaluated = 0

    def record(combo_names: tuple[str, ...], result: PortfolioResult | None, score: float, metric: str, error: str | None):
        nonlocal best_combo, best_result, best_score
        if error is not None:
            warnings.append(f"Combination ({', '.join(combo_names)}) failed and was skipped: {error}")
            return
        leaderboard.append(ComposerCandidateResult(
            combo_names=combo_names, score=score, score_metric=metric,
            net_profit=float(result.combined_statistics.net_profit),
            diversification_ratio=result.diversification_ratio,
        ))
        if score > best_score:
            best_score, best_combo, best_result = score, combo_names, result

    if use_exhaustive:
        search_mode = "exhaustive"
        for k in range(min_legs, max_legs + 1):
            for combo_names in itertools.combinations(names, k):
                combo = tuple(by_name[n] for n in combo_names)
                result, score, metric, error = _evaluate_combo(combo, cfg)
                n_evaluated += 1
                record(combo_names, result, score, metric, error)
    else:
        search_mode = "greedy"
        warnings.append(
            f"{total_combos} possible combinations exceeds max_evaluations={max_evaluations} -- "
            "using greedy forward selection instead of an exhaustive search."
        )
        # Seed with the best pair (a single leg can't be scored by
        # run_portfolio_backtest, which requires >= 2 legs) among a
        # bounded sample of starting pairs so the greedy search itself
        # doesn't silently become an exhaustive one at k=2.
        pair_budget = max(max_evaluations // 2, 1)
        starting_pairs = list(itertools.islice(itertools.combinations(names, 2), pair_budget))
        best_seed_names: tuple[str, ...] = ()
        best_seed_result: PortfolioResult | None = None
        best_seed_score = float("-inf")
        for combo_names in starting_pairs:
            combo = tuple(by_name[n] for n in combo_names)
            result, score, metric, error = _evaluate_combo(combo, cfg)
            n_evaluated += 1
            record(combo_names, result, score, metric, error)
            if score > best_seed_score:
                best_seed_score, best_seed_names, best_seed_result = score, combo_names, result

        if not best_seed_names:
            raise PortfolioComposerError("Every starting pair failed to backtest -- cannot search further.")

        current_names = list(best_seed_names)
        current_score = best_seed_score
        remaining = [n for n in names if n not in current_names]
        while len(current_names) < max_legs and remaining and n_evaluated < max_evaluations:
            improved = False
            best_addition = None
            best_addition_score = current_score
            best_addition_result = None
            for candidate_name in remaining:
                if n_evaluated >= max_evaluations:
                    break
                trial_names = tuple(current_names + [candidate_name])
                combo = tuple(by_name[n] for n in trial_names)
                result, score, metric, error = _evaluate_combo(combo, cfg)
                n_evaluated += 1
                record(trial_names, result, score, metric, error)
                if score > best_addition_score:
                    best_addition_score, best_addition, best_addition_result = score, candidate_name, result
                    improved = True
            if not improved:
                break
            current_names.append(best_addition)
            current_score = best_addition_score
            remaining.remove(best_addition)

    if not leaderboard:
        raise PortfolioComposerError("Every candidate combination failed to backtest -- nothing to compose.")

    leaderboard.sort(key=lambda c: c.score, reverse=True)
    return PortfolioComposerResult(
        best_combo_names=best_combo, best_result=best_result, score_metric=score_metric,
        search_mode=search_mode, leaderboard=leaderboard[:top_k],
        n_candidates_considered=len(candidates), n_combinations_evaluated=n_evaluated,
        warnings=warnings,
    )
