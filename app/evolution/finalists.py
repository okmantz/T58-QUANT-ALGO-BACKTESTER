"""
Finalist Report -- turns an Evolution Lab leaderboard into the two
things Owen asked for on top of a single ranked list:

  1. The full sequential funnel (P(pass eval) -> P(reach funded) ->
     P(first payout) -> P(second payout) -> P(third payout)), not just
     the single evaluation_pass_probability/first_payout_probability
     pair PROP FITNESS ranks on. app.prop.survival_engine already
     computes exactly this (PayoutFunnelStats / FundedSurvivalStats) --
     it's just never been called from the Evolution Lab's own pipeline,
     only from one-off single-strategy reports (app.reports.
     survival_report, app.lab.strategy_lab, the web/desktop "Run &
     Report" flow). This module is the missing wire.

  2. A Pareto frontier over the finalists (app.scoring.pareto) instead
     of pretending PROP FITNESS's single number is "the" answer --
     Conservative / Balanced / Aggressive tradeoffs, per Owen's example
     (85% eval / 38% payout / low return vs. 72% eval / 61% payout /
     higher return vs. 67% eval / 73% payout / higher volatility).

Deliberately run ONLY on the leaderboard (elite_keep candidates, a
handful), not on every candidate ever tested -- the full survival
analysis (thousands of Monte Carlo resamples with the reset-economics
chain) is exactly the kind of expensive, only-worth-it-for-survivors
step app.evolution.prop_fitness's own docstring describes PROP FITNESS
as existing to gate before. Call this once, on demand (end of a run, or
whenever Owen wants to review finalists), not every generation.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.prop.simulator import PropRules
from app.prop.survival_engine import PropSurvivalConfig, run_prop_survival_analysis
from app.prop.rolling_evaluation import run_rolling_evaluation
from app.scoring.pareto import compute_pareto_frontier, label_frontier


@dataclass
class FinalistReport:
    candidate_id: str
    family: str
    fitness_score: float | None
    probability_pass_evaluation: float
    probability_first_payout: float
    probability_second_payout: float
    probability_third_payout: float
    prop_survival_score: float
    max_drawdown_pct: float | None
    # Rolling Evaluation Windows (see app.prop.rolling_evaluation) -- the
    # exact eval ruleset re-run from every real historical starting day,
    # rather than resampled/reordered like the Monte Carlo-based
    # probabilities above. None when window_trading_days wasn't given to
    # build_finalist_reports (the analysis is skipped, not defaulted to
    # a guess) or the analysis couldn't run (e.g. too few distinct
    # trading days for even one window).
    rolling_eval_pass_rate_pct: float | None = None
    rolling_eval_windows_tested: int | None = None
    rolling_eval_worst_period: str | None = None
    rolling_eval_best_period: str | None = None
    label: str | None = None                # Conservative / Balanced / Aggressive
    on_frontier: bool = False
    notes: list = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def build_finalist_reports(
    leaderboard: list,               # list of EvolutionCandidateRecord (duck-typed: .candidate_id/.meta/.trades/.stats/.fitness)
    prop_rules: PropRules,
    survival_cfg: PropSurvivalConfig | None = None,
    top_n: int = 10,
    window_trading_days: int | None = None,
) -> list[FinalistReport]:
    """Runs the full survival-engine funnel on the top `top_n`
    leaderboard entries (by existing PROP FITNESS score, so the
    expensive analysis is spent on the candidates most likely to be
    worth it) and returns one FinalistReport per candidate that had
    enough trades to analyze. A candidate with too few trades to run a
    meaningful survival analysis is skipped (not crashed on) -- the
    same trade-count floor issue app.evolution.prop_fitness already
    penalizes in PROP FITNESS itself.

    window_trading_days: if given, also runs Rolling Evaluation Windows
    (app.prop.rolling_evaluation) per finalist -- Owen's "what fraction
    of REAL historical starting points would this have actually passed
    from" number, distinct from (and, per his own point, often more
    honest than) the Monte-Carlo-resampled probabilities above. Left
    None by default since it's the more expensive of the two analyses
    and the caller may not know the target firm's actual window length
    yet.
    """
    ranked = sorted(
        (r for r in leaderboard if getattr(r, "fitness", None) is not None),
        key=lambda r: r.fitness.final_score, reverse=True,
    )[:top_n]

    reports: list[FinalistReport] = []
    for r in ranked:
        trades = getattr(r, "trades", None) or []
        if len(trades) < 5:
            continue
        try:
            result = run_prop_survival_analysis(trades, prop_rules, survival_cfg)
        except Exception:  # noqa: BLE001 -- a survival-analysis failure must not block reporting the rest
            continue

        rolling_pass_rate = rolling_windows_tested = rolling_worst = rolling_best = None
        if window_trading_days is not None:
            try:
                rolling = run_rolling_evaluation(trades, prop_rules, window_trading_days)
                rolling_pass_rate = rolling.pass_rate_pct
                rolling_windows_tested = rolling.n_windows
                rolling_worst = rolling.worst_starting_period
                rolling_best = rolling.best_starting_period
            except Exception:  # noqa: BLE001 -- rolling eval is an enrichment, never blocks the base report
                pass

        reports.append(FinalistReport(
            candidate_id=r.candidate_id,
            family=(r.meta or {}).get("family", "?"),
            fitness_score=r.fitness.final_score,
            probability_pass_evaluation=result.evaluation.probability_pass_evaluation,
            probability_first_payout=result.funded.probability_first_payout,
            probability_second_payout=result.funded.probability_second_payout,
            probability_third_payout=result.funded.probability_third_payout,
            prop_survival_score=result.prop_survival_score,
            max_drawdown_pct=(r.stats or {}).get("max_drawdown_pct"),
            rolling_eval_pass_rate_pct=rolling_pass_rate,
            rolling_eval_windows_tested=rolling_windows_tested,
            rolling_eval_worst_period=rolling_worst,
            rolling_eval_best_period=rolling_best,
            notes=list(result.notes),
        ))
    return reports


def pareto_frontier_for_finalists(reports: list[FinalistReport]) -> list[FinalistReport]:
    """Tags each report's .on_frontier / .label in place (and also
    returns the list, for chaining) using probability_pass_evaluation
    as the 'safety' axis and probability_first_payout as the 'upside'
    axis -- the same two numbers Owen's own Conservative/Balanced/
    Aggressive example contrasts directly."""
    if not reports:
        return reports
    points = [r.to_dict() | {"candidate_id": r.candidate_id} for r in reports]
    metrics = {"probability_pass_evaluation": "max", "probability_first_payout": "max"}
    frontier = compute_pareto_frontier(points, metrics, key_fn=lambda p: p["candidate_id"])
    frontier = label_frontier(frontier, conservative_metric="probability_pass_evaluation",
                               aggressive_metric="probability_first_payout")
    by_id = {p.key: p for p in frontier}
    for r in reports:
        p = by_id.get(r.candidate_id)
        if p is not None:
            r.on_frontier = not p.dominated
            r.label = p.label
    return reports


def render_finalist_report(reports: list[FinalistReport]) -> str:
    lines = ["Finalists -- sequential funnel + Pareto frontier", ""]
    if not reports:
        lines.append("(no finalist had enough trades for a full survival analysis yet)")
        return "\n".join(lines)
    has_rolling = any(r.rolling_eval_pass_rate_pct is not None for r in reports)
    header = f"{'Candidate':<32}{'Label':<14}{'Pass Eval':>10}{'Payout 1':>10}{'Payout 2':>10}{'Payout 3':>10}"
    if has_rolling:
        header += f"{'Rolling Pass':>14}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in sorted(reports, key=lambda r: (not r.on_frontier, -(r.fitness_score or 0))):
        tag = r.label or ("dominated" if not r.on_frontier else "")
        line = (
            f"{r.candidate_id:<32}{tag:<14}{r.probability_pass_evaluation:>9.1f}%"
            f"{r.probability_first_payout:>9.1f}%{r.probability_second_payout:>9.1f}%{r.probability_third_payout:>9.1f}%"
        )
        if has_rolling:
            rolling_str = f"{r.rolling_eval_pass_rate_pct:.1f}%" if r.rolling_eval_pass_rate_pct is not None else "--"
            line += f"{rolling_str:>14}"
        lines.append(line)
    return "\n".join(lines)
