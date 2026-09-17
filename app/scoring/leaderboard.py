"""
Final Selection Leaderboard -- pipeline reorg plan items #5/#3 ("one
unified leaderboard" / "leaderboard + status taxonomy").

Every strategy that has been through Full Pipeline (whether it started
there, in Forge, in Evolution Lab, in Search Lab -- anything that ends
by calling app.strategy.library.record_backtest_result) already gets a
"last_run" metadata block on its saved library entry. That block is,
after the full_pipeline.py wiring in this same change, the ONE place
every tool's output converges: it now carries t58_score / t58_tier /
parsimony_score / risk_of_ruin_pct alongside the fields it already had.

This module does nothing statistically new -- it just reads that
existing convergence point across the WHOLE library (not one tool's own
local run) and ranks by t58_score, which is what turns ~10 separate
per-tool "leaderboards" (Search Lab's results_db, Evolution Lab's own
leaderboard, Forge's, Portfolio's, etc. -- see the pipeline reorg plan's
finding #5) into one Final Selection view, per the plan's section 26.

A strategy only shows up here once it has a t58_score recorded (i.e. it
has actually been through Full Pipeline at least once) -- a strategy
still sitting in Discovery/Filter with no last_run.t58_score simply
isn't ranked yet, which is correct: this is a FINAL SELECTION view, not
a strategy browser (app.strategy.library.list_saved_strategies already
covers that).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.strategy.library import STRATEGY_TYPES, list_saved_strategies


@dataclass
class LeaderboardEntry:
    strategy_type: str
    filename: str
    status: str
    status_display: str
    t58_score: float
    t58_tier: str | None
    prop_pass_probability: float | None      # eval_pass_probability, from the same last_run block
    payout_probability: float | None
    risk_of_ruin_pct: float | None
    risk_of_ruin_hard_fail: bool
    lookahead_hard_fail: bool
    max_drawdown_pct: float | None
    trade_count: int | None
    parsimony_score: float | None
    verdict: str | None
    report_html: str | None
    modified: float


def build_leaderboard(
    strategy_type: str | None = None,
    top_n: int = 20,
    exclude_ruin_hard_fail: bool = True,
    exclude_lookahead_hard_fail: bool = True,
) -> list[LeaderboardEntry]:
    """Ranked, cross-tool Final Selection leaderboard.

    strategy_type: restrict to one of app.strategy.library.STRATEGY_TYPES,
    or None (default) for all of them -- a manual-builder winner and a
    Python winner are directly comparable once both have a t58_score.

    exclude_ruin_hard_fail: True (default) drops anything that failed
    the risk-of-ruin hard safety gate (pipeline reorg plan section 19) --
    those are never candidates for Final Selection regardless of their
    other numbers. Set False to see them anyway (e.g. for an "everything
    that's been tested" audit view rather than a shortlist).

    exclude_lookahead_hard_fail: True (default) drops anything that
    failed Full Pipeline's lookahead-bias hard safety gate (see
    app.orchestration.full_pipeline._make_verdict) -- same reasoning as
    exclude_ruin_hard_fail: a strategy whose signal depends on future
    data has no numbers worth ranking, regardless of what its (equally
    untrustworthy) t58_score happens to say.
    """
    entries: list[LeaderboardEntry] = []
    for item in list_saved_strategies(strategy_type=strategy_type):
        last_run = (item.metadata or {}).get("last_run") or {}
        t58_score = last_run.get("t58_score")
        if t58_score is None:
            continue  # never been through Full Pipeline (or ran before this field existed) -- not rankable yet
        if exclude_ruin_hard_fail and last_run.get("risk_of_ruin_hard_fail"):
            continue
        if exclude_lookahead_hard_fail and last_run.get("lookahead_hard_fail"):
            continue
        entries.append(LeaderboardEntry(
            strategy_type=item.strategy_type,
            filename=item.name,
            status=item.status,
            status_display=item.status_display,
            t58_score=t58_score,
            t58_tier=last_run.get("t58_tier"),
            prop_pass_probability=last_run.get("eval_pass_probability"),
            payout_probability=last_run.get("first_payout_probability"),
            risk_of_ruin_pct=last_run.get("risk_of_ruin_pct"),
            risk_of_ruin_hard_fail=bool(last_run.get("risk_of_ruin_hard_fail")),
            lookahead_hard_fail=bool(last_run.get("lookahead_hard_fail")),
            max_drawdown_pct=last_run.get("max_dd"),
            trade_count=last_run.get("trades"),
            parsimony_score=last_run.get("parsimony_score"),
            verdict=last_run.get("verdict"),
            report_html=last_run.get("report_html"),
            modified=item.modified,
        ))
    entries.sort(key=lambda e: e.t58_score, reverse=True)
    return entries[:top_n]


def render_leaderboard_table(entries: list[LeaderboardEntry]) -> str:
    """Plain-text render for the UI's output pane / a CLI -- same
    convention as app.search.graveyard.render_graveyard_report and
    app.validation.regime_matrix.RegimeMatrixResult.render_table."""
    if not entries:
        return (
            "No strategies have a T58 Score yet. Run Full Pipeline on at least one strategy first --\n"
            "the leaderboard only ranks strategies that have actually gone through Final Selection\n"
            "scoring, not everything sitting in the library."
        )
    lines = ["FINAL SELECTION LEADERBOARD", ""]
    header = f"{'#':<3}{'Strategy':<34}{'Type':<10}{'Score':>7}{'Tier':>11}{'Pass%':>8}{'Payout%':>9}{'RoR%':>7}{'DD%':>7}{'Trades':>8}{'Parsimony':>11}  Status"
    lines.append(header)
    lines.append("-" * len(header))
    for i, e in enumerate(entries, start=1):
        def _fmt(v, spec="{:.1f}"):
            return spec.format(v) if v is not None else "--"
        lines.append(
            f"{i:<3}{e.filename[:33]:<34}{e.strategy_type:<10}{_fmt(e.t58_score):>7}{(e.t58_tier or '--'):>11}"
            f"{_fmt(e.prop_pass_probability):>8}{_fmt(e.payout_probability):>9}{_fmt(e.risk_of_ruin_pct):>7}"
            f"{_fmt(e.max_drawdown_pct):>7}{(e.trade_count if e.trade_count is not None else '--'):>8}"
            f"{_fmt(e.parsimony_score):>11}  {e.status_display}"
        )
    return "\n".join(lines)
