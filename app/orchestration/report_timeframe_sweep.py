"""
Run & Report timeframe sweep -- "run this strategy on 5m, 15m, 30m, 1h and
4h bars, all resampled from the one file I loaded (e.g. 1-minute MGC), and
show me the numbers side by side."

Deliberately a thin outer loop over the exact same run_backtest ->
run_monte_carlo path the single-timeframe Run & Report route uses, so every
per-timeframe row is directly comparable to a normal run. The route then
generates its full report for the winning timeframe only.

Selection caveat (surfaced in `selection_note`): picking the best of N
timeframes on the same history is itself a mild form of data-mining. The
winner still goes through the normal holdout comparison in the report, but
treat a sweep winner as a candidate to send to Full Pipeline, not a verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.data.timeframe_sweep import build_timeframe_sweep, stamp_manual_timeframe
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules


@dataclass
class TimeframeRow:
    label: str
    bars: int
    trades: int
    net_profit: float
    win_rate: float
    max_dd_pct: float
    eval_pass_probability: float
    first_payout_probability: float
    risk_of_ruin_pct: float
    is_best: bool = False
    # v9.15: per-attempt (gate-metric) pair for DISPLAY. The chain-level
    # fields above still drive the best-timeframe pick (unchanged
    # behavior); the table shows these so its "Eval pass %" column
    # answers the same one-account question as every other surface.
    eval_pass_probability_per_attempt: float = 0.0
    first_payout_probability_per_attempt: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ReportSweepResult:
    rows: list[TimeframeRow]
    best_label: str | None
    best_df: pd.DataFrame | None
    best_strategy: object | None
    skipped: list = field(default_factory=list)
    errors: dict = field(default_factory=dict)
    selection_note: str = ""


def run_report_timeframe_sweep(
    df: pd.DataFrame, strategy, risk: RiskConfig, rules: PropRules, timeframes: list[str],
    adaptive_risk=None, n_sims: int = 1000, mc_method: str = "bootstrap",
    reset_on_breach: bool = False, cancel_event=None, progress_cb=None,
) -> ReportSweepResult:
    plan = build_timeframe_sweep(df, timeframes or [])
    rows: list[TimeframeRow] = []
    errors: dict[str, str] = {}
    frames: dict[str, tuple[pd.DataFrame, object]] = {}

    for target in plan.targets:
        if cancel_event is not None and cancel_event.is_set():
            break
        label = target.label
        run_strategy = stamp_manual_timeframe(strategy, label)
        if progress_cb:
            progress_cb(f"[{label}] backtesting {target.bar_count:,} bars")
        try:
            bt = run_backtest(target.dataframe, run_strategy, risk, adaptive_risk=adaptive_risk)
            if not bt.trades:
                errors[label] = "no trades on this timeframe"
                rows.append(TimeframeRow(label, target.bar_count, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0))
                continue
            mc = run_monte_carlo(
                bt.trades, rules,
                MonteCarloConfig(n_simulations=max(200, min(int(n_sims), 5000)), method=mc_method, reset_on_breach=reset_on_breach),
            )
        except Exception as exc:  # noqa: BLE001 -- one bad timeframe must not kill the others
            errors[label] = str(exc)
            continue
        st = bt.statistics
        rows.append(TimeframeRow(
            label=label, bars=target.bar_count, trades=len(bt.trades), net_profit=float(st.net_profit),
            win_rate=float(st.win_rate), max_dd_pct=float(st.max_drawdown_pct),
            eval_pass_probability=float(mc.evaluation_pass_probability),
            first_payout_probability=float(mc.first_payout_probability), risk_of_ruin_pct=float(mc.risk_of_ruin_pct),
            eval_pass_probability_per_attempt=float(mc.headline_evaluation_pass_probability),
            first_payout_probability_per_attempt=float(mc.headline_first_payout_probability),
        ))
        frames[label] = (target.dataframe, run_strategy)

    candidates = [r for r in rows if r.trades > 0 and r.label in frames]
    best = max(candidates, key=lambda r: (r.eval_pass_probability, r.net_profit), default=None)
    if best is not None:
        best.is_best = True
    note = (
        f"Best of {len(candidates)} timeframe(s) tested on the same history, so its numbers are optimistic by "
        "construction. Send the winner through Full Pipeline (which hides a true holdout) before trusting it."
    ) if best is not None else ""
    return ReportSweepResult(
        rows=rows, best_label=best.label if best else None,
        best_df=frames[best.label][0] if best else None,
        best_strategy=frames[best.label][1] if best else None,
        skipped=plan.skipped, errors=errors, selection_note=note,
    )
