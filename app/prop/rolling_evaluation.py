"""
Rolling Evaluation Windows -- Owen's ask: "the number that decides a prop
evaluation is not expectancy, it is the shape of the worst path inside
the window... slide [the eval rules] across your backtest starting at
every possible start bar. That gives you a pass rate rather than a
verdict."

app.prop.simulator.simulate_account already contains every rule this
needs (daily loss limit, drawdown, profit target, min trading days,
consistency rule) -- it just runs it ONCE, from the first trade. This
module calls that exact same function once per candidate start day
(no reimplementation of eval logic, so a rule change in simulate_account
automatically applies here too) and aggregates the pass/fail verdicts
into the distribution Owen described instead of a single verdict.

Distinct from every other robustness check already in this codebase:
  - Monte Carlo (app.monte_carlo.engine) asks "what if I reorder/resample
    this exact set of trade outcomes?"
  - Walk-forward (app.search.robustness) asks "does this keep working
    when trained/tested through time?"
  - CPCV (app.validation.cpcv) asks "how sensitive is this to which
    slice of history is train vs. test?"
  - THIS asks "if I had actually purchased this evaluation at any real
    historical moment, how often would I have passed?" -- no resampling,
    no reordering, just literally every real starting point in the
    trade sequence that was actually observed.

Performance note: each window re-simulates the (variable-length) tail of
the trade sequence from its own start day, so total cost is roughly
O(n_windows x average_remaining_trades) -- quadratic-ish in the number
of trades for a full scan. `stride` and `max_windows` exist specifically
to make this practical on a multi-year, multi-million-bar dataset (the
same reasoning app.evolution.engine's cpcv_top_n already applies to
CPCV: this is a stage worth spending real compute on, but not on every
single day of a huge dataset every time).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from app.backtest.execution import Trade
from app.prop.simulator import PropRules, precompute_day_structure, simulate_account


@dataclass
class WindowOutcome:
    start_day_index: int
    start_date: str
    passed: bool
    failure_category: str | None    # "daily_loss_limit" | "max_drawdown" | "target_not_reached" | "too_many_days" | None (passed)
    days_to_pass: int | None
    reached_first_payout: bool

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RollingEvaluationResult:
    window_trading_days: int
    n_windows: int
    n_passed: int
    n_failed: int
    pass_rate_pct: float
    first_payout_rate_pct: float
    median_days_to_pass: float | None
    worst_starting_period: str | None    # "YYYY-MM" with the lowest pass rate (min n_windows_per_period samples)
    best_starting_period: str | None
    failure_breakdown: dict              # {category: count}
    windows: list = field(default_factory=list)   # list[WindowOutcome] -- kept for detail views/export

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["windows"] = [w.to_dict() for w in self.windows]
        return d

    def render(self) -> str:
        lines = [
            "T58 STRATEGY RESULT -- Rolling Evaluation Windows",
            "",
            f"Eval Pass Rate       {self.pass_rate_pct:.1f}%",
            "",
            f"Windows Tested       {self.n_windows:,}",
            f"Passed               {self.n_passed:,}",
            f"Failed               {self.n_failed:,}",
            "",
            f"Median Days to Pass  {self.median_days_to_pass:.0f}" if self.median_days_to_pass is not None else "Median Days to Pass  --",
            f"Worst Starting Period  {self.worst_starting_period or '--'}",
            f"Best Starting Period   {self.best_starting_period or '--'}",
            "",
            f"First Payout Rate    {self.first_payout_rate_pct:.1f}%",
        ]
        if self.failure_breakdown:
            lines += ["", "Failure breakdown:"]
            label = {
                "daily_loss_limit": "Hit daily loss",
                "max_drawdown": "Hit max drawdown",
                "target_not_reached": "Didn't reach target",
                "too_many_days": "Too many days",
            }
            for cat, count in sorted(self.failure_breakdown.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {label.get(cat, cat):<22}{count:>6,}")
        return "\n".join(lines)


def run_rolling_evaluation(
    trades: list[Trade],
    rules: PropRules,
    window_trading_days: int,
    stride: int = 1,
    max_windows: int | None = 2000,
) -> RollingEvaluationResult:
    """Slides `rules`'s exact evaluation across every real historical
    starting trading day in `trades` (not resampled/reordered -- the
    genuine historical sequence, just starting from a different day each
    time) and returns the pass-rate distribution Owen described.

    window_trading_days: the evaluation's time limit -- e.g. 30 for a
    firm with a 30-trading-day evaluation window. A start day only
    counts as a valid window if there are ANY trades on/after it (an
    empty tail is not tested -- there's nothing to evaluate).

    stride: only start a window every `stride` trading days instead of
    every single one -- 1 (default) matches Owen's "every possible start
    bar" literally; a larger stride trades completeness for speed on a
    very long dataset.

    max_windows: hard cap on how many start days are actually tested
    (evenly spread across the full range, not just the first N) -- see
    the module docstring's performance note. None disables the cap.
    """
    if not trades:
        raise ValueError("Cannot run a rolling evaluation with zero trades.")
    if window_trading_days <= 0:
        raise ValueError("window_trading_days must be positive.")

    dates = [pd.Timestamp(t.entry_time) for t in trades]
    day_structure = precompute_day_structure(dates)
    n_days = day_structure.n_days
    if n_days <= 1:
        raise ValueError("Not enough distinct trading days to run a rolling evaluation.")

    # Group trade indices by day once, so slicing a tail from day `s`
    # onward is an O(1) index lookup rather than a per-window scan.
    first_trade_index_of_day: list[int] = [0] * n_days
    seen = set()
    for i, day_idx in enumerate(day_structure.day_index_per_trade):
        if day_idx not in seen:
            first_trade_index_of_day[day_idx] = i
            seen.add(day_idx)

    candidate_starts = list(range(0, n_days - 1, max(1, stride)))
    if max_windows is not None and len(candidate_starts) > max_windows:
        # Evenly spread the sampled starts across the full range instead
        # of just the earliest ones, so the result still reflects the
        # WHOLE dataset's history, not only its first portion.
        step = len(candidate_starts) / max_windows
        candidate_starts = [candidate_starts[int(i * step)] for i in range(max_windows)]

    outcomes: list[WindowOutcome] = []
    for start_day in candidate_starts:
        start_idx = first_trade_index_of_day[start_day]
        tail_pnls = [t.pnl for t in trades[start_idx:]]
        tail_dates = dates[start_idx:]
        if not tail_pnls:
            continue
        result = simulate_account(tail_pnls, tail_dates, rules)

        passed = False
        failure_category: str | None = None
        if result.failed and result.failure_day_index is not None and result.failure_day_index < window_trading_days:
            failure_category = "daily_loss_limit" if result.failure_reason == "daily_loss_limit" else "max_drawdown"
        elif result.passed_evaluation and result.days_to_pass is not None and result.days_to_pass <= window_trading_days:
            passed = True
        else:
            # Neither a clean rule-failure nor a clean pass landed inside
            # the window -- the firm would have simply closed the
            # evaluation out at the deadline. Split "never really
            # working" from "was on track, just slow" using whether the
            # account was net positive at all by the window's close (a
            # heuristic, not a firm-defined rule -- see module docstring).
            was_making_progress = result.final_balance > rules.account_size
            failure_category = "too_many_days" if was_making_progress else "target_not_reached"

        outcomes.append(WindowOutcome(
            start_day_index=start_day,
            start_date=str(day_structure.day_dates[start_day].date()),
            passed=passed,
            failure_category=failure_category,
            days_to_pass=result.days_to_pass if passed else None,
            reached_first_payout=passed and result.first_payout_day_index is not None,
        ))

    n_windows = len(outcomes)
    n_passed = sum(1 for o in outcomes if o.passed)
    n_failed = n_windows - n_passed
    pass_rate = (n_passed / n_windows * 100.0) if n_windows else 0.0
    first_payout_rate = (sum(1 for o in outcomes if o.reached_first_payout) / n_windows * 100.0) if n_windows else 0.0
    days_list = sorted(o.days_to_pass for o in outcomes if o.days_to_pass is not None)
    median_days = days_list[len(days_list) // 2] if days_list else None
    failure_breakdown = dict(Counter(o.failure_category for o in outcomes if o.failure_category))

    # Best/worst starting MONTH -- grouped by the calendar month each
    # window started in, requiring at least a handful of samples in that
    # month so one lucky/unlucky window doesn't crown an entire month.
    by_month: dict[str, list[bool]] = {}
    for o in outcomes:
        month = o.start_date[:7]
        by_month.setdefault(month, []).append(o.passed)
    month_rates = {
        m: sum(flags) / len(flags) for m, flags in by_month.items() if len(flags) >= min(3, max(1, n_windows // 20))
    }
    worst_period = min(month_rates, key=month_rates.get) if month_rates else None
    best_period = max(month_rates, key=month_rates.get) if month_rates else None

    return RollingEvaluationResult(
        window_trading_days=window_trading_days, n_windows=n_windows, n_passed=n_passed, n_failed=n_failed,
        pass_rate_pct=pass_rate, first_payout_rate_pct=first_payout_rate, median_days_to_pass=median_days,
        worst_starting_period=worst_period, best_starting_period=best_period,
        failure_breakdown=failure_breakdown, windows=outcomes,
    )
