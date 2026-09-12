"""
Automatic re-tune trigger -- closes the loop from "Strategy Health/Drift
Monitor detected decay" to "here's a re-tuned candidate," instead of
requiring a person to notice a WARNING/CRITICAL flag on the Forward Test
tab and go run Quick Optimize by hand.

app.monitoring.strategy_health.check_strategy_health() already computes
whether a forward-tested strategy's realized return/drawdown/win-rate/
losing-streak has drifted outside its own walk-forward-predicted band.
This module is a thin layer on top of it: when the drift crosses a
configurable severity threshold, it calls the existing, unchanged
app.orchestration.quick_optimize.run_quick_optimize() (the walk-forward-
aware GA re-tuner already used everywhere else in the app) against the
SAME strategy and market data the forward test has been running
against, and returns the result alongside the health check that
triggered it. No re-tuning or drift-detection logic is reimplemented
here.

This never runs on a timer by itself -- it's a single check-then-maybe-
retune step a caller (a UI button, a script, or
app.orchestration.overnight_autopilot) invokes once, when invoked.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.backtest.risk import RiskConfig
from app.forward_test.journal import ForwardTestJournal
from app.monitoring.strategy_health import StrategyHealthResult, check_strategy_health
from app.monte_carlo.engine import MonteCarloResult
from app.orchestration.quick_optimize import QuickOptimizeConfig, QuickOptimizeResult, run_quick_optimize
from app.prop.simulator import PropRules
from app.strategy.base import Strategy

_SEVERITY_ORDER = {"ok": 0, "watch": 1, "warning": 2, "critical": 3}
VALID_THRESHOLDS = tuple(_SEVERITY_ORDER.keys())


@dataclass
class AutoRetuneOutcome:
    health: StrategyHealthResult
    triggered: bool
    trigger_reason: str | None = None
    retune_result: QuickOptimizeResult | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "health": self.health.to_dict(),
            "triggered": self.triggered,
            "trigger_reason": self.trigger_reason,
            "error": self.error,
            "retune_improved": self.retune_result.improved if self.retune_result else None,
        }


def maybe_trigger_retune(
    journal: ForwardTestJournal,
    session_id: int,
    strategy_label: str,
    predicted: MonteCarloResult,
    account_balance: float,
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    prop_rules: PropRules,
    severity_threshold: str = "warning",
    quick_optimize_cfg: QuickOptimizeConfig | None = None,
    progress_cb=None,
) -> AutoRetuneOutcome:
    """
    Runs check_strategy_health(), and if its overall_severity meets or
    exceeds `severity_threshold` ("watch" | "warning" | "critical"),
    immediately runs Quick Optimize against `strategy`/`df` and attaches
    the result. Never raises for a re-tune failure -- an optimizer error
    is recorded on `.error` so a caller (e.g. the overnight autopilot's
    summary report) can say "drift detected, re-tune attempt failed:
    ..." instead of the whole health check crashing over it.

    Raises app.monitoring.strategy_health.StrategyHealthError under the
    same conditions check_strategy_health() itself would (e.g. no closed
    trades yet) -- that part is not swallowed, since "the health check
    itself couldn't run" is a different, earlier failure than "the
    re-tune it triggered failed."
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    if severity_threshold not in VALID_THRESHOLDS:
        raise ValueError(f"severity_threshold must be one of {VALID_THRESHOLDS}, got {severity_threshold!r}")

    health = check_strategy_health(journal, session_id, strategy_label, predicted, account_balance)

    threshold_rank = _SEVERITY_ORDER[severity_threshold]
    triggered = _SEVERITY_ORDER[health.overall_severity] >= threshold_rank
    outcome = AutoRetuneOutcome(health=health, triggered=triggered)
    if not triggered:
        log(f"Strategy Health: '{health.overall_severity}' -- below the '{severity_threshold}' auto-retune threshold, no action taken.")
        return outcome

    worst = max(health.flags, key=lambda f: _SEVERITY_ORDER[f.severity]) if health.flags else None
    outcome.trigger_reason = (
        f"overall drift severity '{health.overall_severity}'"
        + (f" ({worst.metric}: {worst.message})" if worst else "")
    )
    log(f"Drift threshold crossed -- {outcome.trigger_reason}. Starting an automatic Quick Optimize re-tune...")
    try:
        outcome.retune_result = run_quick_optimize(
            df, strategy, risk, prop_rules, cfg=quick_optimize_cfg, progress_cb=progress_cb,
        )
        log(
            "Automatic re-tune finished: "
            + ("found an improved configuration." if outcome.retune_result.improved else "no improvement found.")
        )
    except Exception as exc:  # noqa: BLE001 -- a re-tune failure must never crash the health check that triggered it
        outcome.error = f"{type(exc).__name__}: {exc}"
        log(f"Automatic re-tune failed: {outcome.error}")
    return outcome
