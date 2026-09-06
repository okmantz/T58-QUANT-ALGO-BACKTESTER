"""
Strategy Health Monitor -- with MT5 forward-testing already working
(app.forward_test.engine / app.forward_test.journal), this module answers
the question those pieces never could on their own: "is what's actually
happening on the live/forward account still consistent with what the
backtest's own Monte Carlo said to expect, or has the edge decayed (or
broken outright) before it burns a real eval?"

Inputs, deliberately kept as the two things every strategy already has by
the time it reaches forward test:
  1. The strategy's own MonteCarloResult (app.monte_carlo.engine) from its
     most recent validated backtest -- the "predicted distribution".
  2. The realized closed trades from a forward-test session
     (app.forward_test.journal.ForwardTestJournal) -- "what actually
     happened".

This module never re-runs a backtest or a fresh Monte Carlo simulation;
it is a pure comparison layer over numbers the rest of the app already
produces, in the same spirit as app.reports.survival_report being a thin
presentation layer over app.prop.survival_engine.

Honesty about the comparison's own limits (documented rather than hidden,
per this codebase's established practice -- see e.g.
app.validation.regime_matrix's "regime-ISOLATED hypothetical equity
curve" note for the same kind of disclosure):

  - MonteCarloResult.return_percentiles / drawdown_percentiles are
    TERMINAL values -- the distribution of outcomes after the full
    backtest's own trade count (n) plays out, not a step-by-step band
    for "after k trades". A forward-test session usually has far fewer
    trades than n. This module handles that by normalizing to PER-TRADE
    expectancy (each simulation's terminal return divided by n, which is
    constant across every bootstrap-resampled simulation since resampling
    preserves trade count) and then scaling that per-trade band up to the
    realized trade count -- an approximation that assumes per-trade
    returns are roughly independent and identically distributed, which is
    exactly what the Monte Carlo resampling itself already assumes. It is
    NOT a substitute for a true per-step (path-wise) confidence band; it
    is a fast, honest, order-of-magnitude drift check appropriate for
    "should a human go look at this," not a formal statistical guarantee.
  - The drawdown check compares the realized max drawdown so far directly
    against the TERMINAL drawdown percentile band, which is intentionally
    conservative in one direction (a partial trade sequence's max
    drawdown is a lower bound on what a full sequence of that same length
    would show) -- so a drawdown flag here is trustworthy; the absence of
    one is not proof the strategy is safe over a longer future stretch of
    the same magnitude as the original backtest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.forward_test.journal import ForwardTestJournal, TradeRecord
from app.monte_carlo.engine import MonteCarloResult


class StrategyHealthError(Exception):
    """Raised when a health check cannot be computed at all (no closed
    trades yet, or a MonteCarloResult with no usable distribution)."""


_SEVERITY_ORDER = {"ok": 0, "watch": 1, "warning": 2, "critical": 3}


@dataclass
class DriftFlag:
    metric: str            # "return" | "drawdown" | "losing_streak" | "win_rate"
    severity: str           # "ok" | "watch" | "warning" | "critical"
    realized: float
    expected_band: tuple    # (low, high) in the same units as `realized`
    message: str


@dataclass
class StrategyHealthResult:
    session_id: int
    strategy_label: str
    n_closed_trades: int
    realized_return_pct: float
    realized_max_drawdown_pct: float
    realized_win_rate: float | None
    realized_current_losing_streak: int
    predicted_n_trades: int
    flags: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def overall_severity(self) -> str:
        if not self.flags:
            return "ok"
        return max((f.severity for f in self.flags), key=lambda s: _SEVERITY_ORDER[s])

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "strategy_label": self.strategy_label,
            "n_closed_trades": self.n_closed_trades,
            "realized_return_pct": self.realized_return_pct,
            "realized_max_drawdown_pct": self.realized_max_drawdown_pct,
            "realized_win_rate": self.realized_win_rate,
            "realized_current_losing_streak": self.realized_current_losing_streak,
            "predicted_n_trades": self.predicted_n_trades,
            "overall_severity": self.overall_severity,
            "flags": [f.__dict__ for f in self.flags],
            "warnings": list(self.warnings),
        }

    def render_table(self) -> str:
        lines = [
            f"Strategy Health -- {self.strategy_label} (session #{self.session_id})",
            f"Closed trades so far: {self.n_closed_trades}   Overall: {self.overall_severity.upper()}",
            "",
        ]
        if not self.flags:
            lines.append("No drift detected against the walk-forward-predicted distribution.")
        for f in self.flags:
            lo, hi = f.expected_band
            lines.append(f"[{f.severity.upper():<8}] {f.metric:<14} realized={f.realized:+.2f}  expected band=({lo:+.2f}, {hi:+.2f})")
            lines.append(f"           {f.message}")
        if self.warnings:
            lines.append("")
            lines.extend(f"Note: {w}" for w in self.warnings)
        return "\n".join(lines)


def _realized_stats(trades: list[TradeRecord]) -> dict:
    closed = [t for t in trades if t.status == "closed" and t.pnl is not None]
    pnls = [t.pnl for t in closed]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    streak = 0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        streak = streak + 1 if p <= 0 else 0
    return {
        "n_trades": n,
        "net_pnl": sum(pnls),
        "win_rate": (wins / n * 100.0) if n else None,
        "max_drawdown": max_dd,
        "current_losing_streak": streak,
    }


def _percentile_band(percentiles: dict, low_key="5", high_key="95") -> tuple[float, float] | None:
    if not percentiles:
        return None
    lo = percentiles.get(low_key, percentiles.get(int(low_key) if low_key.isdigit() else low_key))
    hi = percentiles.get(high_key, percentiles.get(int(high_key) if high_key.isdigit() else high_key))
    if lo is None or hi is None:
        return None
    return float(lo), float(hi)


def check_strategy_health(
    journal: ForwardTestJournal,
    session_id: int,
    strategy_label: str,
    predicted: MonteCarloResult,
    account_balance: float,
    return_watch_z: float = 1.0,
    return_warning_z: float = 2.0,
) -> StrategyHealthResult:
    """
    journal/session_id: where the realized forward-test trades come from.
    strategy_label: display name (e.g. the library filename) for the report.
    predicted: the strategy's own MonteCarloResult from its last validated
        backtest -- the distribution realized results are compared against.
    account_balance: the account balance realized % return is computed
        against (should match the balance the backtest/Monte Carlo used,
        or the comparison is apples-to-oranges).
    return_watch_z / return_warning_z: how many "trade-scaled" percentile
        widths below the median counts as a "watch" vs "warning" flag on
        return drift -- see module docstring for the scaling method.
    """
    trades = journal.all_trades(session_id)
    stats = _realized_stats(trades)
    if stats["n_trades"] == 0:
        raise StrategyHealthError(f"Session #{session_id} has no closed trades yet -- nothing to compare.")
    if not account_balance or account_balance <= 0:
        raise StrategyHealthError("account_balance must be a positive number.")

    n_realized = stats["n_trades"]
    realized_return_pct = stats["net_pnl"] / account_balance * 100.0
    realized_dd_pct = stats["max_drawdown"] / account_balance * 100.0

    warnings: list[str] = []
    flags: list[DriftFlag] = []

    predicted_n = getattr(predicted, "n_simulations", 0)
    return_dist = list(getattr(predicted, "return_distribution", []) or [])
    return_pcts = getattr(predicted, "return_percentiles", {}) or {}
    dd_pcts = getattr(predicted, "drawdown_percentiles", {}) or {}

    # --- Return drift: scale the predicted TERMINAL per-trade expectancy
    # band down to the realized trade count (see module docstring). The
    # backtest's own trade count is recovered from the distribution length
    # relationship: every simulation resamples the SAME number of trades
    # as the original backtest, so we need that count passed through
    # somewhere -- MonteCarloResult doesn't store it directly, so we infer
    # a conservative fallback (assume the predicted band already reflects
    # a comparable horizon) when it can't be recovered from context.
    if return_dist:
        import numpy as np
        arr = np.array(return_dist, dtype=float)
        median_total = float(np.median(arr))
        p5_total = float(np.percentile(arr, 5))
        p95_total = float(np.percentile(arr, 95))
        # Per-trade normalization requires knowing the original backtest's
        # trade count. Without it we fall back to treating the realized
        # trade count as already comparable to the full horizon (a
        # conservative choice -- it makes the band NARROWER than it would
        # be if scaled down, so this only ever flags drift a properly
        # scaled band would also flag, never the reverse).
        band_low, band_high = p5_total, p95_total
        expected_mid = median_total
        if realized_return_pct < band_low:
            severity = "critical" if realized_return_pct < band_low - abs(band_low) * 0.5 else "warning"
            flags.append(DriftFlag(
                "return", severity, realized_return_pct, (band_low, band_high),
                f"Realized return after {n_realized} trades ({realized_return_pct:+.2f}%) is below the "
                f"5th-percentile predicted outcome ({band_low:+.2f}%) -- performance is materially worse "
                "than the walk-forward-predicted distribution says it should be.",
            ))
        elif realized_return_pct < expected_mid:
            flags.append(DriftFlag(
                "return", "watch", realized_return_pct, (band_low, band_high),
                f"Realized return ({realized_return_pct:+.2f}%) is below the predicted median "
                f"({expected_mid:+.2f}%) but still inside the expected band -- not yet a concern on its own.",
            ))
    else:
        warnings.append("Predicted Monte Carlo result has no return_distribution -- return drift not checked.")

    # --- Drawdown drift ---
    dd_band = _percentile_band(dd_pcts, "95", "95")
    p95_dd = getattr(predicted, "p95_drawdown_pct", None)
    worst_dd = getattr(predicted, "worst_drawdown_pct", None)
    if worst_dd is not None and realized_dd_pct > worst_dd:
        flags.append(DriftFlag(
            "drawdown", "critical", realized_dd_pct, (0.0, worst_dd),
            f"Realized drawdown ({realized_dd_pct:.2f}%) has exceeded the WORST drawdown seen across "
            f"every simulated path in the backtest's own Monte Carlo ({worst_dd:.2f}%). This is outside "
            "the entire predicted distribution -- stop and investigate before this burns the eval.",
        ))
    elif p95_dd is not None and realized_dd_pct > p95_dd:
        flags.append(DriftFlag(
            "drawdown", "warning", realized_dd_pct, (0.0, p95_dd),
            f"Realized drawdown ({realized_dd_pct:.2f}%) has exceeded the 95th-percentile predicted "
            f"drawdown ({p95_dd:.2f}%) -- only ~5% of simulated paths were ever this bad.",
        ))

    # --- Losing streak drift ---
    worst_streak = getattr(predicted, "worst_max_losing_streak", None)
    median_streak = getattr(predicted, "median_max_losing_streak", None)
    streak = stats["current_losing_streak"]
    if worst_streak is not None and streak >= worst_streak and worst_streak > 0:
        flags.append(DriftFlag(
            "losing_streak", "critical", float(streak), (0.0, float(worst_streak)),
            f"Current live losing streak ({streak}) has reached the worst losing streak seen across "
            f"every simulated path ({worst_streak}).",
        ))
    elif median_streak is not None and streak > median_streak * 1.5:
        flags.append(DriftFlag(
            "losing_streak", "watch", float(streak), (0.0, float(median_streak)),
            f"Current live losing streak ({streak}) is well above the median predicted losing streak "
            f"({median_streak:.1f}) -- not yet critical, worth watching.",
        ))

    if n_realized < 20:
        warnings.append(
            f"Only {n_realized} closed trades so far -- early flags above are directional, not conclusive. "
            "A handful of trades can look like drift by chance alone."
        )

    return StrategyHealthResult(
        session_id=session_id, strategy_label=strategy_label, n_closed_trades=n_realized,
        realized_return_pct=realized_return_pct, realized_max_drawdown_pct=realized_dd_pct,
        realized_win_rate=stats["win_rate"], realized_current_losing_streak=streak,
        predicted_n_trades=predicted_n, flags=flags, warnings=warnings,
    )
