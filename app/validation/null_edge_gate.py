"""Null-edge gate: does the entry logic beat RANDOM entry timing? (2026-10 audit, build item 4)

Every strategy must pass this before a challenge fee is spent. The test keeps everything about the strategy EXCEPT its
timing signal: same number of trades, same long/short mix, same exits (stop/target distances), same sizing and costs,
typical hold time -- but entry bars drawn at random, ``n_seeds`` times through the real engine. The p-value is the share of
random-timing runs that did at least as well as the real strategy (``random_entry_distribution``).

    result = run_null_edge_gate(df, strategy, risk, n_seeds=200)
    result.passed      # p <= p_max (default 0.10) AND enough trades for the test to mean anything
    result.render()    # human-readable block for reports

A strategy whose entries are indistinguishable from chance has no edge to find; paying a fee to learn that is the
cheapest mistake this repository can now refuse to let you make. ``passed`` is False (never an exception) whenever the
test could not run -- no evidence is not a pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_TRADES_FOR_NULL = 20


@dataclass
class NullEdgeResult:
    passed: bool
    p_value: float | None
    n_seeds: int
    n_trades: int
    observed: float | None
    null_mean: float | None
    null_p95: float | None
    metric: str
    p_max: float
    reason: str = ""
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def render(self) -> str:
        if self.p_value is None:
            return f"NULL-EDGE GATE: NOT RUN ({self.reason}) -> treated as FAILED"
        return (
            f"NULL-EDGE GATE: {'PASSED' if self.passed else 'FAILED'}  p = {self.p_value:.3f} (limit {self.p_max:.2f}) over {self.n_seeds} "
            f"random-timing runs; {self.metric}: observed {self.observed:,.0f} vs null mean {self.null_mean:,.0f} / p95 {self.null_p95:,.0f}; "
            f"{self.n_trades} trades." + (f" {self.reason}" if self.reason else "")
        )


def run_null_edge_gate(
    df: pd.DataFrame, strategy, risk, *, n_seeds: int = 200, p_max: float = 0.10, metric: str = "net_profit",
    seed: int = 0, adaptive_risk=None, max_workers: int | None = None, progress_cb=None,
) -> NullEdgeResult:
    from app.backtest.engine import run_backtest
    from app.data.timeframe_resample import prepare_timeframe_aligned_data
    from app.research.director import random_entry_distribution

    def fail(reason, **kw):
        return NullEdgeResult(False, None, n_seeds, kw.get("n_trades", 0), None, None, None, metric, p_max, reason)

    try:
        bt = run_backtest(df, strategy, risk, adaptive_risk=adaptive_risk, lookahead_check="skip")
    except Exception as exc:  # noqa: BLE001
        return fail(f"backtest failed: {type(exc).__name__}: {exc}")
    trades = bt.trades
    if len(trades) < MIN_TRADES_FOR_NULL:
        return fail(f"only {len(trades)} trades (< {MIN_TRADES_FOR_NULL}); a null distribution on so few is meaningless", n_trades=len(trades))
    try:
        d2, _ = prepare_timeframe_aligned_data(df, strategy)
        sr = strategy.generate(d2)
        if getattr(sr, "entry_orders", None) is not None:
            return fail("resting-order strategies are not supported by the random-entry null yet", n_trades=len(trades))
        longs = sum(1 for t in trades if t.direction == 1)
        holds = sorted((t.exit_time - t.entry_time).total_seconds() for t in trades)
        bar_s = max(1.0, float(pd.Series(pd.to_datetime(d2["timestamp"])).diff().dt.total_seconds().median()))
        hold_bars = max(1, int(round(holds[len(holds) // 2] / bar_s)))
        observed = float(getattr(bt.statistics, metric if metric != "net_profit" else "net_profit"))
        dist = random_entry_distribution(
            d2, risk, observed, len(trades), longs / len(trades), hold_bars,
            stop_loss_pips=sr.stop_loss_pips, take_profit_pips=sr.take_profit_pips,
            stop_loss_distance=sr.stop_loss_distance, take_profit_distance=sr.take_profit_distance,
            metric=metric, n_seeds=int(n_seeds), seed=seed, progress_cb=progress_cb, max_workers=max_workers,
        )
    except Exception as exc:  # noqa: BLE001 -- no evidence is not a pass
        return fail(f"null test failed: {type(exc).__name__}: {exc}", n_trades=len(trades))
    p = dist.get("p_value")
    if p is None:
        return fail("null distribution returned no p-value", n_trades=len(trades))
    arr = np.asarray(dist.get("null", []), dtype=float)
    return NullEdgeResult(
        passed=bool(p <= p_max), p_value=float(p), n_seeds=int(dist.get("n_seeds", len(arr))), n_trades=len(trades),
        observed=float(dist.get("observed", observed)), null_mean=float(dist.get("null_mean", np.nanmean(arr) if len(arr) else 0.0)),
        null_p95=float(dist.get("null_p95", np.nanpercentile(arr, 95) if len(arr) else 0.0)), metric=metric, p_max=p_max,
    )
