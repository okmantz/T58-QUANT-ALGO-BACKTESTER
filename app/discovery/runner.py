"""
Run a rule spec on data with the REAL engine and the same sizing/cost model as
every other mode (run_execution for per-bar signals, resting_orders for zone
ideas). Returns trades plus metrics in R (pnl / worst-case risk at the stop) so
results are comparable across instruments and timeframes.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.backtest.execution import Trade, run_execution
from app.backtest.resting_orders import simulate_resting_orders
from app.discovery.rule_spec import build_plan, validate_spec


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLCV resample on the timestamp column (labels = bar OPEN time, closed left)."""
    d = df.set_index(pd.to_datetime(df["timestamp"]))
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in d.columns:
        agg["volume"] = "sum"
    out = d.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open", "close"])
    return out.reset_index().rename(columns={out.index.name or "index": "timestamp"})


def run_spec(df: pd.DataFrame, spec: dict, risk) -> list[Trade]:
    plan = build_plan(df, spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if plan.mode == "resting":
            return simulate_resting_orders(df, plan.orders, risk)
        trades, _ = run_execution(
            df, plan.signals, risk, stop_loss_pips=None, take_profit_pips=None,
            stop_loss_distance=plan.stop_loss_distance, take_profit_distance=plan.take_profit_distance,
        )
    return trades


def trade_r(t: Trade) -> float | None:
    rs = getattr(t, "risk_at_stop_dollars", None)
    if rs:
        return float(t.pnl) / float(rs)
    if t.initial_risk and t.size:
        return float(t.pnl) / (float(t.initial_risk) * float(t.size))
    return None


@dataclass
class Metrics:
    n: int
    net: float
    mean_r: float
    win_rate: float
    profit_factor: float
    sharpe_per_trade: float
    total_r: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def metrics(trades: list[Trade]) -> Metrics:
    rs = np.array([r for r in (trade_r(t) for t in trades) if r is not None], dtype=float)
    pn = np.array([t.pnl for t in trades], dtype=float)
    if len(rs) == 0:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    gp, gl = pn[pn > 0].sum(), -pn[pn < 0].sum()
    sd = rs.std(ddof=1) if len(rs) > 1 else 0.0
    return Metrics(
        n=len(rs), net=float(pn.sum()), mean_r=float(rs.mean()), win_rate=float((pn > 0).mean()),
        profit_factor=float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0),
        sharpe_per_trade=float(rs.mean() / sd) if sd > 0 else 0.0, total_r=float(rs.sum()),
    )
