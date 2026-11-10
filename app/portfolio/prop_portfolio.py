"""Prop-evaluation portfolio construction (Phase 5): the goal is the highest P(pass<=30 calendar days) without breaking a rule,
not the most trades. Works on ``TradeLedger`` objects so any strategy / instrument / exit variant can be combined.

Pieces:
  * ``daily_pnl_matrix`` / ``correlation_matrix``  -- dependence between legs on exit-day P&L (the quantity the account feels).
  * ``apply_shared_risk``  -- ONE risk controller across all legs: concurrent-position cap, open-risk cap, shared daily loss stop.
    Causal: a trade is admitted using only information at its entry time (trades that exited by then, and positions still open).
  * ``regime_gate``        -- activate a leg only in the regimes whose label (looked up on the PRIOR day) is in an allowed set.
  * ``allocate_weights``   -- coarse grid over per-leg size multipliers, ranked by the 30-day pass lower bound.
Selecting weights/gates on the same data that is then evaluated is in-sample; the report says so, and the validation module is where
an out-of-sample read happens.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from app.prop.simulator import PropRules
from app.prop.thirty_day import ThirtyDayConfig, ThirtyDayResult, TradeLedger, evaluate_thirty_day, rank_key


def daily_pnl_matrix(ledgers: Mapping[str, TradeLedger]) -> pd.DataFrame:
    cols = {}
    for name, led in ledgers.items():
        if not len(led):
            cols[name] = pd.Series(dtype=float)
            continue
        d = pd.to_datetime(led.exit_time).normalize()
        cols[name] = pd.Series(led.pnl, index=d).groupby(level=0).sum()
    return pd.DataFrame(cols).fillna(0.0).sort_index()


def correlation_matrix(ledgers: Mapping[str, TradeLedger]) -> pd.DataFrame:
    m = daily_pnl_matrix(ledgers)
    if m.shape[0] < 3:
        return pd.DataFrame(np.nan, index=m.columns, columns=m.columns)
    return m.corr().fillna(0.0)


def diversification_ratio(ledgers: Mapping[str, TradeLedger]) -> float:
    """sum of leg daily-P&L stdevs / stdev of the combined daily P&L (1.0 = no diversification)."""
    m = daily_pnl_matrix(ledgers)
    tot = m.sum(axis=1).std()
    return float(m.std().sum() / tot) if tot and tot > 0 else 1.0


@dataclass
class SharedRiskStats:
    admitted: int = 0
    skipped_concurrency: int = 0
    skipped_open_risk: int = 0
    skipped_daily_stop: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def apply_shared_risk(ledger: TradeLedger, *, max_concurrent: int = 2, max_open_risk_usd: float | None = None,
                      daily_stop_usd: float | None = None) -> tuple:
    """Filter a merged ledger through one shared controller. Returns ``(filtered_ledger, SharedRiskStats)``."""
    stats = SharedRiskStats()
    n = len(ledger)
    if n == 0:
        return ledger, stats
    keep = np.zeros(n, bool)
    open_pos: list = []                      # (exit_time, risk)
    day_pnl: dict = {}
    for i in range(n):
        t0 = ledger.entry_time[i]
        open_pos = [(x, r) for x, r in open_pos if x > t0]
        day = pd.Timestamp(t0).normalize()
        realized = sum(float(ledger.pnl[j]) for j in range(i) if keep[j] and ledger.exit_time[j] <= t0 and
                       pd.Timestamp(ledger.exit_time[j]).normalize() == day)
        risk_i = float(ledger.risk[i]) if np.isfinite(ledger.risk[i]) else 0.0
        if len(open_pos) >= max_concurrent:
            stats.skipped_concurrency += 1
            continue
        if max_open_risk_usd is not None and sum(r for _, r in open_pos) + risk_i > max_open_risk_usd:
            stats.skipped_open_risk += 1
            continue
        if daily_stop_usd is not None and realized <= -abs(daily_stop_usd):
            stats.skipped_daily_stop += 1
            continue
        keep[i] = True
        stats.admitted += 1
        open_pos.append((ledger.exit_time[i], risk_i))
    return ledger.select(keep), stats


def regime_gate(ledger: TradeLedger, labels: pd.Series, allowed: Sequence) -> TradeLedger:
    """Keep trades whose entry-day regime label (the last label strictly BEFORE the entry day: causal) is in ``allowed``."""
    if not len(ledger):
        return ledger
    lab = labels.copy()
    lab.index = pd.to_datetime(lab.index).normalize()
    lab = lab.sort_index()
    days = pd.to_datetime(ledger.entry_time).normalize()
    idx = lab.index.searchsorted(days, side="left") - 1          # strictly before the entry day
    vals = np.array([lab.iloc[k] if k >= 0 else None for k in idx], dtype=object)
    return ledger.select(np.array([v in set(allowed) for v in vals], bool))


@dataclass
class PortfolioCandidate:
    weights: dict
    result: ThirtyDayResult
    shared_risk: dict
    n_trades: int

    def to_dict(self) -> dict:
        return {"weights": self.weights, "p_pass_30": self.result.p_pass_30, "lower_bound": self.result.lower_bound,
                "p_bust_30": self.result.p_bust_30, "median_days": self.result.median_days_to_pass, "n_trades": self.n_trades,
                "shared_risk": self.shared_risk}


@dataclass
class PortfolioReport:
    correlation: pd.DataFrame
    diversification_ratio: float
    candidates: list = field(default_factory=list)
    n_trials: int = 0
    notes: list = field(default_factory=list)

    def best(self):
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict:
        return {"correlation": self.correlation.round(3).to_dict(), "diversification_ratio": self.diversification_ratio,
                "n_trials": self.n_trials, "notes": self.notes, "ranked": [c.to_dict() for c in self.candidates]}


def allocate_weights(ledgers: Mapping[str, TradeLedger], rules: PropRules, *, grid: Sequence[float] = (0.0, 0.5, 1.0, 1.5),
                     max_concurrent: int = 2, max_open_risk_usd: float | None = None, daily_stop_usd: float | None = None,
                     cfg: ThirtyDayConfig | None = None, data_start=None, data_end=None) -> PortfolioReport:
    names = list(ledgers)
    rep = PortfolioReport(correlation=correlation_matrix(ledgers), diversification_ratio=diversification_ratio(ledgers))
    for w in itertools.product(grid, repeat=len(names)):
        if not any(w):
            continue
        parts = [ledgers[nm].scaled(wi) for nm, wi in zip(names, w) if wi > 0 and len(ledgers[nm])]
        merged = TradeLedger.merge(parts)
        filt, st = apply_shared_risk(merged, max_concurrent=max_concurrent, max_open_risk_usd=max_open_risk_usd, daily_stop_usd=daily_stop_usd)
        if not len(filt):
            continue
        res = evaluate_thirty_day(filt, rules, data_start=data_start, data_end=data_end, cfg=cfg or ThirtyDayConfig(), with_sensitivity=False)
        rep.candidates.append(PortfolioCandidate(dict(zip(names, w)), res, st.to_dict(), len(filt)))
        rep.n_trials += 1
    rep.candidates.sort(key=lambda c: rank_key(c.result))
    rep.notes.append(f"{rep.n_trials} weight vectors tried on the SAME data they are scored on: in-sample. Confirm out of sample before trusting.")
    return rep
