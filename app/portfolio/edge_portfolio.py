"""Edge-portfolio layer (report 2026-10-20, sections 4 and 6).

One edge at ~0.2-0.3R and ~100 trades a year takes years to prove (see app.validation.edge_stats). Several UNCORRELATED edges
prove faster and size better. This layer decides which validated edges may sit in the same book and what the BOOK has proven.

Admission (each rule is independent and reported):
  1. positive expectancy lower bound  -- day-block bootstrap LB of mean R > 0 (after the costs the trades already carry)
  2. daily P&L correlation < 0.5      -- against every already-admitted edge (greedy by LB, best first)
  3. random-entry null test beaten    -- caller supplies the flag from the candidate's own pipeline run (None = not run -> not admitted)
  4. per-instrument cap               -- at most ``max_per_instrument`` edges on one instrument (they share the same risk)

Portfolio-level gates (the book is judged as one strategy, with the same evidence standard):
  * pooled trade floor      >= ``min_book_trades`` (the 300-trade floor, portfolio version)
  * joint-day resampling    -- days are resampled TOGETHER across edges so cross-edge correlation and clustering are kept;
                               the lower bound of the book's mean daily R must be > 0
  * book drawdown           -- 95th percentile of the max drawdown (in R) of the joint-resampled equity paths
  * sequential-test planning -- months of forward trading the book needs at its blended edge (SPRT planning numbers)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.validation.edge_stats import day_block_bootstrap_mean, months_to_decide, sprt_boundaries


@dataclass
class EdgeCandidate:
    name: str
    instrument: str
    trades: list
    family: str = ""
    null_beaten: bool | None = None          # random-entry null test result from the candidate's own run


@dataclass
class Admission:
    name: str
    instrument: str
    admitted: bool
    n_trades: int
    mean_r: float | None
    lb_r: float | None
    reasons: list = field(default_factory=list)
    max_corr_with_book: float | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _r_by_day(trades) -> pd.Series:
    """Per-day summed R of a trade list (exit date). The unit that portfolios are resampled in."""
    from app.search.robustness import _r_and_days
    rs, days = _r_and_days(trades)
    if not rs:
        return pd.Series(dtype=float)
    d = pd.to_datetime(pd.Series(days)).dt.normalize()
    return pd.Series(rs, index=d.to_numpy()).groupby(level=0).sum().sort_index()


def _corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).sort_index().fillna(0.0)
    if len(df) < 20 or df.iloc[:, 0].std() == 0 or df.iloc[:, 1].std() == 0:
        return 0.0
    return float(np.corrcoef(df.iloc[:, 0], df.iloc[:, 1])[0, 1])


def admit_edges(cands: list[EdgeCandidate], *, max_corr: float = 0.5, max_per_instrument: int = 2, min_trades: int = 30) -> list[Admission]:
    """Greedy admission, best lower bound first. Returns one Admission per candidate (admitted or not, with reasons)."""
    scored = []
    for c in cands:
        from app.search.robustness import _r_and_days
        rs, days = _r_and_days(c.trades)
        bs = day_block_bootstrap_mean(rs, days, n_boot=1000, seed=11) if len(rs) >= 10 else {}
        scored.append((c, len(rs), bs.get("mean"), bs.get("lb")))
    scored.sort(key=lambda x: (x[3] is None, -(x[3] if x[3] is not None else -9e9)))
    book_days: dict[str, pd.Series] = {}
    per_inst: dict[str, int] = {}
    out: dict[str, Admission] = {}
    for c, n, mean, lb in scored:
        a = Admission(c.name, c.instrument, False, n, mean, lb)
        if n < min_trades:
            a.reasons.append(f"{n} trades < {min_trades}")
        if lb is None or lb <= 0:
            a.reasons.append("expectancy lower bound is not above zero")
        if c.null_beaten is not True:
            a.reasons.append("random-entry null test not beaten" if c.null_beaten is False else "random-entry null test not run")
        if per_inst.get(c.instrument, 0) >= max_per_instrument:
            a.reasons.append(f"instrument cap reached ({max_per_instrument} edges on {c.instrument})")
        mine = _r_by_day(c.trades)
        worst = max((_corr(mine, s) for s in book_days.values()), default=None)
        a.max_corr_with_book = worst
        if worst is not None and worst >= max_corr:
            a.reasons.append(f"daily P&L correlation {worst:.2f} >= {max_corr:.2f} with an admitted edge")
        if not a.reasons:
            a.admitted = True
            book_days[c.name] = mine
            per_inst[c.instrument] = per_inst.get(c.instrument, 0) + 1
        out[c.name] = a
    return [out[c.name] for c in cands]


def joint_day_resample(day_series: dict[str, pd.Series], weights: dict[str, float] | None = None, n_boot: int = 2000, seed: int = 21,
                       block_days: int = 5) -> dict:
    """Resample whole DAYS together across edges (moving blocks of ``block_days``) and summarise the book."""
    if not day_series:
        return {"n_days": 0}
    df = pd.concat(day_series, axis=1).sort_index().fillna(0.0)
    w = np.array([(weights or {}).get(c, 1.0) for c in df.columns], dtype=float)
    daily = df.to_numpy(dtype=float) @ w
    n = len(daily)
    if n < 30:
        return {"n_days": n}
    rng = np.random.default_rng(seed)
    b = max(1, min(int(block_days), n))
    n_blocks = int(np.ceil(n / b))
    means = np.empty(n_boot)
    mdd = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - b + 1, n_blocks)
        path = np.concatenate([daily[s:s + b] for s in starts])[:n]
        means[i] = path.mean()
        eq = np.cumsum(path)
        mdd[i] = float((np.maximum.accumulate(eq) - eq).max())
    return {"n_days": n, "mean_daily_r": float(daily.mean()), "lb_daily_r": float(np.quantile(means, 0.05)),
            "ub_daily_r": float(np.quantile(means, 0.95)), "p_book_positive": float((means > 0).mean()),
            "mdd_r_p50": float(np.quantile(mdd, 0.5)), "mdd_r_p95": float(np.quantile(mdd, 0.95)),
            "daily_sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else None}


@dataclass
class BookVerdict:
    admitted: list
    rejected: list
    n_trades: int
    resample: dict
    stage: str
    reasons: list = field(default_factory=list)
    sprt: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["admitted"] = [a if isinstance(a, dict) else a.to_dict() for a in self.admitted]
        d["rejected"] = [a if isinstance(a, dict) else a.to_dict() for a in self.rejected]
        return d

    def render(self) -> str:
        lines = [f"EDGE PORTFOLIO: {self.stage}  |  {len(self.admitted)} edge(s) admitted, {len(self.rejected)} rejected, {self.n_trades} pooled trades"]
        for a in self.admitted:
            lines.append(f"  + {a.name} ({a.instrument}): mean R {a.mean_r:+.3f}, LB {a.lb_r:+.3f}, n={a.n_trades}")
        for a in self.rejected:
            lines.append(f"  - {a.name} ({a.instrument}): " + "; ".join(a.reasons))
        lines += [f"  * {r}" for r in self.reasons]
        return "\n".join(lines)


def evaluate_book(cands: list[EdgeCandidate], *, max_corr: float = 0.5, max_per_instrument: int = 2, min_book_trades: int = 300,
                  weights: dict[str, float] | None = None, n_boot: int = 2000, seed: int = 21) -> BookVerdict:
    adm = admit_edges(cands, max_corr=max_corr, max_per_instrument=max_per_instrument)
    ok = [a for a in adm if a.admitted]
    bad = [a for a in adm if not a.admitted]
    by_name = {c.name: c for c in cands}
    n_trades = sum(a.n_trades for a in ok)
    days = {a.name: _r_by_day(by_name[a.name].trades) for a in ok}
    rs = joint_day_resample(days, weights, n_boot=n_boot, seed=seed) if ok else {"n_days": 0}
    reasons: list[str] = []
    stage = "LEAD"
    if not ok:
        stage = "REJECT"
        reasons.append("no edge passed admission")
    else:
        if n_trades < min_book_trades:
            reasons.append(f"book has {n_trades} pooled trades, below the {min_book_trades}-trade portfolio floor")
        lb = rs.get("lb_daily_r")
        if lb is None:
            reasons.append("too few common days to resample the book")
        elif lb <= 0:
            reasons.append(f"the lower bound of the book's mean daily R is {lb:+.3f} (not above zero)")
        if not reasons:
            stage = "FORWARD-TEST CANDIDATE"
    sprt = {}
    lbs = [a.lb_r for a in ok if a.lb_r is not None and a.lb_r > 0]
    if ok and rs.get("n_days") and lbs:
        months_span = max(rs["n_days"] / 21.0, 1e-9)
        per_edge_tpm = n_trades / len(ok) / months_span
        h1 = float(np.mean(lbs))                        # H1 = the shrunk/lower-bound edge, not the in-sample mean
        bnd = sprt_boundaries(h1)
        sprt = {"h1_r": h1, "expected_trades_h1": bnd.expected_trades_h1, "expected_trades_h0": bnd.expected_trades_h0,
                "trades_per_month_per_edge": per_edge_tpm,
                "months_to_decide_h1": months_to_decide(bnd.expected_trades_h1, per_edge_tpm, len(ok)),
                "months_to_decide_single_edge": months_to_decide(bnd.expected_trades_h1, per_edge_tpm, 1)}
    return BookVerdict(ok, bad, n_trades, rs, stage, reasons, sprt)
