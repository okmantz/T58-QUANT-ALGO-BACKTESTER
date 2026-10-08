"""Real-account check: do the backtester's trades match what a real account did?

The accuracy plan's only test that proves the MODEL (rather than the code): run the same
strategy over the same ~20 sessions the trader actually traded, then compare.

    compare_trade_lists(real, simulated, ...)  ->  RealCheckReport

`real` / `simulated` are DataFrames (or CSV paths) with at least: entry_time, direction
(or side: long/short/buy/sell/1/-1), entry_price, exit_price, pnl. Optional: size/qty.

Matching: a real trade matches the nearest unused simulated trade of the same direction
whose entry is within `time_tolerance` (default 2 bars' worth, 10 minutes). For matched
pairs the report gives entry/exit price differences (in price units and dollars/contract
when point value is known), pnl difference vs `cost_tolerance_dollars`, and daily P&L
agreement. The verdict is 'agrees within costs' only if >= `min_match_rate` of real trades
matched AND the median |pnl difference| is within the cost tolerance AND daily P&L differs
by no more than the tolerance on at least `min_daily_agreement` of days.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _norm_dir(v) -> int:
    s = str(v).strip().lower()
    if s in ("1", "1.0", "long", "buy", "b", "l"):
        return 1
    if s in ("-1", "-1.0", "short", "sell", "s"):
        return -1
    raise ValueError(f"unrecognised direction {v!r}")


def _load(x) -> pd.DataFrame:
    df = pd.read_csv(x) if isinstance(x, (str, bytes)) or hasattr(x, "read") else x.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "direction" not in df.columns and "side" in df.columns:
        df["direction"] = df["side"]
    for need in ("entry_time", "direction", "entry_price", "exit_price", "pnl"):
        if need not in df.columns:
            raise ValueError(f"trade list is missing column {need!r}")
    out = df.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True).dt.tz_convert(None)
    if "exit_time" in out.columns:
        out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True).dt.tz_convert(None)
    out["direction"] = out["direction"].map(_norm_dir)
    return out.sort_values("entry_time").reset_index(drop=True)


@dataclass
class RealCheckReport:
    n_real: int
    n_sim: int
    n_matched: int
    match_rate: float
    median_abs_pnl_diff: float | None
    median_entry_diff: float | None
    median_exit_diff: float | None
    daily_agreement: float | None
    unmatched_real: list = field(default_factory=list)
    extra_sim: list = field(default_factory=list)
    pairs: list = field(default_factory=list)
    agrees: bool = False
    notes: list = field(default_factory=list)

    def render(self) -> str:
        f = lambda v, p="{:.2f}": "n/a" if v is None else p.format(v)  # noqa: E731
        lines = [
            f"REAL-ACCOUNT CHECK: {'AGREES WITHIN COSTS' if self.agrees else 'DOES NOT AGREE'}",
            f"  real trades {self.n_real}   simulated {self.n_sim}   matched {self.n_matched} ({self.match_rate * 100:.0f}%)",
            f"  median |pnl diff| ${f(self.median_abs_pnl_diff)}   median entry diff {f(self.median_entry_diff)}   median exit diff {f(self.median_exit_diff)}",
            f"  days where daily P&L agrees within tolerance: {f(None if self.daily_agreement is None else self.daily_agreement * 100, '{:.0f}')}%",
        ]
        if self.unmatched_real:
            lines.append(f"  real trades the backtest did not take: {len(self.unmatched_real)} (first: {self.unmatched_real[0]})")
        if self.extra_sim:
            lines.append(f"  backtest trades you did not take: {len(self.extra_sim)} (first: {self.extra_sim[0]})")
        lines += [f"  NOTE: {n}" for n in self.notes]
        return "\n".join(lines)


def compare_trade_lists(
    real, simulated, *, time_tolerance: str | pd.Timedelta = "10min", cost_tolerance_dollars: float = 60.0,
    min_match_rate: float = 0.8, min_daily_agreement: float = 0.8,
) -> RealCheckReport:
    r, s = _load(real), _load(simulated)
    tol = pd.Timedelta(time_tolerance)
    used: set[int] = set()
    pairs, unmatched = [], []
    for i, row in r.iterrows():
        cand = s[(s["direction"] == row["direction"]) & ((s["entry_time"] - row["entry_time"]).abs() <= tol)]
        cand = cand[~cand.index.isin(used)]
        if cand.empty:
            unmatched.append(f"{row['entry_time']} {'long' if row['direction'] == 1 else 'short'}")
            continue
        j = (cand["entry_time"] - row["entry_time"]).abs().idxmin()
        used.add(j)
        sj = s.loc[j]
        pairs.append(dict(
            real_entry=str(row["entry_time"]), sim_entry=str(sj["entry_time"]),
            entry_diff=float(sj["entry_price"] - row["entry_price"]) * row["direction"],
            exit_diff=float(sj["exit_price"] - row["exit_price"]) * row["direction"],
            pnl_diff=float(sj["pnl"] - row["pnl"]), real_pnl=float(row["pnl"]), sim_pnl=float(sj["pnl"]),
        ))
    extra = [f"{s.loc[j, 'entry_time']}" for j in s.index if j not in used]
    n_m = len(pairs)
    match_rate = n_m / len(r) if len(r) else 0.0
    med = lambda k: float(np.median([abs(p[k]) for p in pairs])) if pairs else None  # noqa: E731
    # daily P&L (by exit date when available, else entry date)
    def daily(df):
        d = (df["exit_time"] if "exit_time" in df.columns else df["entry_time"]).dt.normalize()
        return df.groupby(d)["pnl"].sum()
    dr, ds = daily(r), daily(s)
    days = dr.index.union(ds.index)
    diff = (dr.reindex(days).fillna(0.0) - ds.reindex(days).fillna(0.0)).abs()
    daily_agree = float((diff <= cost_tolerance_dollars * max(1, len(r) / max(1, len(days)))).mean()) if len(days) else None
    mp = med("pnl_diff")
    agrees = bool(match_rate >= min_match_rate and mp is not None and mp <= cost_tolerance_dollars
                  and daily_agree is not None and daily_agree >= min_daily_agreement)
    notes = []
    if len(r) < 20:
        notes.append(f"only {len(r)} real trades: agreement on so few trades is weak evidence.")
    if match_rate < min_match_rate:
        notes.append("low match rate: signals, session times, or the data feed differ between the account and the backtest.")
    return RealCheckReport(len(r), len(s), n_m, match_rate, mp, med("entry_diff"), med("exit_diff"), daily_agree,
                           unmatched, extra, pairs, agrees, notes)
