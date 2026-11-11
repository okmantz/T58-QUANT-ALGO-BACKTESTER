"""Exit research on FROZEN entries (Phase 4 of the 30-day prop initiative).

The entry list is an input and is never altered: every policy is replayed over the identical (bar, side, entry price, stop distance)
tuples, so a difference in outcome is caused by the exit and nothing else. Intrabar handling is deliberately pessimistic:

  * stop and target both inside one bar -> the STOP is assumed to fill first;
  * a bar that opens through the stop fills at the OPEN (gap), not at the stop price;
  * stop/trail/break-even changes learned from a bar's range take effect from the NEXT bar (no same-bar look-ahead);
  * every stop-type exit pays ``slippage_points`` extra; every trade pays ``cost_points`` round-turn friction.

Every policy tried is counted (``n_trials``) and reported with the ranking so a winner is read against the size of the search.
The ranking key is the same CI lower bound on P(pass<=30d) that the evaluator uses, never Sharpe or profit factor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from app.prop.simulator import PropRules
from app.prop.thirty_day import ThirtyDayConfig, ThirtyDayResult, TradeLedger, evaluate_thirty_day, rank_key


@dataclass(frozen=True)
class FrozenEntry:
    bar: int                  # index of the bar the position is filled on (fill price = ``price``)
    side: int                 # +1 long, -1 short
    price: float
    stop_dist: float          # initial stop distance in price points (> 0)
    # True (default): the fill is the OPEN of ``bar`` (next-bar-open entries), so the rest of that bar's range (its high/low after the
    # open) is live exposure and is replayed. False: the fill is at/after ``bar``'s close (``price`` = that close), so the first bar that
    # can hurt the position is ``bar + 1``. Before 2026-11-11 every entry was treated as the second kind, which skipped the entry bar's own
    # stop/target touches for open-filled entries.
    fills_at_open: bool = True


@dataclass(frozen=True)
class ExitPolicy:
    """kind: fixed | breakeven | atr_trail | partial | time | vol_adjusted | dynamic_target | regime.

    target_r: take-profit in initial-risk multiples (None = no target).  max_bars: time exit, default 78 bars (~one 5-minute RTH session: prop rules flatten before the close, so an exit that can hold for
    days is not a legal exit); None disables it and is only for tests.
    be_trigger_r / be_offset_r: move the stop to entry + offset after price reached +trigger R.
    trail_atr: trail at this many ATRs from the best price (starts after ``trail_start_r``).
    partial_r / partial_frac: bank ``frac`` of the position at +partial_r R, runner continues under the rest of the policy.
    vol_scale: scale stop and target by (ATR / ref_atr) clipped to [0.5, 2]. dynamic_target_atr: target = k * ATR at entry.
    regime_hi_vol_target_r: used when entry ATR is above the median (regime-dependent exit).
    """
    name: str
    kind: str = "fixed"
    target_r: float | None = 2.0
    max_bars: int | None = 78
    be_trigger_r: float | None = None
    be_offset_r: float = 0.0
    trail_atr: float | None = None
    trail_start_r: float = 0.0
    partial_r: float | None = None
    partial_frac: float = 0.5
    vol_scale: bool = False
    dynamic_target_atr: float | None = None
    regime_hi_vol_target_r: float | None = None


def atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """Causal ATR (value at bar i uses bars <= i)."""
    pc = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    out = pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()
    return out


@dataclass
class ExitResult:
    r: float                  # net R after cost, in units of the (possibly vol-scaled) initial stop
    bars: int
    mae_r: float              # <= 0
    mfe_r: float              # >= 0
    reason: str
    exit_bar: int


def simulate_exit(o, h, l, c, atr, e: FrozenEntry, p: ExitPolicy, *, cost_points: float = 0.0, slippage_points: float = 0.0,
                  ref_atr: float | None = None, median_atr: float | None = None) -> ExitResult:
    n = len(c)
    side = e.side
    entry = e.price
    stop_dist = e.stop_dist
    a0 = float(atr[min(e.bar, n - 1)])
    if p.vol_scale and ref_atr and a0 > 0:
        stop_dist = stop_dist * float(np.clip(a0 / ref_atr, 0.5, 2.0))
    stop_dist = max(stop_dist, 1e-9)
    tgt_r = p.target_r
    if p.dynamic_target_atr is not None and a0 > 0:
        tgt_r = p.dynamic_target_atr * a0 / stop_dist
    if p.regime_hi_vol_target_r is not None and median_atr and a0 > median_atr:
        tgt_r = p.regime_hi_vol_target_r
    stop = entry - side * stop_dist
    target = entry + side * tgt_r * stop_dist if tgt_r is not None else None
    best = entry
    banked = 0.0                   # R banked by partial exits (gross, before cost)
    remaining = 1.0
    partial_done = False
    be_done = False
    mae = 0.0
    mfe = 0.0
    last = min(n - 1, e.bar + (p.max_bars if p.max_bars is not None else n))
    exit_px, reason, exit_bar = None, "data_end", last
    first_bar = e.bar if e.fills_at_open else e.bar + 1
    for i in range(first_bar, last + 1):
        if side > 0:
            adverse, favour = l[i] - entry, h[i] - entry
        else:
            adverse, favour = entry - h[i], entry - l[i]
        # 1) stop first (pessimistic), using the stop as of the START of this bar
        gap_open = o[i]
        stopped = (l[i] <= stop) if side > 0 else (h[i] >= stop)
        if stopped:
            through = (gap_open <= stop) if side > 0 else (gap_open >= stop)
            fill = gap_open if through else stop
            exit_px = fill - side * slippage_points
            mae = min(mae, adverse / stop_dist)
            reason, exit_bar = "stop", i
            break
        mae = min(mae, adverse / stop_dist)
        mfe = max(mfe, favour / stop_dist)
        if target is not None:
            hit = (h[i] >= target) if side > 0 else (l[i] <= target)
            if hit:
                exit_px = target
                reason, exit_bar = "target", i
                break
        # 2) learn from this bar; changes apply to the next bar
        best = max(best, h[i]) if side > 0 else min(best, l[i])
        fav_r = side * (best - entry) / stop_dist
        if p.partial_r is not None and not partial_done and fav_r >= p.partial_r:
            banked += p.partial_frac * p.partial_r
            remaining -= p.partial_frac
            partial_done = True
        if p.be_trigger_r is not None and not be_done and fav_r >= p.be_trigger_r:
            new_stop = entry + side * p.be_offset_r * stop_dist
            stop = max(stop, new_stop) if side > 0 else min(stop, new_stop)
            be_done = True
        if p.trail_atr is not None and fav_r >= p.trail_start_r and atr[i] > 0:
            tr_stop = best - side * p.trail_atr * float(atr[i])
            stop = max(stop, tr_stop) if side > 0 else min(stop, tr_stop)
    if exit_px is None:
        exit_px = c[last]
        reason = "time" if (p.max_bars is not None and last == e.bar + p.max_bars) else "data_end"
    gross_r = banked + remaining * side * (exit_px - entry) / stop_dist
    net_r = gross_r - cost_points / stop_dist
    return ExitResult(float(net_r), int(exit_bar - e.bar), float(mae), float(mfe), reason, int(exit_bar))


def _ledger_from_results(df_ts, entries: Sequence[FrozenEntry], results: Sequence[ExitResult], stop_dists: Sequence[float],
                         risk_usd: float, label: str, instrument: str, cost_points: float) -> TradeLedger:
    ts = pd.to_datetime(df_ts)
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    et = np.array([ts[e.bar] for e in entries], dtype="datetime64[ns]")
    xt = np.array([ts[r.exit_bar] for r in results], dtype="datetime64[ns]")
    r = np.array([x.r for x in results], float)
    pnl = r * risk_usd
    mae = np.minimum(np.array([x.mae_r for x in results]) * risk_usd - cost_points / np.maximum(np.asarray(stop_dists), 1e-9) * risk_usd, pnl)
    mfe = np.maximum(np.array([x.mfe_r for x in results]) * risk_usd, pnl)
    n = len(entries)
    cost = np.array([cost_points / max(sd, 1e-9) * risk_usd for sd in stop_dists], float)
    return TradeLedger(et, xt, pnl, mae, mfe, cost, np.full(n, float(risk_usd)), np.array([e.side for e in entries], int),
                       np.array([label] * n, dtype=object), np.array([instrument] * n, dtype=object))


@dataclass
class ExitCandidate:
    policy: ExitPolicy
    result: ThirtyDayResult
    mean_r: float
    risk_usd: float

    def to_dict(self) -> dict:
        return {"policy": self.policy.name, "kind": self.policy.kind, "p_pass_30": self.result.p_pass_30,
                "lower_bound": self.result.lower_bound, "mean_r": self.mean_r, "risk_usd": self.risk_usd,
                "p_bust_30": self.result.p_bust_30, "median_days": self.result.median_days_to_pass}


@dataclass
class ExitLabReport:
    entries_frozen: int
    n_trials: int
    candidates: list = field(default_factory=list)       # sorted best first by rank_key
    notes: list = field(default_factory=list)

    def best(self) -> ExitCandidate | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict:
        return {"entries_frozen": self.entries_frozen, "n_trials": self.n_trials, "notes": self.notes,
                "ranked": [c.to_dict() for c in self.candidates]}


def default_policy_grid() -> list:
    g = [ExitPolicy("fixed_1R", target_r=1.0), ExitPolicy("fixed_2R", target_r=2.0), ExitPolicy("fixed_3R", target_r=3.0)]
    g += [ExitPolicy("be_1R_t2", "breakeven", target_r=2.0, be_trigger_r=1.0),
          ExitPolicy("be_1R_t3", "breakeven", target_r=3.0, be_trigger_r=1.0, be_offset_r=0.1)]
    g += [ExitPolicy("trail_2atr", "atr_trail", target_r=None, trail_atr=2.0, trail_start_r=0.5),
          ExitPolicy("trail_3atr_t4", "atr_trail", target_r=4.0, trail_atr=3.0, trail_start_r=1.0)]
    g += [ExitPolicy("partial_1R_run3", "partial", target_r=3.0, partial_r=1.0, partial_frac=0.5, be_trigger_r=1.0)]
    g += [ExitPolicy("time_60_t2", "time", target_r=2.0, max_bars=60), ExitPolicy("time_30_t1.5", "time", target_r=1.5, max_bars=30)]
    g += [ExitPolicy("vol_adj_t2", "vol_adjusted", target_r=2.0, vol_scale=True),
          ExitPolicy("dyn_target_3atr", "dynamic_target", target_r=None, dynamic_target_atr=3.0),
          ExitPolicy("regime_t1.5_hi2.5", "regime", target_r=1.5, regime_hi_vol_target_r=2.5)]
    return g


def run_exit_lab(df: pd.DataFrame, entries: Sequence[FrozenEntry], rules: PropRules, *, policies: Sequence[ExitPolicy] | None = None,
                 risk_usd: float = 500.0, cost_points: float = 0.0, slippage_points: float = 0.0, instrument: str = "",
                 cfg: ThirtyDayConfig | None = None, data_start=None, data_end=None, min_entries: int = 30, ledger_fn=None) -> ExitLabReport:
    """Replay the frozen ``entries`` under each policy and rank by the 30-day pass lower bound. Entries are never modified.

    ``ledger_fn(policy, entries) -> (TradeLedger, mean_r)`` lets the caller supply realistic whole-contract sizing / contract caps /
    instrument costs (the scanner's ``build_ledger``); without it a flat ``risk_usd`` per trade is used (fractional contracts: optimistic).
    """
    policies = list(policies) if policies is not None else default_policy_grid()
    entries = list(entries)
    rep = ExitLabReport(entries_frozen=len(entries), n_trials=len(policies))
    if len(entries) < min_entries:
        rep.notes.append(f"Only {len(entries)} frozen entries (< {min_entries}): exit comparisons would be noise; nothing ranked.")
        return rep
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = atr_series(h, l, c)
    ref_atr = float(np.nanmedian(atr))
    entry_atrs = np.array([atr[min(e.bar, len(atr) - 1)] for e in entries])
    med = float(np.nanmedian(entry_atrs))
    for p in policies:
        if ledger_fn is not None:
            led, mean_r = ledger_fn(p, entries)
            if not len(led):
                continue
            ev = evaluate_thirty_day(led, rules, data_start=data_start, data_end=data_end, cfg=cfg or ThirtyDayConfig(), with_sensitivity=False)
            rep.candidates.append(ExitCandidate(p, ev, float(mean_r), risk_usd))
            continue
        res, sds = [], []
        for e in entries:
            r = simulate_exit(o, h, l, c, atr, e, p, cost_points=cost_points, slippage_points=slippage_points, ref_atr=ref_atr, median_atr=med)
            res.append(r)
            a0 = float(atr[min(e.bar, len(atr) - 1)])
            sd = e.stop_dist * (float(np.clip(a0 / ref_atr, 0.5, 2.0)) if p.vol_scale and ref_atr and a0 > 0 else 1.0)
            sds.append(sd)
        led = _ledger_from_results(df["timestamp"], entries, res, sds, risk_usd, p.name, instrument, cost_points)
        ev = evaluate_thirty_day(led, rules, data_start=data_start, data_end=data_end, cfg=cfg or ThirtyDayConfig(), with_sensitivity=False)
        rep.candidates.append(ExitCandidate(p, ev, float(np.mean([x.r for x in res])), risk_usd))
    rep.candidates.sort(key=lambda cnd: rank_key(cnd.result))
    rep.notes.append(f"{len(policies)} exit policies were tried on {len(entries)} frozen entries; read the winner against that search size.")
    return rep
