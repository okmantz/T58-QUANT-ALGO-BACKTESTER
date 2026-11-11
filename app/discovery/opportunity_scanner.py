"""Frequency-aware opportunity scanner (Phase 3 of the 30-day prop initiative).

Question answered: across instruments, sessions and structural setups, which (family, parameters, instrument, session, exit) cells
make a +6%-in-30-days pass *mathematically plausible* -- enough trades per month at a positive NET edge -- and which of them survive
the real prop simulator on DISCOVERY data only?

Design rules (each guards a failure mode named in the audit):
  * entries use information up to the close of the signal bar and fill at the NEXT bar's open (no look-ahead);
  * every cell is costed (round-turn friction in points, stop slippage) and sized in WHOLE contracts against a fixed dollar risk, with
    a contract cap; a signal that cannot be sized is skipped and counted;
  * the scanner is handed discovery data only (``discovery_end``); it never sees development, walk-forward or sealed windows;
  * the number of cells tried (``n_trials``) is part of every report; cells are ranked by the dependence-aware 30-day pass LOWER bound
    and a feasibility upper bound, never by Sharpe / profit factor.
Not a strategy generator: it maps where opportunity frequency and edge coexist, and hands survivors to the exit lab and the
validation gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from app.prop.feasibility import screening_upper_bound
from app.prop.simulator import PropRules
from app.prop.thirty_day import ThirtyDayConfig, ThirtyDayResult, TradeLedger, evaluate_thirty_day, rank_key
from app.research.exit_lab import ExitPolicy, FrozenEntry, atr_series, simulate_exit

# micro-contract point value ($/point), round-turn friction in points (spread/slippage + commission), a conservative default table
INSTRUMENTS = {
    "NQ": {"micro_pv": 2.0, "cost_points": 1.0, "slip_points": 0.5},
    "ES": {"micro_pv": 5.0, "cost_points": 0.7, "slip_points": 0.25},
    "GC": {"micro_pv": 10.0, "cost_points": 0.3, "slip_points": 0.1},
    "CL": {"micro_pv": 100.0, "cost_points": 0.03, "slip_points": 0.01},
}
# A mini future is 10 micros (NQ/ES/GC/CL alike). ``PropRules.max_contracts`` (and every preset, e.g. Lucid 50K "4 mini / 40 micro") counts MINI
# contracts, so the micro-equivalent cap is ``max_contracts * MICROS_PER_MINI``. Before 2026-11-11 the scanner treated it as a micro count,
# i.e. it silently sized at one tenth of the allowed size.
MICROS_PER_MINI = 10
# New futures session begins 18:00 New York time; a trade must be flat before the firm's flatten time inside its own session.
SESSION_ROLL_NY_MIN = 18 * 60
SESSION_OPENS = {"ny_open": "09:30", "ny_pre": "08:30", "london": "03:00", "asia": "20:00"}


@dataclass(frozen=True)
class Cell:
    family: str
    instrument: str
    session: str
    params: tuple
    exit_name: str

    @property
    def key(self) -> str:
        return f"{self.instrument}|{self.family}|{self.session}|{dict(self.params)}|{self.exit_name}"


@dataclass
class ScanResult:
    cell: Cell
    n_signals: int
    n_skipped_sizing: int
    trades_per_30d: float
    win_rate: float
    payoff_r: float
    net_expectancy_r: float
    feasibility_upper_bound: float
    result: ThirtyDayResult | None

    def to_dict(self) -> dict:
        r = self.result
        return {"cell": self.cell.key, "n_signals": self.n_signals, "skipped_sizing": self.n_skipped_sizing,
                "trades_per_30d": round(self.trades_per_30d, 2), "win_rate": round(self.win_rate, 3), "payoff_r": round(self.payoff_r, 3),
                "net_expectancy_r": round(self.net_expectancy_r, 4), "feasibility_upper_bound": round(self.feasibility_upper_bound, 3),
                "p_pass_30": None if r is None else r.p_pass_30, "lower_bound": None if r is None else r.lower_bound,
                "p_bust_30": None if r is None else r.p_bust_30, "median_days": None if r is None else r.median_days_to_pass}


@dataclass
class ScanReport:
    n_trials: int
    discovery_end: str
    results: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    skipped_instruments: dict = field(default_factory=dict)

    def top(self, k: int = 10) -> list:
        return self.results[:k]

    def to_dict(self) -> dict:
        return {"n_trials": self.n_trials, "discovery_end": self.discovery_end, "notes": self.notes,
                "skipped_instruments": self.skipped_instruments, "ranked": [r.to_dict() for r in self.results]}


# ------------------------------------------------------------------------------------------------------------- bar preparation
def resample_ohlc(df: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
    """1-minute (or finer) bars -> ``minutes`` bars; timestamps are bar OPEN times, naive UTC in and out."""
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if d["timestamp"].dt.tz is not None:
        d["timestamp"] = d["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    g = d.set_index("timestamp").resample(f"{minutes}min", label="left", closed="left")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    out = out.dropna().reset_index()
    return out


def _with_ny_time(bars: pd.DataFrame) -> pd.DataFrame:
    ny = bars["timestamp"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    b = bars.copy()
    b["ny_date"] = ny.dt.date
    b["ny_min"] = ny.dt.hour * 60 + ny.dt.minute
    return b


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


# ------------------------------------------------------------------------------------------------------------- signal families
def _day_groups(b: pd.DataFrame):
    return b.groupby("ny_date", sort=True).indices


def signals_orb(b: pd.DataFrame, atr: np.ndarray, *, session: str, or_minutes: int, stop_atr_cap: float = 1.5) -> list:
    """First 5-min CLOSE beyond the opening range [open, open+or_minutes) after it completes; one signal per day; entry next bar open."""
    t0 = _hm(SESSION_OPENS[session])
    out = []
    for _, idx in _day_groups(b).items():
        m = b["ny_min"].to_numpy()[idx]
        in_or = idx[(m >= t0) & (m < t0 + or_minutes)]
        if len(in_or) < max(2, or_minutes // 5 - 1):
            continue
        hi, lo = b["high"].to_numpy()[in_or].max(), b["low"].to_numpy()[in_or].min()
        after = idx[(m >= t0 + or_minutes) & (m < t0 + or_minutes + 180)]
        cl = b["close"].to_numpy()
        for i in after:
            if i + 1 >= len(b):
                break
            side = 1 if cl[i] > hi else (-1 if cl[i] < lo else 0)
            if side:
                entry = float(b["open"].to_numpy()[i + 1])
                sd = (entry - lo) if side > 0 else (hi - entry)
                sd = float(min(max(sd, 0.25 * atr[i]), stop_atr_cap * atr[i]))
                out.append(FrozenEntry(i + 1, side, entry, sd))
                break
    return out


def signals_drive(b: pd.DataFrame, atr: np.ndarray, *, session: str, drive_minutes: int, k_atr: float, reverse: bool) -> list:
    """Opening drive: return over the first ``drive_minutes`` beyond ``k_atr`` x ATR -> continue (or fade, ``reverse``) at its end."""
    t0 = _hm(SESSION_OPENS[session])
    out = []
    for _, idx in _day_groups(b).items():
        m = b["ny_min"].to_numpy()[idx]
        first = idx[m >= t0]
        if len(first) == 0 or b["ny_min"].to_numpy()[first[0]] > t0 + 10:
            continue
        end_i = idx[(m >= t0 + drive_minutes)]
        if len(end_i) == 0 or end_i[0] + 1 >= len(b):
            continue
        j = end_i[0]
        op = float(b["open"].to_numpy()[first[0]])
        cl = float(b["close"].to_numpy()[j - 1])
        move = cl - op
        if abs(move) < k_atr * atr[j - 1]:
            continue
        side = (1 if move > 0 else -1) * (-1 if reverse else 1)
        entry = float(b["open"].to_numpy()[j])
        sd = float(max(0.8 * atr[j - 1] * math.sqrt(drive_minutes / 5.0) * 0.5, 0.5 * atr[j - 1]))
        out.append(FrozenEntry(int(j), side, entry, sd))
    return out


def signals_prior_day_break(b: pd.DataFrame, atr: np.ndarray, *, session: str) -> list:
    t0 = _hm(SESSION_OPENS[session])
    out = []
    dates = sorted(_day_groups(b).keys())
    groups = _day_groups(b)
    for prev, cur in zip(dates[:-1], dates[1:]):
        pidx = groups[prev]
        hi, lo = b["high"].to_numpy()[pidx].max(), b["low"].to_numpy()[pidx].min()
        idx = groups[cur]
        m = b["ny_min"].to_numpy()[idx]
        for i in idx[(m >= t0 + 5) & (m < t0 + 240)]:
            if i + 1 >= len(b):
                break
            c = b["close"].to_numpy()[i]
            side = 1 if c > hi else (-1 if c < lo else 0)
            if side:
                entry = float(b["open"].to_numpy()[i + 1])
                out.append(FrozenEntry(i + 1, side, entry, float(max(atr[i] * 1.0, 1e-6))))
                break
    return out


def signals_compression(b: pd.DataFrame, atr: np.ndarray, *, session: str, lookback: int, ratio: float) -> list:
    """Range of the last ``lookback`` bars < ``ratio`` x lookback x ATR-per-bar, then a close beyond that range; <=1 per day."""
    t0 = _hm(SESSION_OPENS[session])
    hi_a, lo_a, cl_a, op_a = (b[k].to_numpy() for k in ("high", "low", "close", "open"))
    out = []
    for _, idx in _day_groups(b).items():
        m = b["ny_min"].to_numpy()[idx]
        for i in idx[(m >= t0 + lookback * 5) & (m < t0 + 300)]:
            if i - lookback < 0 or i + 1 >= len(b):
                continue
            h, l = hi_a[i - lookback:i].max(), lo_a[i - lookback:i].min()
            if (h - l) < ratio * atr[i] * math.sqrt(lookback):
                side = 1 if cl_a[i] > h else (-1 if cl_a[i] < l else 0)
                if side:
                    entry = float(op_a[i + 1])
                    out.append(FrozenEntry(i + 1, side, entry, float(max(min(entry - l if side > 0 else h - entry, 1.5 * atr[i]), 0.4 * atr[i]))))
                    break
    return out


def signals_gap_fade(b: pd.DataFrame, atr: np.ndarray, *, session: str, k_atr: float) -> list:
    t0 = _hm(SESSION_OPENS[session])
    out = []
    dates = sorted(_day_groups(b).keys())
    groups = _day_groups(b)
    for prev, cur in zip(dates[:-1], dates[1:]):
        pidx = groups[prev]
        pc = float(b["close"].to_numpy()[pidx[-1]])
        idx = groups[cur]
        m = b["ny_min"].to_numpy()[idx]
        op_idx = idx[m >= t0]
        if len(op_idx) < 2 or op_idx[0] + 1 >= len(b):
            continue
        j = op_idx[0]
        gap = float(b["open"].to_numpy()[j]) - pc
        if abs(gap) < k_atr * atr[j]:
            continue
        side = -1 if gap > 0 else 1
        entry = float(b["open"].to_numpy()[j + 1])
        out.append(FrozenEntry(int(j + 1), side, entry, float(max(0.6 * abs(gap), 0.5 * atr[j]))))
    return out


FAMILY_GRID = [
    ("orb", {"or_minutes": 15}), ("orb", {"or_minutes": 30}), ("orb", {"or_minutes": 60}),
    ("drive_cont", {"drive_minutes": 30, "k_atr": 1.0}), ("drive_cont", {"drive_minutes": 30, "k_atr": 2.0}),
    ("drive_rev", {"drive_minutes": 30, "k_atr": 2.0}),
    ("prior_day_break", {}),
    ("compression", {"lookback": 12, "ratio": 0.5}), ("compression", {"lookback": 12, "ratio": 0.7}),
    ("gap_fade", {"k_atr": 1.0}), ("gap_fade", {"k_atr": 2.0}),
]
EXIT_GRID = [ExitPolicy("t1R", target_r=1.0, max_bars=60), ExitPolicy("t1.5R", target_r=1.5, max_bars=60),
             ExitPolicy("t2R_be", "breakeven", target_r=2.0, max_bars=60, be_trigger_r=1.0)]


def _signals(family: str, b: pd.DataFrame, atr: np.ndarray, session: str, params: dict) -> list:
    if family == "orb":
        return signals_orb(b, atr, session=session, **params)
    if family == "drive_cont":
        return signals_drive(b, atr, session=session, reverse=False, **params)
    if family == "drive_rev":
        return signals_drive(b, atr, session=session, reverse=True, **params)
    if family == "prior_day_break":
        return signals_prior_day_break(b, atr, session=session)
    if family == "compression":
        return signals_compression(b, atr, session=session, **params)
    if family == "gap_fade":
        return signals_gap_fade(b, atr, session=session, **params)
    raise ValueError(f"unknown family {family!r}")


def prepare_bars(raw: pd.DataFrame, *, start=None, end=None, minutes: int = 5) -> tuple:
    """``(bars, atr, ny_frame)`` for one instrument, optionally clipped to ``[start, end)``; shared by the scanner and the pipeline."""
    bars = resample_ohlc(raw, minutes)
    if end is not None:
        bars = bars[bars["timestamp"] < pd.Timestamp(end)]
    if start is not None:
        bars = bars[bars["timestamp"] >= pd.Timestamp(start)]
    bars = bars.reset_index(drop=True)
    atr = atr_series(bars["high"].to_numpy(float), bars["low"].to_numpy(float), bars["close"].to_numpy(float))
    return bars, atr, _with_ny_time(bars)


def cell_entries(cell: "Cell", nb: pd.DataFrame, atr: np.ndarray) -> list:
    return _signals(cell.family, nb, atr, cell.session, dict(cell.params))


def _flatten_ny_minute(flatten_time_ct: str | None) -> int | None:
    """Firm flatten time ("HH:MM" America/Chicago) -> minutes after midnight in New York (Chicago + 1h)."""
    if not flatten_time_ct:
        return None
    return _hm(str(flatten_time_ct)) + 60


def _flatten_limits(bars: pd.DataFrame, flatten_ny_min: int) -> tuple:
    """``(allowed, run_end)`` per bar: ``allowed[j]`` = bar j opens before the flatten time of its own futures session (or after the new
    session's 18:00 roll); ``run_end[j]`` = the last bar of the unbroken allowed run containing j (the latest bar a position opened on j
    may still be held through)."""
    ny = pd.DatetimeIndex(bars["timestamp"]).tz_localize("UTC").tz_convert("America/New_York")
    mins = np.asarray(ny.hour * 60 + ny.minute, dtype=np.int64)
    dates = np.asarray([d.toordinal() for d in ny.date], dtype=np.int64)
    sess = dates + (mins >= SESSION_ROLL_NY_MIN).astype(np.int64)
    allowed = (mins < flatten_ny_min) | (mins >= SESSION_ROLL_NY_MIN)
    n = len(bars)
    if n == 0:
        return allowed, np.zeros(0, dtype=np.int64)
    brk = np.concatenate([[0], ((sess[1:] != sess[:-1]) | (allowed[1:] != allowed[:-1])).astype(np.int64)])
    run_id = np.cumsum(brk)
    ends = np.concatenate([np.flatnonzero(np.diff(run_id) != 0), [n - 1]])
    return allowed, ends[run_id]


def build_ledger(bars: pd.DataFrame, atr: np.ndarray, entries: Sequence[FrozenEntry], policy: ExitPolicy, instrument: str, *,
                 risk_usd: float, max_contracts: int, label: str, flatten_time_ct: str | None = None) -> tuple:
    """Size in whole micro contracts (floor), skip what cannot be sized, replay the exit, return ``(ledger, n_skipped, r_list)``.

    Realism rules (2026-11-11 audit):
      * ``max_contracts`` is the firm's MINI-contract limit; the cap in micro-equivalents is ``max_contracts * MICROS_PER_MINI``;
      * ``flatten_time_ct`` ("HH:MM" Chicago, e.g. a preset's ``flatten_time_ct``): no position is held past the flatten time of its own
        futures session -- an exit that would run later is replayed on the bars up to the last pre-flatten bar and closed at its close,
        and an entry that would open after the flatten time is skipped;
      * ONE position at a time: an entry whose bar is not after the previous trade's exit bar is skipped.
    ``n_skipped`` counts every skipped signal (cannot size, after flatten, or overlapping an open position)."""
    spec = INSTRUMENTS[instrument]
    pv, cost_pts, slip = spec["micro_pv"], spec["cost_points"], spec["slip_points"]
    cap_micros = int(max(0, max_contracts)) * MICROS_PER_MINI
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ref_atr = float(np.nanmedian(atr))
    flat_min = _flatten_ny_minute(flatten_time_ct)
    allowed = run_end = None
    if flat_min is not None and len(bars):
        allowed, run_end = _flatten_limits(bars, flat_min)
    et, xt, pnl, mae, mfe, cost, risk, side, rs = [], [], [], [], [], [], [], [], []
    skipped = 0
    last_exit_bar = -1
    ts = bars["timestamp"].to_numpy()
    for e in sorted(entries, key=lambda x: x.bar):
        if e.bar <= last_exit_bar:                       # one position at a time
            skipped += 1
            continue
        if allowed is not None and not allowed[e.bar]:   # would open after the flatten time
            skipped += 1
            continue
        n_c = int(min(cap_micros, math.floor(risk_usd / (e.stop_dist * pv)))) if e.stop_dist > 0 else 0
        if n_c < 1:
            skipped += 1
            continue
        if run_end is not None:
            stop_ix = int(run_end[e.bar]) + 1            # replay only up to (and including) the last pre-flatten bar
            r = simulate_exit(o[:stop_ix], h[:stop_ix], l[:stop_ix], c[:stop_ix], atr[:stop_ix], e, policy, cost_points=cost_pts,
                              slippage_points=slip, ref_atr=ref_atr)
        else:
            r = simulate_exit(o, h, l, c, atr, e, policy, cost_points=cost_pts, slippage_points=slip, ref_atr=ref_atr)
        last_exit_bar = r.exit_bar
        act_risk = n_c * e.stop_dist * pv
        et.append(ts[e.bar])
        xt.append(ts[r.exit_bar])
        pnl.append(r.r * act_risk)
        mae.append(min(r.mae_r * act_risk - cost_pts / e.stop_dist * act_risk, r.r * act_risk))
        mfe.append(max(r.mfe_r * act_risk, r.r * act_risk))
        cost.append(cost_pts / e.stop_dist * act_risk)
        risk.append(act_risk)
        side.append(e.side)
        rs.append(r.r)
    n = len(pnl)
    if not n:
        return TradeLedger.empty(), skipped, []
    led = TradeLedger(np.array(et, dtype="datetime64[ns]"), np.array(xt, dtype="datetime64[ns]"), np.array(pnl), np.array(mae),
                      np.array(mfe), np.array(cost), np.array(risk), np.array(side, int), np.array([label] * n, dtype=object),
                      np.array([instrument] * n, dtype=object))
    return led, skipped, rs


def scan(frames: dict, rules: PropRules, *, discovery_end, discovery_start=None, minutes: int = 5, risk_usd: float = 500.0,
         max_contracts: int = 4, sessions: dict | None = None, families=FAMILY_GRID, exits: Iterable[ExitPolicy] = EXIT_GRID,
         min_trades_per_30d: float = 3.0, cfg: ThirtyDayConfig | None = None, feasibility_sims: int = 600) -> ScanReport:
    """``frames``: ``{instrument: 1m-or-finer DataFrame(timestamp, open, high, low, close)}``. Only bars before ``discovery_end`` are used.

    ``sessions``: ``{instrument: [session names]}`` (default NY open for every instrument, plus London/Asia for GC and NQ).
    """
    end = pd.Timestamp(discovery_end)
    start = pd.Timestamp(discovery_start) if discovery_start is not None else None
    cfg = cfg or ThirtyDayConfig()
    rep = ScanReport(n_trials=0, discovery_end=str(end))
    exits = list(exits)
    for inst in INSTRUMENTS:
        if inst not in frames or frames[inst] is None or len(frames[inst]) == 0:
            rep.skipped_instruments[inst] = "no data supplied"
    for inst, raw in frames.items():
        if inst not in INSTRUMENTS or raw is None or len(raw) == 0:
            continue
        bars = resample_ohlc(raw, minutes)
        bars = bars[bars["timestamp"] < end]
        if start is not None:
            bars = bars[bars["timestamp"] >= start]
        bars = bars.reset_index(drop=True)
        if len(bars) < 2000:
            rep.skipped_instruments[inst] = "fewer than 2000 discovery bars"
            continue
        atr = atr_series(bars["high"].to_numpy(float), bars["low"].to_numpy(float), bars["close"].to_numpy(float))
        nb = _with_ny_time(bars)
        d0, d1 = bars["timestamp"].iloc[0], bars["timestamp"].iloc[-1]
        months = max((d1 - d0).days / 30.0, 1e-9)
        sess_list = (sessions or {}).get(inst) or (["ny_open", "london"] if inst in ("NQ", "GC") else ["ny_open"])
        for session in sess_list:
            for fam, params in families:
                ents = _signals(fam, nb, atr, session, params)
                if len(ents) < 30:
                    continue
                for pol in exits:
                    cell = Cell(fam, inst, session, tuple(sorted(params.items())), pol.name)
                    rep.n_trials += 1
                    led, skipped, rs = build_ledger(bars, atr, ents, pol, inst, risk_usd=risk_usd, max_contracts=max_contracts, label=cell.key,
                                                 flatten_time_ct=getattr(rules, "flatten_time_ct", None))
                    if len(led) < 30:
                        continue
                    r = np.asarray(rs)
                    wins, losses = r[r > 0], r[r <= 0]
                    wr = float(len(wins) / len(r))
                    payoff = float(wins.mean() / max(abs(losses.mean()), 1e-9)) if len(wins) and len(losses) else 0.0
                    net_e = float(r.mean())
                    tpm = float(len(led) / months)
                    if tpm < min_trades_per_30d or net_e <= 0:
                        rep.results.append(ScanResult(cell, len(ents), skipped, tpm, wr, payoff, net_e, 0.0, None))
                        continue
                    ub = screening_upper_bound(trades_per_30d=tpm, expectancy_r_net=net_e, win_prob=wr, payoff_r=payoff, rules=rules, n_sims=feasibility_sims)
                    res = evaluate_thirty_day(led, rules, data_start=d0, data_end=d1, cfg=cfg, with_sensitivity=False)
                    rep.results.append(ScanResult(cell, len(ents), skipped, tpm, wr, payoff, net_e, ub, res))
    rep.results.sort(key=lambda s: (rank_key(s.result) if s.result is not None else (1.0, 0.0, 1e9, 1.0)))
    rep.notes.append(f"{rep.n_trials} cells tried on DISCOVERY data only (< {end.date()}). Survivors are hypotheses: they have not touched "
                     "development, walk-forward or sealed data and the trial count above is the multiple-testing burden.")
    return rep
