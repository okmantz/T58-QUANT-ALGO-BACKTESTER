"""Hypothesis-first edge discovery: measure whether a market inefficiency EXISTS before any strategy is built.

Pipeline: inefficiency -> mechanism -> *measure the raw behaviour* (this module) -> only then build strategies from the families
that exploit confirmed behaviours -> test them under realistic execution and prop-firm constraints (the canonical engine + lifecycle).

Each probe turns a behaviour into per-event forward returns (bps, after a round-trip cost), clusters them by trading day so one busy day
cannot masquerade as many independent observations, and reports n, net bps/event, a day-clustered t-stat, hit rate and a first-half /
second-half stability check. A Bonferroni-style hurdle across the probes keeps "we tried six things" from producing a false positive.
Nothing here trades or builds a strategy; it only answers: is there something to exploit, in which direction, and is it bigger than costs?
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_EVENTS = 80


@dataclass
class Evidence:
    n: int
    gross_bps: float
    net_bps: float
    t_stat: float
    hit_rate: float
    half_agree: bool
    n_days: int

    def to_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class Finding:
    hypothesis: str
    inefficiency: str
    mechanism: str
    direction: str              # what to do when the event occurs: "fade" / "follow" / "long" / "short" / "volatility"
    verdict: str                # confirmed | weak | absent | insufficient
    evidence: Evidence | None
    families: list = field(default_factory=list)
    note: str = ""

    def to_dict(self):
        return {"hypothesis": self.hypothesis, "inefficiency": self.inefficiency, "mechanism": self.mechanism, "direction": self.direction,
                "verdict": self.verdict, "evidence": self.evidence.to_dict() if self.evidence else None, "families": self.families, "note": self.note}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        col = next((c for c in ("timestamp", "datetime", "time", "date") if c in d.columns), None)
        if col is None:
            raise ValueError("Bars need a datetime index or a timestamp column.")
        d.index = pd.to_datetime(d[col])
    d.columns = [str(c).lower() for c in d.columns]
    d = d.sort_index()
    d["day"] = d.index.normalize()
    return d


def _evidence(gross_bps: pd.Series, days: pd.Series, cost_bps: float) -> Evidence | None:
    g = pd.Series(gross_bps).dropna()
    if len(g) < MIN_EVENTS:
        return None
    d = days.loc[g.index] if hasattr(days, "loc") else days
    net = g - cost_bps
    per_day = net.groupby(d.values).sum()           # cluster by day
    n_days = len(per_day)
    se = per_day.std(ddof=1) / math.sqrt(n_days) if n_days > 1 and per_day.std(ddof=1) > 0 else float("nan")
    t = float(per_day.mean() / se) if se and se == se else 0.0
    h = len(net) // 2
    return Evidence(n=len(net), gross_bps=float(g.mean()), net_bps=float(net.mean()), t_stat=t, hit_rate=float((net > 0).mean()),
                    half_agree=bool(np.sign(net.iloc[:h].mean()) == np.sign(net.iloc[h:].mean()) and net.iloc[:h].mean() != 0), n_days=n_days)


def _classify(ev: Evidence | None, hurdle: float) -> str:
    if ev is None:
        return "insufficient"
    if ev.net_bps > 0 and ev.t_stat >= hurdle and ev.half_agree:
        return "confirmed"
    if ev.net_bps > 0 and ev.t_stat >= hurdle * 0.6:
        return "weak"
    return "absent"


# ------------------------------------------------------------------------------------------------------------- probes
def _daily(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("day")
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})


def _gap_probe(d, cost):
    day = _daily(d)
    if len(day) < 120:
        return None, None
    gap = day["open"] / day["close"].shift() - 1
    atr = ((day["high"] - day["low"]) / day["close"]).rolling(14).mean().shift()
    big = gap.abs() > 0.5 * atr
    intraday = day["close"] / day["open"] - 1
    fade = (-np.sign(gap) * intraday * 1e4)[big]
    follow = (np.sign(gap) * intraday * 1e4)[big]
    return (fade, "fade"), (follow, "follow"), day.index.to_series()


def _opening_drive_probe(d, cost):
    d = d.copy(); d["bar"] = d.groupby("day").cumcount()
    first = d[d["bar"] <= 5].groupby("day")["close"].agg(["first", "last"])
    rest = d[d["bar"] > 5].groupby("day")["close"].agg(["first", "last"])
    j = first.join(rest, lsuffix="_o", rsuffix="_r", how="inner").dropna()
    if len(j) < 80:
        return None
    drive = np.sign(j["last_o"] / j["first_o"] - 1)
    cont = drive * (j["last_r"] / j["first_r"] - 1) * 1e4
    return cont, j.index.to_series()


def _autocorr_probe(d, cost, n=12):
    c = d["close"]
    past = c / c.shift(n) - 1
    fut = c.shift(-n) / c - 1
    thr = past.abs().quantile(0.8)
    sel = past.abs() > thr
    ev = (fut * 1e4)[sel]
    follow = np.sign(past[sel]) * ev
    return follow.iloc[::n], d["day"]


def _hour_probe(d, cost):
    r = (d["close"] / d["close"].shift(1) - 1) * 1e4
    hr = d.index.hour
    out = {}
    for h in sorted(set(hr)):
        s = r[hr == h].dropna()
        if len(s) >= MIN_EVENTS:
            out[h] = s
    return out


def _squeeze_probe(d, cost, n=12):
    rng = (d["high"] - d["low"]) / d["close"]
    low = rng.rolling(n).mean() < rng.rolling(500, min_periods=100).quantile(0.2)
    nxt = (d["close"].shift(-n) / d["close"] - 1).abs()
    base = nxt.mean()
    ratio = nxt[low].mean() / base if base else float("nan")
    return float(ratio), int(low.sum())


def _sweep_probe(d, cost):
    day = _daily(d)
    ph = day["high"].shift(); pl = day["low"].shift()
    swept_hi = (day["high"] > ph) & (day["close"] < ph)       # poked above yesterday's high, closed back below
    swept_lo = (day["low"] < pl) & (day["close"] > pl)
    nxt = day["close"].shift(-1) / day["close"] - 1
    rev = pd.concat([(-nxt[swept_hi]), (nxt[swept_lo])]).sort_index() * 1e4
    return rev, day.index.to_series()


def measure_edges(df: pd.DataFrame, cost_bps: float = 1.5, extra_events: pd.DataFrame | None = None) -> dict:
    """Run every probe. ``cost_bps`` is the round-trip cost per event in basis points of price (spread + commission + slippage)."""
    d = _prep(df)
    findings: list[Finding] = []
    n_probes = 7
    hurdle = 2.0 + 0.25 * math.log(n_probes)         # ~2.5: inflated t-hurdle for the number of hypotheses examined

    gp = _gap_probe(d, cost_bps)
    if gp and gp[0] is not None:
        (fade, _), (follow, _), days = gp
        for series, direction, fams in ((fade, "fade", ["overnight_gap_fade", "overnight_gap_fade_atr_gated", "monday_gap_reversal"]),
                                        (follow, "follow", ["gap_and_go_continuation", "gap_and_hold_acceptance"])):
            ev = _evidence(series, days, cost_bps)
            findings.append(Finding(f"gap_{direction}", "Overnight gaps are over/under-reactions to news", "Gap = information shock; liquidity returns at the open",
                                    direction, _classify(ev, hurdle), ev, fams))
    else:
        findings.append(Finding("gap", "Overnight gaps", "", "fade/follow", "insufficient", None, ["overnight_gap_fade", "gap_and_go_continuation"],
                                "Needs >= 120 sessions of intraday bars."))

    od = _opening_drive_probe(d, cost_bps)
    ev = _evidence(od[0], od[1], cost_bps) if od else None
    findings.append(Finding("opening_drive_continuation", "Early order-flow imbalance persists", "Institutional execution is spread over hours",
                            "follow", _classify(ev, hurdle), ev, ["opening_momentum_burst", "daily_momentum_open_drive", "ny_open_drive_continue_or_fade"]))

    ac, days = _autocorr_probe(d, cost_bps)
    ev = _evidence(ac, days, cost_bps)
    if ev and ev.net_bps <= 0 and ev.gross_bps < 0:
        rev = _evidence(-ac, days, cost_bps)
        findings.append(Finding("short_horizon_reversal", "Short-horizon overreaction", "Liquidity-demand price pressure mean-reverts",
                                "fade", _classify(rev, hurdle), rev, ["mean_reversion_band", "vwap_reversion", "rsi_extreme_reversion"]))
    findings.append(Finding("short_horizon_momentum", "Short-horizon autocorrelation", "Orders are split and herding persists",
                            "follow", _classify(ev, hurdle), ev, ["momentum_continuation", "pct_change_momentum_burst", "roc_momentum_continuation"]))

    hours = _hour_probe(d, cost_bps)
    best = None
    for h, s in hours.items():
        for sign, lab in ((1, "long"), (-1, "short")):
            e = _evidence(sign * s, d["day"], cost_bps)
            if e and (best is None or e.t_stat > best[1].t_stat):
                best = (h, e, lab)
    if best:
        hurdle_h = hurdle + 0.25 * math.log(max(2, len(hours) * 2))     # extra penalty: we picked the best of many hours
        findings.append(Finding(f"time_of_day_drift_h{best[0]:02d}", "Systematic intraday drift by hour", "Flows (rebalancing, close/open auctions) are time-patterned",
                                best[2], _classify(best[1], hurdle_h), best[1], ["session_time_effect", "time_stop_scalp"], f"best hour {best[0]:02d}:00"))

    ratio, n_sq = _squeeze_probe(d, cost_bps)
    verdict = "confirmed" if n_sq >= MIN_EVENTS and ratio >= 1.15 else "weak" if n_sq >= MIN_EVENTS and ratio >= 1.05 else "insufficient" if n_sq < MIN_EVENTS else "absent"
    findings.append(Finding("compression_then_expansion", "Volatility clusters: quiet precedes large moves", "Volatility is persistent and mean-reverting",
                            "volatility", verdict, None, ["volatility_contraction_squeeze", "keltner_squeeze_breakout", "bollinger_squeeze_expansion"],
                            f"|move| after compression is {ratio:.2f}x normal over {n_sq} bars (direction still needs a trigger)"))

    sw, days = _sweep_probe(d, cost_bps)
    ev = _evidence(sw, days, cost_bps)
    findings.append(Finding("liquidity_sweep_reversal", "Stop-runs beyond obvious highs/lows reverse", "Resting stops are liquidity for larger players",
                            "fade", _classify(ev, hurdle), ev, ["liquidity_sweep_reversal", "turtle_soup_displacement_fade"]))

    if extra_events is not None and len(extra_events):            # e.g. SEC filing-date events: columns ts, direction (+1/-1), horizon_bars
        px = d["close"]
        rows = []
        for r in extra_events.itertuples():
            i = px.index.searchsorted(pd.Timestamp(r.ts))
            h = int(getattr(r, "horizon_bars", 20))
            if 0 < i < len(px) - h:
                rows.append((px.index[i], float(r.direction) * (px.iloc[i + h] / px.iloc[i] - 1) * 1e4))
        if rows:
            s = pd.Series([v for _, v in rows], index=[t for t, _ in rows])
            ev = _evidence(s, pd.Series(s.index.normalize(), index=s.index), cost_bps)
            findings.append(Finding("event_drift", "Post-event drift (e.g. earnings / filings)", "Slow information diffusion after SEC filings",
                                    "follow", _classify(ev, hurdle), ev, ["momentum_continuation"], "from supplied event list"))

    order = {"confirmed": 0, "weak": 1, "absent": 2, "insufficient": 3}
    findings.sort(key=lambda f: (order[f.verdict], -(f.evidence.t_stat if f.evidence else 0)))
    confirmed = [f for f in findings if f.verdict == "confirmed"]
    weak = [f for f in findings if f.verdict == "weak"]
    fams: list = []
    for f in confirmed + weak:
        for x in f.families:
            if x not in fams:
                fams.append(x)
    return {
        "cost_bps": cost_bps, "t_hurdle": round(hurdle, 2), "n_bars": len(d), "findings": [f.to_dict() for f in findings],
        "n_confirmed": len(confirmed), "n_weak": len(weak), "families": fams,
        "headline": (f"{len(confirmed)} confirmed edge(s), {len(weak)} weak. Build from: " + ", ".join(f.hypothesis for f in confirmed))
        if confirmed else (f"No hypothesis cleared the bar on this data ({len(weak)} weak). Do not expect a strategy built here to survive; "
                           "try another market/timeframe or more history before searching further."),
    }


def hypothesis_first_families(df: pd.DataFrame, cost_bps: float = 1.5, include_weak: bool = True) -> tuple[list, dict]:
    """(families to search, report). Falls back to ALL active families when nothing is confirmed, so 'use everything' still applies."""
    rep = measure_edges(df, cost_bps)
    from app.search import strategy_space as ss
    pool = set(ss.active_family_names())
    fams = [f for f in rep["families"] if f in pool]
    if not include_weak:
        confirmed = [x for f in rep["findings"] if f["verdict"] == "confirmed" for x in f["families"]]
        fams = [f for f in fams if f in confirmed]
    return (fams or sorted(pool)), rep
