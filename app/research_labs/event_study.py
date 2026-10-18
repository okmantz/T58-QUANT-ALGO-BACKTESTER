"""Lab 6 -- Economic Event Study Lab.

For a list of release timestamps (CPI, PCE, NFP, FOMC, Treasury auctions, ...) measure what the market did around
them and compare each event with **matched non-event sessions**: the same clock time on days with no release.

Measured per event:
* pre-event volatility (mean true range over ``pre_bars`` bars before the release),
* post-event expansion (mean true range over ``post_bars`` bars after / pre-event),
* initial reaction (signed return over the first ``reaction_bars`` bars) and continuation (signed return over the
  following ``cont_bars`` bars, in the direction of the initial reaction -> >0 means continuation, <0 means fade),
* adverse excursion (worst move against the initial-reaction direction after the reaction window, in ATR units),
* optional: the same-window return of a simple strategy supplied as a per-bar return series.

Event timestamps are matched in the DATA'S OWN timezone (naive); convert the events file to the same timezone as
the bars. A built-in rule generator is provided for the NFP schedule (first Friday of the month); every other
series (CPI, FOMC, auctions) must be uploaded because their dates are published schedules, not rules.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.research_labs import common as C

LAB = "event_study"
TITLE = "Economic Event Study Lab"
ASSUMPTIONS = [
    "Event timestamps are interpreted in the SAME timezone as the bar timestamps (both naive); the lab does not convert zones.",
    "Each event is aligned to the first bar whose timestamp is >= the release time. Bars before it are 'pre', that bar onward 'post'.",
    "Matched control = the same time of day, same weekday, on sessions with no event that day (up to ``controls_per_event`` per event, nearest in time, deterministic).",
    "Volatility = true range / ATR(14) measured BEFORE the window (so the ATR scale is not contaminated by the event).",
    "Continuation = initial-reaction-direction x subsequent return; positive = trend continued, negative = the reaction faded.",
    "Events with missing bars around them are dropped and counted. Multiple-testing: Benjamini-Hochberg across the metrics tested.",
]


def nfp_schedule(start: str, end: str, hour: int = 8, minute: int = 30) -> pd.DataFrame:
    """First Friday of each month at 08:30 (US/Eastern wall-clock; convert if your bars are not Eastern).
    Approximation: the BLS occasionally shifts the release (holidays); upload the official calendar for precision."""
    rows = []
    for m in pd.period_range(start, end, freq="M"):
        d = pd.Timestamp(m.start_time)
        d += pd.Timedelta(days=(4 - d.weekday()) % 7)
        rows.append({"timestamp": d + pd.Timedelta(hours=hour, minutes=minute), "event": "NFP"})
    return pd.DataFrame(rows)


def parse_events(events: pd.DataFrame | list) -> pd.DataFrame:
    ev = pd.DataFrame(events).copy()
    cols = {c.lower().strip(): c for c in ev.columns}
    tcol = next((cols[k] for k in ("timestamp", "datetime", "time", "date", "release_time") if k in cols), None)
    if tcol is None:
        raise C.ResearchLabError("Events need a 'timestamp' column (release date+time).")
    ev["timestamp"] = pd.to_datetime(ev[tcol], errors="coerce")
    if getattr(ev["timestamp"].dt, "tz", None) is not None:
        ev["timestamp"] = ev["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    ncol = next((cols[k] for k in ("event", "name", "type", "series") if k in cols), None)
    ev["event"] = ev[ncol].astype(str) if ncol else "event"
    ev = ev.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    return ev[["timestamp", "event"]].reset_index(drop=True)


def _measure(df: pd.DataFrame, i: int, atr: float, pre: int, post: int, react: int, cont: int) -> dict | None:
    need_lo, need_hi = i - pre, i + react + cont
    if need_lo < 1 or need_hi >= len(df) or not math.isfinite(atr) or atr <= 0:
        return None
    c, h, l = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))  # tr[k] belongs to bar k+1
    pre_v = float(tr[need_lo - 1:i - 1].mean())
    post_end = min(i + post, len(df) - 1)
    post_v = float(tr[i - 1:post_end].mean()) if post_end > i else float("nan")
    base = c[i - 1]
    r_init = (c[i + react - 1] - base) / base
    r_cont = (c[i + react + cont - 1] - c[i + react - 1]) / base
    sgn = 1.0 if r_init >= 0 else -1.0
    seg_h, seg_l = h[i + react:i + react + cont], l[i + react:i + react + cont]
    adverse = float((c[i + react - 1] - seg_l.min()) / atr) if sgn > 0 else float((seg_h.max() - c[i + react - 1]) / atr)
    return {"pre_vol_atr": pre_v / atr, "post_vol_atr": post_v / atr, "expansion": post_v / pre_v if pre_v > 0 else float("nan"),
            "reaction": r_init, "reaction_abs_atr": abs(c[i + react - 1] - base) / atr, "continuation": sgn * r_cont, "adverse_atr": adverse}


def run(df: pd.DataFrame, events, *, label: str = "series", pre_bars: int = 12, post_bars: int = 12, reaction_bars: int = 3,
        cont_bars: int = 12, controls_per_event: int = 3, strategy_returns: pd.Series | None = None,
        seed: int = C.DEFAULT_SEED, save: bool = True) -> C.ExperimentRecord:
    d = C.prepare_ohlcv(df)
    ev = parse_events(events)
    if len(ev) < 8:
        raise C.ResearchLabError(f"Only {len(ev)} usable events -- an event study needs at least ~8 (preferably 30+).")
    ts = d["timestamp"].to_numpy()
    atr = C.atr_series(d, 14).shift(1).to_numpy()
    ev = ev[(ev["timestamp"] >= d["timestamp"].iloc[0]) & (ev["timestamp"] <= d["timestamp"].iloc[-1])].reset_index(drop=True)
    if len(ev) < 8:
        raise C.ResearchLabError("Fewer than 8 events fall inside the price data's date range.")
    event_days = set(ev["timestamp"].dt.normalize())
    rows, dropped = [], 0
    for _, e in ev.iterrows():
        i = int(np.searchsorted(ts, np.datetime64(e["timestamp"])))
        m = _measure(d, i, atr[i] if i < len(atr) else float("nan"), pre_bars, post_bars, reaction_bars, cont_bars)
        if m is None or (d["timestamp"].iloc[i] - e["timestamp"]) > pd.Timedelta(hours=6):
            dropped += 1
            continue
        rows.append({"timestamp": e["timestamp"], "event": e["event"], "i": i, **m})
    if len(rows) < 8:
        raise C.ResearchLabError(f"Only {len(rows)} events had complete bars around them (dropped {dropped}).")
    evt = pd.DataFrame(rows)
    # matched non-event controls: same weekday + time of day on days without events
    d_day = d["timestamp"].dt.normalize()
    crow = []
    tod_s = (d["timestamp"] - d_day).to_numpy()
    for _, r in evt.iterrows():
        target = r["timestamp"] - r["timestamp"].normalize()
        wd = r["timestamp"].weekday()
        cand_days = [x for x in d_day.unique() if pd.Timestamp(x).weekday() == wd and x not in event_days]
        cand_days.sort(key=lambda x: abs((pd.Timestamp(x) - r["timestamp"].normalize()).days))
        got = 0
        for day in cand_days:
            idx = np.searchsorted(ts, np.datetime64(pd.Timestamp(day) + target))
            if idx >= len(d) or (d["timestamp"].iloc[idx] - (pd.Timestamp(day) + target)) > pd.Timedelta(minutes=30):
                continue
            m = _measure(d, int(idx), atr[idx], pre_bars, post_bars, reaction_bars, cont_bars)
            if m:
                crow.append({"event": r["event"], "i": int(idx), **m})
                got += 1
            if got >= controls_per_event:
                break
    ctl = pd.DataFrame(crow)
    metrics = ["pre_vol_atr", "post_vol_atr", "expansion", "reaction_abs_atr", "adverse_atr"]
    comp, pv = [], []
    for k in metrics:
        a = evt[k].dropna().to_numpy()
        b = ctl[k].dropna().to_numpy() if len(ctl) else np.array([])
        w = C.welch_test(a, b) if len(b) > 3 else {"t": None, "p": None}
        lo, hi = C.diff_ci(a, b, seed=seed, n_boot=400) if len(b) > 3 else (None, None)
        comp.append({"metric": k, "event_mean": float(a.mean()), "control_mean": float(b.mean()) if len(b) else None, "event_n": int(len(a)),
                     "control_n": int(len(b)), "t": w["t"], "p": w["p"], "effect_d": C.cohens_d(a, b) if len(b) > 3 else None,
                     "diff_ci_low": lo, "diff_ci_high": hi})
        pv.append(w["p"])
    for r, q in zip(comp, C.bh_adjust(pv)):
        r["q_value_bh"] = q
        r["verdict"] = ("event differs from matched non-event sessions" if q is not None and q < 0.10 else "no reliable difference from matched sessions")
    cont = evt["continuation"].dropna().to_numpy()
    cont_t = C.one_sample_t(cont)
    lo, hi = C.bootstrap_ci(cont, seed=seed)
    ctl_cont = ctl["continuation"].dropna().to_numpy() if len(ctl) else np.array([])
    reaction = {"continuation_mean": float(cont.mean()), "ci_low": lo, "ci_high": hi, "t": cont_t["t"], "p": cont_t["p"],
                "continue_rate": float((cont > 0).mean()),
                "control_continuation_mean": float(ctl_cont.mean()) if len(ctl_cont) else None,
                "reading": ("reactions tend to CONTINUE" if cont_t["p"] is not None and cont_t["p"] < 0.05 and cont.mean() > 0 else
                            "reactions tend to FADE" if cont_t["p"] is not None and cont_t["p"] < 0.05 and cont.mean() < 0 else
                            "no reliable continuation or fade")}
    by_event = []
    for name, g in evt.groupby("event"):
        by_event.append({"event": name, "n": int(len(g)), "mean_expansion": float(g["expansion"].mean()), "mean_adverse_atr": float(g["adverse_atr"].mean()),
                         "continuation_mean": float(g["continuation"].mean())})
    strat = None
    if strategy_returns is not None and len(strategy_returns):
        sr = pd.Series(strategy_returns).reindex(range(len(d))).fillna(0.0).to_numpy()
        ev_ret = np.array([sr[int(i):int(i) + post_bars].sum() for i in evt["i"]])
        ct_ret = np.array([sr[int(i):int(i) + post_bars].sum() for i in ctl["i"]]) if len(ctl) else np.array([])
        strat = {"event_window_mean": float(ev_ret.mean()), "control_window_mean": float(ct_ret.mean()) if len(ct_ret) else None,
                 **({"p": C.welch_test(ev_ret, ct_ret)["p"]} if len(ct_ret) > 3 else {})}
    cut = int(len(evt) * 0.7)
    oos = {"is_expansion": float(evt["expansion"].iloc[:cut].mean()), "oos_expansion": float(evt["expansion"].iloc[cut:].mean()),
           "oos_events": int(len(evt) - cut), "note": "Chronological 70/30 split of events: does the expansion estimated early hold up later?"}
    warnings = []
    if dropped:
        warnings.append(f"{dropped} events were dropped (bars missing around them).")
    if len(evt) < 30:
        warnings.append(f"Only {len(evt)} events: confidence intervals are wide; treat results as exploratory.")
    exp_row = next(r for r in comp if r["metric"] == "expansion")
    results = {"headline": f"{len(evt)} events vs {len(ctl)} matched controls; post-event volatility expansion {exp_row['event_mean']:.2f}x vs {exp_row['control_mean'] if exp_row['control_mean'] is not None else float('nan'):.2f}x; {reaction['reading']}",
               "events_used": int(len(evt)), "events_dropped": dropped, "event_vs_control": comp, "reaction": reaction, "by_event": by_event,
               "out_of_sample": oos, "strategy": strat,
               "events": [{"timestamp": str(r.timestamp), "event": r.event, "expansion": None if not math.isfinite(r.expansion) else float(r.expansion),
                           "reaction": float(r.reaction), "continuation": float(r.continuation), "adverse_atr": float(r.adverse_atr)}
                          for r in evt.tail(200).itertuples()]}
    return C.record_from(LAB, TITLE, {"pre_bars": pre_bars, "post_bars": post_bars, "reaction_bars": reaction_bars, "cont_bars": cont_bars,
                                      "controls_per_event": controls_per_event, "n_events_in": int(len(ev))}, ASSUMPTIONS, d, results, warnings,
                         seed, label, save=save)


def event_flag_feature(df: pd.DataFrame, events, window_bars: int = 6) -> pd.Series:
    """1.0 on bars from the release until ``window_bars`` later (for strategies / indicators_v10 'event_window')."""
    d = C.prepare_ohlcv(df)
    ev = parse_events(events)
    flag = np.zeros(len(d))
    ts = d["timestamp"].to_numpy()
    for t in ev["timestamp"]:
        i = int(np.searchsorted(ts, np.datetime64(t)))
        flag[i:i + window_bars] = 1.0
    return pd.Series(flag, index=d.index)
