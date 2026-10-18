"""Lab 8 -- Market Seasonality Lab.

Studies calendar-driven patterns -- time of day, weekday, month of year, holiday proximity, option-expiry / month-end
effects and (optionally) release-event days -- and answers the only question that matters: *is this real, or the
kind of pattern you find in any random series if you look at enough calendar slices?*

Method (every choice is listed in ``ASSUMPTIONS`` and travels with the result):

* **Non-overlapping observations.** Intraday buckets (default 60 min) and whole sessions are the unit, not individual
  bars, so serial correlation between neighbouring bars does not inflate significance.
* **Two outcomes** per slice: signed return (bps) and volatility (high-low range, bps).
* **Baselines.** (1) *rest of the sample* -- each slice vs every other slice in its family, after subtracting the
  overall mean; (2) *weekday-matched* -- holiday / expiry / month-end / event days are compared on weekday-adjusted
  residuals, so "Fridays are different" can't masquerade as "expiry is different"; (3) a *family-wise permutation
  null* -- the labels are shuffled 300x and the largest |t| of the family is recorded, giving a p-value for "the best
  slice in this family" rather than for a slice picked after looking.
* **Multiple testing.** Benjamini-Hochberg across EVERY slice tested in the whole run (``q_global``); the number of
  tests and the false positives expected by chance are reported.
* **Persistence.** Chronological in-sample / out-of-sample split. A slice is only "promising" if it is significant
  after correction AND has the same sign out-of-sample. A discovery-then-replication summary shows how many
  in-sample-nominal effects kept their sign later (about half would by coincidence).
* **Yearly consistency.** Share of calendar years in which the slice had the same sign as its overall effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research_labs import common as C

LAB = "seasonality"
TITLE = "Market Seasonality Lab"
ASSUMPTIONS = [
    "Timestamps are used as given (naive, in the data's own timezone). Time-of-day results are only meaningful relative to that clock.",
    "Observations are non-overlapping: an intraday bucket (default 60 min) or a whole session -- never overlapping bar windows.",
    "Outcomes: signed close-to-close return in bps (excess over the family's overall mean) and range volatility in bps (excess over the family's mean, as a ratio).",
    "Baseline 1: each slice vs the rest of its family. Baseline 2: holiday / expiry / month-end / event days are compared on WEEKDAY-ADJUSTED residuals.",
    "Baseline 3: label-permutation null (300 shuffles). The family-wise p-value is the chance a random labelling produces a slice at least as extreme as the best one.",
    "Multiple testing: Benjamini-Hochberg across every slice and outcome tested in this run (q_global). 'Promising' needs q_global < 0.10 AND the same sign out-of-sample.",
    "Holidays use the US federal calendar; options expiry = third Friday of the month; NFP = first Friday. Upload an events CSV for FOMC / CPI / auctions.",
    "Calendar effects are weak and often arbitrage away; a clean result here is a hypothesis to test in the backtester, not a trading rule.",
]

_WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MO = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
N_PERM = 300


# ---------------------------------------------------------------------------------------------------------------
def _sessions(d: pd.DataFrame, bps: np.ndarray) -> pd.DataFrame:
    day = d["timestamp"].dt.normalize()
    g = pd.DataFrame({"day": day, "ret": bps, "hi": d["high"].to_numpy(), "lo": d["low"].to_numpy(), "cl": d["close"].to_numpy()})
    s = g.groupby("day", sort=True).agg(ret=("ret", "sum"), hi=("hi", "max"), lo=("lo", "min"), cl=("cl", "last"), bars=("ret", "size"))
    s["rng"] = (s["hi"] - s["lo"]) / s["cl"] * 1e4
    s = s[s["bars"] >= max(3, int(s["bars"].median() * 0.5))]          # drop half sessions (data edges, early closes)
    return s.reset_index()


def _tod_obs(d: pd.DataFrame, bps: np.ndarray, minutes: int) -> pd.DataFrame:
    b = d["timestamp"].dt.floor(f"{minutes}min")
    g = pd.DataFrame({"b": b, "ret": bps, "rng": (d["high"] - d["low"]).to_numpy() / d["close"].to_numpy() * 1e4})
    a = g.groupby("b", sort=True).agg(ret=("ret", "sum"), rng=("rng", "sum"), n=("ret", "size"))
    full = int(a["n"].median())
    a = a[a["n"] >= max(1, full)].reset_index()                         # only complete buckets
    a["label"] = a["b"].dt.strftime("%H:%M")
    return a


def _holiday_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    from pandas.tseries.holiday import USFederalHolidayCalendar

    return USFederalHolidayCalendar().holidays(start=start - pd.Timedelta(days=10), end=end + pd.Timedelta(days=10))


def _third_friday(days: pd.Series) -> pd.Series:
    return (days.dt.weekday == 4) & (days.dt.day >= 15) & (days.dt.day <= 21)


def _first_friday(days: pd.Series) -> pd.Series:
    return (days.dt.weekday == 4) & (days.dt.day <= 7)


def _neighbour_flags(days: pd.Series, targets: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Trading-day-aware: the session before / after / of each target date, using the sessions actually in the data."""
    dset = days.reset_index(drop=True)
    pos = {t: i for i, t in enumerate(dset)}
    before = np.zeros(len(dset), bool)
    after = np.zeros(len(dset), bool)
    on = np.zeros(len(dset), bool)
    arr = dset.to_numpy()
    for t in targets:
        t = pd.Timestamp(t).normalize()
        j = int(np.searchsorted(arr, np.datetime64(t)))
        if t in pos:
            on[pos[t]] = True
        if 0 < j <= len(dset) and (j < len(dset) or t > dset.iloc[-1]) and j - 1 >= 0 and (j >= len(dset) or arr[j] >= np.datetime64(t)):
            if j - 1 >= 0 and dset.iloc[j - 1] < t:
                before[j - 1] = True
        k = j + 1 if (j < len(dset) and dset.iloc[j] == t) else j
        if k < len(dset) and dset.iloc[k] > t:
            after[k] = True
    idx = days.index
    return pd.Series(before, index=idx), pd.Series(after, index=idx), pd.Series(on, index=idx)


def _perm_family_p(y: np.ndarray, codes: np.ndarray, k: int, seed: int) -> dict:
    """Family-wise permutation test: observed max |Welch-ish t| of any slice vs the rest, against label-shuffled nulls."""
    ok = np.isfinite(y) & (codes >= 0)
    y, codes = y[ok], codes[ok]
    if len(y) < 20 or k < 2:
        return {"p": None, "observed_max_t": None}

    def max_t(lab: np.ndarray) -> float:
        n = np.bincount(lab, minlength=k).astype(float)
        s = np.bincount(lab, weights=y, minlength=k)
        s2 = np.bincount(lab, weights=y * y, minlength=k)
        N, S, S2 = len(y), y.sum(), (y * y).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            m1 = s / n
            v1 = (s2 - n * m1 ** 2) / (n - 1)
            nr, sr, s2r = N - n, S - s, S2 - s2
            m2 = sr / nr
            v2 = (s2r - nr * m2 ** 2) / (nr - 1)
            t = (m1 - m2) / np.sqrt(v1 / n + v2 / nr)
        t = t[(n >= 5) & (nr >= 5) & np.isfinite(t)]
        return float(np.abs(t).max()) if len(t) else 0.0

    obs = max_t(codes)
    rng = np.random.default_rng(seed)
    ge = 0
    lab = codes.copy()
    for _ in range(N_PERM):
        rng.shuffle(lab)
        if max_t(lab) >= obs:
            ge += 1
    return {"p": (1 + ge) / (1 + N_PERM), "observed_max_t": obs}


def _year_consistency(outcome: pd.Series, mask: pd.Series, years: pd.Series, overall_sign: float) -> dict:
    if overall_sign == 0 or not np.isfinite(overall_sign):
        return {"years": 0, "years_same_sign": None}
    sel = mask & outcome.notna()
    tot = same = 0
    for _, g in outcome[sel].groupby(years[sel]):
        if len(g) >= 3:
            tot += 1
            same += int(np.sign(g.mean()) == np.sign(overall_sign))
    return {"years": tot, "years_same_sign": (same / tot) if tot else None}


def _family(name: str, obs: pd.DataFrame, labels: dict[str, pd.Series], *, outcomes: dict[str, pd.Series], oos_fraction: float,
            min_n: int, seed: int, adjust_weekday: pd.Series | None = None, exclusive: bool = False) -> tuple[list[dict], list[dict]]:
    """Runs one family; returns (rows, profile). ``exclusive`` = labels partition the rows (permutation test applies)."""
    rows: list[dict] = []
    allmask = pd.Series(True, index=obs.index)
    years = obs["year"]
    for oname, raw in outcomes.items():
        y = pd.Series(raw, index=obs.index).astype(float)
        if oname == "volatility":
            base = y.mean()
            y = (y / base - 1.0) * 100.0 if base else y          # % above/below the family's average range
            unit = "% vs avg range"
        else:
            y = y - y.mean()
            unit = "bps vs avg"
        if adjust_weekday is not None:
            y = y - y.groupby(adjust_weekday).transform("mean")
        gc = C.group_compare(y, allmask, labels, oos_fraction=oos_fraction, min_n=min_n, seed=seed)
        perm = {"p": None}
        if exclusive:
            names = list(labels)
            codes = np.full(len(y), -1)
            for i, nm in enumerate(names):
                codes[labels[nm].fillna(False).to_numpy()] = i
            perm = _perm_family_p(y.to_numpy(), codes, len(names), seed)
        cut = int(len(y) * (1 - min(max(oos_fraction, 0.1), 0.6)))
        pos = np.arange(len(y))
        for r in gc:
            m = labels[r["group"]].fillna(False).astype(bool)
            is_sel = m & (pos < cut)
            is_rest = (~m) & (pos < cut)
            r["is_p"] = C.welch_test(y[is_sel].dropna().to_numpy(), y[is_rest].dropna().to_numpy())["p"] if is_sel.sum() > 2 else None
            r.update(_year_consistency(y, m, years, r["mean"] if r.get("mean") is not None else 0.0))
            r["family"], r["outcome"], r["unit"] = name, oname, unit
            r["family_perm_p"] = perm["p"]
            r["slice"] = r["group"]
            r["group"] = f"{name}: {r['slice']} ({oname})"
        rows.extend(gc)
    profile = []
    for sl in labels:
        m = labels[sl].fillna(False).astype(bool)
        if m.sum():
            profile.append({"slice": sl, "n": int(m.sum()),
                            **{f"mean_{k}": float(pd.Series(v)[m.to_numpy()].mean()) for k, v in outcomes.items()}})
    return rows, profile


# ---------------------------------------------------------------------------------------------------------------
def run(df: pd.DataFrame, events=None, *, label: str = "series", tod_minutes: int = 60, oos_fraction: float = 0.3, min_n: int = 20,
        fdr_q: float = 0.10, families: tuple[str, ...] = ("time_of_day", "weekday", "month", "holiday", "calendar", "events"),
        seed: int = C.DEFAULT_SEED, save: bool = True) -> C.ExperimentRecord:
    d = C.prepare_ohlcv(df)
    if len(d) < 300:
        raise C.ResearchLabError(f"Only {len(d)} bars -- a seasonality study needs at least a few hundred (months of data for weekday/month effects).")
    bar_min = C.median_bar_minutes(d)
    close = d["close"].to_numpy()
    bps = np.r_[0.0, np.diff(np.log(close))] * 1e4
    warnings: list[str] = []
    all_rows: list[dict] = []
    profiles: dict[str, list] = {}
    skipped: dict[str, str] = {}
    tod_minutes = int(min(max(tod_minutes, 5), 240))

    sess = _sessions(d, bps)
    span_days = (sess["day"].iloc[-1] - sess["day"].iloc[0]).days if len(sess) else 0
    sess["year"] = sess["day"].dt.year
    sess["wd"] = sess["day"].dt.weekday
    sess["month"] = sess["day"].dt.month
    sess_day_outcomes = {"return": sess["ret"].to_numpy(), "volatility": sess["rng"].to_numpy()}
    wd_series = sess["wd"]

    # ---- time of day -------------------------------------------------------------------------------------------
    if "time_of_day" in families:
        if bar_min > tod_minutes * 1.5 or bar_min >= 720:
            skipped["time_of_day"] = f"bars are ~{bar_min:.0f} min -- need intraday data finer than the {tod_minutes}-min bucket."
        else:
            a = _tod_obs(d, bps, tod_minutes)
            if len(a) < 100:
                skipped["time_of_day"] = "too few complete intraday buckets."
            else:
                a["year"] = a["b"].dt.year
                labels = {lb: (a["label"] == lb) for lb in sorted(a["label"].unique())}
                labels = {k: v for k, v in labels.items() if v.sum() >= min_n}
                if labels:
                    rows, prof = _family("time of day", a, labels, outcomes={"return": a["ret"].to_numpy(), "volatility": a["rng"].to_numpy()},
                                         oos_fraction=oos_fraction, min_n=min_n, seed=seed, exclusive=True)
                    all_rows += rows
                    profiles["time_of_day"] = prof
                else:
                    skipped["time_of_day"] = "no clock bucket has enough observations."

    # ---- weekday / month ---------------------------------------------------------------------------------------
    if "weekday" in families:
        if len(sess) < 40:
            skipped["weekday"] = f"only {len(sess)} complete sessions (need ~40+)."
        else:
            labels = {_WD[i]: (sess["wd"] == i) for i in range(7) if (sess["wd"] == i).sum() >= min_n}
            rows, prof = _family("weekday", sess, labels, outcomes=sess_day_outcomes, oos_fraction=oos_fraction, min_n=min_n, seed=seed, exclusive=True)
            all_rows += rows
            profiles["weekday"] = prof
    if "month" in families:
        if span_days < 270:
            skipped["month"] = f"data spans {span_days} days -- month-of-year effects need at least ~1 full year (preferably 5+)."
        else:
            labels = {_MO[m - 1]: (sess["month"] == m) for m in range(1, 13) if (sess["month"] == m).sum() >= min_n}
            if len(labels) >= 3:
                rows, prof = _family("month", sess, labels, outcomes=sess_day_outcomes, oos_fraction=oos_fraction, min_n=min_n, seed=seed, exclusive=True)
                all_rows += rows
                profiles["month"] = prof
                n_years = sess["year"].nunique()
                if n_years < 4:
                    warnings.append(f"Month effects rest on only {n_years} calendar year(s): one strong year can masquerade as a seasonal pattern. Check 'years_same_sign'.")
            else:
                skipped["month"] = "fewer than 3 months have enough sessions."

    # ---- holiday / calendar rules / events (weekday-adjusted baseline) -----------------------------------------
    days = sess["day"]
    if "holiday" in families and len(sess) >= 60:
        hol = _holiday_days(days.iloc[0], days.iloc[-1])
        before, after, on = _neighbour_flags(days, hol)
        near = before | after
        labels = {"session before a holiday": before, "session after a holiday": after, "holiday-adjacent (either)": near}
        if on.sum() >= min_n // 2:
            labels["trades on the holiday itself"] = on
        labels = {k: v for k, v in labels.items() if v.sum() >= max(8, min_n // 2)}
        if labels:
            rows, prof = _family("holiday", sess, labels, outcomes=sess_day_outcomes, oos_fraction=oos_fraction,
                                 min_n=max(8, min_n // 2), seed=seed, adjust_weekday=wd_series)
            all_rows += rows
            profiles["holiday"] = prof
        else:
            skipped["holiday"] = "too few holiday-adjacent sessions in this date range."
    elif "holiday" in families:
        skipped["holiday"] = "needs ~60+ sessions."

    if "calendar" in families and len(sess) >= 60:
        ym = days.dt.to_period("M")
        last_of_month = days == days.groupby(ym).transform("max")
        first_of_month = days == days.groupby(ym).transform("min")
        tom = last_of_month | (days.groupby(ym).cumcount() < 3)
        q_end = last_of_month & days.dt.month.isin([3, 6, 9, 12])
        opex = _third_friday(days)
        triple = opex & days.dt.month.isin([3, 6, 9, 12])
        nfp = _first_friday(days)
        labels = {"monthly options expiry (3rd Friday)": opex, "quarterly triple witching": triple, "month-end session": last_of_month,
                  "first session of month": first_of_month, "turn of month (last + first 3)": tom, "quarter-end session": q_end,
                  "NFP Friday (1st Friday)": nfp}
        labels = {k: v for k, v in labels.items() if v.sum() >= max(8, min_n // 2)}
        if labels:
            rows, prof = _family("calendar rule", sess, labels, outcomes=sess_day_outcomes, oos_fraction=oos_fraction,
                                 min_n=max(8, min_n // 2), seed=seed, adjust_weekday=wd_series)
            all_rows += rows
            profiles["calendar"] = prof
        else:
            skipped["calendar"] = "date range too short for expiry / month-end rules."
    elif "calendar" in families:
        skipped["calendar"] = "needs ~60+ sessions."

    if "events" in families and events is not None and len(sess) >= 40:
        from app.research_labs.event_study import parse_events

        ev = parse_events(events)
        ev = ev[(ev["timestamp"] >= days.iloc[0]) & (ev["timestamp"] <= days.iloc[-1] + pd.Timedelta(days=1))]
        if len(ev) < 8:
            skipped["events"] = f"only {len(ev)} events fall inside the data range (need 8+)."
        else:
            labels = {}
            for nm, g in ev.groupby("event"):
                if len(g) >= 8:
                    b, a_, o = _neighbour_flags(days, pd.DatetimeIndex(g["timestamp"].dt.normalize()))
                    labels[f"{nm} day"] = o
                    labels[f"{nm}: session before"] = b
                    labels[f"{nm}: session after"] = a_
            if len(ev["event"].unique()) > 1:
                b, a_, o = _neighbour_flags(days, pd.DatetimeIndex(ev["timestamp"].dt.normalize()))
                labels["any event day"] = o
            labels = {k: v for k, v in labels.items() if v.sum() >= 8}
            if labels:
                rows, prof = _family("event", sess, labels, outcomes=sess_day_outcomes, oos_fraction=oos_fraction, min_n=8, seed=seed,
                                     adjust_weekday=wd_series)
                all_rows += rows
                profiles["events"] = prof
            else:
                skipped["events"] = "no event series had 8+ matching sessions."
    elif "events" in families and events is not None:
        skipped["events"] = "needs ~40+ sessions."

    if not all_rows:
        raise C.ResearchLabError("Nothing could be tested: " + ("; ".join(f"{k}: {v}" for k, v in skipped.items()) or "no usable data") +
                                 ". Use intraday data spanning several months (a year or more for month effects).")

    # ---- global correction, verdicts, persistence ---------------------------------------------------------------
    qs = C.bh_adjust([r["p"] for r in all_rows])
    for r, q in zip(all_rows, qs):
        r["q_global"] = q
        same = (r["in_sample_mean"] is not None and r["oos_mean"] is not None and np.sign(r["in_sample_mean"]) == np.sign(r["oos_mean"])
                and r["oos_n"] >= max(5, min_n // 3))
        r["oos_same_sign"] = bool(same)
        if r["n"] < (8 if r["family"] in ("holiday", "calendar rule", "event") else min_n):
            r["verdict"] = "too few observations"
        elif q is not None and q < fdr_q and same:
            r["verdict"] = "promising: significant after correction AND same sign out-of-sample"
        elif q is not None and q < fdr_q:
            r["verdict"] = "significant in the full sample but NOT confirmed out-of-sample"
        else:
            r["verdict"] = "no reliable difference (not significant after multiple-testing correction)"
    n_tests = len(all_rows)
    discovered = [r for r in all_rows if r.get("is_p") is not None and r["is_p"] < 0.05 and r["oos_mean"] is not None and r["in_sample_mean"] is not None]
    kept = [r for r in discovered if np.sign(r["in_sample_mean"]) == np.sign(r["oos_mean"])]
    persistence = {
        "slices_tested": n_tests, "expected_false_positives_at_5pct": round(0.05 * n_tests, 1),
        "nominal_in_sample_hits": len(discovered), "kept_sign_out_of_sample": len(kept),
        "replication_rate": (len(kept) / len(discovered)) if discovered else None,
        "survive_correction_and_oos": int(sum(r["verdict"].startswith("promising") for r in all_rows)),
        "reading": ("no in-sample hits to replicate" if not discovered else
                    "in-sample hits replicate no better than a coin flip -- treat them as noise" if len(kept) / len(discovered) < 0.6 else
                    "in-sample hits replicate more often than chance"),
        "note": "If nothing were real, about half of in-sample 'hits' would keep their sign out-of-sample by coincidence.",
    }
    ranked = sorted(all_rows, key=lambda r: (r["q_global"] if r["q_global"] is not None else 9, -abs(r.get("effect_d") or 0)))
    top = [{k: r.get(k) for k in ("group", "n", "mean", "unit", "p", "q_global", "family_perm_p", "in_sample_mean", "oos_mean", "years", "years_same_sign", "verdict")}
           for r in ranked[:12]]
    fam_summary = []
    for fam in dict.fromkeys(r["family"] for r in all_rows):
        rs = [r for r in all_rows if r["family"] == fam]
        fam_summary.append({"family": fam, "slices_x_outcomes": len(rs), "promising": int(sum(r["verdict"].startswith("promising") for r in rs)),
                            "best_q_global": min((r["q_global"] for r in rs if r["q_global"] is not None), default=None),
                            "family_perm_p_return": next((r["family_perm_p"] for r in rs if r["outcome"] == "return" and r["family_perm_p"] is not None), None),
                            "family_perm_p_volatility": next((r["family_perm_p"] for r in rs if r["outcome"] == "volatility" and r["family_perm_p"] is not None), None)})
    for fam, why in skipped.items():
        warnings.append(f"Skipped {fam.replace('_', ' ')}: {why}")
    if span_days < 180:
        warnings.append(f"Only {span_days} days of data: seasonal effects are easy to fake in short samples -- treat everything as exploratory.")
    n_ok = persistence["survive_correction_and_oos"]
    headline = (f"{n_tests} calendar slices tested ({round(0.05 * n_tests, 1)} would look 'significant' at 5% by chance); "
                f"{n_ok} survive multiple-testing correction AND hold out-of-sample" + (f" -- best: {top[0]['group']}" if n_ok and top else "") +
                ("" if n_ok else " -- no seasonal edge you should trust yet"))
    results = {"headline": headline, "family_summary": fam_summary, "top_slices": top, "persistence": persistence, "profiles": profiles,
               "slices": [{k: v for k, v in r.items() if k not in ("slice",)} for r in all_rows], "skipped": skipped,
               "data_span_days": span_days, "sessions": int(len(sess)), "bar_minutes": bar_min}
    return C.record_from(LAB, TITLE, {"tod_minutes": tod_minutes, "oos_fraction": oos_fraction, "min_n": min_n, "fdr_q": fdr_q,
                                      "families": list(families), "n_events": 0 if events is None else int(len(events))},
                         ASSUMPTIONS, d, results, warnings, seed, label, save=save)
