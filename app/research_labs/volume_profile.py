"""Lab 1 -- Volume Profile & Market Profile Lab.

Maps volume by price (volume profile) and time-at-price (TPO / Market Profile) per session, and extracts the levels
traders act on: point of control (POC), value-area high / low (VAH / VAL), high-volume nodes (HVN) and low-volume
nodes (LVN).

The research question it answers (default study): *do liquidity-sweep setups behave differently around value-area
boundaries and low-volume zones?*  Locations are always measured against the PRIOR completed session's profile, so
nothing the study conditions on uses information from the bar being studied (no lookahead).

Data honesty
------------
* Volume profile from OHLCV distributes each bar's volume uniformly across the bar's high-low range. That is an
  approximation of where the volume really traded; with tick / trade-level data the result would be exact. The lab
  says so in every record.
* If the dataset has no volume at all, the profile is built from bar counts (time-at-price) and flagged.
* TPO needs only timestamps + OHLC, so it is exact for the bars you have.
"""
from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.research_labs import common as C

LAB = "volume_profile"
TITLE = "Volume Profile & Market Profile Lab"
ASSUMPTIONS = [
    "Each bar's volume is spread uniformly across its high-low range (OHLCV cannot reveal where inside the bar the "
    "volume actually traded). Trade/tick data would make the profile exact.",
    "Value area = the smallest price band around the POC holding the chosen share (default 70%) of the session's volume.",
    "HVN = local peak of the smoothed profile above 1.2x the mean node; LVN = local trough below 0.6x the mean node.",
    "Study locations are measured against the PRIOR completed session's profile, so no information from the studied "
    "bar or its session leaks into the conditioning variable.",
    "Forward outcomes are measured in the reversal direction of each sweep (SSL sweep -> long, BSL sweep -> short), in "
    "basis points, after the sweep bar closes.",
]


@dataclass
class Profile:
    edges: np.ndarray
    volume: np.ndarray
    poc: float
    vah: float
    val: float
    total: float
    hvn: list
    lvn: list
    volume_missing: bool = False

    @property
    def centers(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2.0

    def to_dict(self, max_rows: int = 80) -> dict:
        c = self.centers
        step = max(1, int(np.ceil(len(c) / max_rows)))
        return {"poc": self.poc, "vah": self.vah, "val": self.val, "total_volume": self.total,
                "hvn": self.hvn, "lvn": self.lvn, "volume_missing": self.volume_missing,
                "rows": [{"price": float(c[i]), "volume": float(self.volume[i:i + step].sum())}
                         for i in range(0, len(c), step)]}


def _spread_volume(lows: np.ndarray, highs: np.ndarray, vols: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n = len(edges) - 1
    b_lo = np.clip(np.searchsorted(edges, lows, side="right") - 1, 0, n - 1)
    b_hi = np.clip(np.searchsorted(edges, highs, side="right") - 1, 0, n - 1)
    b_hi = np.maximum(b_hi, b_lo)
    share = vols / (b_hi - b_lo + 1)
    diff = np.zeros(n + 1)
    np.add.at(diff, b_lo, share)
    np.add.at(diff, b_hi + 1, -share)
    return np.cumsum(diff)[:n]


def value_area(volume: np.ndarray, poc_idx: int, share: float = 0.70) -> tuple[int, int]:
    """Classic expansion: from the POC, repeatedly add the larger of the next two bins above/below."""
    total = volume.sum()
    if total <= 0:
        return poc_idx, poc_idx
    lo = hi = poc_idx
    acc = float(volume[poc_idx])
    n = len(volume)
    while acc < share * total and (lo > 0 or hi < n - 1):
        up = float(volume[hi + 1:hi + 3].sum()) if hi < n - 1 else -1.0
        dn = float(volume[max(lo - 2, 0):lo].sum()) if lo > 0 else -1.0
        if up >= dn and hi < n - 1:
            step = min(2, n - 1 - hi)
            hi += step
            acc += float(volume[hi - step + 1:hi + 1].sum())
        elif lo > 0:
            step = min(2, lo)
            lo -= step
            acc += float(volume[lo:lo + step].sum())
        else:
            break
    return lo, hi


def find_nodes(volume: np.ndarray, hvn_mult: float = 1.2, lvn_mult: float = 0.6) -> tuple[list[int], list[int]]:
    n = len(volume)
    if n < 5 or volume.sum() <= 0:
        return [], []
    k = np.array([0.25, 0.5, 0.25])
    sm = np.convolve(np.pad(volume, 1, mode="edge"), k, mode="valid")
    mean = sm.mean()
    hvn, lvn = [], []
    for i in range(1, n - 1):
        if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] >= hvn_mult * mean:
            hvn.append(i)
        if sm[i] <= sm[i - 1] and sm[i] < sm[i + 1] and sm[i] <= lvn_mult * mean:
            lvn.append(i)
    # an LVN must sit between two HVNs, otherwise it is just the tail of the profile
    if hvn:
        lvn = [i for i in lvn if hvn[0] < i < hvn[-1]]
    else:
        lvn = []
    return hvn, lvn


def build_profile(df: pd.DataFrame, n_bins: int = 60, share: float = 0.70, edges: np.ndarray | None = None) -> Profile:
    lows = df["low"].to_numpy(float)
    highs = df["high"].to_numpy(float)
    vols = df["volume"].to_numpy(float)
    missing = float(vols.sum()) <= 0
    if missing:
        vols = np.ones(len(df))
    if edges is None:
        lo, hi = float(lows.min()), float(highs.max())
        if hi <= lo:
            hi = lo + max(abs(lo) * 1e-6, 1e-9)
        edges = np.linspace(lo, hi, n_bins + 1)
    vol = _spread_volume(lows, highs, vols, edges)
    poc_i = int(np.argmax(vol))
    lo_i, hi_i = value_area(vol, poc_i, share)
    centers = (edges[:-1] + edges[1:]) / 2.0
    hvn_i, lvn_i = find_nodes(vol)
    return Profile(edges=edges, volume=vol, poc=float(centers[poc_i]), vah=float(edges[hi_i + 1]), val=float(edges[lo_i]),
                   total=float(vol.sum()), hvn=[float(centers[i]) for i in hvn_i], lvn=[float(centers[i]) for i in lvn_i],
                   volume_missing=missing)


def tpo_profile(day: pd.DataFrame, bin_size: float, period_minutes: int = 30, share: float = 0.70) -> dict:
    """Market Profile for one session. Each period contributes one TPO to every price bin its range touched."""
    if len(day) == 0 or bin_size <= 0:
        return {}
    lo = np.floor(day["low"].min() / bin_size) * bin_size
    hi = np.ceil(day["high"].max() / bin_size) * bin_size
    n = max(1, int(round((hi - lo) / bin_size)))
    edges = lo + bin_size * np.arange(n + 1)
    start = day["timestamp"].iloc[0]
    period = ((day["timestamp"] - start).dt.total_seconds() // (period_minutes * 60)).astype(int)
    letters = string.ascii_uppercase + string.ascii_lowercase
    counts = np.zeros(n, dtype=int)
    letter_rows = [[] for _ in range(n)]
    for pid, grp in day.groupby(period):
        b_lo = int(np.clip(np.searchsorted(edges, grp["low"].min(), side="right") - 1, 0, n - 1))
        b_hi = int(np.clip(np.searchsorted(edges, grp["high"].max(), side="right") - 1, 0, n - 1))
        ch = letters[int(pid) % len(letters)]
        for b in range(b_lo, b_hi + 1):
            counts[b] += 1
            letter_rows[b].append(ch)
    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_candidates = np.flatnonzero(counts == counts.max())
    mid = (n - 1) / 2.0
    poc_i = int(poc_candidates[np.argmin(np.abs(poc_candidates - mid))])
    lo_i, hi_i = value_area(counts.astype(float), poc_i, share)
    ib_end = start + pd.Timedelta(minutes=60)
    ib = day[day["timestamp"] < ib_end]
    interior = counts[1:-1] if n > 2 else np.array([])
    single_prints = [float(centers[i + 1]) for i, c in enumerate(interior) if c == 1]
    pos = poc_i / max(n - 1, 1)
    shape = "P (POC high: short-covering / upside acceptance)" if pos > 0.67 else \
            "b (POC low: long liquidation / downside acceptance)" if pos < 0.33 else "D (balanced / normal day)"
    return {
        "poc": float(centers[poc_i]), "vah": float(edges[hi_i + 1]), "val": float(edges[lo_i]),
        "ib_high": float(ib["high"].max()) if len(ib) else None, "ib_low": float(ib["low"].min()) if len(ib) else None,
        "single_prints": single_prints, "shape": shape, "periods": int(period.nunique()),
        "rows": [{"price": float(centers[i]), "tpo": int(counts[i]), "letters": "".join(letter_rows[i])}
                 for i in range(n - 1, -1, -1)],
    }


def _nice(x: float) -> float:
    """Round up to 1 / 2 / 2.5 / 5 x 10^k so the bin size is stable against small differences in the data it was derived from."""
    if not np.isfinite(x) or x <= 0:
        return 1e-9
    e = 10 ** np.floor(np.log10(x))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if x <= m * e * 1.0000001:
            return float(m * e)
    return float(10 * e)


def _bin_size(df: pd.DataFrame, divisor: int = 40, warmup_sessions: int = 5) -> float:
    """Price-bin width, fixed from the FIRST few sessions only -- so adding later data can never move an earlier session's POC /
    value area (a bin size taken from the whole sample would be a quiet form of look-ahead)."""
    day = df["timestamp"].dt.normalize()
    first_days = sorted(day.unique())[:warmup_sessions]
    head = df[day.isin(first_days)]
    hday = head["timestamp"].dt.normalize()
    rng = (head.groupby(hday)["high"].max() - head.groupby(hday)["low"].min()).median()
    floor = abs(float(df["close"].iloc[0])) * 1e-6
    return _nice(float(max(rng / divisor, floor, 1e-9)))


def session_levels(df: pd.DataFrame, share: float = 0.70, bin_size: float | None = None) -> pd.DataFrame:
    """Per-session (calendar day) POC / VAH / VAL plus the nearest HVN/LVN lists. Index = session date."""
    bs = bin_size or _bin_size(df)
    day = df["timestamp"].dt.normalize()
    rows = {}
    for d, g in df.groupby(day):
        if len(g) < 3:
            continue
        lo = np.floor(g["low"].min() / bs) * bs
        hi = np.ceil(g["high"].max() / bs) * bs
        n = max(2, int(round((hi - lo) / bs)))
        p = build_profile(g, edges=lo + bs * np.arange(n + 1), share=share)
        rows[d] = {"poc": p.poc, "vah": p.vah, "val": p.val, "hvn": p.hvn, "lvn": p.lvn, "volume": p.total}
    return pd.DataFrame.from_dict(rows, orient="index")


def profile_feature_frame(df: pd.DataFrame, share: float = 0.70) -> pd.DataFrame:
    """Causal per-bar features vs the PRIOR session's profile (what ``indicators_v10`` exposes to every engine)."""
    levels = session_levels(df, share=share)
    out = pd.DataFrame(index=df.index)
    day = df["timestamp"].dt.normalize()
    prev = levels.shift(1)
    atr = C.atr_series(df).replace(0, np.nan)
    out["vp_poc"] = day.map(prev["poc"]).to_numpy(float) if len(prev) else np.nan
    out["vp_vah"] = day.map(prev["vah"]).to_numpy(float) if len(prev) else np.nan
    out["vp_val"] = day.map(prev["val"]).to_numpy(float) if len(prev) else np.nan
    close = df["close"]
    out["vp_dist_poc"] = (close - out["vp_poc"]) / atr
    out["vp_dist_vah"] = (close - out["vp_vah"]) / atr
    out["vp_dist_val"] = (close - out["vp_val"]) / atr
    out["vp_location"] = np.where(close > out["vp_vah"], 1.0, np.where(close < out["vp_val"], -1.0, 0.0))
    out.loc[out["vp_poc"].isna(), "vp_location"] = np.nan
    bs = _bin_size(df)
    prev_lvn = prev["lvn"] if len(prev) else pd.Series(dtype=object)
    prev_hvn = prev["hvn"] if len(prev) else pd.Series(dtype=object)
    lvn_map = day.map(prev_lvn)
    hvn_map = day.map(prev_hvn)
    cl = close.to_numpy(float)

    def near(levels_series: pd.Series) -> np.ndarray:
        res = np.zeros(len(levels_series))
        for i, lv in enumerate(levels_series.to_numpy()):
            if isinstance(lv, list) and lv:
                res[i] = 1.0 if np.min(np.abs(np.asarray(lv) - cl[i])) <= 1.5 * bs else 0.0
        return res

    out["vp_in_lvn"] = near(lvn_map)
    out["vp_in_hvn"] = near(hvn_map)
    return out


def _location_label(loc: float, in_lvn: float) -> str:
    base = "above prior VAH" if loc > 0 else "below prior VAL" if loc < 0 else "inside prior value area"
    return base + (" + at an LVN" if in_lvn else "")


def sweep_value_study(df: pd.DataFrame, horizon: int = 12, lookback: int = 20, oos_fraction: float = 0.3,
                      min_n: int = 15, seed: int = C.DEFAULT_SEED) -> dict:
    feats = profile_feature_frame(df)
    sweeps = C.detect_liquidity_sweeps(df, lookback=lookback)
    outcome = C.sweep_directional_return(df, sweeps, horizon) * 1e4  # bps
    is_event = (sweeps["ssl_sweep"] | sweeps["bsl_sweep"]) & outcome.notna() & feats["vp_location"].notna()
    cut = int(len(df) * (1 - min(max(oos_fraction, 0.1), 0.6)))
    idx = np.arange(len(df))
    groups = {}
    for name, mask in {
        "above prior VAH": feats["vp_location"] > 0, "inside prior value area": feats["vp_location"] == 0,
        "below prior VAL": feats["vp_location"] < 0, "at an LVN": feats["vp_in_lvn"] > 0,
        "away from LVNs": feats["vp_in_lvn"] == 0, "at an HVN": feats["vp_in_hvn"] > 0,
    }.items():
        groups[name] = mask
    rows, pvals = [], []
    all_vals = outcome[is_event].to_numpy()
    for name, mask in groups.items():
        sel = is_event & mask
        in_s, out_s = sel & (idx < cut), sel & (idx >= cut)
        grp, rest = outcome[sel].to_numpy(), outcome[is_event & ~mask].to_numpy()
        w = C.welch_test(grp, rest)
        pvals.append(w["p"])
        d = C.describe_group(grp, seed=seed)
        lo, hi = C.diff_ci(grp, rest, seed=seed, n_boot=400)
        rows.append({"group": name, **d, "vs_rest_t": w["t"], "vs_rest_p": w["p"], "effect_d": C.cohens_d(grp, rest),
                     "diff_ci_low": lo, "diff_ci_high": hi,
                     "in_sample_mean": float(outcome[in_s].mean()) if in_s.sum() else None, "in_sample_n": int(in_s.sum()),
                     "oos_mean": float(outcome[out_s].mean()) if out_s.sum() else None, "oos_n": int(out_s.sum())})
    for row, q in zip(rows, C.bh_adjust(pvals)):
        row["q_value_bh"] = q
        sign_ok = (row["in_sample_mean"] is not None and row["oos_mean"] is not None
                   and np.sign(row["in_sample_mean"]) == np.sign(row["oos_mean"]) and row["oos_n"] >= max(5, min_n // 3))
        row["verdict"] = ("promising: significant after correction AND same sign out-of-sample"
                          if (q is not None and q < 0.10 and row["n"] >= min_n and sign_ok)
                          else "too few events" if row["n"] < min_n
                          else "no reliable difference (not significant after multiple-testing correction, or OOS disagrees)")
    base = C.describe_group(all_vals, seed=seed)
    return {"horizon_bars": horizon, "lookback": lookback, "events": int(is_event.sum()), "baseline_all_sweeps": base, "groups": rows}


def run(df: pd.DataFrame, *, label: str = "", n_bins: int = 60, value_area_pct: float = 70.0, horizon: int = 12,
        sweep_lookback: int = 20, oos_fraction: float = 0.3, tpo_minutes: int = 30, seed: int = C.DEFAULT_SEED,
        save: bool = True) -> C.ExperimentRecord:
    d = C.prepare_ohlcv(df)
    share = min(max(value_area_pct / 100.0, 0.5), 0.95)
    warnings = []
    if float(d["volume"].sum()) <= 0:
        warnings.append("This dataset has no volume: the profile is built from time-at-price (bar counts), not volume.")
    full = build_profile(d.iloc[-min(len(d), 5000):], n_bins=n_bins, share=share)
    last_day = d[d["timestamp"].dt.normalize() == d["timestamp"].dt.normalize().iloc[-1]]
    bs = _bin_size(d)
    tpo = tpo_profile(last_day, bs, tpo_minutes, share) if len(last_day) >= 3 else {}
    study = sweep_value_study(d, horizon=horizon, lookback=sweep_lookback, oos_fraction=oos_fraction, seed=seed)
    if study["events"] < 30:
        warnings.append(f"Only {study['events']} sweep events fall inside this sample -- group results are weak evidence.")
    promising = [g["group"] for g in study["groups"] if str(g["verdict"]).startswith("promising")]
    headline = (f"{study['events']} sweeps studied; promising location(s): {', '.join(promising)}" if promising
                else f"{study['events']} sweeps studied; no location difference survived multiple-testing + out-of-sample")
    results = {"headline": headline, "profile_recent": full.to_dict(), "tpo_last_session": tpo, "sweep_study": study,
              "session_levels_tail": session_levels(d.iloc[-3000:]).tail(8).reset_index().rename(columns={"index": "session"}).to_dict(orient="records")}
    return C.record_from(LAB, TITLE, {"n_bins": n_bins, "value_area_pct": value_area_pct, "horizon": horizon,
                                      "sweep_lookback": sweep_lookback, "oos_fraction": oos_fraction,
                                      "tpo_minutes": tpo_minutes}, ASSUMPTIONS, d, results, warnings, seed, label, save=save)
