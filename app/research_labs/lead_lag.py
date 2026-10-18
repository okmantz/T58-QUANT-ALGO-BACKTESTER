"""Lab 4 -- Cross-Asset Lead-Lag Analyzer.

Does a move in market A tend to PRECEDE a move in market B, at which horizon, and does that survive out-of-sample?

This is a lead-lag *matrix*, not a correlation heat-map. For every ordered pair (leader -> follower) it reports:

* best lag chosen IN-SAMPLE only (largest |t|), with the correlation, its Fisher confidence interval and p-value;
* a multiple-testing correction (Bonferroni across lags inside a pair, then Benjamini-Hochberg across all pairs);
* lagged *predictive performance*: the P&L of trading the follower in the direction of the leader's lagged move
  (hit rate, mean bps per bar, t-stat, after a round-trip cost);
* STABILITY: the same lag re-estimated in K contiguous blocks -- how many keep the sign, and how much it varies;
* the same lag evaluated on held-out OUT-OF-SAMPLE data, never used for selection.

Only relationships that clear the correction, hold out-of-sample and are stable across blocks get the "robust lead"
verdict. Everything else is labelled honestly ("unstable", "no lead").
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from app.research_labs import common as C

LAB = "lead_lag"
TITLE = "Cross-Asset Lead-Lag Analyzer"
ASSUMPTIONS = [
    "Markets are aligned on a common clock by taking the LAST close in each bucket (default 15 minutes) and keeping only "
    "buckets present in every series -- sessions where one market is closed are dropped, not forward-filled.",
    "Returns are log returns of that last close. A lag of k means the leader's return k buckets AGO against the "
    "follower's return now, so the leader's information is strictly older than the follower's outcome.",
    "The lag is selected on the in-sample portion only; out-of-sample numbers are never used to choose anything.",
    "Predictive P&L trades the follower in the direction implied by the leader's lagged move and charges a round-trip "
    "cost (default 1 bp) each time the position flips.",
    "Correlation CIs use Fisher's z with the number of non-overlapping bucket pairs; p-values are corrected for the "
    "number of lags tested and then for the number of pairs (Benjamini-Hochberg).",
]


def align_series(series: dict[str, pd.DataFrame], freq: str = "15min") -> pd.DataFrame:
    cols = {}
    for name, df in series.items():
        d = C.prepare_ohlcv(df)
        s = d.set_index("timestamp")["close"].resample(freq).last().dropna()
        cols[name] = s
    joined = pd.concat(cols, axis=1, join="inner").dropna()
    if len(joined) < 100:
        raise C.ResearchLabError(f"Only {len(joined)} overlapping {freq} buckets across the selected markets -- they barely "
                                 "trade at the same time. Pick a coarser bucket (e.g. 1h or 1D) or markets with overlapping hours.")
    return np.log(joined).diff().dropna()


def _corr_p(r: float, n: int) -> float:
    if n < 5 or not math.isfinite(r):
        return 1.0
    r = max(min(r, 0.999999), -0.999999)
    t = r * math.sqrt((n - 2) / (1 - r * r))
    try:
        from scipy import stats

        return float(2 * stats.t.sf(abs(t), n - 2))
    except Exception:  # noqa: BLE001
        return float(math.erfc(abs(t) / math.sqrt(2.0)))


def _lag_corr(x: np.ndarray, y: np.ndarray, k: int) -> tuple[float, int]:
    if k >= len(x) - 5:
        return float("nan"), 0
    a, b = x[:-k], y[k:]
    if a.std() == 0 or b.std() == 0:
        return float("nan"), len(a)
    return float(np.corrcoef(a, b)[0, 1]), len(a)


def _predictive(x: np.ndarray, y: np.ndarray, k: int, sign: float, cost_bps: float) -> dict:
    sig = np.sign(x[:-k]) * sign
    r = y[k:] * sig
    flips = np.abs(np.diff(sig, prepend=0.0)) > 0
    net = r - flips * (cost_bps * 1e-4)
    if len(net) < 10:
        return {"n": int(len(net)), "mean_bps": None, "hit_rate": None, "t": None}
    t = C.one_sample_t(net)
    return {"n": int(len(net)), "mean_bps": float(net.mean() * 1e4), "hit_rate": float((r > 0).mean()),
            "t": t["t"], "p": t["p"], "gross_mean_bps": float(r.mean() * 1e4)}


def analyze_pair(rets: pd.DataFrame, leader: str, follower: str, max_lag: int = 8, oos_fraction: float = 0.3,
                 blocks: int = 6, cost_bps: float = 1.0) -> dict:
    x_all, y_all = rets[leader].to_numpy(float), rets[follower].to_numpy(float)
    cut = int(len(rets) * (1 - min(max(oos_fraction, 0.1), 0.6)))
    xi, yi, xo, yo = x_all[:cut], y_all[:cut], x_all[cut:], y_all[cut:]
    profile = []
    for k in range(1, max_lag + 1):
        r, n = _lag_corr(xi, yi, k)
        profile.append({"lag": k, "corr_is": r, "n": n, "p": _corr_p(r, n)})
    valid = [p for p in profile if math.isfinite(p["corr_is"])]
    if not valid:
        return {"leader": leader, "follower": follower, "verdict": "no data"}
    best = max(valid, key=lambda p: abs(p["corr_is"]) * math.sqrt(max(p["n"] - 2, 1)))
    k, r_is = best["lag"], best["corr_is"]
    p_pair = min(1.0, best["p"] * len(valid))                      # Bonferroni across lags
    lo, hi = C.fisher_ci(r_is, best["n"])
    r_oos, n_oos = _lag_corr(xo, yo, k)
    p_oos = _corr_p(r_oos, n_oos)
    sign = float(np.sign(r_is)) or 1.0
    pred_is = _predictive(xi, yi, k, sign, cost_bps)
    pred_oos = _predictive(xo, yo, k, sign, cost_bps)
    # stability: same lag in contiguous blocks of the whole sample
    edges = np.linspace(0, len(rets), blocks + 1, dtype=int)
    block_r = []
    for a, b in zip(edges[:-1], edges[1:]):
        r_b, n_b = _lag_corr(x_all[a:b], y_all[a:b], k)
        if math.isfinite(r_b) and n_b > 20:
            block_r.append(r_b)
    same = float(np.mean([np.sign(v) == sign for v in block_r])) if block_r else None
    return {"leader": leader, "follower": follower, "best_lag": k, "corr_is": r_is, "ci_low": lo, "ci_high": hi,
            "p_is_bonferroni": p_pair, "corr_oos": r_oos, "p_oos": p_oos, "n_is": best["n"], "n_oos": n_oos,
            "predictive_is": pred_is, "predictive_oos": pred_oos, "stability_same_sign": same,
            "stability_block_corr_std": float(np.std(block_r)) if block_r else None, "block_corrs": block_r,
            "lag_profile": profile}


def run(series: dict[str, pd.DataFrame], *, freq: str = "15min", max_lag: int = 8, oos_fraction: float = 0.3,
        cost_bps: float = 1.0, blocks: int = 6, seed: int = C.DEFAULT_SEED, save: bool = True) -> C.ExperimentRecord:
    if len(series) < 2:
        raise C.ResearchLabError("Pick at least two markets (e.g. ES, NQ, GC, a Treasury-yield proxy, the dollar, crude oil).")
    rets = align_series(series, freq)
    names = list(rets.columns)
    pairs = [analyze_pair(rets, a, b, max_lag, oos_fraction, blocks, cost_bps) for a, b in itertools.permutations(names, 2)]
    pairs = [p for p in pairs if "best_lag" in p]
    qs = C.bh_adjust([p["p_is_bonferroni"] for p in pairs])
    for p, q in zip(pairs, qs):
        p["q_value_bh"] = q
        oos_ok = (p["corr_oos"] is not None and math.isfinite(p["corr_oos"]) and np.sign(p["corr_oos"]) == np.sign(p["corr_is"])
                  and p["p_oos"] < 0.10)
        stable = p["stability_same_sign"] is not None and p["stability_same_sign"] >= 0.67
        edge = (p["predictive_oos"].get("mean_bps") or 0) > 0
        if q is not None and q < 0.10 and oos_ok and stable and edge:
            p["verdict"] = "robust lead"
        elif q is not None and q < 0.10:
            why = [w for w, ok in (("fails out-of-sample", oos_ok), ("unstable across blocks", stable), ("no net edge after cost", edge)) if not ok]
            p["verdict"] = "unstable: significant in-sample but " + ", ".join(why)
        else:
            p["verdict"] = "no lead"
    pairs.sort(key=lambda p: (p["verdict"] != "robust lead", p["q_value_bh"] if p["q_value_bh"] is not None else 1.0))
    matrix = {a: {b: None for b in names} for a in names}
    for p in pairs:
        matrix[p["leader"]][p["follower"]] = {"lag": p["best_lag"], "corr_is": p["corr_is"], "corr_oos": p["corr_oos"],
                                              "verdict": p["verdict"], "q": p["q_value_bh"], "stability": p["stability_same_sign"]}
    robust = [f"{p['leader']} -> {p['follower']} (lag {p['best_lag']})" for p in pairs if p["verdict"] == "robust lead"]
    headline = (f"{len(robust)} robust lead(s): " + "; ".join(robust[:5])) if robust else \
        f"No robust lead-lag among {len(names)} markets ({len(pairs)} ordered pairs tested) after correction + out-of-sample"
    warnings = []
    if len(rets) < 1000:
        warnings.append(f"Only {len(rets)} aligned returns; lead-lag estimates this short are noisy.")
    results = {"headline": headline, "markets": names, "freq": freq, "aligned_returns": int(len(rets)),
               "pairs_tested": len(pairs), "lags_per_pair": max_lag, "matrix": matrix, "pairs": pairs}
    primary = next(iter(series.values()))
    return C.record_from(LAB, TITLE, {"markets": names, "freq": freq, "max_lag": max_lag, "oos_fraction": oos_fraction,
                                      "cost_bps": cost_bps, "blocks": blocks}, ASSUMPTIONS, primary, results, warnings, seed,
                         ", ".join(names), save=save)


def lagged_signal(follower: pd.DataFrame, leader: pd.DataFrame, lag: int = 1, freq: str = "15min") -> pd.Series:
    """Per-bar sign of the leader's return ``lag`` buckets ago, aligned to the follower's bars (causal: only completed
    leader buckets are used). Use it to gate a follower strategy ('only long when the leader rose'); +1/-1/0."""
    f = C.prepare_ohlcv(follower)
    l = C.prepare_ohlcv(leader).set_index("timestamp")["close"].resample(freq).last().dropna()
    r = np.sign(np.log(l).diff()).shift(lag)
    # a leader bucket is only known after it ends: label it by its END time
    r.index = r.index + pd.Timedelta(freq)
    out = r.reindex(f["timestamp"], method="ffill")
    return pd.Series(out.fillna(0.0).to_numpy(), index=f.index)
