"""Effective number of independent trials (report 2026-10-20, section 4: DSR "keep 0.95 with effective trial clustering").

The Deflated Sharpe Ratio penalises a result for how many things were tried. Counting every genome the optimizer evaluated
over-penalises (hundreds of near-identical parameter sets are ONE idea) while ignoring family-level searching under-penalises.
The right count is the number of distinct return streams: cluster candidates by how correlated their daily P&L is.

    effective_trials = number of clusters at |corr| >= corr_threshold   (average linkage)

``scale_trial_count`` applies the cluster ratio observed on a sample (the GA leaderboard) to the full evaluation count, never
returning less than the number of clusters observed nor more than the raw count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_pnl_series(trades, label: str | None = None) -> pd.Series:
    """Per-calendar-day net P&L of a trade list (exit date), as a Series indexed by normalized date."""
    rows = {}
    for t in trades or []:
        ts = getattr(t, "exit_time", None) or getattr(t, "entry_time", None)
        pnl = getattr(t, "pnl", None)
        if ts is None or pnl is None:
            continue
        d = pd.Timestamp(ts).normalize()
        rows[d] = rows.get(d, 0.0) + float(pnl)
    return pd.Series(rows, dtype=float, name=label).sort_index()


def correlation_matrix(streams: list[pd.Series], min_overlap: int = 20) -> np.ndarray:
    """Pairwise correlation over the union of days (missing = no trade = 0 P&L). Pairs sharing no activity are uncorrelated."""
    if not streams:
        return np.zeros((0, 0))
    df = pd.concat(streams, axis=1).sort_index().fillna(0.0)
    n = df.shape[1]
    if df.shape[0] < min_overlap:
        return np.eye(n)
    c = np.corrcoef(df.to_numpy(dtype=float).T)
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 1.0)
    return c


def effective_trial_count(streams: list[pd.Series], corr_threshold: float = 0.90) -> dict:
    """Cluster return streams by correlation; return the number of clusters and the membership."""
    n = len(streams)
    if n <= 1:
        return {"n_streams": n, "n_clusters": max(n, 1), "labels": [0] * n, "corr_threshold": corr_threshold}
    c = correlation_matrix(streams)
    dist = 1.0 - c
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, 2.0)
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        z = linkage(squareform(dist, checks=False), method="average")
        labels = fcluster(z, t=1.0 - float(corr_threshold), criterion="distance") - 1
    except Exception:  # noqa: BLE001 -- greedy fallback without scipy
        labels = -np.ones(n, dtype=int)
        k = 0
        for i in range(n):
            if labels[i] >= 0:
                continue
            labels[i] = k
            for j in range(i + 1, n):
                if labels[j] < 0 and c[i, j] >= corr_threshold:
                    labels[j] = k
            k += 1
    return {"n_streams": n, "n_clusters": int(len(set(labels.tolist()))), "labels": [int(x) for x in labels], "corr_threshold": corr_threshold}


def scale_trial_count(raw_trials: int, sample_streams: list[pd.Series], corr_threshold: float = 0.90) -> dict:
    """Effective trials for a search of ``raw_trials`` evaluations, using the cluster ratio seen on ``sample_streams``."""
    raw = max(1, int(raw_trials))
    info = effective_trial_count(sample_streams, corr_threshold)
    n, k = max(1, info["n_streams"]), max(1, info["n_clusters"])
    eff = int(round(raw * k / n))
    eff = max(k, min(raw, eff))
    info.update({"raw_trials": raw, "effective_trials": eff})
    return info


def representatives(streams: list[pd.Series], corr_threshold: float = 0.90) -> list[int]:
    """One representative index per correlation cluster -- the FIRST member in input order (a ranked list keeps its best-ranked stream).
    2026-11-04 audit (P0-3): PBO on a leaderboard of near-duplicates (same entry rule, nudged parameters) measures nothing, because the
    'in-sample winner' and its out-of-sample rank are decided by ties between clones. Run it on one stream per cluster instead."""
    info = effective_trial_count(streams, corr_threshold)
    seen: dict[int, int] = {}
    for i, lab in enumerate(info["labels"]):
        seen.setdefault(int(lab), i)
    return sorted(seen.values())


def cluster_leaderboard(trade_lists: list, corr_threshold: float = 0.90) -> list[dict]:
    """Group a ranked leaderboard (best first) by return-stream correlation (2026-11-04 audit, P2-3). Returns one row per entry:
    {"rank", "cluster", "representative"} where ``representative`` marks the best-ranked member of its cluster. A leaderboard of ten
    near-clones is one idea, not ten; showing and promoting clusters keeps the search from crowning (and PBO/DSR from counting) the same
    edge repeatedly."""
    streams = [daily_pnl_series(t) for t in trade_lists]
    info = effective_trial_count(streams, corr_threshold)
    reps = set(representatives(streams, corr_threshold)) if streams else set()
    return [{"rank": i + 1, "cluster": int(info["labels"][i]), "representative": i in reps} for i in range(len(streams))]


# ---------------------------------------------------------------------------------------------------------------------------
# Problem 2 (2026-11): Optimal Number of Clusters (ONC) + sensitivity report
# ---------------------------------------------------------------------------------------------------------------------------

SENSITIVITY_THRESHOLDS = (0.80, 0.90, 0.95)


def onc_cluster_count(streams: list[pd.Series], *, seed: int = 0, max_k: int | None = None) -> dict:
    """Optimal Number of Clusters (Lopez de Prado): k-means on the rows of the correlation-DISTANCE matrix, k chosen to maximise
    the silhouette quality q = mean(s) / std(s). Unlike a fixed average-linkage cut it has no correlation threshold to hand-pick,
    so it is the primary count; the threshold cuts are reported beside it as sensitivity.

    Degenerate shapes are handled explicitly rather than silently: fewer than 3 streams -> one cluster per stream (no basis to merge);
    every pair correlated above 0.95 -> one cluster."""
    n = len(streams)
    if n < 3:
        return {"n_streams": n, "n_clusters": max(n, 1), "labels": list(range(n)), "quality": None, "method": "onc (too few streams)"}
    c = correlation_matrix(streams)
    off = c[np.triu_indices(n, k=1)]
    if off.size and float(off.min()) >= 0.95:
        return {"n_streams": n, "n_clusters": 1, "labels": [0] * n, "quality": None, "method": "onc (all streams >= 0.95 correlated)"}
    dist = np.sqrt(np.clip((1.0 - c) / 2.0, 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_samples
    except Exception:  # noqa: BLE001 -- no sklearn: fall back to the average-linkage count at 0.90, flagged in `method`
        info = effective_trial_count(streams, 0.90)
        return {"n_streams": n, "n_clusters": info["n_clusters"], "labels": info["labels"], "quality": None,
                "method": "average-linkage 0.90 (sklearn unavailable, ONC not run)"}
    best = None
    kmax = min(n - 1, int(max_k) if max_k else n - 1)
    for k in range(2, kmax + 1):
        try:
            lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(dist)
            if len(set(lab.tolist())) < 2:
                continue
            s = silhouette_samples(dist, lab, metric="euclidean")
            sd = float(np.std(s))
            q = float(np.mean(s)) / sd if sd > 1e-12 else float(np.mean(s)) * 1e6
        except Exception:  # noqa: BLE001
            continue
        if best is None or q > best[0]:
            best = (q, k, lab)
    if best is None:
        return {"n_streams": n, "n_clusters": 1, "labels": [0] * n, "quality": None, "method": "onc (no valid split)"}
    return {"n_streams": n, "n_clusters": int(best[1]), "labels": [int(x) for x in best[2]], "quality": float(best[0]), "method": "onc"}


def _scale(raw: int, k: int, n: int) -> int:
    """Cluster ratio seen on the sample applied to the full evaluation count; never below the clusters actually observed, never above raw."""
    raw, k, n = max(1, int(raw)), max(1, int(k)), max(1, int(n))
    return max(k, min(raw, int(round(raw * k / n))))


def effective_trials_report(raw_trials: int, streams: list[pd.Series], *, primary_corr: float = 0.90,
                            thresholds=SENSITIVITY_THRESHOLDS, seed: int = 0) -> dict:
    """The effective trial count the DSR deflates by, with its sensitivity.

    ``used`` is the LARGER (more conservative) of the ONC count and the average-linkage count at ``primary_corr``, each scaled from the
    sampled streams up to ``raw_trials`` and floored at the clusters observed. ``sensitivity`` shows the count at 0.80 / 0.90 / 0.95
    so a reader can see how much the answer depends on the clustering choice. Never less than the measured cluster count, never more
    than the raw count."""
    raw = max(1, int(raw_trials))
    n = len(streams)
    onc = onc_cluster_count(streams, seed=seed)
    sens = {}
    for thr in sorted(set(float(t) for t in thresholds) | {float(primary_corr)}):
        k = effective_trial_count(streams, thr)["n_clusters"] if n >= 2 else max(n, 1)
        sens[f"{thr:.2f}"] = {"clusters": int(k), "effective_trials": _scale(raw, k, n)}
    link = sens[f"{float(primary_corr):.2f}"]
    onc_eff = _scale(raw, onc["n_clusters"], n)
    used = max(onc_eff, link["effective_trials"])
    measured = max(onc["n_clusters"], link["clusters"])
    used = max(measured, min(raw, used))
    return {
        "raw_trials": raw, "n_streams": n, "onc_clusters": int(onc["n_clusters"]), "onc_effective_trials": onc_eff,
        "onc_quality": onc.get("quality"), "onc_method": onc["method"], "primary_corr": float(primary_corr),
        "linkage_clusters": int(link["clusters"]), "sensitivity": sens, "effective_trials": int(used),
        "measured_clusters": int(measured), "labels": onc["labels"],
    }


class StreamSampler:
    """Reservoir sample of candidate daily-P&L streams drawn from ANYWHERE in a search (not only the final leaderboard, which is
    selected for being good and therefore looks more alike than the population that was actually tried). ``offer`` is O(1) and
    keeps at most ``capacity`` streams, uniformly."""

    def __init__(self, capacity: int = 64, seed: int = 0):
        import random
        self.capacity, self._rng, self._seen, self.streams = int(capacity), random.Random(seed), 0, []

    def offer(self, trades) -> None:
        s = daily_pnl_series(trades)
        if s.empty:
            return
        self._seen += 1
        if len(self.streams) < self.capacity:
            self.streams.append(s)
        else:
            j = self._rng.randrange(self._seen)
            if j < self.capacity:
                self.streams[j] = s
