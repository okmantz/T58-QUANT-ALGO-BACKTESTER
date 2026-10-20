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
