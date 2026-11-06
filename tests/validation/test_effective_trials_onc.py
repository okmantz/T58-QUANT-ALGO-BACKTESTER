"""Problem 2: ONC effective-trial count with sensitivity; never below measured clusters, never above raw."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.validation.effective_trials import StreamSampler, effective_trials_report, onc_cluster_count


def _clones(n_ideas=3, per=8, noise=0.25, seed=1, days=400):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=days)
    base = [rng.normal(0, 1, days) for _ in range(n_ideas)]
    return [pd.Series(b + rng.normal(0, noise, days), index=idx) for b in base for _ in range(per)]


def test_onc_finds_the_planted_number_of_ideas():
    assert onc_cluster_count(_clones())["n_clusters"] == 3


def test_report_scales_to_raw_and_exposes_sensitivity():
    r = effective_trials_report(30000, _clones())
    assert r["onc_clusters"] == 3 and set(r["sensitivity"]) >= {"0.80", "0.90", "0.95"}
    assert r["measured_clusters"] <= r["effective_trials"] <= r["raw_trials"]
    assert r["effective_trials"] == 30000 * 3 // 24            # 3 clusters of 24 streams scaled to the raw count


def test_independent_streams_are_not_collapsed():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2023-01-02", periods=400)
    ind = [pd.Series(rng.normal(0, 1, 400), index=idx) for _ in range(20)]
    r = effective_trials_report(30000, ind)
    assert r["effective_trials"] == 30000                      # the larger (average-linkage) count wins; ONC's k=2 is not trusted alone


def test_floor_is_measured_clusters_and_cap_is_raw():
    r = effective_trials_report(5, _clones())
    assert r["effective_trials"] >= r["measured_clusters"] or r["effective_trials"] == 5
    assert r["effective_trials"] <= 5 or r["measured_clusters"] > 5


def test_sampler_is_a_bounded_uniform_reservoir():
    class T:
        def __init__(self, d, p): self.exit_time, self.entry_time, self.pnl = pd.Timestamp(d), pd.Timestamp(d), p
    s = StreamSampler(capacity=5, seed=1)
    for i in range(50):
        s.offer([T("2024-01-02", float(i)), T("2024-01-03", 1.0)])
    assert len(s.streams) == 5
