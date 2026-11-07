import numpy as np
import pandas as pd

from app.discovery_engine.edge_hypotheses import hypothesis_first_families, measure_edges


def _bars(days=260, seed=3, gap_fade=0.0):
    rng = np.random.default_rng(seed)
    rows, px = [], 100.0
    for d in pd.bdate_range("2022-01-03", periods=days):
        gap = rng.normal(0, 0.006)
        o = px * (1 + gap)
        day_drift = -gap * gap_fade
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=78, freq="5min")
        steps = rng.normal(day_drift / 78, 0.0008, 78)
        c = o * np.exp(np.cumsum(steps))
        op = np.r_[o, c[:-1]]
        for t, a, b in zip(idx, op, c):
            rows.append((t, a, max(a, b) * 1.0002, min(a, b) * 0.9998, b, 1000))
        px = c[-1]
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")


def test_planted_gap_fade_is_confirmed_and_noise_is_not():
    planted = measure_edges(_bars(gap_fade=0.8), cost_bps=1.5)
    f = {x["hypothesis"]: x for x in planted["findings"]}
    assert f["gap_fade"]["verdict"] == "confirmed" and f["gap_fade"]["evidence"]["net_bps"] > 0
    assert "overnight_gap_fade" in planted["families"]
    noise = measure_edges(_bars(gap_fade=0.0, seed=11), cost_bps=1.5)
    g = {x["hypothesis"]: x for x in noise["findings"]}
    assert g["gap_fade"]["verdict"] != "confirmed"


def test_falls_back_to_all_families_when_nothing_found():
    fams, rep = hypothesis_first_families(_bars(gap_fade=0.0, seed=21))
    assert len(fams) > 50 and "headline" in rep
