import numpy as np
import pandas as pd

from app.data.spread_from_tape import corwin_schultz, measure_spread


def _tape(spread, n=6000, seed=1):
    rng = np.random.default_rng(seed)
    mid = 100 + np.cumsum(rng.normal(0, 0.05, n))
    return pd.DataFrame({"high": mid + abs(rng.normal(0, 0.08, n)) + spread / 2,
                         "low": mid - abs(rng.normal(0, 0.08, n)) - spread / 2, "close": mid}), mid


def test_estimator_recovers_planted_spread():
    df, _ = _tape(0.10)
    r = measure_spread(df, 0.01, 1.0)
    assert r["source"] == "estimated" and 8.0 < r["floor_pips"] < 11.0
    assert r["stress_pips"] >= r["floor_pips"] and r["verdict"] == "optimistic"


def test_quoted_spread_wins_and_is_exact():
    df, mid = _tape(0.0)
    df["bid"], df["ask"] = mid - 0.02, mid + 0.02
    r = measure_spread(df, 0.01, 1.0)
    assert r["source"] == "quoted" and abs(r["floor_pips"] - 4.0) < 1e-6


def test_covering_configuration_and_too_little_data():
    df, mid = _tape(0.0)
    df["bid"], df["ask"] = mid - 0.005, mid + 0.005
    assert measure_spread(df, 0.01, 5.0)["verdict"] == "covers_stress"
    assert measure_spread(df.head(10)[["high", "low"]], 0.01, 1.0)["verdict"] == "unmeasured"
    assert corwin_schultz(pd.DataFrame({"close": [1, 2, 3]})) is None
