import numpy as np
import pandas as pd

from app.data import importer


def _frame():
    ts = pd.date_range("2025-12-01 00:00", "2026-03-30 23:55", freq="15min")
    ts = ts[ts.dayofweek < 5]
    r = np.random.default_rng(0)
    close = 5000 + np.cumsum(r.normal(0, 0.6, len(ts)))
    # an unadjusted +45pt contract flip at the first bar of 2026-03-12 (inside the March roll window), no session break
    open_ = np.r_[close[0], close[:-1]]
    off = np.where(ts >= pd.Timestamp("2026-03-12"), 45.0, 0.0)
    close, open_ = close + off, open_ + off
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": np.maximum(open_, close) + 0.3, "low": np.minimum(open_, close) - 0.3,
                         "close": close, "volume": 100})


def test_adjustment_is_data_driven_not_filename_gated():
    df = importer._maybe_back_adjust_continuous(_frame(), "data.csv", [])          # a name that says nothing about futures
    assert df.attrs.get("roll_adjusted") is True


def test_non_roll_data_with_plain_name_is_left_alone():
    ts = pd.date_range("2025-12-01", periods=3000, freq="15min")
    r = np.random.default_rng(1)
    c = 1.1 + np.cumsum(r.normal(0, 0.0002, 3000))
    df = pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1e-4, "low": c - 1e-4, "close": c, "volume": 1})
    out = importer._maybe_back_adjust_continuous(df, "eurusd.csv", [])
    assert out is df or not out.attrs.get("roll_adjusted")
