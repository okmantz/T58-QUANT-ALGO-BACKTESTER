"""Lab features exposed as ordinary indicators so every strategy engine can use them."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy import indicators_v10 as v10
from app.strategy.indicators import build_indicator_series


def _frame(n=3000):
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=n, freq="5min"), "open": o,
                         "high": np.maximum(o, c) * 1.0004, "low": np.minimum(o, c) * 0.9996, "close": c, "volume": rng.integers(100, 900, n)})


@pytest.mark.parametrize("kind", list(v10.V10_INDICATOR_KINDS))
def test_every_kind_builds_through_the_normal_indicator_entry_point(kind):
    f = _frame()
    s = build_indicator_series(f, kind, 14, "close", 14)
    assert len(s) == len(f) and not s.isna().any()


def test_self_contained_kinds_are_registered_in_the_search_grammar():
    from app.search import grammar

    for k in v10.V10_SELF_CONTAINED:
        assert k in grammar.NUMERIC_KINDS


def test_volume_profile_features_are_causal():
    f = _frame()
    full = build_indicator_series(f, "vp_dist_poc", 14, "close", 14)
    part = build_indicator_series(f.iloc[:1800].copy(), "vp_dist_poc", 14, "close", 14)
    assert np.allclose(full.iloc[:900].to_numpy(), part.iloc[:900].to_numpy(), equal_nan=True)


def test_event_window_uses_attached_events():
    f = _frame()
    f.attrs["events"] = pd.DataFrame({"timestamp": [f["timestamp"].iloc[500]]})
    s = build_indicator_series(f, "event_window", 4, "close", 4)
    assert s.iloc[500:504].sum() == 4 and s.iloc[:500].sum() == 0
