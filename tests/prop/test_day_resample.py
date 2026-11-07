"""Every Monte Carlo that draws its own paths resamples whole session days (2026-11-06 audit)."""
import numpy as np
import pandas as pd

from app.monte_carlo.day_resample import PathSampler
from app.prop.simulator import PropRules


def _dates():
    # 6 session days, 3 trades each, 10:00-12:00 Chicago-ish UTC times
    out = []
    for d in range(6):
        for h in (15, 16, 17):
            out.append(pd.Timestamp(2024, 1, 2 + d, h))
    return out


def test_day_block_draw_keeps_each_days_trades_together():
    dates = _dates()
    s = PathSampler(dates, PropRules(), "day_block_bootstrap")
    assert s.day_mode and s.n_days == 6
    rng = np.random.default_rng(3)
    for _ in range(20):
        idx, ds = s.draw(rng)
        assert ds is not None and len(idx) % 3 == 0
        for k in range(0, len(idx), 3):
            assert set(idx[k:k + 3]) <= set(range(3 * (idx[k] // 3), 3 * (idx[k] // 3) + 3))   # one source day per slot


def test_non_day_method_matches_old_iid_draw_stream():
    from app.monte_carlo.engine import _resample_indices
    s = PathSampler(_dates(), PropRules(), "bootstrap")
    a, ds = s.draw(np.random.default_rng(5))
    b = _resample_indices(np.random.default_rng(5), len(_dates()), "bootstrap", s._eff_block)
    assert ds is None and np.array_equal(a, b)
