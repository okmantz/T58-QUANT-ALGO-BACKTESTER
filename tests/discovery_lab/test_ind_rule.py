"""The any-indicator family must never peek: signals on bar i are identical with or without the bars after it."""
from __future__ import annotations

import random

import numpy as np

from app.discovery_lab import families as F
from app.discovery_lab.indicator_pool import POOL_KINDS
from tests.backtest.test_indicators_v12 import _intraday


def test_pool_is_large_and_registered():
    assert len(POOL_KINDS) >= 250
    fam = F.FAMILIES["ind_rule"]
    assert fam.group == "pool" and set(fam.categorical) == {"kind", "mode"}
    assert F.neighbours(fam, {"kind": POOL_KINDS[0], "period": 14, "win": 250, "k": 1.5, "mode": "trend"})   # steps numeric settings only
    assert all(q["kind"] == POOL_KINDS[0] and q["mode"] == "trend" for q in F.neighbours(fam, {"kind": POOL_KINDS[0], "period": 14, "win": 250, "k": 1.5, "mode": "trend"}))


def test_ind_rule_signals_are_prefix_invariant():
    df = _intraday(days=40)
    rng = random.Random(3)
    fam = F.FAMILIES["ind_rule"]
    cut = 1800
    for _ in range(40):
        p = {k: rng.choice(v) for k, v in fam.grid.items()}
        lg, sh, _ = F.signals("ind_rule", df, p)
        lg2, sh2, _ = F.signals("ind_rule", df.iloc[:cut].copy(), p)
        assert np.array_equal(lg[:cut], lg2) and np.array_equal(sh[:cut], sh2), p
