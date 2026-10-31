"""Never-worse guard: a mutated champion replaces its parent only on a confirmed lower-bound gain across inner purged folds."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from app.optimize import never_worse as nw


@dataclass
class T:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl: float
    risk_at_stop_dollars: float = 100.0


TS = pd.date_range("2024-01-01", periods=1000, freq="5min", tz="UTC")


def _trades(per_fold_r, n_per_fold=6, folds=5, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    bounds = nw.fold_bounds(TS, folds)
    out = []
    for (a, b), r in zip(bounds, per_fold_r):
        step = (b - a) // (n_per_fold + 2)
        for i in range(n_per_fold):
            e = pd.Timestamp(a + step * (i + 1), tz="UTC")
            x = e + pd.Timedelta(minutes=5)
            out.append(T(e, x, (r + rng.normal(0, noise)) * 100.0))
    return out


def test_trade_r_uses_risk_at_stop_and_rejects_unknown_risk():
    t = T(TS[0], TS[1], 50.0, 100.0)
    assert nw.trade_r(t) == pytest.approx(0.5)
    assert nw.trade_r(T(TS[0], TS[1], 50.0, 0.0)) is None


def test_accepts_clearly_better_champion():
    parent = _trades([0.0, 0.05, -0.05, 0.0, 0.02], noise=0.01, seed=1)
    champ = _trades([0.4, 0.45, 0.35, 0.4, 0.42], noise=0.01, seed=2)
    d = nw.decide(parent, champ, TS)
    assert d.accept and d.lcb_gain_r > 0 and d.n_folds_used == 5


def test_keeps_parent_when_champion_is_worse():
    parent = _trades([0.3, 0.35, 0.25, 0.3, 0.32], noise=0.01, seed=1)
    champ = _trades([0.0, 0.02, -0.02, 0.0, 0.01], noise=0.01, seed=2)
    assert not nw.decide(parent, champ, TS).accept


def test_keeps_parent_when_gain_is_within_noise():
    parent = _trades([0.10, 0.20, 0.00, 0.15, 0.05], noise=0.3, seed=3)
    champ = _trades([0.11, 0.19, 0.01, 0.16, 0.04], noise=0.3, seed=4)
    assert not nw.decide(parent, champ, TS).accept


def test_too_few_usable_folds_keeps_parent():
    parent = _trades([0.1] * 5)
    champ = _trades([0.9] * 5, n_per_fold=1)       # one trade per fold < min_trades_per_fold
    d = nw.decide(parent, champ, TS)
    assert not d.accept and d.n_folds_used < nw.MIN_VALID_FOLDS


def test_unmeasurable_parent_lets_champion_through():
    champ = _trades([0.3, 0.3, 0.3, 0.3, 0.3], noise=0.01)
    d = nw.decide([], champ, TS)
    assert d.accept and d.parent_lcb_r is None


def test_too_little_data_keeps_parent():
    d = nw.decide(_trades([0.1] * 5), _trades([0.5] * 5), TS[:4])
    assert not d.accept


def test_boundary_straddling_trade_is_purged():
    bounds = nw.fold_bounds(TS, 5)
    a, b = bounds[0]
    straddle = T(pd.Timestamp(b - 1000, tz="UTC"), pd.Timestamp(b + 1000, tz="UTC"), 9999.0)
    inside = [T(pd.Timestamp(a + 10 * (i + 1), tz="UTC"), pd.Timestamp(a + 10 * (i + 1) + 5, tz="UTC"), 100.0) for i in range(3)]
    means = nw.fold_mean_r(inside + [straddle], bounds)
    assert means[0] == pytest.approx(1.0)          # the straddler's huge R is not counted in any fold
    assert np.isnan(means[1])


def test_decision_serialises():
    d = nw.decide(_trades([0.1] * 5), _trades([0.5] * 5), TS)
    assert set(d.to_dict()) >= {"accept", "reason", "parent_lcb_r", "champion_lcb_r", "lcb_gain_r"}
