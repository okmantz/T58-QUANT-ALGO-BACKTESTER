"""Evolution Lab extras (audit item 2): parity gate, random controls, tribes, decay, incubation, hour-of-week spread."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.evolution import lab_extras as lx


def _trades(n=40, shift=0.0, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-01-05 10:00")
    out = []
    for i in range(n):
        out.append({"entry_time": t0 + pd.Timedelta(hours=6 * i) + pd.Timedelta(seconds=shift),
                    "direction": 1 if i % 3 else -1, "pnl": float((i % 7) - 2.5 + rng.normal(0, noise))})
    return out


def test_parity_pass_and_fail():
    a = _trades()
    ok = lx.parity_gate(a, _trades(shift=30))
    assert ok.status == "PASS" and ok.match_rate == 1.0
    bad = lx.parity_gate(a, _trades(shift=30, noise=20.0, seed=5))
    assert bad.status == "FAIL" and bad.reasons
    assert lx.parity_gate(a, []).status == "NOT_RUN"
    assert lx.parity_gate(a[:5], _trades()[:5]).status == "INSUFFICIENT"


def test_ready_requires_parity():
    ok = lx.parity_gate(_trades(), _trades(shift=10))
    assert lx.readiness_label(ok, True) == "READY"
    assert lx.readiness_label(ok, False).startswith("NOT READY")
    assert lx.readiness_label(None, True).startswith("NOT READY")
    assert lx.readiness_label({"status": "FAIL"}, True).startswith("NOT READY")


def test_random_control_spec_builds_and_is_seeded(tmp_path):
    from app.search.strategy_space import build_strategy_from_spec
    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=800, freq="5min"),
                       "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10.0})
    a = build_strategy_from_spec(lx.random_control_spec(7), tmp_path).generate(df)
    b = build_strategy_from_spec(lx.random_control_spec(7), tmp_path).generate(df)
    c = build_strategy_from_spec(lx.random_control_spec(8), tmp_path).generate(df)
    assert (a.signals.values == b.signals.values).all() and not (a.signals.values == c.signals.values).all()
    assert set(np.unique(a.signals.values)) <= {-1, 0, 1} and (a.signals != 0).sum() > 0


def test_control_tracker_distinguishes_skill_from_chance():
    rng = np.random.default_rng(0)
    skilled = lx.ControlTracker()
    for _ in range(60):
        skilled.record(True, rng.random() < 0.05, rng.uniform(0.6, 1.1))
        skilled.record(False, rng.random() < 0.40, rng.uniform(0.8, 1.8))
    r = skilled.report()
    assert r["beats_chance"] is True and r["p_value"] < 0.05
    luck = lx.ControlTracker()
    for _ in range(60):
        luck.record(True, rng.random() < 0.20, rng.uniform(0.6, 1.3))
        luck.record(False, rng.random() < 0.20, rng.uniform(0.6, 1.3))
    assert luck.report()["beats_chance"] is False
    thin = lx.ControlTracker()
    thin.record(True, False, 0.9)
    assert thin.report()["beats_chance"] is None and "Not enough data" in thin.report()["verdict"]


def test_tribe_scores_differ():
    from app.evolution.prop_fitness import PropFitnessBreakdown
    steady = PropFitnessBreakdown(0.7, 0.3, 0.9, 0.9, 5.0, 10.0)
    swingy = PropFitnessBreakdown(0.5, 0.8, 0.4, 0.4, 9.0, 10.0)
    s, g = lx.tribe_scores(steady), lx.tribe_scores(swingy)
    assert s["consistency"] > g["consistency"] and g["growth"] > s["growth"]
    assert lx.tribes_active(3, 8) and not lx.tribes_active(8, 8) and not lx.tribes_active(1, 8, enabled=False)


def test_decay_check_15():
    good = [2.0] * 60
    assert lx.decay_check(good).decayed is False
    cold = [2.0] * 50 + [-1.0] * 15
    d = lx.decay_check(cold)
    assert d.decayed is True and d.last_mean < 0
    assert lx.decay_check([1.0] * 10).decayed is None


def test_incubation_ramp():
    assert lx.incubation_multiplier(0) == 0.25 and lx.incubation_multiplier(20) == 0.5
    assert lx.incubation_multiplier(75) == 0.75 and lx.incubation_multiplier(500) == 1.0
    plan = lx.incubation_plan(1.0)
    assert [s["risk_pct"] for s in plan] == [0.25, 0.5, 0.75, 1.0]


def test_hour_of_week_spread_and_recost():
    ts = pd.date_range("2026-01-05", periods=24 * 7 * 6, freq="h")
    rng = np.random.default_rng(3)
    wide = np.where(ts.hour == 3, 3.0, 1.0)
    df = pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 100 + wide * rng.uniform(0.5, 1.5, len(ts)),
                       "low": 100.0, "close": 100.0, "volume": 100.0})
    m = lx.HourOfWeekSpread.fit(df)
    assert abs(float(m.mult.mean()) - 1.0) < 0.35
    assert m.spread_pips(pd.Timestamp("2026-01-06 03:00"), 1.0) > 1.5 * m.spread_pips(pd.Timestamp("2026-01-06 12:00"), 1.0)

    class T:  # minimal trade
        def __init__(self, h, pnl):
            self.entry_time = pd.Timestamp("2026-01-06") + pd.Timedelta(hours=h)
            self.exit_time = self.entry_time + pd.Timedelta(minutes=30)
            self.pnl, self.size = pnl, 100.0
    out = lx.recost_trades([T(3, 10.0), T(12, 10.0)], m, flat_spread_pips=1.0, pip_size=0.01)
    assert out["adjusted_pnls"][0] < out["adjusted_pnls"][1]


def test_curve_store_is_bounded_and_incremental():
    s = lx.EquityCurveStore(capacity=5)
    for i in range(12):
        s.add({"id": i})
    got = s.since(0)
    assert len(got["items"]) == 5 and got["seq"] == 12
    assert [r["id"] for r in s.since(10)["items"]] == [10, 11]
    assert len(lx.curve_from_pnls(list(range(1000)), 10_000.0, 90)) <= 90
