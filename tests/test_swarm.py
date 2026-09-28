"""Tests for app.orchestration.swarm -- the equity-curve swarm and its
in-sample/out-of-sample divider."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.orchestration import swarm as sw

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"


# ---- pure helpers ------------------------------------------------------------

def test_downsample_keeps_first_and_last_and_never_upsamples():
    arr = np.arange(1000, dtype=float)
    out = sw.downsample_equity(arr, 80)
    assert len(out) == 80 and out[0] == 0 and out[-1] == 999
    assert list(sw.downsample_equity([1, 2, 3], 80)) == [1, 2, 3]
    assert sw.downsample_equity([], 80).size == 0


def test_split_index_is_clamped_so_both_segments_exist():
    assert sw.split_index_for(80, 0.2) == 64
    assert sw.split_index_for(2, 0.2) == 1
    assert sw.split_index_for(10, 0.0) == 9 and sw.split_index_for(10, 1.0) == 1
    assert sw.split_index_for(1, 0.2) == 0


def test_summarize_curve_normalises_and_measures_the_holdout_segment():
    # doubles in-sample, then falls 10% on the holdout
    # 100 samples, divider at sample 80: flat, then doubled by the divider, then -10%
    equity = [10000] * 10 + [20000] * 70 + list(np.linspace(20000, 18000, 20))
    c = sw.summarize_curve("c1", "breakout", equity, 10000, points=100)
    assert c.values[0] == 0.0
    assert c.total_return_pct == pytest.approx(80.0, abs=0.5)
    assert c.is_return_pct == pytest.approx(100.0, abs=0.5)
    assert c.oos_return_pct == pytest.approx(-10.0, abs=0.5) and c.holds_up is False


def test_summarize_curve_holds_up_when_profitable_after_the_divider():
    c = sw.summarize_curve("c2", "trend", np.linspace(10000, 12000, 200), 10000)
    assert c.holds_up is True and c.oos_return_pct > 0


@pytest.mark.parametrize("equity,balance", [([], 10000), ([5], 10000), ([1, float("nan"), 3], 10000),
                                            ([1, 2, 3], 0), ([1, 2, 3], -5), ([1, float("inf")], 10000)])
def test_summarize_curve_rejects_undrawable_input(equity, balance):
    assert sw.summarize_curve("x", "f", equity, balance) is None


def test_summarize_curve_survives_a_wiped_out_account():
    c = sw.summarize_curve("dead", "f", np.linspace(10000, 0, 100), 10000)
    assert c is not None and np.isfinite(c.oos_return_pct)


# ---- build_swarm with the engine stubbed --------------------------------------

def _stub_engine(monkeypatch, equity_for):
    class BT:
        def __init__(self, eq): self.equity_curve = pd.DataFrame({"equity": eq}); self.initial_balance = 10000.0

    def fake_backtest(df, strategy, risk):
        return BT(equity_for(strategy))

    monkeypatch.setattr("app.backtest.engine.run_backtest", fake_backtest)
    monkeypatch.setattr("app.search.strategy_space.build_strategy_from_spec",
                        lambda spec, tmp_dir=None: spec["config"]["tag"])


def _cands(n, **extra):
    return [{"candidate_id": f"c{i}", "family": "f", "source_type": "manual", "config": {"tag": i}, **extra}
            for i in range(n)]


def test_build_swarm_caps_curves_and_counts_holds_up(monkeypatch):
    _stub_engine(monkeypatch, lambda tag: np.linspace(10000, 10000 + (tag % 2) * 500 - 250 + 300, 300))
    result = sw.build_swarm(None, None, _cands(10), max_curves=6)
    assert result.requested == 6 and result.computed == 6 and result.failed == 0
    assert len(result.curves) == 6
    d = result.to_dict()
    assert d["points"] == 80 and d["split_index"] == 64
    assert d["holds_up_count"] == result.holds_up_count


def test_a_failing_candidate_is_counted_and_skipped(monkeypatch):
    def equity(tag):
        if tag == 1:
            raise RuntimeError("bad strategy")
        return np.linspace(10000, 11000, 200)

    _stub_engine(monkeypatch, equity)
    result = sw.build_swarm(None, None, _cands(3))
    assert (result.computed, result.failed) == (2, 1)


def test_time_budget_returns_partial_results(monkeypatch):
    _stub_engine(monkeypatch, lambda tag: np.linspace(10000, 11000, 200))
    ticks = iter(range(0, 1000, 10))
    result = sw.build_swarm(None, None, _cands(10), time_budget_seconds=25, clock=lambda: next(ticks))
    assert result.timed_out is True and 0 < result.computed < 10


def test_cancel_event_stops_the_swarm(monkeypatch):
    _stub_engine(monkeypatch, lambda tag: np.linspace(10000, 11000, 200))
    cancel = threading.Event(); cancel.set()
    result = sw.build_swarm(None, None, _cands(5), cancel_event=cancel)
    assert result.computed == 0


def test_progress_callback_and_a_raising_callback(monkeypatch):
    _stub_engine(monkeypatch, lambda tag: np.linspace(10000, 11000, 200))
    seen = []
    sw.build_swarm(None, None, _cands(3), progress_cb=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 3), (2, 3), (3, 3)]

    def boom(d, t): raise RuntimeError("ui gone")
    assert sw.build_swarm(None, None, _cands(2), progress_cb=boom).computed == 2


def test_curves_are_trimmed_to_one_common_length(monkeypatch):
    _stub_engine(monkeypatch, lambda tag: np.linspace(10000, 11000, 30 if tag == 0 else 300))
    result = sw.build_swarm(None, None, _cands(2))
    assert len({len(c.values) for c in result.curves}) == 1
    assert result.points == len(result.curves[0].values)
    assert 0 < result.split_index < result.points


def test_empty_candidate_list():
    result = sw.build_swarm(None, None, [])
    assert result.computed == 0 and result.to_dict()["holds_up_pct"] is None


# ---- real engine, real candidates ---------------------------------------------

def test_real_backtests_produce_a_swarm():
    from app.backtest.risk import RiskConfig
    from app.search.strategy_space import generate_search_space

    df = pd.read_csv(SAMPLE); df["timestamp"] = pd.to_datetime(df["timestamp"])
    space = generate_search_space("family", family="trend_breakout", max_candidates=5, seed=1)
    cands = [{"candidate_id": cid, "family": space.meta[cid]["family"], **spec} for cid, spec in space.candidates.items()]
    result = sw.build_swarm(df, RiskConfig(initial_balance=10000), cands)
    assert result.computed == 5 and result.failed == 0
    assert all(len(c.values) == result.points for c in result.curves)
    assert all(c.values[0] == 0.0 for c in result.curves)


# ---- cache ---------------------------------------------------------------------

def _wait(cache, key, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        e = cache.get(key)
        if e and e["status"] != "running":
            return e
        time.sleep(0.01)
    raise AssertionError("swarm cache entry never finished")


def test_cache_runs_once_then_serves_the_result():
    cache = sw.SwarmCache(); calls = []

    def compute(progress):
        calls.append(1); progress(1, 1)
        return sw.SwarmResult(requested=1, computed=1)

    cache.start("k", compute)
    entry = _wait(cache, "k")
    assert entry["status"] == "ready" and entry["result"]["computed"] == 1
    assert cache.start("k", compute)["status"] == "ready" and len(calls) == 1      # served from cache
    cache.start("k", compute, force=True); _wait(cache, "k")
    assert len(calls) == 2                                                          # force recomputes


def test_cache_does_not_double_start_a_running_key():
    cache = sw.SwarmCache(); gate = threading.Event(); calls = []

    def compute(progress):
        calls.append(1); gate.wait(2)
        return sw.SwarmResult()

    cache.start("k", compute); cache.start("k", compute)
    time.sleep(0.05); gate.set(); _wait(cache, "k")
    assert len(calls) == 1


def test_cache_reports_errors_and_evicts_oldest():
    cache = sw.SwarmCache(max_entries=2)

    def boom(progress): raise ValueError("nope")
    cache.start("a", boom)
    assert _wait(cache, "a")["error"].startswith("ValueError")
    for k in ("b", "c"):
        cache.start(k, lambda p: sw.SwarmResult()); _wait(cache, k)
    assert cache.get("a") is None and cache.get("c") is not None
