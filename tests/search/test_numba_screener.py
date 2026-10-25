"""Parity + safety tests for app.search.numba_screener.

The screener is a SCREEN: these tests pin (1) per-trade parity with the real engines on the supported subset, (2) that it
never kills a candidate the exact Stage-1 path would have passed, (3) that it degrades to "off" without numba, and (4) that
wiring it into Search Lab's Stage-1 batch task leaves the set of survivors unchanged."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")

from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig
from app.backtest.statistics import compute_statistics
from app.backtest.vectorized_fastpath import VectorizedCandidate, run_vectorized_batch
from app.search import batch_runner as br
from app.search import numba_screener as ns


def _gapless_df(n=3000, seed=7, freq="15min", phi=0.0):
    """Gapless OHLC path. ``phi`` > 0 adds momentum (AR(1) steps) so some trend-following candidates have a real edge."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq=freq)
    prev = 100.0
    rows = []
    step = 0.0
    for i in range(n):
        o = prev
        step = phi * step + rng.normal(0, 0.6)
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.4))
        l = min(o, c) - abs(rng.normal(0, 0.4))
        rows.append((ts[i], o, h, l, c, 1000.0))
        prev = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _signals(n, seed=11, flat_prob=0.15):
    rng = np.random.default_rng(seed)
    sig = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(n):
        r = rng.random()
        if r < flat_prob:
            state = 0
        elif r < flat_prob + 0.05:
            state = -1 if state != -1 else 1
        elif state == 0 and r > 0.7:
            state = 1 if rng.random() < 0.5 else -1
        sig[i] = state
    sig[-5:] = 0
    return sig


def _fixed_risk(**kw):
    base = dict(initial_balance=10_000.0, risk_mode="fixed", risk_value=100.0, pip_size=0.01, spread_pips=1.0,
                slippage_pips=0.5, commission_per_trade=0.5, max_trades_per_day=999)
    base.update(kw)
    return RiskConfig(**base)


@pytest.mark.parametrize("sl,tp", [(20.0, 40.0), (15.0, None), (None, 30.0), (None, None)])
def test_per_trade_parity_with_scalar_engine(sl, tp):
    df = _gapless_df()
    sig = _signals(len(df))
    risk = _fixed_risk()
    scalar, _ = run_execution(df, pd.Series(sig), risk, stop_loss_pips=sl, take_profit_pips=tp)
    out = ns.simulate_trades_r(df, sig, risk, stop_loss_pips=sl, take_profit_pips=tp)
    assert out["n_trades"] == len(scalar) > 20
    ts = df["timestamp"].to_numpy()
    for k, t in enumerate(scalar):
        assert pd.Timestamp(ts[out["entry_bar"][k]]) == t.entry_time
        assert pd.Timestamp(ts[out["exit_bar"][k]]) == t.exit_time
        assert out["direction"][k] == t.direction
        # R = pnl / fixed risk budget (units are sized so a full stop costs exactly one budget)
        assert out["r"][k] == pytest.approx(t.pnl / risk.risk_value, abs=1e-9)


def test_parity_with_vectorized_fastpath_incl_per_bar_atr_distances():
    df = _gapless_df(seed=3)
    sig = _signals(len(df), seed=5)
    risk = _fixed_risk(spread_pips=0.5, slippage_pips=0.0)
    atr = (df["high"] - df["low"]).rolling(14, min_periods=1).mean().to_numpy()
    cand = VectorizedCandidate("c", sig, None, None, stop_distances=atr * 1.5, take_distances=atr * 3.0)
    exact = run_vectorized_batch(df, [cand], risk)["c"].trades
    out = ns.simulate_trades_r(df, sig, risk, stop_distances=atr * 1.5, take_distances=atr * 3.0)
    assert out["n_trades"] == len(exact) > 20
    for k, t in enumerate(exact):
        assert out["r"][k] == pytest.approx(t.pnl / risk.risk_value, abs=1e-9)


def test_daily_trade_cap_and_cooldown_parity():
    df = _gapless_df(n=2500, seed=9, freq="30min")
    sig = _signals(len(df), seed=2, flat_prob=0.3)
    for kw in ({"max_trades_per_day": 2}, {"reentry_cooldown_bars": 4}):
        risk = _fixed_risk(**kw)
        scalar, _ = run_execution(df, pd.Series(sig), risk, stop_loss_pips=20.0, take_profit_pips=30.0)
        out = ns.simulate_trades_r(df, sig, risk, stop_loss_pips=20.0, take_profit_pips=30.0)
        assert out["n_trades"] == len(scalar), kw


def test_batch_kernel_equals_single_runs():
    df = _gapless_df(n=2000)
    risk = _fixed_risk()
    cands = [VectorizedCandidate(f"c{i}", _signals(len(df), seed=20 + i), 10.0 + i * 5, 20.0 + i * 5) for i in range(6)]
    rows = ns.screen_arrays(df, cands, risk)
    for i, c in enumerate(cands):
        one = ns.simulate_trades_r(df, c.signals, risk, c.stop_loss_pips, c.take_profit_pips)
        assert rows[i, 0] == one["n_trades"]
        assert rows[i, 1] == pytest.approx(one["sum_r"], abs=1e-9)


def test_screener_is_fast_on_thousands_of_combos():
    import time
    df = _gapless_df(n=20000, seed=1, freq="5min")
    risk = _fixed_risk()
    cands = [VectorizedCandidate(f"c{i}", _signals(len(df), seed=i), 10.0 + (i % 20), 20.0 + (i % 15)) for i in range(40)]
    ns.screen_arrays(df, cands[:2], risk)           # compile / load cache
    t0 = time.time()
    ns.screen_arrays(df, cands, risk)
    assert time.time() - t0 < 5.0                   # 40 candidates x 20k bars: well under a second compiled


def _battery(df, risk, n_cand=120, seed=0):
    """Momentum/reversion candidates (both signs, varied lookback, dead-band, stop, target): a mix of real winners on a
    momentum path, clear losers (the reversed ones) and marginal cases."""
    rng = np.random.default_rng(seed)
    close = df["close"]
    cands = []
    for i in range(n_cand):
        k = int(rng.choice([2, 3, 5, 8, 13]))
        band = float(rng.choice([0.0, 0.3, 0.8]))
        sign = int(rng.choice([1, -1]))
        mv = (close - close.shift(k)).to_numpy()
        sig = np.where(np.isnan(mv), 0, np.where(mv > band, sign, np.where(mv < -band, -sign, 0))).astype(np.int8)
        sig[-5:] = 0
        sl = float(rng.choice([10, 15, 20, 30, 50]))
        tp = float(rng.choice([0, 15, 30, 60])) or None
        cands.append(VectorizedCandidate(f"c{i}", sig, sl, tp))
    return cands


@pytest.mark.parametrize("risk_kw", [
    {},                                                                    # fixed fractional-unit sizing
    {"risk_mode": "percent", "risk_value": 1.0},                           # compounding
    {"daily_loss_limit_pct": 2.0, "risk_mode": "percent", "risk_value": 1.0},   # screener ignores the daily limit
])
def test_never_kills_a_candidate_the_exact_stage1_would_pass(risk_kw):
    df = _gapless_df(n=4000, seed=21, phi=0.25)
    risk = _fixed_risk(**risk_kw)
    cands = _battery(df, risk, seed=4)
    filters = {"min_trades": 30, "min_profit_factor": 1.0, "max_drawdown_buffer_mult": 1e9, "min_t_stat": 0.0}
    prop = type("P", (), {"max_drawdown_pct": 10.0})()
    outs = run_vectorized_batch(df, cands, risk)
    verdicts = ns.screen_candidates(df, cands, risk, ns.ScreenThresholds(min_trades=30))
    assert verdicts is not None
    passed_exact, killed = set(), set()
    for c in cands:
        trades = outs[c.candidate_id].trades
        stats = compute_statistics(trades, outs[c.candidate_id].equity_curve, risk.initial_balance).to_dict() if trades else {}
        rec = br._stage1_score({"candidate_id": c.candidate_id}, stats, bool(trades), filters, prop)
        if rec.get("passed_stage1"):
            passed_exact.add(c.candidate_id)
        if verdicts[c.candidate_id].killed:
            killed.add(c.candidate_id)
    assert passed_exact, "battery must contain real exact-path passers for this test to mean something"
    assert not (passed_exact & killed), f"false kills: {sorted(passed_exact & killed)}"
    assert len(killed) >= 0.2 * len(cands), "screener should cull a useful share of clear losers"


def test_unsupported_shapes_are_not_screened():
    from app.strategy.base import StrategyResult
    sig = pd.Series(np.zeros(10, dtype=np.int8))
    ok = StrategyResult(name="a", source_type="manual", signals=sig, stop_loss_pips=10.0)
    assert ns.screen_supported(_fixed_risk(), ok)
    trail = StrategyResult(name="a", source_type="manual", signals=sig, trailing_stop_distance=pd.Series(np.ones(10)))
    be = StrategyResult(name="a", source_type="manual", signals=sig, breakeven_trigger_r=1.0)
    part = StrategyResult(name="a", source_type="manual", signals=sig, partial_exit={"r_multiple": 1.0, "fraction": 0.5})
    for bad in (trail, be, part):
        assert not ns.screen_supported(_fixed_risk(), bad)
    assert not ns.screen_supported(_fixed_risk(time_stop_hours=4), ok)


def test_degrades_to_off_without_numba(monkeypatch):
    monkeypatch.setenv("T58_DISABLE_NUMBA_SCREENER", "1")
    assert not ns.numba_available()
    df = _gapless_df(n=500)
    assert ns.screen_candidates(df, [VectorizedCandidate("c", _signals(500), 10.0, 20.0)], _fixed_risk(),
                                ns.ScreenThresholds()) is None


def test_judge_only_kills_far_below_the_bar():
    th = ns.ScreenThresholds(min_trades=100)
    # [n, sum_r, sum_r2, gross_win, gross_loss]
    assert ns.judge("a", np.array([40, 4.0, 10.0, 10.0, 6.0]), th).killed            # 40 < 0.7*100 trades
    assert ns.judge("b", np.array([200, -20.0, 80.0, 30.0, 50.0]), th).killed       # PF 0.6, -0.1R
    assert not ns.judge("c", np.array([200, 2.0, 80.0, 52.0, 50.0]), th).killed      # marginal winner
    assert not ns.judge("d", np.array([200, -2.0, 80.0, 45.0, 47.0]), th).killed     # mildly negative: left to the exact path


# ---------------------------------------------------------------- wiring into Stage 1 (worker-process code path)
def _spec(stop, take, fast, slow):
    return {"source_type": "manual", "config": {
        "name": f"x-{fast}-{slow}", "market": "x",
        "entry_conditions": {
            "long": [{"left": {"type": "ema", "period": fast}, "operator": "cross above", "right": {"type": "ema", "period": slow}}],
            "short": [{"left": {"type": "ema", "period": fast}, "operator": "cross below", "right": {"type": "ema", "period": slow}}]},
        "risk_management": {"stop_type": "fixed", "stop_value": stop, "target_type": "fixed", "target_value": take}}}


def test_stage1_batch_survivors_identical_with_and_without_screener(tmp_path):
    df = _gapless_df(n=6000, seed=33)
    risk = _fixed_risk(max_trades_per_day=50)
    prop = br.PropRules(account_size=risk.initial_balance)
    br._WORKER.update({"df": df, "risk": risk, "prop_rules": prop, "tmp_dir": tmp_path})
    items = [(f"c{f}-{s}-{sl}", _spec(sl, sl * 2, f, s)) for f in (3, 5, 8) for s in (13, 21, 34) for sl in (15.0, 30.0, 60.0)]
    filters = {"min_trades": 30, "min_profit_factor": 1.0, "max_drawdown_buffer_mult": 1e9, "min_t_stat": 0.0, "max_cost_in_r": 0.0}
    plain = {r["candidate_id"]: r for r in br._stage1_task_batch(items, filters, None)}
    screened = {r["candidate_id"]: r for r in br._stage1_task_batch(items, filters, {"min_trades": 30})}
    assert set(plain) == set(screened)
    assert {c for c, r in plain.items() if r.get("passed_stage1")} == {c for c, r in screened.items() if r.get("passed_stage1")}
    culled = [c for c, r in screened.items() if r.get("screened_out")]
    assert all(not plain[c].get("passed_stage1") for c in culled)
    # screened records carry no statistics and say why
    assert all(screened[c]["error"].startswith("screened:") for c in culled)
