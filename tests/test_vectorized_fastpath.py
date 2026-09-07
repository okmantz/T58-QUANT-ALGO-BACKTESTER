"""
Parity tests for app.backtest.vectorized_fastpath.

The whole point of the vectorized Stage 1 fast path is that it agrees
with the real bar-by-bar engine (app.backtest.execution.run_execution)
for every candidate it's eligible to handle. These tests build a shared
synthetic price series + signal series, run BOTH engines on it, and
assert the resulting trades match to the cent -- not just "similar
stats". A future change to either engine that breaks this parity should
fail loudly here, not surface as a silently-wrong Search Lab ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig
from app.backtest.vectorized_fastpath import (
    VectorizedCandidate, _position_size_vec, _risk_amount_vec, is_vectorizable, run_vectorized_batch,
)
from app.strategy.base import StrategyResult


def _choppy_df(n=600, seed=7):
    """Noisy-but-bounded price path with plenty of both stop-outs,
    take-profits, and signal-driven exits -- deliberately NOT a clean
    trend, so the fill-resolution logic (gap-through stops, honest
    take-profit fills, signal reversals) gets genuinely exercised on
    both engines."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    price = 100.0
    rows = []
    for i in range(n):
        step = rng.normal(0, 0.6)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.4))
        l = min(o, c) - abs(rng.normal(0, 0.4))
        rows.append((ts[i], o, h, l, c, 1000.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _alternating_signals(n, seed=11, flat_prob=0.15):
    """A signal series with entries, flat stretches, and reversals --
    exercises same-bar reversal fills on both engines identically."""
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
    return sig


@pytest.mark.parametrize("stop_pips,take_pips", [(20.0, 40.0), (15.0, None), (None, 30.0), (None, None)])
def test_vectorized_matches_scalar_engine_trade_by_trade(stop_pips, take_pips):
    df = _choppy_df()
    sig = _alternating_signals(len(df))
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=0.01, spread_pips=1.0,
                       slippage_pips=0.5, commission_per_trade=0.5, max_trades_per_day=999)

    signals_series = pd.Series(sig)
    scalar_trades, _ = run_execution(
        df, signals_series, risk, stop_loss_pips=stop_pips, take_profit_pips=take_pips,
    )

    outcomes = run_vectorized_batch(
        df, [VectorizedCandidate("c1", sig, stop_pips, take_pips)], risk,
    )
    vec_trades = outcomes["c1"].trades

    assert len(vec_trades) == len(scalar_trades)
    for s, v in zip(scalar_trades, vec_trades):
        assert s.entry_time == v.entry_time
        assert s.exit_time == v.exit_time
        assert s.direction == v.direction
        assert s.exit_reason == v.exit_reason
        assert s.entry_price == pytest.approx(v.entry_price, abs=1e-9)
        assert s.exit_price == pytest.approx(v.exit_price, abs=1e-9)
        assert s.size == pytest.approx(v.size, rel=1e-9)
        assert s.pnl == pytest.approx(v.pnl, abs=1e-6)
        assert s.equity_after == pytest.approx(v.equity_after, abs=1e-6)


def test_vectorized_batch_multiple_candidates_are_independent():
    """Two different parameter sets in the same batch call must not leak
    state into each other -- this is the whole premise of vectorizing
    across columns instead of running each in its own process."""
    df = _choppy_df(seed=3)
    sig_a = _alternating_signals(len(df), seed=1)
    sig_b = _alternating_signals(len(df), seed=2)
    risk = RiskConfig(initial_balance=25_000.0, risk_value=0.5, pip_size=0.01, spread_pips=0.5)

    outcomes = run_vectorized_batch(
        df,
        [
            VectorizedCandidate("a", sig_a, 20.0, 40.0),
            VectorizedCandidate("b", sig_b, 10.0, 20.0),
        ],
        risk,
    )

    scalar_a, _ = run_execution(df, pd.Series(sig_a), risk, stop_loss_pips=20.0, take_profit_pips=40.0)
    scalar_b, _ = run_execution(df, pd.Series(sig_b), risk, stop_loss_pips=10.0, take_profit_pips=20.0)

    assert len(outcomes["a"].trades) == len(scalar_a)
    assert len(outcomes["b"].trades) == len(scalar_b)
    if scalar_a:
        assert outcomes["a"].trades[-1].equity_after == pytest.approx(scalar_a[-1].equity_after, abs=1e-6)
    if scalar_b:
        assert outcomes["b"].trades[-1].equity_after == pytest.approx(scalar_b[-1].equity_after, abs=1e-6)


def test_pip_scale_mismatch_detected_in_fast_path():
    """A fixed-pips stop calibrated for FX (pip_size 0.0001-scale
    thinking) run against a whole-dollar/expensive instrument must still
    trip the scale-mismatch flag through the fast path -- this is the
    exact diagnostic Search Lab's Stage 1 log relies on to explain a
    0-survivor run (see tests/test_batch_runner.py)."""
    df = _choppy_df(seed=5)
    # price ~100, pip_size left at FX default -> a "20 pip" stop is
    # laughably tiny against this instrument.
    risk = RiskConfig(initial_balance=10_000.0, pip_size=0.0001)
    sig = _alternating_signals(len(df), seed=4)

    outcomes = run_vectorized_batch(df, [VectorizedCandidate("c", sig, 20.0, 40.0)], risk)
    assert outcomes["c"].scale_mismatch is True


def test_risk_amount_and_position_size_vectorized_match_scalar():
    risk_pct = RiskConfig(initial_balance=10_000.0, risk_mode="percent", risk_value=2.0, pip_size=0.01)
    risk_fixed = RiskConfig(initial_balance=10_000.0, risk_mode="fixed", risk_value=150.0, pip_size=0.01,
                             max_position_size=500.0)
    equities = np.array([10_000.0, 5_000.0, -100.0, 0.0, 20_000.0])
    pips = np.array([10.0, 25.0, 0.0, 50.0, 5.0])

    for risk in (risk_pct, risk_fixed):
        vec_amt = _risk_amount_vec(equities, risk)
        vec_size = _position_size_vec(equities, pips, risk)
        for eq, pip, va, vs in zip(equities, pips, vec_amt, vec_size):
            assert va == pytest.approx(risk.risk_amount(eq), abs=1e-9)
            assert vs == pytest.approx(risk.position_size(eq, pip), abs=1e-9)


def test_is_vectorizable_rejects_dynamic_stop_shapes():
    base = dict(name="x", source_type="python", signals=pd.Series([0]))
    assert is_vectorizable(StrategyResult(**base, stop_loss_pips=20.0, take_profit_pips=40.0))
    assert not is_vectorizable(StrategyResult(**base, stop_loss_distance=pd.Series([0.5])))
    assert not is_vectorizable(StrategyResult(**base, take_profit_distance=pd.Series([0.5])))
    assert not is_vectorizable(StrategyResult(**base, trailing_stop_distance=pd.Series([0.5])))
    assert not is_vectorizable(StrategyResult(**base, breakeven_trigger_r=1.5))
