"""
v5 fastpath tests: fill-lag parity, per-bar ATR stops, tail handling,
adaptive-budget exactness.

Fill-lag context: the v5 fastpath fills market entries and signal-driven
exits at open[i + entry_fill_lag_bars] (default 1, read via
getattr(risk, "entry_fill_lag_bars", 1)), while the scalar engine in this
tree still fills at the signal bar's close (its own fill-lag change lands
separately). Parity tests therefore run on GAPLESS data -- where
open[i+1] == close[i] makes the two fills coincide -- and assert the
intended one-bar timestamp shift explicitly. The gapped-data tests assert
the next-open fill directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig
from app.backtest.vectorized_fastpath import (
    VectorizedCandidate, is_vectorizable, run_vectorized_batch,
)
from app.strategy.base import StrategyResult
from tests.test_vectorized_fastpath import (
    _alternating_signals, _assert_trades_match_lag_shift, _gapless_df,
)


def _drift_df(n=30, seed=1):
    """Tiny deterministic series with real gaps (open[i+1] != close[i])
    and tight ranges, so a wide stop never triggers and every fill is a
    market fill whose price we can predict exactly."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    rows = []
    o = 100.0
    for i in range(n):
        gap = rng.normal(0, 0.05)
        oo = o + gap if i else o
        c = oo + 0.01
        rows.append((ts[i], oo, oo + 0.05, oo - 0.05, c, 1000.0))
        o = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _wide_stop_risk(**kw):
    args = dict(initial_balance=10_000.0, risk_value=1.0, pip_size=0.01,
                spread_pips=0.0, slippage_pips=0.0, commission_per_trade=0.0,
                max_trades_per_day=999)
    args.update(kw)
    return RiskConfig(**args)


def test_market_entry_fills_at_next_bar_open():
    """Entry decided at bar i fills at open[i+1] (+ costs), with
    entry_time == ts[i+1] -- not at close[i]."""
    df = _drift_df()
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    sig[5:] = 1  # long, held; wide stop so nothing else interferes
    risk = _wide_stop_risk()
    outcomes = run_vectorized_batch(
        df, [VectorizedCandidate("c", sig, 500.0, None)], risk)
    trades = outcomes["c"].trades
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == pytest.approx(df["open"][6], abs=1e-9)
    assert t.entry_time == df["timestamp"][6]
    assert t.exit_reason == "end_of_data"
    assert t.exit_price == pytest.approx(df["close"][n - 1], abs=1e-9)


def test_signal_exit_fills_at_next_bar_open():
    """Signal exit decided at bar i fills at open[i+1] (- costs); a
    reversal then re-enters subject to the normal cooldown."""
    df = _drift_df()
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    sig[5:10] = 1
    sig[10:] = -1  # reversal held long enough to clear the cooldown
    risk = _wide_stop_risk()
    outcomes = run_vectorized_batch(
        df, [VectorizedCandidate("c", sig, 500.0, None)], risk)
    trades = outcomes["c"].trades
    assert len(trades) == 2
    long_t, short_t = trades
    assert long_t.exit_reason == "signal"
    assert long_t.exit_price == pytest.approx(df["open"][11], abs=1e-9)
    assert long_t.exit_time == df["timestamp"][11]
    # Reentry was blocked at bar 10 by reentry_cooldown_bars=1, so the
    # short is decided at bar 11 and fills at open[12].
    assert short_t.direction == -1
    assert short_t.entry_price == pytest.approx(df["open"][12], abs=1e-9)
    assert short_t.entry_time == df["timestamp"][12]


def test_tail_entry_skipped_when_no_next_open():
    """An entry decided on the last `fill_lag` bars is skipped -- there
    is no next open to fill at."""
    df = _drift_df()
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    sig[n - 1] = 1  # decided at the final bar: fill bar would be n
    risk = _wide_stop_risk()
    outcomes = run_vectorized_batch(
        df, [VectorizedCandidate("c", sig, 500.0, None)], risk)
    assert outcomes["c"].trades == []
    assert outcomes["c"].equity_curve is None


def test_tail_signal_exit_settles_as_end_of_data():
    """A signal exit decided on the last bar cannot fill at a next open;
    the position stays open and the end-of-data close settles it."""
    df = _drift_df()
    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    sig[5:] = 1
    sig[n - 1] = -1  # reversal on the final bar
    risk = _wide_stop_risk()
    outcomes = run_vectorized_batch(
        df, [VectorizedCandidate("c", sig, 500.0, None)], risk)
    trades = outcomes["c"].trades
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "end_of_data"
    assert t.exit_price == pytest.approx(df["close"][n - 1], abs=1e-9)
    assert t.exit_time == df["timestamp"][n - 1]


def test_perbar_atr_stop_matches_scalar_engine():
    """Per-bar stop/target distance arrays vectorize with trade-by-trade
    parity against the scalar engine (gapless data; the intended +1-bar
    market-fill shift asserted explicitly)."""
    df = _gapless_df()
    n = len(df)
    sig = _alternating_signals(n)
    sig[-5:] = 0
    # ATR-like distances: smoothly varying, always positive.
    idx = np.arange(n)
    sl_dist = pd.Series(0.8 + 0.3 * np.sin(idx / 25.0))
    tp_dist = pd.Series(1.6 + 0.5 * np.cos(idx / 31.0))
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=0.01,
                      spread_pips=1.0, slippage_pips=0.5, commission_per_trade=0.5,
                      max_trades_per_day=999)

    scalar_trades, _ = run_execution(
        df, pd.Series(sig), risk, None, None,
        stop_loss_distance=sl_dist, take_profit_distance=tp_dist,
    )
    outcomes = run_vectorized_batch(
        df, [VectorizedCandidate("c", sig, None, None,
                                stop_distances=sl_dist.to_numpy(),
                                take_distances=tp_dist.to_numpy())],
        risk,
    )
    assert len(scalar_trades) > 20  # sanity: the test actually exercises exits
    _assert_trades_match_lag_shift(scalar_trades, outcomes["c"].trades, df)


def test_perbar_stop_with_fixed_take_pips_matches_scalar():
    """Mixed config: per-bar stop distances + fixed-pips take-profit --
    scalar precedence (per-bar > fixed pips) must hold on both paths."""
    df = _gapless_df(seed=9)
    n = len(df)
    sig = _alternating_signals(n, seed=21)
    sig[-5:] = 0
    sl_dist = pd.Series(0.7 + 0.2 * np.sin(np.arange(n) / 20.0))
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=0.01,
                      spread_pips=1.0, slippage_pips=0.5, commission_per_trade=0.5,
                      max_trades_per_day=999)

    scalar_trades, _ = run_execution(
        df, pd.Series(sig), risk, None, 40.0,
        stop_loss_distance=sl_dist,
    )
    outcomes = run_vectorized_batch(
        df, [VectorizedCandidate("c", sig, None, 40.0,
                                stop_distances=sl_dist.to_numpy())],
        risk,
    )
    assert len(scalar_trades) > 10
    _assert_trades_match_lag_shift(scalar_trades, outcomes["c"].trades, df)


def test_perbar_nan_falls_back_to_fixed_pips():
    """A NaN in the per-bar array means 'no per-bar distance for this
    bar' -- the fixed-pips stop applies there, exactly like the scalar
    engine's NaN -> None handling."""
    df = _gapless_df(seed=4)
    n = len(df)
    sig = _alternating_signals(n, seed=33)
    sig[-5:] = 0
    sl_dist = pd.Series(np.where(np.arange(n) % 3 == 0, np.nan, 0.9))
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=0.01,
                      spread_pips=1.0, slippage_pips=0.5, commission_per_trade=0.5,
                      max_trades_per_day=999)

    scalar_trades, _ = run_execution(
        df, pd.Series(sig), risk, 20.0, None,
        stop_loss_distance=sl_dist,
    )
    outcomes = run_vectorized_batch(
        df, [VectorizedCandidate("c", sig, 20.0, None,
                                stop_distances=sl_dist.to_numpy())],
        risk,
    )
    assert len(scalar_trades) > 10
    _assert_trades_match_lag_shift(scalar_trades, outcomes["c"].trades, df)


def test_adaptive_budget_silent_candidate_costs_nothing_and_stays_exact():
    """A zero-signal candidate resolves to an empty outcome without
    entering the bar loop; every other candidate's trades are identical
    to running it alone (the budget is exact, never approximate)."""
    df = _gapless_df(n=500, seed=6)
    n = len(df)
    sig_a = _alternating_signals(n, seed=41)
    sig_a[-5:] = 0
    sig_b = np.zeros(n, dtype=np.int8)
    sig_b[:60] = _alternating_signals(60, seed=42)  # signals end early -> compaction path
    sig_silent = np.zeros(n, dtype=np.int8)
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=0.01,
                      spread_pips=1.0, slippage_pips=0.5, commission_per_trade=0.5,
                      max_trades_per_day=999)

    cands = [
        VectorizedCandidate("a", sig_a, 20.0, 40.0),
        VectorizedCandidate("b", sig_b, 20.0, 40.0),
        VectorizedCandidate("silent", sig_silent, 20.0, 40.0),
    ]
    outcomes = run_vectorized_batch(df, cands, risk)

    silent = outcomes["silent"]
    assert silent.trades == []
    assert silent.equity_curve is None
    assert silent.scale_mismatch is False

    for cid, sig in (("a", sig_a), ("b", sig_b)):
        ref = run_vectorized_batch(
            df, [VectorizedCandidate(cid, sig, 20.0, 40.0)], risk)[cid].trades
        got = outcomes[cid].trades
        assert len(got) == len(ref)
        for g, r in zip(got, ref):
            assert g.entry_time == r.entry_time
            assert g.exit_time == r.exit_time
            assert g.entry_price == pytest.approx(r.entry_price, abs=1e-12)
            assert g.exit_price == pytest.approx(r.exit_price, abs=1e-12)
            assert g.pnl == pytest.approx(r.pnl, abs=1e-9)


def test_is_vectorizable_perbar_shapes():
    base = dict(name="x", source_type="python", signals=pd.Series([0]))
    assert is_vectorizable(StrategyResult(**base, stop_loss_distance=pd.Series([0.5])))
    assert is_vectorizable(StrategyResult(
        **base, stop_loss_distance=np.array([0.5]),
        take_profit_distance=pd.Series([1.0])))
    # Unknown shapes fall back to the scalar path rather than failing.
    assert not is_vectorizable(StrategyResult(**base, stop_loss_distance=2.5))
