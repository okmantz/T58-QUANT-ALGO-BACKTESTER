"""Parity tests for app.backtest.vectorized_fastpath. / v5 fastpath tests: fill-lag parity, per-bar ATR stops, tail handling,"""
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
from app.backtest.vectorized_fastpath import (
    VectorizedCandidate, is_vectorizable, run_vectorized_batch,
)


# ---- from test_vectorized_fastpath.py ----

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


def _gapless_df(n=600, seed=7):
    """Like _choppy_df, but with open[i+1] == close[i] exactly (no gaps).

    Fill-lag parity note (v5): the fastpath fills market entries and
    signal-exits at open[i+lag] while the scalar engine still fills at
    close[i]. On gapless data those two prices coincide, so a
    trade-by-trade comparison against the scalar engine isolates REAL
    divergences from the intended one-bar fill shift. Tests that need
    gaps use _choppy_df and assert the next-open fill directly."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    rows = []
    prev_close = 100.0
    for i in range(n):
        o = prev_close  # gapless by construction: this bar's open IS last bar's close
        step = rng.normal(0, 0.6)
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.4))
        l = min(o, c) - abs(rng.normal(0, 0.4))
        rows.append((ts[i], o, h, l, c, 1000.0))
        prev_close = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _assert_trades_match_lag_shift(scalar_trades, vec_trades, df):
    """Assert trade-by-trade parity between the scalar engine and the v5
    fastpath. BOTH engines fill market entries and signal-exits at the
    fill bar's open -- ``open[i + entry_fill_lag_bars]`` (default lag 1,
    read off the shared RiskConfig) -- so entry/exit timestamps must be
    IDENTICAL, with identical prices/sizes/pnl/reasons.

    Callers must zero out signals on the last `lag` bars first: entries
    the engines would open at the final close are skipped by both (no
    next open to fill at), and signal-exits decided there settle as
    end_of_data instead of "signal" -- both intended fill-lag semantics,
    not bugs."""
    ts_index = {t: i for i, t in enumerate(df["timestamp"])}
    assert len(vec_trades) == len(scalar_trades)
    for s, v in zip(scalar_trades, vec_trades):
        assert s.direction == v.direction
        assert s.exit_reason == v.exit_reason
        assert s.entry_price == pytest.approx(v.entry_price, abs=1e-9)
        assert s.exit_price == pytest.approx(v.exit_price, abs=1e-9)
        assert s.size == pytest.approx(v.size, rel=1e-9)
        assert s.pnl == pytest.approx(v.pnl, abs=1e-6)
        assert s.equity_after == pytest.approx(v.equity_after, abs=1e-6)
        # Both engines fill at open[i + entry_fill_lag_bars]; resting
        # stop/take fills and the end-of-data close do not move either.
        assert v.entry_time == s.entry_time
        assert v.exit_time == s.exit_time


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
    # v5: gapless data so the fastpath's next-open fill coincides with
    # the scalar engine's next-open fill (both read entry_fill_lag_bars=1
    # off the shared RiskConfig); exact timestamp parity is asserted by
    # _assert_trades_match_lag_shift. Signals on
    # the last bars are zeroed: tail entries are an intended semantic
    # difference (no next open to fill at), not a parity break.
    df = _gapless_df()
    sig = _alternating_signals(len(df))
    sig[-5:] = 0
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=0.01, spread_pips=1.0,
                       slippage_pips=0.5, commission_per_trade=0.5, max_trades_per_day=999)

    signals_series = pd.Series(sig)
    scalar_trades, _ = run_execution(
        df, signals_series, risk, stop_loss_pips=stop_pips, take_profit_pips=take_pips,
    )

    outcomes = run_vectorized_batch(
        df, [VectorizedCandidate("c1", sig, stop_pips, take_pips)], risk,
    )
    _assert_trades_match_lag_shift(scalar_trades, outcomes["c1"].trades, df)


def test_vectorized_batch_multiple_candidates_are_independent():
    """Two different parameter sets in the same batch call must not leak
    state into each other -- this is the whole premise of vectorizing
    across columns instead of running each in its own process."""
    # v5: gapless data + zeroed tail so both engines fill identically
    # (see test_vectorized_matches_scalar_engine_trade_by_trade).
    df = _gapless_df(seed=3)
    sig_a = _alternating_signals(len(df), seed=1)
    sig_b = _alternating_signals(len(df), seed=2)
    sig_a[-5:] = 0
    sig_b[-5:] = 0
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
    0-survivor run (see tests/search/test_batch_runner.py)."""
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


def test_is_vectorizable_allows_perbar_stops_rejects_unsupported():
    """v5: per-bar stop_loss_distance / take_profit_distance arrays (ATR
    stops) are now vectorizable; trailing stops, breakeven triggers and
    partial exits still force the scalar path."""
    base = dict(name="x", source_type="python", signals=pd.Series([0]))
    assert is_vectorizable(StrategyResult(**base, stop_loss_pips=20.0, take_profit_pips=40.0))
    assert is_vectorizable(StrategyResult(**base, stop_loss_distance=pd.Series([0.5])))
    assert is_vectorizable(StrategyResult(
        **base, stop_loss_distance=pd.Series([0.5]), take_profit_distance=pd.Series([1.0])))
    assert not is_vectorizable(StrategyResult(
        **base, stop_loss_distance=pd.Series([0.5]), trailing_stop_distance=pd.Series([0.5])))
    assert not is_vectorizable(StrategyResult(
        **base, stop_loss_distance=pd.Series([0.5]), breakeven_trigger_r=1.5))
    assert not is_vectorizable(StrategyResult(
        **base, stop_loss_distance=pd.Series([0.5]), partial_exit={"r_multiple": 1.0, "fraction": 0.5}))
    assert not is_vectorizable(StrategyResult(**base, trailing_stop_distance=pd.Series([0.5])))
    assert not is_vectorizable(StrategyResult(**base, breakeven_trigger_r=1.5))


def test_position_size_parity_with_contract_size():
    """P0-4: _position_size_vec must floor to whole contracts exactly like
    RiskConfig.position_size -- a trade whose intended risk doesn't reach
    one whole contract sizes to 0 (skipped), never a fractional contract."""
    equities = np.array([10_000.0, 50_000.0, 5_000.0, 20_000.0, 0.0])
    pips = np.array([10.0, 25.0, 5.0, 50.0, 10.0])
    for contract_size in (50.0, 5.0):
        risk = RiskConfig(
            initial_balance=50_000.0, risk_mode="percent", risk_value=1.0,
            pip_size=1.0, contract_size=contract_size,
        )
        vec_size = _position_size_vec(equities, pips, risk)
        for eq, pip, vs in zip(equities, pips, vec_size):
            assert vs == pytest.approx(risk.position_size(eq, pip), abs=1e-9)
        # flooring never rounds up: every nonzero size is a whole multiple
        # of contract_size
        for vs in vec_size:
            if vs > 0:
                assert vs / contract_size == pytest.approx(round(vs / contract_size), abs=1e-9)


# ---- from test_vectorized_fastpath_v5.py ----

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
