"""W4 (realism + validation) tests for the v6 bundle.

Covers:
  C1  fastpath commission parity (scalar vs fastpath, multi-contract)
  C2  fastpath daily-loss handling parity on a deep-dip bar
  C3  scalar: liquidation AT the daily floor (covered via the C2 test)
  C4  intrabar ordering: stop resolves before breakeven/trailing/partial
      tightening on the same bar
  C5  InstrumentSpec spread/slippage defaults applied by
      apply_instrument_spec
  C6  market-impact RuntimeWarning on >30-contract entries
  C7  importer roll-awareness warning
  D1  holdout gate: a losing holdout (>= 10 trades) -> NOT READY
  D3  DSR/PBO candidate pools: leaderboard candidates carry real
      config snapshots (no more ValueError for manual candidates)
  D4b FullPipelineConfig.external_trial_count exists (threading is
      exercised through count_all_trials)
  D5  WF-starved cap: no WF/CPCV and no holdout gate -> at most MARGINAL
"""
from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig
from app.backtest.vectorized_fastpath import (
    VectorizedCandidate,
    run_vectorized_batch,
)
from app.data.instrument_specs import apply_instrument_spec, get_instrument_spec
from app.monte_carlo.engine import MonteCarloResult
from app.orchestration.full_pipeline import FullPipelineConfig, _make_verdict
from app.search.robustness import count_all_trials


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _flat_df(n=60, price=100.0, start="2024-01-01", freq="15min"):
    ts = pd.date_range(start, periods=n, freq=freq)
    rows = [(t, price, price, price, price, 1000.0) for t in ts]
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _mc(risk_of_ruin_pct: float = 5.0, strong: bool = False) -> MonteCarloResult:
    kw = dict(
        n_simulations=1000,
        evaluation_pass_probability=95.0 if strong else 80.0,
        first_payout_probability=90.0 if strong else 60.0,
        failure_before_payout_probability=2.0 if strong else 10.0,
        multiple_payout_probability=80.0 if strong else 5.0,
        median_days_to_pass=10.0,
        median_days_to_first_payout=20.0,
        average_days_to_first_payout=22.0,
        median_return_pct=40.0 if strong else 10.0,
        mean_return_pct=42.0 if strong else 11.0,
        expected_payout=5000.0 if strong else 500.0,
        median_payout=4500.0 if strong else 450.0,
        total_simulated_withdrawals=10000.0 if strong else 1000.0,
        median_drawdown_pct=1.0 if strong else 3.0,
        p95_drawdown_pct=2.0 if strong else 6.0,
        worst_drawdown_pct=3.0 if strong else 9.0,
        risk_of_ruin_pct=risk_of_ruin_pct,
        median_max_losing_streak=2.0 if strong else 4.0,
        worst_max_losing_streak=4 if strong else 8,
        per_attempt_pass_probability=95.0 if strong else 75.0,
        per_attempt_payout_probability=90.0 if strong else 55.0,
        total_independent_attempts=1000,
    )
    return MonteCarloResult(**kw)


def _holdout(net: float, pf: float, ho_trades: int = 12, in_trades: int = 50) -> dict:
    return {
        "holdout_statistics": {
            "total_trades": ho_trades, "net_profit": net, "profit_factor": pf,
        },
        "in_sample_statistics": {"total_trades": in_trades},
        "holdout_period": ("2024-06-01", "2024-07-01"),
    }


# ---------------------------------------------------------------------------
# C1: fastpath commission parity (multi-contract)
# ---------------------------------------------------------------------------

def _choppy_gapless(n=300, seed=9):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    rows = []
    prev_close = 100.0
    for i in range(n):
        o = prev_close
        step = rng.normal(0, 1.2)
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.8))
        l = min(o, c) - abs(rng.normal(0, 0.8))
        rows.append((ts[i], o, h, l, c, 1000.0))
        prev_close = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_fastpath_commission_parity_multicontract():
    """Same multi-contract trade must price identically (pnl AND
    per-trade commission) in the scalar engine and the fastpath: flat
    per-trade + per-contract x contracts, not flat-only."""
    df = _choppy_gapless()
    sig = np.zeros(len(df), dtype=np.int8)
    sig[10] = 1
    sig[60] = 0    # signal exit -> fills at bar 61 open
    sig[120] = -1
    sig[200] = 0
    sig[-5:] = 0
    # fixed $1000 risk, 20-point stop. Costs now sit INSIDE the risk budget
    # (worst case per contract = 5*(20+1.5)+1.42 = $108.92) -> 9 whole
    # contracts (10 would risk $1089): commission = 1.0 + 1.42 * 9 = 13.78.
    risk = RiskConfig(
        initial_balance=100_000.0, risk_mode="fixed", risk_value=1000.0,
        pip_size=1.0, contract_size=5.0,
        commission_per_trade=1.0, commission_per_contract=1.42,
        spread_pips=1.0, slippage_pips=0.5, max_trades_per_day=999,
    )
    scalar_trades, _ = run_execution(
        df, pd.Series(sig), risk, stop_loss_pips=20.0, take_profit_pips=None,
    )
    outcomes = run_vectorized_batch(
        df, [VectorizedCandidate("c1", sig, 20.0, None)], risk,
    )
    vec_trades = outcomes["c1"].trades
    assert len(vec_trades) == len(scalar_trades) >= 2
    for s, v in zip(scalar_trades, vec_trades):
        assert v.exit_reason == s.exit_reason
        assert v.commission == pytest.approx(s.commission, abs=1e-9)
        assert v.commission == pytest.approx(1.0 + 1.42 * 9, abs=1e-9)
        assert v.pnl == pytest.approx(s.pnl, abs=1e-6)


# ---------------------------------------------------------------------------
# C2: fastpath daily-loss handling parity on a deep-dip bar
# ---------------------------------------------------------------------------

def test_fastpath_daily_loss_parity_deep_dip():
    """A bar that dips deep enough to breach the daily-loss floor
    intrabar (but closes back above the stop) must force-close at the
    floor liquidation price with reason daily_loss_limit_forced_close on
    BOTH engines -- the old fastpath had no floating check and would
    have ridden through (or stop-exited) instead."""
    df = _flat_df(40)
    # deep-dip bar: low=50 breaches the $1000 daily floor at the $60
    # liquidation price, then recovers to 99 (above the $80 stop).
    df.loc[20, ["open", "high", "low", "close"]] = [100.0, 101.0, 50.0, 99.0]
    sig = np.zeros(len(df), dtype=np.int8)
    sig[5] = 1  # long, filled at bar 6 open = 100
    sig[6:21] = 1  # hold the long signal through the dip bar (a 0 would
    # mean "flat" and trigger a signal exit before the dip)
    # fixed $500 risk, 20-point stop; costs ($1 flat) sit inside the
    # budget -> 4 whole contracts (5 would risk $501) = 20 units;
    # daily floor $1000 -> liquidation 50 points adverse -> $50.
    risk = RiskConfig(
        initial_balance=50_000.0, risk_mode="fixed", risk_value=500.0,
        pip_size=1.0, contract_size=5.0,
        commission_per_trade=1.0, commission_per_contract=0.0,
        spread_pips=0.0, slippage_pips=0.0,
        daily_loss_limit_pct=2.0, max_trades_per_day=999,
    )
    scalar_trades, _ = run_execution(
        df, pd.Series(sig), risk, stop_loss_pips=20.0, take_profit_pips=None,
    )
    outcomes = run_vectorized_batch(
        df, [VectorizedCandidate("c1", sig, 20.0, None)], risk,
    )
    vec_trades = outcomes["c1"].trades

    assert len(scalar_trades) == 1
    assert len(vec_trades) == 1
    s, v = scalar_trades[0], vec_trades[0]
    for t in (s, v):
        assert t.exit_reason == "daily_loss_limit_forced_close"
        assert t.exit_price == pytest.approx(50.0, abs=1e-9)
        assert t.pnl == pytest.approx((50.0 - 100.0) * 20 - 1.0, abs=1e-6)
    assert v.exit_price == pytest.approx(s.exit_price, abs=1e-9)
    assert v.pnl == pytest.approx(s.pnl, abs=1e-6)


def test_fastpath_day_entry_block_after_daily_loss():
    """After a daily-loss forced close, no new entries open for the rest
    of the day on either engine."""
    df = _flat_df(40)
    df.loc[20, ["open", "high", "low", "close"]] = [100.0, 101.0, 50.0, 99.0]
    sig = np.zeros(len(df), dtype=np.int8)
    sig[5] = 1
    sig[6:21] = 1  # hold the long signal through the dip bar
    sig[25] = 1  # same session day, after the forced close -> must be blocked
    risk = RiskConfig(
        initial_balance=50_000.0, risk_mode="fixed", risk_value=500.0,
        pip_size=1.0, contract_size=5.0,
        commission_per_trade=1.0, commission_per_contract=0.0,
        spread_pips=0.0, slippage_pips=0.0,
        daily_loss_limit_pct=2.0, max_trades_per_day=999,
    )
    scalar_trades, _ = run_execution(
        df, pd.Series(sig), risk, stop_loss_pips=20.0, take_profit_pips=None,
    )
    outcomes = run_vectorized_batch(
        df, [VectorizedCandidate("c1", sig, 20.0, None)], risk,
    )
    assert len(scalar_trades) == 1
    assert len(outcomes["c1"].trades) == 1


# ---------------------------------------------------------------------------
# C3: liquidation AT the floor (scalar), incl. prior day-PnL sign
# ---------------------------------------------------------------------------

def test_liquidate_at_floor_accounts_for_prior_day_pnl():
    """The day is already -$400 realized; the $1000 floor leaves $600 of
    room. A long @100 with 50 units must liquidate at 100 - 600/50 =
    $88 -- NOT at the bar's $50 low (old behavior) and NOT at the
    sign-flipped $76 (which would put the day at -$1400, past the
    floor)."""
    df = _flat_df(60)
    # trade 1: long, stopped out for exactly -$400 on the same day
    df.loc[3, ["open", "high", "low", "close"]] = [100.0, 101.0, 91.0, 99.0]
    # trade 2: long @100, then a deep dip to $50 that recovers to 99
    df.loc[20, ["open", "high", "low", "close"]] = [100.0, 101.0, 50.0, 99.0]
    sig = np.zeros(len(df), dtype=np.int8)
    sig[1] = 1
    sig[2:4] = 1   # hold through trade 1's stop bar
    sig[10] = 1    # trade 2, filled at bar 11 open = 100
    sig[11:21] = 1  # hold through the dip bar
    risk = RiskConfig(
        initial_balance=50_000.0, risk_mode="fixed", risk_value=400.0,
        pip_size=1.0, contract_size=5.0,
        commission_per_trade=0.0, commission_per_contract=0.0,
        spread_pips=0.0, slippage_pips=0.0,
        daily_loss_limit_pct=2.0, max_trades_per_day=999,
    )
    # fixed $400 risk, 8-point stop -> 50 units (10 contracts);
    # max_trade_loss = 3x intended risk = $1200, so the $600 floor loss
    # below is NOT clamped.
    scalar_trades, _ = run_execution(
        df, pd.Series(sig), risk, stop_loss_pips=8.0, take_profit_pips=None,
    )
    outcomes = run_vectorized_batch(
        df, [VectorizedCandidate("c1", sig, 8.0, None)], risk,
    )
    vec_trades = outcomes["c1"].trades
    assert len(scalar_trades) == 2 and len(vec_trades) == 2
    assert scalar_trades[0].pnl == pytest.approx(-400.0, abs=1e-6)
    for t in (scalar_trades[1], vec_trades[1]):
        assert t.exit_reason == "daily_loss_limit_forced_close"
        assert t.exit_price == pytest.approx(88.0, abs=1e-9)
        assert t.pnl == pytest.approx(-600.0, abs=1e-6)
    assert vec_trades[1].exit_price == pytest.approx(scalar_trades[1].exit_price, abs=1e-9)


# ---------------------------------------------------------------------------
# C4: intrabar ordering -- stop before tightening
# ---------------------------------------------------------------------------

def test_intrabar_stop_before_tightening():
    """Bar touches the breakeven trigger (high=112, +12 on a 10-point
    risk) AND the resting stop (low=89 < 90): the resting stop filled
    first, so the trade exits at the full stop loss ($90), not the
    breakeven scratch ($100) the old tighten-first ordering produced."""
    df = _flat_df(30)
    df.loc[3, ["open", "high", "low", "close"]] = [100.0, 112.0, 89.0, 105.0]
    sig = np.zeros(len(df), dtype=np.int8)
    sig[1] = 1  # long, filled at bar 2 open = 100, stop 90 (10-pt risk)
    sig[2:4] = 1  # hold through the test bar (a 0 means "flat" -> signal exit)
    risk = RiskConfig(
        initial_balance=50_000.0, risk_mode="fixed", risk_value=1000.0,
        pip_size=1.0, contract_size=5.0,
        commission_per_trade=1.0, commission_per_contract=0.0,
        spread_pips=0.0, slippage_pips=0.0, max_trades_per_day=999,
    )
    trades, _ = run_execution(
        df, pd.Series(sig), risk,
        stop_loss_pips=10.0, take_profit_pips=None,
        breakeven_trigger_r=1.0,
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "stop_loss"
    # full stop loss: 19 contracts (20 would risk $1001 incl. the $1
    # flat commission) = 95 units: (90 - 100) * 95 - $1 commission
    assert t.exit_price == pytest.approx(90.0, abs=1e-9)
    assert t.pnl == pytest.approx(-951.0, abs=1e-6)


def test_intrabar_tightening_still_applies_when_stop_not_touched():
    """Guard: when the bar touches the BE trigger but NOT the resting
    stop, the breakeven move still happens (scratch exit at $100)."""
    df = _flat_df(30)
    # bar 3 touches the BE trigger (high=112, +12 on a 10-pt risk) but NOT
    # the resting stop (low=101 > 90): stop moves to 100, no exit yet.
    df.loc[3, ["open", "high", "low", "close"]] = [100.0, 112.0, 101.0, 105.0]
    # bar 4 then tags the breakeven stop -> scratch exit at 100.
    df.loc[4, ["open", "high", "low", "close"]] = [105.0, 106.0, 99.0, 104.0]
    sig = np.zeros(len(df), dtype=np.int8)
    sig[1] = 1
    sig[2:4] = 1  # hold through bar 3 (a 0 means "flat" -> signal exit)
    risk = RiskConfig(
        initial_balance=50_000.0, risk_mode="fixed", risk_value=1000.0,
        pip_size=1.0, contract_size=5.0,
        commission_per_trade=1.0, commission_per_contract=0.0,
        spread_pips=0.0, slippage_pips=0.0, max_trades_per_day=999,
    )
    trades, _ = run_execution(
        df, pd.Series(sig), risk,
        stop_loss_pips=10.0, take_profit_pips=None,
        breakeven_trigger_r=1.0,
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "stop_loss"
    # breakeven moved the stop to 100 on bar 3; bar 4's low=99 hits it.
    assert t.exit_price == pytest.approx(100.0, abs=1e-9)
    assert t.pnl == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# C5: spec-driven spread/slippage defaults
# ---------------------------------------------------------------------------

def test_spec_spread_slippage_defaults_applied():
    spec = get_instrument_spec("MGC")
    assert spec is not None
    assert spec.default_spread_ticks == 1.0
    assert spec.default_slippage_ticks == 1.0

    risk = RiskConfig()
    assert risk.spread_pips == 0.0 and risk.slippage_pips == 0.0
    applied = apply_instrument_spec(risk, "MGC")
    # 2026-10-07 cost-unit fix: the spec's defaults are TICKS; MGC's tick is
    # 0.10 points, so 1 tick == 0.10 pip (pip_size 1.0), not 1.0.
    assert applied.spread_pips == pytest.approx(1.0 * spec.tick_size / spec.pip_size)
    assert applied.slippage_pips == pytest.approx(1.0 * spec.tick_size / spec.pip_size)
    # pip_size/contract_size/commission behavior unchanged
    assert applied.pip_size == 1.0 and applied.contract_size == 10.0


def test_spec_spread_slippage_never_overwrite_explicit():
    risk = replace(RiskConfig(), spread_pips=2.5, slippage_pips=0.75)
    applied = apply_instrument_spec(risk, "ES")
    assert applied.spread_pips == pytest.approx(2.5)
    assert applied.slippage_pips == pytest.approx(0.75)


def test_spec_spread_defaults_all_specs():
    from app.data.instrument_specs import KNOWN_INSTRUMENTS
    for symbol, spec in KNOWN_INSTRUMENTS.items():
        assert spec.default_spread_ticks > 0, symbol
        assert spec.default_slippage_ticks >= 0, symbol
        applied = apply_instrument_spec(RiskConfig(), symbol)
        # ticks -> price via tick_size -> pips via pip_size (cost-unit fix)
        assert applied.spread_pips == pytest.approx(
            float(spec.default_spread_ticks) * spec.tick_size / spec.pip_size), symbol
        assert applied.slippage_pips == pytest.approx(
            float(spec.default_slippage_ticks) * spec.tick_size / spec.pip_size), symbol


# ---------------------------------------------------------------------------
# C6: market-impact guardrail
# ---------------------------------------------------------------------------

def test_market_impact_warning_on_large_entry():
    """An entry above 30 whole contracts warns (RuntimeWarning) that the
    fill assumed zero market impact; a small entry stays silent."""
    df = _flat_df(20)
    sig = np.zeros(len(df), dtype=np.int8)
    sig[1] = 1
    # 40 contracts: fixed $4000 risk / 20-pt stop = 200 units / 5 = 40
    big = RiskConfig(
        initial_balance=100_000.0, risk_mode="fixed", risk_value=4000.0,
        pip_size=1.0, contract_size=5.0, max_trades_per_day=999,
    )
    with pytest.warns(RuntimeWarning, match="MARKET IMPACT"):
        run_execution(df, pd.Series(sig), big, stop_loss_pips=20.0, take_profit_pips=None)

    small = replace(big, risk_value=100.0)  # 5 units -> 1 contract
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        run_execution(df, pd.Series(sig), small, stop_loss_pips=20.0, take_profit_pips=None)
    msgs = [str(w.message) for w in log]
    assert not any("MARKET IMPACT" in m for m in msgs)


# ---------------------------------------------------------------------------
# C7: importer roll awareness
# ---------------------------------------------------------------------------

def _roll_like_df():
    """Two quiet sessions joined by an overnight gap with a 5x-median-TR
    displacement landing on the 10th of the month (roll window)."""
    rows = []
    base = pd.Timestamp("2024-06-09 09:00")
    for i in range(30):
        t = base + pd.Timedelta(minutes=15 * i)
        rows.append((t, 100.0, 100.5, 99.5, 100.0, 1000.0))
    # session break: next bar is 2024-06-10 (day 10, in the roll window),
    # opened 5 points higher -- 5x the 1.0 median true range.
    base2 = pd.Timestamp("2024-06-10 09:00")
    for i in range(30):
        t = base2 + pd.Timedelta(minutes=15 * i)
        rows.append((t, 105.0, 105.5, 104.5, 105.0, 1000.0))
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_importer_roll_awareness_warning(tmp_path):
    from app.data.importer import import_csv
    p = tmp_path / "roll.csv"
    _roll_like_df().to_csv(p, index=False)
    with pytest.warns(RuntimeWarning, match="[Rr]oll"):
        result = import_csv(str(p), historical_complete=True)
    assert result.is_valid  # warning only -- never a rejection
    assert any("roll" in w.lower() for w in result.warnings)


def test_importer_no_roll_warning_on_clean_data(tmp_path):
    from app.data.importer import import_csv
    p = tmp_path / "clean.csv"
    _roll_like_df().iloc[:30].to_csv(p, index=False)  # single quiet session
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = import_csv(str(p), historical_complete=True)
    assert result.is_valid
    assert not any("roll" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# D1: holdout gate
# ---------------------------------------------------------------------------

def test_holdout_losing_rejects_not_ready():
    """Holdout ran with >= 10 trades and lost money while in-sample
    traded -> NOT READY with a HARD VALIDATION GATE FAILED reason naming
    the holdout stats (fails on baseline: no gate existed)."""
    verdict, reasons, _sc, _rf, _lf = _make_verdict(
        _mc(strong=True), None, None, holdout=_holdout(net=-500.0, pf=0.8),
    )
    assert verdict == "NOT READY"
    fails = [r for r in reasons if "HARD VALIDATION GATE FAILED" in r]
    assert len(fails) == 1
    assert "holdout" in fails[0].lower()
    assert "$-500.00" in fails[0] and "0.80" in fails[0] and "12 trade(s)" in fails[0]


def test_holdout_losing_on_pf_only_rejects():
    """Profit factor < 1.0 alone (net could be near-zero) still trips the
    gate."""
    verdict, reasons, _sc, _rf, _lf = _make_verdict(
        _mc(strong=True), None, None, holdout=_holdout(net=10.0, pf=0.9),
    )
    assert verdict == "NOT READY"
    assert any("HARD VALIDATION GATE FAILED" in r and "holdout" in r.lower() for r in reasons)


def test_holdout_thin_is_unproven_not_failed():
    """Holdout with < 10 trades can't convict -- no hard failure even
    when it lost money."""
    verdict, reasons, _sc, _rf, _lf = _make_verdict(
        _mc(strong=True), None, None, holdout=_holdout(net=-500.0, pf=0.5, ho_trades=3),
    )
    assert not any(
        "HARD VALIDATION GATE FAILED" in r and "holdout" in r.lower() for r in reasons
    )


def test_holdout_winning_does_not_reject():
    verdict, reasons, _sc, _rf, _lf = _make_verdict(
        _mc(strong=True), None, None, holdout=_holdout(net=500.0, pf=1.8),
    )
    assert not any(
        "HARD VALIDATION GATE FAILED" in r and "holdout" in r.lower() for r in reasons
    )
    assert verdict == "READY"


def test_holdout_gate_respects_advisory_mode():
    verdict, reasons, _sc, _rf, _lf = _make_verdict(
        _mc(strong=True), None, None, holdout=_holdout(net=-500.0, pf=0.8),
        gates_advisory_only=True,
    )
    assert verdict != "NOT READY" or not any(
        "HARD VALIDATION GATE FAILED" in r for r in reasons
    )
    assert any("ADVISORY ONLY" in r and "holdout" in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# D5: WF-starved cap
# ---------------------------------------------------------------------------

def test_wf_starved_cap():
    """No walk-forward, no CPCV, holdout gate didn't run (holdout=None):
    an Elite-tier scorecard is capped at MARGINAL, not READY."""
    verdict, reasons, scorecard, _rf, _lf = _make_verdict(_mc(strong=True), None, None)
    assert scorecard.tier == "Elite"
    assert verdict == "MARGINAL"
    assert any("CAPPED AT MARGINAL" in r and "walk-forward" in r.lower() for r in reasons)


def test_wf_starved_cap_not_applied_when_holdout_gate_ran():
    """A holdout that ran with enough trades and won leaves the verdict
    alone (gate ran -> D5 does not fire)."""
    verdict, _reasons, _sc, _rf, _lf = _make_verdict(
        _mc(strong=True), None, None, holdout=_holdout(net=500.0, pf=1.8),
    )
    assert verdict == "READY"


# ---------------------------------------------------------------------------
# D3: DSR/PBO candidate pools for manual candidates
# ---------------------------------------------------------------------------

def test_leaderboard_candidate_spec_manual_pool():
    """The D3-fixed leaderboard shape (config snapshot per candidate)
    flows through _leaderboard_candidate_spec without raising -- this is
    what the DSR/PBO gates re-backtest."""
    from app.optimize.parameter_space import apply_genome, extract_genome
    from app.optimize.walkforward_ga import WalkforwardGACandidate
    from app.orchestration.full_pipeline import _leaderboard_candidate_spec
    from app.strategy.manual import ManualStrategy

    config = {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }
    strategy = ManualStrategy(config)
    genes = extract_genome(config)
    assert genes, "expected tunable genes in the SMA config"
    # the exact per-candidate snapshot walkforward_ga now takes (D3)
    genome = [float(g.lo) for g in genes]
    cand = WalkforwardGACandidate(
        genome=genome, fitness=1.0, oos_trade_count=5,
        config=apply_genome(strategy.config, genes, genome),
    )
    spec = _leaderboard_candidate_spec(cand, "manual")
    assert spec["source_type"] == "manual"
    assert spec["config"] == apply_genome(strategy.config, genes, genome)
    # and the spec actually builds a runnable strategy (the DSR pool use)
    from app.search.strategy_space import build_strategy_from_spec
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        built = build_strategy_from_spec(spec, td)
        assert built is not None


def test_walkforward_ga_leaderboard_carries_configs():
    """End-to-end D3 proof on a tiny GA run: every leaderboard candidate
    carries a real config snapshot (manual) so the DSR/PBO pool is
    non-empty and re-backtestable."""
    from app.monte_carlo.engine import MonteCarloConfig
    from app.optimize.refinement import RefinementConfig
    from app.optimize.walkforward_ga import run_walkforward_aware_refinement
    from app.orchestration.full_pipeline import _leaderboard_candidate_spec
    from app.prop.simulator import PropRules
    from app.strategy.manual import ManualStrategy

    rng = np.random.default_rng(3)
    n = 900
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        o = price
        c = o + 0.00015 + rng.normal(0, 0.00006)
        h = max(o, c) + abs(rng.normal(0, 0.00003))
        l = min(o, c) - abs(rng.normal(0, 0.00003))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    config = {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }
    strategy = ManualStrategy(config)
    refine_cfg = RefinementConfig(population_size=4, generations=1, search_monte_carlo_sims=10)
    result = run_walkforward_aware_refinement(
        df, strategy, RiskConfig(), PropRules(),
        MonteCarloConfig(n_simulations=10),
        refinement_config=refine_cfg, n_folds=2, parallel=False,
    )
    assert len(result.leaderboard) == 4
    pool = []
    for cand in result.leaderboard:
        assert cand.config is not None, "D3: leaderboard candidate must carry a config snapshot"
        pool.append(_leaderboard_candidate_spec(cand, "manual"))
    assert len(pool) == 4 and all(p["config"] for p in pool)


# ---------------------------------------------------------------------------
# D4(b): external trial count
# ---------------------------------------------------------------------------

def test_external_trial_count_config_and_threading():
    """FullPipelineConfig.external_trial_count defaults to 0 and feeds
    count_all_trials' extra_trial_counts so hand-tuned strategies don't
    get n_trials=1."""
    cfg = FullPipelineConfig()
    assert cfg.external_trial_count == 0
    assert count_all_trials(baseline_count=1, ga_total_evaluations=0) == 1
    assert count_all_trials(
        baseline_count=1, ga_total_evaluations=0,
        extra_trial_counts=(cfg.external_trial_count,),
    ) == 1
    cfg2 = replace(cfg, external_trial_count=25)
    assert count_all_trials(
        baseline_count=1, ga_total_evaluations=0,
        extra_trial_counts=(cfg2.external_trial_count,),
    ) == 26
