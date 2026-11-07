"""2026-11-06 independent audit: skip-memo parity (finding 1), cache thread-safety (finding 3), `dayofweek` removal
(finding 5) and the equity-curve signal-exit timing fix (finding 4).

The skip-memo tests run the SAME inputs through the engine with the memo off (the old path) and on, then require
bit-identical trades, equity curve and counters on a plain config, an adaptive-risk config and a prop-account config.
"""
from __future__ import annotations

import re
import sys
import threading
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.backtest import engine as engine_mod
from app.backtest import execution as ex
from app.backtest.adaptive_risk import AdaptiveRiskConfig, AdaptiveRiskRule
from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules

warnings.simplefilter("ignore")

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ helpers
def _es_frame(n=6000, seed=11, start="2024-03-04 00:00"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="1min")
    close = 4000 + np.cumsum(rng.normal(0, 0.6, n))
    open_ = np.concatenate([[4000.0], close[:-1]])
    high = np.maximum(open_, close) + rng.random(n) * 0.5
    low = np.minimum(open_, close) - rng.random(n) * 0.5
    return pd.DataFrame(dict(timestamp=ts, open=open_, high=high, low=low, close=close, volume=100))


def _persistent_signals(n, block=240, seed=5):
    """Long/short/flat regimes that persist for hundreds of bars -- the shape that made the memo worth having."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=int)
    for s in range(0, n, block):
        out[s:s + block] = rng.choice([-1, 0, 1, 1, -1])
    return pd.Series(out)


def _risk(**kw):
    base = dict(initial_balance=50_000.0, risk_mode="fixed", risk_value=250.0, pip_size=0.25, contract_size=50.0,
                spread_pips=1.0, slippage_pips=1.0, commission_per_contract=2.4, max_trades_per_day=50,
                sizing_mode="skip")
    base.update(kw)
    return RiskConfig(**base)


def _prop_rules():
    return PropRules(account_size=50_000, max_drawdown_pct=4.0, drawdown_type="trailing", daily_loss_limit_pct=2.0,
                     evaluation_profit_target_pct=6.0, dd_basis="floating", daily_loss_basis="floating",
                     daily_loss_action="lock_day")


_COUNTER_KEYS = ("sizing_summary", "entry_block_counts", "zero_size_contract_floor_count", "min_contract_rescue_count",
                 "adaptive_probe_count", "same_bar_both_touch", "account_reset_events", "prop_attempts")


def _run(df, sig, risk, memo, **kw):
    old = ex._SKIP_MEMO_ENABLED
    ex._SKIP_MEMO_ENABLED = memo
    try:
        return ex.run_execution(df, sig, risk, kw.pop("sl_pips", 160), kw.pop("tp_pips", 320), **kw)
    finally:
        ex._SKIP_MEMO_ENABLED = old


def _assert_identical(a, b):
    trades_a, eq_a = a
    trades_b, eq_b = b
    assert [asdict(t) for t in trades_a] == [asdict(t) for t in trades_b]
    assert np.array_equal(eq_a["equity"].to_numpy(), eq_b["equity"].to_numpy())
    for k in _COUNTER_KEYS:
        assert eq_a.attrs.get(k) == eq_b.attrs.get(k), k


def _varying_stop_distance(n, seed=2):
    """Per-bar stop distance in price points: blocks of tight (fits the budget) and wide (always skipped) stops, so the
    memo key changes on bar boundaries and after fills move equity."""
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for s in range(0, n, 97):
        d[s:s + 97] = rng.choice([3.0, 4.0, 40.0, 60.0])
    return pd.Series(d)


# ------------------------------------------------------------------ finding 1: parity
def test_skip_memo_plain_identical_and_actually_skips():
    df = _es_frame()
    sig = _persistent_signals(len(df))
    risk = _risk()
    off = _run(df, sig, risk, memo=False)
    on = _run(df, sig, risk, memo=True)
    _assert_identical(off, on)
    assert on[1].attrs["sizing_summary"]["skip_reasons"], "the workload must exercise the skip path or this proves nothing"
    assert on[1].attrs["entry_block_counts"]["zero_contract_sizing"] > 100


def test_skip_memo_per_bar_distances_identical():
    df = _es_frame(seed=21)
    sig = _persistent_signals(len(df), seed=9)
    dist = _varying_stop_distance(len(df))
    risk = _risk()
    off = _run(df, sig, risk, memo=False, stop_loss_distance=dist, take_profit_distance=dist * 2)
    on = _run(df, sig, risk, memo=True, stop_loss_distance=dist, take_profit_distance=dist * 2)
    _assert_identical(off, on)
    assert len(on[0]) > 0 and on[1].attrs["entry_block_counts"]["zero_contract_sizing"] > 0


def test_skip_memo_price_derived_fallback_stop_identical():
    """No stop at all -> the stop is a % of the entry price, so the price must be part of the memo key."""
    df = _es_frame(seed=33)
    sig = _persistent_signals(len(df), seed=3)
    risk = _risk(risk_value=120.0)
    off = _run(df, sig, risk, memo=False, sl_pips=None, tp_pips=None)
    on = _run(df, sig, risk, memo=True, sl_pips=None, tp_pips=None)
    _assert_identical(off, on)


def test_skip_memo_adaptive_risk_identical():
    df = _es_frame(seed=44)
    sig = _persistent_signals(len(df), seed=4)
    dist = _varying_stop_distance(len(df), seed=6)
    adaptive = AdaptiveRiskConfig(enabled=True, rules=[AdaptiveRiskRule("consecutive_losses", 2, 0.5),
                                                       AdaptiveRiskRule("daily_losses", 3, 0.0)])
    risk = _risk()
    kw = dict(stop_loss_distance=dist, take_profit_distance=dist * 2, adaptive_risk=adaptive)
    off = _run(df, sig, risk, memo=False, **kw)
    on = _run(df, sig, risk, memo=True, **kw)
    _assert_identical(off, on)
    assert len(on[0]) > 0


def test_skip_memo_prop_account_identical():
    df = _es_frame(n=9000, seed=55)
    sig = _persistent_signals(len(df), seed=8)
    dist = _varying_stop_distance(len(df), seed=7)
    risk = _risk(account_model="prop", prop_account_rules=_prop_rules())
    kw = dict(stop_loss_distance=dist, take_profit_distance=dist * 2, attempt_mode="chain")
    off = _run(df, sig, risk, memo=False, **kw)
    on = _run(df, sig, risk, memo=True, **kw)
    _assert_identical(off, on)
    assert on[1].attrs["prop_attempts"], "prop mode must have recorded attempts"


# ------------------------------------------------------------------ finding 3: cache locks
def test_day_index_cache_survives_thread_stress():
    errors: list[str] = []
    frames = [_es_frame(n=60, seed=i, start=f"2024-{1 + i % 12:02d}-{1 + i % 27:02d} 00:00") for i in range(64)]
    for i, f in enumerate(frames):  # make every fingerprint distinct
        f["timestamp"] = f["timestamp"] + pd.Timedelta(seconds=i)
    sig = pd.Series(np.zeros(60, dtype=int))
    risk = _risk()
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)

    def worker(offset):
        try:
            for k in range(40):
                ex.run_execution(frames[(offset * 7 + k) % 64], sig, risk, 160, 320)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    try:
        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old_interval)
    assert not errors, errors[:3]
    assert len(ex._DAY_INDEX_CACHE) <= ex._DAY_INDEX_MAX


def test_lookahead_cache_survives_thread_stress(monkeypatch):
    monkeypatch.setattr(engine_mod, "_LOOKAHEAD_CACHE_MAX", 8)
    errors: list[str] = []
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)

    def worker(offset):
        try:
            for k in range(600):
                key = ("k", (offset * 13 + k) % 64)
                if engine_mod._lookahead_cache_get(key) is None:
                    engine_mod._lookahead_cache_put(key, object())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    try:
        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old_interval)
    assert not errors, errors[:3]
    assert len(engine_mod._LOOKAHEAD_CACHE) <= 8
    assert engine_mod._lookahead_cache_get(None) is None


# ------------------------------------------------------------------ finding 5: dayofweek
def test_no_removed_pandas_dayofweek_attribute_left_in_app():
    pat = re.compile(r"\.dayofweek\b")
    offenders = []
    for p in (REPO / "app").rglob("*.py"):
        for ln, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{p.relative_to(REPO)}:{ln}")
    assert not offenders, offenders[:10]


# ------------------------------------------------------------------ finding 4: equity timing
def _probe_frame():
    """12 bars. Long entry decided on bar 2 (fills bar 3 open = 100). Flat signal from bar 6 (close 100); the exit fills on
    bar 7 open = 110. No costs."""
    ts = pd.date_range("2024-05-06 10:00", periods=12, freq="1min")
    o = np.full(12, 100.0)
    h = np.full(12, 100.0)
    l = np.full(12, 100.0)
    c = np.full(12, 100.0)
    for i in range(7, 12):
        o[i] = h[i] = l[i] = c[i] = 110.0
    df = pd.DataFrame(dict(timestamp=ts, open=o, high=h, low=l, close=c, volume=1))
    sig = pd.Series([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    return df, sig


def _probe_risk():
    return RiskConfig(initial_balance=10_000.0, pip_size=1.0, contract_size=1.0, spread_pips=0.0, slippage_pips=0.0,
                      commission_per_contract=0.0, commission_per_trade=0.0, sizing_mode="fixed_contracts",
                      fixed_contracts=1, risk_mode="fixed", risk_value=1000.0)


def _run_probe(timing):
    df, sig = _probe_frame()
    old = ex._EQUITY_SIGNAL_EXIT_TIMING
    ex._EQUITY_SIGNAL_EXIT_TIMING = timing
    try:
        return ex.run_execution(df, sig, _probe_risk(), 50, None)
    finally:
        ex._EQUITY_SIGNAL_EXIT_TIMING = old


def test_signal_exit_pnl_lands_on_the_fill_bar_not_the_decision_bar():
    trades, eq = _run_probe(True)
    assert len(trades) == 1 and trades[0].exit_reason == "signal"
    pnl = trades[0].pnl
    assert pnl == pytest.approx(10.0) and trades[0].exit_price == pytest.approx(110.0)
    curve = eq["equity"].to_numpy()
    # bar 6 is the decision bar (close 100): the position is still open, so the point carries NO realized gain yet
    assert curve[6] == pytest.approx(10_000.0)
    # bar 7 is the fill bar: the realized P&L lands here
    assert curve[7] == pytest.approx(10_000.0 + pnl)
    assert curve[-1] == pytest.approx(10_000.0 + pnl)


def test_old_timing_is_still_reproducible_behind_the_switch():
    trades, eq = _run_probe(False)
    curve = eq["equity"].to_numpy()
    assert curve[6] == pytest.approx(10_000.0 + trades[0].pnl)  # the documented one-bar peek


def test_timing_fix_changes_only_the_equity_curve_not_the_trades():
    df = _es_frame(seed=77)
    sig = _persistent_signals(len(df), seed=12)
    risk = _risk(risk_value=2500.0)
    old, new = ex._EQUITY_SIGNAL_EXIT_TIMING, None
    try:
        ex._EQUITY_SIGNAL_EXIT_TIMING = False
        t_old, e_old = ex.run_execution(df, sig, risk, 40, 80)
        ex._EQUITY_SIGNAL_EXIT_TIMING = True
        t_new, e_new = ex.run_execution(df, sig, risk, 40, 80)
    finally:
        ex._EQUITY_SIGNAL_EXIT_TIMING = old
    assert [asdict(t) for t in t_old] == [asdict(t) for t in t_new]
    assert any(t.exit_reason == "signal" for t in t_new), "workload must contain signal exits"
    assert not np.array_equal(e_old["equity"].to_numpy(), e_new["equity"].to_numpy())
    # same ledger at the end
    assert e_old["equity"].iloc[-1] == pytest.approx(e_new["equity"].iloc[-1])
