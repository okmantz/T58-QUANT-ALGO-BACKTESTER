"""v9.16 audit fixes: EOD trailing ratchet, session flatten, preset field
parity, loss clamp, 30-day payout replay and dependence-aware intervals."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig, with_prop_safety_defaults
from app.prop.presets import get_preset
from app.prop.simulator import PropRules

warnings.simplefilter("ignore")


def _risk(rules, **kw):
    base = dict(initial_balance=50_000.0, risk_mode="fixed", risk_value=500.0, pip_size=0.25, contract_size=50.0,
                spread_pips=0.0, slippage_pips=0.0, commission_per_contract=0.0, max_trades_per_day=20,
                sizing_mode="fixed_contracts", fixed_contracts=1, account_model="prop", prop_account_rules=rules)
    base.update(kw)
    return RiskConfig(**base)


def _flat_frame(ts, price=4000.0):
    n = len(ts)
    p = np.full(n, price)
    return pd.DataFrame(dict(timestamp=ts, open=p.copy(), high=p.copy(), low=p.copy(), close=p.copy(), volume=1))


def _set(df, i, o, h, l, c):
    df.loc[i, ["open", "high", "low", "close"]] = [o, h, l, c]


def _two_day_frame():
    d1 = pd.date_range("2024-01-02 15:00", periods=24, freq="15min")
    d2 = pd.date_range("2024-01-03 15:00", periods=24, freq="15min")
    df = _flat_frame(d1.append(d2))
    # day 1: +25pt winner, then -25pt loser (net 0, but realized balance peaks +$1,250 intraday)
    _set(df, 4, 4000, 4026, 4000, 4026)
    for i in range(5, 24):
        _set(df, i, 4026, 4026, 4026, 4026)
    _set(df, 10, 4026, 4026, 4000, 4000)
    for i in range(11, 24):
        _set(df, i, 4000, 4000, 4000, 4000)
    # day 2: one -25pt loser
    _set(df, 24 + 4, 4000, 4000, 3974, 3974)
    for i in range(24 + 5, 48):
        _set(df, i, 3974, 3974, 3974, 3974)
    sig = pd.Series(np.zeros(len(df), dtype=int))
    for i in (2, 8, 24 + 2):
        sig.iloc[i:i + 2] = 1      # entry signal held until the (intrabar) stop/target fires
    return df, sig


def test_eod_trailing_peak_moves_only_at_session_close():
    """Day 1 realized balance peaks at 51,250 intraday but closes at 50,000.
    Under an END-OF-DAY trailing MLL the floor must stay 48,000; day 2's -1,250
    (48,750 balance) is then a survivable day, not a bust. The old engine fed
    every close as 'last of day' and ratcheted the peak to 51,250 (floor 49,250)."""
    rules = PropRules(account_size=50_000, evaluation_profit_target_pct=50.0, daily_loss_limit_pct=100.0,
                      max_drawdown_pct=4.0, drawdown_type="trailing", dd_basis="eod",
                      daily_loss_basis="realized", trailing_distance_basis="account",
                      min_trading_days=1, consistency_rule_pct=None)
    df, sig = _two_day_frame()
    trades, eq = run_execution(df, sig, _risk(rules), 100, 100, attempt_mode="single")
    assert len(trades) == 3, [t.exit_reason for t in trades]
    att = eq.attrs["prop_attempts"][0]
    assert att["outcome"] != "failed", att


def test_session_flatten_closes_position_and_blocks_entries():
    rules = PropRules(account_size=50_000, evaluation_profit_target_pct=50.0, daily_loss_limit_pct=100.0,
                      max_drawdown_pct=10.0, drawdown_type="trailing", dd_basis="eod", daily_loss_basis="realized",
                      min_trading_days=1, consistency_rule_pct=None, flatten_time_ct="15:45")
    ts = pd.date_range("2024-01-02 20:00", periods=16, freq="15min")   # 14:00 CT .. 17:45 CT
    df = _flat_frame(ts)
    sig = pd.Series(np.zeros(len(df), dtype=int))
    sig.iloc[1:7] = 1    # holds from 14:30 CT; signal ends exactly at the 15:45 CT flatten bar
    sig.iloc[9] = 1      # 16:15 CT: inside the flatten zone -> entry must be blocked
    trades, eq = run_execution(df, sig, _risk(rules), 400, 400, attempt_mode="single")
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason in ("session_close", "session_flatten_forced_close")
    assert pd.Timestamp(t.exit_time).tz_localize(None) == pd.Timestamp("2024-01-02 21:45")
    assert eq.attrs["session_flatten_close_count"] == 1


def test_flatten_off_by_default_keeps_overnight_position():
    rules = PropRules(account_size=50_000, evaluation_profit_target_pct=50.0, daily_loss_limit_pct=100.0,
                      max_drawdown_pct=10.0, dd_basis="eod", daily_loss_basis="realized", min_trading_days=1,
                      consistency_rule_pct=None)
    ts = pd.date_range("2024-01-02 20:00", periods=16, freq="15min")
    df = _flat_frame(ts)
    sig = pd.Series(np.zeros(len(df), dtype=int))
    sig.iloc[1:] = 1
    trades, _ = run_execution(df, sig, _risk(rules), 400, 400, attempt_mode="single")
    assert trades and trades[0].exit_reason in ("end_of_data", "attempt_end_forced_close")


def test_lucid_preset_carries_every_rule_field():
    p = get_preset("lucid_50k")
    r = p.to_prop_rules()
    assert r.trailing_distance_basis == "account"
    assert r.flatten_time_ct == "15:45"
    # fixed $2,000 trailing distance, locks at 50,100
    assert r.max_drawdown_pct == 4.0 and r.trailing_lock is True


def test_prop_runs_do_not_clamp_losses_at_three_r():
    rules = get_preset("lucid_50k").to_prop_rules()
    r = with_prop_safety_defaults(RiskConfig(initial_balance=50_000.0), rules)
    assert r.max_loss_per_trade_pct == pytest.approx(rules.max_drawdown_pct)
    # an explicit caller value is never overridden
    r2 = with_prop_safety_defaults(RiskConfig(initial_balance=50_000.0, max_loss_per_trade_pct=1.0), rules)
    assert r2.max_loss_per_trade_pct == 1.0


def test_block_bootstrap_ci_is_wider_than_wilson_for_overlapping_starts():
    from app.prop.attempt_replay import _block_bootstrap_ci, _wilson
    rng = np.random.default_rng(1)
    # 60 overlapping starts, but the outcome only changes every ~15 starts (4 independent windows)
    flags = np.repeat(rng.integers(0, 2, 4), 15).astype(float)
    k = int(flags.sum())
    w_lo, w_hi = _wilson(k, len(flags))
    b_lo, b_hi = _block_bootstrap_ci(flags, 15)
    assert (b_hi - b_lo) > (w_hi - w_lo) * 0.9
    assert _block_bootstrap_ci(flags, len(flags)) == (0.0, 1.0)     # one block = no replication


def test_attempt_replay_tracks_first_payout_within_30_days():
    from app.prop.attempt_replay import run_attempt_replay
    rng = np.random.default_rng(3)
    n = 6000
    ts = pd.date_range("2024-01-02 14:30", periods=n, freq="15min")
    drift = 0.35
    c = 4000 + np.cumsum(rng.normal(drift, 2.0, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + 0.5
    l = np.minimum(o, c) - 0.5
    df = pd.DataFrame(dict(timestamp=ts, open=o, high=h, low=l, close=c, volume=1))
    sig = pd.Series(np.where(np.arange(n) % 8 == 0, 1, 0))
    rules = PropRules(account_size=50_000, evaluation_profit_target_pct=3.0, daily_loss_limit_pct=100.0,
                      max_drawdown_pct=6.0, drawdown_type="trailing", dd_basis="eod", daily_loss_basis="realized",
                      min_trading_days=1, consistency_rule_pct=None, payout_frequency_days=2,
                      winning_days_for_payout=0, funded_consistency_rule_pct=None)
    res = run_attempt_replay(df, sig, _risk(rules, sizing_mode="fit_stop"), rules, 20, 40,
                             horizon_days=60, n_starts=10, track_payout=True, payout_windows=(30, 45))
    assert res.payout_tracked and set(res.payout_within_rate) == {30, 45}
    assert res.payout_within_rate[30] <= res.payout_within_rate[45] <= 1.0
    lo, hi = res.payout_within_ci[30]
    assert 0.0 <= lo <= hi <= 1.0
    assert res.n_effective > 0 and res.horizon_days == 60
    assert "First payout" in res.render()


def test_verdict_wrapper_demotes_ready_on_low_30_day_payout_rate(monkeypatch):
    from types import SimpleNamespace

    import app.orchestration.full_pipeline as fp
    monkeypatch.setattr(fp, "_make_verdict_core", lambda *a, **k: ("READY", ["ok"], {}, False, False))
    mc = SimpleNamespace(sample_ok=True, per_attempt_pass_probability=60.0)
    low = {"rate": 0.10, "ci": (0.02, 0.30), "n": 36, "n_effective": 9.0, "window": 30, "source": "out-of-sample holdout"}
    v, reasons, *_ = fp._make_verdict(mc, payout30=low, payout30_min_pct=35.0)
    assert v == "MARGINAL" and any("30-DAY PAYOUT" in r for r in reasons)
    ok = {**low, "rate": 0.55, "ci": (0.40, 0.70)}
    assert fp._make_verdict(mc, payout30=ok, payout30_min_pct=35.0)[0] == "READY"
    # too few independent windows -> no gate (the report still prints the numbers)
    thin = {**low, "n_effective": 1.5}
    assert fp._make_verdict(mc, payout30=thin, payout30_min_pct=35.0, payout30_min_windows=3.0)[0] == "READY"


def test_block_bootstrap_never_claims_zero_width_interval():
    from app.prop.attempt_replay import _block_bootstrap_ci
    lo, hi = _block_bootstrap_ci([0.0] * 48, 1)
    assert lo == 0.0 and hi > 0.03


def test_roll_detection_ignores_weekend_gaps_in_non_roll_months():
    import numpy as np

    from app.data.continuous_contract import detect_rolls
    n = 400
    ts = pd.date_range("2024-02-01 15:00", periods=n, freq="1h")
    base = 4000 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, n))
    df = pd.DataFrame(dict(timestamp=ts, open=base, high=base + 1, low=base - 1, close=base, volume=1))
    # a +30pt gap after a >2h break on Feb 9 (a real weekend gap, not a roll)
    brk = int((pd.Timestamp("2024-02-09 15:00") - ts[0]) / pd.Timedelta("1h"))
    df.loc[brk:, ["open", "high", "low", "close"]] += 30.0
    df = df.drop(index=range(brk - 3, brk)).reset_index(drop=True)
    assert detect_rolls(df, roll_days=None)                                   # size+break alone would call it a roll
    assert detect_rolls(df, roll_months=(3, 6, 9, 12), roll_days=range(4, 22)) == []   # calendar says February is no roll month
