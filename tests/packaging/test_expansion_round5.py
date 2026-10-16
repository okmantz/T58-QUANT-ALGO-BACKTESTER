"""Tests for expansion round 5:
  - 7 new indicators (ADX, Stochastic, CCI, OBV/OBV-EMA, Keltner Channel,
    Donchian Channel levels, SuperTrend) in app.strategy.indicators, and
    their dispatch through app.strategy.manual's condition engine.
  - 7 new strategy families built on them in app.search.strategy_space
    (covered end-to-end for "produces at least one trade" by the existing
    parametrized test in test_strategy_space.py; this file adds indicator-
    level unit coverage the parametrized test doesn't).
  - Search Lab's new graveyard write-back (app.search.batch_runner's
    _write_search_graveyard_entries), the counterpart to Evolution Lab's
    long-standing one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.search.batch_runner import _write_search_graveyard_entries
from app.search.graveyard import graveyard_path_for, load_graveyard
from app.strategy.indicators import (
    adx, build_indicator_series, cci, donchian, keltner, obv, obv_ema, stochastic, supertrend,
)
from app.strategy.manual import ManualStrategy


def _trending_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.cumsum(rng.normal(0.05, 0.5, n))
    close = 100 + drift
    high = close + rng.random(n) * 0.5
    low = close - rng.random(n) * 0.5
    openp = close + rng.normal(0, 0.1, n)
    vol = rng.random(n) * 1000
    return pd.DataFrame({
        "timestamp": ts, "open": openp, "high": high, "low": low, "close": close, "volume": vol,
    })


# ---------------------------------------------------------------------------
# Indicator-level sanity
# ---------------------------------------------------------------------------

def test_adx_is_bounded_0_to_100():
    df = _trending_df()
    result = adx(df, 14)
    valid = result.dropna()
    assert len(valid) > 0
    assert (valid >= 0).all() and (valid <= 100).all()


def test_stochastic_k_and_d_are_bounded_0_to_100():
    df = _trending_df()
    k, d = stochastic(df, 14)
    assert (k >= 0).all() and (k <= 100).all()
    assert (d >= 0).all() and (d <= 100).all()


def test_cci_is_unbounded_and_reacts_to_a_price_spike():
    df = _trending_df(n=60)
    baseline = cci(df, 20)
    df.loc[50, "close"] = df.loc[50, "close"] + 50  # sharp spike well above the recent range
    df.loc[50, "high"] = df.loc[50, "close"] + 0.1
    spiked = cci(df, 20)
    assert spiked.iloc[50] > baseline.iloc[50]


def test_obv_accumulates_signed_volume_and_ema_smooths_it():
    df = _trending_df(n=60)
    series = obv(df)
    smoothed = obv_ema(df, 10)
    assert len(series) == len(df)
    assert smoothed.dropna().shape[0] > 0
    # obv_ema must NOT just be constant/zero -- it should track obv's own trend
    assert smoothed.dropna().std() > 0


def test_obv_is_all_zero_without_a_volume_column():
    df = _trending_df().drop(columns=["volume"])
    assert (obv(df) == 0.0).all()


def test_keltner_upper_is_always_above_lower_where_defined():
    df = _trending_df()
    mid, upper, lower = keltner(df, 20, 2.0)
    valid = upper.notna() & lower.notna()
    assert valid.any()
    assert (upper[valid] > lower[valid]).all()
    assert (mid[valid] > lower[valid]).all() and (mid[valid] < upper[valid]).all()


def test_donchian_upper_and_lower_never_use_the_current_bar():
    """Regression guard for the exact 'close > highest_high(window including
    current bar)' bug class test_strategy_space.py's own docstring warns
    about -- a naive (non-shifted) Donchian upper can never be exceeded by
    that same bar's own close, since the bar's own high is part of the
    window. Confirmed here directly: force one bar's high to a new extreme
    and check donchian_upper does NOT reflect it until the FOLLOWING bar.
    """
    df = _trending_df(n=60)
    df.loc[40, "high"] = 10_000.0  # an obvious new all-time extreme
    mid, upper, lower = donchian(df, 20)
    assert upper.iloc[40] < 10_000.0  # today's own new high must not appear in today's channel
    assert upper.iloc[41] == 10_000.0  # but must appear starting the very next bar


def test_supertrend_direction_is_always_plus_or_minus_one_or_nan():
    df = _trending_df(n=500)
    line, direction = supertrend(df, 10, 3.0)
    valid = direction.dropna()
    assert len(valid) > 0
    assert set(valid.unique()) <= {1.0, -1.0}


def test_supertrend_actually_flips_on_a_realistic_series():
    """Regression guard for the exact warmup-NaN-freeze bug this round hit
    during development: once the ATR-dependent band went NaN briefly
    during warmup, plain `<`/`>` comparisons against NaN (both False)
    silently froze the ratchet at NaN forever, producing zero flips ever."""
    df = _trending_df(n=1000, seed=7)
    _, direction = supertrend(df, 10, 3.0)
    flips = (direction.diff().fillna(0) != 0).sum()
    assert flips > 0


@pytest.mark.parametrize("kind", [
    "adx", "stoch_k", "stoch_d", "cci", "obv", "obv_ema",
    "keltner_mid", "keltner_upper", "keltner_lower",
    "donchian_mid", "donchian_upper", "donchian_lower",
    "supertrend_line", "supertrend_direction",
])
def test_build_indicator_series_dispatches_every_new_kind(kind):
    df = _trending_df()
    series = build_indicator_series(df, kind, period=14, column="close")
    assert len(series) == len(df)


@pytest.mark.parametrize("kind", [
    "adx", "stoch_k", "cci", "obv", "obv_ema",
    "keltner_upper", "donchian_upper", "supertrend_direction",
])
def test_manual_strategy_condition_engine_accepts_every_new_kind(kind):
    """The condition engine (app.strategy.manual) keeps its own separate
    whitelist of recognized operand kinds -- a kind can be fully supported
    by build_indicator_series and still raise StrategyError here if it was
    never added to that whitelist too. Exercises the actual code path a
    generated family's entry/exit condition runs through."""
    df = _trending_df()
    config = {
        "name": "kind dispatch smoke test",
        "entry_conditions": {
            "long": [{"left": {"type": kind, "period": 10}, "operator": ">", "right": {"type": "value", "value": -1e9}}],
            "short": [],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": {"stop_type": "atr", "stop_value": 1.5, "target_type": "atr", "target_value": 2.0},
    }
    strategy = ManualStrategy(config)
    result = strategy.generate(df)
    assert len(result.signals) == len(df)


# ---------------------------------------------------------------------------
# Search Lab graveyard write-back
# ---------------------------------------------------------------------------

def _stage3_record(candidate_id, family, passed, **overrides):
    rec = {
        "candidate_id": candidate_id, "family": family, "passed_stage3_gate": passed,
        "config": {"lookback": 20}, "gate_notes": "", "fitness": 1.23,
        "mc_summary": {"evaluation_pass_probability": 40.0},
        "robustness": {"stability_ratio": 0.5},
        "walk_forward": {"walk_forward_efficiency": -0.2},
    }
    rec.update(overrides)
    return rec


def test_write_search_graveyard_entries_records_only_failures(tmp_path, monkeypatch):
    monkeypatch.setattr("app.search.graveyard.get_app_base_dir", lambda: tmp_path)
    records = [
        _stage3_record("cand-pass", "trend_breakout", True),
        _stage3_record("cand-fail-1", "rsi_extreme_reversion", False),
        _stage3_record("cand-fail-2", "rsi_extreme_reversion", False),
    ]
    path = _write_search_graveyard_entries(records, instrument="EURUSD", timeframe="15m")
    assert path is not None
    assert path == graveyard_path_for("EURUSD", "15m")
    rows = load_graveyard(path)
    ids = {r["candidate_id"] for r in rows}
    assert ids == {"cand-fail-1", "cand-fail-2"}
    assert "cand-pass" not in ids


def test_write_search_graveyard_entries_returns_none_when_everything_passed(tmp_path, monkeypatch):
    monkeypatch.setattr("app.search.graveyard.get_app_base_dir", lambda: tmp_path)
    records = [_stage3_record("cand-pass", "trend_breakout", True)]
    path = _write_search_graveyard_entries(records, instrument="EURUSD", timeframe="15m")
    assert path is None


def test_write_search_graveyard_entries_never_raises_when_logging_fails(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("app.search.graveyard.get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr("app.search.batch_runner.record_rejections", _boom)
    records = [_stage3_record("cand-fail", "trend_breakout", False)]
    path = _write_search_graveyard_entries(records, instrument="EURUSD", timeframe="15m")
    assert path is None  # diagnostic logging failure must never raise out of the search run


def test_write_search_graveyard_entries_shares_the_path_forge_and_evolution_use(tmp_path, monkeypatch):
    """Search Lab's graveyard file must accumulate in the SAME per-instrument
    file Forge Strategy already writes to (app.search.graveyard.graveyard_path_for),
    not a separate one -- otherwise a Search Lab run and a Forge run against
    the same instrument would each re-discover the other's dead ends."""
    monkeypatch.setattr("app.search.graveyard.get_app_base_dir", lambda: tmp_path)
    records = [_stage3_record("cand-fail", "trend_breakout", False)]
    path = _write_search_graveyard_entries(records, instrument="XAUUSD", timeframe="1h")
    assert path == graveyard_path_for("XAUUSD", "1h")
