"""Tests for the Expansion round 8 indicators added to
app.strategy.indicators (HMA/DEMA/TEMA/KAMA, Vortex, Elder Ray, TTM
Squeeze, VWMA, Accumulation/Distribution + Chaikin Oscillator, Fisher
Transform, Connors RSI, Average Daily Range) and their dispatch through
build_indicator_series / app.strategy.manual's operand whitelist."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy.indicators import (
    accumulation_distribution,
    average_daily_range,
    build_indicator_series,
    chaikin_oscillator,
    connors_rsi,
    dema,
    elder_ray,
    fisher_transform,
    hma,
    kama,
    tema,
    ttm_squeeze,
    vortex,
    vwma,
)
from app.strategy.manual import ManualStrategy


def _trending_df(n=250, seed=1, drift=0.05):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="15min")
    close = 100 + np.cumsum(np.full(n, drift) + rng.normal(0, 0.1, n))
    high = close + np.abs(rng.normal(0, 0.1, n))
    low = close - np.abs(rng.normal(0, 0.1, n))
    openp = close + rng.normal(0, 0.02, n)
    volume = rng.uniform(50, 150, n)
    return pd.DataFrame({
        "timestamp": ts, "open": openp, "high": high, "low": low, "close": close, "volume": volume,
    })


def _flat_df(n=100, price=100.0):
    ts = pd.date_range("2026-01-01", periods=n, freq="15min")
    close = pd.Series([price] * n)
    return pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 0.01, "low": close - 0.01,
        "close": close, "volume": pd.Series([10.0] * n),
    })


# ---------------------------------------------------------------------------
# Lower-lag moving averages
# ---------------------------------------------------------------------------

def test_hma_tracks_a_constant_series_at_the_constant_value():
    df = _flat_df()
    out = hma(df["close"], period=20)
    assert np.allclose(out.dropna().to_numpy(), 100.0)


def test_hma_has_less_lag_than_a_plain_wma_on_a_trend():
    from app.strategy.indicators import wma
    df = _trending_df()
    h = hma(df["close"], period=20)
    w = wma(df["close"], 20)
    valid = h.notna() & w.notna()
    hma_gap = (df["close"][valid] - h[valid]).abs().mean()
    wma_gap = (df["close"][valid] - w[valid]).abs().mean()
    assert hma_gap < wma_gap


def test_dema_and_tema_reduce_lag_versus_plain_ema_on_a_trend():
    from app.strategy.indicators import ema
    df = _trending_df()
    e = ema(df["close"], 20)
    d = dema(df["close"], 20)
    t = tema(df["close"], 20)
    valid = e.notna() & d.notna() & t.notna()
    ema_gap = (df["close"][valid] - e[valid]).abs().mean()
    dema_gap = (df["close"][valid] - d[valid]).abs().mean()
    tema_gap = (df["close"][valid] - t[valid]).abs().mean()
    assert dema_gap < ema_gap
    assert tema_gap < ema_gap


def test_kama_is_flat_on_pure_chop_and_moves_on_a_clean_trend():
    chop = _flat_df(n=80)
    chop["close"] = 100 + np.tile([0.1, -0.1], 40)
    trend = _trending_df(n=80, drift=0.2)
    k_chop = kama(chop["close"], period=10)
    k_trend = kama(trend["close"], period=10)
    chop_move = (k_chop.dropna().iloc[-1] - k_chop.dropna().iloc[0])
    trend_move = (k_trend.dropna().iloc[-1] - k_trend.dropna().iloc[0])
    assert abs(chop_move) < abs(trend_move)


# ---------------------------------------------------------------------------
# Vortex / Elder Ray
# ---------------------------------------------------------------------------

def test_vortex_plus_exceeds_vortex_minus_in_a_clean_uptrend():
    df = _trending_df(drift=0.3)
    vi_plus, vi_minus = vortex(df, period=14)
    assert vi_plus.dropna().mean() > vi_minus.dropna().mean()


def test_elder_ray_bull_power_is_positive_on_average_in_a_clean_uptrend():
    df = _trending_df(drift=0.3)
    bull, bear = elder_ray(df, period=13)
    assert bull.dropna().mean() > 0


# ---------------------------------------------------------------------------
# TTM Squeeze
# ---------------------------------------------------------------------------

def test_ttm_squeeze_flags_on_during_a_genuinely_quiet_period():
    n = 100
    ts = pd.date_range("2026-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(5)
    close = 100 + rng.normal(0, 0.01, n)
    df = pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 0.005, "low": close - 0.005,
        "close": close, "volume": pd.Series([10.0] * n),
    })
    squeeze_on, momentum = ttm_squeeze(df, bb_period=20, kc_period=20, momentum_period=12)
    assert squeeze_on.iloc[30:].mean() > 0.5
    assert len(momentum) == n


def test_ttm_squeeze_on_is_boolean_and_momentum_is_finite_where_defined():
    df = _trending_df()
    squeeze_on, momentum = ttm_squeeze(df)
    assert squeeze_on.dtype == bool
    assert np.isfinite(momentum.dropna().to_numpy()).all()


# ---------------------------------------------------------------------------
# VWMA / ADL / Chaikin Oscillator
# ---------------------------------------------------------------------------

def test_vwma_equals_sma_when_volume_is_constant():
    from app.strategy.indicators import sma
    df = _trending_df()
    df["volume"] = 10.0
    v = vwma(df, period=20)
    s = sma(df["close"], 20)
    assert np.allclose(v.dropna().to_numpy(), s.dropna().to_numpy())


def test_vwma_falls_back_to_plain_price_weighting_with_no_volume_column():
    from app.strategy.indicators import sma
    df = _trending_df().drop(columns=["volume"])
    v = vwma(df, period=20)
    s = sma(df["close"], 20)
    assert np.allclose(v.dropna().to_numpy(), s.dropna().to_numpy())


def test_accumulation_distribution_rises_when_closes_land_near_the_bar_high():
    n = 50
    ts = pd.date_range("2026-01-01", periods=n, freq="15min")
    high = pd.Series([101.0] * n)
    low = pd.Series([99.0] * n)
    close = pd.Series([100.9] * n)
    volume = pd.Series([10.0] * n)
    df = pd.DataFrame({"timestamp": ts, "open": close, "high": high, "low": low, "close": close, "volume": volume})
    adl = accumulation_distribution(df)
    assert adl.iloc[-1] > adl.iloc[5]


def test_chaikin_oscillator_is_positive_when_adl_is_accelerating_upward():
    n = 60
    ts = pd.date_range("2026-01-01", periods=n, freq="15min")
    high = pd.Series(np.linspace(101, 111, n))
    low = pd.Series(np.linspace(99, 109, n))
    close = pd.Series(np.linspace(100.9, 110.9, n))
    volume = pd.Series(np.linspace(10, 50, n))
    df = pd.DataFrame({"timestamp": ts, "open": close, "high": high, "low": low, "close": close, "volume": volume})
    osc = chaikin_oscillator(df)
    assert osc.dropna().iloc[-1] > 0


# ---------------------------------------------------------------------------
# Fisher Transform / Connors RSI / ADR
# ---------------------------------------------------------------------------

def test_fisher_transform_swings_at_extremes_and_signal_is_prior_bar():
    n = 60
    ts = pd.date_range("2026-01-01", periods=n, freq="15min")
    close = pd.Series(100 + 2 * np.sin(np.linspace(0, 6 * np.pi, n)))
    df = pd.DataFrame({"timestamp": ts, "open": close, "high": close + 0.1, "low": close - 0.1,
                        "close": close, "volume": pd.Series([10.0] * n)})
    fisher, signal = fisher_transform(df, period=10)
    assert len(fisher) == n
    assert fisher.abs().max() > 0.5
    assert (signal.iloc[1:].to_numpy() == fisher.shift(1).iloc[1:].to_numpy()).all()


def test_connors_rsi_is_low_after_a_losing_streak_and_high_after_a_winning_one():
    n = 120
    ts = pd.date_range("2026-01-01", periods=n, freq="15min")
    down = 100 - np.arange(60) * 0.2
    up = down[-1] + np.arange(60) * 0.2
    close = pd.Series(np.concatenate([down, up]))
    df = pd.DataFrame({"timestamp": ts, "open": close, "high": close + 0.05, "low": close - 0.05,
                        "close": close, "volume": pd.Series([10.0] * n)})
    crsi = connors_rsi(df, rank_period=50)
    assert crsi.iloc[59] < 40
    assert crsi.iloc[-1] > 60


def test_average_daily_range_is_constant_within_a_day():
    ts = pd.date_range("2026-01-01", periods=4 * 24, freq="1h")
    rng = np.random.default_rng(9)
    high = pd.Series(100 + rng.uniform(0, 2, len(ts)))
    low = high - rng.uniform(0.5, 1.5, len(ts))
    close = (high + low) / 2
    df = pd.DataFrame({"timestamp": ts, "open": close, "high": high, "low": low, "close": close,
                        "volume": pd.Series([10.0] * len(ts))})
    adr = average_daily_range(df, period=2)
    day3_mask = pd.to_datetime(df["timestamp"]).dt.normalize() == pd.Timestamp("2026-01-03")
    assert adr[day3_mask].nunique() == 1


# ---------------------------------------------------------------------------
# Dispatch registry (build_indicator_series) + manual.py operand wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", [
    "hma", "dema", "tema", "kama",
    "vortex_plus", "vortex_minus",
    "elder_bull_power", "elder_bear_power",
    "ttm_squeeze_on", "ttm_squeeze_momentum",
    "vwma", "adl", "chaikin_oscillator",
    "fisher_transform", "fisher_transform_signal",
    "connors_rsi", "adr",
])
def test_build_indicator_series_dispatches_every_round_8_kind(kind):
    df = _trending_df()
    series = build_indicator_series(df, kind, period=14)
    assert len(series) == len(df)


@pytest.mark.parametrize("kind", [
    "hma", "vortex_plus", "elder_bull_power", "ttm_squeeze_on",
    "vwma", "chaikin_oscillator", "fisher_transform", "connors_rsi", "adr",
])
def test_manual_strategy_accepts_every_round_8_operand_kind(kind):
    df = _trending_df()
    cfg = {
        "name": "round 8 operand smoke test",
        "entry_conditions": {
            "long": [{"left": {"type": kind, "period": 14}, "operator": ">", "right": {"type": "value", "value": -1e9}}],
        },
        "exit_conditions": {"long": [], "short": []},
        "stop_loss_pips": 20, "take_profit_pips": 40,
    }
    strategy = ManualStrategy(cfg)
    result = strategy.generate(df)
    assert len(result.signals) == len(df)
