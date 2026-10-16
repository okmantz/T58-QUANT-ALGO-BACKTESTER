"""w10-astra tests: regime-oscillator + VWAP-profile ports and the B3 search
pre-screen.

Covers the task's mandatory cases:
* RSI zone values in each regime (synthetic trending data),
* divergence detector causality (no lookahead -- truncation invariance),
* VWAP / POC sanity,
* pre-screen skip (unfundable candidate skipped with a counted reason),
plus Manual-builder dispatch wiring for the new operand kinds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.quant_lab import regime_oscillators as ro
from app.quant_lab import vwap_profile as vp
from app.search.batch_runner import _prescreen_funding
from app.strategy.manual import ManualStrategy


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ohlc(n: int, close: np.ndarray, start: str = "2024-01-02 00:00", freq: str = "15min",
          volume: float = 100.0) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq=freq)
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, volume),
    })


def _uptrend(n: int = 200, pullback_at: int = 150) -> pd.DataFrame:
    """Steady climb with real oscillation (so the fractal-swing detector
    finds HH/HL structure), then a moderate pullback -- RSI runs hot,
    then dips into the bull pullback-buy zone."""
    pullback_at = min(pullback_at, max(n - 30, 10))
    i = np.arange(n, dtype=float)
    # climb with oscillation; pullback gives back ~40% of the climb
    trend = np.where(i < pullback_at, 0.4 * i,
                     0.4 * pullback_at - 0.4 * 0.4 * pullback_at * np.minimum((i - pullback_at) / 20.0, 1.0))
    close = 100.0 + trend + 1.5 * np.sin(i / 3.0)
    return _ohlc(n, close)


# ---------------------------------------------------------------------------
# 1. RSI zone values in each regime
# ---------------------------------------------------------------------------

def test_classify_rsi_zone_scalar_bull_bands():
    assert ro.classify_rsi_zone_scalar(45.0, "bull") == ro.BULL_PULLBACK_BUY
    assert ro.classify_rsi_zone_scalar(38.0, "bull") == ro.BULL_PULLBACK_BUY
    assert ro.classify_rsi_zone_scalar(60.0, "bull") == ro.BULL_HEALTHY
    assert ro.classify_rsi_zone_scalar(80.0, "bull") == ro.OVERBOUGHT_NO_CHASE
    assert ro.classify_rsi_zone_scalar(20.0, "bull") == ro.NEUTRAL


def test_classify_rsi_zone_scalar_bear_bands():
    assert ro.classify_rsi_zone_scalar(55.0, "bear") == ro.BEAR_RALLY_SELL
    assert ro.classify_rsi_zone_scalar(35.0, "bear") == ro.BEAR_HEALTHY
    assert ro.classify_rsi_zone_scalar(20.0, "bear") == ro.OVERSOLD_NO_CHASE
    assert ro.classify_rsi_zone_scalar(70.0, "bear") == ro.NEUTRAL


def test_classify_rsi_zone_scalar_range_bands():
    assert ro.classify_rsi_zone_scalar(25.0, "range") == ro.RANGE_OVERSOLD
    assert ro.classify_rsi_zone_scalar(75.0, "range") == ro.RANGE_OVERBOUGHT
    assert ro.classify_rsi_zone_scalar(50.0, "range") == ro.NEUTRAL


def test_classify_rsi_zone_scalar_accepts_t58_trend_vocabulary():
    # The existing detector emits up/down/range -- all must map.
    assert ro.classify_rsi_zone_scalar(45.0, "up") == ro.BULL_PULLBACK_BUY
    assert ro.classify_rsi_zone_scalar(55.0, "down") == ro.BEAR_RALLY_SELL
    assert ro.classify_rsi_zone_scalar(25.0, "range") == ro.RANGE_OVERSOLD


def test_classify_rsi_zone_scalar_missing_rsi_is_insufficient():
    assert ro.classify_rsi_zone_scalar(None, "bull") == ro.INSUFFICIENT_DATA
    assert ro.classify_rsi_zone_scalar(float("nan"), "bear") == ro.INSUFFICIENT_DATA


def test_zone_bounds_are_tunable():
    bounds = dict(ro.DEFAULT_ZONE_BOUNDS)
    bounds["bull_pullback"] = (30.0, 40.0)
    assert ro.classify_rsi_zone_scalar(45.0, "bull", bounds) == ro.NEUTRAL
    assert ro.classify_rsi_zone_scalar(35.0, "bull", bounds) == ro.BULL_PULLBACK_BUY


def test_trend_regime_series_detects_uptrend():
    df = _uptrend(n=200)
    trend = ro.trend_regime_series(df, left=5, right=5)
    assert len(trend) == len(df)
    # After warmup + confirmation lag, the steady climb must read bull
    # (the pullback section later legitimately flips to bear).
    assert (trend.iloc[60:140] == "bull").all()


def test_rsi_zone_series_flags_bull_pullback_on_synthetic_trend():
    df = _uptrend(n=200, pullback_at=150)
    trend = pd.Series("bull", index=df.index)  # pin the regime; test the zones
    zones = ro.rsi_zone_series(df, rsi_period=14, trend=trend)
    assert len(zones) == len(df)
    # The hot climb parks RSI high; the pullback must dip into the buy zone.
    assert (zones == ro.BULL_PULLBACK_BUY).any()
    assert (zones == ro.OVERBOUGHT_NO_CHASE).any() or (zones == ro.BULL_HEALTHY).any()
    buy_flags = ro.regime_flag_series(df, "rsi_zone_buy", trend=trend)
    assert set(buy_flags.unique()) <= {0.0, 1.0}
    assert (buy_flags == 1.0).any()
    # Buy flags must agree with the zone labels, bar for bar.
    assert ((buy_flags == 1.0) == (zones == ro.BULL_PULLBACK_BUY)).all()


def test_rsi_zone_series_flags_bear_rally_on_synthetic_downtrend():
    n = 200
    close = 200.0 - 0.4 * np.arange(150)
    bounce = np.linspace(0, 12.0, 20)  # sharp rally into the sell zone
    close = np.concatenate([close, close[-1] + bounce, np.full(n - 170, close[-1] + 12.0)])
    df = _ohlc(n, close)
    trend = pd.Series("bear", index=df.index)
    zones = ro.rsi_zone_series(df, rsi_period=14, trend=trend)
    assert (zones == ro.BEAR_RALLY_SELL).any()
    sell_flags = ro.regime_flag_series(df, "rsi_zone_sell", trend=trend)
    assert ((sell_flags == 1.0) == (zones == ro.BEAR_RALLY_SELL)).all()


def test_rsi_regime_codes():
    df = _uptrend(n=200)
    codes = ro.regime_flag_series(df, "rsi_regime")
    assert set(codes.unique()) <= {-1.0, 0.0, 1.0}
    assert (codes.iloc[60:140] == 1.0).all()


# ---------------------------------------------------------------------------
# 2. Divergence detector -- algorithm + causality
# ---------------------------------------------------------------------------

def test_divergence_detects_bullish_case():
    # Price: lower low. Oscillator: higher low -> bullish.
    closes = np.array([10., 9., 8., 9., 10., 9., 7.5, 8.5, 9., 10.])
    osc = np.array([50., 45., 40., 44., 48., 46., 43., 46., 48., 50.])
    out = ro.divergence_signals(closes, osc, lookback=10)
    assert out[-1] == 1.0


def test_divergence_detects_bearish_case():
    # Price: higher high. Oscillator: lower high -> bearish.
    closes = np.array([10., 11., 12., 11., 10., 11., 12.5, 11.5, 11., 10.5])
    osc = np.array([50., 55., 60., 56., 52., 54., 57., 54., 52., 50.])
    out = ro.divergence_signals(closes, osc, lookback=10)
    assert out[-1] == -1.0


def test_divergence_is_none_without_enough_history_or_signal():
    closes = np.linspace(100, 110, 20)
    osc = np.linspace(50, 55, 20)
    out = ro.divergence_signals(closes, osc, lookback=20)
    assert (out[:19] == 0.0).all()  # insufficient history -> NONE
    assert out[19] == 0.0  # steady grind, no divergence


def test_divergence_matches_scalar_reference():
    # Cross-check the vectorized port against a direct transcription of the
    # astra scalar algorithm on random data.
    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0, 1, 300))
    osc = 50 + np.cumsum(rng.normal(0, 2, 300))
    lookback, eps = 20, 1e-9
    half = lookback // 2

    def scalar(px, oc):
        n = lookback
        fpx, spx = px[:half], px[half:]
        foc, soc = oc[:half], oc[half:]
        i1 = int(np.argmin(fpx))
        i2 = half + int(np.argmin(spx))
        if spx[i2 - half] < fpx[i1] - eps and soc[i2 - half] > foc[i1] + eps:
            return 1.0
        j1 = int(np.argmax(fpx))
        j2 = half + int(np.argmax(spx))
        if spx[j2 - half] > fpx[j1] + eps and soc[j2 - half] < foc[j1] - eps:
            return -1.0
        return 0.0

    vec = ro.divergence_signals(closes, osc, lookback=lookback)
    for i in range(lookback - 1, 300, 17):
        assert vec[i] == scalar(closes[i - lookback + 1: i + 1], osc[i - lookback + 1: i + 1]), i


def test_divergence_is_causal_truncation_invariance():
    # The behavioral lookahead contract: bar i's signal must be identical
    # whether computed on the full frame or on the frame truncated at i.
    rng = np.random.default_rng(42)
    n = 400
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    df = _ohlc(n, close, freq="1h")
    full = ro.rsi_divergence_series(df, rsi_period=14, lookback=20)
    for cut in (100, 199, 250, 399):
        trunc = ro.rsi_divergence_series(df.iloc[: cut + 1], rsi_period=14, lookback=20)
        assert (trunc.values == full.iloc[: cut + 1].values).all(), f"cut={cut}"


def test_rsi_divergence_series_values_are_tristate():
    df = _uptrend(n=120)
    s = ro.rsi_divergence_series(df)
    assert set(s.unique()) <= {-1.0, 0.0, 1.0}
    assert len(s) == len(df)


# ---------------------------------------------------------------------------
# 3. VWAP profile sanity
# ---------------------------------------------------------------------------

def test_session_keys_roll_at_1700_ct():
    # 2024-01-02 21:00 UTC = 15:00 CT (session A); 23:30 UTC = 17:30 CT (session B).
    ts = pd.to_datetime(["2024-01-02 21:00", "2024-01-02 21:30", "2024-01-02 23:30", "2024-01-03 00:00"])
    keys = vp.session_keys(pd.Series(ts))
    assert keys.iloc[0] == keys.iloc[1]
    assert keys.iloc[2] == keys.iloc[3]
    assert keys.iloc[1] != keys.iloc[2]
    # 18:00 CT and next-morning 10:00 CT belong to the SAME session.
    ts2 = pd.to_datetime(["2024-01-03 00:00", "2024-01-03 16:00"])  # 18:00 CT, 10:00 CT
    keys2 = vp.session_keys(pd.Series(ts2))
    assert keys2.iloc[0] == keys2.iloc[1]


def test_session_vwap_matches_hand_computation():
    n = 60
    ts = pd.date_range("2024-01-03 00:00", periods=n, freq="15min")  # one CT session
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    df = pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 0.4, "low": close - 0.4,
        "close": close, "volume": rng.uniform(50, 150, n),
    })
    prof = vp.session_vwap_profile(df)
    typ = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]
    for i in (10, 30, 59):
        w = slice(0, i + 1)
        expect_vwap = (typ[w] * vol[w]).sum() / vol[w].sum()
        assert prof["vwap"].iloc[i] == pytest.approx(expect_vwap)
        var = ((typ[w] ** 2 * vol[w]).sum() / vol[w].sum()) - expect_vwap ** 2
        expect_sig = max(var, 0.0) ** 0.5
        assert prof["sigma"].iloc[i] == pytest.approx(expect_sig)
        assert prof["vah"].iloc[i] == pytest.approx(expect_vwap + expect_sig)
        assert prof["val"].iloc[i] == pytest.approx(expect_vwap - expect_sig)
        assert prof["upper_2"].iloc[i] == pytest.approx(expect_vwap + 2 * expect_sig)
        assert prof["lower_2"].iloc[i] == pytest.approx(expect_vwap - 2 * expect_sig)


def test_session_vwap_resets_at_roll():
    # Flat price in session A, different flat price in session B: VWAP must
    # jump to the new session's level instead of blending. Hourly bars from
    # 2024-01-02 18:00 UTC (= 12:00 CT); the 17:00 CT roll lands on index 5.
    ts = pd.date_range("2024-01-02 18:00", periods=16, freq="1h")
    close = np.array([100.0] * 5 + [200.0] * 11)
    df = pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 0.1, "low": close - 0.1,
        "close": close, "volume": np.full(16, 10.0),
    })
    prof = vp.session_vwap_profile(df)
    keys = vp.session_keys(df["timestamp"])
    roll_bar = int(np.flatnonzero(keys.values[1:] != keys.values[:-1])[0] + 1)
    assert roll_bar == 5
    assert prof["vwap"].iloc[roll_bar - 1] == pytest.approx(100.0, abs=0.2)
    assert prof["vwap"].iloc[roll_bar] == pytest.approx(200.0, abs=0.2)


def test_poc_within_session_range_and_tracks_volume():
    n = 120
    # 5min bars starting 00:00 CT: 10 hours, one single session.
    ts = pd.date_range("2024-01-03 06:00", periods=n, freq="5min")
    # Two humps: heavy volume low, light volume high -> POC should sit low.
    close = np.concatenate([np.full(60, 100.0), np.full(60, 110.0)])
    volume = np.concatenate([np.full(60, 1000.0), np.full(60, 10.0)])
    df = pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": volume,
    })
    assert vp.session_keys(df["timestamp"]).nunique() == 1  # guard: one session
    prof = vp.session_vwap_profile(df, poc_buckets=30)
    poc = prof["poc"].dropna()
    assert len(poc) == n
    assert ((poc >= 99.0) & (poc <= 111.0)).all()
    # With 100x the volume at 100, the developing POC must settle near 100.
    assert poc.iloc[-1] == pytest.approx(100.0, abs=2.0)


def test_vwap_zscore_and_position_flags():
    df = _uptrend(n=120)
    prof = vp.session_vwap_profile(df)
    z = prof["zscore"]
    assert ((z == (df["close"] - prof["vwap"]) / prof["sigma"]) | (z.isna())).all()
    above = vp.vwap_operand_series(df, "vwap_above", profile=prof)
    below = vp.vwap_operand_series(df, "vwap_below", profile=prof)
    assert ((above == 1.0) == (df["close"] > prof["vwap"])).all()
    assert ((below == 1.0) == (df["close"] < prof["vwap"])).all()
    outside = vp.vwap_operand_series(df, "vwap_outside_value_area", profile=prof)
    assert set(outside.unique()) <= {0.0, 1.0}


def test_vwap_profile_is_causal_truncation_invariance():
    rng = np.random.default_rng(11)
    n = 300
    ts = pd.date_range("2024-01-03 00:00", periods=n, freq="15min")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 0.4, "low": close - 0.4,
        "close": close, "volume": rng.uniform(50, 150, n),
    })
    full = vp.session_vwap_profile(df)
    for cut in (99, 199, 299):
        trunc = vp.session_vwap_profile(df.iloc[: cut + 1])
        for col in ("vwap", "vah", "val", "poc"):
            a = full[col].iloc[: cut + 1].to_numpy()
            b = trunc[col].to_numpy()
            assert np.allclose(a, b, equal_nan=True), (col, cut)


# ---------------------------------------------------------------------------
# 4. Manual-builder dispatch wiring
# ---------------------------------------------------------------------------

def _manual_cfg_with_operand(operand: dict) -> dict:
    return {
        "name": "w10 dispatch test",
        "entry_conditions": {
            "long": [{"left": operand, "operator": "is true", "right": {"type": "value", "value": 1}}],
        },
        "exit_conditions": {"long": [], "short": []},
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }


def test_manual_dispatches_rsi_zone_buy():
    df = _uptrend(n=200, pullback_at=150)
    cfg = _manual_cfg_with_operand({"type": "rsi_zone_buy", "rsi_period": 14})
    result = ManualStrategy(cfg).generate(df)
    assert len(result.signals) == len(df)
    # Some bar must actually trigger the buy-zone condition.
    assert (result.signals != 0).any()


def test_manual_dispatches_rsi_divergence_and_regime():
    df = _uptrend(n=200)
    for kind in ("rsi_divergence", "rsi_regime", "rsi_zone_sell"):
        cfg = _manual_cfg_with_operand({"type": kind})
        result = ManualStrategy(cfg).generate(df)
        assert len(result.signals) == len(df)


def test_manual_dispatches_vwap_kinds():
    df = _uptrend(n=200)
    for kind in ("vwap_above", "vwap_below", "vwap_outside_value_area"):
        cfg = _manual_cfg_with_operand({"type": kind, "roll_hour": 17})
        result = ManualStrategy(cfg).generate(df)
        assert len(result.signals) == len(df)
    # Numeric kinds compare against the vwap level itself.
    df2 = _uptrend(n=200)
    cfg = {
        "name": "vwap cross test",
        "entry_conditions": {
            "long": [{"left": {"type": "close"}, "operator": ">",
                      "right": {"type": "session_vwap", "roll_hour": 17}}],
        },
        "exit_conditions": {"long": [], "short": []},
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }
    result = ManualStrategy(cfg).generate(df2)
    assert len(result.signals) == len(df2)


def test_manual_operand_tunable_params_flow_through():
    df = _uptrend(n=200, pullback_at=150)
    cfg = _manual_cfg_with_operand({
        "type": "rsi_zone_buy", "rsi_period": 14, "div_lookback": 20,
        "zone_bounds": {"bull_pullback": (30.0, 40.0)},
    })
    result = ManualStrategy(cfg).generate(df)
    assert len(result.signals) == len(df)


# ---------------------------------------------------------------------------
# 5. B3 search pre-screen
# ---------------------------------------------------------------------------

def _spec_with_fixed_stop(stop_pips: float) -> dict:
    return {
        "source_type": "manual",
        "config": {
            "name": "prescreen test",
            "risk_management": {"stop_type": "fixed", "stop_value": stop_pips},
        },
    }


def _risk_for_prescreen(risk_value: float = 0.25) -> RiskConfig:
    # MGC-micro-like: pip_size=1.0 price point, contract_size=5 $/point.
    return RiskConfig(initial_balance=50_000.0, risk_value=risk_value,
                      pip_size=1.0, contract_size=5.0)


def test_prescreen_skips_unfundable_candidate_with_reason():
    df = _ohlc(50, 100 + np.arange(50) * 0.1)
    risk = _risk_for_prescreen(risk_value=0.25)  # $125 risk vs 200pt stop = 0.625 units < 5
    base = {"candidate_id": "c1"}
    rec = _prescreen_funding(base, _spec_with_fixed_stop(200.0), risk, df)
    assert rec is not None
    assert rec["passed_stage1"] is False
    assert rec["prescreen_skipped"] is True
    assert rec["prescreen_skip_reason"] == "unfundable"
    assert "cannot afford" in rec["prescreen_detail"]
    assert rec["candidate_id"] == "c1"


def test_prescreen_passes_fundable_candidate():
    df = _ohlc(50, 100 + np.arange(50) * 0.1)
    risk = _risk_for_prescreen(risk_value=2.0)  # $1000 risk vs 200pt stop = 5 units >= 5
    rec = _prescreen_funding({"candidate_id": "c2"}, _spec_with_fixed_stop(200.0), risk, df)
    assert rec is None


def test_prescreen_passes_through_when_no_contract_size():
    df = _ohlc(50, 100 + np.arange(50) * 0.1)
    risk = RiskConfig(initial_balance=50_000.0, risk_value=0.25, pip_size=1.0, contract_size=None)
    rec = _prescreen_funding({"candidate_id": "c3"}, _spec_with_fixed_stop(200.0), risk, df)
    assert rec is None  # no whole-contract flooring in play


def test_prescreen_handles_atr_stops_via_median_atr():
    df = _ohlc(200, 100 + np.cumsum(np.random.default_rng(5).normal(0, 0.5, 200)))
    risk = _risk_for_prescreen(risk_value=0.25)
    spec = {
        "source_type": "manual",
        "config": {
            "name": "atr stop test",
            "risk_management": {"stop_type": "atr", "stop_value": 50.0, "stop_atr_period": 14},
        },
    }
    rec = _prescreen_funding({"candidate_id": "c4"}, spec, risk, df)
    assert rec is not None  # 50x ATR median at 0.25% risk cannot fund 1 contract
    assert rec["prescreen_skip_reason"] == "unfundable"
    # ...but a wide risk budget passes the same stop.
    rich = _risk_for_prescreen(risk_value=50.0)
    assert _prescreen_funding({"candidate_id": "c4"}, spec, rich, df) is None


def test_prescreen_passes_through_non_manual_and_stop_free_specs():
    df = _ohlc(50, 100 + np.arange(50) * 0.1)
    risk = _risk_for_prescreen(risk_value=0.25)
    assert _prescreen_funding({"candidate_id": "c5"},
                              {"source_type": "python", "code_text": "x=1"}, risk, df) is None
    assert _prescreen_funding({"candidate_id": "c6"},
                              {"source_type": "manual", "config": {"name": "no rm"}}, risk, df) is None


def test_prescreen_mirrors_engine_flooring_exactly():
    # Property: the pre-screen skips exactly when RiskConfig.position_size()
    # (the engine's own sizing) would floor this candidate's median stop to
    # zero contracts.
    rng = np.random.default_rng(9)
    df = _ohlc(100, 100 + np.cumsum(rng.normal(0, 0.5, 100)))
    for risk_value, stop_pips in [(0.1, 50.0), (0.25, 200.0), (1.0, 100.0), (2.0, 400.0), (5.0, 150.0)]:
        risk = _risk_for_prescreen(risk_value=risk_value)
        rec = _prescreen_funding({"candidate_id": "cx"}, _spec_with_fixed_stop(stop_pips), risk, df)
        engine_units = risk.position_size(risk.initial_balance, stop_pips)
        engine_floored = engine_units < risk.contract_size - 1e-9
        assert (rec is not None) == engine_floored, (risk_value, stop_pips, engine_units)


def test_prescreen_fires_inside_stage1_task(monkeypatch):
    # End-to-end through _stage1_task (the scalar Stage 1 path): an
    # unfundable candidate returns the skip record without running a
    # backtest at all.
    import app.search.batch_runner as br
    df = _ohlc(200, 100 + np.cumsum(np.random.default_rng(12).normal(0, 0.5, 200)))
    risk = _risk_for_prescreen(risk_value=0.25)
    monkeypatch.setitem(br._WORKER, "df", df)
    monkeypatch.setitem(br._WORKER, "risk", risk)
    from app.prop.simulator import PropRules
    monkeypatch.setitem(br._WORKER, "prop_rules", PropRules())
    monkeypatch.setitem(br._WORKER, "tmp_dir", None)

    ran = []
    monkeypatch.setattr(br, "build_strategy_from_spec",
                        lambda spec, tmp_dir: ran.append(spec) or (_ for _ in ()).throw(AssertionError("must not build")))
    rec = br._stage1_task("unfundable-1", _spec_with_fixed_stop(200.0), {})
    assert rec["prescreen_skipped"] is True
    assert rec["prescreen_skip_reason"] == "unfundable"
    assert rec["passed_stage1"] is False
    assert ran == []  # the backtest path was never reached
