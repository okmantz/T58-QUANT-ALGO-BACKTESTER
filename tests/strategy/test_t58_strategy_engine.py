import numpy as np
import pandas as pd
import pytest

from app.ai import t58_strategy_engine as t58


def _trending_frame(n=300, start=1.0800, drift=0.00003, noise=0.0004, seed=1):
    """Synthetic H1 frame with a gentle uptrend plus noise -- enough bars
    for EMA200 to be defined and for swing highs/lows to form."""
    rng = np.random.default_rng(seed)
    closes = start + np.cumsum(np.full(n, drift)) + rng.normal(0, noise, n)
    highs = closes + np.abs(rng.normal(0, noise, n))
    lows = closes - np.abs(rng.normal(0, noise, n))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": np.full(n, 100.0),
    })


def _displacement_m15_frame(direction="bullish", n=30):
    """M15 frame whose LAST candle is a clean, big-bodied displacement
    candle closing strongly in `direction`."""
    rng = np.random.default_rng(2)
    base = 1.0800
    closes = base + rng.normal(0, 0.0002, n)
    highs = closes + 0.0003
    lows = closes - 0.0003
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    if direction == "bullish":
        opens[-1], lows[-1], highs[-1], closes[-1] = base, base - 0.0002, base + 0.0035, base + 0.0032
    else:
        opens[-1], lows[-1], highs[-1], closes[-1] = base, base - 0.0035, base + 0.0002, base - 0.0032
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-05", periods=n, freq="15min", tz="UTC"),
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": np.full(n, 50.0),
    })


def test_ema_context_reads_bullish_alignment_from_uptrend():
    frame = _trending_frame()
    snapshot = t58.build_market_snapshot("EURUSD", frame, None, macro_bias="neutral")
    assert snapshot.ema.alignment == "bullish"
    assert snapshot.ema.price_vs_50 == "above"


def test_location_zone_is_premium_or_discount_not_always_extended():
    frame = _trending_frame()
    snapshot = t58.build_market_snapshot("EURUSD", frame, None)
    assert snapshot.location.zone in ("premium", "discount", "equilibrium")
    assert snapshot.location.range_high >= snapshot.location.range_low


def test_direction_without_confirmation_never_reaches_ready():
    """Reproduces the strategy doc's explicit non-negotiable example:
    macro/HTF/EMA aligned but no sweep/confirmation must stay WAIT, never
    'long now'."""
    frame = _trending_frame()
    snapshot = t58.build_market_snapshot("EURUSD", frame, m15_frame=None, macro_bias="bullish")
    assessment = t58.assess(snapshot)
    assert assessment.status in ("WAIT", "DEVELOPING", "EXTENDED", "PASS")
    assert assessment.status != "READY"


def test_neutral_macro_never_produces_a_directional_ready_or_ban_on_direction():
    frame = _trending_frame()
    snapshot = t58.build_market_snapshot("EURUSD", frame, None, macro_bias="neutral")
    assessment = t58.assess(snapshot)
    assert assessment.direction == "none"
    assert assessment.status == "PASS"


def test_m15_confirmation_detects_bullish_displacement():
    m15 = _displacement_m15_frame("bullish")
    ctx = t58._m15_confirmation(m15)
    assert ctx.displacement is True
    assert ctx.direction == "bullish"


def test_m15_confirmation_is_none_without_data():
    ctx = t58._m15_confirmation(None)
    assert ctx.displacement is False
    assert ctx.direction == "none"


def test_score_is_bounded_0_to_100():
    frame = _trending_frame()
    for bias in ("bullish", "bearish", "neutral"):
        snapshot = t58.build_market_snapshot("EURUSD", frame, None, macro_bias=bias)
        assessment = t58.assess(snapshot)
        assert 0 <= assessment.score <= 100


def test_high_impact_news_never_increases_score():
    frame = _trending_frame()
    calm = t58.assess(t58.build_market_snapshot("EURUSD", frame, None, macro_bias="bullish", news_risk="none"))
    risky = t58.assess(t58.build_market_snapshot("EURUSD", frame, None, macro_bias="bullish", news_risk="high"))
    assert risky.score <= calm.score
    assert risky.status != "READY"
