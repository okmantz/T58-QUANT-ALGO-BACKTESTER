"""Macro (higher-timeframe) filter: causal, only removes entries, works through run_backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy.base import StrategyResult
from app.strategy.macro_filter import apply_macro_filter, macro_bias


def _bars(n=6000, seed=5):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": 1.0})


DF = _bars()


def _result(df):
    sig = np.zeros(len(df), int)
    sig[100::40] = 1
    sig[120::40] = -1
    return StrategyResult(signals=pd.Series(sig, index=df.index), name="t", source_type="python")


@pytest.mark.parametrize("mode", ["ema", "ema_slope", "supertrend", "donchian", "adx_ema", "macd", "rsi"])
def test_modes_run_and_only_remove(mode):
    r = _result(DF)
    out, note = apply_macro_filter(r, DF, "1h", mode, 20)
    assert (out.signals != 0).sum() <= (r.signals != 0).sum() + 0
    assert "Macro filter" in note


def test_no_lookahead():
    """Changing bars after t must not change the bias at or before t."""
    b1 = macro_bias(DF, "1h", "ema", 20)
    df2 = DF.copy()
    df2.loc[4000:, "close"] += 50.0
    df2.loc[4000:, ["open", "high", "low"]] += 50.0
    b2 = macro_bias(df2, "1h", "ema", 20)
    assert np.array_equal(b1[:3990], b2[:3990])


def test_macro_must_be_coarser():
    with pytest.raises(ValueError):
        macro_bias(DF, "5m", "ema", 20)


def test_strict_blocks_at_least_as_much():
    r = _result(DF)
    loose, _ = apply_macro_filter(r, DF, "1h", "ema", 20, strict=False)
    strict, _ = apply_macro_filter(r, DF, "1h", "ema", 20, strict=True)
    assert (strict.signals != 0).sum() <= (loose.signals != 0).sum()


def test_regime_filter_only_removes():
    from app.strategy.macro_filter import apply_regime_filter
    r = _result(DF)
    for mode in ("vol_normal_or_high", "vol_high", "vol_low", "trending", "ranging"):
        out, note = apply_regime_filter(r, DF, mode)
        assert (out.signals != 0).sum() <= (r.signals != 0).sum()
        assert "Regime filter" in note
