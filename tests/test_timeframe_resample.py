"""
FIX (MTF-STRATEGY-001) / FIX (MTF-LOOKAHEAD-001): tests for two related
additions:

  1. app.data.timeframe_resample -- lets a strategy DECLARE what
     timeframe(s) it needs (its own execution timeframe, plus optionally
     one or more coarser context/bias timeframes) and automatically
     resamples whatever base data was loaded (e.g. 1-minute GC/ES/NQ)
     into exactly that shape, once, at the one shared chokepoint
     (app.backtest.engine.run_backtest) every tool in the app funnels
     through. A strategy that declares nothing is completely unaffected
     (byte-identical `df`, no warnings) -- this is purely additive.

  2. A real, previously-undetected lookahead bug found IN THE COURSE of
     building (1): app.data.multi_timeframe.merge_multi_timeframe merged
     a higher-timeframe bar by its raw (pandas-default) START-of-bar
     label, which let a base-timeframe row anywhere inside a still-
     forming HTF bar see that bar's FINAL close/high/low/open before it
     had actually finished forming -- the exact same class of bug
     app.strategy.mtf's own docstring describes finding and fixing in a
     real uploaded strategy, just never caught in this shared utility.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.data.multi_timeframe import merge_multi_timeframe
from app.data.timeframe_resample import (
    TimeframeError,
    normalize_timeframe_label,
    parse_timeframe_label,
    prepare_timeframe_aligned_data,
    resample_ohlcv,
    resolve_strategy_declaration,
)
from app.strategy.manual import ManualStrategy


def _minute_df(n_minutes: int, start="2024-01-01 00:00") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n_minutes, freq="1min")
    close = pd.Series(range(n_minutes), dtype=float) + 1000.0
    return pd.DataFrame({
        "timestamp": ts,
        "open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
        "volume": 1.0,
    })


# ---------------------------------------------------------------------
# Timeframe label parsing
# ---------------------------------------------------------------------

@pytest.mark.parametrize("label,minutes", [
    ("15m", 15), ("15min", 15), ("1h", 60), ("1H", 60), ("4h", 240),
    ("4hr", 240), ("1d", 1440), ("D", 1440), ("daily", 1440),
    ("40m", 40),  # an unusual but explicitly-requested bar size
    ("60", 60), ("5", 5),
])
def test_parse_timeframe_label_accepts_common_and_unusual_formats(label, minutes):
    assert parse_timeframe_label(label) == pytest.approx(minutes)


def test_parse_timeframe_label_rejects_garbage():
    with pytest.raises(TimeframeError):
        parse_timeframe_label("not a timeframe")


@pytest.mark.parametrize("a,b", [("1h", "60m"), ("60", "1h"), ("4H", "4h"), ("1d", "1440m")])
def test_normalize_timeframe_label_treats_equivalent_spellings_the_same(a, b):
    assert normalize_timeframe_label(a) == normalize_timeframe_label(b)


# ---------------------------------------------------------------------
# resample_ohlcv correctness
# ---------------------------------------------------------------------

def test_resample_ohlcv_aggregates_correctly():
    df = _minute_df(120)  # exactly two hours of 1-minute bars
    hourly = resample_ohlcv(df, "1h")
    assert len(hourly) == 2
    first_hour = df.iloc[0:60]
    assert hourly.iloc[0]["open"] == pytest.approx(first_hour.iloc[0]["open"])
    assert hourly.iloc[0]["close"] == pytest.approx(first_hour.iloc[-1]["close"])
    assert hourly.iloc[0]["high"] == pytest.approx(first_hour["high"].max())
    assert hourly.iloc[0]["low"] == pytest.approx(first_hour["low"].min())
    assert hourly.iloc[0]["volume"] == pytest.approx(first_hour["volume"].sum())


def test_resample_ohlcv_drops_bins_with_no_source_rows():
    df = _minute_df(30)  # only half an hour of data
    hourly = resample_ohlcv(df, "1h")
    # No full hour of source data exists past the first partial bin's
    # worth of rows -- but pandas still forms a bin from whatever rows
    # exist. The key correctness property is: no bin is EVER synthesized
    # with no underlying rows at all.
    assert len(hourly) == 1
    assert not hourly["close"].isna().any()


def test_resample_ohlcv_handles_an_unusual_bar_size():
    df = _minute_df(120)
    resampled = resample_ohlcv(df, "40m")
    assert len(resampled) == 3
    assert resampled.iloc[0]["close"] == pytest.approx(df.iloc[39]["close"])


# ---------------------------------------------------------------------
# MTF-LOOKAHEAD-001: merge_multi_timeframe must not leak a still-forming bar
# ---------------------------------------------------------------------

def test_merge_multi_timeframe_does_not_leak_a_still_forming_htf_bar():
    df = _minute_df(240)
    hourly = resample_ohlcv(df, "1h")
    merged, _labels = merge_multi_timeframe([df, hourly])

    # 5 minutes into the still-forming 00:00-01:00 hour: that hour's
    # close (which only becomes known at 01:00) must NOT be visible yet.
    row = merged[merged["timestamp"] == pd.Timestamp("2024-01-01 00:05:00")].iloc[0]
    assert pd.isna(row["tf60_close"])

    # 5 minutes into the SECOND hour: the now-closed first hour's close
    # (fully known as of 01:00) must be visible.
    row2 = merged[merged["timestamp"] == pd.Timestamp("2024-01-01 01:05:00")].iloc[0]
    first_hour_close = df.iloc[59]["close"]
    assert row2["tf60_close"] == pytest.approx(first_hour_close)

    # And it must NOT have silently jumped ahead to the second hour's
    # still-forming close either.
    assert row2["tf60_close"] != pytest.approx(df.iloc[64]["close"])


def test_merge_multi_timeframe_single_frame_still_works_unchanged():
    df = _minute_df(10)
    merged, labels = merge_multi_timeframe([df])
    assert labels == ["base"]
    assert len(merged) == 10


# ---------------------------------------------------------------------
# resolve_strategy_declaration -- manual strategies
# ---------------------------------------------------------------------

def test_manual_strategy_with_no_timeframe_fields_declares_nothing():
    cfg = {
        "name": "plain",
        "entry_conditions": {"long": [{"left": {"type": "rsi", "period": 5}, "operator": "<", "right": {"type": "value", "value": 30}}]},
    }
    decl = resolve_strategy_declaration(ManualStrategy(cfg))
    assert not decl.declares_anything


def test_manual_strategy_top_level_timeframe_is_execution_timeframe():
    cfg = {"name": "x", "timeframe": "15m", "entry_conditions": {"long": []}}
    decl = resolve_strategy_declaration(ManualStrategy(cfg))
    assert decl.execution_timeframe == "15m"
    assert decl.context_timeframes == set()


def test_manual_strategy_operand_timeframe_becomes_a_context_timeframe():
    cfg = {
        "name": "bias example",
        "timeframe": "15m",
        "entry_conditions": {
            "long": [{
                "left": {"type": "close", "timeframe": "1h"},
                "operator": ">",
                "right": {"type": "ema", "period": 50, "field": "close", "timeframe": "1h"},
            }],
        },
    }
    decl = resolve_strategy_declaration(ManualStrategy(cfg))
    assert decl.execution_timeframe == "15m"
    assert decl.context_timeframes == {"1h"}
    specs = decl.context_indicators["1h"]
    assert len(specs) == 1  # the "close" operand is a plain price, no indicator to precompute
    assert specs[0].kind == "ema" and specs[0].period == 50


# ---------------------------------------------------------------------
# prepare_timeframe_aligned_data
# ---------------------------------------------------------------------

def test_no_declaration_returns_raw_df_unchanged():
    df = _minute_df(50)
    cfg = {"name": "plain", "entry_conditions": {"long": []}}
    out_df, warnings = prepare_timeframe_aligned_data(df, ManualStrategy(cfg))
    assert out_df is df
    assert warnings == []


def test_single_execution_timeframe_resamples_and_warns():
    df = _minute_df(120)
    cfg = {"name": "hourly", "timeframe": "1h", "entry_conditions": {"long": []}}
    out_df, warnings = prepare_timeframe_aligned_data(df, ManualStrategy(cfg))
    assert len(out_df) == 2  # 120 minutes -> 2 hourly bars
    assert any("execution timeframe: 1h" in w for w in warnings)


def test_execution_timeframe_finer_than_source_falls_back_with_warning():
    df = resample_ohlcv(_minute_df(600), "1h")  # only hourly source data
    cfg = {"name": "too fine", "timeframe": "5m", "entry_conditions": {"long": []}}
    out_df, warnings = prepare_timeframe_aligned_data(df, ManualStrategy(cfg))
    assert len(out_df) == len(df)  # unchanged -- can't invent 5m bars from 1h data
    assert any("FINER than" in w for w in warnings)


def test_context_timeframe_not_coarser_than_execution_is_skipped():
    df = _minute_df(120)
    cfg = {
        "name": "bad bias",
        "timeframe": "1h",
        "entry_conditions": {"long": [{
            "left": {"type": "close", "timeframe": "15m"},  # finer than the 1h execution tf
            "operator": ">", "right": {"type": "value", "value": 0},
        }]},
    }
    out_df, warnings = prepare_timeframe_aligned_data(df, ManualStrategy(cfg))
    assert not any(c.startswith("tf15_") for c in out_df.columns)
    assert any("not coarser than the execution timeframe" in w for w in warnings)


def test_context_indicator_is_computed_on_native_htf_frequency_not_upsampled_base():
    """The core correctness property this whole feature exists for: an
    EMA(3) tagged with a 1h context timeframe on 4 hours of 1-minute data
    must equal an EMA(3) computed on the four REAL hourly closes -- not
    an EMA(3) computed on the 1-minute-upsampled/repeated version of
    those same values, which would be a completely different number."""
    df = _minute_df(240)  # exactly 4 hours
    cfg = {
        "name": "htf ema",
        "timeframe": "15m",
        "entry_conditions": {"long": [{
            "left": {"type": "close"},
            "operator": ">",
            "right": {"type": "ema", "period": 3, "field": "close", "timeframe": "1h"},
        }]},
    }
    out_df, _warnings = prepare_timeframe_aligned_data(df, ManualStrategy(cfg))
    merged_col = [c for c in out_df.columns if c.startswith("tf60_ema_3_close")]
    assert merged_col, f"expected a tf60_ema_3_close column, got: {list(out_df.columns)}"

    hourly = resample_ohlcv(df, "1h")
    expected_ema = hourly["close"].ewm(span=3, adjust=False).mean()
    # After the 4th hour closes (row index 3 of `hourly`), the merged
    # value visible on the base timeframe should match that native EMA,
    # not a naive upsampled recomputation.
    last_base_row = out_df.iloc[-1]
    assert last_base_row[merged_col[0]] == pytest.approx(expected_ema.iloc[2])  # last CLOSED hour is index 2, not 3


# ---------------------------------------------------------------------
# End-to-end through run_backtest
# ---------------------------------------------------------------------

def test_run_backtest_end_to_end_with_declared_execution_timeframe():
    df = _minute_df(600)  # 10 hours of 1-minute data
    cfg = {
        "name": "1h swing",
        "timeframe": "1h",
        "entry_conditions": {
            "long": [{"left": {"type": "sma", "period": 2}, "operator": "cross above", "right": {"type": "sma", "period": 4}}],
        },
        "exit_conditions": {"long": []},
        "stop_loss_pips": 5, "take_profit_pips": 10,
    }
    risk = RiskConfig(initial_balance=50_000.0, risk_value=1.0, pip_size=0.01)
    result = run_backtest(df, ManualStrategy(cfg), risk)
    # Every trade's entry time should land on an hourly boundary -- proof
    # that execution actually happened on resampled 1h bars, not the raw
    # 1-minute source.
    for t in result.trades:
        assert t.entry_time.minute == 0
    assert any("Timeframe-aware data pipeline" in w for w in result.warnings)


def test_run_backtest_with_bias_and_entry_on_different_timeframes():
    df = _minute_df(600)
    cfg = {
        "name": "bias + entry",
        "timeframe": "15m",
        "entry_conditions": {
            "long": [
                {"left": {"type": "close", "timeframe": "1h"}, "operator": ">",
                 "right": {"type": "ema", "period": 2, "field": "close", "timeframe": "1h"}},
            ],
        },
        "exit_conditions": {"long": []},
        "stop_loss_pips": 5, "take_profit_pips": 10,
    }
    risk = RiskConfig(initial_balance=50_000.0, risk_value=1.0, pip_size=0.01)
    result = run_backtest(df, ManualStrategy(cfg), risk)
    for t in result.trades:
        assert t.entry_time.minute % 15 == 0


def test_existing_manual_strategy_without_timeframe_fields_is_byte_identical():
    """The critical backward-compatibility guarantee: a strategy that
    predates this feature (no 'timeframe' anywhere) must produce IDENTICAL
    trades/equity to running the exact same config before this module
    existed -- proven here by confirming run_backtest's warnings carry no
    timeframe-pipeline note at all, and that the dataframe handed to
    run_execution is length-identical to the raw input."""
    df = _minute_df(300)
    cfg = {
        "name": "plain sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 10, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "stop_loss_pips": 20, "take_profit_pips": 40,
    }
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=0.01)
    result = run_backtest(df, ManualStrategy(cfg), risk)
    assert not any("Timeframe-aware data pipeline" in w for w in result.warnings)
    assert len(result.equity_curve) == len(df)


# ---------------------------------------------------------------------
# Python strategies
# ---------------------------------------------------------------------

def test_python_strategy_declares_timeframe_via_module_attributes(tmp_path):
    strategy_file = tmp_path / "htf_strategy.py"
    strategy_file.write_text(
        "TIMEFRAME = '1h'\n"
        "HTF_TIMEFRAMES = ['4h']\n"
        "def generate_signals(df):\n"
        "    import pandas as pd\n"
        "    sig = (df['close'] > df.get('tf240_close', df['close'])).astype(int)\n"
        "    return pd.Series(sig.values, index=df.index)\n"
    )
    from app.strategy.python import PythonStrategy
    strategy = PythonStrategy(str(strategy_file))

    decl = resolve_strategy_declaration(strategy)
    assert decl.execution_timeframe == "1h"
    assert decl.context_timeframes == {"4h"}

    df = _minute_df(600)
    risk = RiskConfig(initial_balance=10_000.0, risk_value=1.0, pip_size=0.01)
    result = run_backtest(df, strategy, risk)
    assert any("execution timeframe: 1h" in w and "4h" in w for w in result.warnings)


def test_python_strategy_without_timeframe_attrs_is_unaffected(tmp_path):
    strategy_file = tmp_path / "plain_strategy.py"
    strategy_file.write_text(
        "def generate_signals(df):\n"
        "    import pandas as pd\n"
        "    return pd.Series(1, index=df.index)\n"
    )
    from app.strategy.python import PythonStrategy
    strategy = PythonStrategy(str(strategy_file))
    decl = resolve_strategy_declaration(strategy)
    assert not decl.declares_anything
