from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy.pinescript import PineScriptStrategy
from app.strategy.translator import (
    TranslationError, parse_pinescript, to_mql5, to_pinescript, to_python, translate_strategy,
)

BASIC_CONFIG = {
    "name": "EMA Cross RSI Filter",
    "market": {"direction": "Both"},
    "entry_conditions": {
        "long": [
            {"left": {"type": "ema", "period": 20, "field": "close"}, "operator": "crosses above",
             "right": {"type": "ema", "period": 50, "field": "close"}},
            {"left": {"type": "rsi", "period": 14, "field": "close"}, "operator": "<", "right": 70},
        ],
        "long_connectors": ["AND"],
        "short": [
            {"left": {"type": "ema", "period": 20, "field": "close"}, "operator": "crosses below",
             "right": {"type": "ema", "period": 50, "field": "close"}},
        ],
    },
    "exit_conditions": {
        "long": [{"left": {"type": "rsi", "period": 14, "field": "close"}, "operator": ">", "right": 80}],
        "short": [{"left": {"type": "rsi", "period": 14, "field": "close"}, "operator": "<", "right": 20}],
    },
    "risk_management": {
        "stop_type": "atr", "stop_value": 1.5, "stop_atr_period": 14,
        "target_type": "atr", "target_value": 3.0, "target_atr_period": 14,
        "max_bars_in_trade": 40,
    },
}


def _synthetic_df(n=400):
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({
        "timestamp": idx, "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": 1000.0,
    })


def test_to_pinescript_basic_shape():
    code = to_pinescript(BASIC_CONFIG)
    assert code.startswith("//@version=5")
    assert "ta.ema(close, 20)" in code
    assert "ta.crossover(ema_20_close, ema_50_close)" in code
    assert 'strategy.entry("Long", strategy.long)' in code
    assert "T58_SL_ATR_MULT=1.5" in code
    assert "barsInTrade" in code


def test_to_pinescript_round_trips_through_own_parser():
    """The translator's output should itself be loadable by T58's own
    PineScript parser -- proof the generated code isn't just cosmetically
    plausible but actually uses ta.crossover/ta.crossunder and plain
    comparisons, the same primitives T58's own parser recognizes. Uses a
    config with NO risk_management: T58's own Pine parser (unlike
    TradingView's real compiler) only ever recognizes stop/target via its
    T58_SL_PIPS-style comment directives, not the real strategy.exit()
    price-level computation this translator emits for TradingView's
    benefit -- so those lines are valid, TradingView-compilable code that
    simply isn't expected to re-parse through T58's own line-based
    subset, and max_bars_in_trade's `var`/`:=` state likewise isn't part
    of that subset. Entry/exit condition translation is this test's
    concern; risk-management code shape is covered by the shape tests
    above instead."""
    config = {**BASIC_CONFIG, "risk_management": {}}
    code = to_pinescript(config)
    strat = PineScriptStrategy(code)
    df = _synthetic_df()
    result = strat.generate(df)
    assert len(result.signals) == len(df)
    assert set(result.signals.unique()).issubset({-1, 0, 1})


def test_to_mql5_basic_shape():
    code = to_mql5(BASIC_CONFIG)
    assert "#include <Trade\\Trade.mqh>" in code
    assert "iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_EMA, PRICE_CLOSE)" in code
    assert "iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE)" in code
    assert "trade.Buy(" in code and "trade.Sell(" in code
    assert "T58_SL_ATR_MULT=1.5" in code


def test_to_mql5_atr_risk_handles_independent_of_conditions():
    """ATR-multiple stop/target must get a real indicator handle even when
    no condition in the strategy itself references ATR."""
    config = {
        "name": "Plain MA Cross",
        "entry_conditions": {
            "long": [{"left": {"type": "sma", "period": 10, "field": "close"}, "operator": "crosses above",
                      "right": {"type": "sma", "period": 30, "field": "close"}}],
            "short": [{"left": {"type": "sma", "period": 10, "field": "close"}, "operator": "crosses below",
                       "right": {"type": "sma", "period": 30, "field": "close"}}],
        },
        "exit_conditions": {},
        "risk_management": {"stop_type": "atr", "stop_value": 2.0, "stop_atr_period": 14,
                             "target_type": "atr", "target_value": 4.0, "target_atr_period": 21},
    }
    code = to_mql5(config)
    assert "iATRValueUnavailable" not in code
    assert "h_risk_atr_14" in code
    assert "h_risk_atr_21" in code


def test_direction_restriction_forces_flat_side():
    config = dict(BASIC_CONFIG)
    config["market"] = {"direction": "Long"}
    pine = to_pinescript(config)
    assert "shortEntryCond = false" in pine
    mql = to_mql5(config)
    assert "bool shortEntryCond = false;" in mql


def test_trailing_and_breakeven_emit_todo_not_fabricated_logic():
    config = dict(BASIC_CONFIG)
    config["risk_management"] = dict(BASIC_CONFIG["risk_management"])
    config["risk_management"]["trailing_stop"] = {"enabled": True, "value": 1.0, "atr_period": 14}
    config["risk_management"]["break_even"] = {"enabled": True, "trigger_r": 1.0}
    pine = to_pinescript(config)
    assert "TODO(T58): trailing stop" in pine
    assert "TODO(T58): break-even" in pine


@pytest.mark.parametrize("kind", ["liquidity_sweep", "break_of_structure", "fair_value_gap", "session_high", "atr_regime"])
def test_unsupported_smc_operand_kinds_raise(kind):
    config = {
        "name": "Bad",
        "entry_conditions": {"long": [{"left": {"type": kind}, "operator": ">", "right": 0}]},
        "exit_conditions": {},
        "risk_management": {},
    }
    with pytest.raises(TranslationError):
        to_pinescript(config)
    with pytest.raises(TranslationError):
        to_mql5(config)


def test_expression_string_config_rejected():
    with pytest.raises(TranslationError):
        to_pinescript({"long_entry": "close > sma_20"})


def test_unsupported_operator_raises():
    config = {
        "name": "Bad Op",
        "entry_conditions": {"long": [{"left": "close", "operator": "weird_op", "right": 1}]},
        "exit_conditions": {},
        "risk_management": {},
    }
    with pytest.raises(TranslationError):
        to_pinescript(config)


VWAP_CONFIG = {
    "name": "VWAP Reclaim",
    "market": {"direction": "long"},
    "entry_conditions": {
        "long": [{"left": "close", "operator": "crosses above", "right": {"type": "vwap"}}],
    },
    "exit_conditions": {
        "long": [{"left": "close", "operator": "crosses below", "right": {"type": "vwap"}}],
    },
    "risk_management": {},
}


def test_vwap_renders_native_call_in_every_target():
    pine = to_pinescript(VWAP_CONFIG)
    assert "ta.vwap(hlc3)" in pine

    mql5 = to_mql5(VWAP_CONFIG)
    assert "ComputeVWAP" in mql5
    # helper should only be emitted when a strategy actually uses VWAP
    assert "ComputeVWAP" not in to_mql5(BASIC_CONFIG)

    python_code = to_python(VWAP_CONFIG)
    assert "vwap(work)" in python_code
    assert "    atr, bollinger, crossover, crossunder, ema, highest_high, lowest_low, macd, rsi, sma, vwap, wma,\n" in python_code


def test_vwap_round_trips_through_every_direction():
    pine = translate_strategy("manual", VWAP_CONFIG, "pinescript")
    mql5 = translate_strategy("pinescript", pine, "mql5")  # uses the embedded round-trip directive
    python_code = translate_strategy("mql5", mql5, "python")
    pine_again = translate_strategy("python", python_code, "pinescript")
    assert "ta.vwap(hlc3)" in pine_again


def test_hand_written_pine_vwap_parses_via_real_symbolic_parse():
    hand_written = (
        '//@version=5\n'
        'strategy("Hand VWAP", overlay=true)\n'
        'vw = ta.vwap(hlc3)\n'
        'longCond = close > vw\n'
        'if longCond\n'
        '    strategy.entry("Long", strategy.long)\n'
        'strategy.close("Long", when=close < vw)\n'
    )
    config = parse_pinescript(hand_written)
    assert config["entry_conditions"]["long"][0]["right"]["type"] == "vwap"
    # and it keeps translating onward from there
    mql5 = translate_strategy("manual", config, "mql5")
    assert "ComputeVWAP" in mql5
