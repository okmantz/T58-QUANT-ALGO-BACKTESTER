"""Regression tests for the Universal Strategy Translator overhaul.

Owen's report (2026-10-06): translating shipped strategies to other
languages "gave several different errors" -- in fact ZERO shipped manual
strategies translated. Root causes covered here:

* MACD/Bollinger operands whose config carries (or defaults) a period
  crashed every target with a KeyError -- declarations registered under
  fixed ("macd", field, 0)-style keys while operand lookup keys include
  the period (the v9 champion's macd_histogram defaults to 14).
* Whole operand families the condition builder emits (vwma, donchian_*,
  adx, bos/choch, liquidity_sweep, fvg, order_block, swings, regimes,
  session levels, "is true") were refused outright.

The parity tests compare the translator's condition groups against the
engine's own ManualStrategy evaluation bar-for-bar -- the generated
Python must compute the same entries/exits the backtester does.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import app.strategy.indicators as ind
from app.strategy.manual import ManualStrategy
from app.strategy.translator import (
    TranslationError,
    _collect_indicators,
    _python_declare_indicators,
    _python_render_group,
    parse_manual_config,
    to_mql5,
    to_pinescript,
    to_python,
    translate_strategy,
)

REPO = Path(__file__).resolve().parent.parent
MANUAL_DIR = REPO / "strategies" / "manual"
SAMPLE_CSV = REPO / "data" / "examples" / "EURUSD_5M_sample.csv"

STRATEGY_FILES = sorted(p for p in MANUAL_DIR.glob("*.json") if p.name != "v9_champion_params.json")
TARGETS = ["pinescript", "mql5", "python"]

_INDICATOR_FNS = (
    "atr", "bollinger", "crossover", "crossunder", "ema", "highest_high",
    "lowest_low", "macd", "rsi", "sma", "vwap", "wma",
    "adx", "average_volume", "candle_range", "donchian", "percentage_change", "vwma",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _generated_groups(config: dict, frame: pd.DataFrame):
    """The translator's four condition groups, evaluated on `frame` by
    executing exactly the declarations/renderers to_python emits."""
    parsed = parse_manual_config(config)
    indicators = _collect_indicators(parsed)
    lines, var_map = _python_declare_indicators(indicators)
    ns = {fn: getattr(ind, fn) for fn in _INDICATOR_FNS}
    ns["pd"] = pd
    ns["work"] = frame.copy()
    exec("\n".join(lines), ns)
    return [
        eval(_python_render_group(g, var_map), ns)
        for g in (parsed.long_entry, parsed.long_exit, parsed.short_entry, parsed.short_exit)
    ]


def _engine_groups(config: dict, frame: pd.DataFrame):
    strat = ManualStrategy(config)
    work = strat._build_indicators(frame.copy())
    return list(strat._build_visual_signals(work))


def _assert_group_parity(config: dict, frame: pd.DataFrame):
    engine = _engine_groups(config, frame)
    generated = _generated_groups(config, frame)
    for label, eng, gen in zip(("long_entry", "long_exit", "short_entry", "short_exit"), engine, generated):
        e = pd.Series(eng).fillna(False).astype(bool).reset_index(drop=True)
        g = pd.Series(gen).fillna(False).astype(bool).reset_index(drop=True)
        mismatches = int((e != g).sum())
        assert mismatches == 0, f"{label}: {mismatches} bars differ from the engine"


# ---------------------------------------------------------------------------
# The shipped corpus must translate everywhere (params file refuses cleanly)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", STRATEGY_FILES, ids=lambda p: p.name)
@pytest.mark.parametrize("target", TARGETS)
def test_shipped_strategies_translate(path, target):
    out = translate_strategy("manual", _load(path), target)
    assert isinstance(out, str) and len(out) > 200


def test_params_file_refuses_cleanly_not_crashes():
    cfg = _load(MANUAL_DIR / "v9_champion_params.json")
    for target in TARGETS:
        with pytest.raises(TranslationError):
            translate_strategy("manual", cfg, target)


def test_champion_macd_histogram_period_keyerror_regression():
    cfg = _load(MANUAL_DIR / "v9_champion_momentum_continuation.json")
    assert "ta.macd" in to_pinescript(cfg)
    assert "iMACD" in to_mql5(cfg)
    compile(to_python(cfg), "<champion>", "exec")


# ---------------------------------------------------------------------------
# Generated Python: compiles, runs, and matches the engine's conditions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", STRATEGY_FILES, ids=lambda p: p.name)
def test_generated_python_runs_on_sample_data(path, tmp_path):
    frame = pd.read_csv(SAMPLE_CSV)
    code = to_python(_load(path))
    module_file = tmp_path / "generated_strategy.py"
    module_file.write_text(code)
    ns: dict = {}
    exec(compile(code, str(module_file), "exec"), ns)
    signals = pd.Series(ns["generate_signals"](frame.copy()))
    assert len(signals) == len(frame)
    assert set(signals.dropna().unique()) <= {-1, 0, 1}


@pytest.mark.parametrize("path", STRATEGY_FILES, ids=lambda p: p.name)
def test_condition_groups_match_engine(path):
    frame = pd.read_csv(SAMPLE_CSV)
    _assert_group_parity(_load(path), frame)


def test_extended_kind_families_match_engine():
    """Families no shipped strategy exercises together: session levels,
    regimes under 'is true', fractal swing events, day-of-week."""
    frame = pd.read_csv(SAMPLE_CSV)
    config = {
        "name": "Extended Families",
        "entry_conditions": {
            "long": [
                {"left": {"type": "close"}, "operator": ">",
                 "right": {"type": "session_high", "session_start": "08:30", "session_end": "15:00"}},
                {"left": {"type": "atr_regime", "period": 14}, "operator": "is true", "right": 1},
                {"left": {"type": "swing_bos", "lookback": 5}, "operator": "is true", "right": 1},
                {"left": {"type": "day_of_week", "days": [1, 3]}, "operator": "is true", "right": 1},
            ],
            "long_connectors": ["AND", "AND", "AND"],
            "short": [
                {"left": {"type": "volatility_regime", "period": 10}, "operator": "<",
                 "right": {"type": "value", "value": 0}},
                {"left": {"type": "close"}, "operator": "<",
                 "right": {"type": "opening_range_low", "session_start": "09:30", "session_end": "10:30"}},
                {"left": {"type": "previous_day_high"}, "operator": ">", "right": {"type": "value", "value": 0}},
            ],
            "short_connectors": ["AND", "AND"],
        },
        "exit_conditions": {
            "long": [{"left": {"type": "swing_choch", "lookback": 5}, "operator": "is true", "right": 1}],
            "short": [{"left": {"type": "time_of_day", "session_start": "15:00", "session_end": "16:00"},
                       "operator": "is true", "right": 1}],
        },
        "risk_management": {},
    }
    _assert_group_parity(config, frame)
    for renderer in (to_pinescript, to_mql5):
        assert renderer(config)


def test_bos_bullish_fires_at_known_break_bar():
    # Flat bars then a close through the prior 3-bar high at bar 6.
    rows = []
    closes = [10, 10, 10, 10, 10, 10, 12, 10]
    for i, c in enumerate(closes):
        rows.append({
            "timestamp": pd.Timestamp("2024-01-02 09:30") + pd.Timedelta(minutes=5 * i),
            "open": 10.0, "high": max(10.5, c), "low": 9.5, "close": float(c), "volume": 100.0,
        })
    frame = pd.DataFrame(rows)
    config = {
        "name": "BOS probe",
        "entry_conditions": {
            "long": [{"left": {"type": "bos", "lookback": 3, "direction": "bullish"},
                      "operator": "is true", "right": 1}],
        },
        "exit_conditions": {},
        "risk_management": {},
    }
    _assert_group_parity(config, frame)
    gen_long = pd.Series(_generated_groups(config, frame)[0]).fillna(False).astype(bool)
    assert bool(gen_long.iloc[6]) is True
    assert int(gen_long.sum()) == 1


# ---------------------------------------------------------------------------
# Round-trips through the embedded config directive
# ---------------------------------------------------------------------------

def test_round_trip_manual_pine_mql5_and_python_pine():
    cfg = _load(MANUAL_DIR / "Gold Trend Breakout.json")
    pine = translate_strategy("manual", cfg, "pinescript")
    assert "void OnTick" in translate_strategy("pinescript", pine, "mql5")
    py = translate_strategy("manual", cfg, "python")
    assert "strategy(" in translate_strategy("python", py, "pinescript")


# ---------------------------------------------------------------------------
# Still-refused cases stay clean TranslationErrors (never crashes)
# ---------------------------------------------------------------------------

def test_ib_contraction_ratio_still_refused_cleanly():
    config = {
        "name": "IB",
        "entry_conditions": {"long": [{"left": {"type": "ib_contraction_ratio"}, "operator": "<", "right": 0.8}]},
        "exit_conditions": {},
        "risk_management": {},
    }
    for renderer in (to_pinescript, to_mql5, to_python):
        with pytest.raises(TranslationError):
            renderer(config)


def test_timeframe_pinned_operand_refused_not_silently_wrong():
    config = {
        "name": "MTF",
        "entry_conditions": {"long": [{"left": {"type": "ema", "period": 20, "timeframe": "1h"},
                                       "operator": ">", "right": 0}]},
        "exit_conditions": {},
        "risk_management": {},
    }
    for renderer in (to_pinescript, to_mql5, to_python):
        with pytest.raises(TranslationError):
            renderer(config)
