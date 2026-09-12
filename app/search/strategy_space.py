"""
Strategy Space Generator (Search Lab -- Stage 0).

Feeds the batch runner's Stage 1 cheap filter. Two modes:

  "single" -- wrap ONE user-supplied strategy (Manual config, or a built
              Python/PineScript/MQL5 Strategy instance) as a size-1 space,
              so the exact same Stage 1-5 pipeline (filter -> GA refine ->
              validation gate -> leaderboard -> promote) can be used to
              rigorously re-validate a single hand-built strategy of ANY
              supported source type, not just to search a family.

  "family" -- combinatorially expand a search space into many concrete
              candidates. Two distinct ways to do this:

                1. Named hypothesis family (Manual strategies only): a
                   named economic hypothesis + a grid of parameter values
                   (see the FAMILIES registry below).
                2. Grid around one given strategy (`strategy=` argument,
                   any source type -- Manual, Python, PineScript, MQL5):
                   discovers that strategy's own tunable numeric
                   parameters (the same discovery Step 6's Iterative
                   Refinement GA already uses) and grid-searches a
                   discretized range around each one.

Every candidate produced by either mode is represented uniformly as a
"candidate spec" dict so the rest of the Search Lab (batch_runner,
robustness, results_db) never has to care how a candidate was generated:

    {"source_type": "manual", "config": {...}}
    {"source_type": "python", "code_text": "...", "code_extension": ".py"}
    {"source_type": "pinescript", "code_text": "...", "code_extension": ".pine"}
    {"source_type": "mql5", "code_text": "...", "code_extension": ".mq5"}

`build_strategy_from_spec()` turns any of these back into a live Strategy
instance ready to run through app.backtest.engine.run_backtest. This is
the ONE place that dispatch happens; batch_runner.py and robustness.py
both import and use it rather than special-casing source types themselves.

Deliberately NOT random indicator soup for the named-family path. Pure
brute-force combinatorics over arbitrary indicators mostly finds noise
faster than signal, especially at the win-rate/RR targets this app is
built around (see /areas/prop-firm-falsification-kit.md's own findings:
nine instrument/signal combinations tested by hand, one validated edge).
Each named family encodes a specific, named trading hypothesis; the grid
only varies ITS parameters. The "grid around a given strategy" path
sidesteps this concern differently: it only ever varies parameters of a
hypothesis the user themselves already wrote (a Manual config, or a real
Python/PineScript/MQL5 file), never invents new logic.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.optimize.code_parameter_space import (
    CodeGene, apply_code_genome, discover_code_genes,
)
from app.optimize.parameter_space import GeneMeta, apply_genome, extract_genome
from app.strategy.base import Strategy
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy

CANDIDATE_SOURCE_TYPES = {"manual", "python", "pinescript", "mql5"}
_CODE_SOURCE_TYPES = {"python", "pinescript", "mql5"}
_CODE_EXTENSIONS = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}


class StrategySpaceError(Exception):
    """Raised for an invalid search-space request (unknown family, bad config, etc.)."""


# ---------------------------------------------------------------------------
# Space container
# ---------------------------------------------------------------------------

@dataclass
class SearchSpace:
    mode: str                        # "single" | "family"
    family: str | None                # family name, "all", "<source_type>_grid", or None for single mode
    candidates: dict[str, dict]       # candidate_id -> candidate spec dict (see module docstring)
    meta: dict[str, dict]             # candidate_id -> {"family": str, "params": dict}
    total_generated: int              # size of the full grid before any sampling cap
    sampled: bool                     # True if total_generated > requested max_candidates


# ---------------------------------------------------------------------------
# Candidate spec <-> live Strategy instance (the one place source_type is
# dispatched on for building a runnable strategy)
# ---------------------------------------------------------------------------

def _source_text_for_strategy(strategy: Strategy) -> str:
    """The exact, unmodified source text for a code-based strategy instance."""
    if strategy.source_type == "python":
        return Path(strategy.file_path).read_text(encoding="utf-8", errors="ignore")
    return strategy.code  # PineScriptStrategy / MQL5Strategy already hold this in memory


def spec_from_strategy(strategy: Strategy) -> dict:
    """Builds a uniform candidate spec dict from any already-built Strategy instance."""
    st = strategy.source_type
    if st == "manual":
        return {"source_type": "manual", "config": copy.deepcopy(strategy.config)}
    if st in _CODE_SOURCE_TYPES:
        return {
            "source_type": st,
            "code_text": _source_text_for_strategy(strategy),
            "code_extension": _CODE_EXTENSIONS[st],
        }
    raise StrategySpaceError(f"Unsupported strategy source type '{st}'.")


def build_strategy_from_spec(spec: dict, tmp_dir: str | Path | None = None) -> Strategy:
    """
    The single dispatch point that turns a candidate spec dict back into a
    live, runnable Strategy instance. `tmp_dir` is only required for
    Python candidates (PythonStrategy only accepts a file path) -- a fresh
    uniquely-named temp file is written there per call; PineScript/MQL5
    build directly from in-memory text, and Manual builds directly from
    the config dict, so `tmp_dir` is unused for those.
    """
    st = spec.get("source_type", "manual")
    if st == "manual":
        return ManualStrategy(spec["config"])
    if st == "python":
        if tmp_dir is None:
            raise StrategySpaceError(
                "Building a Python strategy candidate requires a writable tmp_dir "
                "(PythonStrategy only accepts a file path)."
            )
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / f"candidate_{uuid.uuid4().hex}.py"
        path.write_text(spec["code_text"], encoding="utf-8")
        return PythonStrategy(path)
    if st == "pinescript":
        return PineScriptStrategy(spec["code_text"])
    if st == "mql5":
        return MQL5Strategy(spec["code_text"])
    raise StrategySpaceError(f"Unknown source_type '{st}' in candidate spec.")




@dataclass(frozen=True)
class SkeletonSpec:
    name: str
    label: str
    description: str
    param_grid: dict[str, list]
    build: Callable[[dict], dict]
    valid: Callable[[dict], bool] = field(default=lambda params: True)
    requires_pair_data: bool = False   # True only for the stat_pairs family -- see
    # app.data.pairs.merge_pair_series; generate_search_space()/batch_runner surface a
    # clear error up front if this family is requested without a "pair_close" column
    # merged into the working DataFrame, instead of letting it fail bar-by-bar as NaNs.

    def combinations(self) -> list[dict]:
        keys = list(self.param_grid.keys())
        value_lists = [self.param_grid[k] for k in keys]
        out = []
        for combo in itertools.product(*value_lists):
            params = dict(zip(keys, combo))
            if self.valid(params):
                out.append(params)
        return out


def _risk_management(
    stop_atr_mult: float, target_atr_mult: float, stop_atr_period: int = 14,
    target_atr_period: int = 14, max_bars_in_trade: int | None = None,
) -> dict:
    return {
        "stop_type": "atr",
        "stop_value": stop_atr_mult,
        "stop_atr_period": stop_atr_period,
        "target_type": "atr",
        "target_value": target_atr_mult,
        "target_atr_period": target_atr_period,
        "opposite_signal_exit": True,
        **({"max_bars_in_trade": max_bars_in_trade} if max_bars_in_trade else {}),
    }


def _ind(kind: str, period: int, field_name: str = "close") -> dict:
    return {"type": kind, "period": period, "field": field_name}


def _val(value: float) -> dict:
    return {"type": "value", "value": value}


def _cond(left: dict, operator: str, right: dict) -> dict:
    return {"left": left, "operator": operator, "right": right}


# ---------------------------------------------------------------------------
# Family A: Trend Breakout
#   Donchian-style N-bar breakout, only taken in the direction of an EMA
#   trend filter. Mirrors the trend-aligned breakout family that already
#   showed a real (if imperfect) out-of-sample edge on Gold in
#   /areas/prop-firm-falsification-kit.md.
# ---------------------------------------------------------------------------

def _breakout_flag(lookback: int, direction: str) -> dict:
    """
    A proper N-bar breakout EXCLUDING the current bar, via the existing
    "break_of_structure" primitive (app/strategy/manual.py::_advanced_boolean
    computes prior_high/prior_low as high.shift(1).rolling(lookback).max()/
    low.shift(1).rolling(lookback).min() -- already correctly non-lookahead
    and already excludes the bar itself, unlike a naive
    "close > highest_high(lookback)" check, which can never fire: a plain
    rolling max/min over a window THAT INCLUDES the current bar can never be
    exceeded by that same bar's own close, since the bar's own high/low is
    part of the window being compared against).
    """
    return {"type": "bos", "lookback": lookback, "direction": direction}


def _build_trend_breakout(p: dict) -> dict:
    lookback, ema_fast, ema_slow = p["lookback"], p["ema_fast"], p["ema_slow"]
    return {
        "name": f"Trend Breakout (lb={lookback}, ema {ema_fast}/{ema_slow})",
        "entry_conditions": {
            "long": [
                _cond(_breakout_flag(lookback, "bullish"), "is true", _val(1)),
                _cond(_ind("ema", ema_fast), ">", _ind("ema", ema_slow)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_breakout_flag(lookback, "bearish"), "is true", _val(1)),
                _cond(_ind("ema", ema_fast), "<", _ind("ema", ema_slow)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_TREND_BREAKOUT = SkeletonSpec(
    name="trend_breakout",
    label="Trend Breakout (Donchian + EMA filter)",
    description=(
        "N-bar price breakout taken only in the direction of a slower EMA trend filter, "
        "with an ATR stop and ATR target. Trades infrequently but with a real, named "
        "directional hypothesis behind every signal."
    ),
    param_grid={
        "lookback": [10, 20, 40],
        "ema_fast": [20, 50],
        "ema_slow": [100, 200],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0, 4.0],
    },
    build=_build_trend_breakout,
    valid=lambda p: p["ema_fast"] < p["ema_slow"],
)


# ---------------------------------------------------------------------------
# Family B: MTF Pullback / Trend Continuation
#   Trade pullbacks (RSI dips/pops) that occur WITHIN an established EMA
#   trend, rather than breakouts of it. Single-timeframe proxy for the
#   5m/15m/1h confluence idea in /areas/pinescript-strategy.md -- extend by
#   adding a genuinely higher-timeframe operand once multi-CSV context is
#   wired through the search pipeline (see integration notes).
# ---------------------------------------------------------------------------

def _build_mtf_pullback(p: dict) -> dict:
    ema_fast, ema_slow = p["ema_fast"], p["ema_slow"]
    rsi_period, rsi_low, rsi_high = p["rsi_period"], p["rsi_pullback_low"], p["rsi_pullback_high"]
    return {
        "name": f"Trend Pullback (ema {ema_fast}/{ema_slow}, rsi{rsi_period} {rsi_low}/{rsi_high})",
        "entry_conditions": {
            "long": [
                _cond(_ind("ema", ema_fast), ">", _ind("ema", ema_slow)),
                _cond(_ind("rsi", rsi_period), "<", _val(rsi_low)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("ema", ema_fast), "<", _ind("ema", ema_slow)),
                _cond(_ind("rsi", rsi_period), ">", _val(rsi_high)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("rsi", rsi_period), ">", _val(rsi_high))],
            "short": [_cond(_ind("rsi", rsi_period), "<", _val(rsi_low))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_MTF_PULLBACK = SkeletonSpec(
    name="mtf_pullback",
    label="Trend Pullback (EMA trend + RSI dip/pop)",
    description=(
        "Buys RSI pullbacks within an established EMA uptrend, sells RSI pops within an "
        "established EMA downtrend -- a continuation, not a reversal, hypothesis."
    ),
    param_grid={
        "ema_fast": [20, 50],
        "ema_slow": [100, 200],
        "rsi_period": [7, 14],
        "rsi_pullback_low": [30, 40],
        "rsi_pullback_high": [60, 70],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [24, 48],
    },
    build=_build_mtf_pullback,
    valid=lambda p: p["ema_fast"] < p["ema_slow"] and p["rsi_pullback_low"] < p["rsi_pullback_high"],
)


# ---------------------------------------------------------------------------
# Family C: Mean-Reversion Band Fade
#   Fades price extremes outside a Bollinger Band, filtered by RSI to avoid
#   fading a band walk during a strong trend. Mirrors the Bollinger
#   mean-reversion family already tried (and retired) in Owen's prop-firm
#   strategy graveyard -- kept here so the search can re-test it
#   systematically across a real grid instead of a single hand-picked config.
# ---------------------------------------------------------------------------

def _build_mean_reversion_band(p: dict) -> dict:
    bb_period, bb_std = p["bb_period"], p["bb_std"]
    rsi_period, oversold, overbought = p["rsi_period"], p["oversold"], 100 - p["oversold"]
    return {
        "name": f"Band Fade (bb{bb_period}x{bb_std}, rsi{rsi_period})",
        "entry_conditions": {
            "long": [
                _cond(_ind("close", 1), "<", {"type": "bollinger_lower", "period": bb_period, "field": "close"}),
                _cond(_ind("rsi", rsi_period), "<", _val(oversold)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("close", 1), ">", {"type": "bollinger_upper", "period": bb_period, "field": "close"}),
                _cond(_ind("rsi", rsi_period), ">", _val(overbought)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), ">", {"type": "bollinger_mid", "period": bb_period, "field": "close"})],
            "short": [_cond(_ind("close", 1), "<", {"type": "bollinger_mid", "period": bb_period, "field": "close"})],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_MEAN_REVERSION_BAND = SkeletonSpec(
    name="mean_reversion_band",
    label="Mean-Reversion Band Fade (Bollinger + RSI filter)",
    description=(
        "Fades price closing outside a Bollinger Band, filtered by RSI extremity, targeting "
        "reversion to the band midline. The exact family already retired once in the prop-firm "
        "strategy graveyard -- included so the search can re-falsify it across a real grid "
        "rather than resting on one earlier hand-picked config."
    ),
    param_grid={
        "bb_period": [14, 20, 30],
        "bb_std": [2.0, 2.5],
        "rsi_period": [7, 14],
        "oversold": [20, 30],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.0],
    },
    build=_build_mean_reversion_band,
)


# ---------------------------------------------------------------------------
# Family D: Volatility Breakout
#   A Donchian-style breakout taken ONLY while the market's own ATR regime
#   is expanding relative to its recent baseline -- a distinct hypothesis
#   from Family A's trend-breakout, which filters by trend DIRECTION (EMA
#   alignment) rather than by volatility STATE. Many real breakout edges
#   depend on catching genuine range expansion, not just any N-bar high;
#   trading every breakout regardless of the prevailing volatility regime
#   is a common way that family fails in a way this family is built to
#   avoid.
# ---------------------------------------------------------------------------

def _build_volatility_breakout(p: dict) -> dict:
    lookback, atr_period, expansion_mult = p["lookback"], p["atr_period"], p["expansion_mult"]
    return {
        "name": f"Volatility Breakout (lb={lookback}, atr{atr_period} x{expansion_mult} expansion)",
        "entry_conditions": {
            "long": [
                _cond(_breakout_flag(lookback, "bullish"), "is true", _val(1)),
                _cond({"type": "atr_regime", "period": atr_period, "expansion_mult": expansion_mult}, "==", _val(1)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_breakout_flag(lookback, "bearish"), "is true", _val(1)),
                _cond({"type": "atr_regime", "period": atr_period, "expansion_mult": expansion_mult}, "==", _val(1)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p.get("max_bars")),
    }


_VOLATILITY_BREAKOUT = SkeletonSpec(
    name="volatility_breakout",
    label="Volatility Breakout (Donchian + ATR expansion filter)",
    description=(
        "N-bar breakout taken only while ATR is running materially above its own recent "
        "baseline (an expanding-volatility regime), regardless of trend direction. A "
        "distinct hypothesis from Family A: this filters by volatility STATE, not trend "
        "direction, so it can fire on genuine range-expansion moves a pure trend filter "
        "would miss or wrongly admit."
    ),
    param_grid={
        "lookback": [10, 20, 40],
        "atr_period": [14, 20],
        "expansion_mult": [1.05, 1.15],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0, 4.0],
        "max_bars": [None, 48],
    },
    build=_build_volatility_breakout,
)


# ---------------------------------------------------------------------------
# Family E: Session / Time-of-Day Effect
#   Trades an opening-range breakout, but ONLY within a specific session
#   window (e.g. the first hours after a session open). A genuinely
#   different hypothesis class from A-D: it bets that a particular clock-
#   time window has structurally different order flow (session opens,
#   overlap windows) rather than betting on any price- or volatility-based
#   pattern that could occur at any hour.
# ---------------------------------------------------------------------------

def _build_session_time_effect(p: dict) -> dict:
    start, end = p["session_start"], p["session_end"]
    return {
        "name": f"Session Effect ({start}-{end} opening-range breakout)",
        "entry_conditions": {
            "long": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("close", 1), ">",
                      {"type": "opening_range_high", "session_start": start, "session_end": end}),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("close", 1), "<",
                      {"type": "opening_range_low", "session_start": start, "session_end": end}),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [], "short": [],
            "long_connectors": [], "short_connectors": [],
        },
        "risk_management": _risk_management(
            p["stop_atr_mult"], p["target_atr_mult"],
            max_bars_in_trade=p["max_bars"],
        ),
        # A clock-time flat-by exit is essential for a session strategy --
        # without it a trade opened near the session window's own close
        # can run indefinitely into hours the hypothesis says nothing about.
        "_time_based_exit": p["flat_time"],
    }


def _apply_time_based_exit(config: dict) -> dict:
    flat_time = config.pop("_time_based_exit", None)
    if flat_time:
        config["risk_management"]["time_based_exit"] = {"enabled": True, "time": flat_time}
    return config


_SESSION_TIME_EFFECT = SkeletonSpec(
    name="session_time_effect",
    label="Session / Time-of-Day Effect (opening-range breakout, session-gated)",
    description=(
        "Bets on a specific clock-time window (e.g. a session open) having structurally "
        "different order flow than the rest of the day: takes an opening-range breakout, "
        "but ONLY within that session window, and force-flattens by a configured clock "
        "time so the trade can't run on into hours the hypothesis makes no claim about. "
        "A genuinely different edge source from A-D -- this one is about WHEN, not what "
        "price or volatility is doing."
    ),
    param_grid={
        "session_start": ["08:30", "13:30"],
        "session_end": ["10:30", "15:30"],
        "flat_time": ["16:00"],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [12, 24],
    },
    build=lambda p: _apply_time_based_exit(_build_session_time_effect(p)),
    valid=lambda p: p["session_start"] < p["session_end"],
)


# ---------------------------------------------------------------------------
# Family F: Volume-Imbalance
#   Trades in the direction of a rolling signed-volume imbalance (more
#   volume trading through up-close bars than down-close bars, or vice
#   versa) once it exceeds a threshold -- an order-flow-pressure hypothesis,
#   independent of price pattern or volatility regime. Requires the market
#   data to include a real volume column; on volume-less feeds (some FX
#   sources report tick count as a volume proxy, which still works here)
#   app.strategy.indicators.volume_delta degrades to a flat 0.0 series, so
#   this family will simply generate zero trades rather than fail loudly --
#   Stage 1's cheap filter drops zero-trade candidates on its own.
# ---------------------------------------------------------------------------

def _build_volume_imbalance(p: dict) -> dict:
    period, threshold = p["period"], p["threshold"]
    return {
        "name": f"Volume Imbalance (period={period}, thresh={threshold})",
        "entry_conditions": {
            "long": [
                _cond({"type": "volume_delta", "period": period}, ">", _val(threshold)),
                _cond(_ind("close", 1), ">", {"type": "price", "field": "open"}),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "volume_delta", "period": period}, "<", _val(-threshold)),
                _cond(_ind("close", 1), "<", {"type": "price", "field": "open"}),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond({"type": "volume_delta", "period": period}, "<", _val(0.0))],
            "short": [_cond({"type": "volume_delta", "period": period}, ">", _val(0.0))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_VOLUME_IMBALANCE = SkeletonSpec(
    name="volume_imbalance",
    label="Volume Imbalance (signed-volume pressure)",
    description=(
        "Trades in the direction of a rolling signed-volume imbalance (volume weighted "
        "toward up-close vs. down-close bars) once it clears a threshold, exiting when "
        "the imbalance flips back through zero. An order-flow-pressure hypothesis, "
        "independent of the price-pattern and volatility-regime hypotheses in the other "
        "families -- degrades to zero trades (not an error) on volume-less data."
    ),
    param_grid={
        "period": [10, 20, 40],
        "threshold": [0.15, 0.3, 0.45],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [24, 48],
    },
    build=_build_volume_imbalance,
)


# ---------------------------------------------------------------------------
# Family G: Statistical Pairs / Relative Value
#   Mean-reverts the PRIMARY instrument against a second, correlated
#   instrument's price -- a genuinely different edge source from every
#   other family here, none of which look outside the single instrument
#   being tested at all. Requires the working DataFrame to already have a
#   second instrument's close merged in as a "pair_close" column (see
#   app.data.pairs.merge_pair_series) BEFORE this family is run; the
#   engine itself stays single-instrument (see that module's docstring for
#   why), so only the PRIMARY leg is ever actually traded here -- this is
#   an honest, explicitly-scoped proxy for a full two-leg pairs trade, not
#   one, and is documented as such rather than silently pretending to
#   trade both legs.
# ---------------------------------------------------------------------------

def _build_stat_pairs(p: dict) -> dict:
    period, entry_z, exit_z = p["period"], p["entry_z"], p["exit_z"]
    return {
        "name": f"Stat Pairs Relative Value (period={period}, entry_z={entry_z})",
        "entry_conditions": {
            "long": [_cond({"type": "pair_zscore", "period": period}, "<", _val(-entry_z))],
            "short": [_cond({"type": "pair_zscore", "period": period}, ">", _val(entry_z))],
        },
        "exit_conditions": {
            "long": [_cond({"type": "pair_zscore", "period": period}, ">", _val(-exit_z))],
            "short": [_cond({"type": "pair_zscore", "period": period}, "<", _val(exit_z))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_STAT_PAIRS = SkeletonSpec(
    name="stat_pairs",
    label="Statistical Pairs / Relative Value (requires merged pair data)",
    description=(
        "Mean-reverts the primary instrument's price ratio against a second, correlated "
        "instrument once their relative-value z-score stretches past a threshold, exiting "
        "as it reverts toward zero. REQUIRES a 'pair_close' column already merged into the "
        "working data via app.data.pairs.merge_pair_series() -- only the primary instrument "
        "is actually traded (this app's engine is single-instrument), so treat this as a "
        "relative-value ENTRY FILTER on the primary leg, not a full two-leg pairs trade."
    ),
    param_grid={
        "period": [30, 50, 100],
        "entry_z": [1.5, 2.0, 2.5],
        "exit_z": [0.25, 0.5],
        "stop_atr_mult": [1.5, 2.0],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [48, 96],
    },
    build=_build_stat_pairs,
    valid=lambda p: p["exit_z"] < p["entry_z"],
    requires_pair_data=True,
)


# ---------------------------------------------------------------------------
# Family G: Liquidity Sweep Reversal
#   A standalone liquidity-hypothesis family -- distinct from Family A's
#   trend-direction breakout and Family D's volatility-state breakout,
#   this one bets purely on stop-hunt/liquidity-grab behavior: a swing
#   level is run through (triggering resting stops) and price immediately
#   reclaims it, which is read as evidence the run was liquidity-driven
#   rather than the start of a genuine breakout. Uses the same
#   `liquidity_sweep` primitive app.strategy.manual._advanced_boolean
#   already implements and app.strategy.dna already recognizes as a
#   distinct "liquidity" gene -- this family just gives that gene its own
#   dedicated, independently-searchable hypothesis instead of only ever
#   appearing as an ingredient inside a hand-built strategy.
# ---------------------------------------------------------------------------

def _build_liquidity_sweep_reversal(p: dict) -> dict:
    lookback = p["lookback"]
    return {
        "name": f"Liquidity Sweep Reversal (lookback={lookback})",
        "entry_conditions": {
            "long": [_cond({"type": "liquidity_sweep", "lookback": lookback, "direction": "bullish"}, "is true", _val(1))],
            "short": [_cond({"type": "liquidity_sweep", "lookback": lookback, "direction": "bearish"}, "is true", _val(1))],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_LIQUIDITY_SWEEP_REVERSAL = SkeletonSpec(
    name="liquidity_sweep_reversal",
    label="Liquidity Sweep Reversal (stop-hunt + reclaim)",
    description=(
        "Enters on a pure liquidity-sweep signal -- price runs through a recent swing high/low "
        "(the resting-stop level) and immediately closes back on the other side of it, read as a "
        "stop-hunt rather than a real breakout. A standalone liquidity hypothesis, independent of "
        "trend direction or volatility state, so it can be searched and scored on its own instead "
        "of only appearing as one ingredient inside a hand-built discretionary strategy."
    ),
    param_grid={
        "lookback": [10, 20, 30],
        "stop_atr_mult": [0.75, 1.0, 1.5],
        "target_atr_mult": [1.5, 2.0, 3.0],
        "max_bars": [None, 24, 48],
    },
    build=_build_liquidity_sweep_reversal,
)


# ---------------------------------------------------------------------------
# Family H: Momentum Continuation
#   Trades WITH an established momentum surge (RSI extremity confirmed by
#   a MACD histogram in agreement), rather than fading it (Family C) or
#   buying a pullback within a slower EMA trend (Family B). A genuinely
#   different edge source: this one bets that momentum which has already
#   shown up persists a while longer, the time-series-momentum effect
#   referenced in the T58 Quant Trading Masterclass material (AQR's
#   published time-series-momentum evidence across futures markets).
# ---------------------------------------------------------------------------

def _build_momentum_continuation(p: dict) -> dict:
    rsi_period, rsi_threshold = p["rsi_period"], p["rsi_threshold"]
    return {
        "name": f"Momentum Continuation (rsi{rsi_period}>{rsi_threshold}, macd hist confirm)",
        "entry_conditions": {
            "long": [
                _cond(_ind("rsi", rsi_period), ">", _val(rsi_threshold)),
                _cond({"type": "macd_histogram"}, ">", _val(0.0)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("rsi", rsi_period), "<", _val(100 - rsi_threshold)),
                _cond({"type": "macd_histogram"}, "<", _val(0.0)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("rsi", rsi_period), "<", _val(50))],
            "short": [_cond(_ind("rsi", rsi_period), ">", _val(50))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_MOMENTUM_CONTINUATION = SkeletonSpec(
    name="momentum_continuation",
    label="Momentum Continuation (RSI extremity + MACD histogram confirmation)",
    description=(
        "Trades WITH an already-established momentum surge (RSI past a threshold, confirmed by a "
        "same-direction MACD histogram) rather than fading it or waiting for a pullback -- a "
        "distinct hypothesis from the mean-reversion and trend-pullback families: that recent "
        "momentum tends to persist a while longer, not immediately revert."
    ),
    param_grid={
        "rsi_period": [7, 14],
        "rsi_threshold": [55, 60, 65],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 24],
    },
    build=_build_momentum_continuation,
)


# ---------------------------------------------------------------------------
# Family I: VWAP Reversion
#   Mean-reverts toward the session VWAP once price stretches too far from
#   it (in ATR terms), gated by an ATR-regime filter so it stands down
#   during unusually high volatility -- the T58 Quant Trading Masterclass
#   material's own warning about mean reversion ("can get destroyed
#   during trends") applied directly: don't fade a VWAP stretch that's
#   really the start of a genuine expansion move.
# ---------------------------------------------------------------------------

def _build_vwap_reversion(p: dict) -> dict:
    distance_mult, atr_period = p["distance_atr_mult"], p["atr_period"]
    return {
        "name": f"VWAP Reversion (dist={distance_mult}x ATR{atr_period})",
        "entry_conditions": {
            "long": [
                _cond(_ind("close", 1), "<", _ind("vwap", 1)),
                _cond(
                    {"type": "atr", "period": atr_period},
                    ">",
                    _val(0.0),
                ),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("close", 1), ">", _ind("vwap", 1)),
            ],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), ">", _ind("vwap", 1))],
            "short": [_cond(_ind("close", 1), "<", _ind("vwap", 1))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_VWAP_REVERSION = SkeletonSpec(
    name="vwap_reversion",
    label="VWAP Reversion (session VWAP mean reversion)",
    description=(
        "Mean-reverts toward the session VWAP once price closes on the far side of it, exiting "
        "back at VWAP -- the session-anchored counterpart to Family C's Bollinger-band reversion. "
        "Note: the visual-builder condition DSL only exposes a directional (above/below VWAP) "
        "comparison, not a raw price-minus-VWAP distance operand, so `distance_atr_mult` is "
        "reserved for a future distance-gated version rather than actually gating entries here yet "
        "-- today this trades every VWAP-side crossing, filtered only by the ATR stop/target sizing "
        "itself scaling with `atr_period`."
    ),
    param_grid={
        "distance_atr_mult": [1.0, 1.5, 2.0],
        "atr_period": [14, 20],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [None, 24],
    },
    build=_build_vwap_reversion,
)


# ---------------------------------------------------------------------------
# Family J: Market Structure Shift
#   A dedicated Change-of-Character (CHoCH) family -- distinct from Family
#   A's plain N-bar break-of-structure, CHoCH specifically requires a
#   PRIOR opposing structure (a higher-high sequence, then a break below
#   the higher-low that formed it) before treating the break as a
#   genuine structural shift rather than just any new extreme. Gives the
#   "market_structure" DNA gene (app.strategy.dna) its own dedicated,
#   independently-searchable hypothesis.
# ---------------------------------------------------------------------------

def _build_market_structure_shift(p: dict) -> dict:
    lookback = p["lookback"]
    return {
        "name": f"Market Structure Shift / CHoCH (lookback={lookback})",
        "entry_conditions": {
            "long": [_cond({"type": "change_of_character", "lookback": lookback, "direction": "bullish"}, "is true", _val(1))],
            "short": [_cond({"type": "change_of_character", "lookback": lookback, "direction": "bearish"}, "is true", _val(1))],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_MARKET_STRUCTURE_SHIFT = SkeletonSpec(
    name="market_structure_shift",
    label="Market Structure Shift (Change of Character / CHoCH)",
    description=(
        "Enters on a Change-of-Character: a prior opposing structure (e.g. a run of higher highs) "
        "breaking the swing low that formed it, read as the first evidence the structure itself has "
        "flipped -- a stricter, more specific claim than Family A's plain N-bar break-of-structure, "
        "which fires on any new N-bar extreme regardless of what structure preceded it."
    ),
    param_grid={
        "lookback": [10, 20, 30],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 48],
    },
    build=_build_market_structure_shift,
)


# ---------------------------------------------------------------------------
# Family K: Previous-Day Range Breakout
#   A daily-structural breakout: yesterday's high/low, not an intraday
#   N-bar Donchian window (Family A) or a volatility-gated one (Family D).
#   A classic "prior day's high/low" level trade -- genuinely different
#   TIMEFRAME of structure than any of the intraday-lookback families.
# ---------------------------------------------------------------------------

def _build_prev_day_range_breakout(p: dict) -> dict:
    max_bars = p["max_bars"]
    return {
        "name": "Previous-Day Range Breakout",
        "entry_conditions": {
            "long": [_cond(_ind("close", 1), "cross above", {"type": "previous_day_high"})],
            "short": [_cond(_ind("close", 1), "cross below", {"type": "previous_day_low"})],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=max_bars),
    }


_PREV_DAY_RANGE_BREAKOUT = SkeletonSpec(
    name="prev_day_range_breakout",
    label="Previous-Day Range Breakout (yesterday's high/low)",
    description=(
        "Enters the moment price crosses above yesterday's high or below yesterday's low -- a "
        "daily-STRUCTURE breakout, distinct in timeframe from every intraday N-bar lookback "
        "family here (A, D). Classic prior-session-level trade, no trend or volatility filter."
    ),
    param_grid={
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [1.5, 2.5, 3.5],
        "max_bars": [None, 48, 96],
    },
    build=_build_prev_day_range_breakout,
)


# ---------------------------------------------------------------------------
# Family L: MACD Crossover Trend
#   The MACD line crossing its own signal line, filtered by a slower EMA
#   trend so the crossover is only taken with the prevailing trend --
#   distinct from Family H (which reads MACD histogram SIGN as a momentum
#   confirmation alongside RSI extremity, not a crossover EVENT) and from
#   Family B (RSI dip/pop pullback, no MACD at all).
# ---------------------------------------------------------------------------

def _build_macd_cross_trend(p: dict) -> dict:
    ema_trend = p["ema_trend"]
    return {
        "name": f"MACD Cross Trend (ema{ema_trend} filter)",
        "entry_conditions": {
            "long": [
                _cond({"type": "macd"}, "cross above", {"type": "macd_signal"}),
                _cond(_ind("close", 1), ">", _ind("ema", ema_trend)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "macd"}, "cross below", {"type": "macd_signal"}),
                _cond(_ind("close", 1), "<", _ind("ema", ema_trend)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond({"type": "macd"}, "cross below", {"type": "macd_signal"})],
            "short": [_cond({"type": "macd"}, "cross above", {"type": "macd_signal"})],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_MACD_CROSS_TREND = SkeletonSpec(
    name="macd_cross_trend",
    label="MACD Crossover Trend (MACD/signal cross + EMA filter)",
    description=(
        "Trades the MACD line crossing its own signal line, only in the direction of a slower "
        "EMA trend filter, exiting on the opposite crossover. A crossover-EVENT momentum "
        "hypothesis, distinct from Family H's histogram-SIGN confirmation approach."
    ),
    param_grid={
        "ema_trend": [50, 100, 200],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 48],
    },
    build=_build_macd_cross_trend,
)


# ---------------------------------------------------------------------------
# Family M: Swing Structure Fade
#   Fades price AT a just-confirmed swing high/low -- buy the swing low,
#   sell the swing high -- a contrarian, structure-anchored reversal
#   distinct from every band/VWAP-based reversion family (C, I): the
#   reference point here is a confirmed price PIVOT, not a statistical
#   envelope, and distinct from Family J's CHoCH (which requires a prior
#   OPPOSING structure break, not just any confirmed pivot).
# ---------------------------------------------------------------------------

def _build_swing_structure_fade(p: dict) -> dict:
    lookback = p["lookback"]
    return {
        "name": f"Swing Structure Fade (lookback={lookback})",
        "entry_conditions": {
            "long": [_cond({"type": "swing_low", "lookback": lookback}, "is true", _val(1))],
            "short": [_cond({"type": "swing_high", "lookback": lookback}, "is true", _val(1))],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_SWING_STRUCTURE_FADE = SkeletonSpec(
    name="swing_structure_fade",
    label="Swing Structure Fade (buy confirmed swing lows, sell swing highs)",
    description=(
        "A contrarian pivot-fade: buys a just-CONFIRMED swing low, sells a just-confirmed swing "
        "high, no trend or band filter. The reference point is a real, non-lookahead-confirmed "
        "price pivot rather than a statistical band (C) or VWAP (I) -- a genuinely different "
        "reversal mechanism from either."
    ),
    param_grid={
        "lookback": [5, 10, 15],
        "stop_atr_mult": [0.75, 1.0, 1.5],
        "target_atr_mult": [1.5, 2.0, 3.0],
        "max_bars": [None, 24, 48],
    },
    build=_build_swing_structure_fade,
)


# ---------------------------------------------------------------------------
# Family N: Fair Value Gap Imbalance Continuation
#   Enters in the direction of a freshly-printed Fair Value Gap (a 3-candle
#   price imbalance / displacement), filtered by a slower EMA trend --
#   betting that a genuine displacement candle continues, not that price
#   returns to "fill" the gap. NOTE on scope: the visual-builder DSL only
#   exposes the FVG as a point-in-time boolean event, not a persisting
#   price zone/level, so a "wait for price to return and fill the gap"
#   version of this hypothesis isn't representable here -- this family is
#   honestly scoped to the continuation reading only, the same kind of
#   explicit scope note Family G (stat_pairs) already gives for its own
#   single-leg limitation.
# ---------------------------------------------------------------------------

def _build_fvg_imbalance_continuation(p: dict) -> dict:
    ema_trend = p["ema_trend"]
    return {
        "name": f"FVG Imbalance Continuation (ema{ema_trend} filter)",
        "entry_conditions": {
            "long": [
                _cond({"type": "fair_value_gap", "direction": "bullish"}, "is true", _val(1)),
                _cond(_ind("close", 1), ">", _ind("ema", ema_trend)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "fair_value_gap", "direction": "bearish"}, "is true", _val(1)),
                _cond(_ind("close", 1), "<", _ind("ema", ema_trend)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_FVG_IMBALANCE_CONTINUATION = SkeletonSpec(
    name="fvg_imbalance_continuation",
    label="Fair Value Gap Imbalance Continuation (displacement + EMA filter)",
    description=(
        "Enters in the direction of a freshly-printed Fair Value Gap (3-candle price imbalance), "
        "filtered by a slower EMA trend -- bets the displacement continues. SCOPE NOTE: the FVG "
        "primitive here is a point-in-time event, not a persisting zone, so this is honestly the "
        "continuation reading of the hypothesis, not a 'wait for the gap to fill' reversion read."
    ),
    param_grid={
        "ema_trend": [50, 100],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [None, 24, 48],
    },
    build=_build_fvg_imbalance_continuation,
)


# ---------------------------------------------------------------------------
# Family O: Order Block Reaction
#   Enters immediately on the deterministic order-block proxy already
#   implemented in app.strategy.manual._advanced_boolean (the last opposite
#   candle right before a displacement candle) -- an SMC-style structural
#   continuation trade distinct from every other family here, giving the
#   "order_block" DNA gene (app.strategy.dna) its own dedicated,
#   independently-searchable hypothesis the same way Family G already does
#   for "liquidity_sweep" and Family J does for CHoCH. SCOPE NOTE: same
#   point-in-time-event limitation as Family N -- no persisting zone/retest
#   is modeled, this is the immediate-reaction reading.
# ---------------------------------------------------------------------------

def _build_order_block_reaction(p: dict) -> dict:
    lookback = p["lookback"]
    return {
        "name": f"Order Block Reaction (lookback={lookback})",
        "entry_conditions": {
            "long": [_cond({"type": "order_block", "lookback": lookback, "direction": "bullish"}, "is true", _val(1))],
            "short": [_cond({"type": "order_block", "lookback": lookback, "direction": "bearish"}, "is true", _val(1))],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_ORDER_BLOCK_REACTION = SkeletonSpec(
    name="order_block_reaction",
    label="Order Block Reaction (SMC displacement-origin candle)",
    description=(
        "Enters on the deterministic order-block proxy: the last opposite candle immediately "
        "preceding a displacement move. Gives the 'order_block' DNA gene its own dedicated, "
        "independently-searchable hypothesis rather than only ever appearing as one ingredient "
        "inside a hand-built discretionary strategy. SCOPE NOTE: an immediate-reaction reading, "
        "not a 'wait for price to retest the zone' version -- see Family N's same caveat."
    ),
    param_grid={
        "lookback": [10, 20, 30],
        "stop_atr_mult": [0.75, 1.0, 1.5],
        "target_atr_mult": [1.5, 2.0, 3.0],
        "max_bars": [None, 24, 48],
    },
    build=_build_order_block_reaction,
)


# ---------------------------------------------------------------------------
# Family P: Volatility Contraction Squeeze Breakout
#   The mirror-image bet of Family D: instead of requiring volatility to
#   already be EXPANDING at the breakout bar, this requires it to still be
#   CONTRACTED (a "squeeze") relative to its own baseline -- catching a
#   breakout AS the regime is turning, not after it has already turned.
#   Expect a much lower trade count than Family D by construction (most
#   real squeezes resolve into an expansion bar, which then no longer
#   reads as contracted) -- Stage 1's cheap filter will simply drop this
#   family if it's too rare on a given dataset, same as every other
#   thin-signal family here.
# ---------------------------------------------------------------------------

def _build_volatility_contraction_squeeze(p: dict) -> dict:
    lookback, atr_period, contraction_mult = p["lookback"], p["atr_period"], p["contraction_mult"]
    return {
        "name": f"Volatility Squeeze Breakout (lb={lookback}, atr{atr_period} x{contraction_mult} contraction)",
        "entry_conditions": {
            "long": [
                _cond(_breakout_flag(lookback, "bullish"), "is true", _val(1)),
                _cond({"type": "atr_regime", "period": atr_period, "contraction_mult": contraction_mult}, "==", _val(-1)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_breakout_flag(lookback, "bearish"), "is true", _val(1)),
                _cond({"type": "atr_regime", "period": atr_period, "contraction_mult": contraction_mult}, "==", _val(-1)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p.get("max_bars")),
    }


_VOLATILITY_CONTRACTION_SQUEEZE = SkeletonSpec(
    name="volatility_contraction_squeeze",
    label="Volatility Squeeze Breakout (Donchian + ATR contraction filter)",
    description=(
        "The mirror image of Family D: an N-bar breakout taken only while ATR is STILL running "
        "materially below its own recent baseline (a squeeze), catching a breakout as the "
        "volatility regime turns rather than after it already has. Expect materially fewer "
        "trades than Family D by construction -- that's the hypothesis being tested, not a bug."
    ),
    param_grid={
        "lookback": [10, 20, 40],
        "atr_period": [14, 20],
        "contraction_mult": [0.75, 0.85],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 48],
    },
    build=_build_volatility_contraction_squeeze,
)


# ---------------------------------------------------------------------------
# Family Q: Overnight / Session-Open Gap Fade
#   Fades the gap between the current session-open price and the PRIOR
#   day's close, only within a short session-open window -- a classic gap-
#   fade mean-reversion, anchored to the previous close rather than a
#   Bollinger band (C) or VWAP (I), and gated by clock time rather than by
#   any price or volatility state (E's session family is an opening-range
#   BREAKOUT, the opposite direction bet, at the same kind of session
#   anchor).
# ---------------------------------------------------------------------------

def _build_overnight_gap_fade(p: dict) -> dict:
    start, end = p["session_start"], p["session_end"]
    return {
        "name": f"Overnight Gap Fade ({start}-{end} vs prior close)",
        "entry_conditions": {
            "long": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("close", 1), "<", {"type": "previous_day_close"}),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("close", 1), ">", {"type": "previous_day_close"}),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), ">", {"type": "previous_day_close"})],
            "short": [_cond(_ind("close", 1), "<", {"type": "previous_day_close"})],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
        "_time_based_exit": p["flat_time"],
    }


_OVERNIGHT_GAP_FADE = SkeletonSpec(
    name="overnight_gap_fade",
    label="Overnight / Session-Open Gap Fade (fade toward prior close)",
    description=(
        "Within a short session-open window, fades whichever side of the prior day's close price "
        "opened on, targeting a reversion back to that prior close, force-flattened by a clock "
        "time. Anchored to yesterday's CLOSE (not a band or VWAP), and a fade rather than E's "
        "opening-range BREAKOUT at the same kind of session anchor -- opposite direction bet."
    ),
    param_grid={
        "session_start": ["08:30", "13:30"],
        "session_end": ["09:00", "14:00"],
        "flat_time": ["16:00"],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.0],
        "max_bars": [12, 24],
    },
    build=lambda p: _apply_time_based_exit(_build_overnight_gap_fade(p)),
    valid=lambda p: p["session_start"] < p["session_end"],
)


# ---------------------------------------------------------------------------
# Family R: WMA Ribbon Trend Alignment
# ---------------------------------------------------------------------------

def _build_wma_ribbon_trend(p: dict) -> dict:
    fast, mid, slow = p["wma_fast"], p["wma_mid"], p["wma_slow"]
    return {
        "name": f"WMA Ribbon Trend (wma {fast}/{mid}/{slow})",
        "entry_conditions": {
            "long": [
                _cond(_ind("wma", fast), "cross above", _ind("wma", mid)),
                _cond(_ind("wma", mid), ">", _ind("wma", slow)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("wma", fast), "cross below", _ind("wma", mid)),
                _cond(_ind("wma", mid), "<", _ind("wma", slow)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("wma", fast), "cross below", _ind("wma", mid))],
            "short": [_cond(_ind("wma", fast), "cross above", _ind("wma", mid))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_WMA_RIBBON_TREND = SkeletonSpec(
    name="wma_ribbon_trend",
    label="WMA Ribbon Trend (3-MA stacked alignment, WMA)",
    description=(
        "Enters when a fast WMA crosses its own mid WMA WHILE the mid WMA already leads the slow "
        "WMA (a 3-MA stack aligning), exiting on the fast/mid cross reversing. Uses Weighted "
        "Moving Averages, not used by any EMA-based family here (A, B) -- a distinct smoothing "
        "characteristic (WMA reacts faster to recent bars than EMA at the same period)."
    ),
    param_grid={
        "wma_fast": [10, 20],
        "wma_mid": [30, 50],
        "wma_slow": [100, 150],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 48],
    },
    build=_build_wma_ribbon_trend,
    valid=lambda p: p["wma_fast"] < p["wma_mid"] < p["wma_slow"],
)


# ---------------------------------------------------------------------------
# Family S: Percentage-Change Momentum Burst
#   Trades a raw N-bar percentage-change ignition once it clears a
#   threshold -- a pure rate-of-change momentum-ignition hypothesis, using
#   neither RSI nor MACD (H, L) nor a price-structure breakout (A, D, K).
# ---------------------------------------------------------------------------

def _build_pct_change_momentum_burst(p: dict) -> dict:
    period, threshold = p["period"], p["threshold_pct"]
    return {
        "name": f"Pct-Change Momentum Burst (period={period}, thresh={threshold}%)",
        "entry_conditions": {
            "long": [_cond({"type": "percentage_change", "period": period}, ">", _val(threshold))],
            "short": [_cond({"type": "percentage_change", "period": period}, "<", _val(-threshold))],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_PCT_CHANGE_MOMENTUM_BURST = SkeletonSpec(
    name="pct_change_momentum_burst",
    label="Percentage-Change Momentum Burst (raw rate-of-change ignition)",
    description=(
        "Enters the instant an N-bar percentage price change clears a threshold in either "
        "direction -- a pure raw rate-of-change ignition hypothesis, using neither RSI/MACD (the "
        "momentum-continuation families) nor a price-structure breakout (the Donchian families). "
        "Simplest possible momentum-ignition bet, included as a baseline the other, more elaborate "
        "momentum families should be able to beat."
    ),
    param_grid={
        "period": [5, 10, 20],
        "threshold_pct": [0.5, 1.0, 1.5],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [None, 24],
    },
    build=_build_pct_change_momentum_burst,
)


# ---------------------------------------------------------------------------
# Family T: Pure RSI Extreme Reversion
#   Fades RSI extremity alone, with NO band/channel filter at all -- the
#   simplest possible mean-reversion hypothesis, included as a baseline
#   Family C's (Bollinger + RSI) more elaborate version should be able to
#   beat. Distinct from every other RSI-using family here: B uses RSI as
#   a PULLBACK-within-trend filter, H uses RSI as a momentum-persistence
#   confirmation -- this is the only family that trades RSI extremity on
#   its own, as a standalone reversal signal.
# ---------------------------------------------------------------------------

def _build_rsi_extreme_reversion(p: dict) -> dict:
    rsi_period, oversold = p["rsi_period"], p["oversold"]
    overbought = 100 - oversold
    return {
        "name": f"Pure RSI Extreme Reversion (rsi{rsi_period}, {oversold}/{overbought})",
        "entry_conditions": {
            "long": [_cond(_ind("rsi", rsi_period), "<", _val(oversold))],
            "short": [_cond(_ind("rsi", rsi_period), ">", _val(overbought))],
        },
        "exit_conditions": {
            "long": [_cond(_ind("rsi", rsi_period), ">", _val(50))],
            "short": [_cond(_ind("rsi", rsi_period), "<", _val(50))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_RSI_EXTREME_REVERSION = SkeletonSpec(
    name="rsi_extreme_reversion",
    label="Pure RSI Extreme Reversion (RSI alone, no band/channel filter)",
    description=(
        "Fades RSI extremity alone -- no Bollinger band (C), no channel, no trend filter -- "
        "exiting back at RSI 50. The simplest possible mean-reversion hypothesis, included as a "
        "baseline Family C's more elaborate dual Bollinger+RSI gate should be able to beat. The "
        "only family here trading RSI extremity as a standalone reversal signal rather than a "
        "pullback filter (B) or a momentum confirmation (H)."
    ),
    param_grid={
        "rsi_period": [7, 14, 21],
        "oversold": [20, 25, 30],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.0],
        "max_bars": [None, 24, 48],
    },
    build=_build_rsi_extreme_reversion,
)


# ---------------------------------------------------------------------------
# Families U-Z: Prop-Eval-Shaped Scalp Families
#
# Every family above (A-T) was hand-designed around a general trading
# hypothesis (trend, momentum, mean reversion, structure, liquidity...) and
# then given SOME ATR stop/target -- but the stop/target shape those
# families use (target_atr_mult usually 1.5-4x a 1-2x stop, i.e. "let a
# winner run") is a general-purpose trading shape, not a shape built for
# what a prop-firm evaluation actually needs: a HIGH win rate, a TIGHT
# risk/reward, and a FAST time-to-target, so the account can clear its
# profit target before a bad losing streak has enough room to hit the
# daily-loss/max-drawdown limit first. A strategy can have a perfectly
# real, positive-expectancy edge and still be a poor prop-eval fit if it
# needs 40 bars and a 3:1 winner to pay for four losers along the way.
#
# These six families reuse the exact same manual-condition primitives as
# A-T (liquidity sweep, session/opening-range levels, CHoCH, fair value
# gap, EMA trend, ATR/volatility regime) but are deliberately built the
# other way around: pick (or add) a confirmation filter that raises the
# entry's win probability, THEN size the exit for a tight target (often
# target_atr_mult < stop_atr_mult -- a sub-1.0 reward:risk that needs a
# high win rate to be worth taking at all) and a short max_bars_in_trade,
# so a candidate either proves out fast or gets cut loose fast rather than
# tying up eval capital for dozens of bars either way. None of this is a
# guarantee any of the six actually clears Stage 3 -- that's exactly what
# Search Lab/Evolution Lab are for -- but it means the search space itself
# finally contains hypotheses shaped for the objective Owen is actually
# scoring against (eval_pass_probability), not just more variations on
# hypotheses shaped for raw net profit.
# ---------------------------------------------------------------------------

# Family U: Liquidity Sweep Quick Reclaim
#   Family G's stop-hunt-and-reclaim signal, re-shaped for speed: a
#   shorter lookback (fires more often, on smaller/faster sweeps) and a
#   sub-1.0 reward:risk with a short max_bars_in_trade, instead of Family
#   G's wider 1.5-3x target meant to let a reclaim run.

def _build_liquidity_sweep_quick_reclaim(p: dict) -> dict:
    lookback = p["lookback"]
    return {
        "name": f"Liquidity Sweep Quick Reclaim (lookback={lookback})",
        "entry_conditions": {
            "long": [_cond({"type": "liquidity_sweep", "lookback": lookback, "direction": "bullish"}, "is true", _val(1))],
            "short": [_cond({"type": "liquidity_sweep", "lookback": lookback, "direction": "bearish"}, "is true", _val(1))],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_LIQUIDITY_SWEEP_QUICK_RECLAIM = SkeletonSpec(
    name="liquidity_sweep_quick_reclaim",
    label="Liquidity Sweep Quick Reclaim (fast scalp variant)",
    description=(
        "The prop-eval-shaped sibling of Family G: the same stop-hunt-and-reclaim signal on a "
        "shorter, more frequent lookback, but with a sub-1.0 reward:risk (a tight target relative "
        "to the stop) and a short max_bars_in_trade cap -- built to prove out or fail fast rather "
        "than sizing for a large winning move."
    ),
    param_grid={
        "lookback": [5, 8, 12],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [0.5, 0.75],
        "max_bars": [4, 8, 12],
    },
    build=_build_liquidity_sweep_quick_reclaim,
)


# Family V: Range Midpoint Fade
#   Fades yesterday's high/low back toward the middle of the established
#   range, but ONLY while an ATR-regime filter confirms the market is
#   currently CONTRACTED (calm/range-bound) -- a range fade taken during
#   an expanding/trending regime is exactly the failure mode the T58 Quant
#   Trading Masterclass material warns mean reversion is prone to. Gating
#   on regime is what raises this fade's win probability enough to be
#   worth the tight target below.

def _build_range_midpoint_fade(p: dict) -> dict:
    atr_period = p["atr_period"]
    return {
        "name": f"Range Midpoint Fade (atr{atr_period} contraction-gated)",
        "entry_conditions": {
            "long": [
                _cond(_ind("close", 1), "<", {"type": "previous_day_low"}),
                _cond({"type": "atr_regime", "period": atr_period}, "<=", _val(0)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("close", 1), ">", {"type": "previous_day_high"}),
                _cond({"type": "atr_regime", "period": atr_period}, "<=", _val(0)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_RANGE_MIDPOINT_FADE = SkeletonSpec(
    name="range_midpoint_fade",
    label="Range Midpoint Fade (contraction-gated mean reversion scalp)",
    description=(
        "Fades a close beyond yesterday's high/low back toward the range, but stands down whenever "
        "an ATR-regime filter reads the market as currently EXPANDING (only trades a calm/neutral "
        "regime, atr_regime <= 0) -- unlike Family C's always-on Bollinger fade, this one skips the "
        "volatility regime mean reversion is most likely to get run over in. Tight target, short "
        "max_bars: built to fade quiet-range noise quickly, not to catch a big reversal."
    ),
    param_grid={
        "atr_period": [14, 20],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [0.6, 0.9],
        "max_bars": [6, 12, 16],
    },
    build=_build_range_midpoint_fade,
)


# Family W: Opening Range Retest Confirmation
#   Family E's opening-range breakout, but ENTRY REQUIRES A RETEST: the
#   triggering bar must have traded back down to (long) / up to (short)
#   the opening-range level and still closed beyond it, rather than firing
#   on the very first breakout bar. A retest-confirmed breakout is a
#   textbook way to filter out the false breakouts that are a big part of
#   why naive breakout entries have a lower win rate -- this family
#   accepts fewer, later signals in exchange for a meaningfully higher
#   probability of being right, then exploits that quickly with a tight
#   target instead of letting Family E's wider target ride.

def _build_opening_range_retest_confirmation(p: dict) -> dict:
    start, end = p["session_start"], p["session_end"]
    return {
        "name": f"Opening Range Retest Confirmation ({start}-{end})",
        "entry_conditions": {
            "long": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("close", 1), ">",
                      {"type": "opening_range_high", "session_start": start, "session_end": end}),
                _cond(_ind("low", 1), "<=",
                      {"type": "opening_range_high", "session_start": start, "session_end": end}),
            ],
            "long_connectors": ["AND", "AND"],
            "short": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("close", 1), "<",
                      {"type": "opening_range_low", "session_start": start, "session_end": end}),
                _cond(_ind("high", 1), ">=",
                      {"type": "opening_range_low", "session_start": start, "session_end": end}),
            ],
            "short_connectors": ["AND", "AND"],
        },
        "exit_conditions": {"long": [], "short": [], "long_connectors": [], "short_connectors": []},
        "risk_management": _risk_management(
            p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"],
        ),
        "_time_based_exit": p["flat_time"],
    }


_OPENING_RANGE_RETEST_CONFIRMATION = SkeletonSpec(
    name="opening_range_retest_confirmation",
    label="Opening Range Retest Confirmation (confirmation-gated session breakout scalp)",
    description=(
        "Family E's opening-range breakout, but the entry bar must also have traded back to the "
        "opening-range level and still closed beyond it -- a retest-and-hold, not the first-touch "
        "breakout. Fewer signals than Family E, but each one has already survived one immediate "
        "failure test, which is exactly the mechanism that raises a breakout entry's win rate. "
        "Tight target and short max_bars exploit that confirmation quickly instead of sizing for "
        "a large post-breakout move the way Family E does."
    ),
    param_grid={
        "session_start": ["08:30", "13:30"],
        "session_end": ["10:30", "15:30"],
        "flat_time": ["16:00"],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [0.75, 1.0],
        "max_bars": [6, 12],
    },
    build=lambda p: _apply_time_based_exit(_build_opening_range_retest_confirmation(p)),
    valid=lambda p: p["session_start"] < p["session_end"],
)


# Family X: Micro-Pullback Continuation Scalp
#   A one-bar-dip continuation, deliberately smaller-scope than Family B's
#   RSI-pullback continuation: inside an established EMA trend, buy the
#   very next bar after a single counter-trend (red, in an uptrend) candle,
#   betting on an immediate one-to-a-few-bar resumption rather than
#   waiting out a full RSI pullback/pop cycle. Trades far more often than
#   Family B, each one sized for a small, fast target.

def _build_micro_pullback_continuation(p: dict) -> dict:
    ema_fast, ema_slow = p["ema_fast"], p["ema_slow"]
    return {
        "name": f"Micro-Pullback Continuation (ema {ema_fast}/{ema_slow})",
        "entry_conditions": {
            "long": [
                _cond(_ind("ema", ema_fast), ">", _ind("ema", ema_slow)),
                _cond({"type": "candle_direction", "direction": "bearish"}, "is true", _val(1)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("ema", ema_fast), "<", _ind("ema", ema_slow)),
                _cond({"type": "candle_direction", "direction": "bullish"}, "is true", _val(1)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_MICRO_PULLBACK_CONTINUATION = SkeletonSpec(
    name="micro_pullback_continuation",
    label="Micro-Pullback Continuation (one-bar-dip trend scalp)",
    description=(
        "Buys the single bar after one counter-trend candle inside an established EMA uptrend "
        "(sells the mirror case in a downtrend), betting on immediate resumption rather than "
        "waiting for a full RSI pullback cycle like Family B does. Trades much more often, each "
        "one sized as a small, fast scalp rather than a swing continuation trade."
    ),
    param_grid={
        "ema_fast": [20, 50],
        "ema_slow": [100, 200],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [0.5, 0.8],
        "max_bars": [4, 8, 10],
    },
    build=_build_micro_pullback_continuation,
    valid=lambda p: p["ema_fast"] < p["ema_slow"],
)


# Family Y: Change-of-Character Reversal Scalp
#   Family J's CHoCH signal, gated by a volatility-regime filter requiring
#   CONTRACTION at the moment of the shift -- a structural reversal that
#   shows up during a quiet regime is read as more likely a genuine,
#   orderly rotation than one appearing mid-expansion (which is more often
#   just chop). Tight target and short max_bars exploit the early read
#   quickly rather than sizing for a full structural reversal the way
#   Family J does.

def _build_change_of_character_reversal_scalp(p: dict) -> dict:
    lookback, vol_period = p["lookback"], p["vol_period"]
    return {
        "name": f"CHoCH Reversal Scalp (lookback={lookback}, contraction-gated)",
        "entry_conditions": {
            "long": [
                _cond({"type": "change_of_character", "lookback": lookback, "direction": "bullish"}, "is true", _val(1)),
                _cond({"type": "volatility_regime", "period": vol_period}, "<=", _val(0)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "change_of_character", "lookback": lookback, "direction": "bearish"}, "is true", _val(1)),
                _cond({"type": "volatility_regime", "period": vol_period}, "<=", _val(0)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_CHANGE_OF_CHARACTER_REVERSAL_SCALP = SkeletonSpec(
    name="change_of_character_reversal_scalp",
    label="Change of Character Reversal Scalp (contraction-gated CHoCH)",
    description=(
        "Family J's Change-of-Character signal, but stands down whenever a volatility-regime "
        "filter reads the market as currently EXPANDING (only trades a calm/neutral regime, "
        "volatility_regime <= 0) -- a structural shift during a quiet regime reads as more likely "
        "a genuine rotation than the same signal firing mid-expansion, which is more often chop. "
        "Tight target, short max_bars: exploits the early read quickly instead of sizing for a "
        "full reversal like Family J does."
    ),
    param_grid={
        "lookback": [10, 20],
        "vol_period": [14, 20],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [0.75, 1.0],
        "max_bars": [6, 12, 16],
    },
    build=_build_change_of_character_reversal_scalp,
)


# Family Z: Fair Value Gap Quick-Fill Fade
#   The mirror-image hypothesis of Family N: instead of betting a fresh
#   Fair Value Gap's displacement continues, this fades INTO the gap,
#   betting the imbalance gets at least partially filled -- a textbook
#   "gaps get filled" mean-reversion read rather than Family N's
#   continuation read of the exact same primitive. A very short-lived
#   setup by construction (a gap either fills within a handful of bars or
#   the continuation reading was right instead), hence the shortest
#   max_bars_in_trade of any family here.

def _build_fvg_quick_fill_fade(p: dict) -> dict:
    return {
        "name": "FVG Quick-Fill Fade",
        "entry_conditions": {
            # Fade a BEARISH fvg (gap down) expecting a bounce back up to fill it.
            "long": [_cond({"type": "fair_value_gap", "direction": "bearish"}, "is true", _val(1))],
            # Fade a BULLISH fvg (gap up) expecting a pullback down to fill it.
            "short": [_cond({"type": "fair_value_gap", "direction": "bullish"}, "is true", _val(1))],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_FVG_QUICK_FILL_FADE = SkeletonSpec(
    name="fvg_quick_fill_fade",
    label="Fair Value Gap Quick-Fill Fade (imbalance mean-reversion scalp)",
    description=(
        "Fades INTO a freshly-printed Fair Value Gap, betting the imbalance gets at least "
        "partially filled -- the mean-reversion reading of the same primitive Family N reads as a "
        "continuation signal. A gap either fills within a handful of bars or this hypothesis was "
        "wrong, so this family runs the shortest max_bars_in_trade of any family in this module."
    ),
    param_grid={
        "stop_atr_mult": [1.0, 1.3],
        "target_atr_mult": [0.5, 0.8],
        "max_bars": [4, 6, 10],
    },
    build=_build_fvg_quick_fill_fade,
)


# ---------------------------------------------------------------------------
# Expansion round 2 (Sep 2026, requested batch): 8 more named hypothesis
# families, chosen specifically to cover ground the first 26 didn't --
# either a canonical taxonomy group with ZERO existing family
# (relative_strength) or the deliberate MIRROR IMAGE of an existing
# hypothesis (trading the same primitive the opposite direction), so the
# Search Lab's coverage of app.strategy.family_taxonomy.FAMILY_GROUPS
# stops being lopsided toward mean-reversion/breakout and starts actually
# spanning the full taxonomy. Every comparison below follows this file's
# existing scale-safety convention: a raw price/ATR-scale indicator is
# only ever compared to ANOTHER price/ATR-scale indicator, never to a
# hardcoded absolute number (that's exactly the FX-pip-vs-gold-price bug
# class root-caused and fixed in app/backtest/execution.py -- see
# /areas/prop-algo-backtester.md) -- only genuinely dimensionless reads
# (RSI 0-100, a z-score, relative_volume as a ratio, a regime flag) are
# ever compared to a plain number.
# ---------------------------------------------------------------------------


def _build_relative_strength_momentum(p: dict) -> dict:
    period, entry_z, exit_z = p["period"], p["entry_z"], p["exit_z"]
    return {
        "name": f"Relative Strength Momentum (period={period}, entry_z={entry_z})",
        "entry_conditions": {
            "long": [_cond({"type": "pair_zscore", "period": period}, ">", _val(entry_z))],
            "short": [_cond({"type": "pair_zscore", "period": period}, "<", _val(-entry_z))],
        },
        "exit_conditions": {
            "long": [_cond({"type": "pair_zscore", "period": period}, "<", _val(exit_z))],
            "short": [_cond({"type": "pair_zscore", "period": period}, ">", _val(-exit_z))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_RELATIVE_STRENGTH_MOMENTUM = SkeletonSpec(
    name="relative_strength_momentum",
    label="Relative Strength Ratio Momentum (requires merged pair data)",
    description=(
        "The deliberate MIRROR IMAGE of Family G (stat_pairs): instead of fading a stretched "
        "relative-value z-score back toward zero, this BUYS the primary instrument once its "
        "ratio to a second instrument has ALREADY stretched significantly positive (betting "
        "it keeps outperforming) and shorts once the ratio has stretched significantly "
        "negative -- a relative-strength MOMENTUM hypothesis rather than a mean-reversion one, "
        "on the exact same pair_zscore primitive with the entry direction inverted. Fills the "
        "'relative_strength' taxonomy group, which previously had zero families in it. Same "
        "single-leg execution caveat as stat_pairs: only the primary instrument is traded."
    ),
    param_grid={
        "period": [30, 50, 100],
        "entry_z": [1.5, 2.0, 2.5],
        "exit_z": [0.5, 0.75],
        "stop_atr_mult": [1.5, 2.0],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [48, 96],
    },
    build=_build_relative_strength_momentum,
    valid=lambda p: p["exit_z"] < p["entry_z"],
    requires_pair_data=True,
)


def _build_volume_climax_reversal(p: dict) -> dict:
    vol_period, vol_mult, atr_period, max_bars = p["vol_period"], p["vol_mult"], p["atr_period"], p["max_bars"]
    return {
        "name": f"Volume Climax Reversal (rel-vol{vol_period}>{vol_mult}x, vol-expansion gated)",
        "entry_conditions": {
            # A down-close candle on climactic volume, during a volatility-EXPANSION
            # regime, reads as capitulation/exhaustion selling rather than the start of a
            # sustained new leg down -- fade it long. Mirrored for shorts.
            "long": [
                _cond(_ind("relative_volume", vol_period), ">", _val(vol_mult)),
                _cond({"type": "volatility_regime", "period": atr_period}, "==", _val(1)),
                _cond({"type": "candle_direction", "direction": "bearish"}, "is true", _val(1)),
            ],
            "long_connectors": ["AND", "AND"],
            "short": [
                _cond(_ind("relative_volume", vol_period), ">", _val(vol_mult)),
                _cond({"type": "volatility_regime", "period": atr_period}, "==", _val(1)),
                _cond({"type": "candle_direction", "direction": "bullish"}, "is true", _val(1)),
            ],
            "short_connectors": ["AND", "AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=max_bars),
    }


_VOLUME_CLIMAX_REVERSAL = SkeletonSpec(
    name="volume_climax_reversal",
    label="Volume Climax Exhaustion Reversal (relative-volume spike fade)",
    description=(
        "Fades a single candle that printed on climactic relative volume (a real multiple of "
        "its own recent average -- a dimensionless ratio, not a raw volume count) during a "
        "volatility-expansion regime, betting the move was exhaustion/capitulation rather than "
        "the start of a sustained trend. A genuinely different reversal trigger from Family C "
        "(a statistical band) or Family M (a confirmed swing pivot) -- this one fires on ORDER-"
        "FLOW INTENSITY at a single bar, nothing else."
    ),
    param_grid={
        "vol_period": [14, 20],
        "vol_mult": [2.0, 3.0],
        "atr_period": [14, 20],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [12, 24],
    },
    build=_build_volume_climax_reversal,
)


def _build_vwap_trend_continuation(p: dict) -> dict:
    ema_fast, ema_slow = p["ema_fast"], p["ema_slow"]
    rsi_period, rsi_pullback_low, rsi_pullback_high = p["rsi_period"], p["rsi_pullback_low"], p["rsi_pullback_high"]
    max_bars = p["max_bars"]
    return {
        "name": f"VWAP Trend Continuation (ema {ema_fast}/{ema_slow}, rsi{rsi_period})",
        "entry_conditions": {
            "long": [
                _cond(_ind("ema", ema_fast), ">", _ind("ema", ema_slow)),
                _cond(_ind("close", 1), ">", _ind("vwap", 1)),
                _cond(_ind("rsi", rsi_period), "<", _val(rsi_pullback_low)),
            ],
            "long_connectors": ["AND", "AND"],
            "short": [
                _cond(_ind("ema", ema_fast), "<", _ind("ema", ema_slow)),
                _cond(_ind("close", 1), "<", _ind("vwap", 1)),
                _cond(_ind("rsi", rsi_period), ">", _val(rsi_pullback_high)),
            ],
            "short_connectors": ["AND", "AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("rsi", rsi_period), ">", _val(rsi_pullback_high))],
            "short": [_cond(_ind("rsi", rsi_period), "<", _val(rsi_pullback_low))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=max_bars),
    }


_VWAP_TREND_CONTINUATION = SkeletonSpec(
    name="vwap_trend_continuation",
    label="VWAP Trend Continuation (stay-above-VWAP pullback buy)",
    description=(
        "The deliberate MIRROR IMAGE of Family I (vwap_reversion): instead of fading a stretch "
        "AWAY from VWAP, this buys a shallow RSI pullback WHILE price is still holding above "
        "VWAP inside an established EMA uptrend (mirrored for downtrends/shorts) -- a "
        "continuation reading of the same anchor price, not a mean-reversion one."
    ),
    param_grid={
        "ema_fast": [20, 50],
        "ema_slow": [100, 200],
        "rsi_period": [7, 14],
        "rsi_pullback_low": [35, 45],
        "rsi_pullback_high": [55, 65],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [24, 48],
    },
    build=_build_vwap_trend_continuation,
    valid=lambda p: p["ema_fast"] < p["ema_slow"] and p["rsi_pullback_low"] < p["rsi_pullback_high"],
)


def _build_bollinger_band_walk(p: dict) -> dict:
    bb_period, ema_trend, max_bars = p["bb_period"], p["ema_trend"], p["max_bars"]
    return {
        "name": f"Bollinger Band Walk Continuation (bb{bb_period}, ema{ema_trend} filter)",
        "entry_conditions": {
            "long": [
                _cond(_ind("close", 1), ">", {"type": "bollinger_upper", "period": bb_period, "field": "close"}),
                _cond(_ind("close", 1), ">", _ind("ema", ema_trend)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("close", 1), "<", {"type": "bollinger_lower", "period": bb_period, "field": "close"}),
                _cond(_ind("close", 1), "<", _ind("ema", ema_trend)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), "<", {"type": "bollinger_mid", "period": bb_period, "field": "close"})],
            "short": [_cond(_ind("close", 1), ">", {"type": "bollinger_mid", "period": bb_period, "field": "close"})],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=max_bars),
    }


_BOLLINGER_BAND_WALK = SkeletonSpec(
    name="bollinger_band_walk_continuation",
    label="Bollinger Band Walk Continuation (trend-following band ride)",
    description=(
        "The deliberate MIRROR IMAGE of Family C (mean_reversion_band): instead of fading a "
        "close outside the Bollinger Band, this TRADES WITH a close that's already outside it, "
        "provided a slower EMA trend filter agrees -- the classic 'band walk' reading of the "
        "same statistical envelope, betting the move keeps riding the band rather than "
        "reverting. Exits back at the midline either way, same as Family C's exit."
    ),
    param_grid={
        "bb_period": [14, 20, 30],
        "ema_trend": [50, 100],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 48],
    },
    build=_build_bollinger_band_walk,
)


def _build_gap_and_go(p: dict) -> dict:
    start, end, flat_time, max_bars = p["session_start"], p["session_end"], p["flat_time"], p["max_bars"]
    return {
        "name": f"Gap-and-Go Continuation ({start}-{end})",
        "entry_conditions": {
            "long": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("open", 1), ">", {"type": "previous_day_close"}),
                _cond(_ind("close", 1), ">", _ind("open", 1)),
            ],
            "long_connectors": ["AND", "AND"],
            "short": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("open", 1), "<", {"type": "previous_day_close"}),
                _cond(_ind("close", 1), "<", _ind("open", 1)),
            ],
            "short_connectors": ["AND", "AND"],
        },
        "exit_conditions": {"long": [], "short": [], "long_connectors": [], "short_connectors": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=max_bars),
        "_time_based_exit": flat_time,
    }


_GAP_AND_GO_CONTINUATION = SkeletonSpec(
    name="gap_and_go_continuation",
    label="Gap-and-Go Continuation (session-gated, flat-by-clock-time)",
    description=(
        "The deliberate MIRROR IMAGE of Family K (overnight_gap_fade): instead of fading an "
        "overnight gap back toward yesterday's close, this trades WITH the gap, entering only "
        "within an early session window and only if the opening candle itself hasn't already "
        "reversed direction -- betting a gap that's holding is more likely to run than fill. "
        "Force-flattens by clock time, same discipline as Family E."
    ),
    param_grid={
        "session_start": ["08:30", "13:30"],
        "session_end": ["09:30", "14:30"],
        "flat_time": ["16:00"],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [12, 24],
    },
    build=lambda p: _apply_time_based_exit(_build_gap_and_go(p)),
    valid=lambda p: p["session_start"] < p["session_end"],
)


def _build_volume_confirmed_breakout(p: dict) -> dict:
    lookback, vol_period, vol_mult, max_bars = p["lookback"], p["vol_period"], p["vol_mult"], p["max_bars"]
    return {
        "name": f"Volume-Confirmed Breakout (lb={lookback}, rel-vol{vol_period}>{vol_mult}x)",
        "entry_conditions": {
            "long": [
                _cond(_breakout_flag(lookback, "bullish"), "is true", _val(1)),
                _cond(_ind("relative_volume", vol_period), ">", _val(vol_mult)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_breakout_flag(lookback, "bearish"), "is true", _val(1)),
                _cond(_ind("relative_volume", vol_period), ">", _val(vol_mult)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=max_bars),
    }


_VOLUME_CONFIRMED_BREAKOUT = SkeletonSpec(
    name="volume_confirmed_breakout",
    label="Volume-Confirmed Breakout (Donchian + relative-volume filter)",
    description=(
        "An N-bar breakout gated by relative volume (a real multiple of its own recent "
        "average) instead of Family A's trend-direction (EMA) filter or Family D's ATR-state "
        "filter -- a third, independent way to separate a 'real' breakout from a false one: "
        "conviction read from ORDER FLOW rather than price direction or volatility state."
    ),
    param_grid={
        "lookback": [10, 20, 40],
        "vol_period": [14, 20],
        "vol_mult": [1.5, 2.0],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 48],
    },
    build=_build_volume_confirmed_breakout,
)


def _build_wma_sma_divergence_trend(p: dict) -> dict:
    period, ema_trend, max_bars = p["period"], p["ema_trend"], p["max_bars"]
    return {
        "name": f"WMA/SMA Divergence Trend (period={period}, ema{ema_trend} filter)",
        "entry_conditions": {
            "long": [
                _cond(_ind("wma", period), ">", _ind("sma", period)),
                _cond(_ind("close", 1), ">", _ind("ema", ema_trend)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("wma", period), "<", _ind("sma", period)),
                _cond(_ind("close", 1), "<", _ind("ema", ema_trend)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("wma", period), "<", _ind("sma", period))],
            "short": [_cond(_ind("wma", period), ">", _ind("sma", period))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=max_bars),
    }


_WMA_SMA_DIVERGENCE_TREND = SkeletonSpec(
    name="wma_sma_divergence_trend",
    label="WMA/SMA Divergence Trend (recency-weighted acceleration filter)",
    description=(
        "Compares a recency-weighted moving average (WMA) against a plain equal-weighted one "
        "(SMA) of the SAME period and length: when WMA pulls ahead of SMA, recent bars are "
        "moving faster than older ones -- an ACCELERATING trend -- confirmed by a slower EMA "
        "trend filter. A different mechanism from Family L (wma_ribbon_trend, which stacks "
        "three WMAs of different periods) and from every other trend family here, none of "
        "which compare two DIFFERENT averaging methods of the same window."
    ),
    param_grid={
        "period": [10, 20, 30],
        "ema_trend": [50, 100, 200],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 48],
    },
    build=_build_wma_sma_divergence_trend,
)


def _build_higher_low_structure_continuation(p: dict) -> dict:
    lookback, ema_trend, max_bars = p["lookback"], p["ema_trend"], p["max_bars"]
    return {
        "name": f"Higher-Low Structure Continuation (lookback={lookback}, ema{ema_trend} filter)",
        "entry_conditions": {
            "long": [
                _cond({"type": "swing_low", "lookback": lookback}, "is true", _val(1)),
                _cond(_ind("close", 1), ">", _ind("ema", ema_trend)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "swing_high", "lookback": lookback}, "is true", _val(1)),
                _cond(_ind("close", 1), "<", _ind("ema", ema_trend)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=max_bars),
    }


_HIGHER_LOW_STRUCTURE_CONTINUATION = SkeletonSpec(
    name="higher_low_structure_continuation",
    label="Higher-Low Structure Continuation (trend-filtered swing-pivot buy)",
    description=(
        "Buys a just-CONFIRMED swing low only while price is still above a slower EMA trend "
        "filter (mirrored: sells a confirmed swing high only below it) -- a continuation-via-"
        "structure hypothesis, distinct from Family M (swing_structure_fade, the same pivot "
        "primitive with NO trend filter, trading every pivot as a contrarian fade) and from "
        "Family B/Family V (which trigger off an RSI/EMA-distance reading, never off a "
        "confirmed price pivot)."
    ),
    param_grid={
        "lookback": [5, 10, 15],
        "ema_trend": [50, 100, 200],
        "stop_atr_mult": [0.75, 1.0, 1.5],
        "target_atr_mult": [1.5, 2.0, 3.0],
        "max_bars": [None, 24, 48],
    },
    build=_build_higher_low_structure_continuation,
)


# ---------------------------------------------------------------------------
# Family AI: Order Flow Absorption
#   A genuinely distinct order-flow hypothesis from Family F (Volume
#   Imbalance) above: THIS family requires an above-average PARTICIPATION
#   spike (relative_volume clearing a threshold -- real size trading
#   through right now, not just an ordinary bar) to occur AT THE SAME TIME
#   as a strong directional signed-volume imbalance (volume_delta clearing
#   its own threshold in the same direction) -- i.e. real size is actively
#   absorbing one side of the order book right now, not merely an
#   ordinary up/down-close bar with nothing unusual behind it. Exits as
#   soon as EITHER signal fades (participation drops back toward normal,
#   OR the imbalance itself flips back through zero), since either one
#   fading on its own means the order-flow conviction behind the move is
#   gone even if price hasn't reversed yet. Requires real volume data --
#   like Family F, degrades to zero trades (not an error) on volume-less
#   feeds where app.strategy.indicators.relative_volume/volume_delta both
#   fall back to their flat default series.
# ---------------------------------------------------------------------------

def _build_order_flow_absorption(p: dict) -> dict:
    vol_period, imbalance_period = p["vol_period"], p["imbalance_period"]
    rel_vol_threshold, imbalance_threshold = p["rel_vol_threshold"], p["imbalance_threshold"]
    return {
        "name": f"Order Flow Absorption (relvol>{rel_vol_threshold}, imb>{imbalance_threshold})",
        "entry_conditions": {
            "long": [
                _cond({"type": "relative_volume", "period": vol_period}, ">", _val(rel_vol_threshold)),
                _cond({"type": "volume_delta", "period": imbalance_period}, ">", _val(imbalance_threshold)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "relative_volume", "period": vol_period}, ">", _val(rel_vol_threshold)),
                _cond({"type": "volume_delta", "period": imbalance_period}, "<", _val(-imbalance_threshold)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [
                _cond({"type": "relative_volume", "period": vol_period}, "<", _val(1.0)),
                _cond({"type": "volume_delta", "period": imbalance_period}, "<", _val(0.0)),
            ],
            "long_connectors": ["OR"],
            "short": [
                _cond({"type": "relative_volume", "period": vol_period}, "<", _val(1.0)),
                _cond({"type": "volume_delta", "period": imbalance_period}, ">", _val(0.0)),
            ],
            "short_connectors": ["OR"],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_ORDER_FLOW_ABSORPTION = SkeletonSpec(
    name="order_flow_absorption",
    label="Order Flow Absorption (participation spike + signed imbalance)",
    description=(
        "Trades in the direction of a signed-volume imbalance ONLY when it coincides with an "
        "above-average participation spike -- relative_volume clearing a threshold at the same "
        "time volume_delta clears its own directional threshold -- i.e. real size actively "
        "absorbing one side of the book right now, not an ordinary up/down-close bar. Exits as "
        "soon as EITHER the participation spike fades back toward normal or the imbalance itself "
        "flips, since either one fading alone means the order-flow conviction behind the move is "
        "gone. A distinct hypothesis from Family F (Volume Imbalance), which trades the imbalance "
        "alone with no participation-spike confirmation requirement at all."
    ),
    param_grid={
        "vol_period": [10, 20],
        "imbalance_period": [10, 20],
        "rel_vol_threshold": [1.5, 2.0, 2.5],
        "imbalance_threshold": [0.2, 0.35],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [24, 48],
    },
    build=_build_order_flow_absorption,
)


# ---------------------------------------------------------------------------
# Expansion round 3 (multi-instrument search push): 5 more named families,
# each an established, safely-implemented primitive combined with exactly
# ONE filter it did not previously have -- the same "take a proven base
# hypothesis and add one more independently-motivated filter" pattern
# already used throughout this file (order_flow_absorption on top of
# volume_imbalance, volume_confirmed_breakout on top of trend_breakout,
# higher_low_structure_continuation on top of swing_structure_fade), rather
# than inventing new indicator primitives. Two traps deliberately avoided
# here that a naive new family could fall into: (1) raw highest_high/
# lowest_low compared directly to the current bar's own close is a
# tautology that can never fire, since the rolling window includes the
# current bar itself -- see the _breakout_flag() docstring above for why
# every breakout-style family here goes through the shift(1)-based "bos"
# primitive instead; no family below compares a raw price extreme to
# close for this reason. (2) Fair Value Gap is a point-in-time event, not
# a persisting zone (see Family N/Z's own scope notes), so a genuine
# "price returns later to fill an old gap" hypothesis isn't representable
# with today's primitive -- nothing below attempts that reading either.
# ---------------------------------------------------------------------------

def _build_order_block_trend_continuation(p: dict) -> dict:
    lookback, ema_trend = p["lookback"], p["ema_trend"]
    return {
        "name": f"Order Block Trend Continuation (lookback={lookback}, ema{ema_trend} filter)",
        "entry_conditions": {
            "long": [
                _cond({"type": "order_block", "lookback": lookback, "direction": "bullish"}, "is true", _val(1)),
                _cond(_ind("close", 1), ">", _ind("ema", ema_trend)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "order_block", "lookback": lookback, "direction": "bearish"}, "is true", _val(1)),
                _cond(_ind("close", 1), "<", _ind("ema", ema_trend)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_ORDER_BLOCK_TREND_CONTINUATION = SkeletonSpec(
    name="order_block_trend_continuation",
    label="Order Block Trend Continuation (SMC displacement + EMA filter)",
    description=(
        "Family O (order_block_reaction) trades the bare order-block primitive with no trend "
        "filter at all -- the one remaining SMC primitive here (alongside FVG and CHoCH) without "
        "a trend-filtered sibling, unlike Family N/AB's fvg_imbalance_continuation. This is that "
        "sibling: the same displacement-origin-candle signal, only taken in the direction of a "
        "slower EMA trend, on the theory that an order-block reaction WITH the prevailing trend "
        "is more likely to actually continue than one taken in isolation."
    ),
    param_grid={
        "lookback": [10, 20, 30],
        "ema_trend": [50, 100, 200],
        "stop_atr_mult": [0.75, 1.0, 1.5],
        "target_atr_mult": [1.5, 2.0, 3.0],
        "max_bars": [None, 24, 48],
    },
    build=_build_order_block_trend_continuation,
)


def _build_volume_confirmed_trend_pullback(p: dict) -> dict:
    ema_fast, ema_slow = p["ema_fast"], p["ema_slow"]
    rsi_period, rsi_low, rsi_high = p["rsi_period"], p["rsi_pullback_low"], p["rsi_pullback_high"]
    vol_period, vol_mult = p["vol_period"], p["vol_mult"]
    return {
        "name": f"Volume-Confirmed Trend Pullback (ema {ema_fast}/{ema_slow}, rsi{rsi_period}, relvol{vol_period}>{vol_mult}x)",
        "entry_conditions": {
            "long": [
                _cond(_ind("ema", ema_fast), ">", _ind("ema", ema_slow)),
                _cond(_ind("rsi", rsi_period), "<", _val(rsi_low)),
                _cond({"type": "relative_volume", "period": vol_period}, ">", _val(vol_mult)),
            ],
            "long_connectors": ["AND", "AND"],
            "short": [
                _cond(_ind("ema", ema_fast), "<", _ind("ema", ema_slow)),
                _cond(_ind("rsi", rsi_period), ">", _val(rsi_high)),
                _cond({"type": "relative_volume", "period": vol_period}, ">", _val(vol_mult)),
            ],
            "short_connectors": ["AND", "AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("rsi", rsi_period), ">", _val(rsi_high))],
            "short": [_cond(_ind("rsi", rsi_period), "<", _val(rsi_low))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_VOLUME_CONFIRMED_TREND_PULLBACK = SkeletonSpec(
    name="volume_confirmed_trend_pullback",
    label="Volume-Confirmed Trend Pullback (EMA trend + RSI dip/pop + relative-volume filter)",
    description=(
        "Family B (mtf_pullback) buys any RSI pullback inside an EMA trend, regardless of how "
        "much real size is behind the resumption bar. This adds ONE more independently-motivated "
        "filter -- the same relative-volume confirmation Family AI (order_flow_absorption) and "
        "volume_confirmed_breakout already use elsewhere -- requiring the pullback/resumption bar "
        "itself to trade on above-average relative volume, on the theory that a low-conviction "
        "(low-volume) pullback bounce is more likely to fail than one real size is stepping into."
    ),
    param_grid={
        "ema_fast": [20, 50],
        "ema_slow": [100, 200],
        "rsi_period": [7, 14],
        "rsi_pullback_low": [30, 40],
        "rsi_pullback_high": [60, 70],
        "vol_period": [14, 20],
        "vol_mult": [1.25, 1.5],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 48],
    },
    build=_build_volume_confirmed_trend_pullback,
    valid=lambda p: p["ema_fast"] < p["ema_slow"] and p["rsi_pullback_low"] < p["rsi_pullback_high"],
)


def _build_session_gated_liquidity_sweep(p: dict) -> dict:
    lookback, start, end, flat_time = p["lookback"], p["session_start"], p["session_end"], p["flat_time"]
    return {
        "name": f"Session-Gated Liquidity Sweep ({start}-{end}, lookback={lookback})",
        "entry_conditions": {
            "long": [
                _cond({"type": "liquidity_sweep", "lookback": lookback, "direction": "bullish"}, "is true", _val(1)),
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "liquidity_sweep", "lookback": lookback, "direction": "bearish"}, "is true", _val(1)),
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
        "_time_based_exit": flat_time,
    }


_SESSION_GATED_LIQUIDITY_SWEEP = SkeletonSpec(
    name="session_gated_liquidity_sweep",
    label="Session-Gated Liquidity Sweep (stop-hunt reclaim, clock-windowed)",
    description=(
        "Every session-anchored family here (E, K, Q, AA) is built on a plain price level "
        "(opening range, previous day's close) -- none of them combine a clock-time window with "
        "an SMC stop-hunt primitive. This does: Family G's liquidity-sweep-and-reclaim signal, "
        "restricted to a specific session window and force-flattened by a clock time, betting "
        "the classic 'stop hunt before the real session move' pattern is time-of-day dependent "
        "(e.g. a sweep right at a session open reads differently than the same sweep at 3am)."
    ),
    param_grid={
        "lookback": [5, 10, 20],
        "session_start": ["07:00", "08:30", "13:30"],
        "session_end": ["09:00", "10:30", "15:00"],
        "flat_time": ["16:00"],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [12, 24],
    },
    build=lambda p: _apply_time_based_exit(_build_session_gated_liquidity_sweep(p)),
    valid=lambda p: p["session_start"] < p["session_end"],
)


def _build_macd_histogram_zero_cross_trend(p: dict) -> dict:
    ema_trend = p["ema_trend"]
    return {
        "name": f"MACD Histogram Zero-Cross Trend (ema{ema_trend} filter)",
        "entry_conditions": {
            "long": [
                _cond({"type": "macd_histogram"}, "cross above", _val(0.0)),
                _cond(_ind("close", 1), ">", _ind("ema", ema_trend)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "macd_histogram"}, "cross below", _val(0.0)),
                _cond(_ind("close", 1), "<", _ind("ema", ema_trend)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond({"type": "macd_histogram"}, "cross below", _val(0.0))],
            "short": [_cond({"type": "macd_histogram"}, "cross above", _val(0.0))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_MACD_HISTOGRAM_ZERO_CROSS_TREND = SkeletonSpec(
    name="macd_histogram_zero_cross_trend",
    label="MACD Histogram Zero-Cross Trend (histogram-sign flip + EMA filter)",
    description=(
        "Family L (macd_cross_trend) triggers on the MACD line crossing its OWN signal line -- a "
        "different, and usually earlier-or-later, event than the histogram (the distance between "
        "those two lines) crossing zero, since the histogram can flip sign on a deceleration move "
        "well before or after the two lines themselves actually cross. This trades that separate "
        "event instead, still filtered by a slower EMA trend, as an independent crossover-timing "
        "hypothesis rather than a reparametrization of Family L."
    ),
    param_grid={
        "ema_trend": [50, 100, 200],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 48],
    },
    build=_build_macd_histogram_zero_cross_trend,
)


def _build_atr_regime_trend_pullback(p: dict) -> dict:
    ema_fast, ema_slow = p["ema_fast"], p["ema_slow"]
    rsi_period, rsi_low, rsi_high = p["rsi_period"], p["rsi_pullback_low"], p["rsi_pullback_high"]
    atr_period = p["atr_period"]
    return {
        "name": f"ATR-Regime Trend Pullback (ema {ema_fast}/{ema_slow}, rsi{rsi_period}, atr{atr_period} not-contracted)",
        "entry_conditions": {
            "long": [
                _cond(_ind("ema", ema_fast), ">", _ind("ema", ema_slow)),
                _cond(_ind("rsi", rsi_period), "<", _val(rsi_low)),
                _cond({"type": "atr_regime", "period": atr_period}, "!=", _val(-1)),
            ],
            "long_connectors": ["AND", "AND"],
            "short": [
                _cond(_ind("ema", ema_fast), "<", _ind("ema", ema_slow)),
                _cond(_ind("rsi", rsi_period), ">", _val(rsi_high)),
                _cond({"type": "atr_regime", "period": atr_period}, "!=", _val(-1)),
            ],
            "short_connectors": ["AND", "AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("rsi", rsi_period), ">", _val(rsi_high))],
            "short": [_cond(_ind("rsi", rsi_period), "<", _val(rsi_low))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_ATR_REGIME_TREND_PULLBACK = SkeletonSpec(
    name="atr_regime_trend_pullback",
    label="ATR-Regime Trend Pullback (EMA trend + RSI dip/pop, skips contracted regimes)",
    description=(
        "Family B (mtf_pullback) trades every qualifying RSI pullback inside an EMA trend "
        "regardless of the current volatility regime -- including a flat, choppy market where a "
        "'trend' reading from two EMAs is closer to noise. This adds an atr_regime primitive gate "
        "(the same regime primitive Family P/Family V already use, applied here to a continuation "
        "hypothesis instead of a breakout or fade one): stands down only while ATR reads as "
        "CONTRACTED relative to its own baseline (atr_regime == -1), trading through both the "
        "neutral and expansion states -- a deliberately looser gate than requiring active "
        "expansion outright, since the hypothesis being tested is 'skip the range-bound chop', "
        "not 'only trade full-blown expansion'."
    ),
    param_grid={
        "ema_fast": [20, 50],
        "ema_slow": [100, 200],
        "rsi_period": [7, 14],
        "rsi_pullback_low": [30, 40],
        "rsi_pullback_high": [60, 70],
        "atr_period": [14, 20],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 48],
    },
    build=_build_atr_regime_trend_pullback,
    valid=lambda p: p["ema_fast"] < p["ema_slow"] and p["rsi_pullback_low"] < p["rsi_pullback_high"],
)


# ---------------------------------------------------------------------------
# Expansion round 4 (Sep 2026): 6 more families, each built on a
# primitive this module already supports but had never actually used in
# any family before -- session_high/session_low (running intraday
# session extreme, distinct from Family K's FIXED prior-day level and
# every rolling-N-bar liquidity_sweep/bos family), candle_range (a
# single bar's own high-low, never yet compared against ATR), and
# average_volume (never yet compared against itself at two different
# periods) -- plus two new FILTER combinations (relative-volume gating
# on the FVG and order-block primitives, which previously only ever had
# an EMA-filtered or unfiltered sibling) and one new filter pairing
# (VWAP trend regime gating a Bollinger Band pullback, which previously
# only ever paired with RSI or with nothing).
# ---------------------------------------------------------------------------

def _build_session_extreme_fade(p: dict) -> dict:
    start, end = p["session_start"], p["session_end"]
    return {
        "name": f"Session Extreme Fade ({start}-{end})",
        "entry_conditions": {
            # A fresh session low PRINTED this bar (low == session_low, exactly --
            # session_low is literally derived from this bar's own low via a
            # same-bar expanding cummin, so an == comparison is comparing a
            # value against itself when it just updated, not a fragile
            # float-rounding check) but the bar closed back ABOVE that low --
            # a same-bar rejection off today's running extreme, mirrored for
            # a fresh session high closing back below it.
            "long": [
                _cond(_ind("low", 1), "==", {"type": "session_low", "session_start": start, "session_end": end}),
                _cond(_ind("close", 1), ">", {"type": "session_low", "session_start": start, "session_end": end}),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("high", 1), "==", {"type": "session_high", "session_start": start, "session_end": end}),
                _cond(_ind("close", 1), "<", {"type": "session_high", "session_start": start, "session_end": end}),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_SESSION_EXTREME_FADE = SkeletonSpec(
    name="session_extreme_fade",
    label="Session Extreme Fade (running intraday session high/low rejection)",
    description=(
        "Fades a same-bar rejection off TODAY's running session high or low -- an expanding "
        "intraday extreme that resets every session, distinct in both timeframe and mechanism "
        "from Family K's FIXED prior-day high/low breakout and from every rolling-N-bar "
        "liquidity_sweep/break_of_structure family here (G, J, AC): those look back a fixed bar "
        "count regardless of the clock, this looks at 'the extreme so far today,' however many "
        "bars that is."
    ),
    param_grid={
        "session_start": ["00:00", "08:30"],
        "session_end": ["23:59", "16:00"],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [1.5, 2.5, 3.5],
        "max_bars": [None, 24, 48],
    },
    build=_build_session_extreme_fade,
)


def _build_vwap_bollinger_pullback(p: dict) -> dict:
    bb_period, bb_std = p["bb_period"], p["bb_std"]
    return {
        "name": f"VWAP-Filtered Bollinger Pullback (bb{bb_period}x{bb_std})",
        "entry_conditions": {
            # A bounce back inside the band from below it (a pullback low),
            # taken only while price is still holding above VWAP (the same
            # continuation-vs-reversion trend gate Family I/AB already use,
            # here paired with Bollinger Bands instead of RSI/EMA -- neither
            # mean_reversion_band (RSI-filtered fade) nor
            # bollinger_band_walk_continuation (EMA-filtered breakout) pairs
            # the bands with VWAP).
            "long": [
                _cond(_ind("close", 1), "cross above", {"type": "bollinger_lower", "period": bb_period, "field": "close"}),
                _cond(_ind("close", 1), ">", {"type": "vwap"}),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("close", 1), "cross below", {"type": "bollinger_upper", "period": bb_period, "field": "close"}),
                _cond(_ind("close", 1), "<", {"type": "vwap"}),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), "<", {"type": "vwap"})],
            "short": [_cond(_ind("close", 1), ">", {"type": "vwap"})],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_VWAP_BOLLINGER_PULLBACK = SkeletonSpec(
    name="vwap_bollinger_pullback",
    label="VWAP-Filtered Bollinger Pullback (band bounce + VWAP trend gate)",
    description=(
        "Buys/sells a Bollinger Band pullback bounce, taken only in the direction VWAP already "
        "confirms (price above VWAP for longs, below for shorts) -- a continuation reading of "
        "the same band-touch event Family C (mean_reversion_band, RSI-filtered) and Family AC "
        "(bollinger_band_walk_continuation, EMA-filtered) already use, this time gated by VWAP "
        "instead, which neither of those pairs it with."
    ),
    param_grid={
        "bb_period": [14, 20],
        "bb_std": [2.0, 2.5],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [None, 24, 48],
    },
    build=_build_vwap_bollinger_pullback,
)


def _build_volume_confirmed_fvg_continuation(p: dict) -> dict:
    vol_period, vol_mult = p["vol_period"], p["vol_mult"]
    return {
        "name": f"Volume-Confirmed FVG Continuation (relvol{vol_period}>{vol_mult}x)",
        "entry_conditions": {
            # Family N (fvg_imbalance_continuation) filters the same FVG
            # primitive with an EMA trend; this filters it with relative
            # volume instead -- a genuinely different confirmation signal
            # (participation, not trend direction) for the same displacement
            # event.
            "long": [
                _cond({"type": "fair_value_gap", "direction": "bullish"}, "is true", _val(1)),
                _cond({"type": "relative_volume", "period": vol_period}, ">", _val(vol_mult)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "fair_value_gap", "direction": "bearish"}, "is true", _val(1)),
                _cond({"type": "relative_volume", "period": vol_period}, ">", _val(vol_mult)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_VOLUME_CONFIRMED_FVG_CONTINUATION = SkeletonSpec(
    name="volume_confirmed_fvg_continuation",
    label="Volume-Confirmed FVG Continuation (displacement + relative-volume filter)",
    description=(
        "The same Fair Value Gap displacement primitive Family N already trades, filtered by "
        "relative volume (above-average participation on the displacement) instead of Family "
        "N's EMA trend filter -- a participation-based confirmation rather than a "
        "trend-direction one for the same underlying event."
    ),
    param_grid={
        "vol_period": [10, 20],
        "vol_mult": [1.5, 2.0],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [None, 24, 48],
    },
    build=_build_volume_confirmed_fvg_continuation,
)


def _build_volume_confirmed_order_block_reaction(p: dict) -> dict:
    lookback = p["lookback"]
    vol_period, vol_mult = p["vol_period"], p["vol_mult"]
    return {
        "name": f"Volume-Confirmed Order Block Reaction (lookback={lookback}, relvol{vol_period}>{vol_mult}x)",
        "entry_conditions": {
            # Family O (order_block_reaction) trades this primitive bare;
            # Family AD (order_block_trend_continuation) filters it with an
            # EMA trend. This is the third, remaining reading: filtered by
            # relative volume instead, the same "add a participation gate,
            # not a trend gate" idea volume_confirmed_fvg_continuation above
            # applies to the FVG primitive.
            "long": [
                _cond({"type": "order_block", "lookback": lookback, "direction": "bullish"}, "is true", _val(1)),
                _cond({"type": "relative_volume", "period": vol_period}, ">", _val(vol_mult)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "order_block", "lookback": lookback, "direction": "bearish"}, "is true", _val(1)),
                _cond({"type": "relative_volume", "period": vol_period}, ">", _val(vol_mult)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_VOLUME_CONFIRMED_ORDER_BLOCK_REACTION = SkeletonSpec(
    name="volume_confirmed_order_block_reaction",
    label="Volume-Confirmed Order Block Reaction (SMC displacement + relative-volume filter)",
    description=(
        "The same order-block displacement-origin-candle primitive Families O and AD already "
        "trade (bare, and EMA-filtered respectively), this time filtered by relative volume -- "
        "the third and final remaining reading of this primitive's three natural confirmation "
        "states (none / trend / participation)."
    ),
    param_grid={
        "lookback": [10, 20, 30],
        "vol_period": [10, 20],
        "vol_mult": [1.5, 2.0],
        "stop_atr_mult": [0.75, 1.0, 1.5],
        "target_atr_mult": [1.5, 2.0, 3.0],
        "max_bars": [None, 24, 48],
    },
    build=_build_volume_confirmed_order_block_reaction,
)


def _build_wide_range_bar_exhaustion_fade(p: dict) -> dict:
    atr_period = p["atr_period"]
    return {
        "name": f"Wide-Range Bar Exhaustion Fade (atr{atr_period})",
        "entry_conditions": {
            # A single bar's own high-low range (candle_range -- never yet
            # used by any family here) exceeding the recent average true
            # range reads as a climactic, one-bar exhaustion move rather
            # than the start of a sustained new leg -- fade a down-close
            # climactic bar long, mirrored for an up-close one short.
            # Distinct data source from Family Q/volume_climax_reversal,
            # which reads the same "climax" idea off relative VOLUME
            # instead of the bar's own price range. (The DSL's condition
            # engine only supports a direct left-operator-right comparison
            # between two series -- no per-operand scaling factor -- so the
            # "how extreme" knob here is the ATR averaging period itself,
            # not a multiplier on top of it.)
            "long": [
                _cond({"type": "candle_range"}, ">", {"type": "atr", "period": atr_period}),
                _cond({"type": "candle_direction", "direction": "bearish"}, "is true", _val(1)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "candle_range"}, ">", {"type": "atr", "period": atr_period}),
                _cond({"type": "candle_direction", "direction": "bullish"}, "is true", _val(1)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_WIDE_RANGE_BAR_EXHAUSTION_FADE = SkeletonSpec(
    name="wide_range_bar_exhaustion_fade",
    label="Wide-Range Bar Exhaustion Fade (single-bar range climax vs. ATR)",
    description=(
        "Fades a single bar whose own high-low range exceeded the recent average true range and "
        "closed near its extreme -- a price-range reading of the same 'climactic, one-bar "
        "exhaustion' idea Family Q (volume_climax_reversal) reads off relative volume instead. "
        "candle_range (a bar's own high-low) is a primitive no other family here uses."
    ),
    param_grid={
        "atr_period": [10, 14, 20, 30],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.0],
        "max_bars": [None, 12, 24],
    },
    build=_build_wide_range_bar_exhaustion_fade,
)


def _build_volume_trend_breakout_confirmation(p: dict) -> dict:
    lookback = p["lookback"]
    vol_fast, vol_slow = p["vol_fast"], p["vol_slow"]
    return {
        "name": f"Volume-Trend Breakout Confirmation (lb={lookback}, avgvol {vol_fast}/{vol_slow})",
        "entry_conditions": {
            # A Donchian breakout (the same bos primitive Family A/D/C use),
            # confirmed by a RISING volume trend -- average_volume(fast) >
            # average_volume(slow), i.e. participation has been building
            # into the breakout, not just a single loud bar -- rather than
            # by relative volume on the breakout bar alone (which
            # volume_confirmed_breakout already covers) or by an ATR
            # expansion/contraction regime (Families D/C). average_volume
            # compared against itself at two periods is a primitive
            # pairing no family here uses yet.
            "long": [
                _cond(_breakout_flag(lookback, "bullish"), "is true", _val(1)),
                _cond({"type": "average_volume", "period": vol_fast}, ">", {"type": "average_volume", "period": vol_slow}),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_breakout_flag(lookback, "bearish"), "is true", _val(1)),
                _cond({"type": "average_volume", "period": vol_fast}, ">", {"type": "average_volume", "period": vol_slow}),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_VOLUME_TREND_BREAKOUT_CONFIRMATION = SkeletonSpec(
    name="volume_trend_breakout_confirmation",
    label="Volume-Trend Breakout Confirmation (Donchian + rising average-volume filter)",
    description=(
        "The same N-bar Donchian breakout Family A trades, confirmed by a RISING volume trend "
        "(a fast average-volume above a slow one -- participation building into the move) "
        "rather than by relative volume on the breakout bar alone (volume_confirmed_breakout) "
        "or an ATR volatility regime (Families D/C). average_volume compared against itself at "
        "two different periods is a primitive pairing no family here has used before."
    ),
    param_grid={
        "lookback": [10, 20, 30],
        "vol_fast": [10, 20],
        "vol_slow": [30, 50],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0],
    },
    build=_build_volume_trend_breakout_confirmation,
    valid=lambda p: p["vol_fast"] < p["vol_slow"],
)


# ---------------------------------------------------------------------------
# Expansion round 5 (indicator-widening push): 7 new families built on 7
# indicators this module's search space had never used before -- ADX,
# Stochastic, CCI, OBV, Keltner Channel, Donchian Channel (continuous
# levels, not just the existing `bos` breakout boolean), and SuperTrend.
# Each pairs a genuinely distinct indicator with a mechanism (trend-strength
# gate, oscillator reversion, unbounded-oscillator reversion, volume/price
# divergence confirmation, ATR-band squeeze, channel-level breakout,
# flip-based trend-following) not already covered by an existing family --
# see each SkeletonSpec's own description for the distinction.
# ---------------------------------------------------------------------------

def _build_adx_trend_strength_breakout(p: dict) -> dict:
    lookback, ema_fast, ema_slow = p["lookback"], p["ema_fast"], p["ema_slow"]
    adx_period, adx_min = p["adx_period"], p["adx_min"]
    return {
        "name": f"ADX Trend-Strength Breakout (lb={lookback}, adx{adx_period}>={adx_min})",
        "entry_conditions": {
            "long": [
                _cond(_breakout_flag(lookback, "bullish"), "is true", _val(1)),
                _cond(_ind("ema", ema_fast), ">", _ind("ema", ema_slow)),
                _cond(_ind("adx", adx_period), ">=", _val(adx_min)),
            ],
            "long_connectors": ["AND", "AND"],
            "short": [
                _cond(_breakout_flag(lookback, "bearish"), "is true", _val(1)),
                _cond(_ind("ema", ema_fast), "<", _ind("ema", ema_slow)),
                _cond(_ind("adx", adx_period), ">=", _val(adx_min)),
            ],
            "short_connectors": ["AND", "AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_ADX_TREND_STRENGTH_BREAKOUT = SkeletonSpec(
    name="adx_trend_strength_breakout",
    label="ADX Trend-Strength Breakout (Donchian + EMA + ADX gate)",
    description=(
        "The same trend-aligned Donchian breakout as Family A, gated by ADX so it only fires "
        "when the market is actually trending strongly (ADX above threshold) rather than "
        "chopping -- ADX is a pure trend-STRENGTH filter (no direction of its own), a primitive "
        "no prior family used."
    ),
    param_grid={
        "lookback": [10, 20, 40],
        "ema_fast": [20, 50],
        "ema_slow": [100, 200],
        "adx_period": [14],
        "adx_min": [20, 25, 30],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0],
    },
    build=_build_adx_trend_strength_breakout,
    valid=lambda p: p["ema_fast"] < p["ema_slow"],
)


def _build_stochastic_extreme_reversion(p: dict) -> dict:
    period, smooth, oversold, overbought = p["period"], p["smooth"], p["oversold"], p["overbought"]
    return {
        "name": f"Stochastic Extreme Reversion (k{period}, {oversold}/{overbought})",
        "entry_conditions": {
            "long": [_cond(_ind("stoch_k", period), "crosses above", _val(oversold))],
            "long_connectors": [],
            "short": [_cond(_ind("stoch_k", period), "crosses below", _val(overbought))],
            "short_connectors": [],
        },
        "exit_conditions": {
            "long": [_cond(_ind("stoch_k", period), ">", _val(overbought))],
            "short": [_cond(_ind("stoch_k", period), "<", _val(oversold))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_STOCHASTIC_EXTREME_REVERSION = SkeletonSpec(
    name="stochastic_extreme_reversion",
    label="Stochastic Extreme Reversion (%K crossing back from oversold/overbought)",
    description=(
        "Enters when the Stochastic %K crosses back above an oversold floor (long) or below an "
        "overbought ceiling (short) -- a bounded-oscillator reversion trigger distinct from "
        "rsi_extreme_reversion's RSI-based one (Stochastic and RSI diverge meaningfully on "
        "range-bound vs trending data)."
    ),
    param_grid={
        "period": [9, 14, 21],
        "smooth": [3],
        "oversold": [15, 20],
        "overbought": [80, 85],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.0],
        "max_bars": [16, 32],
    },
    build=_build_stochastic_extreme_reversion,
    valid=lambda p: p["oversold"] < p["overbought"],
)


def _build_cci_extreme_reversion(p: dict) -> dict:
    period, threshold = p["period"], p["threshold"]
    return {
        "name": f"CCI Extreme Reversion (cci{period}, +/-{threshold})",
        "entry_conditions": {
            "long": [_cond(_ind("cci", period), "crosses above", _val(-threshold))],
            "long_connectors": [],
            "short": [_cond(_ind("cci", period), "crosses below", _val(threshold))],
            "short_connectors": [],
        },
        "exit_conditions": {
            "long": [_cond(_ind("cci", period), ">", _val(0))],
            "short": [_cond(_ind("cci", period), "<", _val(0))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_CCI_EXTREME_REVERSION = SkeletonSpec(
    name="cci_extreme_reversion",
    label="CCI Extreme Reversion (unbounded oscillator, +/- threshold)",
    description=(
        "Fades a CCI extreme back through zero -- CCI is unbounded (unlike RSI/Stochastic's "
        "fixed 0-100 scale), so its 'how extreme' reading behaves differently in strongly "
        "trending regimes; a distinct reversion hypothesis from the two bounded-oscillator "
        "families above."
    ),
    param_grid={
        "period": [14, 20, 30],
        "threshold": [100, 150, 200],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.0],
        "max_bars": [16, 32],
    },
    build=_build_cci_extreme_reversion,
)


def _build_obv_divergence_trend_confirmation(p: dict) -> dict:
    obv_fast, obv_slow = p["obv_fast"], p["obv_slow"]
    lookback = p["lookback"]
    return {
        "name": f"OBV Divergence Trend Confirmation (obv {obv_fast}/{obv_slow}, lb={lookback})",
        "entry_conditions": {
            # A structure breakout (bos) confirmed by OBV's own trend agreeing
            # (a fast EMA-of-OBV above a slow one) -- i.e. cumulative buying/
            # selling volume was already leaning the same direction BEFORE the
            # price breakout, the volume-leads-price idea behind classic OBV
            # divergence analysis, distinct from volume_confirmed_breakout's
            # single-bar relative-volume check.
            "long": [
                _cond(_breakout_flag(lookback, "bullish"), "is true", _val(1)),
                _cond(_ind("obv_ema", obv_fast), ">", _ind("obv_ema", obv_slow)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_breakout_flag(lookback, "bearish"), "is true", _val(1)),
                _cond(_ind("obv_ema", obv_fast), "<", _ind("obv_ema", obv_slow)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_OBV_DIVERGENCE_TREND_CONFIRMATION = SkeletonSpec(
    name="obv_divergence_trend_confirmation",
    label="OBV Divergence Trend Confirmation (structure breakout + OBV trend agreement)",
    description=(
        "A structure breakout taken only when On-Balance Volume's own trend (fast EMA-of-OBV "
        "vs slow) already agrees with the breakout direction -- the classic 'volume leads "
        "price' idea, distinct from a single-bar relative-volume confirmation."
    ),
    param_grid={
        "lookback": [10, 20, 30],
        "obv_fast": [10, 20],
        "obv_slow": [30, 50],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0],
    },
    build=_build_obv_divergence_trend_confirmation,
    valid=lambda p: p["obv_fast"] < p["obv_slow"],
)


def _build_keltner_squeeze_breakout(p: dict) -> dict:
    kc_period, kc_mult, don_period = p["kc_period"], p["kc_mult"], p["don_period"]
    return {
        "name": f"Keltner Squeeze Breakout (kc{kc_period}x{kc_mult}, don{don_period})",
        "entry_conditions": {
            # An ATR-band (Keltner) squeeze/breakout hypothesis: price clears
            # the Keltner upper/lower band -- distinct from
            # volatility_contraction_squeeze, which gates on ATR CONTRACTING
            # then a plain bos breakout; this instead uses the Keltner band's
            # own continuous level as the breakout trigger, which widens and
            # narrows with directional range rather than close-to-close
            # dispersion (Bollinger), so it disagrees with a Bollinger-based
            # squeeze on genuinely different bars.
            "long": [_cond(_ind("close", 1), ">", _ind("keltner_upper", kc_period))],
            "long_connectors": [],
            "short": [_cond(_ind("close", 1), "<", _ind("keltner_lower", kc_period))],
            "short_connectors": [],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), "<", _ind("keltner_mid", kc_period))],
            "short": [_cond(_ind("close", 1), ">", _ind("keltner_mid", kc_period))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_KELTNER_SQUEEZE_BREAKOUT = SkeletonSpec(
    name="keltner_squeeze_breakout",
    label="Keltner Squeeze Breakout (ATR-band breakout, not Bollinger)",
    description=(
        "Enters when price clears an ATR-based Keltner Channel band -- an ATR band widens with "
        "directional range rather than close-to-close dispersion, so it fires on different bars "
        "than the existing Bollinger/Donchian-based breakout families."
    ),
    param_grid={
        "kc_period": [20, 30],
        "kc_mult": [1.5, 2.0, 2.5],
        "don_period": [20],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0],
    },
    build=_build_keltner_squeeze_breakout,
)


def _build_donchian_channel_turtle_breakout(p: dict) -> dict:
    entry_period, exit_period = p["entry_period"], p["exit_period"]
    return {
        "name": f"Donchian Turtle Breakout (entry={entry_period}, exit={exit_period})",
        "entry_conditions": {
            # Classic turtle-trader system: enter on a close beyond the
            # N-bar Donchian upper/lower level, exit on a close back through
            # a SHORTER Donchian channel -- distinct from `_breakout_flag`
            # (`bos`)'s one-shot boolean in that it exposes the actual
            # channel LEVEL, enabling the asymmetric entry/exit channel
            # widths this system is defined by (Families A/H/Z all use one
            # symmetric lookback for both).
            "long": [_cond(_ind("close", 1), ">", _ind("donchian_upper", entry_period))],
            "long_connectors": [],
            "short": [_cond(_ind("close", 1), "<", _ind("donchian_lower", entry_period))],
            "short_connectors": [],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), "<", _ind("donchian_lower", exit_period))],
            "short": [_cond(_ind("close", 1), ">", _ind("donchian_upper", exit_period))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_DONCHIAN_CHANNEL_TURTLE_BREAKOUT = SkeletonSpec(
    name="donchian_channel_turtle_breakout",
    label="Donchian Turtle Breakout (asymmetric entry/exit channel)",
    description=(
        "The original turtle-trader system: enter on a close beyond a longer Donchian channel, "
        "exit on a close back through a shorter one -- an asymmetric-channel-width hypothesis "
        "no other breakout family here expresses (they all use one lookback for both)."
    ),
    param_grid={
        "entry_period": [20, 40, 55],
        "exit_period": [10, 20],
        "stop_atr_mult": [1.5, 2.0],
        "target_atr_mult": [3.0, 4.0],
    },
    build=_build_donchian_channel_turtle_breakout,
    valid=lambda p: p["exit_period"] < p["entry_period"],
)


def _build_supertrend_trend_following(p: dict) -> dict:
    st_period, st_mult = p["st_period"], p["st_mult"]
    return {
        "name": f"SuperTrend Trend Following (st{st_period}x{st_mult})",
        "entry_conditions": {
            # SuperTrend's own direction series flips exactly at the bar
            # price closes through the ratcheting trailing-stop line -- a
            # flip-based trend-following trigger distinct from every
            # crossover/breakout family above, since the line's position
            # depends on its own PRIOR value and prior direction (a stateful
            # ratchet), not a stateless rolling window.
            "long": [_cond(_ind("supertrend_direction", st_period), "crosses above", _val(0))],
            "long_connectors": [],
            "short": [_cond(_ind("supertrend_direction", st_period), "crosses below", _val(0))],
            "short_connectors": [],
        },
        "exit_conditions": {
            "long": [_cond(_ind("supertrend_direction", st_period), "<", _val(0))],
            "short": [_cond(_ind("supertrend_direction", st_period), ">", _val(0))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_SUPERTREND_TREND_FOLLOWING = SkeletonSpec(
    name="supertrend_trend_following",
    label="SuperTrend Trend Following (flip-based ratcheting stop)",
    description=(
        "Enters on a SuperTrend direction flip (price closing through its own ratcheting "
        "trailing-stop line) and stays with the trend until the next flip -- a stateful, "
        "flip-based trend-following mechanism distinct from every crossover/breakout family "
        "above."
    ),
    param_grid={
        "st_period": [7, 10, 14],
        "st_mult": [2.0, 3.0, 4.0],
        "stop_atr_mult": [1.5, 2.0],
        "target_atr_mult": [3.0, 4.0],
    },
    build=_build_supertrend_trend_following,
)


# ---------------------------------------------------------------------------
# Expansion round 6: 5 more families on 5 more indicators this module had
# never used before -- Williams %R, Rate of Change, Awesome Oscillator,
# Chaikin Money Flow, Parabolic SAR. Same "one indicator, one distinct
# mechanism" convention as round 5.
# ---------------------------------------------------------------------------

def _build_williams_r_extreme_reversion(p: dict) -> dict:
    period, oversold, overbought = p["period"], p["oversold"], p["overbought"]
    return {
        "name": f"Williams %R Extreme Reversion (wr{period}, {oversold}/{overbought})",
        "entry_conditions": {
            "long": [_cond(_ind("williams_r", period), "crosses above", _val(oversold))],
            "long_connectors": [],
            "short": [_cond(_ind("williams_r", period), "crosses below", _val(overbought))],
            "short_connectors": [],
        },
        "exit_conditions": {
            "long": [_cond(_ind("williams_r", period), ">", _val(overbought))],
            "short": [_cond(_ind("williams_r", period), "<", _val(oversold))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_WILLIAMS_R_EXTREME_REVERSION = SkeletonSpec(
    name="williams_r_extreme_reversion",
    label="Williams %R Extreme Reversion (-100..0 scale, distinct thresholds from Stochastic)",
    description=(
        "Fades Williams %R back from an extreme on its own -100..0 scale -- the identical "
        "underlying high/low-range position Stochastic uses, but with that scale's own -20/-80 "
        "conventional thresholds rather than Stochastic's 80/20."
    ),
    param_grid={
        "period": [10, 14, 21],
        "oversold": [-85, -80],
        "overbought": [-20, -15],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.0],
        "max_bars": [16, 32],
    },
    build=_build_williams_r_extreme_reversion,
)


def _build_roc_momentum_continuation(p: dict) -> dict:
    period, threshold = p["period"], p["threshold"]
    return {
        "name": f"ROC Momentum Continuation (roc{period}, +/-{threshold}%)",
        "entry_conditions": {
            # A raw percentage-change momentum trigger -- distinct from
            # every RSI/Stochastic/CCI-based family above, none of which
            # measure a simple lookback percentage change.
            "long": [_cond(_ind("roc", period), "crosses above", _val(threshold))],
            "long_connectors": [],
            "short": [_cond(_ind("roc", period), "crosses below", _val(-threshold))],
            "short_connectors": [],
        },
        "exit_conditions": {
            "long": [_cond(_ind("roc", period), "<", _val(0))],
            "short": [_cond(_ind("roc", period), ">", _val(0))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_ROC_MOMENTUM_CONTINUATION = SkeletonSpec(
    name="roc_momentum_continuation",
    label="ROC Momentum Continuation (raw % change, not range-position)",
    description=(
        "Enters once the Rate of Change (a raw percentage move over a fixed lookback) clears a "
        "threshold in either direction -- a pure momentum trigger distinct from every range-"
        "position oscillator (RSI/Stochastic/Williams %R/CCI) used elsewhere in this module."
    ),
    param_grid={
        "period": [5, 10, 20],
        "threshold": [1.0, 2.0, 3.0],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0],
    },
    build=_build_roc_momentum_continuation,
)


def _build_awesome_oscillator_zero_cross(p: dict) -> dict:
    return {
        "name": "Awesome Oscillator Zero Cross",
        "entry_conditions": {
            # AO's own conventional trigger -- a zero-line cross of
            # SMA(5)-SMA(34) of midpoint price, fixed periods by
            # definition (see awesome_oscillator's own docstring).
            "long": [_cond(_ind("awesome_oscillator", 1), "crosses above", _val(0))],
            "long_connectors": [],
            "short": [_cond(_ind("awesome_oscillator", 1), "crosses below", _val(0))],
            "short_connectors": [],
        },
        "exit_conditions": {
            "long": [_cond(_ind("awesome_oscillator", 1), "<", _val(0))],
            "short": [_cond(_ind("awesome_oscillator", 1), ">", _val(0))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_AWESOME_OSCILLATOR_ZERO_CROSS = SkeletonSpec(
    name="awesome_oscillator_zero_cross",
    label="Awesome Oscillator Zero Cross (fixed-period SMA(5)/SMA(34) of midpoint)",
    description=(
        "Enters on Awesome Oscillator's own zero-line cross -- SMA(5) minus SMA(34) of the "
        "bar's own midpoint (high+low)/2, fixed periods by the indicator's own convention -- "
        "distinct from MACD's EMA-based fast/slow difference computed on CLOSE."
    ),
    param_grid={
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0, 4.0],
    },
    build=_build_awesome_oscillator_zero_cross,
)


def _build_cmf_volume_confirmation(p: dict) -> dict:
    lookback, cmf_period, cmf_threshold = p["lookback"], p["cmf_period"], p["cmf_threshold"]
    return {
        "name": f"CMF Volume Confirmation (lb={lookback}, cmf{cmf_period}>={cmf_threshold})",
        "entry_conditions": {
            # A structure breakout confirmed by Chaikin Money Flow already
            # leaning the same direction -- distinct from
            # volume_confirmed_breakout's single-bar relative-volume check
            # and from obv_divergence_trend_confirmation's cumulative,
            # unbounded OBV-EMA trend: CMF is a BOUNDED, WINDOWED measure
            # of where within each bar's own range the volume traded.
            "long": [
                _cond(_breakout_flag(lookback, "bullish"), "is true", _val(1)),
                _cond(_ind("cmf", cmf_period), ">", _val(cmf_threshold)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_breakout_flag(lookback, "bearish"), "is true", _val(1)),
                _cond(_ind("cmf", cmf_period), "<", _val(-cmf_threshold)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_CMF_VOLUME_CONFIRMATION = SkeletonSpec(
    name="cmf_volume_confirmation",
    label="CMF Volume Confirmation (bounded windowed accumulation/distribution)",
    description=(
        "A structure breakout confirmed by Chaikin Money Flow already leaning the same "
        "direction -- a bounded, windowed accumulation/distribution measure, distinct from "
        "OBV's unbounded running total and from a single-bar relative-volume check."
    ),
    param_grid={
        "lookback": [10, 20, 30],
        "cmf_period": [20],
        "cmf_threshold": [0.05, 0.1, 0.15],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0],
    },
    build=_build_cmf_volume_confirmation,
)


def _build_parabolic_sar_trend_following(p: dict) -> dict:
    af_start, af_step, af_max = p["af_start"], p["af_step"], p["af_max"]
    return {
        "name": f"Parabolic SAR Trend Following (af={af_start}/{af_step}/{af_max})",
        "entry_conditions": {
            # A flip-based, ACCELERATING trailing stop -- distinct from
            # SuperTrend's fixed ATR multiple, since the step size itself
            # grows every bar the trend continues (see parabolic_sar's own
            # docstring).
            "long": [_cond(_ind("psar_direction", 1), "crosses above", _val(0))],
            "long_connectors": [],
            "short": [_cond(_ind("psar_direction", 1), "crosses below", _val(0))],
            "short_connectors": [],
        },
        "exit_conditions": {
            "long": [_cond(_ind("psar_direction", 1), "<", _val(0))],
            "short": [_cond(_ind("psar_direction", 1), ">", _val(0))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_PARABOLIC_SAR_TREND_FOLLOWING = SkeletonSpec(
    name="parabolic_sar_trend_following",
    label="Parabolic SAR Trend Following (accelerating flip-based stop)",
    description=(
        "Enters on a Parabolic SAR flip -- an accelerating trailing stop whose step size grows "
        "every bar the trend continues, the original Wilder stop-and-reverse system, distinct "
        "from SuperTrend's fixed ATR-multiple band."
    ),
    param_grid={
        "af_start": [0.01, 0.02],
        "af_step": [0.01, 0.02],
        "af_max": [0.1, 0.2, 0.3],
        "stop_atr_mult": [1.5, 2.0],
        "target_atr_mult": [3.0, 4.0],
    },
    build=_build_parabolic_sar_trend_following,
)


FAMILIES: dict[str, SkeletonSpec] = {
    _TREND_BREAKOUT.name: _TREND_BREAKOUT,
    _MTF_PULLBACK.name: _MTF_PULLBACK,
    _MEAN_REVERSION_BAND.name: _MEAN_REVERSION_BAND,
    _VOLATILITY_BREAKOUT.name: _VOLATILITY_BREAKOUT,
    _SESSION_TIME_EFFECT.name: _SESSION_TIME_EFFECT,
    _VOLUME_IMBALANCE.name: _VOLUME_IMBALANCE,
    _STAT_PAIRS.name: _STAT_PAIRS,
    _LIQUIDITY_SWEEP_REVERSAL.name: _LIQUIDITY_SWEEP_REVERSAL,
    _MOMENTUM_CONTINUATION.name: _MOMENTUM_CONTINUATION,
    _VWAP_REVERSION.name: _VWAP_REVERSION,
    _MARKET_STRUCTURE_SHIFT.name: _MARKET_STRUCTURE_SHIFT,
    # -- Expansion round: 10 new families spanning daily-structure breakout,
    # crossover-event momentum, pivot-fade reversal, SMC imbalance/order-flow
    # continuation, volatility-contraction (squeeze) breakout, session-anchored
    # gap fade, an alternate MA-smoothing trend style, raw rate-of-change
    # ignition, and a standalone RSI-extremity reversion -- see each
    # SkeletonSpec's own comment block above for what makes it a genuinely
    # distinct hypothesis rather than a reparametrization of an existing one.
    _PREV_DAY_RANGE_BREAKOUT.name: _PREV_DAY_RANGE_BREAKOUT,
    _MACD_CROSS_TREND.name: _MACD_CROSS_TREND,
    _SWING_STRUCTURE_FADE.name: _SWING_STRUCTURE_FADE,
    _FVG_IMBALANCE_CONTINUATION.name: _FVG_IMBALANCE_CONTINUATION,
    _ORDER_BLOCK_REACTION.name: _ORDER_BLOCK_REACTION,
    _VOLATILITY_CONTRACTION_SQUEEZE.name: _VOLATILITY_CONTRACTION_SQUEEZE,
    _OVERNIGHT_GAP_FADE.name: _OVERNIGHT_GAP_FADE,
    _WMA_RIBBON_TREND.name: _WMA_RIBBON_TREND,
    _PCT_CHANGE_MOMENTUM_BURST.name: _PCT_CHANGE_MOMENTUM_BURST,
    _RSI_EXTREME_REVERSION.name: _RSI_EXTREME_REVERSION,
    # -- Prop-eval-shaped scalp expansion (Families U-Z): high win-rate /
    # tight RR / fast time-to-target hypotheses, built specifically for
    # the eval_pass_probability objective rather than raw net profit --
    # see each SkeletonSpec's own comment block above.
    _LIQUIDITY_SWEEP_QUICK_RECLAIM.name: _LIQUIDITY_SWEEP_QUICK_RECLAIM,
    _RANGE_MIDPOINT_FADE.name: _RANGE_MIDPOINT_FADE,
    _OPENING_RANGE_RETEST_CONFIRMATION.name: _OPENING_RANGE_RETEST_CONFIRMATION,
    _MICRO_PULLBACK_CONTINUATION.name: _MICRO_PULLBACK_CONTINUATION,
    _CHANGE_OF_CHARACTER_REVERSAL_SCALP.name: _CHANGE_OF_CHARACTER_REVERSAL_SCALP,
    _FVG_QUICK_FILL_FADE.name: _FVG_QUICK_FILL_FADE,
    # -- Expansion round 2 (8 new families): fills the previously-empty
    # relative_strength taxonomy group, plus 4 deliberate mirror-image
    # hypotheses of existing families (relative_strength_momentum vs
    # stat_pairs, vwap_trend_continuation vs vwap_reversion,
    # bollinger_band_walk_continuation vs mean_reversion_band,
    # gap_and_go_continuation vs overnight_gap_fade) and 3 new
    # independent mechanisms (volume_climax_reversal, volume_confirmed_
    # breakout, wma_sma_divergence_trend, higher_low_structure_continuation)
    # -- see each SkeletonSpec's own description for what makes it distinct.
    _RELATIVE_STRENGTH_MOMENTUM.name: _RELATIVE_STRENGTH_MOMENTUM,
    _VOLUME_CLIMAX_REVERSAL.name: _VOLUME_CLIMAX_REVERSAL,
    _VWAP_TREND_CONTINUATION.name: _VWAP_TREND_CONTINUATION,
    _BOLLINGER_BAND_WALK.name: _BOLLINGER_BAND_WALK,
    _GAP_AND_GO_CONTINUATION.name: _GAP_AND_GO_CONTINUATION,
    _VOLUME_CONFIRMED_BREAKOUT.name: _VOLUME_CONFIRMED_BREAKOUT,
    _WMA_SMA_DIVERGENCE_TREND.name: _WMA_SMA_DIVERGENCE_TREND,
    _HIGHER_LOW_STRUCTURE_CONTINUATION.name: _HIGHER_LOW_STRUCTURE_CONTINUATION,
    # -- New: a genuinely order-flow-specific family (participation spike
    # AND directional imbalance required together, not either alone) --
    # see its own comment block above for how it differs from Family F.
    _ORDER_FLOW_ABSORPTION.name: _ORDER_FLOW_ABSORPTION,
    # -- Expansion round 3 (multi-instrument search push): 5 more families,
    # each a proven base hypothesis plus exactly one new filter -- see each
    # SkeletonSpec's own comment block above for what makes it distinct.
    _ORDER_BLOCK_TREND_CONTINUATION.name: _ORDER_BLOCK_TREND_CONTINUATION,
    _VOLUME_CONFIRMED_TREND_PULLBACK.name: _VOLUME_CONFIRMED_TREND_PULLBACK,
    _SESSION_GATED_LIQUIDITY_SWEEP.name: _SESSION_GATED_LIQUIDITY_SWEEP,
    _MACD_HISTOGRAM_ZERO_CROSS_TREND.name: _MACD_HISTOGRAM_ZERO_CROSS_TREND,
    _ATR_REGIME_TREND_PULLBACK.name: _ATR_REGIME_TREND_PULLBACK,
    # -- Expansion round 4: 6 more families, each built on a primitive
    # this module already supported but no prior family actually used --
    # see the comment block above these six for the full rationale.
    _SESSION_EXTREME_FADE.name: _SESSION_EXTREME_FADE,
    _VWAP_BOLLINGER_PULLBACK.name: _VWAP_BOLLINGER_PULLBACK,
    _VOLUME_CONFIRMED_FVG_CONTINUATION.name: _VOLUME_CONFIRMED_FVG_CONTINUATION,
    _VOLUME_CONFIRMED_ORDER_BLOCK_REACTION.name: _VOLUME_CONFIRMED_ORDER_BLOCK_REACTION,
    _WIDE_RANGE_BAR_EXHAUSTION_FADE.name: _WIDE_RANGE_BAR_EXHAUSTION_FADE,
    _VOLUME_TREND_BREAKOUT_CONFIRMATION.name: _VOLUME_TREND_BREAKOUT_CONFIRMATION,
    # -- Expansion round 5 (7 new families, 7 new indicators: ADX, Stochastic,
    # CCI, OBV, Keltner Channel, Donchian Channel levels, SuperTrend) -- see
    # each SkeletonSpec's own comment block above for what makes it distinct.
    _ADX_TREND_STRENGTH_BREAKOUT.name: _ADX_TREND_STRENGTH_BREAKOUT,
    _STOCHASTIC_EXTREME_REVERSION.name: _STOCHASTIC_EXTREME_REVERSION,
    _CCI_EXTREME_REVERSION.name: _CCI_EXTREME_REVERSION,
    _OBV_DIVERGENCE_TREND_CONFIRMATION.name: _OBV_DIVERGENCE_TREND_CONFIRMATION,
    _KELTNER_SQUEEZE_BREAKOUT.name: _KELTNER_SQUEEZE_BREAKOUT,
    _DONCHIAN_CHANNEL_TURTLE_BREAKOUT.name: _DONCHIAN_CHANNEL_TURTLE_BREAKOUT,
    _SUPERTREND_TREND_FOLLOWING.name: _SUPERTREND_TREND_FOLLOWING,
    # -- Expansion round 6 (5 more families, 5 more indicators: Williams %R,
    # ROC, Awesome Oscillator, CMF, Parabolic SAR).
    _WILLIAMS_R_EXTREME_REVERSION.name: _WILLIAMS_R_EXTREME_REVERSION,
    _ROC_MOMENTUM_CONTINUATION.name: _ROC_MOMENTUM_CONTINUATION,
    _AWESOME_OSCILLATOR_ZERO_CROSS.name: _AWESOME_OSCILLATOR_ZERO_CROSS,
    _CMF_VOLUME_CONFIRMATION.name: _CMF_VOLUME_CONFIRMATION,
    _PARABOLIC_SAR_TREND_FOLLOWING.name: _PARABOLIC_SAR_TREND_FOLLOWING,
}

# Families that need something beyond the plain OHLCV df -- checked by
# generate_search_space() and by batch_runner.run_search() so a family
# needing "pair_close" merged in first fails with one clear message up
# front, instead of quietly producing zero-trade candidates for every
# single grid point.
FAMILIES_REQUIRING_PAIR_DATA = {name for name, spec in FAMILIES.items() if spec.requires_pair_data}


def list_families() -> dict[str, str]:
    """name -> human-readable label, for a UI dropdown / CLI --list-families."""
    return {name: spec.label for name, spec in FAMILIES.items()}


def family_description(name: str) -> str:
    if name not in FAMILIES:
        raise StrategySpaceError(f"Unknown strategy family '{name}'. Known families: {list(FAMILIES)}")
    return FAMILIES[name].description


# ---------------------------------------------------------------------------
# Hypothesis questions -- Forge Strategy (app.orchestration.forge) surfaces
# these verbatim so every candidate a run generates is traceable back to a
# plain-English, falsifiable market question ("after X, does Y tend to
# happen?"), not just a family label. Every family above was already built
# around exactly this kind of question (see each SkeletonSpec's
# `description` and the comment block above it) -- this dict just phrases
# that same intent as an explicit question for the UI/report, instead of
# requiring a person to infer it from the family's descriptive prose.
#
# Deliberately hand-written per family (not derived from `description`
# programmatically): a good hypothesis question names the SETUP and the
# EXPECTED behavior in one sentence, which the prose descriptions don't
# consistently do in a mechanically-extractable way. Falls back to
# `family_description` for any family not yet given an explicit entry here
# (e.g. a family added later) -- see `hypothesis_question` below -- so this
# dict is safe to extend incrementally and never raises for a valid family.
# ---------------------------------------------------------------------------
HYPOTHESIS_QUESTIONS: dict[str, str] = {
    "trend_breakout": "When price breaks an N-bar range in the direction of a slower EMA trend, does it continue?",
    "mtf_pullback": "When price pulls back (RSI dip/pop) inside an established EMA trend, does it resume the trend?",
    "mean_reversion_band": "When price closes outside a mean-reversion band, does it revert back toward the mean?",
    "volatility_breakout": "When an N-bar range breaks while ATR is expanding, does the move continue?",
    "session_time_effect": "Does price behave differently (directionally) during a specific session/time window?",
    "volume_imbalance": "When volume imbalance confirms a directional move, does price continue in that direction?",
    "stat_pairs": "When the spread between two correlated instruments diverges, does it revert?",
    "liquidity_sweep_reversal": "After a liquidity sweep (a stop-hunt through a swing high/low that reclaims), does price reverse?",
    "momentum_continuation": "When RSI extremity is confirmed by MACD histogram agreement, does the momentum persist?",
    "vwap_reversion": "When price stretches far from VWAP, does it revert back toward VWAP?",
    "market_structure_shift": "After a break of market structure (higher-high/lower-low shift), does the new direction persist?",
    "prev_day_range_breakout": "When price breaks yesterday's high or low, does it continue in that direction?",
    "macd_cross_trend": "When MACD crosses its signal line in the direction of a slower EMA trend, does price follow through?",
    "swing_structure_fade": "When price makes a failed swing-structure attempt, does it fade back the other way?",
    "fvg_imbalance_continuation": "When price returns to fill a fair-value gap within a trend, does the trend continue?",
    "order_block_reaction": "When price returns to an order block, does it react (reverse) from that level?",
    "volatility_contraction_squeeze": "After a volatility squeeze (ATR contraction), does the eventual breakout continue?",
    "overnight_gap_fade": "When price gaps away from the prior close at the session open, does the gap tend to fade?",
    "wma_ribbon_trend": "When a WMA ribbon aligns in one direction, does price continue trending with it?",
    "pct_change_momentum_burst": "After a sharp percentage-change burst, does price continue in that direction?",
    "rsi_extreme_reversion": "When RSI reaches an extreme reading, does price mean-revert?",
    "liquidity_sweep_quick_reclaim": "After a fast liquidity sweep with an immediate reclaim, does price reverse quickly?",
    "range_midpoint_fade": "When price is stretched away from a range's midpoint, does it fade back toward the midpoint?",
    "opening_range_retest_confirmation": "After a breakout of the opening range retests and confirms the level, does it continue?",
    "micro_pullback_continuation": "During an active trend, does a shallow micro-pullback resolve back in the trend's direction?",
    "change_of_character_reversal_scalp": "After a short-term change of character (CHoCH), does price reverse for a quick scalp?",
    "fvg_quick_fill_fade": "When price quickly fills a fair-value gap against the prevailing move, does it fade?",
    "relative_strength_momentum": "When one instrument shows relative strength versus its pair, does that strength persist?",
    "volume_climax_reversal": "After a volume climax (exhaustion spike), does price reverse?",
    "vwap_trend_continuation": "When price holds above/below VWAP in a trend, does the trend continue?",
    "bollinger_band_walk_continuation": "When price 'walks the band' (repeated closes near a Bollinger band), does the trend persist?",
    "gap_and_go_continuation": "When price gaps at the open and immediately continues in the gap's direction, does it keep going?",
    "volume_confirmed_breakout": "When a breakout is confirmed by above-average volume, does it continue further than an unconfirmed one?",
    "wma_sma_divergence_trend": "When a fast WMA diverges from a slower SMA, does price continue in the direction of the divergence?",
    "higher_low_structure_continuation": "During an uptrend/downtrend, does a higher-low (or lower-high) structure event predict continuation?",
    "order_flow_absorption": "When aggressive volume is absorbed at a level without price breaking through, does price reverse?",
    "order_block_trend_continuation": "When price reacts off an order block that aligns with the prevailing trend, does the trend continue?",
    "volume_confirmed_trend_pullback": "When a trend pullback is confirmed by volume, does the trend resume?",
    "session_gated_liquidity_sweep": "Does a liquidity-sweep reversal work better when it's gated to a specific trading session?",
    "macd_histogram_zero_cross_trend": "When the MACD histogram crosses zero in the direction of a slower trend, does price follow through?",
    "atr_regime_trend_pullback": "Does a trend-pullback entry perform differently depending on the prevailing ATR (volatility) regime?",
    "session_extreme_fade": "When price reaches a session's high/low extreme, does it fade back toward the session's range?",
    "vwap_bollinger_pullback": "When price pulls back to VWAP inside a Bollinger band, does it resume the prevailing trend?",
    "volume_confirmed_fvg_continuation": "When a fair-value-gap fill is confirmed by volume, does the underlying trend continue?",
    "volume_confirmed_order_block_reaction": "When an order-block reaction is confirmed by volume, is the reversal more reliable?",
    "wide_range_bar_exhaustion_fade": "After an unusually wide-range bar (exhaustion), does price fade back against that bar's direction?",
    "volume_trend_breakout_confirmation": "Does a trend breakout confirmed by a volume surge outperform one without volume confirmation?",
    "adx_trend_strength_breakout": "Does a trend-aligned breakout perform better when ADX confirms the market is actually trending strongly?",
    "stochastic_extreme_reversion": "When Stochastic %K crosses back from an oversold/overbought extreme, does price revert?",
    "cci_extreme_reversion": "When CCI reaches an unbounded extreme reading, does price mean-revert?",
    "obv_divergence_trend_confirmation": "When On-Balance Volume's own trend already agrees with a structure breakout, does the breakout hold up better?",
    "keltner_squeeze_breakout": "When price clears an ATR-based Keltner Channel band, does the move continue?",
    "donchian_channel_turtle_breakout": "Does the classic turtle system (enter on a long Donchian channel, exit on a shorter one) still work?",
    "supertrend_trend_following": "When SuperTrend's ratcheting trailing-stop line flips direction, does the new trend persist?",
    "williams_r_extreme_reversion": "When Williams %R reaches an extreme on its own -100..0 scale, does price revert?",
    "roc_momentum_continuation": "When Rate of Change clears a threshold, does the raw percentage-change momentum continue?",
    "awesome_oscillator_zero_cross": "When the Awesome Oscillator crosses its zero line, does the new momentum direction continue?",
    "cmf_volume_confirmation": "Does a breakout confirmed by Chaikin Money Flow already leaning the same direction outperform one without it?",
    "parabolic_sar_trend_following": "When Parabolic SAR's accelerating trailing stop flips, does the new trend persist?",
}


def hypothesis_question(name: str) -> str:
    """Plain-English 'does X tend to happen after Y' question for a family
    -- falls back to that family's `description` (never raises for any
    valid family name) so a newly-added family without an explicit entry
    above still renders something sensible instead of a KeyError."""
    if name not in FAMILIES:
        raise StrategySpaceError(f"Unknown strategy family '{name}'. Known families: {list(FAMILIES)}")
    return HYPOTHESIS_QUESTIONS.get(name) or family_description(name)


def family_grid_size(name: str) -> int:
    """Full (pre-sampling) candidate count for one family -- lets the UI show a preview count."""
    if name not in FAMILIES:
        raise StrategySpaceError(f"Unknown strategy family '{name}'. Known families: {list(FAMILIES)}")
    return len(FAMILIES[name].combinations())


# ---------------------------------------------------------------------------
# Grid-around-a-given-strategy (Manual, Python, PineScript, or MQL5)
#
# Reuses the exact same parameter-discovery machinery Step 6's Iterative
# Refinement GA already uses (app.optimize.parameter_space for Manual,
# app.optimize.code_parameter_space for the three code sources) -- both
# gene types expose the same .lo / .hi / .is_int / .base_value / .label
# shape, so the discretization and Cartesian-product logic below is
# entirely source-type-agnostic; only the final "apply a genome" and
# "what does a candidate look like" steps differ.
# ---------------------------------------------------------------------------

def _genes_for_strategy(strategy: Strategy) -> list:
    if strategy.source_type == "manual":
        return extract_genome(strategy.config)
    if strategy.source_type in _CODE_SOURCE_TYPES:
        return discover_code_genes(strategy)
    raise StrategySpaceError(f"Unsupported strategy source type '{strategy.source_type}'.")


def _discretize_gene(gene, n_points: int) -> list[float]:
    """n_points evenly-spaced values across [gene.lo, gene.hi], rounded for
    integer genes and de-duplicated (rounding can collapse points for a
    narrow integer range, e.g. a period whose search range is only 2-4)."""
    n_points = max(int(n_points), 1)
    lo, hi = gene.lo, gene.hi
    if n_points == 1 or hi <= lo:
        raw = [gene.base_value]
    else:
        raw = list(np.linspace(lo, hi, n_points))
    values = [float(round(v)) if gene.is_int else float(v) for v in raw]

    seen: set[float] = set()
    out: list[float] = []
    for v in values:
        key = round(v, 8)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out or [float(gene.base_value)]


def _grid_combinations(genes: list, n_points: int) -> list[list[float]]:
    per_gene_values = [_discretize_gene(g, n_points) for g in genes]
    return [list(combo) for combo in itertools.product(*per_gene_values)]


def _apply_generic_genome(source_type: str, base: Any, genes: list, genome: list[float]) -> Any:
    """Returns a new Manual config dict (manual) or patched source text
    (python/pinescript/mql5) with `genome` written into `base` at the
    positions `genes` describes."""
    if source_type == "manual":
        return apply_genome(base, genes, genome)
    return apply_code_genome(base, genes, genome)


def _grid_space_around_strategy(
    strategy: Strategy, grid_points_per_gene: int, max_candidates: int, seed: int,
) -> SearchSpace:
    genes = _genes_for_strategy(strategy)
    if not genes:
        raise StrategySpaceError(
            f"No tunable numeric parameters were found on this {strategy.source_type} strategy to "
            "grid-search. Manual needs at least one indicator period or a Fixed/ATR stop-loss/"
            "take-profit value; Python needs a top-level SCREAMING_SNAKE_CASE numeric constant; "
            "PineScript needs an input.int()/input.float() value; MQL5 needs an iMA()/iRSI() period "
            "or a T58_SL_PIPS/T58_TP_PIPS directive."
        )

    base = strategy.config if strategy.source_type == "manual" else _source_text_for_strategy(strategy)
    combos = _grid_combinations(genes, grid_points_per_gene)

    total_generated = len(combos)
    sampled = False
    if total_generated > max_candidates:
        rng = random.Random(seed)
        combos = rng.sample(combos, max_candidates)
        sampled = True

    family_label = f"{strategy.source_type}_grid"
    candidates: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for genome in combos:
        applied = _apply_generic_genome(strategy.source_type, base, genes, genome)
        if strategy.source_type == "manual":
            spec = {"source_type": "manual", "config": applied}
            digest_source = json.dumps(applied, sort_keys=True, default=str)
        else:
            spec = {
                "source_type": strategy.source_type, "code_text": applied,
                "code_extension": _CODE_EXTENSIONS[strategy.source_type],
            }
            digest_source = applied
        digest = hashlib.sha1(digest_source.encode()).hexdigest()[:10]
        cid = f"{family_label}-{digest}"
        candidates[cid] = spec
        meta[cid] = {"family": family_label, "params": {g.label: v for g, v in zip(genes, genome)}}

    return SearchSpace(
        mode="family", family=family_label,
        candidates=candidates, meta=meta,
        total_generated=total_generated, sampled=sampled,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_search_space(
    mode: str,
    family: str | None = None,
    single_config: dict | None = None,
    strategy: Strategy | None = None,
    max_candidates: int = 2000,
    seed: int = 42,
    grid_points_per_gene: int = 3,
    has_pair_data: bool = False,
    exclude_families: "set[str] | None" = None,
) -> SearchSpace:
    """
    mode="single":
        strategy=<a built Strategy instance> -- wraps it (Manual, Python,
            PineScript, or MQL5 alike) as a size-1 space. Preferred over
            single_config for anything that isn't Manual.
        single_config=<dict> -- legacy path: wraps a Manual Strategy
            config dict directly, without needing a live Strategy
            instance. Kept for backward compatibility (e.g. --cli's
            DEFAULT_MANUAL_STRATEGY). Ignored if `strategy` is given.

    mode="family":
        strategy=<a built Strategy instance> -- grid-searches THAT
            strategy's own tunable numeric parameters (any source type).
            `family` is ignored when `strategy` is given.
        family=<name> or "all" or None -- (Manual only) expands one named
            hypothesis family, or every family, into its full parameter
            grid. Used only when `strategy` is not given.

    In both family paths, if the full grid exceeds `max_candidates`, a
    random (seeded, reproducible) sample of exactly `max_candidates`
    combinations is taken instead of just the first N in itertools.product
    order -- an arbitrary "first N" slice systematically favors whatever
    the first grid dimension happens to be, which biases the search before
    it even starts.

    exclude_families: only ever applied when family is None/"all" (never
        overrides an explicit single-family request) -- drops any of
        those names from the "every family" set, e.g. dead-end families
        from app.search.family_health.apply_family_exclusions(). Falls
        back to searching everything if excluding these would leave zero
        families, same safety rule apply_family_exclusions itself uses.
    """
    if mode == "single":
        if strategy is not None:
            spec = spec_from_strategy(strategy)
        elif single_config is not None:
            if not isinstance(single_config, dict):
                raise StrategySpaceError(
                    "single_config must be a Manual Strategy config dict (or pass `strategy` instead "
                    "for a non-Manual strategy)."
                )
            spec = {"source_type": "manual", "config": copy.deepcopy(single_config)}
        else:
            raise StrategySpaceError(
                "mode='single' requires either `strategy` (a built Strategy instance of any "
                "supported source type) or single_config (a Manual Strategy config dict)."
            )
        cid = "single-00000"
        return SearchSpace(
            mode="single", family=None,
            candidates={cid: spec},
            meta={cid: {"family": "single", "params": {}}},
            total_generated=1, sampled=False,
        )

    if mode != "family":
        raise StrategySpaceError(f"Unknown search mode '{mode}' (expected 'single' or 'family').")

    if strategy is not None:
        return _grid_space_around_strategy(strategy, grid_points_per_gene, max_candidates, seed)

    families_to_run = list(FAMILIES.keys()) if family in (None, "all") else [family]
    for fam in families_to_run:
        if fam not in FAMILIES:
            raise StrategySpaceError(f"Unknown strategy family '{fam}'. Known families: {list(FAMILIES)}")

    if exclude_families and family in (None, "all"):
        # Only ever applied to an "every family" request, same as the
        # pair-data skip right below -- an explicit single-family request
        # is never second-guessed, even if that family happens to be
        # flagged. Never allowed to empty the space entirely (mirrors
        # app.search.family_health.apply_family_exclusions' own safety
        # rule) -- if literally everything would be excluded, search
        # everything instead of raising "zero valid candidates" below.
        survivors = [f for f in families_to_run if f not in exclude_families]
        if survivors:
            families_to_run = survivors

    if not has_pair_data:
        requested_pair_families = [f for f in families_to_run if f in FAMILIES_REQUIRING_PAIR_DATA]
        if requested_pair_families:
            if family in (None, "all"):
                # Searching "all" families with no pair data merged in: skip the
                # pair-only families rather than failing the entire search --
                # every other family still works fine on plain OHLCV data.
                families_to_run = [f for f in families_to_run if f not in FAMILIES_REQUIRING_PAIR_DATA]
            else:
                raise StrategySpaceError(
                    f"Family '{family}' requires a second instrument's price merged into the "
                    "working data first (see app.data.pairs.merge_pair_series) and "
                    "has_pair_data=True passed here. Merge pair data before searching this family."
                )

    combos_by_family: dict[str, list[dict]] = {
        fam: list(FAMILIES[fam].combinations()) for fam in families_to_run
    }
    total_generated = sum(len(v) for v in combos_by_family.values())
    sampled = False
    if total_generated == 0:
        raise StrategySpaceError("The requested family/grid produced zero valid parameter combinations.")

    rng = random.Random(seed)
    if total_generated <= max_candidates:
        all_items = [(fam, params) for fam, params_list in combos_by_family.items() for params in params_list]
    else:
        # BALANCED per-family sampling ("water-filling"), not a single
        # pooled random.sample() over every family's combinations
        # concatenated together. A pooled sample is size-biased: a family
        # whose grid happens to have 1024 combinations (e.g. it exposes
        # more tunable parameters) is ~85x more likely to appear in the
        # sample than a family with only 12, which has nothing to do with
        # which hypothesis is actually promising -- it's purely an
        # artifact of how many knobs that family's config happens to
        # expose. Left uncorrected, this is exactly what starves Evolution
        # Lab's and Search Lab's random-immigrant/fresh-candidate slots
        # down to just the two or three biggest-grid families over time
        # (compounded further by elite/breeding selection, which is
        # capped separately -- see EvolutionConfig.max_elite_frac_per_family).
        #
        # Water-filling: give every family an equal slice of max_candidates;
        # any family smaller than its slice contributes everything it has
        # and its leftover slice is redistributed evenly across the
        # families that still have room, repeated until the budget is
        # fully assigned or every family is exhausted.
        fams_sorted = sorted(combos_by_family.keys(), key=lambda f: len(combos_by_family[f]))
        allocation: dict[str, int] = {fam: 0 for fam in fams_sorted}
        remaining_quota = max_candidates
        if remaining_quota < len(fams_sorted):
            # BUGFIX: the budget can't afford even one candidate per family.
            # The old code below (`share = max(1, remaining_quota //
            # remaining_fam_count)`) forced every remaining family to get AT
            # LEAST 1 regardless of how small max_candidates was, which
            # silently blew straight through the cap whenever there were
            # more families than budget -- e.g. max_candidates=4 across 45
            # registered families returned 45 candidates (one per family,
            # every one of them ignoring the caller's cap), not 4. Instead,
            # pick a random (seeded, reproducible) subset of exactly
            # `remaining_quota` families and give each of those one
            # candidate, so the total never exceeds max_candidates.
            chosen_fams = rng.sample(fams_sorted, remaining_quota)
            for fam in chosen_fams:
                allocation[fam] = 1
            remaining_quota = 0
            remaining_fam_count = 0
        else:
            remaining_fam_count = len(fams_sorted)
            for fam in fams_sorted:
                share = max(1, remaining_quota // remaining_fam_count)
                take = min(len(combos_by_family[fam]), share)
                allocation[fam] = take
                remaining_quota -= take
                remaining_fam_count -= 1
        # Any quota left over (every family capped below its equal share,
        # or integer-division remainder) goes to the families with the
        # most untapped combinations left, so the budget is still fully
        # used rather than silently under-sampling.
        if remaining_quota > 0:
            headroom = sorted(
                ((fam, len(combos_by_family[fam]) - allocation[fam]) for fam in fams_sorted),
                key=lambda x: x[1], reverse=True,
            )
            for fam, room in headroom:
                if remaining_quota <= 0:
                    break
                extra = min(room, remaining_quota)
                allocation[fam] += extra
                remaining_quota -= extra

        all_items = []
        for fam, take in allocation.items():
            params_list = combos_by_family[fam]
            chosen = params_list if take >= len(params_list) else rng.sample(params_list, take)
            all_items.extend((fam, params) for params in chosen)
        sampled = True

    candidates: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for fam, params in all_items:
        # Content-addressed, not positional: the ID is derived from the
        # family name + the exact parameter combination, so the same
        # combination always gets the same ID regardless of sample order or
        # seed, and two different combinations can never collide onto the
        # same ID. A positional index (e.g. "trend_breakout-00007") would
        # mean the ID's meaning depends on *which run* produced it -- two
        # differently-seeded runs would reuse the same IDs for different
        # underlying strategies, which is both confusing to a person reading
        # the leaderboard and wrong for the results DB's resumability goal
        # (see app/search/results_db.py's module docstring).
        digest = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]
        cid = f"{fam}-{digest}"
        candidates[cid] = {"source_type": "manual", "config": FAMILIES[fam].build(params)}
        meta[cid] = {"family": fam, "params": params}

    return SearchSpace(
        mode="family",
        family=(family if family not in (None, "all") else "all"),
        candidates=candidates, meta=meta,
        total_generated=total_generated, sampled=sampled,
    )
