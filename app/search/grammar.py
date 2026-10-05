"""
Compositional strategy grammar (v5, B1-1).

This is the first path in this app that can INVENT strategy structure --
add/remove/swap conditions, flip AND<->OR, graft subtrees between parents --
instead of only tuning numbers on the 84 frozen SkeletonSpecs in
app.search.strategy_space (whose GA/TPE/CMA-ES optimizers mutate numeric
leaves only; see app.optimize.parameter_space.GENE_KEY_RULES).

The grammar generates plain Manual strategy config dicts -- the exact same
schema app.strategy.manual.ManualStrategy already consumes
(entry_conditions.{long,short} condition lists + connectors, exit_conditions,
filters, risk_management) -- so anything it produces round-trips through
the production builder with zero translation layer, and `validate()` below
proves that by actually running the builder.

Deliberate exclusions (documented, not accidental):
- "ichimoku_chikou": the Oct 3 audit verified a real lookahead leak here
  (close.shift(-26) reachable from the Manual builder while the lookahead
  gate skips manual strategies). The grammar never emits it. If a future
  fix makes chikou safe, register it via register_operand_kind().
- "pair_ratio"/"pair_zscore": require a "pair_close" column merged into the
  working DataFrame (see app.data.pairs.merge_pair_series) -- a grammar
  config must be self-contained. ("correlation" is excluded for the same
  reason -- rolling_correlation needs the same merged column.)
- "news_minutes_since_high_impact"/"news_minutes_until_high_impact":
  require merged economic-calendar columns -- same reason.

EXTENSION POINT (future terminals): OPERAND_BUILDERS maps an operand-kind
name -> builder callable (rng -> operand dict). The astra port list
(regime-adaptive RSI zones, causal divergence detector, VWAP value-area
bands, time-invalidation stop) lands here as new terminals, NOT as new
frozen templates -- call register_operand_kind(name, builder) and the
grammar, the structural operators, and validate() all pick it up
automatically.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.strategy.indicators import BOUNDED_OSCILLATOR_RANGES
from app.strategy.manual import ManualStrategy

# ---------------------------------------------------------------------------
# Operand terminal sets.
#
# Provenance: copied from the dispatch table in
# app.strategy.manual.ManualStrategy._series_from_operand (the kinds that
# route to build_indicator_series), plus the boolean/session primitives the
# same method handles explicitly. A test asserts every kind listed here is
# dispatchable by the real builder (the 50-config round-trip test).
# ---------------------------------------------------------------------------

# Raw market-data terminals.
PRICE_KINDS = ("price", "open", "high", "low", "close", "volume")

# Numeric indicator terminals (build_indicator_series kinds), minus the
# deliberate exclusions documented in the module docstring.
INDICATOR_KINDS = (
    "ema", "sma", "wma", "rsi", "vwap", "macd", "macd_signal", "macd_histogram",
    "atr", "bollinger_mid", "bollinger_upper", "bollinger_lower",
    "highest_high", "lowest_low", "average_volume", "candle_range",
    "percentage_change", "relative_volume", "volume_delta",
    "adx", "stoch_k", "stoch_d", "cci", "obv", "obv_ema",
    "keltner_mid", "keltner_upper", "keltner_lower",
    "donchian_mid", "donchian_upper", "donchian_lower",
    "supertrend_line", "supertrend_direction",
    "williams_r", "roc", "awesome_oscillator", "cmf",
    "psar_line", "psar_direction",
    "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b",
    "fib_382", "fib_500", "fib_618",
    "pivot_point", "pivot_r1", "pivot_s1", "pivot_r2", "pivot_s2",
    "heikin_ashi_open", "heikin_ashi_high", "heikin_ashi_low", "heikin_ashi_close",
    "mfi", "trix", "ultimate_oscillator",
    "aroon_up", "aroon_down", "aroon_oscillator",
    "choppiness_index", "dpo", "anchored_vwap",
    "linreg_mid", "linreg_upper", "linreg_lower",
    "chandelier_long", "chandelier_short",
    "volume_profile_poc", "volume_profile_vah", "volume_profile_val",
    "hma", "dema", "tema", "kama",
    "vortex_plus", "vortex_minus",
    "elder_bull_power", "elder_bear_power",
    "ttm_squeeze_on", "ttm_squeeze_momentum",
    "vwma", "adl", "chaikin_oscillator",
    "fisher_transform", "fisher_transform_signal",
    "connors_rsi", "adr",
    "swing_bos", "swing_choch",
    # v7 (worker D, 2026-10-05): new indicator kinds from
    # app.strategy.indicators_v7 (StochRSI, Wilder +/-DI, linear-regression
    # slope, Hurst exponent, KST, Coppock). Dispatched via the v7 fallback
    # in build_indicator_series and routed by ManualStrategy's
    # _series_from_operand -- the round-trip test below proves each is
    # dispatchable. The Workstream-C list members that already had
    # builders (Supertrend, TTM Squeeze, Connors RSI, Williams %R, CCI,
    # ADX, Aroon, Keltner, Donchian, Parabolic SAR, Heikin-Ashi,
    # linreg channels, Choppiness Index, Elder Ray, TRIX, DPO, VWAP bands)
    # were already registered above and are unchanged.
    "stochrsi_k", "stochrsi_d",
    "plus_di", "minus_di",
    "linreg_slope", "hurst_exponent",
    "kst", "coppock",
    # w10-astra terminals (Oct-4 analysis, Part A ports #2/#3) -- regime-
    # adaptive RSI zones + session VWAP profile levels. NOT
    # build_indicator_series kinds: they need operand-level parameters
    # (zone_bounds/div_lookback/trend window; roll_hour/band sigmas/
    # poc_buckets) that the (frame, kind, period, column, lookback)
    # signature cannot carry, so ManualStrategy._series_from_operand
    # dispatches them via _regime_oscillator_series/_vwap_profile_series
    # (see app.strategy.manual and app.quant_lab.regime_oscillators /
    # app.quant_lab.vwap_profile). They live in this tuple (and hence in
    # NUMERIC_KINDS below) so random_operand() can draw them and
    # swap_operand_kind() can swap to them like any numeric terminal.
    # rsi_regime / rsi_divergence are TRISTATE (+1/-1/0) -- see
    # THRESHOLD_BOUNDS below for how thresholds against them are drawn.
    "rsi_regime", "rsi_divergence",
    "session_vwap", "vwap_sigma", "vwap_vah", "vwap_val",
    "vwap_upper_2", "vwap_lower_2", "vwap_poc", "vwap_zscore",
)

# Boolean terminals -- used with "is true"/"is false" (or == 0/1).
BOOLEAN_KINDS = (
    "time_of_day", "day_of_week", "candle_direction",
    "swing_high", "swing_low",
    "liquidity_sweep", "break_of_structure", "bos",
    "change_of_character", "choch",
    "fair_value_gap", "fvg", "order_block",
    "atr_regime", "volatility_regime",
    # w10-astra: regime-oscillator zone flags + VWAP position flags --
    # 1.0/0.0 per-bar series, safe for "is true"/"is false".
    "rsi_zone_buy", "rsi_zone_sell",
    "vwap_above", "vwap_below", "vwap_outside_value_area",
    # A6: atr_regime / volatility_regime are TRISTATE (+1 expansion /
    # -1 contraction / 0 neutral). "is true" on the raw tristate treats
    # -1 as true (bool(-1) is True), so a contraction gate written
    # against atr_regime would also fire on expansions. Rather than
    # change global "is true" semantics in manual.py, each leg gets its
    # own dedicated 1/0 terminal (dispatched in manual.py via
    # _regime_series; registered here like any boolean terminal).
    "atr_expansion", "atr_contraction",
)

# Session-level numeric terminals (need a session window; numeric output).
SESSION_KINDS = (
    "session_high", "session_low",
    "previous_day_high", "previous_day_low", "previous_day_close",
    "opening_range_high", "opening_range_low",
    "ib_contraction_ratio",
)

NUMERIC_KINDS = PRICE_KINDS + INDICATOR_KINDS + SESSION_KINDS
CONSTANT_KINDS = ("value", "constant", "number")
ALL_OPERAND_KINDS = set(NUMERIC_KINDS) | set(BOOLEAN_KINDS) | set(CONSTANT_KINDS)

# ---------------------------------------------------------------------------
# A6: grammar-local threshold bounds. BOUNDED_OSCILLATOR_RANGES
# (app.strategy.indicators) is the app-wide map; this module extends it
# for the kinds it draws random thresholds for: adx / aroon_up /
# aroon_down are 0-100 oscillators, cmf is a -1..1 money-flow
# oscillator, and the astra rsi_regime / rsi_divergence terminals are
# tristate (-1/0/+1) flags. Deliberately LOCAL to this module (not a
# change to app.strategy.indicators): the bounded-threshold rule is
# enforced by THIS module's random_threshold() (draw) and validate()
# (dead-code rejection); the app-wide gene-mutation bounds in
# app.optimize.parameter_space keep their own map untouched.
# ---------------------------------------------------------------------------
THRESHOLD_BOUNDS: dict[str, tuple[float, float]] = {
    **BOUNDED_OSCILLATOR_RANGES,
    "adx": (0.0, 100.0),
    "aroon_up": (0.0, 100.0),
    "aroon_down": (0.0, 100.0),
    "cmf": (-1.0, 1.0),
    "rsi_regime": (-1.0, 1.0),
    "rsi_divergence": (-1.0, 1.0),
    # v7 (worker D): bounded 0-100 oscillators among the new
    # indicators_v7 kinds -- see V7_BOUNDED_RANGES there. Kept local to
    # this module for the same reason as the entries above.
    "stochrsi_k": (0.0, 100.0),
    "stochrsi_d": (0.0, 100.0),
    "plus_di": (0.0, 100.0),
    "minus_di": (0.0, 100.0),
}

OPERATORS = (">", ">=", "<", "<=", "cross above", "cross below", "is true", "is false")
CONNECTORS = ("AND", "OR")

# Every operator app.strategy.manual.ManualStrategy._compare dispatches --
# validate() accepts all of these (the grammar itself only EMITS the
# canonical OPERATORS above, but hand-built/template configs using the
# aliases are perfectly valid builder input and must not be rejected).
VALID_OPERATORS = frozenset({
    ">", "greater than", "gt",
    ">=", "greater than or equal", "gte",
    "<", "less than", "lt",
    "<=", "less than or equal", "lte",
    "==", "equal to", "equals", "eq",
    "!=", "not equal", "neq",
    "cross above", "crosses above",
    "cross below", "crosses below",
    "is true", "true",
    "is false", "false",
})

# Comparators that make sense between two numerics.
_NUMERIC_OPS = (">", ">=", "<", "<=", "cross above", "cross below")
_MIRROR_OP = {">": "<", ">=": "<=", "<": ">", "<=": ">=",
              "cross above": "cross below", "cross below": "cross above"}

# Kinds whose "period" is really a different window semantic -- kept so
# generated operands pass the right knob name through.
_LOOKBACK_KINDS = {
    "swing_high", "swing_low", "liquidity_sweep", "break_of_structure", "bos",
    "change_of_character", "choch", "fair_value_gap", "fvg", "order_block",
    "swing_bos", "swing_choch", "ib_contraction_ratio",
}

# ---------------------------------------------------------------------------
# v7 (worker D): multi-timeframe grammar support.
#
# Feasibility verdict (2026-10-05): feasible WITHOUT new machinery. The
# app already has a lookahead-safe MTF pipeline -- operand-level
# "timeframe" keys are honored by ManualStrategy._series_from_operand
# (MTF-STRATEGY-001), and app.data.timeframe_resample.
# prepare_timeframe_aligned_data computes the HTF indicator on genuine
# native-frequency bars (never the base timeframe's upsampled values)
# and merges it on as a tfNN_ column before generate() ever runs. The
# grammar just needs to (a) occasionally EMIT the timeframe key and
# (b) run the same data-prep in validate()'s round-trip. Both are done
# here, guarded by _MTF_PROBABILITY so the emission rate is tunable and
# generate_random(allow_mtf=False) turns it off entirely.
# ---------------------------------------------------------------------------
MTF_CONTEXT_TIMEFRAMES: tuple[str, ...] = ("1h", "4h", "1d")

_MTF_PROBABILITY: float = 0.08


def set_mtf_probability(p: float) -> float:
    """Set the per-indicator-operand probability of emitting a coarser
    context timeframe (0.0 = never, 1.0 = always). Returns the previous
    value. generate_random(allow_mtf=False) is the scoped alternative."""
    global _MTF_PROBABILITY
    prev = _MTF_PROBABILITY
    _MTF_PROBABILITY = float(min(max(p, 0.0), 1.0))
    return prev


def enable_mtf_grammar(p: float = 0.08) -> None:
    """Opt in to MTF operand emission at probability `p` (the default)."""
    set_mtf_probability(p)


def disable_mtf_grammar() -> None:
    """Turn MTF operand emission off entirely."""
    set_mtf_probability(0.0)


# Indicator kinds MTF emission skips: raw price columns (a bare HTF price
# is rarely the useful bias primitive -- the HTF *indicator* is), session
# levels (their session_start/session_end semantics are base-timeframe),
# and the lookback-window structure detectors.
_MTF_EXCLUDED_KINDS = frozenset(PRICE_KINDS) | frozenset(SESSION_KINDS) | _LOOKBACK_KINDS

# ---------------------------------------------------------------------------
# Extension point: operand terminal registry.
#
# Maps kind name -> builder(rng) -> operand dict. The built-in kinds are
# registered below; future terminals (regime RSI zones, divergence, VWAP
# value-area, ...) register here and are immediately usable by
# generate_random(), the structural operators, and validate().
# ---------------------------------------------------------------------------

OperandBuilder = Callable[[random.Random], dict]
OPERAND_BUILDERS: dict[str, OperandBuilder] = {}


def register_operand_kind(name: str, builder: OperandBuilder) -> None:
    """Register a new operand terminal for the grammar.

    `builder` receives a random.Random and returns an operand dict the
    Manual builder can dispatch (e.g. {"type": name, ...}). After
    registration the kind is usable by generate_random(), every structural
    operator, and validate(). Re-registering an existing name replaces it.
    """
    if not callable(builder):
        raise TypeError("operand builder must be callable")
    OPERAND_BUILDERS[str(name).lower().strip()] = builder


def _operand(kind: str, rng: random.Random, **overrides: Any) -> dict:
    """Build one operand dict for `kind`, sampling its knobs from `rng`."""
    kind = str(kind).lower().strip()
    op: dict[str, Any] = {"type": kind}
    if kind in _LOOKBACK_KINDS:
        op["lookback"] = int(overrides.get("lookback", rng.choice([5, 10, 14, 20, 30])))
    elif kind in NUMERIC_KINDS and kind not in PRICE_KINDS:
        op["period"] = int(overrides.get("period", rng.choice([5, 8, 10, 14, 20, 21, 50, 100, 200])))
        op["field"] = overrides.get("field", "close")
    if kind in ("candle_direction", "liquidity_sweep", "break_of_structure", "bos",
                "change_of_character", "choch", "fair_value_gap", "fvg", "order_block"):
        op["direction"] = overrides.get("direction", rng.choice(["both", "bullish", "bearish"]))
    if kind in ("day_of_week",):
        op["days"] = overrides.get("days", sorted(rng.sample(range(7), rng.randint(1, 3))))
    if kind in ("time_of_day",) or kind in SESSION_KINDS:
        op["session_start"] = overrides.get("session_start", rng.choice(["08:30", "09:30", "02:00", "18:00"]))
        op["session_end"] = overrides.get("session_end", rng.choice(["11:00", "15:00", "10:00", "23:59"]))
    if kind in ("atr_regime", "volatility_regime", "atr_expansion", "atr_contraction"):
        op["period"] = int(overrides.get("period", rng.choice([10, 14, 20])))
        op["expansion_mult"] = overrides.get("expansion_mult", round(rng.uniform(1.1, 1.5), 2))
        op["contraction_mult"] = overrides.get("contraction_mult", round(rng.uniform(0.5, 0.9), 2))
    # v7: occasionally tag an indicator operand with a coarser context
    # timeframe -- the classic "HTF bias indicator" primitive. The MTF
    # pipeline (MTF-STRATEGY-001 + app.data.timeframe_resample) computes
    # it on genuine native-frequency bars, lookahead-safe; validate()'s
    # round-trip runs the same prep. Skipped for price/session/lookback
    # kinds (see _MTF_EXCLUDED_KINDS).
    if ("timeframe" not in overrides and kind in NUMERIC_KINDS
            and kind not in _MTF_EXCLUDED_KINDS
            and rng.random() < _MTF_PROBABILITY):
        op["timeframe"] = rng.choice(MTF_CONTEXT_TIMEFRAMES)
    op.update(overrides)
    return op


def _register_builtin_terminals() -> None:
    for kind in NUMERIC_KINDS:
        register_operand_kind(kind, (lambda k: (lambda rng: _operand(k, rng)))(kind))
    for kind in BOOLEAN_KINDS:
        register_operand_kind(kind, (lambda k: (lambda rng: _operand(k, rng)))(kind))


_register_builtin_terminals()


def _register_astra_terminals() -> None:
    """Register the w10-astra terminals (Oct-4 analysis, Part A ports
    #2/#3): regime-adaptive RSI zones + causal RSI divergence + session
    VWAP profile levels/flags.

    Called once at import, right after _register_builtin_terminals() --
    the astra kinds are already in the NUMERIC_KINDS / BOOLEAN_KINDS
    category tuples (so random_operand() can draw them), and this
    replaces the generic _operand() builders the builtin loop assigned
    with ones that sample the operand-level knobs manual.py's
    _regime_oscillator_series / _vwap_profile_series actually read
    (rsi_period / div_lookback / trend window; roll_hour / band sigmas /
    poc_buckets). validate() round-trips every emitted operand through
    the real Manual builder, so a bad knob choice here surfaces as an
    import-time-visible test failure, not a silent runtime miss.

    Boolean kinds (drawn with "is true"/"is false"): rsi_zone_buy,
    rsi_zone_sell, vwap_above, vwap_below, vwap_outside_value_area.
    Tristate/numeric kinds (+1/-1/0, compared numerically): rsi_regime,
    rsi_divergence. Plain numeric (price-scale levels): session_vwap,
    vwap_sigma, vwap_vah, vwap_val, vwap_upper_2, vwap_lower_2,
    vwap_poc, vwap_zscore.
    """
    def _rsi_builder(kind: str) -> OperandBuilder:
        def _build(rng: random.Random) -> dict:
            return {
                "type": kind,
                "rsi_period": int(rng.choice([7, 14, 21])),
                "div_lookback": int(rng.choice([10, 20, 30])),
                "trend_left": 5,
                "trend_right": 5,
            }
        return _build

    def _vwap_builder(kind: str) -> OperandBuilder:
        def _build(rng: random.Random) -> dict:
            return {
                "type": kind,
                "roll_hour": 17,
                "value_area_sigma": 1.0,
                "extreme_sigma": 2.0,
                "poc_buckets": 30,
            }
        return _build

    for _kind in ("rsi_regime", "rsi_divergence", "rsi_zone_buy", "rsi_zone_sell"):
        register_operand_kind(_kind, _rsi_builder(_kind))
    for _kind in ("session_vwap", "vwap_sigma", "vwap_vah", "vwap_val",
                  "vwap_upper_2", "vwap_lower_2", "vwap_poc", "vwap_zscore",
                  "vwap_above", "vwap_below", "vwap_outside_value_area"):
        register_operand_kind(_kind, _vwap_builder(_kind))


_register_astra_terminals()


def random_operand(rng: random.Random, category: str = "numeric") -> dict:
    """Draw one random operand from the terminal registry.

    category: "numeric" (price/indicator/session kinds), "boolean", or
    "any". Unknown registry entries default to the numeric treatment.
    """
    if category == "boolean":
        kinds = [k for k in BOOLEAN_KINDS if k in OPERAND_BUILDERS]
    elif category == "any":
        kinds = list(OPERAND_BUILDERS)
    else:
        kinds = [k for k in NUMERIC_KINDS if k in OPERAND_BUILDERS]
    kind = rng.choice(kinds)
    return OPERAND_BUILDERS[kind](rng)


def random_threshold(rng: random.Random, left_kind: str) -> dict:
    """A sensible right-hand side for a comparison against `left_kind`.

    Bounded oscillators get a threshold inside their real range (an
    out-of-range threshold would be dead code -- see
    app.strategy.manual.validate_bounded_conditions); everything else
    gets either another numeric operand or a plausible constant.
    """
    bounds = THRESHOLD_BOUNDS.get(left_kind)
    if bounds is not None:
        lo, hi = bounds
        return {"type": "value", "value": round(rng.uniform(lo, hi), 2)}
    roll = rng.random()
    if roll < 0.55:
        # Indicator-vs-indicator (the transferable pattern -- no
        # instrument-specific price level baked in).
        return random_operand(rng, "numeric")
    if left_kind in _PRICE_SCALE_KINDS:
        # A6: price-level terminals (session VWAP profile levels) draw
        # constant thresholds from the data's own price scale -- a fixed
        # 0.1-3.0 constant against a ~2000 price level is dead code
        # (always true / always false), and the numeric GA stage re-tunes
        # the level per dataset afterwards anyway.
        scale = _price_scale()
        return {"type": "value", "value": round(rng.uniform(0.8, 1.2) * scale, 2)}
    # Plausible constants for ratio/count-style terminals; a plain price
    # comparison against a fixed level is allowed but rare (levels don't
    # transfer across instruments -- the numeric GA stage can tune them).
    return {"type": "value", "value": round(rng.uniform(0.1, 3.0), 2)}


# Price-level terminals whose constant thresholds must be drawn from the
# data's own price scale (see random_threshold above).
_PRICE_SCALE_KINDS = frozenset({
    "session_vwap", "vwap_sigma", "vwap_vah", "vwap_val",
    "vwap_upper_2", "vwap_lower_2", "vwap_poc",
})

_price_scale_frame: pd.DataFrame | None = None


def _price_scale() -> float:
    """Median close of the canonical synthetic frame -- the price scale
    _PRICE_SCALE_KINDS draw constant thresholds from. The synthetic
    frame is the same one validate() round-trips through, so a
    generated threshold is always meaningful on at least that frame."""
    global _price_scale_frame
    if _price_scale_frame is None:
        _price_scale_frame = _synthetic_ohlcv()
    return float(_price_scale_frame["close"].median())


def random_condition(rng: random.Random) -> dict:
    """One random valid entry/exit condition dict."""
    if rng.random() < 0.22:
        # Boolean terminal with is true/is false.
        left = random_operand(rng, "boolean")
        return {
            "left": left,
            "operator": rng.choice(["is true", "is false"]),
            "right": {"type": "value", "value": 1},
        }
    left = random_operand(rng, "numeric")
    op = rng.choice(_NUMERIC_OPS)
    return {"left": left, "operator": op, "right": random_threshold(rng, left.get("type", ""))}


def _random_condition_list(rng: random.Random, max_conditions: int, block_pool: dict | None,
                           side: str) -> tuple[list[dict], list[str]]:
    """A condition list + connectors, optionally seeded from a template block."""
    conditions: list[dict] = []
    if block_pool and rng.random() < 0.5:
        blocks = [b for b in block_pool.get("entry_blocks", []) if b.get("side") == side]
        if blocks:
            block = copy.deepcopy(rng.choice(blocks))
            conditions = block.get("conditions", [])
    n = rng.randint(1, max(1, max_conditions))
    while len(conditions) < n:
        conditions.append(random_condition(rng))
    conditions = conditions[:max(n, 1)]
    connectors = [rng.choice(CONNECTORS) for _ in range(max(0, len(conditions) - 1))]
    return conditions, connectors


def random_risk_block(rng: random.Random, block_pool: dict | None = None) -> dict:
    """A random valid risk_management block."""
    if block_pool and rng.random() < 0.4:
        blocks = block_pool.get("risk_blocks", [])
        if blocks:
            return copy.deepcopy(rng.choice(blocks))
    rm: dict[str, Any] = {}
    if rng.random() < 0.7:
        rm["stop_type"] = "atr"
        rm["stop_value"] = round(rng.uniform(0.75, 3.0), 2)
        rm["stop_atr_period"] = rng.choice([10, 14, 20])
    else:
        rm["stop_type"] = "fixed"
        rm["stop_value"] = round(rng.uniform(20.0, 200.0), 1)
    if rng.random() < 0.7:
        rm["target_type"] = "atr"
        rm["target_value"] = round(rng.uniform(1.0, 4.0), 2)
        rm["target_atr_period"] = rng.choice([10, 14, 20])
    else:
        rm["target_type"] = "fixed"
        rm["target_value"] = round(rng.uniform(40.0, 400.0), 1)
    if rng.random() < 0.4:
        rm["trailing_stop"] = {
            "enabled": True,
            "value": round(rng.uniform(1.0, 2.5), 2),
            "atr_period": rng.choice([10, 14, 20]),
        }
    if rng.random() < 0.4:
        rm["break_even"] = {"enabled": True, "trigger_r": round(rng.uniform(0.5, 2.0), 2)}
    if rng.random() < 0.25:
        rm["partial_exit"] = {
            "enabled": True,
            "r_multiple": round(rng.uniform(1.0, 2.5), 2),
            "fraction": round(rng.uniform(0.25, 0.75), 2),
            "move_stop_to_breakeven": True,
        }
    if rng.random() < 0.6:
        rm["max_bars_in_trade"] = rng.choice([24, 48, 96, 192])
    rm["opposite_signal_exit"] = rng.random() < 0.8
    if rng.random() < 0.15:
        rm["time_based_exit"] = {
            "enabled": True,
            "time": rng.choice(["15:00", "15:30", "16:00", "21:00"]),
        }
    return rm


# Regime vocabulary mirrors app.validation.regime_matrix's label values
# (trend/volatility/environment dims; session dim omitted -- its labels
# are session-name specific).
_REGIME_DIM_VALUES = {
    "trend": ["strong_bullish", "strong_bearish", "weak_bullish", "weak_bearish", "neutral"],
    "volatility": ["extreme", "high", "very_low"],
    "environment": ["compression", "expansion", "breakout"],
}


def random_filter_block(rng: random.Random) -> dict:
    """A random valid filters block (may be empty)."""
    filters: dict[str, Any] = {}
    roll = rng.random()
    if roll < 0.35:
        # Mostly weekend exclusion (the common real use), sometimes a
        # weekday -- distinct from the day_of_week CONDITION kind (an
        # entry rule) per manual.py's docstring.
        days = [5, 6] if rng.random() < 0.7 else sorted(rng.sample(range(7), rng.randint(1, 2)))
        filters["days_of_week"] = {"exclude": days}
    elif roll < 0.6:
        cells = []
        for _ in range(rng.randint(1, 2)):
            dim = rng.choice(list(_REGIME_DIM_VALUES))
            cells.append({dim: rng.choice(_REGIME_DIM_VALUES[dim])})
        filters["regime_exclude"] = cells
    return filters


def generate_random(
    rng: random.Random | None = None,
    max_conditions_per_side: int = 3,
    block_pool: dict | None = None,
    seed: int | None = None,
    allow_mtf: bool = True,
) -> dict:
    """Generate one random, valid Manual strategy config dict.

    rng: random.Random instance (preferred -- pass one for reproducible
    streams); seed: convenience alternative (ignored if rng is given).
    block_pool: optional building_block_pool() output -- with ~50%
    probability per side, entry conditions are drawn from decomposed
    template blocks instead of pure random terminals, so the grammar
    invents around proven ingredients rather than pure noise.
    allow_mtf: when False, no operand gets a "timeframe" key (the MTF
    emission probability is forced to 0.0 for the duration of this call
    only; the module-level setting is restored afterwards).
    """
    prev_prob = None
    if not allow_mtf:
        prev_prob = set_mtf_probability(0.0)
    try:
        return _generate_random_inner(rng, max_conditions_per_side, block_pool, seed)
    finally:
        if prev_prob is not None:
            set_mtf_probability(prev_prob)


def _generate_random_inner(
    rng: random.Random | None,
    max_conditions_per_side: int,
    block_pool: dict | None,
    seed: int | None,
) -> dict:
    """Body of generate_random (see that docstring). Split out so the
    allow_mtf flag can scope the MTF probability knob around it."""
    if rng is None:
        rng = random.Random(seed)
    long_conds, long_conns = _random_condition_list(rng, max_conditions_per_side, block_pool, "long")
    short_conds, short_conns = _random_condition_list(rng, max_conditions_per_side, block_pool, "short")

    entry: dict[str, Any] = {"long": long_conds, "short": short_conds}
    if long_conns:
        entry["long_connectors"] = long_conns
    if short_conns:
        entry["short_connectors"] = short_conns

    exits: dict[str, Any] = {"long": [], "short": []}
    if block_pool and rng.random() < 0.4:
        blocks = block_pool.get("exit_blocks", [])
        if blocks:
            b = copy.deepcopy(rng.choice(blocks))
            exits[b.get("side", "long")] = b.get("conditions", [])
            if b.get("connectors"):
                exits[f"{b.get('side', 'long')}_connectors"] = b["connectors"]
    elif rng.random() < 0.3:
        side = rng.choice(["long", "short"])
        exits[side] = [random_condition(rng)]

    config: dict[str, Any] = {
        "name": f"Grammar strategy {rng.randrange(10**8):08x}",
        "entry_conditions": entry,
        "exit_conditions": exits,
        "risk_management": random_risk_block(rng, block_pool),
        "market": {"direction": "Both"},
    }
    filters = random_filter_block(rng)
    if filters:
        config["filters"] = filters

    errors = validate(config)
    if errors:
        # Should never happen -- generate_random only emits shapes
        # validate() accepts. Raising (not silently returning) keeps a
        # generator/validator drift from hiding.
        raise AssertionError(f"generate_random produced an invalid config: {errors[:3]}")
    return config


def mirror_condition_for_short(condition: dict, rng: random.Random) -> dict:
    """Mirror a long-side condition for the short side (flip comparators)."""
    out = copy.deepcopy(condition)
    op = str(out.get("operator", ">")).strip()
    out["operator"] = _MIRROR_OP.get(op, op)
    return out


# ---------------------------------------------------------------------------
# Validation -- structural checks + a real round-trip through the Manual
# builder on synthetic data (proves the config is dispatchable, not just
# well-shaped).
# ---------------------------------------------------------------------------

def _synthetic_ohlcv(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-02", periods=n, freq="15min")
    drift = np.linspace(0, 30, n)
    noise = np.cumsum(rng.normal(0, 0.5, n))
    close = 2000.0 + drift + noise
    spread = np.abs(rng.normal(0.4, 0.2, n))
    return pd.DataFrame({
        "timestamp": ts,
        "open": close - rng.normal(0, 0.2, n),
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": np.abs(rng.normal(100, 20, n)),
    })


def _validate_operand(operand: Any, where: str, errors: list[str]) -> None:
    if isinstance(operand, (int, float, np.number)):
        return
    if not isinstance(operand, dict):
        errors.append(f"{where}: operand must be a dict or number, got {type(operand).__name__}")
        return
    kind = str(operand.get("type", operand.get("source", ""))).lower().strip()
    if kind not in ALL_OPERAND_KINDS and kind not in OPERAND_BUILDERS:
        errors.append(f"{where}: unknown operand kind '{kind}'")
        return
    period = operand.get("period")
    if period is not None:
        try:
            if int(period) < 1:
                errors.append(f"{where}: period must be >= 1, got {period!r}")
        except (TypeError, ValueError):
            errors.append(f"{where}: period must be an integer, got {period!r}")
    lookback = operand.get("lookback")
    if lookback is not None:
        try:
            if int(lookback) < 1:
                errors.append(f"{where}: lookback must be >= 1, got {lookback!r}")
        except (TypeError, ValueError):
            errors.append(f"{where}: lookback must be an integer, got {lookback!r}")
    if kind in CONSTANT_KINDS:
        try:
            float(operand.get("value", 0))
        except (TypeError, ValueError):
            errors.append(f"{where}: constant operand needs a numeric 'value'")
    direction = operand.get("direction")
    if direction is not None and str(direction).lower() not in ("both", "bullish", "bearish"):
        errors.append(f"{where}: direction must be both/bullish/bearish, got {direction!r}")
    if kind == "day_of_week":
        days = operand.get("days", [operand.get("day", 0)])
        try:
            if any(int(d) < 0 or int(d) > 6 for d in days):
                errors.append(f"{where}: day_of_week days must be 0..6, got {days!r}")
        except (TypeError, ValueError):
            errors.append(f"{where}: day_of_week days must be integers 0..6")
    for key in ("session_start", "session_end"):
        if operand.get(key) is not None:
            try:
                pd.to_datetime(str(operand[key])).time()
            except (TypeError, ValueError):
                errors.append(f"{where}: {key} must be HH:MM, got {operand[key]!r}")
    # (Bounded-oscillator threshold checking lives in _validate_condition,
    # where BOTH sides of the comparison are visible -- an indicator
    # operand carries no "value" of its own; the threshold sits on the
    # constant operand on the other side.)


def _validate_condition(condition: Any, where: str, errors: list[str]) -> None:
    if not isinstance(condition, dict):
        errors.append(f"{where}: condition must be a dict, got {type(condition).__name__}")
        return
    operator = str(condition.get("operator", "")).strip().lower()
    if operator not in VALID_OPERATORS:
        # The builder accepts a few aliases ("greater than", "gt", ...);
        # the grammar only emits the canonical OPERATORS set, but all
        # builder-supported spellings validate.
        errors.append(f"{where}: unsupported operator {condition.get('operator')!r}")
        return
    _validate_operand(condition.get("left"), f"{where}.left", errors)
    _validate_operand(condition.get("right"), f"{where}.right", errors)
    # Bounded-oscillator thresholds outside the oscillator's real range are
    # dead code (see app.strategy.manual.validate_bounded_conditions) --
    # the grammar must never emit them, and hand-fed configs get rejected
    # here, not warned. The threshold sits on the CONSTANT operand; the
    # oscillator may be on either side of the comparison.
    for osc_node, val_node in (
        (condition.get("left"), condition.get("right")),
        (condition.get("right"), condition.get("left")),
    ):
        if not isinstance(osc_node, dict) or not isinstance(val_node, dict):
            continue
        osc_kind = str(osc_node.get("type", "")).lower().strip()
        bounds = THRESHOLD_BOUNDS.get(osc_kind)
        if bounds is None:
            continue
        if str(val_node.get("type", "")).lower().strip() not in CONSTANT_KINDS:
            continue
        try:
            thr = float(val_node.get("value", 0))
        except (TypeError, ValueError):
            continue
        lo, hi = bounds
        if thr < lo or thr > hi:
            errors.append(
                f"{where}: threshold {thr:g} outside {osc_kind}'s [{lo:g}, {hi:g}] -- "
                "this condition can never be true"
            )


def _validate_condition_block(block: Any, where: str, errors: list[str]) -> None:
    if not isinstance(block, dict):
        errors.append(f"{where}: must be a dict of side -> condition list")
        return
    for side in ("long", "short"):
        conditions = block.get(side, [])
        if not isinstance(conditions, list):
            errors.append(f"{where}.{side}: must be a list")
            continue
        for i, cond in enumerate(conditions):
            _validate_condition(cond, f"{where}.{side}[{i}]", errors)
        conns = block.get(f"{side}_connectors", [])
        if conns:
            if not isinstance(conns, list) or len(conns) != max(0, len(conditions) - 1):
                errors.append(
                    f"{where}.{side}_connectors: need 0 or len(conditions)-1 entries, "
                    f"got {len(conns) if isinstance(conns, list) else conns!r}"
                )
            elif any(str(c).upper() not in CONNECTORS for c in conns):
                errors.append(f"{where}.{side}_connectors: entries must be AND/OR")


def _validate_risk_block(rm: Any, errors: list[str]) -> None:
    if rm is None:
        return
    if not isinstance(rm, dict):
        errors.append("risk_management: must be a dict")
        return
    for key in ("stop_type", "target_type"):
        v = rm.get(key)
        if v not in (None, "", "fixed", "atr"):
            errors.append(f"risk_management.{key}: must be 'fixed'/'atr', got {v!r}")
    for key in ("stop_value", "target_value"):
        v = rm.get(key)
        if v not in (None, ""):
            try:
                if float(v) <= 0:
                    errors.append(f"risk_management.{key}: must be positive, got {v!r}")
            except (TypeError, ValueError):
                errors.append(f"risk_management.{key}: must be numeric, got {v!r}")
    for key in ("stop_atr_period", "target_atr_period"):
        v = rm.get(key)
        if v is not None:
            try:
                if int(v) < 1:
                    errors.append(f"risk_management.{key}: must be >= 1")
            except (TypeError, ValueError):
                errors.append(f"risk_management.{key}: must be an integer")
    ts = rm.get("trailing_stop") or {}
    if isinstance(ts, dict) and ts.get("enabled"):
        try:
            if float(ts.get("value", 0)) <= 0:
                errors.append("risk_management.trailing_stop.value: must be positive")
        except (TypeError, ValueError):
            errors.append("risk_management.trailing_stop.value: must be numeric")
    be = rm.get("break_even") or {}
    if isinstance(be, dict) and be.get("enabled"):
        try:
            if float(be.get("trigger_r", -1)) < 0:
                errors.append("risk_management.break_even.trigger_r: must be >= 0")
        except (TypeError, ValueError):
            errors.append("risk_management.break_even.trigger_r: must be numeric")
    pe = rm.get("partial_exit") or {}
    if isinstance(pe, dict) and pe.get("enabled"):
        try:
            r_mult, frac = float(pe.get("r_multiple", 0)), float(pe.get("fraction", 0))
            if r_mult <= 0 or not 0 < frac <= 1:
                errors.append("risk_management.partial_exit: need r_multiple > 0 and 0 < fraction <= 1")
        except (TypeError, ValueError):
            errors.append("risk_management.partial_exit: r_multiple/fraction must be numeric")
    mb = rm.get("max_bars_in_trade")
    if mb is not None:
        try:
            if int(mb) < 1:
                errors.append("risk_management.max_bars_in_trade: must be >= 1")
        except (TypeError, ValueError):
            errors.append("risk_management.max_bars_in_trade: must be an integer")
    tbe = rm.get("time_based_exit") or {}
    if isinstance(tbe, dict) and tbe.get("enabled"):
        try:
            pd.to_datetime(str(tbe.get("time", ""))).time()
        except (TypeError, ValueError):
            errors.append(f"risk_management.time_based_exit.time: must be HH:MM, got {tbe.get('time')!r}")
    for legacy in ("stop_loss_pips", "take_profit_pips"):
        v = rm.get(legacy)
        if v is not None:
            try:
                float(v)
            except (TypeError, ValueError):
                errors.append(f"risk_management.{legacy}: must be numeric")


_REGIME_DIM_NAMES = {"trend", "volatility", "session", "environment"}


def _validate_filters(filters: Any, errors: list[str]) -> None:
    if filters is None:
        return
    if not isinstance(filters, dict):
        errors.append("filters: must be a dict")
        return
    dow = filters.get("days_of_week") or {}
    if isinstance(dow, dict):
        for d in dow.get("exclude", []) or []:
            try:
                if int(d) < 0 or int(d) > 6:
                    errors.append(f"filters.days_of_week.exclude: days must be 0..6, got {d!r}")
            except (TypeError, ValueError):
                errors.append(f"filters.days_of_week.exclude: days must be integers 0..6, got {d!r}")
    cells = filters.get("regime_exclude") or []
    if not isinstance(cells, list):
        errors.append("filters.regime_exclude: must be a list of {dim: value} dicts")
    else:
        for i, cell in enumerate(cells):
            if not isinstance(cell, dict) or not cell:
                errors.append(f"filters.regime_exclude[{i}]: must be a non-empty dict")
            elif any(k not in _REGIME_DIM_NAMES for k in cell):
                errors.append(
                    f"filters.regime_exclude[{i}]: dims must be a subset of "
                    f"{sorted(_REGIME_DIM_NAMES)}, got {sorted(cell)}"
                )


def _config_declares_timeframe(config: dict) -> bool:
    """True if any entry/exit condition operand carries a "timeframe" key
    (the v7 MTF primitive)."""
    for _where, node in _iter_condition_operands(config):
        if node.get("timeframe"):
            return True
    return False


def validate(config: dict) -> list[str]:
    """Validate a Manual strategy config dict. Returns a list of error
    strings -- empty means valid.

    Beyond shape checks this round-trips through the REAL Manual builder
    (ManualStrategy.generate on synthetic OHLCV): a config that the
    builder cannot dispatch is invalid, however well-shaped. Structural
    operators call this after every mutation and fall back when it is
    non-empty, so validity here is the guarantee the search relies on.
    """
    errors: list[str] = []
    if not isinstance(config, dict):
        return ["config must be a dict"]
    entry = config.get("entry_conditions")
    if not entry:
        errors.append("entry_conditions: missing or empty")
    else:
        _validate_condition_block(entry, "entry_conditions", errors)
        longs = (entry.get("long") or []) if isinstance(entry, dict) else []
        shorts = (entry.get("short") or []) if isinstance(entry, dict) else []
        if not longs and not shorts:
            errors.append("entry_conditions: at least one side needs a condition")
    exits = config.get("exit_conditions")
    if exits:
        _validate_condition_block(exits, "exit_conditions", errors)
    _validate_risk_block(config.get("risk_management"), errors)
    _validate_filters(config.get("filters"), errors)

    if not errors:
        # The round-trip: prove the builder can actually dispatch every
        # operand kind/operator in this config. Synthetic data is enough --
        # this checks dispatch validity, not edge quality.
        try:
            strat = ManualStrategy(config)
            df = _synthetic_ohlcv()
            if _config_declares_timeframe(config):
                # v7 MTF: run the same upstream data prep the production
                # backtest path runs (app.backtest.engine.run_backtest) --
                # it computes each HTF-tagged indicator on genuine
                # native-frequency bars and merges it on as a lookahead-safe
                # tfNN_ column. Without this the builder would raise on the
                # missing merged column. Lazy import: keeps this module's
                # import graph unchanged for non-MTF configs.
                from app.data.timeframe_resample import prepare_timeframe_aligned_data
                df, _mtf_warnings = prepare_timeframe_aligned_data(df, strat)
            result = strat.generate(df)
            signals = result.signals
            if not set(pd.unique(signals.astype(int))).issubset({-1, 0, 1}):
                errors.append("builder round-trip: signals contain values outside {-1, 0, 1}")
        except Exception as exc:  # noqa: BLE001 -- any builder failure is a validity failure
            errors.append(f"builder round-trip failed: {type(exc).__name__}: {exc}")
    return errors


def lint_warnings(config: dict) -> list[str]:
    """Soft warnings for a config that is valid but suspicious -- currently
    the bounded-oscillator dead-code check plus the chikou lookahead flag
    plus the all-NaN/constant series check below. Informational only;
    validate() does not call this."""
    from app.strategy.manual import validate_bounded_conditions
    warnings = list(validate_bounded_conditions(config or {}))

    def _has_chikou(node: Any) -> bool:
        if isinstance(node, dict):
            if str(node.get("type", "")).lower().strip() == "ichimoku_chikou":
                return True
            return any(_has_chikou(v) for v in node.values())
        if isinstance(node, list):
            return any(_has_chikou(v) for v in node)
        return False

    if _has_chikou(config):
        warnings.append(
            "ichimoku_chikou present: known lookahead leak (close.shift(-26)) -- "
            "the grammar never emits this kind; see the module docstring."
        )
    warnings.extend(_constancy_warnings(config or {}))
    return warnings


def _iter_condition_operands(config: dict):
    """Yield (where, operand_dict) for every left/right operand in every
    entry/exit condition."""
    for block_name in ("entry_conditions", "exit_conditions"):
        block = config.get(block_name) or {}
        if not isinstance(block, dict):
            continue
        for side in ("long", "short"):
            conditions = block.get(side) or []
            if not isinstance(conditions, list):
                continue
            for i, cond in enumerate(conditions):
                if not isinstance(cond, dict):
                    continue
                for role in ("left", "right"):
                    node = cond.get(role)
                    if isinstance(node, dict):
                        yield f"{block_name}.{side}[{i}].{role}", node


def _constancy_warnings(config: dict) -> list[str]:
    """A6: warn when a condition's computed series is all-NaN or constant
    on the canonical synthetic frame -- a constant series makes its
    condition dead code (always true / always false), and an all-NaN
    series means the terminal produced nothing tradeable at all.
    Informational only (lint, not validate): the check runs against the
    synthetic frame, so a series that is constant THERE but varies on
    real data would be a false positive as a hard error."""
    warnings: list[str] = []
    try:
        df = _synthetic_ohlcv()
        strat = ManualStrategy(config)
    except Exception:  # noqa: BLE001 -- lint must never blow up the caller
        return warnings
    for where, node in _iter_condition_operands(config):
        kind = str(node.get("type", "")).lower().strip()
        if kind in CONSTANT_KINDS or kind not in ALL_OPERAND_KINDS:
            continue
        try:
            series = strat._series_from_operand(df, node, where.rsplit(".", 1)[-1])
            vals = pd.Series(series).dropna()
        except Exception:  # noqa: BLE001 -- an undispatchable operand is validate()'s problem, not lint's
            continue
        if len(vals) == 0:
            warnings.append(
                f"{where}: '{kind}' series is all-NaN on the reference frame -- this condition never fires"
            )
        elif vals.nunique() <= 1:
            warnings.append(
                f"{where}: '{kind}' series is constant on the reference frame -- this condition is dead code"
            )
    return warnings


# ---------------------------------------------------------------------------
# Building-block pool: decompose the 84 frozen SkeletonSpecs into reusable
# condition/filter/risk blocks the grammar can draw from. The templates
# themselves are NOT modified or deleted -- this only reads them.
# ---------------------------------------------------------------------------

def building_block_pool(max_combos_per_family: int = 4) -> dict:
    """Decompose every SkeletonSpec family into reusable blocks.

    Returns {"entry_blocks": [...], "exit_blocks": [...], "risk_blocks":
    [...], "filter_blocks": [...], "families_used": [...]}. Entry/exit
    blocks are {"side", "conditions", "connectors"} dicts; risk/filter
    blocks are the raw sub-dicts, all deep-copied. Families needing merged
    pair/calendar columns are skipped (their blocks reference operand
    kinds the grammar deliberately excludes).
    """
    from app.search.strategy_space import FAMILIES

    pool: dict[str, list] = {
        "entry_blocks": [], "exit_blocks": [], "risk_blocks": [],
        "filter_blocks": [], "families_used": [],
    }
    seen: set[str] = set()

    def _add(key: str, block: Any) -> None:
        import json
        try:
            fingerprint = key + ":" + json.dumps(block, sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            return
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        pool[key].append(copy.deepcopy(block))

    for family_name, spec in FAMILIES.items():
        if getattr(spec, "requires_pair_data", False) or getattr(spec, "requires_calendar_data", False):
            continue
        try:
            combos = spec.combinations()
        except Exception:  # noqa: BLE001
            continue
        if not combos:
            continue
        step = max(1, len(combos) // max(1, max_combos_per_family))
        for params in combos[::step][:max_combos_per_family]:
            try:
                cfg = spec.build(params)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(cfg, dict):
                continue
            # Only keep blocks that pass the grammar's own validator --
            # the pool must never inject something generate_random()
            # couldn't have emitted itself.
            entry = cfg.get("entry_conditions") or {}
            for side in ("long", "short"):
                conds = entry.get(side) or []
                if conds:
                    _add("entry_blocks", {
                        "side": side,
                        "conditions": conds,
                        "connectors": list(entry.get(f"{side}_connectors") or []),
                        "family": family_name,
                    })
            exits = cfg.get("exit_conditions") or {}
            for side in ("long", "short"):
                conds = exits.get(side) or []
                if conds:
                    _add("exit_blocks", {
                        "side": side,
                        "conditions": conds,
                        "connectors": list(exits.get(f"{side}_connectors") or []),
                        "family": family_name,
                    })
            if cfg.get("risk_management"):
                _add("risk_blocks", cfg["risk_management"])
            if cfg.get("filters"):
                _add("filter_blocks", cfg["filters"])
        pool["families_used"].append(family_name)
    return pool


def random_block(pool: dict, block_type: str, rng: random.Random) -> dict | None:
    """Draw one deep-copied block of `block_type` from a pool (None if empty).

    block_type: "entry_blocks" | "exit_blocks" | "risk_blocks" | "filter_blocks".
    """
    blocks = (pool or {}).get(block_type) or []
    if not blocks:
        return None
    return copy.deepcopy(rng.choice(blocks))
