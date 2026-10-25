"""
Parameter space extraction for Iterative Refinement.

app.strategy.manual.ManualStrategy takes a plain, JSON-serializable nested
dict (see app/strategy/manual.py and app/ui/main_window.py::_build_strategy
for the exact shape). That happens to make it an ideal substrate for a
genetic-algorithm-style optimizer: every tunable number in a Manual
Strategy -- indicator periods, comparison thresholds, stop/target values,
ATR multiples, trailing-stop distance, break-even trigger, max bars in
trade -- is a plain numeric leaf somewhere in that dict.

This module walks a config dict, finds every leaf whose *key name* is a
known tunable parameter (see GENE_KEY_RULES below), and turns it into a
Gene: a (path, bounds, integer-or-continuous) triple. A "genome" is then
just a flat list of floats, one per gene, in the same order as the gene
list -- which is exactly the representation a GA needs for crossover and
mutation. apply_genome() writes a genome back into a full config dict.

Only Manual Strategy Builder configs are supported. Python / PineScript /
MQL5 strategies are opaque code with no declared parameter schema, so
there is nothing here to walk -- see app/optimize/refinement.py for how
that limitation is surfaced to the user.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from app.strategy.indicators import BOUNDED_OSCILLATOR_RANGES
from app.strategy.indicators_v12 import (
    V12_BOUNDED_RANGES, V12_FLAGS, V12_NO_PERIOD, V12_PERIOD_RANGES, V12_VALUE_RANGES,
)

# A period of 1 or 2 on a bounded oscillator (see BOUNDED_OSCILLATOR_RANGES)
# is degenerate, not just aggressive: e.g. a 1-period RSI collapses to
# essentially a coin-flip of whether the immediately preceding bar closed
# up or down, so it whipsaws every other bar regardless of any real trend
# or reversal. Left to the generic period rule's min_lo=1 floor, a GA is
# free to "discover" this degenerate case because it happens to score
# well on a narrow validation slice, while it behaves as near-random
# noise -- and a guaranteed cost bleed once entries/exits fire on every
# other bar -- once run against the full history.
# (see 2026-09-15 Quick-Optimize-vs-Full-Pipeline diagnosis this fixes,
# where the GA produced an RSI(1) exit condition).
MIN_OSCILLATOR_PERIOD = 3

# A raw-percentage / unbounded-momentum indicator's comparison threshold
# (e.g. the 1.0 in "ROC(10) crosses above 1.0%") sitting very close to
# zero is a DIFFERENT kind of degenerate case than an oscillator's period
# above, but the same root problem: left to the generic "value" gene's
# rel_span rule, a GA is free to mutate a meaningfully non-zero threshold
# (1-3%) down toward 0 because a de facto zero-cross trigger can score
# well on a narrow slice, while in reality it fires on nearly every bar
# of an unbounded oscillating series -- an explosively large trade count
# on a multi-million-bar 1-minute dataset that can take a worker minutes
# per candidate instead of seconds, exactly the "worker appears hung /
# stall recovery" pattern repeatedly seen on roc_momentum_continuation
# candidates during Search Lab's genetic-algorithm refinement stage (see
# the Search Lab log this fixes, where every stalled Stage 2 batch was
# that same family). Some families (awesome_oscillator_zero_cross,
# trix_zero_cross_momentum, day_of_week_seasonality's zero-anchored
# comparisons) DELIBERATELY compare against exactly 0 -- this floor is
# only ever applied when the strategy's OWN base threshold is already
# non-zero (see _gene_for below), so it never touches an intentional
# zero-cross design; it only stops the GA from mutating an already-
# meaningful non-zero threshold down into a degenerate near-zero one.
ZERO_SENSITIVE_MIN_ABS_THRESHOLD: dict[str, float] = {
    "roc": 0.5,   # percent -- half of roc_momentum_continuation's smallest shipped threshold (1.0%)
}


# v12 (2026-10-24): registered valid ranges for the new indicators. A comparison threshold paired with one of these is
# clamped into INDICATOR_VALUE_RANGES (soft, practical range of the series) or, for exactly bounded series, into
# BOUNDED_OSCILLATOR_RANGES (autocorr_lag1 -1..1, round_dist_* -0.5..0.5); a period gene is clamped into
# INDICATOR_PERIOD_RANGES. Single source of truth: app.strategy.indicators_v12.
INDICATOR_VALUE_RANGES: dict[str, tuple[float, float]] = dict(V12_VALUE_RANGES)
INDICATOR_PERIOD_RANGES: dict[str, tuple[int, int]] = dict(V12_PERIOD_RANGES)
assert all(k in BOUNDED_OSCILLATOR_RANGES for k in V12_BOUNDED_RANGES), "v12 bounded range missing from indicators.py"


class RefinementError(Exception):
    """Raised when Iterative Refinement cannot proceed (e.g. no tunable parameters)."""


# ---------------------------------------------------------------------------
# Which dict keys are treated as tunable numeric parameters, and how a
# search range is derived from their current value.
#
#   is_int      -- round mutated/random values to the nearest integer
#   lo_mult /
#   hi_mult     -- for strictly-positive values: search range is
#                  [value * lo_mult, value * hi_mult]
#   min_lo      -- floor for the low end of that range (never search below this)
#   rel_span    -- for values that can be zero/negative (comparison thresholds):
#                  search range is [value - span, value + span] where
#                  span = max(abs(value) * rel_span, abs_span)
#   abs_span    -- absolute floor for rel_span-based spans
#   abs_floor   -- fallback low end when the current value is <= 0 for a
#                  lo_mult/hi_mult-style key (e.g. an unset stop distance)
# ---------------------------------------------------------------------------
GENE_KEY_RULES: dict[str, dict] = {
    "period":            dict(is_int=True,  lo_mult=0.3, hi_mult=3.0, min_lo=1,   abs_floor=1),
    "lookback":          dict(is_int=True,  lo_mult=0.3, hi_mult=3.0, min_lo=1,   abs_floor=1),
    "stop_atr_period":   dict(is_int=True,  lo_mult=0.3, hi_mult=3.0, min_lo=1,   abs_floor=1),
    "target_atr_period": dict(is_int=True,  lo_mult=0.3, hi_mult=3.0, min_lo=1,   abs_floor=1),
    "atr_period":        dict(is_int=True,  lo_mult=0.3, hi_mult=3.0, min_lo=1,   abs_floor=1),
    "max_bars_in_trade": dict(is_int=True,  lo_mult=0.3, hi_mult=3.0, min_lo=1,   abs_floor=1),
    "stop_value":        dict(is_int=False, lo_mult=0.3, hi_mult=3.0, min_lo=0.1, abs_floor=0.1),
    "target_value":      dict(is_int=False, lo_mult=0.3, hi_mult=3.0, min_lo=0.1, abs_floor=0.1),
    "stop_loss_pips":    dict(is_int=False, lo_mult=0.3, hi_mult=3.0, min_lo=0.1, abs_floor=0.1),
    "take_profit_pips":  dict(is_int=False, lo_mult=0.3, hi_mult=3.0, min_lo=0.1, abs_floor=0.1),
    "trigger_r":         dict(is_int=False, lo_mult=0.3, hi_mult=3.0, min_lo=0.05, abs_floor=0.05),
    # "value" covers both the trailing-stop ATR multiple AND every
    # visual-builder condition's comparison constant (e.g. the "55" in
    # "RSI(14) > 55"), which can legitimately be zero or negative
    # (e.g. "MACD Histogram > 0"), so it gets the relative/absolute-span
    # treatment rather than a pure multiplicative range.
    "value":             dict(is_int=False, rel_span=0.5, abs_span=1.0),
}

# AUDIT FIX (Oct 2026, GA flag-gene bug). Operand kinds whose series is a 0/1 (or
# -1/0/1) FLAG, not a continuous quantity. The comparison constant paired with one
# of these ("time_of_day == 1.0", "supertrend_direction == 1", "rsi_zone_buy == 1")
# is a logical selector, not a tunable threshold. The generic "value" rule gave it a
# +/-50% window, so a GA could mutate `time_of_day == 1.0` into `== 0.627`, a
# comparison that can NEVER be true -- silently deleting that side of the strategy
# (the shipped "NQ1! Opening Range Breakout" short side was dead for exactly this
# reason). Their paired "value" gene is therefore frozen.
FLAG_OPERAND_KINDS = frozenset({
    "time_of_day", "day_of_week", "candle_direction",
    "swing_high", "swing_low", "liquidity_sweep", "break_of_structure", "bos",
    "change_of_character", "choch", "fair_value_gap", "fvg", "order_block",
    "atr_regime", "volatility_regime", "atr_expansion", "atr_contraction",
    "rsi_regime", "rsi_zone_buy", "rsi_zone_sell", "rsi_divergence",
    "vwap_above", "vwap_below", "vwap_outside_value_area",
    "supertrend_direction", "psar_direction", "ttm_squeeze_on",
    "swing_bos", "swing_choch",
    *V12_FLAGS,                     # nr4 / nr7 / daily_nr4 / daily_nr7
})

# Operand kinds that ignore `period`/`lookback` entirely (app.strategy.manual returns
# the raw column / constant / flag without reading them). Tuning those genes only
# burns search budget on dead dimensions and inflates the trial count that the
# Deflated-Sharpe gate corrects for.
NO_PERIOD_OPERAND_KINDS = frozenset({
    "price", "open", "high", "low", "close", "volume",
    "value", "constant", "number",
    "time_of_day", "day_of_week", "candle_direction",
    *V12_NO_PERIOD,                 # round_dist_* and the NR flags ignore period/lookback
})


def _is_flag_operand(node) -> bool:
    return isinstance(node, dict) and str(node.get("type", "")).lower().strip() in FLAG_OPERAND_KINDS


# Path-key segments that must never be descended into / mutated even if a
# nested dict happens to reuse one of the names above for something else.
# (Not currently needed by the known config shape, but kept as a guard
# rail for future config fields.)
_EXCLUDED_KEYS = {"name", "description", "author", "version", "market", "type",
                   "field", "direction", "source_type"}


@dataclass(frozen=True)
class GeneMeta:
    path: tuple                # sequence of dict keys / list indices to this value
    kind: str                  # the matched key name, e.g. "period", "stop_value"
    is_int: bool
    lo: float
    hi: float
    base_value: float
    label: str                 # human-readable dotted path, for reports/logs


def _label(path: tuple) -> str:
    parts: list[str] = []
    for key in path:
        if isinstance(key, int):
            parts[-1] = f"{parts[-1]}[{key}]"
        else:
            parts.append(str(key))
    return ".".join(parts)


def _bounds_for(key: str, value: float) -> tuple[float, float, bool]:
    rule = GENE_KEY_RULES[key]
    is_int = bool(rule["is_int"] if "is_int" in rule else False)

    if "rel_span" in rule:
        span = max(abs(value) * rule["rel_span"], rule["abs_span"])
        lo, hi = value - span, value + span
        return lo, hi, is_int

    lo_mult, hi_mult = rule["lo_mult"], rule["hi_mult"]
    min_lo, abs_floor = rule["min_lo"], rule["abs_floor"]

    if value > 0:
        lo = max(value * lo_mult, min_lo)
        hi = max(value * hi_mult, lo + min_lo)
    else:
        lo = float(abs_floor)
        hi = max(float(abs_floor) * 10.0, 10.0)

    if is_int:
        lo = float(max(1, round(lo)))
        hi = float(max(lo + 1, round(hi)))

    return lo, hi, is_int


def _oscillator_bound(node) -> tuple[float, float] | None:
    """If `node` is an operand dict whose "type" is a bounded oscillator
    (see BOUNDED_OSCILLATOR_RANGES), return its valid (lo, hi) range --
    otherwise None. Used so a "value" gene being compared against that
    operand can never mutate outside a range the oscillator could
    actually take, however loosely the generic rel_span rule would
    otherwise have allowed."""
    if not isinstance(node, dict):
        return None
    kind = str(node.get("type", "")).lower().strip()
    return BOUNDED_OSCILLATOR_RANGES.get(kind) or INDICATOR_VALUE_RANGES.get(kind)


def _zero_sensitive_min_abs(node) -> float | None:
    """If `node` is an operand dict whose "type" is a zero-sensitive
    unbounded-momentum indicator (see ZERO_SENSITIVE_MIN_ABS_THRESHOLD),
    return the minimum absolute comparison-threshold magnitude the GA
    must keep -- otherwise None."""
    if not isinstance(node, dict):
        return None
    kind = str(node.get("type", "")).lower().strip()
    return ZERO_SENSITIVE_MIN_ABS_THRESHOLD.get(kind)


def extract_genome(config: dict) -> list[GeneMeta]:
    """
    Walk a Manual Strategy config dict and return one GeneMeta per tunable
    numeric leaf found, in a stable (depth-first, dict-insertion) order.

    Two bounded-oscillator-aware refinements on top of the generic
    per-key rules in GENE_KEY_RULES (see BOUNDED_OSCILLATOR_RANGES and
    MIN_OSCILLATOR_PERIOD above for why these exist):

    - A condition shaped like {"left": ..., "operator": ..., "right": ...}
      (exactly the shape entry_conditions/exit_conditions items use) has
      its two sides inspected for each other's indicator type. If one
      side is a bounded oscillator (RSI, Stochastic, MFI, ...), the other
      side's "value" gene (the comparison threshold) is clamped into that
      oscillator's actual range -- so a GA can never mutate, say, an RSI
      threshold to something outside 0-100, which would otherwise make
      that whole branch of the condition impossible (permanently
      True/False) without anything flagging it.
    - A "period" gene that belongs to a bounded-oscillator operand itself
      (e.g. the 14 in {"type": "rsi", "period": 14, ...}) is floored at
      MIN_OSCILLATOR_PERIOD rather than the generic rule's min_lo=1, so
      the GA cannot collapse it into a degenerate 1- or 2-bar lookback.
    """
    genes: list[GeneMeta] = []

    def _gene_for(k: str, v: float, new_path: tuple, value_bounds, osc_kind, zero_min: float | None) -> GeneMeta:
        lo, hi, is_int = _bounds_for(k, float(v))
        if k == "value" and value_bounds is not None:
            clamped_lo = max(lo, value_bounds[0])
            clamped_hi = min(hi, value_bounds[1])
            if clamped_hi > clamped_lo:
                lo, hi = clamped_lo, clamped_hi
            else:
                # The base value itself already sits outside the
                # oscillator's range (a pre-existing invalid config) --
                # search the oscillator's own full range rather than a
                # window that can't contain anything valid.
                lo, hi = value_bounds
        elif k in ("period", "lookback") and osc_kind in BOUNDED_OSCILLATOR_RANGES:
            lo = max(lo, float(MIN_OSCILLATOR_PERIOD))
            hi = max(hi, lo + 1)
        if k == "period" and osc_kind in INDICATOR_PERIOD_RANGES:
            p_lo, p_hi = INDICATOR_PERIOD_RANGES[osc_kind]
            c_lo, c_hi = max(lo, float(p_lo)), min(hi, float(p_hi))
            lo, hi = (c_lo, c_hi) if c_hi > c_lo else (float(p_lo), float(p_hi))
        if k == "value" and zero_min is not None and v != 0:
            # Keep the same sign as the strategy's own base threshold --
            # only push the range AWAY from zero, never flip which side
            # of zero it searches (see ZERO_SENSITIVE_MIN_ABS_THRESHOLD).
            if v > 0:
                lo = max(lo, zero_min)
                if hi <= lo:
                    hi = lo + zero_min
            else:
                hi = min(hi, -zero_min)
                if lo >= hi:
                    lo = hi - zero_min
        return GeneMeta(
            path=new_path, kind=k, is_int=is_int,
            lo=lo, hi=hi, base_value=float(v),
            label=_label(new_path),
        )

    def walk(node, path: tuple, value_bounds: tuple[float, float] | None = None, zero_min: float | None = None,
             freeze_value: bool = False):
        if isinstance(node, dict):
            is_condition = "left" in node and "right" in node and "operator" in node
            if is_condition:
                left_bound = _oscillator_bound(node.get("left"))
                right_bound = _oscillator_bound(node.get("right"))
                left_zero_min = _zero_sensitive_min_abs(node.get("left"))
                right_zero_min = _zero_sensitive_min_abs(node.get("right"))
                walk(node.get("left"), path + ("left",), value_bounds=right_bound, zero_min=right_zero_min,
                     freeze_value=_is_flag_operand(node.get("right")))
                walk(node.get("right"), path + ("right",), value_bounds=left_bound, zero_min=left_zero_min,
                     freeze_value=_is_flag_operand(node.get("left")))
                for k, v in node.items():
                    if k in ("left", "right") or k in _EXCLUDED_KEYS:
                        continue
                    new_path = path + (k,)
                    if k in GENE_KEY_RULES and isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
                        genes.append(_gene_for(k, v, new_path, None, None, None))
                    walk(v, new_path)
                return
            osc_kind = str(node.get("type", "")).lower().strip() if "type" in node else None
            for k, v in node.items():
                if k in _EXCLUDED_KEYS:
                    continue
                new_path = path + (k,)
                if (
                    k in GENE_KEY_RULES
                    and isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and v is not None
                    and not (k == "value" and freeze_value)
                    and not (k in ("period", "lookback") and osc_kind in NO_PERIOD_OPERAND_KINDS)
                ):
                    genes.append(_gene_for(k, v, new_path, value_bounds, osc_kind, zero_min))
                walk(v, new_path, value_bounds=value_bounds, zero_min=zero_min, freeze_value=freeze_value)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, path + (i,), value_bounds=value_bounds, zero_min=zero_min, freeze_value=freeze_value)

    walk(config, ())
    return genes


def _get_by_path(obj, path: tuple):
    cur = obj
    for key in path:
        cur = cur[key]
    return cur


def _set_by_path(obj, path: tuple, value) -> None:
    cur = obj
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


def apply_genome(base_config: dict, genes: list[GeneMeta], genome: list[float]) -> dict:
    """
    Deep-copies base_config and writes each genome value into the path its
    corresponding GeneMeta describes. genome must be the same length and
    order as genes (as returned by extract_genome for this same config).
    """
    if len(genome) != len(genes):
        raise RefinementError(
            f"Genome length ({len(genome)}) does not match the number of "
            f"tunable parameters ({len(genes)})."
        )
    cfg = copy.deepcopy(base_config)
    for gene, raw_value in zip(genes, genome):
        value = int(round(raw_value)) if gene.is_int else float(raw_value)
        _set_by_path(cfg, gene.path, value)
    return cfg
