"""v13 indicators (2026-10-25): the report's "Top 100 indicators to add" -- one module that merges the registries.

Implementation lives in ``indicators_v13_core`` (groups A-F: volatility estimators, adaptive filters, cycles, randomness,
oscillators, microstructure) and ``indicators_v13_market`` (groups G-K plus 62-63: structure, cross-asset/macro/Ext, crypto,
seasonality, regime models). This module is what the rest of the app imports; it mirrors the v12 constants
(``V13_INDICATOR_BUILDERS`` ... ``V13_DESCRIPTIONS``) so the dispatcher, manual strategies, the GA's parameter space, the grammar
and the UI condition builder consume it the same way.

Each report item maps to one or more kinds (``V13_REPORT_NUMBER[kind]`` -> 1..100; ``items_covered()`` proves all 100 are present).

Registration checklist for every kind (the report's "for every addition"):
  1. registered with the grammar via ``register_operand_kind`` (``register_v13_with_grammar``), so Search/Evolution can draw it;
  2. exact bounds in ``V13_BOUNDED_RANGES`` / soft ranges in ``V13_VALUE_RANGES`` (consumed by app.optimize.parameter_space and
     indicators.BOUNDED_OSCILLATOR_RANGES) so a GA can never mutate a threshold outside what the series can produce;
  3. a DIRECT lookahead check (tests/backtest/test_indicators_v13.py: prefix-invariance on every kind + a manual-strategy
     behavioural check) -- the lookahead gate skips manual strategies, so this cannot be left to the engine;
  4. its extra terminals are CHARGED to the lifetime trial ledger (``ledger_charge``): a larger grammar is more implicit trials.
Ext kinds need a feed (app.data.macro_data.EXT_FEEDS, with release-time lags); companion kinds need frames in ``frame.attrs``.
"""
from __future__ import annotations

from app.strategy import indicators_v13_core as _core
from app.strategy import indicators_v13_market as _market   # noqa: F401  (importing registers the group G-K builders)
from app.strategy import indicators_v13_support as _support  # noqa: F401  (family-support primitives, report number 0)

V13_INDICATOR_BUILDERS = _core.BUILDERS
V13_INDICATOR_KINDS: tuple[str, ...] = tuple(V13_INDICATOR_BUILDERS)
V13_DESCRIPTIONS = _core.DESCRIPTIONS
V13_REPORT_NUMBER = _core.NUMBER
V13_BOUNDED_RANGES = _core.BOUNDED
V13_VALUE_RANGES = _core.VALUE_RANGES
V13_PERIOD_RANGES = {k: v for k, v in _core.PERIOD_RANGES.items() if v != (0, 0)}
V13_NO_PERIOD: tuple[str, ...] = tuple(_core.NO_PERIOD)
V13_FLAGS: tuple[str, ...] = tuple(_core.FLAGS)
V13_EXT_KINDS: frozenset = frozenset(_market.EXT_KINDS)
V13_COMPANION_KINDS: frozenset = frozenset(_market.COMPANION_KINDS)
V13_INSTRUMENT_TIED_KINDS: frozenset = frozenset(_market.INSTRUMENT_TIED)

# Kinds the random grammar must NOT draw blindly: they return NaN without an external feed / companion frame, and the families
# that use them are hand-built (the family files pass the feed). Everything else is drawable by the grammar.
V13_SUPPORT_KINDS: frozenset = frozenset(_support.V13_SUPPORT_KINDS)      # number 0: family building blocks, NOT among the 100 and never drawn by the grammar
V13_GRAMMAR_EXCLUDED: frozenset = V13_EXT_KINDS | V13_COMPANION_KINDS | V13_SUPPORT_KINDS


def items_covered() -> set[int]:
    return {v for v in V13_REPORT_NUMBER.values() if v > 0}


def kinds_for_item(n: int) -> tuple[str, ...]:
    return tuple(k for k, v in V13_REPORT_NUMBER.items() if v == n)


def ledger_charge(n_new_terminals: int | None = None) -> int:
    """Implicit trials a bigger grammar adds. Each drawable terminal multiplies the search space the GA/grammar explores; charge
    one trial per terminal per registered family-scale unit (conservative and deterministic, never zero)."""
    k = len(V13_INDICATOR_KINDS) - len(V13_GRAMMAR_EXCLUDED) if n_new_terminals is None else int(n_new_terminals)
    return max(1, k) * 10


def register_v13_with_grammar() -> int:
    """Register every drawable v13 kind with ``app.search.grammar.register_operand_kind``; returns how many were registered.

    The operand builder draws a period inside the kind's registered range (or none for no-period kinds) so a drawn operand is
    always valid. Idempotent."""
    from app.search import grammar

    def make(kind: str):
        rng_lo, rng_hi = V13_PERIOD_RANGES.get(kind, (5, 60))
        no_period = kind in V13_NO_PERIOD

        def build(rng):
            op = {"type": kind}
            if not no_period:
                op["period"] = int(rng.choice(sorted({rng_lo, (rng_lo + rng_hi) // 4 or rng_lo, (rng_lo + rng_hi) // 2, rng_hi})))
            return op
        return build

    n = 0
    for kind in V13_INDICATOR_KINDS:
        if kind in V13_GRAMMAR_EXCLUDED:
            continue
        grammar.register_operand_kind(kind, make(kind))
        n += 1
    return n
