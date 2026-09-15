"""
Covers the 2026-09-15 Quick-Optimize-vs-Full-Pipeline diagnosis: a GA
search (Quick Optimize, Full Pipeline, Search Lab, Evolution Lab, Forge,
Overnight -- all of which share app.optimize.parameter_space's
extract_genome/apply_genome) could mutate a comparison threshold on a
bounded oscillator (RSI, Stochastic, MFI, ...) outside that oscillator's
actual range -- e.g. "RSI > 102.56", which can never be true since RSI
cannot exceed 100 -- silently turning that branch of the strategy's logic
into permanent dead code while every other number kept reporting
normally. It could also collapse an oscillator's period gene down to a
degenerate 1-bar lookback (e.g. a 1-period RSI, which is essentially a
coin-flip of whether the previous bar closed up or down), producing a
strategy that whipsaws on noise and bleeds transaction costs on
essentially every trade.

Both are fixed at the source: app.optimize.parameter_space.extract_genome
now clamps each gene's search range to the compared oscillator's real
range and floors oscillator periods at MIN_OSCILLATOR_PERIOD, so no
future GA-produced genome (from ANY of the search tools above -- they all
call this same function) can reproduce either failure mode. A second,
independent check -- app.strategy.manual.validate_bounded_conditions,
wired into app.backtest.engine.run_backtest -- catches the same class of
bug in a hand-typed or already-saved config that never went through the
GA at all.
"""
import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig, has_impossible_condition
from app.optimize.parameter_space import (
    MIN_OSCILLATOR_PERIOD,
    extract_genome,
)
from app.strategy.manual import ManualStrategy, validate_bounded_conditions

# The exact short-entry RSI condition from the strategy that produced the
# Quick-Optimize-vs-Full-Pipeline discrepancy: RSI(1) > 102.559... --
# mathematically impossible, since RSI is bounded to [0, 100].
_IMPOSSIBLE_RSI_CONDITION = {
    "left": {"type": "rsi", "period": 1, "field": "close"},
    "operator": ">",
    "right": {"type": "value", "value": 102.55949550153764},
}

_VALID_RSI_CONDITION = {
    "left": {"type": "rsi", "period": 6, "field": "close"},
    "operator": "<",
    "right": {"type": "value", "value": 30.51166136500698},
}


def _minimal_manual_config(short_condition=None):
    return {
        "name": "test strategy",
        "entry_conditions": {
            "long": [_VALID_RSI_CONDITION],
            "long_connectors": [],
            "short": [short_condition] if short_condition else [],
            "short_connectors": [],
        },
        "exit_conditions": {
            "long": [
                {
                    "left": {"type": "rsi", "period": 1, "field": "close"},
                    "operator": ">",
                    "right": {"type": "value", "value": 60.781147976025835},
                }
            ],
            "short": [],
        },
        "risk_management": {
            "stop_type": "atr",
            "stop_value": 6.265060019040547,
            "stop_atr_period": 29,
            "target_type": "atr",
            "target_value": 1.8199151881831175,
            "target_atr_period": 10,
            "opposite_signal_exit": True,
            "max_bars_in_trade": 24,
        },
    }


def _ohlcv_df(n=600, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="1min")
    price = 1800.0
    rows = []
    for i in range(n):
        step = rng.normal(0, 0.5)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.2))
        l = min(o, c) - abs(rng.normal(0, 0.2))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------
# extract_genome: bounded-value clamping
# ---------------------------------------------------------------------

def test_rsi_threshold_gene_is_clamped_to_valid_rsi_range():
    config = _minimal_manual_config(short_condition=_IMPOSSIBLE_RSI_CONDITION)
    genes = extract_genome(config)
    value_genes = [g for g in genes if g.kind == "value" and g.path[-2:] == ("right", "value")]
    # Every "value" gene compared against an RSI operand must search
    # strictly within RSI's real [0, 100] range, regardless of how far
    # outside that range the existing (already-invalid) base value sits.
    for gene in value_genes:
        assert gene.lo >= 0.0
        assert gene.hi <= 100.0
        assert gene.lo < gene.hi


def test_valid_rsi_threshold_search_range_still_stays_in_bounds():
    config = _minimal_manual_config()
    genes = extract_genome(config)
    value_genes = [g for g in genes if g.kind == "value"]
    assert value_genes, "expected at least one 'value' gene from the long entry/exit RSI conditions"
    for gene in value_genes:
        assert gene.lo >= 0.0
        assert gene.hi <= 100.0


def test_oscillator_period_gene_floored_at_min_oscillator_period():
    # The exit condition's RSI period is 1 (the exact degenerate value
    # from the reported strategy) -- the gene's search range must never
    # go below MIN_OSCILLATOR_PERIOD even though the generic period rule
    # alone would allow a floor of 1.
    config = _minimal_manual_config(short_condition=_IMPOSSIBLE_RSI_CONDITION)
    genes = extract_genome(config)
    period_genes = [g for g in genes if g.kind == "period" and g.path[-2] == "left"]
    assert period_genes
    for gene in period_genes:
        assert gene.lo >= MIN_OSCILLATOR_PERIOD


def test_non_oscillator_period_gene_unaffected():
    # An EMA period gene has no bounded range and no oscillator-period
    # floor -- this must keep behaving exactly as before.
    config = {
        "entry_conditions": {
            "long": [
                {
                    "left": {"type": "ema", "period": 69, "field": "close"},
                    "operator": ">",
                    "right": {"type": "ema", "period": 360, "field": "close"},
                }
            ],
            "long_connectors": [],
            "short": [],
            "short_connectors": [],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": {},
    }
    genes = extract_genome(config)
    ema_period_genes = [g for g in genes if g.kind == "period"]
    assert ema_period_genes
    fast = next(g for g in ema_period_genes if g.base_value == 69)
    assert fast.lo == float(round(max(69 * 0.3, 1)))  # unchanged generic rule, no oscillator floor applied


# ---------------------------------------------------------------------
# validate_bounded_conditions / has_impossible_condition
# ---------------------------------------------------------------------

def test_validate_bounded_conditions_flags_impossible_rsi_threshold():
    config = _minimal_manual_config(short_condition=_IMPOSSIBLE_RSI_CONDITION)
    warnings = validate_bounded_conditions(config)
    assert warnings
    assert any("can never be true" in w and "short entry" in w for w in warnings)


def test_validate_bounded_conditions_silent_on_valid_config():
    config = _minimal_manual_config()
    assert validate_bounded_conditions(config) == []


def test_run_backtest_surfaces_impossible_condition_warning():
    df = _ohlcv_df()
    strategy = ManualStrategy(_minimal_manual_config(short_condition=_IMPOSSIBLE_RSI_CONDITION))
    risk = RiskConfig(initial_balance=50_000.0, risk_mode="percent", risk_value=0.5, pip_size=1.0)
    result = run_backtest(df, strategy, risk)
    assert has_impossible_condition(result.warnings)


def test_run_backtest_no_false_positive_on_valid_strategy():
    df = _ohlcv_df()
    strategy = ManualStrategy(_minimal_manual_config())
    risk = RiskConfig(initial_balance=50_000.0, risk_mode="percent", risk_value=0.5, pip_size=1.0)
    result = run_backtest(df, strategy, risk)
    assert not has_impossible_condition(result.warnings)
