"""Tests for the regime-conditional-trading primitive: app.strategy.base.
resolve_excluded_regimes / apply_regime_exclusion, its wiring into
app.backtest.engine.run_backtest, and app.validation.regime_matrix.
RegimeMatrixResult.as_exclude_filter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.strategy.base import apply_regime_exclusion, resolve_excluded_regimes
from app.strategy.manual import ManualStrategy
from app.validation.regime_matrix import run_regime_matrix


def _trending_df(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.linspace(0, 40, n)
    noise = np.cumsum(rng.normal(0, 0.5, n))
    price = 1900 + drift + noise
    high = price + np.abs(rng.normal(0.3, 0.15, n))
    low = price - np.abs(rng.normal(0.3, 0.15, n))
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": high, "low": low, "close": price, "volume": 100.0,
    })


def _always_long_config(regime_exclude=None):
    config = {
        "name": "always long",
        "long_entry": "close > 0", "long_exit": "close < -1",
        "short_entry": "close < -1", "short_exit": "close > 0",
        "stop_loss_pips": 50, "take_profit_pips": 100,
    }
    if regime_exclude is not None:
        config["filters"] = {"regime_exclude": regime_exclude}
    return config


# ---------------------------------------------------------------------------
# resolve_excluded_regimes
# ---------------------------------------------------------------------------

def test_manual_strategy_declares_nothing_by_default():
    strategy = ManualStrategy(_always_long_config())
    assert resolve_excluded_regimes(strategy) == []


def test_manual_strategy_declares_regime_exclude():
    strategy = ManualStrategy(_always_long_config(regime_exclude=[{"volatility": "extreme"}]))
    assert resolve_excluded_regimes(strategy) == [{"volatility": "extreme"}]


def test_manual_strategy_multi_dim_cell():
    cells = [{"trend": "strong_bearish", "environment": "compression"}]
    strategy = ManualStrategy(_always_long_config(regime_exclude=cells))
    assert resolve_excluded_regimes(strategy) == cells


def test_pinescript_directive_parses_multiple_cells():
    class FakePine:
        source_type = "pinescript"
        code = "// T58_EXCLUDE_REGIMES=volatility:extreme;trend:strong_bearish,environment:compression\n//@version=5"

    result = resolve_excluded_regimes(FakePine())
    assert result == [{"volatility": "extreme"}, {"trend": "strong_bearish", "environment": "compression"}]


def test_python_strategy_module_attr_hook():
    class FakePython:
        source_type = "python"

        @staticmethod
        def module_attr(name, default=None):
            if name == "EXCLUDE_REGIMES":
                return [{"volatility": "extreme"}]
            return default

    assert resolve_excluded_regimes(FakePython()) == [{"volatility": "extreme"}]


def test_no_source_type_returns_empty():
    class Bare:
        pass

    assert resolve_excluded_regimes(Bare()) == []


# ---------------------------------------------------------------------------
# apply_regime_exclusion
# ---------------------------------------------------------------------------

def test_apply_regime_exclusion_noop_when_nothing_declared():
    df = _trending_df()
    strategy = ManualStrategy(_always_long_config())
    signals = pd.Series(1, index=df.index)
    out = apply_regime_exclusion(df, signals, strategy)
    assert out is signals  # returned completely unchanged, not even copied


def test_apply_regime_exclusion_flattens_matching_bars():
    df = _trending_df()
    strategy = ManualStrategy(_always_long_config(regime_exclude=[{"trend": "strong_bullish"}]))
    signals = pd.Series(1, index=df.index)
    out = apply_regime_exclusion(df, signals, strategy)
    assert out.sum() < signals.sum()  # some bars got flattened
    assert set(out.unique()) <= {0, 1}


def test_apply_regime_exclusion_never_raises_on_bad_dim():
    df = _trending_df()
    strategy = ManualStrategy(_always_long_config(regime_exclude=[{"not_a_real_dimension": "whatever"}]))
    signals = pd.Series(1, index=df.index)
    out = apply_regime_exclusion(df, signals, strategy)
    # An unknown dimension name matches nothing -- should behave as if that
    # cell simply never fires, not crash.
    assert out.equals(signals)


def test_apply_regime_exclusion_too_little_data_degrades_gracefully():
    df = _trending_df(n=5)  # far too few bars for label_regimes' quantile bins
    strategy = ManualStrategy(_always_long_config(regime_exclude=[{"volatility": "extreme"}]))
    signals = pd.Series(1, index=df.index)
    out = apply_regime_exclusion(df, signals, strategy)  # must not raise
    assert len(out) == len(signals)


# ---------------------------------------------------------------------------
# End-to-end: run_backtest actually gates entries off
# ---------------------------------------------------------------------------

def test_run_backtest_wires_regime_exclusion_through():
    df = _trending_df()
    baseline = run_backtest(df, ManualStrategy(_always_long_config()), RiskConfig())
    excluded = run_backtest(
        df, ManualStrategy(_always_long_config(regime_exclude=[{"trend": "strong_bullish"}])), RiskConfig(),
    )
    # Gating off an entire trend regime on a strongly-trending synthetic
    # series should materially change the trade count versus baseline.
    assert len(excluded.trades) != len(baseline.trades) or len(baseline.trades) == 0


# ---------------------------------------------------------------------------
# RegimeMatrixResult.as_exclude_filter
# ---------------------------------------------------------------------------

def test_as_exclude_filter_matches_disable_regimes_dims():
    df = _trending_df()
    strategy = ManualStrategy(_always_long_config())
    result = run_regime_matrix(df, strategy, RiskConfig())
    if result is None:
        pytest.skip("Not enough valid bars to classify any regime on this synthetic dataset.")
    assert result.as_exclude_filter() == [c.dims for c in result.disable_regimes()]


# ---------------------------------------------------------------------------
# UPGRADE (regime-exclusion-primitive-had-no-ui): app.web.server.
# _apply_regime_exclusion_to_strategy -- the web route's "bake the
# recommended exclusion into a copy of the strategy" helper. Confirms
# each source_type's round-trip through resolve_excluded_regimes (the
# same reader apply_regime_exclusion uses at backtest time) rather than
# just eyeballing the generated text.
# ---------------------------------------------------------------------------

def test_apply_regime_exclusion_to_manual_strategy_merges_filter():
    from app.web.server import _apply_regime_exclusion_to_strategy

    strategy = ManualStrategy(_always_long_config())
    cells = [{"trend": "strong_bullish"}, {"volatility": "extreme"}]
    new_strategy, text, ext = _apply_regime_exclusion_to_strategy(strategy, cells)

    assert ext == "json"
    assert resolve_excluded_regimes(new_strategy) == cells
    # The original strategy object must be untouched.
    assert resolve_excluded_regimes(strategy) == []
    # The returned text is valid, re-loadable JSON with the filter baked in.
    import json
    reloaded = ManualStrategy(json.loads(text))
    assert resolve_excluded_regimes(reloaded) == cells


def test_apply_regime_exclusion_to_manual_strategy_preserves_existing_filters():
    """An existing regime_exclude list (from an earlier pass) must be
    preserved and merged with, not clobbered."""
    from app.web.server import _apply_regime_exclusion_to_strategy

    strategy = ManualStrategy(_always_long_config(regime_exclude=[{"session": "asia"}]))
    new_strategy, _text, _ext = _apply_regime_exclusion_to_strategy(strategy, [{"trend": "strong_bullish"}])
    assert resolve_excluded_regimes(new_strategy) == [
        {"session": "asia"}, {"trend": "strong_bullish"},
    ]


def test_apply_regime_exclusion_to_python_strategy(tmp_path):
    from app.strategy.python import PythonStrategy
    from app.web.server import _apply_regime_exclusion_to_strategy

    code = (
        "def generate_signals(df):\n"
        "    import pandas as pd\n"
        "    return pd.Series(1, index=df.index)\n"
    )
    p = tmp_path / "strat.py"
    p.write_text(code)
    strategy = PythonStrategy(p)
    cells = [{"environment": "trending"}]

    new_strategy, text, ext = _apply_regime_exclusion_to_strategy(strategy, cells)

    assert ext == "py"
    assert "EXCLUDE_REGIMES" in text
    assert resolve_excluded_regimes(new_strategy) == cells


def test_apply_regime_exclusion_to_pinescript_strategy():
    from app.strategy.pinescript import PineScriptStrategy
    from app.web.server import _apply_regime_exclusion_to_strategy

    strategy = PineScriptStrategy("//@version=5\nstrategy(\"t\")\n")
    cells = [{"trend": "strong_bullish", "volatility": "high"}, {"session": "asia"}]

    new_strategy, text, ext = _apply_regime_exclusion_to_strategy(strategy, cells)

    assert ext == "pine"
    assert "T58_EXCLUDE_REGIMES=" in text
    assert resolve_excluded_regimes(new_strategy) == cells


def test_apply_regime_exclusion_to_mql5_strategy():
    from app.strategy.mql5 import MQL5Strategy
    from app.web.server import _apply_regime_exclusion_to_strategy

    strategy = MQL5Strategy("void OnTick() {}\n")
    cells = [{"volatility": "low"}]

    new_strategy, text, ext = _apply_regime_exclusion_to_strategy(strategy, cells)

    assert ext == "mq5"
    assert resolve_excluded_regimes(new_strategy) == cells
