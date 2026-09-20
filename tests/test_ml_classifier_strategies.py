"""Tests for strategies/python/ml_classifier_direction.py (logistic
regression baseline), ml_classifier_gbdt_direction.py (LightGBM
gradient-boosted trees), and ml_classifier_random_forest.py (bagged-tree
"Forest") -- shared no-lookahead contract and RETRAIN_PER_FOLD wiring,
parametrized across all three files since they're built to the identical
contract on purpose (see the GBDT/Forest files' own module docstrings for
why each is a separate file rather than a model= parameter on the
logistic one)."""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.strategy.python import PythonStrategy
from app.validation.walk_forward_opt import _run_fold_test, _wants_retrain_context, build_folds

STRATEGY_PATHS = [
    "strategies/python/ml_classifier_direction.py",
    "strategies/python/ml_classifier_gbdt_direction.py",
    "strategies/python/ml_classifier_random_forest.py",
]

# The Forest file's feature set includes a 200-span EMA, so it needs a
# longer warm-up than the other two families' n=800 default before it has
# any valid (non-NaN) feature rows to train on at all.
MIN_SYNTHETIC_N = {
    "strategies/python/ml_classifier_random_forest.py": 1400,
}


def _load_module(path):
    spec = importlib.util.spec_from_file_location(f"ml_test_import_{hash(path)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_df(n=1500, seed=11):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="15min")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + np.abs(rng.normal(0, 0.2, n))
    low = close - np.abs(rng.normal(0, 0.2, n))
    openp = close + rng.normal(0, 0.05, n)
    volume = rng.uniform(50, 150, n)
    return pd.DataFrame({
        "timestamp": ts, "open": openp, "high": high, "low": low, "close": close, "volume": volume,
    })


@pytest.mark.parametrize("path", STRATEGY_PATHS)
def test_module_declares_retrain_per_fold(path):
    strat = PythonStrategy(path)
    assert strat.module_flag("RETRAIN_PER_FOLD") is True


@pytest.mark.parametrize("path", STRATEGY_PATHS)
def test_too_little_data_stands_down_flat(path):
    module = _load_module(path)
    df = _synthetic_df(n=50)  # below the n < 200 floor both files enforce
    signals = module.generate_signals(df)
    assert (signals == 0).all()


@pytest.mark.parametrize("path", STRATEGY_PATHS)
def test_training_segment_is_always_flat(path):
    module = _load_module(path)
    df = _synthetic_df(n=MIN_SYNTHETIC_N.get(path, 800))
    signals = module.generate_signals(df.copy())
    train_end = int(len(df) * module.TRAIN_FRAC)
    assert (signals.iloc[:train_end] == 0).all()


@pytest.mark.parametrize("path", STRATEGY_PATHS)
def test_signals_are_only_minus1_0_1(path):
    module = _load_module(path)
    df = _synthetic_df(n=MIN_SYNTHETIC_N.get(path, 800))
    signals = module.generate_signals(df.copy())
    assert set(signals.unique()) <= {-1, 0, 1}


@pytest.mark.parametrize("path", STRATEGY_PATHS)
def test_explicit_wf_train_end_index_overrides_train_frac_split(path):
    module = _load_module(path)
    df = _synthetic_df(n=MIN_SYNTHETIC_N.get(path, 800))
    explicit_boundary = int(len(df) * 0.625)
    df.attrs["wf_train_end_index"] = explicit_boundary
    signals = module.generate_signals(df.copy())
    assert (signals.iloc[:explicit_boundary] == 0).all()


@pytest.mark.parametrize("path", STRATEGY_PATHS)
def test_walk_forward_fold_trades_stay_inside_the_test_window(path):
    strat = PythonStrategy(path)
    assert _wants_retrain_context(strat) is True
    df = _synthetic_df(n=max(MIN_SYNTHETIC_N.get(path, 800), 1500))
    folds = build_folds(df, n_folds=3, window_mode="rolling", train_frac=0.6)
    assert folds
    risk = RiskConfig(initial_balance=10_000, pip_size=0.01, spread_pips=1.0, slippage_pips=0.5)
    for fold in folds:
        result = _run_fold_test(fold, strat, risk)
        test_start = pd.to_datetime(fold.test_df["timestamp"]).iloc[0]
        test_end = pd.to_datetime(fold.test_df["timestamp"]).iloc[-1]
        for trade in result.trades:
            assert test_start <= trade.entry_time <= test_end
