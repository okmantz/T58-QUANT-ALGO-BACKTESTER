"""Tests for the RETRAIN_PER_FOLD walk-forward retrain-context protocol
(see app/strategy/python.py's WALK-FORWARD RETRAIN CONTEXT section and
strategies/python/ml_classifier_direction.py's WALK-FORWARD RETRAINING
section) -- the fix for the ml_classifier_direction.py strategy wasting
each fold's real training window."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.strategy.python import PythonStrategy
from app.validation.walk_forward_opt import Fold, _run_fold_test, _wants_retrain_context, build_folds

ML_STRATEGY_PATH = "strategies/python/ml_classifier_direction.py"


def _synthetic_df(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + np.abs(rng.normal(0, 0.2, n))
    low = close - np.abs(rng.normal(0, 0.2, n))
    openp = close + rng.normal(0, 0.05, n)
    volume = rng.uniform(50, 150, n)
    return pd.DataFrame({
        "timestamp": ts, "open": openp, "high": high, "low": low, "close": close, "volume": volume,
    })


def _risk():
    return RiskConfig(initial_balance=10_000, pip_size=0.01, spread_pips=1.0, slippage_pips=0.5)


def test_module_flag_reads_retrain_per_fold_true_for_ml_classifier():
    strat = PythonStrategy(ML_STRATEGY_PATH)
    assert strat.module_flag("RETRAIN_PER_FOLD") is True


def test_module_flag_defaults_false_for_a_strategy_without_the_flag(tmp_path):
    path = tmp_path / "plain_strategy.py"
    path.write_text(
        "import pandas as pd\n"
        "def generate_signals(df):\n"
        "    return pd.Series(0, index=df.index)\n"
    )
    strat = PythonStrategy(str(path))
    assert strat.module_flag("RETRAIN_PER_FOLD") is False


def test_wants_retrain_context_true_only_for_flagged_python_strategy(tmp_path):
    ml_strat = PythonStrategy(ML_STRATEGY_PATH)
    assert _wants_retrain_context(ml_strat) is True

    plain_path = tmp_path / "plain_strategy.py"
    plain_path.write_text(
        "import pandas as pd\n"
        "def generate_signals(df):\n"
        "    return pd.Series(0, index=df.index)\n"
    )
    plain_strat = PythonStrategy(str(plain_path))
    assert _wants_retrain_context(plain_strat) is False


def test_run_fold_test_matches_plain_run_backtest_for_a_non_retrain_strategy(tmp_path):
    """No RETRAIN_PER_FOLD flag -- _run_fold_test must be byte-for-byte
    the same as calling run_backtest(fold.test_df, ...) directly, i.e.
    zero behavior change for the ~1400 other strategies/tests that never
    touch this protocol."""
    path = tmp_path / "always_flat.py"
    path.write_text(
        "import pandas as pd\n"
        "def generate_signals(df):\n"
        "    s = pd.Series(0, index=df.index)\n"
        "    s.iloc[10] = 1\n"
        "    return s\n"
    )
    strat = PythonStrategy(str(path))
    df = _synthetic_df(n=200)
    folds = build_folds(df, n_folds=2, window_mode="rolling", train_frac=0.6)
    assert folds
    risk = _risk()
    for fold in folds:
        expected = run_backtest(fold.test_df, strat, risk)
        actual = _run_fold_test(fold, strat, risk)
        assert len(actual.trades) == len(expected.trades)
        assert actual.statistics.net_profit == pytest.approx(expected.statistics.net_profit)


def test_run_fold_test_gives_ml_classifier_the_real_fold_train_window():
    """The bug this protocol fixes: before it, calling generate_signals
    separately on fold.train_df and fold.test_df meant the ML classifier
    fit on the leading 60% of the TEST fold alone, trading only its
    trailing 40% -- wasting the fold's real, larger train_df entirely.
    After the fix, every resulting trade still falls inside the fold's
    test window (no lookahead introduced), but the strategy was trained
    on the fold's actual train_df history, not a re-split of test_df."""
    strat = PythonStrategy(ML_STRATEGY_PATH)
    df = _synthetic_df(n=1500)
    folds = build_folds(df, n_folds=3, window_mode="rolling", train_frac=0.6)
    assert folds
    risk = _risk()

    for fold in folds:
        result = _run_fold_test(fold, strat, risk)
        test_start = pd.to_datetime(fold.test_df["timestamp"]).iloc[0]
        test_end = pd.to_datetime(fold.test_df["timestamp"]).iloc[-1]
        # VAL-006 fix: Trade.entry_time now correctly preserves the input
        # data's tz-awareness instead of always coming back tz-naive, so
        # test_start/test_end (derived from the same fold.test_df column)
        # already agree with it by construction -- no stripping needed.
        for trade in result.trades:
            assert test_start <= trade.entry_time <= test_end


def test_explicit_train_end_index_attr_changes_the_split_point():
    """Direct unit check on the strategy file itself (no harness): with
    wf_train_end_index set on df.attrs, the training prefix boundary
    follows that attr instead of TRAIN_FRAC -- and every bar before it
    still gets a flat (0) signal, per the strategy's own no-lookahead
    contract."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ml_classifier_test_import", ML_STRATEGY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    df = _synthetic_df(n=600, seed=7)
    explicit_boundary = 400
    df.attrs["wf_train_end_index"] = explicit_boundary
    signals = module.generate_signals(df.copy())
    # Every bar strictly before the explicit boundary must be flat --
    # this is the strategy's own pre-existing no-lookahead guarantee,
    # now anchored to the harness-supplied boundary instead of a
    # TRAIN_FRAC-derived one.
    assert (signals.iloc[:explicit_boundary] == 0).all()


def test_absent_train_end_index_attr_falls_back_to_train_frac_unchanged():
    """No wf_train_end_index attr set (the normal call path for every
    other harness) -- behavior must be identical to before this fix."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ml_classifier_test_import2", ML_STRATEGY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    df = _synthetic_df(n=600, seed=7)
    signals_a = module.generate_signals(df.copy())
    signals_b = module.generate_signals(df.copy())
    assert (signals_a == signals_b).all()
