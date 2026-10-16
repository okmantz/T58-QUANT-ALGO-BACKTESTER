"""Regression test for the quick-optimize Python tmp_dir crash.

Bug: run_quick_optimize() called build_strategy_from_spec(final_spec) WITHOUT
tmp_dir. For Python strategies that raises StrategySpaceError ("Building a
Python strategy candidate requires a writable tmp_dir (PythonStrategy only
accepts a file path)") -- every Quick Optimize run on a Python strategy died
at the final re-validation step.

Fix: run_quick_optimize now creates final_tmp_dir via mkdtemp (same pattern as
Full Pipeline) for non-manual strategies and cleans it up before returning.
"""
import numpy as np
import pandas as pd

from app.backtest.risk import RiskConfig
from app.orchestration.quick_optimize import QuickOptimizeConfig, run_quick_optimize
from app.prop.simulator import PropRules
from app.strategy.python import PythonStrategy

_MINIMAL_PY = '''"""Minimal python strategy for the quick_optimize tmp_dir regression test."""
import pandas as pd

STRATEGY_NAME = "qo_tmpdir_regression"
FAST = 5
SLOW = 20


def generate_signals(df: pd.DataFrame) -> pd.Series:
    fast = df["close"].rolling(FAST, min_periods=1).mean()
    slow = df["close"].rolling(SLOW, min_periods=1).mean()
    sig = pd.Series(0, index=df.index)
    sig[fast > slow] = 1
    sig[fast < slow] = -1
    return sig
'''


def _trending_df(n=600, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    price, rows = 100.0, []
    for i in range(n):
        drift = 0.02 if (i // 60) % 2 == 0 else -0.015
        o = price
        c = o + drift + rng.normal(0, 0.05)
        h = max(o, c) + abs(rng.normal(0, 0.02))
        l = min(o, c) - abs(rng.normal(0, 0.02))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_run_quick_optimize_python_strategy_completes(tmp_path):
    """The reported crash: used to raise StrategySpaceError at the final
    build_strategy_from_spec(final_spec) call. Must now complete and return
    a result."""
    strat_file = tmp_path / "qo_tmpdir_regression.py"
    strat_file.write_text(_MINIMAL_PY, encoding="utf-8")
    strategy = PythonStrategy(strat_file)

    df = _trending_df()
    cfg = QuickOptimizeConfig(
        ga_population=4,
        ga_generations=1,
        ga_search_mc_sims=20,
        final_mc_sims=50,
        n_folds=2,
        save_to_library=False,
        parallel=False,
        random_seed=42,
    )
    result = run_quick_optimize(df, strategy, RiskConfig(), PropRules(), cfg, progress_cb=None)
    assert result is not None
    assert result.optimized_trades >= 0
