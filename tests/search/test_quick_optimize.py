import shutil

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration.quick_optimize import QuickOptimizeConfig, run_quick_optimize
from app.prop.simulator import PropRules
from app.strategy import library
from app.strategy.manual import ManualStrategy


@pytest.fixture(autouse=True)
def clean_library_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    base_dir = library.get_strategy_library_dir()
    yield
    shutil.rmtree(base_dir, ignore_errors=True)


def _trending_df(n=1500, seed=3, drift=0.00015, base_price=1.1000):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = base_price
    rows = []
    for i in range(n):
        step = drift * (1 if (i // 40) % 2 == 0 else -1) + rng.normal(0, 0.00006) * base_price
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00003)) * base_price
        l = min(o, c) - abs(rng.normal(0, 0.00003)) * base_price
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config(fast=5, slow=15, stop_loss_pips=None):
    cfg = {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": fast, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": slow, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
    }
    if stop_loss_pips is not None:
        cfg["stop_loss_pips"] = stop_loss_pips
    return cfg


def _cfg(**overrides):
    base = dict(ga_population=4, ga_generations=1, ga_search_mc_sims=20, final_mc_sims=200, n_folds=3)
    base.update(overrides)
    return QuickOptimizeConfig(**base)


# ---------------------------------------------------------------------------
# #3 -- fixed random_seed default
# ---------------------------------------------------------------------------

def test_default_random_seed_is_42():
    assert QuickOptimizeConfig().random_seed == 42


def test_same_seed_reproduces_identical_results(tmp_path):
    df = _trending_df()
    strategy_a = ManualStrategy(_sma_config())
    strategy_b = ManualStrategy(_sma_config())
    res_a = run_quick_optimize(df, strategy_a, RiskConfig(), PropRules(), _cfg(save_to_library=False), progress_cb=None)
    res_b = run_quick_optimize(df, strategy_b, RiskConfig(), PropRules(), _cfg(save_to_library=False), progress_cb=None)
    assert res_a.optimized_trades == res_b.optimized_trades
    assert res_a.optimized_net_profit == pytest.approx(res_b.optimized_net_profit)
    assert res_a.final_parameters == res_b.final_parameters


# ---------------------------------------------------------------------------
# #4 -- instrument-mismatch warning escalation
# ---------------------------------------------------------------------------

def test_instrument_mismatch_surfaced_prominently_on_gold_scale_instrument(tmp_path):
    # Gold-like price scale (~2000) with an FX-default pip_size (0.0001)
    # and a tight fixed-pips stop -- the exact failure mode from the
    # Quick-Optimize-vs-Full-Pipeline diagnosis.
    df = _trending_df(base_price=2000.0, drift=0.3)
    strategy = ManualStrategy(_sma_config(stop_loss_pips=25))
    result = run_quick_optimize(
        df, strategy, RiskConfig(pip_size=0.0001), PropRules(), _cfg(save_to_library=False), progress_cb=None,
    )
    assert result.instrument_mismatch_warning is not None
    assert "pip_size" in result.instrument_mismatch_warning.lower()


def test_no_mismatch_warning_when_pip_size_is_correct(tmp_path):
    df = _trending_df()  # ordinary FX-scale price (~1.10), default pip_size 0.0001 is correct here
    strategy = ManualStrategy(_sma_config(stop_loss_pips=25))
    result = run_quick_optimize(
        df, strategy, RiskConfig(pip_size=0.0001), PropRules(), _cfg(save_to_library=False), progress_cb=None,
    )
    assert result.instrument_mismatch_warning is None
