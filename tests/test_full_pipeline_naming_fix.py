"""
Covers the app.orchestration.full_pipeline half of the 2026-09-17
Quick-Optimize-vs-Full-Pipeline naming-drift fix (point 3): a GA-mutated
manual configuration's saved filename and internal "name" field must
carry app.strategy.library.provenance_stamped_name's stamp, not the
pre-mutation display name verbatim. See
tests/test_quick_optimize_not_validated_fix.py for the Quick Optimize
half of this same fix.
"""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration.full_pipeline import FullPipelineConfig, run_full_pipeline
from app.prop.simulator import PropRules
from app.strategy import library
from app.strategy.manual import ManualStrategy


@pytest.fixture(autouse=True)
def clean_library_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    base_dir = library.get_strategy_library_dir()
    yield
    shutil.rmtree(base_dir, ignore_errors=True)


def _trending_df(n=2400, seed=3, drift=0.00015):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift * (1 if (i // 40) % 2 == 0 else -1) + rng.normal(0, 0.00006)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00003))
        l = min(o, c) - abs(rng.normal(0, 0.00003))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config(fast=5, slow=15):
    return {
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


def test_mutated_manual_config_saved_under_stamped_name(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    cfg = FullPipelineConfig(
        ga_population=4, ga_generations=1, ga_search_mc_sims=20, final_mc_sims=200,
        n_folds=3, save_to_library=True, random_seed=11,
    )
    result = run_full_pipeline(df, strategy, RiskConfig(), PropRules(), tmp_path / "fp_out", cfg,
                                report_basename="naming_fix_test")
    if result.refinement_ran and result.saved_library_path is not None:
        assert "full_pipeline" in result.saved_library_path.name
        assert "seed_11" in result.saved_library_path.name
        saved = json.loads(result.saved_library_path.read_text())
        # Original display name ("sma cross") should still be the PREFIX --
        # provenance_stamped_name appends, never replaces.
        assert saved["name"].startswith("sma cross [")
        assert "full_pipeline" in saved["name"]
        assert "seed=11" in saved["name"]
