"""v9.4: Quick Optimize must produce exactly ONE optimized strategy --
no version spam (Owen: "It produces several version of the same
strategy instead of just 1... I don't want a million versions of the
same strategy here") -- and saved names follow his simple formula:
instrument + timeframe + leading indicator + (TEST), e.g.
"ES1! 15m RSI (OPTIMIZED).json".
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRISTINE = Path.home() / "workspace" / "tmp-pristine"

from app.backtest.risk import RiskConfig  # noqa: E402
from app.orchestration.quick_optimize import (  # noqa: E402
    QuickOptimizeConfig, run_quick_optimize, run_quick_optimize_sweep,
)
from app.prop.simulator import PropRules  # noqa: E402
from app.strategy import library  # noqa: E402
from app.strategy.library import save_strategy_text  # noqa: E402
from app.strategy.manual import ManualStrategy  # noqa: E402


@pytest.fixture(autouse=True)
def clean_library_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    base_dir = library.get_strategy_library_dir()
    yield
    shutil.rmtree(base_dir, ignore_errors=True)


def _trending_df(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    price = 100.0
    rows = []
    for i in range(n):
        step = 0.02 + rng.normal(0, 0.05)
        o = price
        c = o + step
        rows.append((ts[i], o, max(o, c) + 0.01, min(o, c) - 0.01, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _rsi_config():
    return {
        "name": "rsi reversion",
        "indicators": [
            {"type": "rsi", "period": 14, "column": "close", "as": "rsi"},
        ],
        "long_entry": "rsi < 35",
        "long_exit": "rsi > 55",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }


def _cfg(**overrides):
    base = dict(ga_population=4, ga_generations=1, ga_search_mc_sims=20,
                final_mc_sims=100, n_folds=3, save_to_library=True, random_seed=7)
    base.update(overrides)
    return QuickOptimizeConfig(**base)


def _optimized_files() -> list[Path]:
    lib_dir = library.get_strategy_library_dir()
    return sorted(
        p for p in lib_dir.rglob("*.json")
        if ".versions" not in p.parts and not p.name.endswith(".meta.json")
    )


def test_repeat_runs_keep_exactly_one_optimized_file():
    df = _trending_df()
    r1 = run_quick_optimize(df, ManualStrategy(_rsi_config()), RiskConfig(), PropRules(),
                            _cfg(), progress_cb=None, instrument_label="ES1!.csv")
    r2 = run_quick_optimize(df, ManualStrategy(_rsi_config()), RiskConfig(), PropRules(),
                            _cfg(random_seed=8), progress_cb=None, instrument_label="ES1!.csv")
    files = _optimized_files()
    assert len(files) == 1, [p.name for p in files]
    assert files[0].name == "ES1! 15m RSI (OPTIMIZED).json"
    assert r1.saved_library_path == r2.saved_library_path == files[0]
    assert "exactly one optimized copy" in (r2.saved_library_note or "")


def test_replace_checkbox_replaces_the_original_file():
    df = _trending_df()
    import json
    original = save_strategy_text(json.dumps(_rsi_config(), indent=2), "my rsi.json", "manual")
    result = run_quick_optimize(
        df, ManualStrategy(_rsi_config()), RiskConfig(), PropRules(),
        _cfg(replace_existing=True, library_ref=("manual", "my rsi.json")),
        progress_cb=None, instrument_label="ES1!.csv",
    )
    assert result.saved_library_path == original
    assert _optimized_files() == [original]


def _sma_config():
    return {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }


def test_timeframe_sweep_saves_only_the_winner():
    df = _trending_df(n=4000)
    sweep = run_quick_optimize_sweep(
        df, ManualStrategy(_sma_config()), RiskConfig(), PropRules(),
        ["30min", "1h"], cfg=_cfg(), progress_cb=None, instrument_label="ES1!.csv",
    )
    files = _optimized_files()
    assert len(files) == 1, [p.name for p in files]
    assert sweep.best_result.saved_library_path == files[0]
    for label, res in sweep.per_timeframe.items():
        if label != sweep.best_timeframe:
            assert res.saved_library_path is None


def test_fails_on_upstream_v92():
    if not PRISTINE.exists():
        pytest.skip("pristine upstream copy not available")
    probe = "tests/test_v94_single_optimized_output.py"
    dest = PRISTINE / probe
    shutil.copy2(ROOT / probe, dest)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", probe, "-x", "-q",
             "-k", "not fails_on_upstream"],
            cwd=PRISTINE, capture_output=True, text=True, timeout=900)
    finally:
        dest.unlink(missing_ok=True)
    assert r.returncode != 0, r.stdout[-1200:]
