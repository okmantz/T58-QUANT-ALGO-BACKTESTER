import shutil

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration.full_pipeline import FullPipelineConfig, run_full_pipeline
from app.prop.simulator import PropRules
from app.scoring.leaderboard import build_leaderboard, render_leaderboard_table
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


def _cfg(**overrides):
    base = dict(
        n_folds=3, ga_population=4, ga_generations=1, ga_search_mc_sims=20,
        final_mc_sims=200, oos_check_folds=3, holdout_frac=0.2,
    )
    base.update(overrides)
    return FullPipelineConfig(**base)


def test_leaderboard_empty_library_returns_empty_list_and_friendly_message():
    assert build_leaderboard() == []
    assert "No strategies" in render_leaderboard_table([])


def test_leaderboard_ranks_a_strategy_after_full_pipeline_run(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path, _cfg(), progress_cb=None,
    )
    entries = build_leaderboard()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.strategy_type == "manual"
    assert 0.0 <= entry.t58_score <= 100.0
    assert entry.t58_tier in ("Elite", "Strong", "Promising", "Research", "Reject")
    table = render_leaderboard_table(entries)
    assert "FINAL SELECTION LEADERBOARD" in table
    # render_leaderboard_table intentionally truncates the Strategy column
    # to 33 characters (see its own f"{e.filename[:33]:<34}" formatting) --
    # a full filename longer than that (e.g. a provenance-stamped one from
    # the 2026-09-17 naming-drift fix, app.strategy.library.
    # provenance_stamped_name) is expected to appear truncated in the
    # table, not verbatim.
    assert entry.filename[:33] in table


def test_leaderboard_excludes_risk_of_ruin_hard_fails_by_default(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    # An unreachable cap of -1 forces every run to hard-fail the ruin gate
    # regardless of its actual Monte Carlo result -- deterministic way to
    # exercise the exclusion without depending on a specific ruin number.
    run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path, _cfg(risk_of_ruin_cap=-1.0), progress_cb=None,
    )
    assert build_leaderboard(exclude_ruin_hard_fail=True) == []
    entries_including = build_leaderboard(exclude_ruin_hard_fail=False)
    assert len(entries_including) == 1
    assert entries_including[0].risk_of_ruin_hard_fail is True
