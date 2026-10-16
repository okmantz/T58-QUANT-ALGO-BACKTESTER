"""Speed optimizations (2026-10-06) -- same answers, less wall-clock.

1. GA flatline early stop (app.optimize.walkforward_ga): a population
   where EVERY candidate scores exactly 0.0 fitness for `flatline_
   patience` straight generations has no selection gradient; the
   remaining generations only re-score mutations of a dead population.
   Owen's 2020-2026 ES run printed best=0.000/mean=0.000 for 7 straight
   generations (760s) before the pipeline moved on.

2. Full Pipeline / Quick Optimize dedup: when the search returns the
   baseline configuration unchanged, the "final" full backtest (and, in
   Quick Optimize, the final Monte Carlo) used to be re-run bit-for-bit.
   They now reuse the baseline results. No gate, threshold, or
   simulation count changes anywhere.
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

import app.optimize.walkforward_ga as wga
import app.orchestration.full_pipeline as fp
from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.refinement import RefinementConfig
from app.optimize.walkforward_ga import _flatlined, run_walkforward_aware_refinement
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


def _trending_df(n=900, seed=3, drift=0.00015):
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


# ---------------------------------------------------------------------------
# _flatlined unit tests
# ---------------------------------------------------------------------------

class _S:
    def __init__(self, best, mean):
        self.best_fitness = best
        self.mean_fitness = mean


def test_flatlined_requires_full_patience_of_exact_zeros():
    assert _flatlined([_S(0.0, 0.0)] * 3, 3)
    assert not _flatlined([_S(0.0, 0.0)] * 2, 3)          # not enough history yet
    assert not _flatlined([_S(0.0, 0.0), _S(0.0, 0.0), _S(0.1, 0.0)], 3)  # a live best breaks it
    assert not _flatlined([_S(0.0, 0.0), _S(0.0, 0.2), _S(0.0, 0.0)], 3)  # a live mean breaks it
    assert not _flatlined([_S(0.0, 0.0)] * 5, 0)          # patience 0 disables the stop


# ---------------------------------------------------------------------------
# GA integration: dead population stops early; disabling restores full run
# ---------------------------------------------------------------------------

def _run_ga(monkeypatch, patience):
    monkeypatch.setattr(wga, "compute_fitness", lambda *a, **k: 0.0)
    refine_cfg = RefinementConfig(
        population_size=4, generations=6, search_monte_carlo_sims=20,
        flatline_patience=patience, auto_shrink_on_low_trades=False,
    )
    return run_walkforward_aware_refinement(
        _trending_df(), ManualStrategy(_sma_config()), RiskConfig(), PropRules(),
        MonteCarloConfig(n_simulations=20), refinement_config=refine_cfg,
        n_folds=3, parallel=False,
    )


def test_ga_stops_early_when_every_candidate_scores_zero(monkeypatch):
    result = _run_ga(monkeypatch, patience=3)
    # Generations 0,1,2 evaluated, then the stop fired (of 0..6 possible).
    assert len(result.generation_history) == 3
    assert any("exactly 0.0 fitness" in w for w in result.warnings)


def test_ga_flatline_stop_can_be_disabled(monkeypatch):
    result = _run_ga(monkeypatch, patience=0)
    assert len(result.generation_history) == 7  # generations 0..6, unchanged behavior


# ---------------------------------------------------------------------------
# Full Pipeline: identical final configuration reuses the baseline backtest
# ---------------------------------------------------------------------------

def test_full_pipeline_reuses_baseline_backtest_when_final_matches(tmp_path, monkeypatch):
    calls = {"n": 0}
    real_run_backtest = fp.run_backtest

    def counting(df_, strategy_, risk_, **kw):
        calls["n"] += 1
        return real_run_backtest(df_, strategy_, risk_, **kw)

    monkeypatch.setattr(fp, "run_backtest", counting)
    cfg = FullPipelineConfig(
        n_folds=3, ga_population=4, ga_generations=1, ga_search_mc_sims=20,
        final_mc_sims=200, holdout_frac=0.2,
        skip_optimization=True, reserve_true_holdout=False,
    )
    result = run_full_pipeline(
        _trending_df(n=2400), ManualStrategy(_sma_config()), RiskConfig(),
        PropRules(), tmp_path, cfg, progress_cb=None,
    )
    assert result.final_bt is result.baseline_bt
    assert calls["n"] == 1, (
        f"expected only the Step 1 baseline backtest at pipeline level, got {calls['n']} "
        "(Step 3 re-ran an identical configuration)"
    )


# ---------------------------------------------------------------------------
# Verdict guidance: structured parts for the web UI (string form unchanged)
# ---------------------------------------------------------------------------

from types import SimpleNamespace

import app.orchestration.pipeline_guide as pg


def _ruin_result():
    return SimpleNamespace(
        lookahead_hard_fail=False,
        risk_of_ruin_hard_fail=True,
        risk_of_ruin_cap=20.0,
        final_mc=SimpleNamespace(risk_of_ruin_pct=100.0, reset_on_breach=True),
        scorecard=SimpleNamespace(score=2.3, tier="Reject"),
        warnings=[],
        final_bt=SimpleNamespace(warnings=[]),
        final_holdout={},
    )


def test_after_full_pipeline_parts_ruin_branch_is_structured():
    parts = pg.after_full_pipeline_parts("NOT READY", False, result=_ruin_result())
    assert "risk of ruin (100.0%)" in parts["headline"]
    assert "20% cap" in parts["headline"]
    assert len(parts["steps"]) >= 2
    assert parts["closing"].startswith("After any change")
    # The single-string form (desktop UI) still carries every piece.
    text = pg.after_full_pipeline("NOT READY", False, result=_ruin_result())
    assert parts["headline"] in text
    for step in parts["steps"]:
        assert step in text
    assert "Next steps, in order: 1)" in text


def test_after_full_pipeline_parts_ready_and_generic():
    ready = pg.after_full_pipeline_parts("READY", True)
    assert ready["headline"] == "Verdict: READY."
    assert any("saved to the Strategy Library" in p for p in ready["points"])
    assert "forward-test" in pg.after_full_pipeline("READY", True)
    generic = pg.after_full_pipeline_parts("NOT READY", False)
    assert generic["headline"] == "Verdict: NOT READY."
    assert generic["steps"] == []
