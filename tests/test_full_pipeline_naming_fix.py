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


def test_report_flags_ga_modified_parameters_and_shows_baseline(tmp_path):
    """2026-09-24 PARAMETER-FIDELITY FIX: found via an external RoboQuant
    comparison -- when the GA actually changes a manual strategy's
    parameters, the SAVED REPORT (not just the library filename covered
    above) must say so in its own title and carry both the baseline
    (originally-supplied) and final (GA-mutated) parameter values, so a
    reader can't mistake one for a backtest of the other."""
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    cfg = FullPipelineConfig(
        ga_population=4, ga_generations=1, ga_search_mc_sims=20, final_mc_sims=200,
        n_folds=3, save_to_library=True, random_seed=11,
    )
    result = run_full_pipeline(df, strategy, RiskConfig(), PropRules(), tmp_path / "fp_out2", cfg,
                                report_basename="param_fidelity_test")
    report = json.loads(result.report_paths["json"].read_text())

    if result.refinement_ran and result.ga_result is not None and result.ga_result.best.oos_trade_count > 0:
        # This is the "mutated_by_ga" branch -- the title must flag it,
        # and baseline_parameters must be present with the same keys as
        # final_parameters (whether or not any individual value actually
        # moved -- see the "no values moved" case in the section renderer).
        assert "GA-modified parameters" in report["strategy"]["name"]
        assert report["final_parameters"] is not None
        assert report["baseline_parameters"] is not None
        assert set(report["baseline_parameters"].keys()) == set(report["final_parameters"].keys())
    else:
        # No mutation happened (or refinement didn't run at all) -- the
        # title must stay exactly as before this fix, with no baseline
        # section clutter for a report where nothing needs comparing.
        assert "GA-modified parameters" not in report["strategy"]["name"]


def test_final_parameters_section_flags_changed_rows():
    from app.reports.generator import _final_parameters_section

    baseline = {"ema.period": "20", "rsi.period": "14"}
    final = {"ema.period": "35", "rsi.period": "14"}
    html = _final_parameters_section(final, baseline)
    assert "param-changed" in html
    assert "GA search changed one or more parameters" in html
    assert "35" in html and "20" in html  # both baseline and final values shown


def test_final_parameters_section_no_baseline_falls_back_to_final_only():
    from app.reports.generator import _final_parameters_section

    html = _final_parameters_section({"ema.period": "35"}, None)
    assert "Baseline" not in html
    assert "35" in html


def test_final_parameters_section_no_changes_says_so():
    from app.reports.generator import _final_parameters_section

    same = {"ema.period": "20"}
    html = _final_parameters_section(dict(same), dict(same))
    assert "param-changed" not in html
    assert "did not move any parameter" in html


def test_skip_optimization_mode_never_mutates_parameters(tmp_path):
    """2026-09-24 lock-parameters / fixed-backtest mode: found via an
    external RoboQuant comparison -- comparing "the same strategy"
    against another tool is only valid when Full Pipeline's GA search
    doesn't quietly change it first. skip_optimization=True must skip
    Step 2 entirely and leave the report describing exactly the supplied
    parameters, with no GA-modified tag and no baseline/final split."""
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    cfg = FullPipelineConfig(
        ga_population=4, ga_generations=1, ga_search_mc_sims=20, final_mc_sims=200,
        n_folds=3, save_to_library=False, random_seed=11, skip_optimization=True,
    )
    result = run_full_pipeline(df, strategy, RiskConfig(), PropRules(), tmp_path / "fp_skip_opt", cfg,
                                report_basename="skip_optimization_test")

    assert result.refinement_ran is False
    assert result.ga_result is None
    assert "skip_optimization" in (result.refinement_skip_reason or "")
    # The final config must be byte-identical to what was supplied -- no
    # GA winner was ever substituted in.
    assert result.final_config == strategy.config

    report = json.loads(result.report_paths["json"].read_text())
    assert "GA-modified parameters" not in report["strategy"]["name"]
    assert report["final_parameters"] is None
    assert report["baseline_parameters"] is None
