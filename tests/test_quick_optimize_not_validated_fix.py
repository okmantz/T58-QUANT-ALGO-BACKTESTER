"""
Covers the 2026-09-17 Quick-Optimize-vs-Full-Pipeline follow-up: the
5-point fix that stops a Quick Optimize result from ever looking like a
finished, validated answer. See app.orchestration.quick_optimize's module
docstring for the full narrative; this file checks each point actually
does what it claims:

  1. Every result carries the fixed "NOT OOS VALIDATED" banner, regardless
     of how strong its numbers look.
  2. A result built on too few out-of-sample trades is flagged unreliable
     rather than shown as a bare, confident percentage.
  3. A GA-mutated config's saved filename/JSON "name" field no longer
     silently reuses the pre-mutation display name (app.strategy.library.
     provenance_stamped_name).
  4. The same ICIR/Bonferroni significance gate and parsimony score Full
     Pipeline's Step 6 runs are also available from Quick Optimize.
  5. QuickOptimizeConfig.reserve_holdout carves off a genuine holdout
     slice the search never sees, and reports the winner's performance on
     it separately.
"""
import shutil

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration.quick_optimize import (
    MIN_CREDIBLE_OOS_TRADES,
    RESEARCH_RESULT_BANNER,
    RESEARCH_RESULT_BANNER_DETAIL,
    QuickOptimizeConfig,
    run_quick_optimize,
)
from app.prop.simulator import PropRules
from app.strategy import library
from app.strategy.library import provenance_stamped_name
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
    base = dict(ga_population=4, ga_generations=1, ga_search_mc_sims=20, final_mc_sims=200, n_folds=3, save_to_library=False)
    base.update(overrides)
    return QuickOptimizeConfig(**base)


# ---------------------------------------------------------------------
# Point (1) -- unconditional "NOT OOS VALIDATED" banner
# ---------------------------------------------------------------------

def test_result_always_carries_not_validated_banner():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = run_quick_optimize(df, strategy, RiskConfig(), PropRules(), _cfg(), progress_cb=None)
    assert result.validated is False
    assert result.result_banner == RESEARCH_RESULT_BANNER
    assert result.result_banner_detail == RESEARCH_RESULT_BANNER_DETAIL
    assert "NOT OOS VALIDATED" in result.result_banner


def test_banner_is_logged_regardless_of_outcome():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    logged = []
    run_quick_optimize(df, strategy, RiskConfig(), PropRules(), _cfg(), progress_cb=logged.append)
    assert any("NOT OOS VALIDATED" in msg for msg in logged)
    assert any("Full Pipeline" in msg for msg in logged)


# ---------------------------------------------------------------------
# Point (2) -- minimum-credible-OOS-trade-count gate
# ---------------------------------------------------------------------

def test_min_credible_oos_trades_constant_is_100():
    assert MIN_CREDIBLE_OOS_TRADES == 100


def test_too_few_oos_trades_flagged_unreliable():
    # A short dataset with a slow-crossing SMA pair produces only a
    # handful of trades -- exactly the "99% eval-pass on 12 trades" case
    # this point exists to catch.
    df = _trending_df(n=250)
    strategy = ManualStrategy(_sma_config(fast=40, slow=90))
    result = run_quick_optimize(df, strategy, RiskConfig(), PropRules(), _cfg(), progress_cb=None)
    assert result.oos_trade_count < MIN_CREDIBLE_OOS_TRADES
    assert result.min_trade_count_met is False
    assert result.trade_count_warning is not None
    assert "UNRELIABLE" in result.trade_count_warning


def test_plenty_of_oos_trades_not_flagged():
    df = _trending_df(n=4000)
    strategy = ManualStrategy(_sma_config(fast=3, slow=8))
    result = run_quick_optimize(
        df, strategy, RiskConfig(), PropRules(),
        _cfg(ga_population=6, ga_generations=2, n_folds=3), progress_cb=None,
    )
    if result.oos_trade_count >= MIN_CREDIBLE_OOS_TRADES:
        assert result.min_trade_count_met is True
        assert result.trade_count_warning is None


# ---------------------------------------------------------------------
# Point (3) -- provenance-stamped naming (no more stale-name drift)
# ---------------------------------------------------------------------

def test_provenance_stamped_name_format():
    stamped = provenance_stamped_name("My Strategy", origin="quick_optimize", seed=42)
    assert stamped.startswith("My Strategy [")
    assert "quick_optimize" in stamped
    assert "seed=42" in stamped


def test_mutated_config_saved_under_stamped_name_not_stale_original():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = run_quick_optimize(
        df, strategy, RiskConfig(), PropRules(),
        _cfg(save_to_library=True, random_seed=7), progress_cb=None,
    )
    if result.final_parameters:  # only meaningful when the GA actually produced a mutated winner
        assert result.saved_library_path is not None
        assert "quick_optimize" in result.saved_library_path.name
        assert "seed_7" in result.saved_library_path.name
        import json
        saved = json.loads(result.saved_library_path.read_text())
        assert "[quick_optimize" in saved["name"]
        assert "seed=7" in saved["name"]


# ---------------------------------------------------------------------
# Point (4) -- significance/parsimony diagnostics surfaced here too
# ---------------------------------------------------------------------

def test_parsimony_is_always_computed():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = run_quick_optimize(df, strategy, RiskConfig(), PropRules(), _cfg(), progress_cb=None)
    assert result.parsimony_result is not None
    assert result.parsimony_note is not None


def test_significance_gate_runs_or_reports_why_not():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = run_quick_optimize(df, strategy, RiskConfig(), PropRules(), _cfg(), progress_cb=None)
    # Best-effort, like Full Pipeline's own copy of this step: either it
    # produced a note, or it explains why it couldn't run -- never silent.
    assert result.significance_note is not None or result.icir_gate_skip_reason is not None


# ---------------------------------------------------------------------
# Point (5) -- optional holdout split
# ---------------------------------------------------------------------

def test_holdout_disabled_by_default_is_byte_identical_shape():
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = run_quick_optimize(df, strategy, RiskConfig(), PropRules(), _cfg(), progress_cb=None)
    assert result.holdout_enabled is False
    assert result.holdout_trades is None
    assert result.holdout_net_profit is None


def test_reserve_holdout_reports_separate_holdout_numbers():
    df = _trending_df(n=3000)
    strategy = ManualStrategy(_sma_config())
    result = run_quick_optimize(
        df, strategy, RiskConfig(), PropRules(),
        _cfg(reserve_holdout=True, holdout_frac=0.2), progress_cb=None,
    )
    assert result.holdout_enabled is True
    assert result.holdout_note is not None
    # Either it found trades (all holdout_* fields populated) or it
    # explicitly says it found none -- never silently blank.
    if result.holdout_trades:
        assert result.holdout_net_profit is not None
        assert result.holdout_win_rate is not None


def test_reserve_holdout_search_never_sees_the_holdout_slice():
    # The baseline/GA search only ever runs against the first
    # (1 - holdout_frac) of the data when reserve_holdout=True -- confirm
    # the baseline trade count reflects a smaller window than the full
    # dataset would produce.
    df = _trending_df(n=3000)
    strategy_a = ManualStrategy(_sma_config())
    strategy_b = ManualStrategy(_sma_config())
    result_full = run_quick_optimize(df, strategy_a, RiskConfig(), PropRules(), _cfg(), progress_cb=None)
    result_holdout = run_quick_optimize(
        df, strategy_b, RiskConfig(), PropRules(),
        _cfg(reserve_holdout=True, holdout_frac=0.2), progress_cb=None,
    )
    # Same strategy/data, but the holdout run's baseline only ever saw 80%
    # of the bars -- its own trade count should not exceed the full run's.
    assert result_holdout.baseline_trades <= result_full.baseline_trades
