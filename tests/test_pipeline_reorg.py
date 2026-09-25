import shutil

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration.full_pipeline import FullPipelineConfig, _make_verdict, run_full_pipeline
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


def _cfg(**overrides):
    base = dict(
        n_folds=3, ga_population=4, ga_generations=1, ga_search_mc_sims=20,
        final_mc_sims=200, oos_check_folds=3, holdout_frac=0.2,
    )
    base.update(overrides)
    return FullPipelineConfig(**base)


# ---------------------------------------------------------------------------
# #1/#2 -- scorecard verdict + risk-of-ruin hard gate
# ---------------------------------------------------------------------------

def test_ruin_hard_fail_forces_not_ready_regardless_of_scorecard(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    # -1.0 is unreachable -- guarantees the hard gate fires even if the
    # strategy's actual Monte Carlo risk of ruin happens to be 0%.
    result = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path, _cfg(risk_of_ruin_cap=-1.0), progress_cb=None,
    )
    assert result.verdict == "NOT READY"
    assert result.risk_of_ruin_hard_fail is True
    assert any("HARD SAFETY GATE FAILED" in r for r in result.verdict_reasons)
    # The scorecard is still computed and attached (for the report/leaderboard)
    # even though it wasn't what decided this verdict.
    assert result.scorecard is not None


def test_scorecard_attached_and_verdict_matches_tier_mapping(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path, _cfg(), progress_cb=None,
    )
    assert result.scorecard is not None
    tier = result.scorecard.tier
    if result.risk_of_ruin_hard_fail:
        assert result.verdict == "NOT READY"
    elif tier in ("Elite", "Strong"):
        assert result.verdict == "READY"
    elif tier in ("Promising", "Research"):
        assert result.verdict == "MARGINAL"
    else:
        assert result.verdict == "NOT READY"


def test_make_verdict_never_requires_every_metric_to_pass_simultaneously():
    """Regression guard for the old 5-way-AND behavior this replaces:
    a strategy with strong pass/payout probabilities and a real but
    imperfect walk-forward result should NOT be forced to NOT READY just
    because one supporting signal (ICIR) is missing/failed -- see
    pipeline reorg plan section 4."""
    from dataclasses import dataclass

    @dataclass
    class _FakeMC:
        evaluation_pass_probability: float = 90.0
        first_payout_probability: float = 80.0
        risk_of_ruin_pct: float = 5.0
        return_percentiles: dict = None

        def __post_init__(self):
            if self.return_percentiles is None:
                self.return_percentiles = {25: 8.0, 50: 10.0, 75: 12.0}

    @dataclass
    class _FakeWF:
        n_folds: int = 4
        walk_forward_efficiency: float = 0.85
        is_stable: bool = True
        stability_threshold: float = 0.4

    verdict, reasons, scorecard, hard_fail, lookahead_hard_fail = _make_verdict(
        _FakeMC(), _FakeWF(), icir_gate=None, risk_of_ruin_cap=20.0,
    )
    assert hard_fail is False
    assert lookahead_hard_fail is False
    assert verdict in ("READY", "MARGINAL")  # not dragged to NOT READY by the missing ICIR gate alone
    assert scorecard.n_components_used >= 4


# ---------------------------------------------------------------------------
# Minimum trade-count floor for READY (2026-09-24) -- see
# FullPipelineConfig.min_trades_for_ready's own docstring for why this
# exists: found via an external RoboQuant comparison where a strategy on
# just 40 trades over 6 years, with one trade responsible for a third of
# its gross profit, would otherwise have scored well enough to be
# considered for READY.
# ---------------------------------------------------------------------------

def _elite_leaning_mc_and_wf():
    from dataclasses import dataclass

    @dataclass
    class _FakeMC:
        evaluation_pass_probability: float = 95.0
        first_payout_probability: float = 90.0
        risk_of_ruin_pct: float = 2.0
        return_percentiles: dict = None

        def __post_init__(self):
            if self.return_percentiles is None:
                self.return_percentiles = {25: 10.0, 50: 15.0, 75: 20.0}

    @dataclass
    class _FakeWF:
        n_folds: int = 6
        walk_forward_efficiency: float = 0.9
        is_stable: bool = True
        stability_threshold: float = 0.4

    return _FakeMC(), _FakeWF()


def _fake_statistics(total_trades: int):
    from dataclasses import dataclass

    @dataclass
    class _FakeStats:
        total_trades: int

        def to_dict(self):
            return {}

    return _FakeStats(total_trades=total_trades)


def test_min_trades_floor_caps_thin_sample_at_marginal():
    mc, wf = _elite_leaning_mc_and_wf()
    verdict, reasons, scorecard, hard_fail, lookahead_hard_fail = _make_verdict(
        mc, wf, icir_gate=None, risk_of_ruin_cap=20.0,
        statistics=_fake_statistics(40), min_trades_for_ready=100,
    )
    assert scorecard.tier in ("Elite", "Strong"), "test setup should have earned READY on the score alone"
    assert verdict == "MARGINAL"
    assert any("CAPPED AT MARGINAL" in r for r in reasons)
    assert any("40 trade" in r for r in reasons)


def test_min_trades_floor_does_not_affect_a_healthy_sample():
    mc, wf = _elite_leaning_mc_and_wf()
    verdict, reasons, scorecard, hard_fail, lookahead_hard_fail = _make_verdict(
        mc, wf, icir_gate=None, risk_of_ruin_cap=20.0,
        statistics=_fake_statistics(250), min_trades_for_ready=100,
    )
    assert verdict == "READY"
    assert not any("CAPPED AT MARGINAL" in r for r in reasons)


def test_min_trades_floor_never_downgrades_an_already_marginal_verdict():
    """A strategy that wasn't going to be READY anyway shouldn't get an
    extra, misleading 'capped' reason attached -- its score already
    explains the verdict."""
    from dataclasses import dataclass

    @dataclass
    class _WeakMC:
        evaluation_pass_probability: float = 40.0
        first_payout_probability: float = 20.0
        risk_of_ruin_pct: float = 15.0
        return_percentiles: dict = None

        def __post_init__(self):
            if self.return_percentiles is None:
                self.return_percentiles = {25: -2.0, 50: 1.0, 75: 3.0}

    verdict, reasons, scorecard, hard_fail, lookahead_hard_fail = _make_verdict(
        _WeakMC(), None, icir_gate=None, risk_of_ruin_cap=20.0,
        statistics=_fake_statistics(10), min_trades_for_ready=100,
    )
    assert verdict != "READY"
    assert not any("CAPPED AT MARGINAL" in r for r in reasons)


def test_min_trades_floor_is_configurable_via_full_pipeline_config():
    assert FullPipelineConfig().min_trades_for_ready == 100
    assert FullPipelineConfig(min_trades_for_ready=10).min_trades_for_ready == 10



# ---------------------------------------------------------------------------

def test_cpcv_as_supporting_diagnostic_does_not_replace_walk_forward(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path,
        _cfg(cpcv_supporting_enabled=True, cpcv_n_groups=4, cpcv_n_test_groups=1), progress_cb=None,
    )
    # Walk-forward still ran as primary (Step 4 always runs); CPCV ran as
    # an extra, low-weight supporting component -- never both scored into
    # the same "primary generalization" slot.
    assert result.oos_validation is not None or result.oos_validation_skip_reason is not None
    if result.cpcv_result is not None:
        assert result.scorecard.components.get("cpcv_supporting", {}).get("value") is not None


def test_cpcv_as_primary_method_feeds_the_generalization_slot(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path,
        _cfg(primary_robustness_method="cpcv", cpcv_n_groups=4, cpcv_n_test_groups=1), progress_cb=None,
    )
    if result.cpcv_result is not None:
        # CPCV filled the walk_forward_stability slot instead of the
        # ordinary walk-forward check (never both -- see _make_verdict).
        assert result.scorecard.components["walk_forward_stability"]["value"] is not None
    else:
        assert result.cpcv_skip_reason is not None


def test_regime_diagnostics_attached_and_never_affects_verdict(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    with_regime = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path / "a", _cfg(regime_diagnostics_enabled=True), progress_cb=None,
    )
    without_regime = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path / "b", _cfg(regime_diagnostics_enabled=False), progress_cb=None,
    )
    assert with_regime.regime_result is not None or with_regime.regime_skip_reason is not None
    assert without_regime.regime_result is None and without_regime.regime_skip_reason is None
    # Same strategy, same data, same everything else -- regime diagnostics
    # being on or off must never change the verdict.
    assert with_regime.verdict == without_regime.verdict


# ---------------------------------------------------------------------------
# Circularity fix -- reserve_true_holdout (see CIRCULARITY_AUDIT.md)
# ---------------------------------------------------------------------------

def test_reserve_true_holdout_keeps_steps_1_to_4_off_the_final_holdout_bars(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path, _cfg(reserve_true_holdout=True), progress_cb=None,
    )
    assert result.final_holdout is not None
    dev_cutoff = pd.Timestamp(result.final_holdout["in_sample_period"][1])
    # Every trade the final (Step 3) backtest produced must have entered
    # strictly within the dev slice -- none of Steps 1-4 ever saw a bar
    # past this cutoff.
    assert all(pd.Timestamp(t.entry_time) <= dev_cutoff for t in result.final_bt.trades)
    if result.oos_validation is not None:
        # The post-hoc walk-forward check (Step 4) is built entirely from
        # dev_df, so it can't possibly span past the same cutoff either.
        assert result.oos_validation.n_folds > 0
    # Step 5's own holdout comparison DID get the reserved tail -- its
    # holdout_bars is exactly the ~holdout_frac of the full dataset, not 0.
    assert result.final_holdout["holdout_bars"] > 0
    expected_holdout_bars = len(df) - len(df) * (1 - _cfg().holdout_frac)
    assert result.final_holdout["holdout_bars"] == pytest.approx(expected_holdout_bars, abs=2)


def test_reserve_true_holdout_false_restores_old_behavior(tmp_path):
    """Regression guard for the escape hatch -- turning the fix off must
    not crash and must still produce a valid 3-way verdict."""
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path, _cfg(reserve_true_holdout=False), progress_cb=None,
    )
    assert result.verdict in ("READY", "MARGINAL", "NOT READY")
    assert result.final_holdout is not None


def test_reserve_true_holdout_default_is_on():
    assert FullPipelineConfig().reserve_true_holdout is True


def test_parsimony_attached_to_result_and_scorecard(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path, _cfg(), progress_cb=None,
    )
    assert result.parsimony is not None
    if result.parsimony.score is not None:
        assert result.scorecard.components["parsimony"]["value"] == pytest.approx(result.parsimony.score)
