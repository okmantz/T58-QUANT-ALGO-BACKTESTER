"""B1-2 (w4-forge) tests.

Covers the four w4-forge work items from the 2026-10-04 deep analysis:
  1. Forge's final-MC sort ranks on PER-ATTEMPT pass probability, not the
     chain-level number (app/orchestration/forge.py::_final_mc_sort_key).
  2. CPCV-skip now means NOT robust, and a locked-OOS "NOT TESTED" now
     REJECTS the candidate (was: skip = pass / keep-with-flag).
  3. SearchStageConfig.min_first_payout_probability is 50.0 on the 0-100
     Monte Carlo scale (was 0.5 -- a units bug = 0.5% gate), enforced in
     Stage 3's gate and exposed on the web Search Lab form.
  4. floating_drawdown_mode threads from SearchStageConfig through to the
     PropRules every search simulator/MC call uses ("realized" default =
     byte-identical to today).
"""
from __future__ import annotations

import tempfile
from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration import forge as forge_mod
from app.orchestration.forge import (
    ForgeConfig,
    _cpcv_is_robust,
    _final_mc_sort_key,
    _locked_oos_check,
)
from app.prop.simulator import PropRules
from app.search.batch_runner import SearchStageConfig, _search_prop_rules


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _trending_df(n=300, seed=7):
    import numpy as np

    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = 0.00015 + rng.normal(0, 0.00003)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00002))
        l = min(o, c) - abs(rng.normal(0, 0.00002))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_cross_spec():
    return {
        "source_type": "manual",
        "config": {
            "name": "sma_cross",
            "indicators": [
                {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
                {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
            ],
            "long_entry": "sma_fast > sma_slow",
            "long_exit": "sma_fast < sma_slow",
            "risk_management": {
                "stop_type": "fixed", "stop_value": 20,
                "target_type": "fixed", "target_value": 40,
            },
        },
    }


# ---------------------------------------------------------------------------
# 1. per-attempt final-MC sort
# ---------------------------------------------------------------------------

def test_final_mc_sort_ranks_on_per_attempt_not_chain_level():
    """Chain-level says A wins (90 vs 20); per-attempt says B wins
    (60 vs 10). The sort must follow per-attempt."""
    recs = [
        {"candidate_id": "A",
         "final_mc": {"evaluation_pass_probability": 90.0, "per_attempt_pass_probability": 10.0}},
        {"candidate_id": "B",
         "final_mc": {"evaluation_pass_probability": 20.0, "per_attempt_pass_probability": 60.0}},
    ]
    recs.sort(key=_final_mc_sort_key, reverse=True)
    assert [r["candidate_id"] for r in recs] == ["B", "A"]


def test_final_mc_sort_key_falls_back_to_chain_level():
    """Dicts predating the per-attempt field still sort on the legacy
    chain-level field instead of crashing."""
    rec = {"final_mc": {"evaluation_pass_probability": 42.0}}
    assert _final_mc_sort_key(rec) == 42.0


# ---------------------------------------------------------------------------
# 2a. CPCV skip => not robust
# ---------------------------------------------------------------------------

def test_cpcv_skip_is_not_robust():
    assert _cpcv_is_robust(None) is False


def test_cpcv_result_robustness_passes_through():
    assert _cpcv_is_robust(SimpleNamespace(is_robust=True)) is True
    assert _cpcv_is_robust(SimpleNamespace(is_robust=False)) is False


# ---------------------------------------------------------------------------
# 2b. locked-OOS "NOT TESTED" => rejected
# ---------------------------------------------------------------------------

def _fake_bt(n_trades):
    return SimpleNamespace(
        trades=[SimpleNamespace(pnl=1.0)] * n_trades,
        statistics=SimpleNamespace(net_profit=10.0, to_dict=lambda: {"net_profit": 10.0}),
    )


def _check_kwargs(tmp_path, **overrides):
    kw = dict(
        rec={"candidate_id": "c1", "family": "testfam", "config": {}},
        locked_df=_trending_df(n=50),
        risk=RiskConfig(),
        prop_rules=PropRules(),
        config=ForgeConfig(locked_oos_min_pass_rate=25.0),
        family_scores={},
        diagnoses=[],
        graveyard_path=tmp_path / "g.jsonl",
    )
    kw.update(overrides)
    return kw


def test_locked_oos_not_tested_rejects_candidate(tmp_path, monkeypatch):
    """Too few holdout trades => "NOT TESTED" => rejected: the caller must
    NOT keep the candidate, and a diagnosis + graveyard entry are recorded."""
    monkeypatch.setattr(forge_mod, "build_strategy_from_spec", lambda spec: object())
    monkeypatch.setattr(forge_mod, "run_backtest", lambda df, strategy, risk: _fake_bt(2))

    kw = _check_kwargs(tmp_path)
    status = _locked_oos_check(**kw)

    assert status == "NOT TESTED"
    assert kw["rec"]["locked_oos"] == {"status": "NOT TESTED", "pass_rate_pct": None}
    # caller-side contract: only "PASSED" is kept
    survivors = {}
    if status == "PASSED":
        survivors[kw["rec"]["candidate_id"]] = kw["rec"]
    assert survivors == {}
    # rejection is diagnosed + graveyard-logged, not silent
    assert len(kw["diagnoses"]) == 1
    assert "NOT TESTED" in kw["diagnoses"][0].verdict or "not testable" in kw["diagnoses"][0].verdict
    grave_rows = (tmp_path / "g.jsonl").read_text().strip().splitlines()
    assert len(grave_rows) == 1
    assert '"oos_result": "not_tested"' in grave_rows[0]


def test_locked_oos_no_holdout_backtest_rejects_candidate(tmp_path, monkeypatch):
    """holdout_bt None (empty locked df slice) is also NOT TESTED => rejected."""
    monkeypatch.setattr(forge_mod, "build_strategy_from_spec", lambda spec: object())
    monkeypatch.setattr(forge_mod, "run_backtest", lambda df, strategy, risk: None)

    kw = _check_kwargs(tmp_path)
    assert _locked_oos_check(**kw) == "NOT TESTED"
    assert kw["rec"]["locked_oos"]["status"] == "NOT TESTED"


def test_locked_oos_passed_still_keeps_candidate(tmp_path, monkeypatch):
    """Sanity: the PASSED path is unchanged -- high holdout pass rate keeps."""
    monkeypatch.setattr(forge_mod, "build_strategy_from_spec", lambda spec: object())
    monkeypatch.setattr(forge_mod, "run_backtest", lambda df, strategy, risk: _fake_bt(10))
    monkeypatch.setattr(
        forge_mod, "run_rolling_evaluation",
        lambda trades, rules, **k: SimpleNamespace(pass_rate_pct=90.0),
    )

    kw = _check_kwargs(tmp_path)
    status = _locked_oos_check(**kw)
    assert status == "PASSED"
    assert kw["rec"]["locked_oos"] == {"status": "PASSED", "pass_rate_pct": 90.0}
    assert kw["diagnoses"] == []


def test_locked_oos_failed_still_rejects_candidate(tmp_path, monkeypatch):
    """Sanity: the FAILED path is unchanged -- low holdout pass rate rejects."""
    monkeypatch.setattr(forge_mod, "build_strategy_from_spec", lambda spec: object())
    monkeypatch.setattr(forge_mod, "run_backtest", lambda df, strategy, risk: _fake_bt(10))
    monkeypatch.setattr(
        forge_mod, "run_rolling_evaluation",
        lambda trades, rules, **k: SimpleNamespace(pass_rate_pct=5.0),
    )

    kw = _check_kwargs(tmp_path)
    status = _locked_oos_check(**kw)
    assert status == "FAILED"
    assert kw["rec"]["locked_oos"]["status"] == "FAILED"


# ---------------------------------------------------------------------------
# 3. payout floor 50.0 on the 0-100 MC scale
# ---------------------------------------------------------------------------

def test_payout_floor_default_is_50_on_0_100_scale():
    assert SearchStageConfig().min_first_payout_probability == 50.0
    assert SearchStageConfig().min_eval_pass_probability == 70.0


def _stage3_with_fake_mc(monkeypatch):
    """Run _stage3_task against a real backtest but a stubbed Monte Carlo
    with per-attempt pass 80% / payout 10%: above the eval floor, below the
    payout floor. Returns (result_default_floor, result_floor_zero)."""
    from app.search import batch_runner
    from app.search.batch_runner import _init_worker, _stage3_task

    fake_mc = SimpleNamespace(
        evaluation_pass_probability=80.0,
        first_payout_probability=10.0,
        per_attempt_pass_probability=80.0,
        per_attempt_payout_probability=10.0,
        expected_payout=50.0,
        risk_of_ruin_pct=5.0,
        median_drawdown_pct=3.0,
        n_simulations=20,
    )
    monkeypatch.setattr(batch_runner, "run_monte_carlo", lambda trades, rules, cfg: fake_mc)

    base_cfg = {
        "full_mc_sims": 20, "random_seed": 1, "fitness_metric": "eval_pass_probability",
        "walk_forward_folds": 0, "robustness_neighbors": 0,
        "stage3_min_trades": 3, "stage3_min_profit_factor": 0.5,
        "stage3_max_drawdown_buffer_mult": 2.0, "stage3_require_positive_net": False,
        "reset_on_breach": False, "min_eval_pass_probability": 70.0,
    }
    with tempfile.TemporaryDirectory() as td:
        df = _trending_df(n=300)
        df_path = f"{td}/data.pkl"
        df.to_pickle(df_path)
        _init_worker(df_path, {}, {}, td)
        spec = _sma_cross_spec()
        # default floor (key absent -> cfg.get default 50.0)
        r_default = _stage3_task("floor-default", spec, dict(base_cfg))
        # explicit 50.0 floor
        r_50 = _stage3_task("floor-50", spec, {**base_cfg, "min_first_payout_probability": 50.0})
        # floor disabled -> control, must pass
        r_zero = _stage3_task("floor-zero", spec, {**base_cfg, "min_first_payout_probability": 0.0})
    return r_default, r_50, r_zero


def test_payout_floor_50_rejects_low_payout_candidate(monkeypatch):
    r_default, r_50, r_zero = _stage3_with_fake_mc(monkeypatch)
    # 10% per-attempt payout < 50% floor => rejected on level, not stability
    assert r_default["passed_stage3_gate"] is False
    assert r_50["passed_stage3_gate"] is False
    assert "first-payout probability 10.0% below acceptance floor 50.0%" in r_default["gate_notes"]


def test_payout_floor_zero_control_passes(monkeypatch):
    _, _, r_zero = _stage3_with_fake_mc(monkeypatch)
    # Same candidate, floor disabled => the payout leg no longer rejects
    assert r_zero["passed_stage3_gate"] is True


# ---------------------------------------------------------------------------
# 4. floating_drawdown_mode threading
# ---------------------------------------------------------------------------

def test_floating_drawdown_mode_defaults_to_realized():
    assert SearchStageConfig().floating_drawdown_mode == "realized"
    assert PropRules().floating_drawdown_mode == "realized"


def test_floating_drawdown_mode_invalid_value_raises():
    with pytest.raises(ValueError, match="floating_drawdown_mode"):
        SearchStageConfig(floating_drawdown_mode="bogus")


def test_floating_drawdown_mode_adverse_reaches_prop_rules():
    """The adverse value is stamped onto the PropRules the search's
    simulate_account / run_monte_carlo calls consume (in-process and, via
    asdict, the worker processes)."""
    stamped = _search_prop_rules(PropRules(), SearchStageConfig(floating_drawdown_mode="adverse"))
    assert stamped.floating_drawdown_mode == "adverse"
    assert asdict(stamped)["floating_drawdown_mode"] == "adverse"


def test_floating_drawdown_mode_default_is_byte_identical():
    """Default config leaves PropRules untouched in value -- identical to
    today's behavior."""
    stamped = _search_prop_rules(PropRules(), SearchStageConfig())
    assert stamped.floating_drawdown_mode == "realized"
    assert stamped == PropRules()
