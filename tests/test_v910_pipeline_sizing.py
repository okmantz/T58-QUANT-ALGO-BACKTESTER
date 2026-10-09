"""v9.10 regression tests: Full Pipeline must COMPLETE for stop-less
strategies instead of BLOCKing at preflight.

Root cause (Owen's 2026-10-09 screenshots): a manual strategy defining
no stop got the engine's silent 1%-of-price placeholder (~78 pts on ES
= ~$3,900 on one mini), fit_stop refused to shrink below 20% of that
invented distance, and >20% of signals skipped -> preflight BLOCK.

Fix under test:
  * apply_instrument_spec populates micro_contract_size /
    micro_commission_per_contract (micro_fallback was dead code before).
  * Full Pipeline replaces the placeholder with a visible 2 x ATR(14)
    stop default for stop-less manual strategies (logged, recorded).
  * At the preflight gate, sizing remedies are tried in order --
    micro fallback (same stop), then fit_stop at the 10% hard floor --
    and only a run that still cannot afford one micro BLOCKs.
"""
import shutil

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig, build_run_context
from app.data.instrument_specs import apply_instrument_spec
from app.optimize.parameter_space import RefinementError
from app.orchestration.full_pipeline import FullPipelineConfig, run_full_pipeline
from app.prop.presets import get_preset
from app.strategy import library
from app.strategy.manual import ManualStrategy
from app.validation.preflight import (
    FIT_STOP_HARD_FLOOR,
    apply_no_stop_default,
    sizing_remedy_candidates,
    strategy_defines_stop,
)


@pytest.fixture(autouse=True)
def clean_library_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    base_dir = library.get_strategy_library_dir()
    yield
    shutil.rmtree(base_dir, ignore_errors=True)


STOPLESS_SMA = {
    "name": "5m SMA (OPTIMIZED)",
    "indicators": [
        {"type": "sma", "period": 15, "column": "close", "as": "sma_fast"},
        {"type": "sma", "period": 21, "column": "close", "as": "sma_slow"},
    ],
    "long_entry": "sma_fast > sma_slow",
    "long_exit": "sma_fast < sma_slow",
    "short_entry": "sma_fast < sma_slow",
    "short_exit": "sma_fast > sma_slow",
}


def _es_df(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    steps, level, t = [], 7800.0, 0
    while t < n:
        seg = rng.integers(300, 900)
        drift = rng.normal(0.004, 0.01)
        vol = rng.uniform(0.9, 2.2)
        for _ in range(min(seg, n - t)):
            level = max(6000.0, level + drift * level / 100.0 + rng.normal(0, vol))
            steps.append(level)
            t += 1
    close = np.array(steps[:n])
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    spread = np.abs(rng.normal(0.9, 0.5, n))
    ts = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamp": ts, "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close, "volume": 1000.0,
    })


def _lucid_risk(**over):
    kw = dict(initial_balance=50000.0, risk_mode="percent", risk_value=1.0,
              max_trades_per_day=10, sizing_mode="fit_stop")
    kw.update(over)
    rules = get_preset("lucid_50k").to_prop_rules()
    return build_run_context(RiskConfig(**kw), prop_rules=rules, instrument="ES1!"), rules


def _cfg(**over):
    base = dict(
        # conftest sets T58_PREFLIGHT_ENFORCE=0 suite-wide; the gate is
        # the subject under test here, so opt back in explicitly.
        preflight_enforce=True,
        skip_optimization=True, baseline_mc_sims=50, final_mc_sims=200,
        n_folds=3, oos_check_folds=3, attempt_replay_starts=4,
        null_n_seeds=10, seed_rescore_seeds=2, seed_rescore_sims=100,
        preflight_min_trades=10,
    )
    base.update(over)
    return FullPipelineConfig(**base)


# ---------------------------------------------------------------- units

def test_apply_instrument_spec_populates_micro_fields():
    risk = apply_instrument_spec(RiskConfig(), "ES")
    assert risk.micro_contract_size == 5.0          # MES is $5/point
    assert risk.micro_commission_per_contract == pytest.approx(1.42)
    # an explicit caller value is never overwritten
    risk2 = apply_instrument_spec(RiskConfig(micro_contract_size=9.0), "ES")
    assert risk2.micro_contract_size == 9.0


def test_strategy_defines_stop_detection():
    assert strategy_defines_stop(ManualStrategy(STOPLESS_SMA)) is False
    with_rm = dict(STOPLESS_SMA, risk_management={"stop_type": "atr", "stop_value": 2.0})
    assert strategy_defines_stop(ManualStrategy(with_rm)) is True
    with_pips = dict(STOPLESS_SMA, stop_loss_pips=25)
    assert strategy_defines_stop(ManualStrategy(with_pips)) is True
    with_zone = dict(STOPLESS_SMA, zone_entry={"type": "retrace"})
    assert strategy_defines_stop(ManualStrategy(with_zone)) is True


def test_no_stop_default_is_visible_atr_and_non_mutating():
    df = _es_df()
    strategy = ManualStrategy(STOPLESS_SMA)
    new_strategy, note = apply_no_stop_default(strategy, df)
    assert note is not None and note["rule"] == "atr" and note["multiple"] == 2.0
    assert note["median_stop_points"] is not None and 2.0 < note["median_stop_points"] < 30.0
    # the placeholder it replaces is an order of magnitude wider
    assert note["placeholder_stop_points"] > 3 * note["median_stop_points"]
    assert strategy_defines_stop(new_strategy) is True
    assert "risk_management" not in strategy.config  # original untouched


def test_remedy_candidates_order_micro_then_floor():
    risk, _ = _lucid_risk()
    kinds = [k for k, _, _ in sizing_remedy_candidates(risk, "ES1!")]
    assert kinds == ["micro_fallback", "fit_stop_floor"]
    _, detail, cand = sizing_remedy_candidates(risk, "ES1!")[0]
    assert detail["micro_symbol"] == "MES" and cand.micro_contract_size == 5.0
    _, detail2, cand2 = sizing_remedy_candidates(risk, "ES1!")[1]
    assert detail2["fit_stop_min_fraction"] == FIT_STOP_HARD_FLOOR == 0.10
    assert cand2.fit_stop_min_fraction == 0.10


# ------------------------------------------------------- end-to-end

def test_full_pipeline_stopless_completes_with_stop_default(tmp_path):
    risk, rules = _lucid_risk()
    logs = []
    result = run_full_pipeline(_es_df(), ManualStrategy(STOPLESS_SMA), risk, rules,
                               tmp_path, _cfg(), progress_cb=logs.append, instrument="ES1!")
    assert result.verdict in ("READY", "MARGINAL", "NOT READY")
    assert len(result.baseline_bt.trades) > 0
    assert result.stop_default_applied is not None
    assert result.stop_default_applied["rule"] == "atr"
    assert any("STOP DEFAULT" in line for line in logs)
    halt = result.baseline_bt.equity_curve.attrs.get("sizing_halt", {})
    assert halt.get("skip_ratio", 1.0) <= 0.20


def test_full_pipeline_rr_model_sizes_micro_when_mini_exceeds_planned(tmp_path):
    # $50 budget (0.1% of $50k): one ES mini at the ATR stop (~$320)
    # cannot fit, one MES micro (~$32) can -- Owen's model sizes the
    # micro natively in the baseline, no gate remedy needed.
    risk, rules = _lucid_risk(risk_value=0.1)
    logs = []
    result = run_full_pipeline(_es_df(), ManualStrategy(STOPLESS_SMA), risk, rules,
                               tmp_path, _cfg(), progress_cb=logs.append, instrument="ES1!")
    assert result.verdict in ("READY", "MARGINAL", "NOT READY")
    assert result.sizing_adjustment is None
    assert any("RISK MODEL" in line for line in logs)
    ss = result.baseline_bt.equity_curve.attrs.get("sizing_summary", {})
    assert ss.get("micro_fallback", 0) >= 1


# ------------------------------------------------- Owen's risk model

def _es_risk(**over):
    kw = dict(initial_balance=50000.0, risk_mode="percent", risk_value=1.0,
              sizing_mode="rr_planned", planned_target_dollars=300.0)
    kw.update(over)
    return apply_instrument_spec(RiskConfig(**kw), "ES")


def test_rr_planned_scales_risk_with_setup():
    # Owen's example: $300 target at 1:2 RR plans $150 of risk, not the
    # $500 cap. Stop 10 pts / target 20 pts on ES: one mini risks ~$529
    # (over planned); micros ($5/pt) fit 2 contracts (~$108), not 3.
    dec = _es_risk().size_for_stop(50000.0, 10.0, tp_distance=20.0)
    assert dec.used_micro and dec.contracts == 2
    assert dec.risk_at_stop <= 150.0
    assert dec.stop_distance == 10.0  # the stop itself is never moved


def test_rr_planned_no_target_risks_up_to_cap_on_mini():
    dec = _es_risk().size_for_stop(50000.0, 6.4)  # ~$320 on one mini
    assert dec.contracts == 1 and not dec.used_micro
    assert dec.stop_distance == 6.4


def test_rr_planned_cap_anchored_to_account_not_remaining_equity():
    # Equity down to $20k mid-attempt: the cap is still 1% of the $50k
    # account ($500), so the same trade still sizes one mini.
    dec = _es_risk().size_for_stop(20000.0, 6.4)
    assert dec.contracts == 1 and not dec.used_micro


def test_rr_planned_skips_when_even_one_micro_exceeds_planned():
    dec = _es_risk(risk_value=0.02).size_for_stop(50000.0, 10.0, tp_distance=20.0)
    assert dec.units == 0 and dec.skip_reason == "stop_exceeds_planned_risk"


def test_remedy_candidates_empty_for_rr_planned():
    assert sizing_remedy_candidates(_es_risk(), "ES1!") == []


def test_full_pipeline_still_blocks_when_even_micro_too_wide(tmp_path):
    # $10 budget, already in micro mode: even one MES at the stop busts
    # it, and no remedy remains -- the honest BLOCK must stand.
    risk, rules = _lucid_risk(risk_value=0.02, sizing_mode="micro_fallback")
    with pytest.raises(RefinementError, match="stopped at preflight"):
        run_full_pipeline(_es_df(), ManualStrategy(STOPLESS_SMA), risk, rules,
                          tmp_path, _cfg(), progress_cb=None, instrument="ES1!")


def test_explicit_stop_strategy_is_untouched(tmp_path):
    cfg = dict(STOPLESS_SMA, risk_management={"stop_type": "atr", "stop_value": 1.5,
                                              "stop_atr_period": 14})
    risk, rules = _lucid_risk()
    logs = []
    result = run_full_pipeline(_es_df(), ManualStrategy(cfg), risk, rules,
                               tmp_path, _cfg(), progress_cb=logs.append, instrument="ES1!")
    assert result.verdict in ("READY", "MARGINAL", "NOT READY")
    assert result.stop_default_applied is None
    assert not any("STOP DEFAULT" in line for line in logs)
