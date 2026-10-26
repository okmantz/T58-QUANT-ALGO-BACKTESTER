"""The promotion sequence: order, short-circuit, tied-family handling, crashes count as failures."""
from __future__ import annotations

import numpy as np
import pytest

from app.discovery_engine import promotion as pr
from app.discovery_engine.engine_steps import build_steps


def _ok(name):
    return lambda c, ctx: pr.StepResult(name, True, detail="ok")


def _bad(name):
    return lambda c, ctx: pr.StepResult(name, False, detail="nope")


def _all_ok():
    return pr.Steps(**{n: _ok(n) for n in pr.STEP_ORDER})


def test_all_pass():
    r = pr.run_promotion_sequence({"spec": {"config": {"x": 1}}}, _all_ok())
    assert r.passed and [s.name for s in r.steps] == list(pr.STEP_ORDER) and r.forward_test_required


def test_stops_at_first_failure_and_skips_rest():
    s = _all_ok()
    s.null_gate = _bad("null_gate")
    r = pr.run_promotion_sequence({"spec": {}}, s)
    assert not r.passed and r.failed_step == "null_gate"
    assert r.step("pooled_edge").skipped and r.step("sealed_slice").skipped and not r.step("reproduction").skipped
    with pytest.raises(pr.PromotionFailed):
        r.raise_for_failure()


def test_crashing_or_missing_step_is_a_failure():
    s = _all_ok()
    s.sequence_deflated = lambda c, ctx: 1 / 0
    r = pr.run_promotion_sequence({"spec": {}}, s)
    assert r.failed_step == "sequence_deflated" and "crashed" in r.failed_reason
    s2 = _all_ok()
    s2.sealed_slice = None
    assert pr.run_promotion_sequence({"spec": {}}, s2).failed_step == "sealed_slice"


def test_instrument_tied_family_skips_pooling_but_stays_mandatory_forward():
    cand = {"spec": {"config": {"entry_conditions": {"long": [{"left": "gex_flip_distance"}]}}}}
    s = _all_ok()
    s.pooled_edge = None                    # not needed for a tied family
    seen = {}
    s.sealed_slice = lambda c, ctx: seen.update(tied=ctx["tied"]) or pr.StepResult("sealed_slice", True)
    r = pr.run_promotion_sequence(cand, s)
    assert r.passed and not r.poolable and r.forward_test_required and seen["tied"] and r.step("pooled_edge").skipped


def test_tied_detection_by_marker_and_by_family_name():
    assert pr.is_instrument_tied({"a": "minutes_since_high_impact_news"})
    assert pr.is_instrument_tied({}, "gex_flip")
    assert not pr.is_instrument_tied({"config": {"entry": "rsi"}}, "mean_reversion")


def test_loop_stop_finalizer_raises_unless_all_steps_pass():
    s = _all_ok()
    assert pr.finalize_loop_stop({"spec": {}}, s).passed
    s.sealed_slice = _bad("sealed_slice")
    with pytest.raises(pr.PromotionFailed):
        pr.finalize_loop_stop({"spec": {}}, s)


def test_deflated_step_rejects_noise_and_accepts_strong_edge():
    rng = np.random.default_rng(3)
    pool = rng.normal(0, 1, (900, 300))
    noise_best = pool[:, int(np.argmax(pool.mean(axis=0)))]
    assert not pr.deflated_step(noise_best, pool, 5000).passed
    strong = rng.normal(0.25, 1, 900)
    assert pr.deflated_step(strong, pool, 50).passed
    assert not pr.deflated_step(strong[:10], pool, 50).passed


def test_pooled_step_requires_enough_markets():
    cfg = pr.PromotionConfig()
    rng = np.random.default_rng(1)
    per = {f"M{i}": (list(rng.normal(0.3, 1, 60)), list(range(60))) for i in range(3)}
    r = pr.pooled_step(per, cfg)
    assert not r.passed and "market" in r.detail
    per = {f"M{i}": (list(rng.normal(0.3, 1, 60)), [d + i * 5 for d in range(60)]) for i in range(8)}
    assert pr.pooled_step(per, cfg).passed


def test_sealed_step_is_stricter_for_tied_families():
    cfg = pr.PromotionConfig()
    rng = np.random.default_rng(4)
    rs = list(rng.normal(0.12, 1.0, 40))
    days = list(range(40))
    assert not pr.sealed_step(rs, days, tied=True, cfg=cfg).passed          # <50 trades is never enough for a tied family
    strong = list(rng.normal(0.6, 1.0, 120))
    assert pr.sealed_step(strong, list(range(120)), tied=False, cfg=cfg).passed


def test_real_steps_fail_closed_without_provenance_or_sealed_slice():
    import pandas as pd
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=50, freq="h"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    st = build_steps(dev_df=df, sealed_df=None, risk=None)
    r = pr.run_promotion_sequence({"spec": {"config": {}}}, st)
    assert not r.passed and r.failed_step == "reproduction"
