"""The power-aware sequential forward test that replaced the unreachable "95% lower bound >= 65%" forward gate."""
import numpy as np
import pytest

from app.validation.prop30_validation import ForwardRegistry
from app.validation.sequential_forward import (ACCEPT_TARGET, CONTINUE, INCONCLUSIVE, REJECT_TARGET, SequentialDesign, decide, design_summary,
                                               independent_attempts, operating_characteristic, posterior_prob_at_least)


def test_the_old_gate_was_unreachable_and_the_new_design_is_powered():
    """Old gate: at n=12 it needs 12/12 passes, so a true-80% strategy cleared it ~7% of the time. The new design confirms it ~99% of the time."""
    from app.prop.thirty_day import wilson_interval
    assert wilson_interval(11 / 12, 12)[0] < 0.65 <= wilson_interval(12 / 12, 12)[0]          # only 12/12 reaches the 65% lower bound
    assert 0.8 ** 12 < 0.07                                                                      # P(12/12 | true p = 0.8) ~ 6.9%
    oc = operating_characteristic(SequentialDesign(), 0.8)
    assert oc["p_confirm"] > 0.95


def test_error_rates_are_fixed_in_advance():
    d = SequentialDesign()
    s = design_summary(d)
    assert s["false_confirmation_at_p0"] <= d.alpha + 1e-9                                        # P(confirm | p = p0) <= alpha
    assert s["power_at_p1"] >= 1 - d.beta - 0.05 and s["meets_error_budget"]
    assert operating_characteristic(d, 0.30)["p_confirm"] < 0.01


def test_operating_characteristic_is_a_probability_distribution():
    for p in (0.1, 0.4, 0.65, 0.9):
        oc = operating_characteristic(SequentialDesign(), p)
        assert oc["p_confirm"] + oc["p_reject"] + oc["p_inconclusive"] == pytest.approx(1.0, abs=1e-9)
        assert 1 <= oc["expected_attempts"] <= SequentialDesign().n_max


def test_operating_characteristic_matches_simulation():
    d, rng = SequentialDesign(), np.random.default_rng(11)
    wins = sum(decide(d, rng.random(d.n_max) < 0.65)["decision"] == ACCEPT_TARGET for _ in range(4000)) / 4000
    assert wins == pytest.approx(operating_characteristic(d, 0.65)["p_confirm"], abs=0.03)


def test_first_boundary_crossing_decides_and_later_results_cannot_overturn_it():
    d = SequentialDesign()
    run = [True] * 9
    assert decide(d, run)["decision"] == ACCEPT_TARGET
    n_stop = decide(d, run)["stopped_at"]
    assert decide(d, run[:n_stop] + [False] * 20)["decision"] == ACCEPT_TARGET
    assert decide(d, [False] * 5)["decision"] == REJECT_TARGET
    assert decide(d, [True, False])["decision"] == CONTINUE
    assert decide(SequentialDesign(n_max=4), [True, False, True, False])["decision"] == INCONCLUSIVE     # budget spent: NOT a confirmation


def test_design_validation():
    for bad in (dict(p0=0.7, p1=0.65), dict(alpha=0.6), dict(n_max=0)):
        with pytest.raises(ValueError):
            SequentialDesign(**bad)


def test_posterior_is_information_only():
    assert posterior_prob_at_least(12, 12, 0.65) > 0.9 and posterior_prob_at_least(2, 12, 0.65) < 0.01


def test_overlapping_attempts_are_not_independent():
    rows = [{"start": "2026-01-02"}, {"start": "2026-01-10"}, {"start": "2026-02-05"}, {"start": "2026-02-20"}]
    kept, n_excl = independent_attempts(rows, 30)
    assert [r["start"] for r in kept] == ["2026-01-02", "2026-02-05"] and n_excl == 2


def test_registry_preregisters_the_design_and_refuses_to_change_it(tmp_path):
    f = ForwardRegistry(tmp_path / "f.json")
    d = f.register_design("c", SequentialDesign(n_max=30))
    assert f.register_design("c", SequentialDesign(n_max=30)) == d                                # idempotent
    with pytest.raises(ValueError):
        f.register_design("c", SequentialDesign(n_max=60))                                        # cannot be moved after results arrive
    assert f.design("c") == d and f.design("other") is None


def test_registry_summary_confirms_after_enough_independent_passes_and_ignores_overlap(tmp_path):
    f = ForwardRegistry(tmp_path / "f.json")
    f.register_design("c")
    day = np.datetime64("2026-01-02")
    for i in range(10):
        f.record("c", str(day + np.timedelta64(31 * i, "D")), "passed", 12.0)
    f.record("c", str(day + np.timedelta64(31 * 3 + 5, "D")), "failed", None)                      # overlaps attempt #3: excluded
    s = f.summary("c")
    assert s["sequential"]["decision"] == ACCEPT_TARGET and s["sequential"]["n_excluded_overlap"] == 1 and s["n_resolved"] == 11


def test_a_slow_pass_is_a_miss_for_the_30_day_objective(tmp_path):
    f = ForwardRegistry(tmp_path / "f.json")
    day = np.datetime64("2026-01-02")
    for i in range(6):
        f.record("c", str(day + np.timedelta64(31 * i, "D")), "passed", 45.0)
    assert f.summary("c")["sequential"]["decision"] == REJECT_TARGET and f.summary("c")["sequential"]["passes"] == 0


def test_legacy_registry_rows_without_a_kind_are_still_attempts(tmp_path):
    import json
    p = tmp_path / "f.json"
    p.write_text(json.dumps([{"config_id": "c", "start": "2026-01-02", "outcome": "passed", "days": 10.0}]))
    assert ForwardRegistry(p).summary("c")["n_resolved"] == 1
