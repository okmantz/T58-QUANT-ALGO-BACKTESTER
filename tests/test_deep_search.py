"""Deep search (item 12): PropFit scoring, the two-stage ranking (a
curve-fit must lose to a robust variant under verification), and the
honest no-winner path. The scoring core is pure; the runner is driven
with stub evaluators so the stage logic is tested without an engine.
"""
from __future__ import annotations

import pytest

from app.optimize import deep_search as ds


# ------------------------------------------------------------ PropFit

def test_propfit_known_metrics_known_score():
    score, comps = ds.propfit_from_metrics(
        eval_pass=0.80, payout=0.40, profit_factor=1.5, n_trades=100,
        expectancy=25.0, max_dd_pct=1.0, dd_limit_pct=4.0, stressed_net=500.0,
    )
    # 45*.8 + 30*.4 + 15*(1.5/3) + 10*1.0 = 36 + 12 + 7.5 + 10
    assert score == pytest.approx(65.5)
    assert comps["sufficiency"] == 1.0
    assert comps["penalties"] == 0.0


def test_propfit_pf_capped_and_sufficiency_scaled():
    score, _ = ds.propfit_from_metrics(
        eval_pass=0.0, payout=0.0, profit_factor=9.0, n_trades=50,
        expectancy=1.0, max_dd_pct=1.0, dd_limit_pct=4.0, stressed_net=10.0,
    )
    # PF capped at 3 -> 15 * 3/3 = 15; sufficiency 50/100 -> 5
    assert score == pytest.approx(20.0)


def test_propfit_negative_expectancy_floors_to_zero():
    score, comps = ds.propfit_from_metrics(
        eval_pass=0.9, payout=0.9, profit_factor=2.0, n_trades=200,
        expectancy=-1.0, max_dd_pct=1.0, dd_limit_pct=4.0, stressed_net=100.0,
    )
    assert score == 0.0
    assert comps["floored"] is True


def test_propfit_penalties():
    score, comps = ds.propfit_from_metrics(
        eval_pass=0.80, payout=0.40, profit_factor=1.5, n_trades=100,
        expectancy=25.0, max_dd_pct=3.7, dd_limit_pct=4.0, stressed_net=-5.0,
    )
    # 65.5 - 10 (DD within 10% of the 4.0 limit) - 30 (stress underwater)
    assert score == pytest.approx(25.5)
    assert comps["penalties"] == 40.0


# ------------------------------------------------------- verified score

def test_verified_score_evaporates_below_cpcv_1():
    assert ds.verified_score(90.0, cpcv_oos_pf=0.8, stress_net=100.0,
                             seed_std_pts=1.0) == 0.0
    assert ds.verified_score(90.0, cpcv_oos_pf=None, stress_net=100.0,
                             seed_std_pts=1.0) == 0.0


def test_verified_score_dispersion_and_stress_drag():
    assert ds.verified_score(90.0, cpcv_oos_pf=1.4, stress_net=100.0,
                             seed_std_pts=3.0) == pytest.approx(87.0)
    assert ds.verified_score(90.0, cpcv_oos_pf=1.4, stress_net=-1.0,
                             seed_std_pts=3.0) == pytest.approx(57.0)


# ------------------------------------------------------------ two-stage

def _row(label, propfit, dataset="d.csv", timeframe="5m"):
    return {"variant": {"label": label}, "label": label, "dataset": dataset,
            "timeframe": timeframe, "n_trades": 150, "eval_pass": 0.8,
            "payout": 0.4, "profit_factor": 1.5, "propfit": propfit}


def test_curve_fit_loses_to_robust_under_verification():
    variants = [{"label": "curvefit"}, {"label": "robust"}, {"label": "weak"}]
    screen = {"curvefit": 90.0, "robust": 70.0, "weak": 40.0}

    def evaluate(v, phase):
        return _row(v["label"], screen[v["label"]])

    verify_map = {
        "curvefit": {"cpcv_oos_pf": 0.8, "stress_net": 100.0, "seed_std_pts": 1.0, "eval_pass_verified": 0.5},
        "robust": {"cpcv_oos_pf": 1.4, "stress_net": 100.0, "seed_std_pts": 2.0, "eval_pass_verified": 0.5},
        "weak": {"cpcv_oos_pf": 1.2, "stress_net": 100.0, "seed_std_pts": 1.0, "eval_pass_verified": 0.5},
    }
    out = ds.run_search(variants, evaluate, lambda r: verify_map[r["label"]])
    assert out["winner"]["label"] == "robust"
    assert out["winner"]["verified"] == pytest.approx(68.0)
    curvefit = [r for r in out["rows"] if r["label"] == "curvefit"][0]
    assert curvefit["verified"] == 0.0  # edge evaporated; shown, not crowned


def test_zero_verified_eval_pass_is_not_above_water():
    """A variant no verified attempt ever passes with must not be
    crowned, however good its screen/CPCV/stress look (found live: the
    screen's single MC seed said 67%, five verify seeds said 0%)."""
    row = {"verified": 40.0, "cpcv_oos_pf": 1.3, "stress_net": 100.0,
           "eval_pass_verified": 0.0}
    assert ds.is_above_water(row) is False
    row["eval_pass_verified"] = 0.25
    assert ds.is_above_water(row) is True


def test_no_winner_path_is_honest():
    variants = [{"label": "a"}, {"label": "b"}]

    def evaluate(v, phase):
        return _row(v["label"], 80.0)

    out = ds.run_search(
        variants, evaluate,
        lambda r: {"cpcv_oos_pf": 0.5, "stress_net": -1.0, "seed_std_pts": 1.0},
    )
    assert out["winner"] is None
    assert "No robust winner found in 2 variants" in out["message"]
    assert len(out["rows"]) == 2  # least-bad still shown


def test_runner_prunes_then_ranks_on_full_data():
    variants = [{"label": f"v{i}"} for i in range(50)]
    seen_phases = []

    def evaluate(v, phase):
        seen_phases.append(phase)
        return _row(v["label"], float(v["label"][1:]))

    out = ds.run_search(variants, evaluate, None, prune_above=10, finalist_count=5)
    assert "slice" in seen_phases and "full" in seen_phases
    assert out["rows"][0]["label"] == "v49"
    assert out["n_variants"] == 50


def test_sample_genomes_base_first_and_in_bounds():
    class G:
        def __init__(self, lo, hi, base, is_int=False):
            self.lo, self.hi, self.base_value, self.is_int = lo, hi, base, is_int

    genes = [G(5, 40, 20, is_int=True), G(0.5, 3.0, 1.0)]
    genomes = ds.sample_genomes(genes, 25, seed=1)
    assert len(genomes) == 25
    assert genomes[0] == [20.0, 1.0]
    for g in genomes:
        assert 5 <= g[0] <= 40 and g[0] == int(g[0])
        assert 0.5 <= g[1] <= 3.0
