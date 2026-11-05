from types import SimpleNamespace

import numpy as np
import pandas as pd

from app.discovery_engine.trial_accounting import effective_trials_sensitivity, trial_inputs
from app.optimize.elite_reeval import reevaluate_elites
from app.optimize.ga_vs_random import BENCHMARKS, compare, make_genes


def test_ga_beats_random_at_equal_budget_on_smooth_and_multimodal():
    for name in ("sphere", "rastrigin"):
        r = compare(BENCHMARKS[name], make_genes(5), population=12, generations=6, runs=16, seed=3)
        assert r.budget == 12 * 7
        assert r.ga_wins > r.random_wins and r.sign_test_p < 0.05, (name, r.verdict)


def test_compare_is_deterministic():
    a = compare(BENCHMARKS["sphere"], make_genes(3), population=8, generations=3, runs=4, seed=1)
    b = compare(BENCHMARKS["sphere"], make_genes(3), population=8, generations=3, runs=4, seed=1)
    assert a.per_seed == b.per_seed


def test_elite_reeval_demotes_a_lucky_winner():
    e = [SimpleNamespace(genome=[1], fitness=0.90), SimpleNamespace(genome=[2], fitness=0.80)]
    score = lambda g, seed: 0.20 if g == [1] else 0.78  # noqa: E731
    best, rep = reevaluate_elites(e, score, repeats=2)
    assert best.genome == [2] and rep["swapped"] and rep["elites"][0]["luck"] > 0.4


def test_elite_reeval_keeps_winner_when_scores_hold_and_survives_scorer_errors():
    e = [SimpleNamespace(genome=[1], fitness=0.9), SimpleNamespace(genome=[2], fitness=0.5)]

    def boom(g, s):
        raise RuntimeError("x")
    best, rep = reevaluate_elites(e, boom, repeats=2)
    assert best.genome == [1] and not rep["swapped"]


def _streams(n_groups=3, per=4, days=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days)
    out = []
    for _ in range(n_groups):
        base = rng.normal(size=days)
        for _ in range(per):
            out.append(pd.Series(base + rng.normal(scale=0.2, size=days), index=idx))
    return out


def test_sensitivity_rows_and_monotone_in_threshold():
    rows = effective_trials_sensitivity(1000, _streams())
    assert [r["corr_threshold"] for r in rows] == [0.80, 0.90, 0.95]
    effs = [r["effective_trials"] for r in rows]
    assert effs == sorted(effs)  # stricter cut-off never counts fewer independent tries
    assert effective_trials_sensitivity(1000, _streams()[:2]) == []
    assert trial_inputs(100, streams=_streams()).to_dict()["sensitivity"]
