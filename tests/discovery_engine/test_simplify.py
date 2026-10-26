from __future__ import annotations

from app.discovery_engine import simplify as sm


def _spec(n_long=6):
    return {"source_type": "manual", "config": {
        "entry_conditions": {"long": [{"id": i} for i in range(n_long)], "short": [{"id": 100}]},
        "exit_conditions": {"long": [{"id": 50}, {"id": 51}, {"id": 52}], "short": []}}}


def test_cap_drops_latest_and_does_not_mutate():
    s = _spec()
    out, removed = sm.cap_structure(s, sm.StructureCap(4, 2))
    assert removed == 3 and len(out["config"]["entry_conditions"]["long"]) == 4 and len(out["config"]["exit_conditions"]["long"]) == 2
    assert len(s["config"]["entry_conditions"]["long"]) == 6 and not sm.within_cap(s) and sm.within_cap(out)


def test_prune_removes_noise_conditions_but_keeps_the_one_that_matters():
    spec = _spec(4)
    def score(sp):                         # only condition id 0 carries edge; every other condition costs a little
        ids = [c["id"] for c in sp["config"]["entry_conditions"]["long"]]
        return (0.10 if 0 in ids else -0.05) - 0.001 * len(ids)
    r = sm.prune(spec, score, tolerance=0.002)
    assert [c["id"] for c in r.spec["config"]["entry_conditions"]["long"]] == [0]
    assert len(r.removed) >= 3 and r.score_after >= r.score_before
    assert len(spec["config"]["entry_conditions"]["long"]) == 4


def test_prune_respects_minimum_and_unmeasurable():
    spec = _spec(1)
    r = sm.prune(spec, lambda sp: 0.1)
    assert len(r.spec["config"]["entry_conditions"]["long"]) == 1
    assert sm.prune(spec, lambda sp: float("-inf")).removed == []
