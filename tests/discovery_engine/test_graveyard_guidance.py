from __future__ import annotations

from app.discovery_engine.graveyard_guidance import GraveyardGuidance
from app.search.graveyard import param_signature


def _rows():
    cfg = {"len": 14.0}
    sig = param_signature("fam_dead", cfg)
    rows = [{"family": "fam_dead", "param_signature": sig, "avg_r_net": -0.2, "n_trades": 150} for _ in range(8)]
    rows += [{"family": "fam_near", "param_signature": "fam_near|len=20", "doubled_cost_avg_r": 0.03, "n_trades": 180} for _ in range(3)]
    rows += [{"family": "fam_near", "param_signature": "fam_near|len=30", "avg_r_net": -0.1, "n_trades": 150}]
    return rows, cfg


def test_penalty_is_graduated_and_capped():
    rows, cfg = _rows()
    g = GraveyardGuidance.from_rows(rows)
    assert g.penalty_r("fam_dead", cfg) == min(g.max_penalty_r, (8 - g.free_deaths) * g.per_death_r)
    assert g.penalty_r("fam_dead", {"len": 99.0}) == 0.0
    assert 0 < g.penalty_r("fam_dead", cfg) <= g.max_penalty_r


def test_family_weight_prefers_near_misses_and_unseen_is_neutral():
    rows, _ = _rows()
    g = GraveyardGuidance.from_rows(rows)
    assert g.family_weight("fam_near") > g.family_weight("fam_dead")
    assert g.family_weight("never_seen") == 0.5
    assert g.reorder(["fam_dead", "fam_near"], lambda x: x)[0] == "fam_near"


def test_empty_guidance_is_inert():
    g = GraveyardGuidance.from_rows([])
    assert g.penalty_r("x", {}) == 0.0 and "no prior" in g.summary()
