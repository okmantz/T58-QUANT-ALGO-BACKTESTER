from app.prop.presets import list_presets
from app.prop.provenance import GLOBAL_NOT_MODELED, provenance, provenance_table


def test_every_preset_has_a_row_and_flags_match_state():
    t = provenance_table()
    assert t["n_presets"] == len(list_presets()) == len(t["presets"])
    for r in t["presets"]:
        assert r["unverified"] == (r["state"] == "unverified")
        assert isinstance(r["not_modeled"], list)
    assert t["n_unverified"] == sum(r["unverified"] for r in t["presets"])
    assert t["global_not_modeled"] == list(GLOBAL_NOT_MODELED)


def test_unmodeled_list_follows_the_preset_fields():
    import dataclasses
    p = list_presets()[0]
    off = dataclasses.replace(p, flatten_time_ct=None, max_contracts=None)
    on = dataclasses.replace(p, flatten_time_ct="15:45", max_contracts=5)
    assert any("flatten" in s.lower() for s in provenance(off)["not_modeled"])
    assert not any("flatten" in s.lower() for s in provenance(on)["not_modeled"])
