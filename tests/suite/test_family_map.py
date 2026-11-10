from app.search import family_intelligence as fi


def _rows(fam, n, r=0.1, inst="MES", sig=True, cost=50.0, p=0.01, prop=60.0):
    return [{"family": fam, "avg_r_net": r, "instrument": inst if not isinstance(inst, list) else inst[i % len(inst)],
             "param_signature": f"s{i}" if sig else "same", "doubled_cost_net_profit": cost, "null_gate_p": p,
             "prop_sim_pass_pct": prop} for i in range(n)]


def test_small_family_shows_no_rates():
    out = fi.summarize(_rows("trend_breakout", 3))
    f = out["families"][0]
    assert f["profitable"] is None and f["significant"] is None
    assert "Too few" in f["verdict"]


def test_rates_tags_and_redundancy():
    rows = _rows("trend_breakout", 10, inst=["MES", "MNQ", "MGC"]) + _rows("mean_reversion_band", 10, r=-0.2, sig=False, cost=-5, p=0.6)
    out = fi.summarize(rows)
    by = {f["family"]: f for f in out["families"]}
    good = next(f for k, f in by.items() if f["profitable"] == 1.0)
    bad = next(f for k, f in by.items() if f["profitable"] == 0.0)
    assert "profitable" in good["tags"] and "survives higher costs" in good["tags"] and "works across instruments" in good["tags"]
    assert "redundant" in bad["tags"] and bad["survives_cost"] == 0.0
    assert good["redundancy"] == 0.0 and bad["redundancy"] > 0.8


def test_missing_figures_not_counted_as_failures():
    rows = _rows("trend_breakout", 6)
    for r in rows:
        r["doubled_cost_net_profit"] = None
    assert fi.summarize(rows)["families"][0]["survives_cost"] is None


def test_page_and_api(monkeypatch):
    import app.search.graveyard as gy
    monkeypatch.setattr(gy, "load_graveyard", lambda *a, **k: _rows("trend_breakout", 6))
    from app.web.server import app
    c = app.test_client()
    assert c.get("/family-map").status_code == 200
    d = c.get("/api/family-map").get_json()
    assert d["ok"] and d["families"]
