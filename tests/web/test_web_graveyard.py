"""Tests for the /graveyard web page -- browsing the persistent, shared
Strategy Graveyard files (see app.search.graveyard.graveyard_path_for)."""
from app.search.graveyard import GraveyardEntry, graveyard_path_for, record_rejections
from app.web.server import app


def test_graveyard_page_loads_with_no_files():
    client = app.test_client()
    r = client.get("/graveyard")
    assert r.status_code == 200
    assert b"Strategy Graveyard" in r.data


def test_graveyard_page_lists_files_and_clusters(tmp_path, monkeypatch):
    monkeypatch.setattr("app.search.graveyard.get_app_base_dir", lambda: tmp_path)
    path = graveyard_path_for("ES1!", "1m")
    record_rejections([
        GraveyardEntry(
            candidate_id=f"c{i}", family="rsi_extreme_reversion", generation=None,
            stage_died="cpcv", reason="failed CPCV robustness",
            param_signature="rsi_extreme_reversion|lookback=68.0",
        )
        for i in range(4)
    ], path=path)

    client = app.test_client()
    r = client.get("/graveyard")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "rsi_extreme_reversion" in body
    assert "4x" in body


def test_graveyard_page_family_filter(tmp_path, monkeypatch):
    monkeypatch.setattr("app.search.graveyard.get_app_base_dir", lambda: tmp_path)
    path = graveyard_path_for("ES1!", "1m")
    record_rejections([
        GraveyardEntry(candidate_id="c1", family="fam_a", generation=None, stage_died="cpcv", reason="x",
                        param_signature="fam_a|x=1.0"),
        GraveyardEntry(candidate_id="c2", family="fam_b", generation=None, stage_died="stress", reason="y",
                        param_signature="fam_b|y=2.0"),
    ], path=path)

    client = app.test_client()
    r = client.get(f"/graveyard?path={path}&family=fam_b")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "fam_b" in body
    # Split on the actual "Dead neighborhoods" *heading*, not the first
    # match of that phrase anywhere on the page -- the sidebar nav also
    # has a "Dead neighborhoods & why they failed" link, which sits
    # before the family-filter dropdown (itself expected to always list
    # every family, including the filtered-out one, as an <option>).
    assert "fam_a" not in body.split("<h2>Dead neighborhoods")[1]


def test_graveyard_lists_rows_with_friendly_labels_and_detail_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr("app.search.graveyard.get_app_base_dir", lambda: tmp_path)
    path = graveyard_path_for("EURUSD/EURUSD240.csv", "4h")
    record_rejections([
        GraveyardEntry(candidate_id="c1", family="rsi_extreme_reversion", generation=None, stage_died="not_ranked",
                       reason="Lost money in both halves.", primary_failure="no edge after costs",
                       param_signature="rsi_extreme_reversion|rsi_period=7.0,oversold=20.0", prop_sim_pass_pct=12.0),
    ], path=path)
    client = app.test_client()
    body = client.get("/graveyard").get_data(as_text=True)
    assert "Rsi extreme reversion" in body                      # plain-English strategy name, not the identifier
    assert "No edge after trading costs" in body                # plain-English reason
    assert "EURUSD \u00b7 4h" in body                             # market label, not 'eurusd-eurusd240-csv / 4h'
    assert ">eurusd-eurusd240-csv" not in body
    assert 'data-id="c1"' in body and 'id="gy-panel"' in body   # clickable row + side panel
    d = client.get(f"/graveyard/detail?path={path}&id=c1").get_json()
    assert d["family_title"] == "Rsi extreme reversion" and d["can_rebuild"] is False
    assert any(f["label"] == "Eval pass chance" and f["value"] == 12 for f in d["facts"])
    c = client.get(f"/graveyard/curve?path={path}&id=c1").get_json()
    assert c["ok"] is False and "exact settings" in c["message"]  # honest, no invented curve
    assert client.get("/graveyard/detail?path=/etc/passwd&id=c1").status_code == 404
    assert client.get(f"/graveyard/detail?path={path}&id=nope").status_code == 404


def test_graveyard_curve_rebuilds_for_a_bundled_seed_row():
    import pytest
    from pathlib import Path
    if not (Path(__file__).resolve().parents[2] / "data" / "raw" / "EURUSD" / "EURUSD240.csv").exists():
        pytest.skip("bundled sample data not present")
    import json
    from app.search import graveyard_detail as gd
    seed = Path(gd.__file__).parent / "seed_data" / "graveyard_seed.jsonl"
    row = next(json.loads(l) for l in seed.open(encoding="utf-8") if "EURUSD/EURUSD240.csv" in l)
    out = gd.rebuild_curve(row)
    assert out["ok"] and len(out["points"]) > 20 and out["stats"]["trades"] > 0
