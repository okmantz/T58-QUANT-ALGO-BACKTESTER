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
    assert "fam_a" not in body.split("Dead neighborhoods")[1]
