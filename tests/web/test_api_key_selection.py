"""Choosing which saved data-provider key a run uses: masked key hint, and Discovery Engine coverage."""
from pathlib import Path

from app.data import api_sources

ROOT = Path(__file__).resolve().parents[2]


def test_key_hint_masks_and_never_leaks(monkeypatch):
    monkeypatch.setattr(api_sources, "_key_for", lambda p: "SECRETKEY-abcd1234" if p.needs_key else "free")
    for item in api_sources.provider_list():
        blob = str(item)
        assert "SECRETKEY" not in blob
        if item["id"] == "dukascopy":
            assert item["key_hint"] == "" or item["key_hint"] == "no key needed"
        else:
            assert item["key_hint"].endswith("1234") and item["key_hint"].startswith("•")


def test_unsaved_key_has_no_hint(monkeypatch):
    monkeypatch.setattr(api_sources, "_key_for", lambda p: "")
    rows = {r["id"]: r for r in api_sources.provider_list()}
    assert rows["polygon"]["ready"] is False and rows["polygon"]["key_hint"] == ""


def test_picker_covers_discovery_engine_dataset_select():
    js = (ROOT / "app/web/static/api-source.js").read_text(encoding="utf-8")
    assert 'select[name="dataset"]:not([multiple])' in js
    html = (ROOT / "app/web/templates/discovery_engine.html").read_text(encoding="utf-8")
    assert 'name="dataset"' in html and "{% if dataset_groups %}\n      <select" not in html


def test_discovery_start_resolves_provider_token():
    src = (ROOT / "app/web/discovery_engine_routes.py").read_text(encoding="utf-8")
    assert '__api__:' in src and "resolve_token" in src


def test_provider_list_survives_a_broken_key_store(monkeypatch):
    def boom(p):
        raise RuntimeError("keyring exploded")
    monkeypatch.setattr(api_sources, "_key_for", boom)
    rows = api_sources.provider_list()
    assert len(rows) == len(api_sources.PROVIDERS) and all(r["key_hint"] == "" for r in rows if r["id"] != "dukascopy")


def test_picker_script_reports_problems_instead_of_hiding():
    js = (ROOT / "app/web/static/api-source.js").read_text(encoding="utf-8")
    assert "FALLBACK_PROVIDERS" in js and "warnFallback" in js and "reportProblem" in js


def test_discovery_start_with_provider_pick_downloads_first(monkeypatch):
    from app.web.server import app
    seen = {}

    def fake_resolve(token, http_get=None):
        seen["token"] = token
        raise RuntimeError("no key saved for Polygon")
    monkeypatch.setattr(api_sources, "resolve_token", fake_resolve)
    r = app.test_client().post("/discovery-engine/start", data={"dataset": "__api__:polygon|AAPL|5m||"})
    assert seen.get("token", "").startswith("__api__:polygon")
    assert r.status_code == 400 and b"download" in r.data.lower()
