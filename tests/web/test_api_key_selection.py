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
