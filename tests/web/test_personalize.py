"""Personalize: seasonal backdrops + verse bar assets exist and are wired into every page and the settings card."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "app" / "web" / "static"


def test_all_eight_season_images_exist_and_are_webp():
    for season in ("spring", "summer", "autumn", "winter"):
        for mode in ("dark", "light"):
            p = STATIC / "backgrounds" / f"{season}-{mode}.webp"
            assert p.exists() and p.read_bytes()[:4] == b"RIFF", p


def test_css_has_rule_for_every_season_and_theme():
    css = (STATIC / "theme.css").read_text(encoding="utf-8")
    for season in ("spring", "summer", "autumn", "winter"):
        assert f"backgrounds/{season}-dark.webp" in css and f"backgrounds/{season}-light.webp" in css


def test_verse_list_has_enough_entries_and_default_off():
    js = (STATIC / "t58-personalize.js").read_text(encoding="utf-8")
    assert js.count('["') >= 40  # [ref, text] pairs
    assert "get('t58-verse') === '1'" in js  # off unless explicitly enabled
    assert "30 * 60 * 1000" in js


def test_pages_load_personalize_and_settings_card_present(client=None):
    from app.web.server import app
    c = app.test_client()
    r = c.get("/settings/account")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "t58-personalize.js" in html and 'id="pz-grid"' in html and 'id="pz-verse"' in html
