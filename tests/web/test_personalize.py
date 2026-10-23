"""Personalize: seasonal backdrops + verse bar assets exist and are wired into every page and the settings card."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "app" / "web" / "static"


def test_all_eight_season_images_exist_and_are_webp():
    for season in ("spring", "summer", "autumn", "winter"):
        for mode in ("dark", "light"):
            p = STATIC / "backgrounds" / f"{season}-{mode}.webp"
            assert p.exists() and p.read_bytes()[:4] == b"RIFF", p


def test_season_images_are_artwork_box_sized_and_fade_at_the_edges():
    """Every backdrop is the 2048x768 artwork box, solid in the middle and transparent at the edges (so it melts in)."""
    from PIL import Image
    for season in ("spring", "summer", "autumn", "winter"):
        for mode in ("dark", "light"):
            p = STATIC / "backgrounds" / f"{season}-{mode}.webp"
            with Image.open(p) as im:
                assert im.size == (2048, 768) and im.mode == "RGBA", (p, im.size, im.mode)
                a = im.getchannel("A")
                assert a.getpixel((1024, 300)) == 255, p      # solid in the middle of the picture
                assert a.getpixel((1024, 767)) == 0, p        # dissolved at the bottom
                assert a.getpixel((1024, 0)) == 0, p          # ...and at the top
                assert a.getpixel((0, 300)) == 0 and a.getpixel((2047, 300)) == 0, p  # ...and at both sides


def test_backdrop_swap_fade_is_wired_in_css_and_js():
    css = (STATIC / "theme.css").read_text(encoding="utf-8")
    js = (STATIC / "t58-personalize.js").read_text(encoding="utf-8")
    assert "html.t58-bg-swap body::before" in css and "@keyframes t58-bg-swap" in css
    assert "prefers-reduced-motion" in css
    assert "t58-bg-swap" in js and "MutationObserver" in js


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
