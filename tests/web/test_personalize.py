"""Personalize: seasonal backdrops + verse bar assets exist and are wired into every page and the settings card."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "app" / "web" / "static"


def test_all_eight_season_images_exist_and_are_webp():
    for season in ("spring", "summer", "autumn", "winter"):
        for mode in ("dark", "light"):
            p = STATIC / "backgrounds" / f"{season}-{mode}.webp"
            assert p.exists() and p.read_bytes()[:4] == b"RIFF", p


def test_season_images_are_sharp_full_frames_with_thumbs():
    """Backdrops are full-frame photos (not the old 2048x768 faded strips) and each has a small thumb for the picker."""
    from PIL import Image
    for season in ("spring", "summer", "autumn", "winter"):
        for mode in ("dark", "light"):
            p = STATIC / "backgrounds" / f"{season}-{mode}.webp"
            with Image.open(p) as im:
                assert im.width >= 1400 and im.width / im.height > 1.4, (p, im.size)
            assert p.stat().st_size < 700_000, p
            t = STATIC / "backgrounds" / f"{season}-{mode}-thumb.webp"
            assert t.exists() and t.stat().st_size < 60_000, t


def test_backdrop_swap_fade_is_wired_in_css_and_js():
    css = (STATIC / "theme.css").read_text(encoding="utf-8")
    js = (STATIC / "t58-personalize.js").read_text(encoding="utf-8")
    assert "html.t58-bg-swap body::before" in css and "@keyframes t58-bg-swap" in css
    assert "position: fixed" in css.split("seasonal scenes")[1][:1800]  # the scene stays put while the page scrolls
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
    assert 'id="ac-row"' in html and 'id="ac-hex"' in html and 'id="pz-fx"' in html
    # requested order: picture, profile, security, personalize, subscription, app
    pos = [html.index(m) for m in ("<h2>Profile picture</h2>", "<h2>Profile</h2>", "<h2>Security</h2>",
                                   "<h2>Personalize</h2>", "<h2>Subscription</h2>", "<h2>App</h2>")]
    assert pos == sorted(pos)


def test_ambient_motion_is_compositor_only_and_respects_user_settings():
    css = (STATIC / "theme.css").read_text(encoding="utf-8")
    js = (STATIC / "t58-personalize.js").read_text(encoding="utf-8")
    block = css[css.index(".t58-ambient {"):]
    assert "pointer-events: none" in block and "prefers-reduced-motion" in block
    assert "@keyframes t58-fall" in block and "top:" not in block.split("@keyframes t58-fall")[1].split("}")[0]
    for kind in ("snow", "leaf", "petal", "firefly"):
        assert kind in js
    assert "saveData" in js and "visibilitychange" in js and "t58-fx" in js


def test_accent_presets_and_custom_hex_api():
    js = (STATIC / "t58-personalize.js").read_text(encoding="utf-8")
    for hexv in ("#35e0b0", "#4d90ff", "#9b8cff", "#35d878", "#f5b83d", "#ff5d73"):
        assert hexv in js
    assert "setAccent" in js and "data-ui-accent" in js
    head = (ROOT / "app" / "web" / "templates" / "_head_common.html").read_text(encoding="utf-8")
    assert "t58-accent-vars" in head  # accent painted before first paint -> no teal flash


def test_floating_panels_are_opaque():
    """The verdict popup (and other floating panels) must not let page text show through."""
    chrome = (STATIC / "t58-chrome.js").read_text(encoding="utf-8")
    assert ".t58-ve{" in chrome and "var(--pop-bg" in chrome.split(".t58-ve{")[1][:300]
    css = (STATIC / "theme.css").read_text(encoding="utf-8")
    assert "--pop-bg:" in css


def test_default_mountains_are_fixed_behind_the_page():
    """Both the night and day mountain backdrops stay put while the page scrolls (position: fixed, not absolute)."""
    import re
    css = (STATIC / "theme.css").read_text(encoding="utf-8")
    for img in ("dashboard-mountains.webp", "dashboard-mountains-light.webp"):
        i = css.index(f'url("/static/{img}")')
        block = css[css.rfind("body::before {", 0, i):i]
        assert re.search(r"position:\s*fixed", block), img
        assert "position: absolute" not in block, img
