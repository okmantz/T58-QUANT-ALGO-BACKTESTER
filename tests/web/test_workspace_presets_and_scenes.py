"""2026-10-30: Workspace presets, the four new scenes, the wall-clock sky and the Focus redesign are wired end to end."""
import pathlib
import re

from app import achievements
from app.web.server import app

ROOT = pathlib.Path(__file__).resolve().parents[2]
STATIC = ROOT / "app" / "web" / "static"
PRESETS = ("obsidian", "cyberpunk", "nordic", "evergreen", "sovereign")
NEW_SCENES = ("endless-sands", "coastal-italy", "bioluminescence")


def test_every_preset_is_an_unlockable_reward_with_real_achievements():
    assert set(achievements.WORKSPACE_UNLOCKS) == set(PRESETS)
    for key, spec in achievements.WORKSPACE_UNLOCKS.items():
        assert spec["ach"], key
        for aid in spec["ach"]:
            assert aid in achievements.CATALOGUE, (key, aid)
    st = achievements.unlock_state()
    assert set(st["workspaces"]) == set(PRESETS) and set(st["locked_workspaces"]) <= set(PRESETS)


def test_new_scenes_are_earned_and_nordic_kingdom_is_not_locked():
    for name in NEW_SCENES:
        assert name in achievements.BACKGROUND_UNLOCKS
    assert "nordic-kingdom" not in achievements.BACKGROUND_UNLOCKS          # unlocked on sign-in
    for name in NEW_SCENES + ("nordic-kingdom", "sakura-reverie-hd"):
        for suffix in ("dark", "light", "thumb"):
            f = STATIC / "backgrounds" / f"{name}-{suffix}.webp"
            assert f.is_file() and f.stat().st_size > 5_000, f


def test_css_defines_every_preset_and_scene():
    ws = (STATIC / "workspace-presets.css").read_text()
    for p in PRESETS:
        assert f'html[data-ws="{p}"]' in ws, p
    theme = (STATIC / "theme.css").read_text()
    for name in NEW_SCENES + ("nordic-kingdom",):
        assert f'html[data-bg="{name}"]:not([data-theme="light"])' in theme and f'html[data-bg="{name}"][data-theme="light"]' in theme, name
    assert "sakura-reverie-hd-dark.webp" in theme and "sakura-reverie-dark.webp" not in theme


def test_scene_registry_and_sky_clock():
    js = (STATIC / "t58-personalize.js").read_text()
    for name in NEW_SCENES + ("nordic-kingdom",):
        assert f"'{name}'" in js, name
    for p in PRESETS:
        assert f"{p}: '" in js
    canvas = (STATIC / "t58-scene-canvas.js").read_text()
    assert "skyAt(Date.now() / 1000)" in canvas                    # the day cycle follows the wall clock, so it never restarts per page
    for kind in ("nordic", "sands", "italy", "biolum"):
        assert f"if (kind === '{kind}')" in canvas, kind
    assert "function strike(" in canvas and "function drawWalker(" in canvas and "function drawBoat(" in canvas


def test_account_page_and_head_expose_presets():
    c = app.test_client()
    html = c.get("/settings/account").get_data(as_text=True)
    assert 'id="ws-grid"' in html and all(f'data-ws="{p}"' in html for p in PRESETS)
    assert 'data-bg="nordic-kingdom"' in html and all(f'data-bg="{n}"' in html for n in NEW_SCENES)
    assert html.index('id="ut-grid"') < html.index('id="ws-grid"')           # presets sit underneath the interface themes
    assert "/static/workspace-presets.css" in html
    assert re.search(r'"ws":\s*\[', html)                                   # the lock list now carries the presets


def test_focus_redesign_keeps_its_api_contract():
    js = (STATIC / "pg-library.js").read_text()
    for needle in ("/api/library/focus", "/api/library/focus/clear", "focus-active", "focus-sound", "fx-dial", "fx-week"):
        assert needle in js, needle
