"""Single source of truth for the desktop app's colors (extracted from main_window.py).

Every UI module declares the palette names it uses as plain module globals and calls
``theme.bind(globals())``. ``apply_theme()`` then rewrites those globals in every bound module, so a
Dark/Light toggle reaches all tabs/widgets exactly like the old single-module ``globals().update()`` did.

This module deliberately imports no tkinter, so it is safe to import from anywhere (e.g. the licensing
gate that runs before the main window loads).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Theme system -- every color used across the whole app is one of the names
# below, looked up as a plain module global. THEMES holds the full palette
# for each mode; apply_theme() (near the bottom of this block) overwrites
# these globals in place and triggers a full UI rebuild, which is how a
# single Dark/Light toggle reaches every tab, chart, and widget at once
# without threading a theme object through thousands of call sites.
# ---------------------------------------------------------------------------
THEMES = {
    # UPGRADE (Sep 2026 UI pass): core semantic colors realigned to the
    # exact same palette as the web app's theme.css, so opening either
    # build feels like the same product (same brand teal/violet/coral/
    # amber, same background/panel/border/text ramps). Only the core
    # brand colors moved -- BLUE, METAL/METAL_BRIGHT, and the decorative
    # NEON_* dashboard-tile set are unchanged (they have no equivalent in
    # the web palette and changing them risked clashing with the existing
    # glow/KPI rendering for no real consistency gain).
    #
    # UPGRADE (2026-09-15, brand kit): ACCENT (the primary brand accent)
    # and NEON_CYAN (the logo's second brand color) retuned to the exact
    # T58 brand palette -- cyan #00D4FF / purple #7B3DFF -- from the
    # official logo/icon assets (see /assets/brand/palette.md). GREEN/
    # RED/AMBER/BLUE and the rest of the NEON_* set are semantic (pass/
    # fail/warning) or purely decorative and intentionally untouched --
    # see the note above.
    "dark": {
        "BG": "#05070A",             # == web --bg
        "PANEL": "#10141C",          # card surface == web --panel-2 (web cards sit on --panel-2, not --panel)
        "SIDEBAR": "#0D1017",        # sidebar / title strip == web --panel
        "PANEL_2": "#10141C",        # == web --panel-2
        "PANEL_3": "#151A24",        # == web --panel-3
        "PANEL_HOVER": "#1B212C",    # hover state for interactive surfaces (buttons, rows)
        "BORDER": "#1C2230",         # == web --border
        "BORDER_LIGHT": "#2A3242",   # == web --border-light
        "TEXT": "#E7EBF2",           # == web --text
        "TEXT_MUTED": "#8A93A6",     # == web --text-muted
        "TEXT_DIM": "#5B6478",       # == web --text-dim
        "METAL": "#B8BDC5",
        "METAL_BRIGHT": "#E7E9ED",
        "LOG_BG": "#070A0E",         # background for the monospace live-log/output Text widgets
        "GREEN": "#35E0B0",          # == web --teal -- pass/validated/success everywhere
        "RED": "#FF6F6F",            # == web --coral -- fail/danger/destructive everywhere
        "BLUE": "#6FA8FF",
        "AMBER": "#F0B429",          # == web --amber -- warning everywhere
        # WEB-PARITY MAKEOVER: the primary accent is now the web app's teal --
        # every primary button, focus ring, selected-stage highlight and
        # default nav highlight on the web is --teal, so the desktop's was
        # the one place that read as a different product (violet).
        "ACCENT": "#35E0B0",          # == web --teal -- primary accent / CTA
        "ACCENT_HOVER": "#5BEBC4",
        "ACCENT_DIM": "#0F3D34",      # low-opacity-style accent for subtle fills/left-bars
        "ACCENT_INK": "#04120E",     # == web primary-button ink (near-black on bright teal)
        "VIOLET": "#7B3DFF",         # == web --violet (brand purple: live/info, brand-mark gradient)
        # Neon accent set -- used for the glowing card borders / ring progress /
        # per-metric coloring on the Dashboard tab, matching the neon-dark
        # reference mockups. Kept separate from the semantic GREEN/RED/AMBER above
        # (which mean pass/fail/warning everywhere else in the app) -- these are
        # purely decorative variety across KPI tiles, the way the mockups give
        # every stat card a different hue rather than making hue mean something.
        # NEON_CYAN doubles as the T58 logo's second brand color (see
        # /assets/brand/palette.md) -- see app.web.static.theme.css's
        # --neon-cyan, which this must stay in sync with.
        "NEON_CYAN": "#00D4FF",
        "NEON_MAGENTA": "#FF2BD6",
        "NEON_LIME": "#B6FF3C",
        "NEON_VIOLET": "#8A5CFF",
        "NEON_AMBER": "#FFB547",
    },
    "light": {
        "BG": "#F5F7FA",             # == web light --bg
        "PANEL": "#FFFFFF",          # == web light --panel
        "SIDEBAR": "#FFFFFF",
        "PANEL_2": "#FFFFFF",        # == web light --panel-2
        "PANEL_3": "#EEF2F6",        # == web light --panel-3
        "PANEL_HOVER": "#E3E8EF",
        "BORDER": "#DDE3EA",         # == web light --border
        "BORDER_LIGHT": "#C9D2DB",   # == web light --border-light
        "TEXT": "#101820",           # == web light --text
        "TEXT_MUTED": "#52616C",     # == web light --text-muted
        "TEXT_DIM": "#7C8894",       # == web light --text-dim
        "METAL": "#6B7280",
        "METAL_BRIGHT": "#2F333B",
        "LOG_BG": "#FBFBFC",
        "GREEN": "#0F9F78",          # == web light --teal
        "RED": "#E0453F",            # == web light --coral
        "BLUE": "#2C64D6",
        "AMBER": "#B8790F",          # == web light --amber
        "ACCENT": "#0F9F78",         # == web light --teal
        "ACCENT_HOVER": "#0A7259",
        "ACCENT_DIM": "#D3F0E7",
        "ACCENT_INK": "#FFFFFF",
        "VIOLET": "#602BEF",         # == web light --violet
        # Same decorative role as the dark theme's neon set, deliberately
        # darkened/desaturated from true neon so they stay legible as text
        # and card borders against a near-white background instead of
        # glaring -- same hue identity per tile, tuned for contrast rather
        # than raw brightness.
        "NEON_CYAN": "#0079A3",
        "NEON_MAGENTA": "#B01C93",
        "NEON_LIME": "#5D8A12",
        "NEON_VIOLET": "#6438C9",
        "NEON_AMBER": "#A66A16",
    },
}

_THEME_SETTINGS_FILENAME = "ui_theme.json"


def _load_theme_name() -> str:
    """Never raises -- defaults to 'dark' (this app's original look) if
    nothing saved yet, the settings file is corrupt, or the persistence
    layer isn't reachable for any reason."""
    try:
        import json as _json

        from app.data.storage import get_app_base_dir

        path = get_app_base_dir() / "data" / "config" / _THEME_SETTINGS_FILENAME
        if path.exists():
            name = _json.loads(path.read_text(encoding="utf-8")).get("theme")
            if name in THEMES:
                return name
    except Exception:
        pass
    return "dark"


def _save_theme_name(name: str) -> None:
    try:
        import json as _json

        from app.data.storage import get_app_base_dir

        config_dir = get_app_base_dir() / "data" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / _THEME_SETTINGS_FILENAME).write_text(_json.dumps({"theme": name}), encoding="utf-8")
    except Exception:
        pass  # persistence is a convenience -- never worth crashing the toggle over


CURRENT_THEME = _load_theme_name()

# Modules whose globals receive the palette (see bind()).
_BOUND: list[dict] = []


def bind(module_globals: dict) -> None:
    """Register a module's globals dict; it is populated now and refreshed on every apply_theme()."""
    if not any(g is module_globals for g in _BOUND):
        _BOUND.append(module_globals)
    module_globals.update(THEMES[CURRENT_THEME])
    module_globals["CURRENT_THEME"] = CURRENT_THEME


def apply_theme(name: str) -> None:
    """Overwrites every color constant (as module globals in every bound module) with the named
    theme's palette. Callers must rebuild any already-built widgets themselves afterward (see
    MainWindow._toggle_theme) -- this only updates the source of truth those widgets read their
    colors from; it cannot recolor widgets whose colors were baked in at construction time."""
    global CURRENT_THEME
    if name not in THEMES:
        return
    CURRENT_THEME = name
    for g in _BOUND:
        g.update(THEMES[name])
        g["CURRENT_THEME"] = name
    _save_theme_name(name)


apply_theme(CURRENT_THEME)

FONT = "Segoe UI"
MONO = "Consolas"

# Sidebar width in px -- == web theme.css --sidebar-w (240px).
SIDEBAR_WIDTH = 240
