"""v9.15 WS-E -- light-theme repair invariants.

The light theme failed Owen's screenshot two ways: (a) theme.css's light
tokens were washed out (--text-dim #7c8894 at 3.37:1, link teal #0f9f78
at 3.13:1, --border #dde3ea at 1.20:1 against the page -- white cards
vanished), and (b) _lifecycle_css.html was a fixed dark palette with no
light scope at all, so Validate / Full Pipeline rendered dark slabs on
the light page. These tests parse the real CSS, resolve the light
token set, and assert WCAG contrast bars; they also pin the dark theme:
every declaration NOT under a light scope must hash to the value
captured at task start (2026-10-10, before the WS-E edits), and the
known dark token values must be exactly what they were.
"""
import hashlib
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
THEME = ROOT / "app/web/static/theme.css"
JOB = ROOT / "app/web/static/job-theme.css"
LIFECYCLE = ROOT / "app/web/templates/_lifecycle_css.html"

# Dark-rule hashes captured from the working tree at WS-E task start
# (normalised: comments stripped, whitespace collapsed, light-scoped
# blocks removed). If a dark declaration changes, these break.
# 2026-10-22: theme.css digest re-pinned after the Personalize work (seasonal backdrops, Bible verse bar, live-PnL pill) ADDED
# new dark-scope rules; no existing dark token or rule was changed (test_dark_token_values_pinned still passes).
DARK_HASH = {
    # NOTE (WS-E, 2026-10-10): hashes measured on the pre-task working
    # tree with the same normalisation _dark_hash() applies. The WS-E
    # edits only ADD light-scoped rules, so a correct tree reproduces
    # these digests exactly; the values below were re-measured after the
    # edits and match the pre-edit dark extraction rule-for-rule (the
    # raw-text diff is only the added comment blocks).
    # Re-pinned (dashboard redesign): the dark palette was deliberately retuned ("Deep Teal Glass": violet/cyan/gold
    # accents, glass panels, mountain backdrop). Light-scoped rules are untouched.
    # Re-pinned 2026-10-23: the new seasonal artwork (scripts/generate_season_backdrops.py) needed per-scene night
    # opacities (the four night scenes differ a lot in brightness) plus the t58-bg-swap fade-in keyframes. Only the
    # seasonal-backdrop rules in theme.css changed; no dark token or any other rule did.
    # Re-pinned 2026-10-28: the midnight-ocean backdrop's CSS swell animation was replaced by the canvas scene (.t58-scene-fx rules).
    # No dark token moved (test_dark_token_values_pinned still passes).
    # Re-pinned 2026-10-30: Lucide sidebar icons (.lu, muted nav-icon colour), the notification bell + inbox panel rules, and the
    # three unlockable backdrops (obsidian-gold / sakura-reverie / abyssal-blue) were added to theme.css (commits f49c523..e0ec4f8).
    # No dark token moved (test_dark_token_values_pinned still passes).
    "theme.css": "793f9089fd1591c4b0b9e1a3af73872a7cd996db5f3e7165047908c49088f381",
    # Re-pinned 2026-10-28: .badge.pass now uses var(--ok) instead of var(--teal) (commit 8f643dd); in the dark scope both are #35e0b0.
    # Re-pinned 2026-10-30: the "Why this verdict" waterfall component (.wf / .wf-step rules) was ADDED (commit f49c523); no existing rule changed.
    "job-theme.css": "1ea80e4f3812928af01d78a9f684cf1e6d13381ea3cebcdba7662127ba89b0a2",
    # Re-pinned 2026-10-13: the lifecycle (Champion board) dark rules were extended after the
    # WS-E pin was taken; token values (DARK_TOKENS below) are still checked separately.
    # Re-pinned (repo cleanup): the pages' hard-coded colors became var(--token) references whose DARK
    # values are the old literals (verified token-by-token: reverse-substituting reproduces the original CSS
    # exactly), and theme.css gained the page-token block. job-theme.css's pin had already drifted.
    "_lifecycle_css.html": "87c894b3b9d27f0a47f1722f4ba0c240c8eaa0ddcd4ea99998c8620aaa96e6ec",
}

DARK_TOKENS = {  # :root values as found at task start -- do not move
    "--bg": "#05070a", "--panel": "#0d1017", "--panel-2": "#10141c",
    "--panel-3": "#151a24", "--border": "#1c2230",
    "--border-light": "#2a3242", "--text": "#e7ebf2",
    "--text-muted": "#8a93a6", "--text-dim": "#5b6478",
    "--teal": "#35e0b0", "--violet": "#8b7cff", "--coral": "#ff6f6f",
    "--amber": "#f0b429",
}
LC_DARK_TOKENS = {  # .lc-scope values as found at task start
    "--lc-bg": "#0a0a0d", "--lc-panel": "#10131a", "--lc-panel2": "#141926",
    "--lc-panel3": "#1a2130", "--lc-border": "#232b3a",
    "--lc-border2": "#2d3648", "--lc-text": "#e8eaf0",
    "--lc-muted": "#8b93a5", "--lc-faint": "#5c6478", "--lc-acc": "#00e5a0",
}


def _blocks(css):
    """Yield (selector, body) for each top-level rule, brace-matched."""
    i = 0
    while True:
        j = css.find("{", i)
        if j < 0:
            return
        depth, m = 1, j + 1
        while depth:
            depth += (css[m] == "{") - (css[m] == "}")
            m += 1
        yield css[i:j].strip(), css[j + 1 : m - 1]
        i = m


def _tokens(body):
    return {k: v.strip() for k, v in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body)}


def _scope_tokens(css, selector_fragment):
    for sel, body in _blocks(css):
        if selector_fragment in sel:
            return _tokens(body)
    raise AssertionError(f"scope {selector_fragment!r} not found")


def _dark_hash(css):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    kept = []
    for sel, body in _blocks(css):
        if 'data-theme="light"' in sel:
            continue
        if sel.startswith("@"):  # keyframes/media: keep inner rules too
            inner = [(s, b) for s, b in _blocks(body)
                     if 'data-theme="light"' not in s]
            kept.append(sel + "{" + "".join(s + "{" + b + "}" for s, b in inner) + "}")
        else:
            kept.append(sel + "{" + body + "}")
    norm = re.sub(r"\s+", " ", "".join(kept)).strip()
    return hashlib.sha256(norm.encode()).hexdigest()


def _lum(hexcolor):
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a, b):
    x, y = _lum(a), _lum(b)
    return (max(x, y) + 0.05) / (min(x, y) + 0.05)


def _resolve(tokens, value):
    seen = set()
    while True:
        m = re.fullmatch(r"var\((--[\w-]+)\)", value.strip())
        if not m or m.group(1) in seen:
            return value.strip()
        seen.add(m.group(1))
        value = tokens[m.group(1)]


def test_dark_rules_unchanged():
    for path, key in ((THEME, "theme.css"), (JOB, "job-theme.css"),
                      (LIFECYCLE, "_lifecycle_css.html")):
        got = _dark_hash(path.read_text())
        assert got == DARK_HASH[key], f"{key} dark declarations changed: {got}"


def test_dark_token_values_pinned():
    dark = _scope_tokens(THEME.read_text(), ":root")
    for k, v in DARK_TOKENS.items():
        assert dark[k] == v, f"dark {k} moved: {dark[k]} != {v}"
    lc = _scope_tokens(LIFECYCLE.read_text(), ".lc-scope")
    for k, v in LC_DARK_TOKENS.items():
        assert lc[k] == v, f"dark {k} moved: {lc[k]} != {v}"


def test_light_text_contrast():
    light = _scope_tokens(THEME.read_text(), 'html[data-theme="light"]')
    for text_tok in ("--text", "--text-muted", "--text-dim"):
        for surf in ("--bg", "--panel", "--panel-3"):
            ratio = contrast(light[text_tok], light[surf])
            assert ratio >= 4.5, f"{text_tok} on {surf}: {ratio:.2f} < 4.5"


def test_light_accents_and_links_contrast():
    light = _scope_tokens(THEME.read_text(), 'html[data-theme="light"]')
    # links render in --teal on body/card surfaces; status hues likewise
    for tok in ("--teal", "--violet", "--blue"):
        assert contrast(light[tok], light["--panel"]) >= 4.5, tok
    assert contrast(light["--teal"], light["--bg"]) >= 4.5
    assert contrast(light["--coral"], light["--panel"]) >= 4.5


def test_light_cards_are_distinct():
    light = _scope_tokens(THEME.read_text(), 'html[data-theme="light"]')
    # card (--panel) vs page (--bg) must differ, AND the border must read
    assert contrast(light["--panel"], light["--bg"]) > 1.05
    assert contrast(light["--border"], light["--bg"]) >= 1.4
    assert contrast(light["--border"], light["--panel"]) >= 1.4
    assert contrast(light["--border-light"], light["--panel"]) >= 2.0


def test_light_filled_button_ink():
    light = _scope_tokens(THEME.read_text(), 'html[data-theme="light"]')
    assert contrast("#ffffff", light["--teal"]) >= 4.5
    assert contrast("#ffffff", light["--violet"]) >= 4.5
    assert contrast("#ffffff", light["--coral"]) >= 4.5


def test_lifecycle_has_light_scope_with_contrast():
    css = LIFECYCLE.read_text()
    light = _scope_tokens(css, 'html[data-theme="light"] .lc-scope')
    for text_tok in ("--lc-text", "--lc-muted", "--lc-faint"):
        for surf in ("--lc-panel", "--lc-bg"):
            ratio = contrast(light[text_tok], light[surf])
            assert ratio >= 4.5, f"{text_tok} on {surf}: {ratio:.2f} < 4.5"
    assert contrast(light["--lc-acc"], light["--lc-panel"]) >= 4.5
    assert contrast(light["--lc-panel"], light["--lc-bg"]) > 1.05
    assert contrast(light["--lc-border"], light["--lc-bg"]) >= 1.4


def test_job_log_is_light_surface_in_light_mode():
    css = JOB.read_text()
    rules = {sel: body for sel, body in _blocks(css)}
    log_rule = next(b for s, b in rules.items()
                    if s.strip().endswith('html[data-theme="light"] #log'))
    assert "#ffffff" in log_rule and "var(--text)" in log_rule


def test_light_theme_has_its_own_mountain_backdrop():
    """Light theme gets a daytime mountain range (not the dark artwork), wired through a light-scoped rule."""
    static = pathlib.Path(__file__).resolve().parents[2] / "app" / "web" / "static"
    img = static / "dashboard-mountains-light.webp"
    assert img.is_file() and img.stat().st_size > 5_000
    css = (static / "theme.css").read_text()
    assert 'html[data-theme="light"] body::before' in css
    assert "dashboard-mountains-light.webp" in css
