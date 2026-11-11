"""Readability guard: secondary text tokens must stay readable (WCAG AA, 4.5:1) on the solid panel behind them, in every UI theme."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "app" / "web" / "static"


def _lum(h: str) -> float:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    ch = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    ch = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in ch]
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]


def _ratio(a: str, b: str) -> float:
    la, lb = _lum(a), _lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _blocks(css: str):
    for m in re.finditer(r"\{([^{}]*)\}", css):
        body = m.group(1)
        d = dict(re.findall(r"(--[a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{3,6})\b", body))
        if "--text-muted" in d and "--panel-3" in d:
            yield d


@pytest.mark.parametrize("token", ["--text-muted", "--text-dim"])
def test_secondary_text_is_readable_on_the_panel_in_every_theme(token):
    seen = 0
    for name in ("theme.css", "ui-themes.css"):
        for d in _blocks((STATIC / name).read_text(encoding="utf-8")):
            if token in d:
                seen += 1
                r = _ratio(d[token], d["--panel-3"])
                assert r >= 4.5, f"{name}: {token} {d[token]} on {d['--panel-3']} is only {r:.2f}:1"
    assert seen >= 3


def test_the_shared_readability_stylesheet_is_loaded_everywhere():
    head = (STATIC.parent / "templates" / "_head_common.html").read_text(encoding="utf-8")
    assert "t58-ui.css" in head
