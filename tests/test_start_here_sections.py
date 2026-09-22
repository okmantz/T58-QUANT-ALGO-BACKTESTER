"""Tests confirming the section-wide "Start Here" pages (one per top-level
sidebar section -- Create, Test, Optimize, Validate, Champion, Deployment,
Strategy Graveyard, Quant Lab, Account) render correctly, and that the
per-tool embedded cards this replaced are actually gone from those tools'
own pages (each tool's section now has ONE section-wide orientation page
instead of a small card duplicated onto one tool within it)."""
from __future__ import annotations

import pytest

from app.web.server import _SECTION_START_HERE, app


@pytest.mark.parametrize("section", [
    "create", "test", "champion", "deployment", "graveyard", "quantlab", "account",
])
def test_dedicated_start_here_page_renders(section):
    app.config["TESTING"] = True
    client = app.test_client()
    r = client.get(f"/start-here/{section}")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Start Here" in body
    data = _SECTION_START_HERE[section]
    assert data["description"][:40] in body
    for tool in data["tools"]:
        assert f'href="{tool["href"]}"' in body
    # Renders through Jinja's normal auto-escaping -- must never come out
    # double-escaped.
    assert "&amp;amp;" not in body


def test_unknown_section_is_a_clean_404():
    app.config["TESTING"] = True
    client = app.test_client()
    r = client.get("/start-here/not-a-real-section")
    assert r.status_code == 404


@pytest.mark.parametrize("path,section_key", [
    ("/optimize", "optimize"), ("/validate", "validate"),
])
def test_optimize_and_validate_hubs_are_framed_as_start_here(path, section_key):
    """Optimize and Validate already had their own hub/picker pages before
    this section-wide Start Here pass -- rather than getting a second,
    separate Start Here page, those pages themselves now carry the
    "Start Here" framing directly."""
    app.config["TESTING"] = True
    client = app.test_client()
    r = client.get(path)
    assert r.status_code == 200
    assert "Start Here" in r.get_data(as_text=True)


@pytest.mark.parametrize("path", ["/", "/forge", "/search", "/evolution", "/full-pipeline"])
def test_old_embedded_start_here_card_is_gone_from_individual_tool_pages(path):
    """These 5 tools each used to carry their OWN small embedded Start
    Here card (see _start_here.html's macro) -- now that every tool's
    section has its own section-wide Start Here page, the embedded
    per-tool copy would just be duplicated, out-of-date orientation info
    sitting on one tool instead of the section, so it was removed."""
    app.config["TESTING"] = True
    client = app.test_client()
    r = client.get(path)
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "start-here-card" not in body
    assert "t58ToggleStartHere" not in body
