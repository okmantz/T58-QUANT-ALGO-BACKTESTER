"""Regression: every link on the Validate Start Here page must resolve to a
real route (the walk-forward cards used to point at /wfo and /wfga, which do
not exist, so clicking them showed "page not found")."""
from __future__ import annotations

import re
from pathlib import Path

from app.web import server

TEMPLATE = Path(server.__file__).parent / "templates" / "validate_hub.html"


def _hrefs():
    return re.findall(r'href="(/[^"{}#?]*)"', TEMPLATE.read_text(encoding="utf-8"))


def test_every_validate_hub_href_matches_a_route():
    adapter = server.app.url_map.bind("localhost")
    missing = []
    for href in _hrefs():
        if href.startswith("/static/"):
            continue
        try:
            adapter.match(href, method="GET")
        except Exception:
            missing.append(href)
    assert not missing, f"validate_hub.html links to nonexistent routes: {missing}"


def test_walk_forward_cards_use_real_routes():
    hrefs = _hrefs()
    assert "/walk-forward-opt" in hrefs and "/walk-forward-ga" in hrefs
    assert "/wfo" not in hrefs and "/wfga" not in hrefs


def test_validate_page_and_its_links_return_200():
    client = server.app.test_client()
    page = client.get("/validate")
    assert page.status_code == 200
    for href in ("/walk-forward-opt", "/walk-forward-ga", "/cpcv", "/pbo", "/sensitivity",
                 "/parameter-robustness", "/regime-matrix", "/payout-probability"):
        assert client.get(href).status_code == 200, href
