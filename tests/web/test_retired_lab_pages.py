"""Search Lab / Evolution Lab / Forge / Speed Run (and the multi-instrument twins) are retired as pages: they send people to Strategy Discovery."""
from __future__ import annotations

import pytest

from app.web.server import app

RETIRED = ["/search", "/search/multi-instrument", "/evolution", "/evolution/multi-instrument", "/forge", "/speed-run", "/speed-run/multi-instrument"]


@pytest.fixture
def client():
    return app.test_client()


@pytest.mark.parametrize("path", RETIRED)
def test_retired_page_redirects_to_strategy_discovery(client, path):
    r = client.get(path)
    assert r.status_code == 302 and r.headers["Location"].startswith("/discovery-engine")


def test_strategy_discovery_explains_where_the_lab_went(client):
    html = client.get(client.get("/forge").headers["Location"]).get_data(as_text=True)
    assert "now lives inside Strategy Discovery" in html


def test_retired_labs_are_gone_from_the_sidebar_and_guides(client):
    html = client.get("/discovery-engine").get_data(as_text=True)
    for href in ('href="/forge"', 'href="/speed-run"', 'href="/evolution"', 'href="/search"'):
        assert href not in html
    from app.orchestration.section_guides import SECTION_GUIDES
    hrefs = {t["href"] for g in SECTION_GUIDES.values() for key in ("tools", "more_tools") for t in g.get(key, [])}
    assert not hrefs & set(RETIRED)


def test_research_agent_and_director_are_one_page(client):
    assert client.get("/research-agent").headers["Location"] == "/research?mode=ask"
    ask, analyze = (client.get(f"/research?mode={m}").get_data(as_text=True) for m in ("ask", "analyze"))
    assert "AI Research" in ask and "AI Research" in analyze
    assert 'action="/research-agent/start"' in ask and 'action="/research/run"' in analyze
