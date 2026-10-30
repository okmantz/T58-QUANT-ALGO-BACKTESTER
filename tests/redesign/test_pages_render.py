"""Smoke tests for the pages touched by the Oct-2026 redesign: they must render and carry the new pieces."""
import pytest


def _client():
    from app.web.server import app
    app.config["TESTING"] = True
    return app.test_client()


@pytest.mark.parametrize("path,needle", [
    ("/achievements", "Certificates &amp; medals"),
    ("/discovery-engine", "Ideas from your research"),
    ("/achievements/certificate.svg", "<svg"),
])
def test_page_renders(path, needle):
    r = _client().get(path)
    assert r.status_code == 200, path
    assert needle in r.get_data(as_text=True)


def test_medal_route_unknown_404():
    assert _client().get("/achievements/medal/nope.svg").status_code == 404


@pytest.mark.parametrize("path,needle", [
    ("/support", "Report an issue"),
    ("/resources", "Start here"),
    ("/hedge-fund", "Portfolio Builder"),
    ("/quant-lab", "Search"),
    ("/research-labs", "Reproducible"),
    ("/dashboard", "Good"),
])
def test_redesigned_pages_render(path, needle):
    r = _client().get(path, follow_redirects=True)
    assert r.status_code == 200, path
    assert needle in r.get_data(as_text=True)
