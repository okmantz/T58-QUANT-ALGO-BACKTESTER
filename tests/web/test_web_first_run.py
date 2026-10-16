"""First-run flow of the web app: license -> Dashboard (or Mobile Access),
the one-time guided tour, and the Dashboard's workflow cards."""
from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from app.licensing import client as license_client
from app.web import onboarding
from app.web import server as server_module
from app.web.server import app


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    monkeypatch.setenv("T58_LICENSE_SERVER_URL", "https://license.example.invalid")
    server_module._license_ok_cached = None
    monkeypatch.setattr(onboarding, "_flag_path", lambda: tmp_path / "web_onboarding.json")
    yield
    server_module._license_ok_cached = None


def _activate(next_value):
    with patch.object(license_client, "activate", return_value=(True, "Activated.")):
        return app.test_client().post("/activate/submit", data={
            "email": "a@b.co", "license_key": "T58-AAAA", "next": next_value,
        }, follow_redirects=False)


@pytest.mark.parametrize("next_value", ["/", "", "/activate", "//evil.example", "http://evil.example/x", "\\\\evil"])
def test_activation_never_lands_on_run_and_report_or_offsite(next_value):
    r = _activate(next_value)
    assert r.status_code in (301, 302, 303, 307, 308)
    assert r.headers["Location"] == "/dashboard"


def test_activation_can_land_on_mobile_access():
    assert _activate("/mobile-access").headers["Location"] == "/mobile-access"


def test_unlicensed_launcher_url_goes_through_activation_to_dashboard():
    with patch.object(license_client, "validate", return_value=(False, "Not activated.")):
        with patch.object(license_client, "load_state") as state:
            state.return_value.email = ""
            page = app.test_client().get("/activate?next=/")
    body = page.get_data(as_text=True)
    assert 'name="next" value="/dashboard"' in body


def test_launcher_opens_the_dashboard():
    import inspect

    from app.web import launcher

    assert 'webbrowser.open(f"http://127.0.0.1:{port}/dashboard")' in inspect.getsource(launcher.main)


def test_tour_autostarts_on_first_dashboard_visit_then_never_again():
    server_module._license_ok_cached = True
    c = app.test_client()
    assert "window.T58_TOUR_AUTOSTART = true" in c.get("/dashboard").get_data(as_text=True)
    # Still true until the tour is actually finished/skipped (reload keeps it).
    assert "window.T58_TOUR_AUTOSTART = true" in c.get("/dashboard").get_data(as_text=True)
    r = c.post("/api/onboarding/tour-seen", json={"how": "skipped"})
    assert r.get_json() == {"ok": True}
    assert "window.T58_TOUR_AUTOSTART = false" in c.get("/dashboard").get_data(as_text=True)
    # The Take-the-tour button is always there to replay it.
    assert "data-t58-tour-start" in c.get("/dashboard").get_data(as_text=True)


def test_workflow_cards_link_to_their_real_pages():
    server_module._license_ok_cached = True
    body = app.test_client().get("/dashboard").get_data(as_text=True)
    cards = re.findall(r'<a class="step" href="([^"]+)"><span class="n">(\d\d)</span><span class="t">([^<]+)</span>', body)
    assert [(n, t, h) for h, n, t in cards] == [
        ("01", "Create", "/start-here/create"),
        ("02", "Test", "/full-pipeline"),
        ("03", "Optimize", "/optimize-simple"),
        ("04", "Validate", "/validate-simple"),
        ("05", "Champion", "/champion-simple"),
        ("06", "Forward test", "/forward-test"),
        ("07", "Deploy live", "/deploy-live"),
    ]
    c = app.test_client()
    for href, _n, _t in cards:
        # /deploy-live answers 403 with its "turn on web live deploy" page
        # by design until live deployment is enabled -- still a real page.
        assert c.get(href).status_code in (200, 302, 403), href


def test_tour_steps_point_at_real_sidebar_links():
    from pathlib import Path

    web_dir = Path(server_module.__file__).resolve().parent
    sidebar = (web_dir / "templates" / "_sidebar.html").read_text(encoding="utf-8")
    js = (web_dir / "static" / "t58-tour.js").read_text(encoding="utf-8")
    for href in set(re.findall(r'navLink\("([^"]+)"\)', js)):
        assert f'href="{href}"' in sidebar, href
