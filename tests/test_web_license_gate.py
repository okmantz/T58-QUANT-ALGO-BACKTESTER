from unittest.mock import patch

import pytest

from app.licensing import client as license_client
from app.web import server as server_module
from app.web.server import app


@pytest.fixture(autouse=True)
def reset_license_cache():
    """The gate caches the license check for the life of the process
    (see app.web.server._license_gate's docstring) -- reset that cache
    before and after every test so tests can't leak the cached
    True/False into each other."""
    server_module._license_ok_cached = None
    yield
    server_module._license_ok_cached = None


def test_unlicensed_get_redirects_to_activate_form():
    with patch.object(license_client, "validate", return_value=(False, "Not activated.")):
        c = app.test_client()
        r = c.get("/dashboard", follow_redirects=False)
    assert r.status_code in (301, 302, 303, 307, 308)
    assert "/activate" in r.headers["Location"]


def test_unlicensed_post_returns_401_json_not_html_redirect():
    with patch.object(license_client, "validate", return_value=(False, "This license has been revoked.")):
        c = app.test_client()
        r = c.post("/settings/api-keys/test", data={"service": "fred"})
    assert r.status_code == 401
    assert r.get_json()["error"] == "not_licensed"


def test_activate_form_reachable_even_when_unlicensed():
    with patch.object(license_client, "validate", return_value=(False, "Not activated.")):
        with patch.object(license_client, "load_state") as mock_state:
            mock_state.return_value.email = ""
            c = app.test_client()
            r = c.get("/activate")
    assert r.status_code == 200
    assert b"Activate T58" in r.data


def test_static_assets_reachable_even_when_unlicensed():
    with patch.object(license_client, "validate", return_value=(False, "Not activated.")):
        c = app.test_client()
        r = c.get("/manifest.json")
    assert r.status_code != 401
    assert r.status_code not in (301, 302, 303, 307, 308)


def test_activate_submit_success_unlocks_and_redirects_to_next():
    with patch.object(license_client, "activate", return_value=(True, "Activated.")):
        c = app.test_client()
        r = c.post("/activate/submit", data={
            "email": "owen@example.com", "license_key": "T58-AAAA-BBBB-CCCC-DDDD", "next": "/dashboard",
        }, follow_redirects=False)
    assert r.status_code in (301, 302, 303, 307, 308)
    assert r.headers["Location"] == "/dashboard"
    assert server_module._license_ok_cached is True

    # Cache now set -- a subsequent request should NOT call validate()
    # again (matches the desktop build's "check once at launch" convention).
    with patch.object(license_client, "validate") as mock_validate:
        c.get("/dashboard")
        mock_validate.assert_not_called()


def test_activate_submit_failure_shows_error_and_stays_locked():
    with patch.object(license_client, "activate", return_value=(False, "That license key wasn't found.")):
        c = app.test_client()
        r = c.post("/activate/submit", data={
            "email": "owen@example.com", "license_key": "BAD-KEY", "next": "/dashboard",
        })
    assert r.status_code == 401
    assert b"That license key wasn&#39;t found." in r.data or b"That license key wasn't found." in r.data
    assert server_module._license_ok_cached is None


def test_licensed_request_passes_through_to_account_lock_gate():
    """Once licensed, the request should reach the (separate, optional)
    account-lock gate exactly as before this feature existed -- i.e. the
    license gate must not itself block a normal, licensed request."""
    with patch.object(license_client, "validate", return_value=(True, "Active.")):
        c = app.test_client()
        r = c.get("/dashboard")
    assert r.status_code == 200
