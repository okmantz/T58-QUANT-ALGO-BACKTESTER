from unittest.mock import patch

import pytest

from app.licensing import client as license_client
from app.web import server as server_module
from app.web.server import app


@pytest.fixture(autouse=True)
def reset_license_cache(monkeypatch):
    """The gate caches the license check for the life of the process
    (see app.web.server._license_gate's docstring) -- reset that cache
    before and after every test so tests can't leak the cached
    True/False into each other."""
    # v9.14: the web gate engages on first launch on EVERY build with
    # no valid stored activation (the v9.9 no-URL skip was reversed).
    # These tests simulate a configured (distributed) build, so a URL
    # is set anyway; validate()/activate() themselves are patched
    # per-test below and never touch the network.
    monkeypatch.setenv("T58_LICENSE_SERVER_URL", "https://license.example.invalid")
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


def _follow_redirects_manually(client, url, limit=10):
    """Walk the redirect chain by hand so a loop is a test failure instead of a hang."""
    chain = []
    for _ in range(limit):
        r = client.get(url, follow_redirects=False)
        chain.append((url, r.status_code))
        if r.status_code not in (301, 302, 303, 307, 308):
            return r, chain
        url = r.headers["Location"]
    raise AssertionError(f"redirect loop (> {limit} hops): {chain}")


def test_unlicensed_with_a_stale_lock_password_lands_on_activation_not_the_lock_page():
    """A reinstall can leave an old lock password behind. The FIRST screen must still be the license key + email page, never a
    password prompt, and the redirect chain must not loop."""
    from app.accounts.settings import AccountSettings

    locked = AccountSettings(password_hash="pbkdf2$sha256$1000$aa$bb")
    with patch.object(license_client, "validate", return_value=(False, "Not activated.")), \
         patch.object(license_client, "validate_fast_start", return_value=(False, "Not activated.")), \
         patch("app.accounts.settings.load_account_settings", return_value=locked), \
         patch.object(license_client, "load_state") as mock_state:
        mock_state.return_value.email = ""
        c = app.test_client()
        for start in ("/lock?next=/activate", "/activate", "/dashboard", "/"):
            r, chain = _follow_redirects_manually(c, start)
            assert r.status_code == 200, (start, chain)
            body = r.data.lower()
            assert b"license" in body, (start, chain)
            assert b'name="password"' not in body, (start, chain)


def test_first_activation_clears_a_leftover_lock_password(monkeypatch, tmp_path):
    """After the license is accepted on a fresh install, the leftover password must be gone so the app opens straight away."""
    from app.accounts import settings as account_settings

    monkeypatch.setattr(account_settings, "get_app_base_dir", lambda: tmp_path)
    account_settings.save_account_settings(account_settings.AccountSettings(
        display_name="Old", password_hash=account_settings.hash_password("old-secret")))
    assert account_settings.load_account_settings().has_password
    monkeypatch.setattr(license_client, "load_state", lambda: license_client.LicenseState())
    monkeypatch.setattr(license_client, "_activate_impl", lambda *a, **k: (True, "Activated."))
    ok, _msg = license_client.activate("me@example.com", "T58-TEST-KEY")
    assert ok
    after = account_settings.load_account_settings()
    assert not after.has_password
    assert after.display_name == "Old"        # only the lock is dropped, never the profile


def test_reactivating_an_existing_install_keeps_its_lock_password(monkeypatch, tmp_path):
    from app.accounts import settings as account_settings

    monkeypatch.setattr(account_settings, "get_app_base_dir", lambda: tmp_path)
    account_settings.save_account_settings(account_settings.AccountSettings(password_hash=account_settings.hash_password("keep-me")))
    monkeypatch.setattr(license_client, "load_state",
                        lambda: license_client.LicenseState(email="me@example.com", license_key="T58-OLD-KEY"))
    monkeypatch.setattr(license_client, "_activate_impl", lambda *a, **k: (True, "Activated."))
    ok, _msg = license_client.activate("me@example.com", "T58-NEW-KEY")
    assert ok
    assert account_settings.load_account_settings().has_password


def test_licensed_and_locked_still_shows_the_lock_page():
    from app.accounts.settings import AccountSettings

    locked = AccountSettings(password_hash="pbkdf2$sha256$1000$aa$bb")
    with patch.object(license_client, "validate", return_value=(True, "Active.")), \
         patch("app.accounts.settings.load_account_settings", return_value=locked):
        c = app.test_client()
        r, chain = _follow_redirects_manually(c, "/dashboard")
    assert r.status_code == 200 and b'name="password"' in r.data.lower(), chain


def test_lock_next_param_is_same_site_only():
    from app.accounts.settings import AccountSettings

    with patch("app.accounts.settings.load_account_settings", return_value=AccountSettings()):
        server_module._license_ok_cached = True
        c = app.test_client()
        for evil in ("//evil.example", "http://evil.example/x", "/\\evil.example"):
            r = c.get("/lock", query_string={"next": evil}, follow_redirects=False)
            assert r.status_code in (301, 302, 303, 307, 308)
            assert r.headers["Location"] == "/dashboard", evil
