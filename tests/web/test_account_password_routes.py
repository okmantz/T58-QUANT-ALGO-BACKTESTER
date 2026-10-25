"""Regression: /settings/account/change_password and /settings/account/delete.

Before the fix, both routes sat inside the lock gate's blanket "/settings/account"
exemption and never checked the current password, so anyone who could reach the
URL could wipe the lock with no credentials:

    curl -X POST http://HOST:PORT/settings/account/change_password -d new_password= -d confirm_password=
    curl -X POST http://HOST:PORT/settings/account/delete

`app.test_client(use_cookies=False)` is exactly that: every request is cookieless,
like a bare curl invocation.
"""
from __future__ import annotations

import pytest

from app.accounts import settings as account_settings
from app.web.server import app

CHANGE = "/settings/account/change_password"
DELETE = "/settings/account/delete"


@pytest.fixture(autouse=True)
def _isolated_account_file(monkeypatch, tmp_path):
    monkeypatch.setattr(account_settings, "get_app_base_dir", lambda: tmp_path)
    yield


def _curl():
    """A cookieless client -- what `curl -X POST ... -d ...` looks like to the server."""
    return app.test_client(use_cookies=False)


def _owner():
    """A browser that keeps its session cookie."""
    return app.test_client()


def _set_lock(password="s3cret-pass"):
    """First-time set (no password yet): open by design."""
    c = _owner()
    r = c.post(CHANGE, data={"new_password": password, "confirm_password": password})
    assert r.status_code == 200
    assert account_settings.load_account_settings().has_password
    return c


def test_first_time_set_is_open_and_stores_a_verifiable_hash():
    c = _set_lock("first-pass")
    s = account_settings.load_account_settings()
    assert account_settings.verify_password("first-pass", s.password_hash)
    # that browser is unlocked; a cookieless one is not
    assert c.get("/settings/account").status_code == 200
    assert _curl().get("/dashboard").status_code == 302


def test_first_time_set_rejects_empty_and_mismatched_passwords():
    c = _owner()
    r = c.post(CHANGE, data={"new_password": "", "confirm_password": ""})
    assert r.status_code == 400
    assert not account_settings.load_account_settings().has_password
    r = c.post(CHANGE, data={"new_password": "abc", "confirm_password": "xyz"})
    assert r.status_code == 400
    assert not account_settings.load_account_settings().has_password


def test_curl_cannot_clear_the_lock_without_a_cookie_or_password():
    _set_lock()
    before = account_settings.load_account_settings().password_hash
    curl = _curl()
    for data in (
        {"new_password": "", "confirm_password": ""},                      # the original exploit
        {"new_password": "evil", "confirm_password": "evil"},              # overwrite with attacker's password
        {"current_password": "wrong", "new_password": "", "confirm_password": ""},
    ):
        r = curl.post(CHANGE, data=data)
        assert r.status_code == 401, data
        assert r.get_json()["error"] == "locked"
        assert account_settings.load_account_settings().password_hash == before


def test_curl_cannot_delete_the_account_without_a_cookie_or_password():
    c = _set_lock()
    c.post("/settings/account/save", data={"display_name": "Owen", "username": "o", "email": "o@x.co", "company": "T58"})
    before = account_settings.load_account_settings()
    r = _curl().post(DELETE)
    assert r.status_code == 401
    r = _curl().post(DELETE, data={"current_password": "s3cret-pass"})
    assert r.status_code == 401  # even the right password is useless without an unlocked session
    after = account_settings.load_account_settings()
    assert after.password_hash == before.password_hash and after.display_name == "Owen"


def test_trailing_slash_does_not_bypass_the_gate():
    _set_lock()
    r = _curl().post(CHANGE + "/", data={"new_password": "", "confirm_password": ""})
    assert r.status_code in (401, 404, 405)
    assert account_settings.load_account_settings().has_password


def test_unlocked_session_still_needs_the_current_password_to_change():
    c = _set_lock()
    before = account_settings.load_account_settings().password_hash
    for data in (
        {"new_password": "new", "confirm_password": "new"},                                  # missing
        {"current_password": "nope", "new_password": "new", "confirm_password": "new"},      # wrong
        {"new_password": "", "confirm_password": ""},                                        # empty new, missing current
        {"current_password": "nope", "new_password": "", "confirm_password": ""},            # empty new, wrong current
    ):
        r = c.post(CHANGE, data=data)
        assert r.status_code == 403, data
        assert account_settings.load_account_settings().password_hash == before


def test_unlocked_session_still_needs_the_current_password_to_delete():
    c = _set_lock()
    c.post("/settings/account/save", data={"display_name": "Owen"})
    for data in ({}, {"current_password": "nope"}):
        r = c.post(DELETE, data=data)
        assert r.status_code == 403, data
        s = account_settings.load_account_settings()
        assert s.has_password and s.display_name == "Owen"
    assert c.get("/dashboard").status_code == 200  # still unlocked: a failed delete doesn't log out


def test_correct_current_password_allows_change_clear_and_delete():
    c = _set_lock("old-pass")
    # change
    r = c.post(CHANGE, data={"current_password": "old-pass", "new_password": "new-pass", "confirm_password": "new-pass"})
    assert r.status_code == 200
    s = account_settings.load_account_settings()
    assert account_settings.verify_password("new-pass", s.password_hash)
    assert not account_settings.verify_password("old-pass", s.password_hash)
    # mismatch is still rejected even with the right current password
    r = c.post(CHANGE, data={"current_password": "new-pass", "new_password": "a", "confirm_password": "b"})
    assert r.status_code == 400
    assert account_settings.verify_password("new-pass", account_settings.load_account_settings().password_hash)
    # clear (empty new password, current verified)
    r = c.post(CHANGE, data={"current_password": "new-pass", "new_password": "", "confirm_password": ""})
    assert r.status_code == 200
    assert not account_settings.load_account_settings().has_password
    # lock gone: cookieless access works again
    assert _curl().get("/dashboard").status_code == 200


def test_delete_with_the_current_password_wipes_profile_and_lock():
    c = _set_lock("pw")
    c.post("/settings/account/save", data={"display_name": "Owen", "email": "o@x.co"})
    r = c.post(DELETE, data={"current_password": "pw"})
    assert r.status_code == 200
    s = account_settings.load_account_settings()
    assert not s.has_password and s.display_name == "" and s.email == ""


def test_delete_with_no_password_set_is_still_allowed():
    c = _owner()
    c.post("/settings/account/save", data={"display_name": "Owen"})
    assert c.post(DELETE).status_code == 200
    assert account_settings.load_account_settings().display_name == ""


def test_security_form_shows_current_password_only_when_a_lock_exists():
    c = _owner()
    assert 'name="current_password"' not in c.get("/settings/account").get_data(as_text=True)
    c = _set_lock()
    assert 'name="current_password"' in c.get("/settings/account").get_data(as_text=True)
