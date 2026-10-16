"""v9.14 WS-2: the license screen is BACK on first launch.

Owen's explicit reversal of the v9.9 source-build bypass: a fresh
install with no valid stored activation must see the Activate T58
screen (desktop) / redirect to /activate (web) on every build --
with or without a license server URL -- and a master key must
activate fully offline, persist across relaunch, and reject wrong
keys. Everything here runs through the REAL client mechanism
(activate()/validate() against isolated tmp storage); the only
substitution is a test master key supplied via the documented
T58_MASTER_LICENSE_KEY_HASH env override, exactly what the fixtures
in tests/conftest.py do for the whole suite.
"""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

TEST_KEY = "T58-TESTK-EY42"


def _hash(key: str) -> str:
    return hashlib.sha256(key.strip().upper().encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def fresh_license_state(monkeypatch, tmp_path):
    """A brand-new install: empty (tmp) app-data dir, no keyring, no
    server URL, no master-key env override, no cached anything."""
    from app.licensing import client

    monkeypatch.setattr(client, "get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(client, "_try_keyring", lambda: None)
    monkeypatch.delenv("T58_LICENSE_SERVER_URL", raising=False)
    monkeypatch.delenv("T58_MASTER_LICENSE_KEY_HASH", raising=False)
    client._session_state = None
    client._session_only_active = False
    # The suite-wide conftest fixture activates a fixture key into this
    # same tmp dir first -- wipe it so these tests really start from a
    # brand-new install.
    client.clear_state()
    yield
    client._session_state = None
    client._session_only_active = False


def _use_test_master_key(monkeypatch):
    monkeypatch.setenv("T58_MASTER_LICENSE_KEY_HASH", _hash(TEST_KEY))


# ----------------------------------------------------------------------
# Desktop gate (app.licensing.gate.ensure_licensed)
# ----------------------------------------------------------------------

def test_fresh_install_blocks_headless_launch(monkeypatch, capsys):
    from app.licensing import gate

    assert gate.ensure_licensed(interactive=False) is False
    out = capsys.readouterr().out
    assert "master key" in out and "offline" in out  # says the one way in


def test_first_launch_window_gets_offline_note(monkeypatch):
    from app.licensing import gate

    seen = {}

    def fake_window(**kw):
        seen.update(kw)
        return False  # user closed it without activating

    monkeypatch.setattr(gate, "show_activation_window", fake_window)
    assert gate.ensure_licensed(interactive=True) is False
    assert "OFFLINE" in seen["initial_info"] and "master key" in seen["initial_info"]
    assert seen["initial_message"] == ""  # first run: no red error greeting


def test_master_key_activates_offline_and_survives_relaunch(monkeypatch):
    from app.licensing import client, gate

    _use_test_master_key(monkeypatch)

    def fake_window(**kw):
        # Exactly what the real window does on Activate.
        ok, _msg = client.activate("owner@example.com", TEST_KEY)
        return ok

    monkeypatch.setattr(gate, "show_activation_window", fake_window)
    assert gate.ensure_licensed(interactive=True) is True

    # Relaunch: brand-new in-process state (nothing session-only), the
    # persisted activation must validate silently with no window.
    monkeypatch.setattr(gate, "show_activation_window", lambda **kw: pytest.fail("window must not reappear"))
    assert gate.ensure_licensed(interactive=True) is True
    assert gate.ensure_licensed(interactive=False) is True


def test_master_key_persists_via_stored_state(monkeypatch):
    from app.licensing import client

    _use_test_master_key(monkeypatch)
    ok, msg = client.activate("owner@example.com", TEST_KEY)
    assert ok, msg
    # No server configured anywhere -- activation never touched one.
    assert client._state_path().exists()
    ok, msg = client.validate()
    assert ok and "master key" in msg


class _RejectingHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (http.server API)
        body = json.dumps({"ok": False, "error": "not_found"}).encode()
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture
def rejecting_server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _RejectingHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("T58_LICENSE_SERVER_URL", f"http://127.0.0.1:{srv.server_port}")
    yield srv
    srv.shutdown()


def test_wrong_key_rejected_by_server(monkeypatch, rejecting_server):
    from app.licensing import client

    _use_test_master_key(monkeypatch)
    ok, msg = client.activate("owner@example.com", "T58-WRON9-XXXX")
    assert not ok and "wasn't found" in msg
    assert not client.load_state().license_key


def test_wrong_key_rejected_with_no_server_configured(monkeypatch):
    from app.licensing import client

    _use_test_master_key(monkeypatch)
    ok, msg = client.activate("owner@example.com", "T58-WRON9-XXXX")
    assert not ok  # only the master key can succeed offline
    assert not client.load_state().license_key


# ----------------------------------------------------------------------
# Master key format (v9.14: shortened)
# ----------------------------------------------------------------------

def test_shorter_master_key_shape_and_hash():
    from app.licensing import client, gate

    # Placeholder shows the shape the owner actually types now.
    assert gate.KEY_PLACEHOLDER == "T58-XXXXX-XXXX"
    assert len(gate.KEY_PLACEHOLDER) <= 16
    h = client._DEFAULT_MASTER_KEY_HASH
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    assert client.master_key_configured()


# ----------------------------------------------------------------------
# Web gate (app.web.server._license_gate) -- first launch end to end
# ----------------------------------------------------------------------

@pytest.fixture
def web():
    from app.web import server as server_module

    server_module._license_ok_cached = None
    yield server_module
    server_module._license_ok_cached = None


def test_web_first_launch_redirects_then_activates_with_master_key(monkeypatch, web):
    from flask import Flask  # noqa: F401  (web app already built by server module)

    _use_test_master_key(monkeypatch)
    c = web.app.test_client()

    r = c.get("/dashboard", follow_redirects=False)
    assert r.status_code in (301, 302, 303, 307, 308)
    assert "/activate" in r.headers["Location"]

    assert c.get("/activate").status_code == 200

    r = c.post("/activate/submit", data={
        "email": "owner@example.com", "license_key": TEST_KEY, "next": "/dashboard",
    }, follow_redirects=False)
    assert r.status_code in (301, 302, 303, 307, 308)
    assert r.headers["Location"] == "/dashboard"
    assert c.get("/dashboard").status_code == 200

    # Relaunch (fresh process cache): stored activation still validates.
    web._license_ok_cached = None
    assert c.get("/dashboard").status_code == 200


def test_web_wrong_key_stays_locked(monkeypatch, web):
    _use_test_master_key(monkeypatch)
    c = web.app.test_client()
    r = c.post("/activate/submit", data={
        "email": "owner@example.com", "license_key": "T58-WRON9-XXXX", "next": "/dashboard",
    })
    assert r.status_code == 401
    assert web._license_ok_cached is not True
    r = c.get("/dashboard", follow_redirects=False)
    assert r.status_code in (301, 302, 303, 307, 308)


def test_web_unlicensed_post_is_401_json(monkeypatch, web):
    _use_test_master_key(monkeypatch)
    c = web.app.test_client()
    r = c.post("/settings/api-keys/test", data={"service": "fred"})
    assert r.status_code == 401
    assert r.get_json()["error"] == "not_licensed"
