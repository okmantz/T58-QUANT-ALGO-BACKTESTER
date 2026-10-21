"""Serverless allowlist licensing (client.py "Allowlist mode" + build_tools/license_keys.py).

Keys are valid iff the SHA-256 of the key is on a published text file. Revoking
= deleting the line. These tests run the real client against a real local HTTP
server so the whole activate -> validate -> revoke -> offline-grace path is
exercised, not just mocks. FAKE keys only.
"""
import http.server
import importlib.util
import socketserver
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.licensing import client

_spec = importlib.util.spec_from_file_location(
    "license_keys_tool", Path(__file__).resolve().parents[2] / "build_tools" / "license_keys.py")
keys_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(keys_tool)

KEY_A = "T58-AAAAA-BBBBB-CCCCC-DDDDD"
KEY_B = "T58-EEEEE-FFFFF-GGGGG-HHHHH"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(client, "_try_keyring", lambda: None)
    for var in ("T58_LICENSE_ALLOWLIST_URL", "T58_LICENSE_SERVER_URL"):
        monkeypatch.delenv(var, raising=False)
    client._session_state = None
    client._session_only_active = False
    yield
    client._session_state = None
    client._session_only_active = False


class _Server:
    """Tiny local static server whose file content the test can change or kill."""

    def __init__(self):
        outer = self
        self.body = ""
        self.status = 200

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(outer.status)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(outer.body.encode())

            def log_message(self, *a):
                pass

        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/allowlist.txt"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.stopped = False

    def set_keys(self, *keys):
        self.body = "# header\n" + "\n".join(client.key_hash(k) for k in keys) + "\n"

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch):
    s = _Server()
    s.set_keys(KEY_A, KEY_B)
    monkeypatch.setenv("T58_LICENSE_ALLOWLIST_URL", s.url)
    yield s
    s.stop()


def test_valid_key_activates_and_validates(server):
    ok, msg = client.activate("a@example.com", "  " + KEY_A.lower() + " ")
    assert ok, msg
    ok, msg = client.validate()
    assert ok, msg


def test_unknown_key_rejected(server):
    ok, msg = client.activate("a@example.com", "T58-ZZZZZ-ZZZZZ-ZZZZZ-ZZZZZ")
    assert not ok
    assert "recognized" in msg or "revoked" in msg


def test_revoking_removes_access_on_next_check(server):
    assert client.activate("a@example.com", KEY_A)[0]
    assert client.validate()[0]
    server.set_keys(KEY_B)  # KEY_A's line deleted from the published file
    ok, msg = client.validate()
    assert not ok and "revoked" in msg.lower()
    assert client.load_state().status == "revoked"
    # and it stays dead even if the network then drops (revoked is never forgiven)
    server.stop()
    assert not client.validate()[0]


def test_other_key_unaffected_by_revocation(server):
    server.set_keys(KEY_B)
    assert client.activate("b@example.com", KEY_B)[0]
    assert client.validate()[0]


def test_offline_grace_period_then_lockout(server):
    assert client.activate("a@example.com", KEY_A)[0]
    server.stop()
    ok, msg = client.validate()
    assert ok and "Offline" in msg  # inside the grace window
    state = client.load_state()
    state.last_validated_at = (datetime.now(timezone.utc) - timedelta(days=client.OFFLINE_GRACE_DAYS + 1)).isoformat()
    client.save_state(state)
    ok, msg = client.validate()
    assert not ok and "license list" in msg


def test_http_error_or_empty_file_is_not_a_mass_revocation(server):
    assert client.activate("a@example.com", KEY_A)[0]
    server.status = 404
    ok, msg = client.validate()
    assert ok and "Offline" in msg  # treated as unreachable, not "revoked"
    server.status = 200
    server.body = "# accidentally emptied\n"
    ok, msg = client.validate()
    assert ok and "Offline" in msg
    assert client.load_state().status == "active"


def test_cannot_activate_when_list_unreachable(monkeypatch):
    monkeypatch.setenv("T58_LICENSE_ALLOWLIST_URL", "http://127.0.0.1:9/allowlist.txt")
    ok, msg = client.activate("a@example.com", KEY_A)
    assert not ok and "license list" in msg


def test_session_only_activation_not_persisted(server):
    assert client.activate("a@example.com", KEY_A, remember=False)[0]
    assert client.validate()[0]
    client._session_state = None
    client._session_only_active = False
    assert client.load_state().license_key == ""


def test_deactivate_clears_local_state_without_a_server(server):
    assert client.activate("a@example.com", KEY_A)[0]
    ok, _ = client.deactivate()
    assert ok and client.load_state().license_key == ""


def test_allowlist_mode_counts_as_configured(monkeypatch):
    assert client.check_license_server_configured()[0] is False
    monkeypatch.setenv("T58_LICENSE_ALLOWLIST_URL", "https://example.com/allowlist.txt")
    assert client.check_license_server_configured()[0] is True


def test_parser_ignores_comments_junk_and_bad_hashes():
    good = client.key_hash(KEY_A)
    text = f"# comment\n\n{good}  # trailing note\nnot-a-hash\n{'g' * 64}\n{good.upper()}\n"
    assert client._parse_allowlist(text) == {good}


def test_parser_reads_google_sheet_csv_export():
    a, b, c = client.key_hash(KEY_A), client.key_hash(KEY_B), client.key_hash("T58-IIIII-JJJJJ-KKKKK-LLLLL")
    csv_text = (f"hash,issued_to,status\r\n{a},jane@example.com,\r\n"
                f"\"{b}\",bob@example.com,revoked\r\n{c},,\r\n")
    assert client._parse_allowlist(csv_text) == {a, c}  # the row marked revoked is excluded


# ---------------------------------------------------------------- key tool
def test_key_format_and_uniqueness():
    keys = {keys_tool.make_key() for _ in range(2000)}
    assert len(keys) == 2000
    for k in list(keys)[:50]:
        parts = k.split("-")
        assert parts[0] == "T58" and len(parts) == 5 and all(len(p) == 5 for p in parts[1:])
        assert not set("01OIL") & set("".join(parts[1:]))


def test_tool_hash_matches_client_hash():
    assert keys_tool.key_hash(KEY_A) == client.key_hash(KEY_A)


def test_generate_assign_revoke_roundtrip(tmp_path):
    keys_tool.main(["--dir", str(tmp_path), "generate", "--count", "500"])
    rows = keys_tool.read_rows(tmp_path / "keys_PRIVATE.csv")
    assert len(rows) == 500
    allow = (tmp_path / "allowlist.txt").read_text()
    assert allow.count("\n") >= 500
    # no plaintext key ever appears in the public file
    assert not any(r["key"] in allow for r in rows)

    victim = rows[7]["key"]
    keys_tool.main(["--dir", str(tmp_path), "assign", victim, "jane@example.com"])
    assert keys_tool.key_hash(victim) in (tmp_path / "allowlist.txt").read_text()
    keys_tool.main(["--dir", str(tmp_path), "revoke", victim])
    allow = (tmp_path / "allowlist.txt").read_text()
    assert keys_tool.key_hash(victim) not in allow
    assert keys_tool.key_hash(rows[8]["key"]) in allow

    # generating more later keeps existing keys and adds new ones
    keys_tool.main(["--dir", str(tmp_path), "generate", "--count", "10"])
    assert len(keys_tool.read_rows(tmp_path / "keys_PRIVATE.csv")) == 510
    assert keys_tool.key_hash(victim) not in (tmp_path / "allowlist.txt").read_text()


def test_generated_key_works_end_to_end(tmp_path, monkeypatch):
    keys_tool.main(["--dir", str(tmp_path), "generate", "--count", "5"])
    key = keys_tool.read_rows(tmp_path / "keys_PRIVATE.csv")[0]["key"]
    s = _Server()
    s.body = (tmp_path / "allowlist.txt").read_text()
    monkeypatch.setenv("T58_LICENSE_ALLOWLIST_URL", s.url)
    try:
        assert client.activate("x@example.com", key)[0]
        keys_tool.main(["--dir", str(tmp_path), "revoke", key])
        s.body = (tmp_path / "allowlist.txt").read_text()
        assert not client.validate()[0]
    finally:
        s.stop()
