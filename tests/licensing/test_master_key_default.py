"""The owner's master key works with zero configuration (Oct 2026).

client._DEFAULT_MASTER_KEY_HASH is a compiled-in fallback so the master
key activates offline in every build -- from-source runs, CI-built .exes,
any machine -- with no env vars and no license server. Precedence stays:
env var > build_config.py (repo-secret bake) > compiled-in default.

These tests use a FAKE key/hash and monkeypatch the constant -- the real
master key and its hash must never appear in the test suite.
"""
import hashlib

import pytest

from app.licensing import client


FAKE_KEY = "T58-TEST-0000-1111-2222-333344445555"


def _fake_hash(key=FAKE_KEY):
    return hashlib.sha256(key.strip().upper().encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(client, "_try_keyring", lambda: None)
    # Simulate a build with no env var and no build_config.py bake.
    monkeypatch.delenv("T58_MASTER_LICENSE_KEY_HASH", raising=False)
    client._session_state = None
    client._session_only_active = False
    yield
    client._session_state = None
    client._session_only_active = False


def test_default_hash_is_valid_sha256_hex():
    h = client._DEFAULT_MASTER_KEY_HASH
    assert isinstance(h, str) and len(h) == 64
    int(h, 16)  # raises if not hex


def test_default_used_when_nothing_else_configured(monkeypatch):
    monkeypatch.setattr(client, "_DEFAULT_MASTER_KEY_HASH", _fake_hash())
    assert client._master_key_hash() == _fake_hash()
    assert client.master_key_configured()
    # Normalization: lowercase + surrounding whitespace still activates.
    ok, msg = client.activate("owner@example.com", "  " + FAKE_KEY.lower() + "  ")
    assert ok, msg
    ok, msg = client.validate()
    assert ok, msg


def test_env_var_beats_default(monkeypatch):
    monkeypatch.setattr(client, "_DEFAULT_MASTER_KEY_HASH", _fake_hash())
    other = _fake_hash("T58-OTHER-AAAA-BBBB-CCCC-DDDDEEEEFFFF")
    monkeypatch.setenv("T58_MASTER_LICENSE_KEY_HASH", other)
    assert client._master_key_hash() == other
    # The default key must NOT activate when the env var points elsewhere.
    ok, _msg = client.activate("owner@example.com", FAKE_KEY)
    assert not ok


def test_wrong_key_rejected_against_default(monkeypatch):
    monkeypatch.setattr(client, "_DEFAULT_MASTER_KEY_HASH", _fake_hash())
    assert not client._is_master_key("T58-WRONG-0000-0000-0000-000000000000")
    assert client._is_master_key(FAKE_KEY)
