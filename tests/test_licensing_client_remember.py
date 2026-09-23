import pytest

from app.licensing import client


@pytest.fixture(autouse=True)
def isolate_license_storage(tmp_path, monkeypatch):
    """Redirects the persistent storage paths away from anything real,
    and resets the module's session-only override before AND after every
    test so tests can't leak state into each other via the module-level
    _session_state/_session_only_active globals."""
    monkeypatch.setattr(client, "get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(client, "_try_keyring", lambda: None)
    client._session_state = None
    client._session_only_active = False
    yield
    client._session_state = None
    client._session_only_active = False


MASTER_KEY = "MASTER-KEY-FOR-TESTS"


@pytest.fixture(autouse=True)
def use_master_key(monkeypatch):
    """Activation goes through the master-key path (no real license
    server contacted) so these tests exercise the remember/session-only
    plumbing in isolation from network/server behavior."""
    import hashlib
    monkeypatch.setattr(client, "_master_key_hash", lambda: hashlib.sha256(MASTER_KEY.encode()).hexdigest())


def test_remember_true_persists_to_disk():
    ok, _msg = client.activate("owen@example.com", MASTER_KEY, remember=True)
    assert ok
    assert client._state_path().exists()

    # A brand-new "process" (module state reset, as if the app relaunched)
    # still finds the activation.
    client._session_state = None
    client._session_only_active = False
    state = client.load_state()
    assert state.email == "owen@example.com"
    assert state.license_key == MASTER_KEY
    ok, _msg = client.validate()
    assert ok


def test_remember_false_does_not_write_to_disk():
    ok, _msg = client.activate("owen@example.com", MASTER_KEY, remember=False)
    assert ok
    assert not client._state_path().exists()
    assert not client._key_fallback_path().exists()


def test_remember_false_still_works_for_rest_of_process():
    ok, _msg = client.activate("owen@example.com", MASTER_KEY, remember=False)
    assert ok
    # validate() re-reads load_state() internally -- must see the
    # in-memory session state, not "not activated".
    ok, msg = client.validate()
    assert ok, msg


def test_remember_false_state_gone_after_simulated_relaunch():
    ok, _msg = client.activate("owen@example.com", MASTER_KEY, remember=False)
    assert ok

    # Simulate the process actually exiting and a new one starting: the
    # module-level session override resets to its defaults.
    client._session_state = None
    client._session_only_active = False

    state = client.load_state()
    assert state.email == ""
    assert state.license_key == ""
    ok, _msg = client.validate()
    assert not ok


def test_remember_default_is_true():
    ok, _msg = client.activate("owen@example.com", MASTER_KEY)
    assert ok
    assert client._state_path().exists()


def test_clear_state_drops_session_only_override_too():
    client.activate("owen@example.com", MASTER_KEY, remember=False)
    assert client._session_only_active
    client.clear_state()
    assert not client._session_only_active
    state = client.load_state()
    assert state.license_key == ""
