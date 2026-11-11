"""The master key is a clean slate on every uninstall + reinstall (no vault, no half-restored state); every other key keeps its progress."""
import json

import pytest

from app.licensing import fresh_start
from app.persistence import vault


@pytest.fixture
def env(tmp_path, monkeypatch):
    base, ud = tmp_path / "install", tmp_path / "ud"
    (base / "data" / "config").mkdir(parents=True)
    (base / "reports").mkdir()
    (base / "reports" / "run1.json").write_text("{}")
    (base / "data" / "prop_desk").mkdir()
    (base / "data" / "prop_desk" / "desk.json").write_text("{}")
    (base / "data" / "config" / "current_strategy.json").write_text("{}")
    import app.data.storage as storage
    monkeypatch.setattr(storage, "get_app_base_dir", lambda: base)
    monkeypatch.setattr(storage, "user_data_dir", lambda: ud)
    monkeypatch.setattr(vault, "_base", lambda: base)
    monkeypatch.setenv("T58_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("T58_VAULT_AUTO", "0")
    monkeypatch.delenv("T58_NO_FRESH_START", raising=False)
    monkeypatch.setenv("T58_FORCE_FRESH_START", "1")
    return base, ud


def _sign_in(monkeypatch, key, master):
    monkeypatch.setattr(vault, "_credentials", lambda: ("owen@example.com", key, "dev1"))
    monkeypatch.setattr(vault, "account_email", lambda: "owen@example.com")
    from app.licensing import client
    monkeypatch.setattr(client, "_is_master_key", lambda k: master and k == key)


def test_master_key_never_writes_or_reads_the_vault(env, monkeypatch):
    base, _ud = env
    _sign_in(monkeypatch, "T58-OWNER", master=True)
    assert vault.is_master_identity()
    assert vault.backup_local()["skipped"] == "master_key"
    assert not vault.vault_root().exists() or not any(vault.vault_root().rglob("*.json"))
    # even a vault copy left behind by an older build is ignored
    old = vault.vault_root() / vault.account_id() / "files" / "reports"
    old.mkdir(parents=True)
    (old / "ghost.json").write_text("{}")
    (base / "reports" / "run1.json").unlink()
    assert vault.restore_local().get("master_key") is True
    assert not (base / "reports" / "ghost.json").exists()


def test_other_keys_keep_their_progress_filed_under_key_and_email(env, monkeypatch):
    base, _ud = env
    _sign_in(monkeypatch, "T58-CUSTOMER", master=False)
    assert not vault.is_master_identity()
    assert vault.backup_local()["copied"] >= 1
    (base / "reports" / "run1.json").unlink()
    assert vault.restore_local()["restored"] >= 1
    assert (base / "reports" / "run1.json").exists()


def test_reinstall_triggers_a_clean_slate_once(env):
    base, ud = env
    assert fresh_start.run_if_needed() is True
    assert not fresh_start.needed()
    marker = json.loads((ud / fresh_start.MARKER_NAME).read_text())
    assert marker["install_id"] == fresh_start.install_id() != ""
    (base / "reports" / "again.json").write_text("{}")
    assert fresh_start.run_if_needed() is False and (base / "reports" / "again.json").exists()   # normal run: saves normally
    # uninstall + reinstall: the install folder (and its id file) is gone, the per-user marker is not
    (base / fresh_start.INSTALL_ID_NAME).unlink()
    (base / "data" / "prop_desk").mkdir(exist_ok=True)
    (base / "data" / "prop_desk" / "desk.json").write_text('{"accounts": [1]}')
    assert fresh_start.needed()
    assert fresh_start.run_if_needed() is True
    assert not (base / "data" / "prop_desk").exists() and not (base / "reports" / "again.json").exists()
    assert not fresh_start.needed()


def test_clean_slate_covers_prop_desk_and_current_strategy(env):
    base, _ud = env
    fresh_start.run_if_needed(force=True)
    assert not (base / "data" / "prop_desk").exists()
    assert not (base / "data" / "config" / "current_strategy.json").exists()
