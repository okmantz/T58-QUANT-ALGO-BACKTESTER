"""Run vault: survives a 'reinstall', never copies secrets, never clobbers bigger local files, rejects hostile zips,
and round-trips through the real license-server endpoints."""
import base64
import hashlib
import importlib
import io
import sys
import zipfile
from pathlib import Path

import pytest

from app.persistence import vault


@pytest.fixture
def env(tmp_path, monkeypatch):
    app1 = tmp_path / "install1"
    app1.mkdir()
    monkeypatch.setenv("T58_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("T58_VAULT_AUTO", "0")
    cur = {"base": app1}
    monkeypatch.setattr(vault, "_base", lambda: cur["base"])
    monkeypatch.setattr(vault, "account_email", lambda: "owen@example.com")
    monkeypatch.setattr(vault, "_credentials", lambda: ("owen@example.com", "T58-AAAA", "dev1"))
    return tmp_path, cur


def _seed(base: Path):
    (base / "reports").mkdir(parents=True)
    (base / "reports" / "run1.json").write_text('{"pf": 1.8}')
    (base / "data/evolution").mkdir(parents=True)
    (base / "data/evolution/strategy_graveyard__es__5m.jsonl").write_text("a\nb\nc\n")
    (base / "data/config").mkdir(parents=True)
    (base / "data/config/validation_checklist.json").write_text("{}")
    for secret in ("license_key.txt", "license_state.json", "api_key_openai.txt", "alpaca_credentials.json", "mt5_password.txt"):
        (base / "data/config" / secret).write_text("SECRET")
    (base / "data/raw").mkdir(parents=True)
    (base / "data/raw/big.csv").write_text("x")


def test_reinstall_roundtrip_restores_runs_but_not_secrets(env):
    tmp, cur = env
    _seed(cur["base"])
    res = vault.backup_local()
    assert res["copied"] == 3
    saved = {p.relative_to(vault.files_dir()).as_posix() for p in vault.files_dir().rglob("*") if p.is_file()}
    assert saved == {"reports/run1.json", "data/evolution/strategy_graveyard__es__5m.jsonl", "data/config/validation_checklist.json"}

    fresh = tmp / "install2"          # uninstall + reinstall = a brand-new empty folder
    fresh.mkdir()
    cur["base"] = fresh
    out = vault.restore_local()
    assert out["restored"] == 3
    assert (fresh / "reports/run1.json").read_text() == '{"pf": 1.8}'
    assert not (fresh / "data/config/license_key.txt").exists()


def test_restore_never_replaces_a_larger_local_file(env):
    tmp, cur = env
    _seed(cur["base"])
    vault.backup_local()
    g = cur["base"] / "data/evolution/strategy_graveyard__es__5m.jsonl"
    g.write_text("a\nb\nc\nd\ne\nf\n")              # the user kept working; local is bigger
    vault.restore_local()
    assert g.read_text().count("\n") == 6
    g.write_text("a\n")                              # fresh install made a tiny one -> vault copy wins
    vault.restore_local()
    assert g.read_text().count("\n") == 3


def test_incremental_backup_skips_unchanged(env):
    _tmp, cur = env
    _seed(cur["base"])
    vault.backup_local()
    again = vault.backup_local()
    assert again["copied"] == 0 and again["unchanged"] == 3


def test_accounts_are_separate(env, monkeypatch):
    _tmp, cur = env
    _seed(cur["base"])
    vault.backup_local()
    monkeypatch.setattr(vault, "account_email", lambda: "someone@else.com")
    monkeypatch.setattr(vault, "_credentials", lambda: ("someone@else.com", "T58-OTHER", "dev1"))   # v10: the key is the identity
    assert vault.local_summary()["files"] == 0


def test_zip_export_import_and_hostile_zip(env):
    tmp, cur = env
    _seed(cur["base"])
    buf, stats = vault.build_zip()
    names = zipfile.ZipFile(io.BytesIO(buf.getvalue())).namelist()
    assert "reports/run1.json" in names and not any("license" in n or "password" in n or "api_key" in n for n in names)
    fresh = tmp / "other"; fresh.mkdir(); cur["base"] = fresh
    assert vault.restore_zip(io.BytesIO(buf.getvalue()))["restored"] == 3

    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../escape.txt", "x")
        z.writestr("reports/../../escape2.txt", "x")
        z.writestr("/abs.txt", "x")
        z.writestr("data/config/license_key.txt", "x")
        z.writestr("app/web/server.py", "x")
        z.writestr("reports/ok.json", "{}")
    r = vault.restore_zip(io.BytesIO(evil.getvalue()))
    assert r["restored"] == 1 and r["rejected"] == 5
    assert not (tmp / "escape.txt").exists() and not (fresh / "app").exists()
    assert vault.restore_zip(io.BytesIO(b"nope"))["ok"] is False


def test_cloud_roundtrip_through_license_server(env, tmp_path, monkeypatch):
    tmp, cur = env
    monkeypatch.setenv("T58_LICENSE_DB_PATH", str(tmp / "lic.db"))
    monkeypatch.setenv("T58_VAULT_DIR_SERVER", "x")
    import importlib.util
    ls = Path(__file__).resolve().parents[2] / "license_server"
    saved_db = sys.modules.get("db")

    def _load(name, fname):
        spec = importlib.util.spec_from_file_location(name, ls / fname)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    srv_db = _load("db", "db.py")                  # license_server/app.py does `import db`
    srv_db.VAULT_DIR = tmp / "server_vault"
    srv = _load("t58_license_server_app", "app.py")
    try:
        lic = srv_db.create_license("owen@example.com", plan="Pro")
        key = lic["license_key"]
        client = srv.app.test_client()

        def fake_call(path, payload, timeout=120.0):
            payload = dict(payload, license_key=key, device_id="dev1")
            r = client.post(path, json=payload)
            return True, r.get_json(), ""
        monkeypatch.setattr(vault, "_cloud_call", fake_call)
        monkeypatch.setattr(vault, "cloud_available", lambda: (True, ""))

        _seed(cur["base"])
        assert vault.cloud_info()["exists"] is False
        assert vault.cloud_push()["ok"] is True
        assert vault.cloud_info()["exists"] is True

        fresh = tmp / "reinstalled"; fresh.mkdir(); cur["base"] = fresh
        res = vault.cloud_pull()
        assert res["ok"] and res["restored"] == 3
        assert (fresh / "reports/run1.json").exists()

        # wrong device / bad payload / wrong email are refused by the server
        bad = client.post("/v1/vault/get", json={"email": "owen@example.com", "license_key": key, "device_id": "other"})
        assert bad.status_code in (200, 403)
        srv_db.bind_device(key, "dev1")
        assert client.post("/v1/vault/get", json={"email": "owen@example.com", "license_key": key, "device_id": "other"}).status_code == 403
        assert client.post("/v1/vault/get", json={"email": "evil@example.com", "license_key": key, "device_id": "dev1"}).status_code == 403
        assert client.post("/v1/vault/put", json={"email": "owen@example.com", "license_key": key, "device_id": "dev1",
                                                   "data_b64": base64.b64encode(b"not a zip").decode()}).status_code == 400
    finally:
        sys.modules.pop("t58_license_server_app", None)
        if saved_db is not None:
            sys.modules["db"] = saved_db
        else:
            sys.modules.pop("db", None)


# ---------------------------------------------------------------------------------------------------------------------
# v10: master-key reinstall, profile files, sealed secrets
# ---------------------------------------------------------------------------------------------------------------------
def _seed_profile(base: Path):
    (base / "account_settings.json").write_text('{"display_name": "Owen", "username": "owen", "password_hash": "pbkdf2$x"}')
    (base / "data/config").mkdir(parents=True, exist_ok=True)
    (base / "data/config/profile_picture.png").write_bytes(b"\x89PNG-pretend")
    (base / "data/ui_prefs.json").write_text('{"t58-bg": "midnight-ocean", "t58-accent": "violet"}')
    (base / "data/achievements.json").write_text('{"first_run": 1}')
    (base / "data/run_history.json").write_text('[{"id": 1}, {"id": 2}]')


def test_profile_files_and_run_history_survive_a_reinstall(env):
    tmp, cur = env
    _seed_profile(cur["base"])
    vault.backup_local()
    fresh = tmp / "install2"
    fresh.mkdir()
    cur["base"] = fresh
    out = vault.restore_local()
    assert out["restored"] >= 5
    assert "Owen" in (fresh / "account_settings.json").read_text()
    assert (fresh / "data/config/profile_picture.png").read_bytes() == b"\x89PNG-pretend"
    assert "midnight-ocean" in (fresh / "data/ui_prefs.json").read_text()
    assert (fresh / "data/run_history.json").exists()
    assert (fresh / "data/achievements.json").exists()


def test_vault_is_filed_by_license_key_so_a_master_key_with_a_new_email_still_finds_it(env, monkeypatch):
    tmp, cur = env
    _seed_profile(cur["base"])
    vault.backup_local()
    fresh = tmp / "install2"
    fresh.mkdir()
    cur["base"] = fresh
    # same key, but the person typed a different email this time
    monkeypatch.setattr(vault, "_credentials", lambda: ("other@example.com", "T58-AAAA", "dev9"))
    monkeypatch.setattr(vault, "account_email", lambda: "other@example.com")
    assert vault.restore_local()["restored"] >= 5
    # a DIFFERENT key never sees this vault
    other = tmp / "install3"
    other.mkdir()
    cur["base"] = other
    monkeypatch.setattr(vault, "_credentials", lambda: ("x@example.com", "T58-ZZZZ", "dev9"))
    monkeypatch.setattr(vault, "account_email", lambda: "x@example.com")
    assert vault.restore_local()["restored"] == 0


def test_legacy_email_keyed_vault_is_still_found(env, monkeypatch):
    tmp, cur = env
    _seed_profile(cur["base"])
    legacy = vault.vault_root() / vault._email_id("owen@example.com") / "files"
    (legacy / "data").mkdir(parents=True)
    (legacy / "data/run_history.json").write_text('[{"id": 1}]')
    fresh = tmp / "install2"
    fresh.mkdir()
    cur["base"] = fresh
    assert vault.restore_local()["restored"] == 1
    assert (fresh / "data/run_history.json").exists()


def test_nothing_is_filed_before_sign_in(env, monkeypatch):
    tmp, cur = env
    _seed_profile(cur["base"])
    monkeypatch.setattr(vault, "_credentials", lambda: ("", "", ""))
    monkeypatch.setattr(vault, "account_email", lambda: "")
    assert vault.backup_local()["files"] == 0
    assert not (vault.vault_root() / "anonymous").exists()
    assert vault.on_startup() == {"waiting_for_sign_in": True}


def test_sealed_secrets_roundtrip_and_wrong_key_fails():
    blob = vault.seal(b'{"api_keys": {"fred_api_key": "abc"}}', "T58-AAAA")
    assert b"fred_api_key" not in blob and b"abc" not in blob
    assert vault.unseal(blob, "T58-AAAA") == b'{"api_keys": {"fred_api_key": "abc"}}'
    assert vault.unseal(blob, "T58-BBBB") is None
    assert vault.unseal(blob[:-1] + bytes([blob[-1] ^ 1]), "T58-AAAA") is None     # tampered
    assert vault.unseal(b"garbage", "T58-AAAA") is None


def test_api_keys_are_sealed_not_plain_and_restore_only_into_blank_fields(env, monkeypatch, tmp_path):
    tmp, cur = env
    from app.accounts import api_keys as ak
    monkeypatch.setattr(ak, "_config_dir", lambda: cur["base"] / "data" / "config")
    monkeypatch.setattr(ak, "_try_keyring", lambda: None)
    (cur["base"] / "data/config").mkdir(parents=True, exist_ok=True)
    ak.save_settings(ak.ApiKeysSettings(fred_api_key="FRED-123", openai_api_key="OPENAI-456"))
    vault.backup_local()
    for p in vault.account_dir().rglob("*"):
        if p.is_file():
            assert b"FRED-123" not in p.read_bytes() and b"OPENAI-456" not in p.read_bytes()
    assert (vault.account_dir() / vault.SECRETS_NAME).is_file()

    fresh = tmp / "install2"
    (fresh / "data/config").mkdir(parents=True)
    cur["base"] = fresh
    monkeypatch.setattr(ak, "_config_dir", lambda: fresh / "data" / "config")
    assert ak.load_settings().fred_api_key == ""
    out = vault.restore_local()
    assert "api_keys" in out["secrets"]
    got = ak.load_settings()
    assert got.fred_api_key == "FRED-123" and got.openai_api_key == "OPENAI-456"
    # a value the person already set on the new install is never overwritten
    ak.save_settings(ak.ApiKeysSettings(fred_api_key="NEWER", openai_api_key="OPENAI-456"))
    vault.restore_local()
    assert ak.load_settings().fred_api_key == "NEWER"


def test_zip_carries_sealed_secrets_and_restores_them(env, monkeypatch):
    tmp, cur = env
    from app.accounts import api_keys as ak
    monkeypatch.setattr(ak, "_config_dir", lambda: cur["base"] / "data" / "config")
    monkeypatch.setattr(ak, "_try_keyring", lambda: None)
    (cur["base"] / "data/config").mkdir(parents=True, exist_ok=True)
    ak.save_settings(ak.ApiKeysSettings(claude_api_key="CLAUDE-789"))
    _seed_profile(cur["base"])
    buf, _ = vault.build_zip()
    raw = buf.getvalue()
    assert b"CLAUDE-789" not in raw
    fresh = tmp / "install2"
    (fresh / "data/config").mkdir(parents=True)
    cur["base"] = fresh
    monkeypatch.setattr(ak, "_config_dir", lambda: fresh / "data" / "config")
    res = vault.restore_zip(io.BytesIO(raw))
    assert res["ok"] and "api_keys" in res["secrets"]
    assert ak.load_settings().claude_api_key == "CLAUDE-789"


def test_activate_with_the_master_key_triggers_the_restore(tmp_path, monkeypatch):
    """The bug behind 'I reinstalled and nothing saved': only the license-server path ever called on_activated()."""
    from app.licensing import client
    calls = []
    monkeypatch.setattr(vault, "on_activated", lambda: calls.append(1))
    monkeypatch.setattr(client, "_is_master_key", lambda k: True)
    monkeypatch.setattr(client, "save_state", lambda st: None)
    monkeypatch.setattr(client, "device_id", lambda: "dev")
    ok, _msg = client.activate("owen@example.com", "ANYKEY")
    assert ok and calls == [1]
    calls.clear()
    ok, _ = client.activate("", "")
    assert not ok and calls == []
