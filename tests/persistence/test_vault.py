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
