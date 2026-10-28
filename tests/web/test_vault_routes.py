"""Saved-runs card + /api/vault/* routes (local layer; account sync is covered in tests/persistence)."""
import io

import pytest

from app.persistence import vault


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("T58_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("T58_VAULT_AUTO", "0")
    base = tmp_path / "app"
    (base / "reports").mkdir(parents=True)
    (base / "reports" / "a.json").write_text("{}")
    monkeypatch.setattr(vault, "_base", lambda: base)
    # v10: nothing is filed until someone has signed in (the license key is the vault's identity)
    monkeypatch.setattr(vault, "_credentials", lambda: ("owen@example.com", "T58-KEY", "dev1"))
    monkeypatch.setattr(vault, "account_email", lambda: "owen@example.com")
    from app.web import server
    server._license_ok_cached = True
    server.app.config["TESTING"] = True
    c = server.app.test_client()
    with c.session_transaction() as s:
        s["csrf_token"] = "tok"
    return c, base


def _post(c, path, **kw):
    return c.post(path, headers={"X-CSRF-Token": "tok"}, **kw)


def test_status_backup_export_import(client):
    c, base = client
    st = c.get("/api/vault/status").get_json()
    assert st["ok"] and st["app_files"] == 1 and st["cloud_available"] is False
    assert _post(c, "/api/vault/backup").get_json()["local"]["copied"] == 1
    zbytes = c.get("/api/vault/export").data
    (base / "reports" / "a.json").unlink()
    r = _post(c, "/api/vault/import", data={"file": (io.BytesIO(zbytes), "x.zip")}, content_type="multipart/form-data").get_json()
    assert r["ok"] and r["restored"] == 1 and (base / "reports" / "a.json").exists()
    (base / "reports" / "a.json").unlink()
    assert _post(c, "/api/vault/restore", json={}).get_json()["local"]["restored"] == 1


def test_cloud_switch_refused_without_account(client):
    c, _ = client
    r = _post(c, "/api/vault/cloud/settings", json={"cloud_auto": True})
    assert r.status_code == 400 and not r.get_json()["ok"]


def test_settings_page_has_saved_runs_card(client):
    c, _ = client
    html = c.get("/settings/account").get_data(as_text=True)
    assert 'id="saved-runs"' in html and "/api/vault/export" in html
