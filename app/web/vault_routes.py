"""Account settings -> "Saved runs" controls for the run vault (app.persistence.vault).

  GET  /api/vault/status            JSON: local copy, account sync state, last times
  POST /api/vault/backup            copy current state to the local vault now (+ account, when switched on)
  POST /api/vault/restore           bring back anything missing from the local vault (and the account, when on)
  POST /api/vault/cloud/settings    {cloud_auto: bool}  switch account sync on/off (on = upload right away)
  POST /api/vault/cloud/push|pull   explicit upload / download
  GET  /api/vault/export            download everything as a zip
  POST /api/vault/import            upload a zip made by export
CSRF is enforced globally (X-CSRF-Token header on fetch()).
"""
from __future__ import annotations

import io
import time

from flask import Blueprint, jsonify, request, send_file

from app.persistence import vault

vault_bp = Blueprint("vault", __name__)


def _fail(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


@vault_bp.route("/api/vault/status")
def vault_status():
    out = vault.status()
    out["ok"] = True
    out["cloud"] = vault.cloud_info() if (request.args.get("cloud") == "1" and out["cloud_available"]) else None
    return jsonify(out)


@vault_bp.route("/api/vault/backup", methods=["POST"])
def vault_backup():
    res = {"ok": True, "local": vault.backup_local()}
    if vault.load_settings().get("cloud_auto"):
        res["cloud"] = vault.cloud_push()
    return jsonify(res)


@vault_bp.route("/api/vault/restore", methods=["POST"])
def vault_restore():
    res = {"ok": True, "local": vault.restore_local()}
    if vault.load_settings().get("cloud_auto") or (request.get_json(silent=True) or {}).get("cloud"):
        res["cloud"] = vault.cloud_pull()
    return jsonify(res)


@vault_bp.route("/api/vault/cloud/settings", methods=["POST"])
def vault_cloud_settings():
    want = bool((request.get_json(silent=True) or {}).get("cloud_auto"))
    if want:
        ok, why = vault.cloud_available()
        if not ok:
            return _fail(why)
    vault.save_settings(cloud_auto=want)
    res = {"ok": True, "cloud_auto": want}
    if want:
        res["push"] = vault.cloud_push()
        if not res["push"].get("ok"):
            vault.save_settings(cloud_auto=False)
            return jsonify({"ok": False, "error": res["push"].get("error", "Upload failed."), "cloud_auto": False}), 502
    return jsonify(res)


@vault_bp.route("/api/vault/cloud/push", methods=["POST"])
def vault_cloud_push():
    res = vault.cloud_push()
    return jsonify(res), (200 if res.get("ok") else 502)


@vault_bp.route("/api/vault/cloud/pull", methods=["POST"])
def vault_cloud_pull():
    res = vault.cloud_pull()
    return jsonify(res), (200 if res.get("ok") else 502)


@vault_bp.route("/api/vault/export")
def vault_export():
    buf, _stats = vault.build_zip()
    return send_file(io.BytesIO(buf.getvalue()), mimetype="application/zip", as_attachment=True,
                     download_name=f"t58-saved-runs-{time.strftime('%Y%m%d-%H%M')}.zip")


@vault_bp.route("/api/vault/import", methods=["POST"])
def vault_import():
    f = request.files.get("file")
    if not f:
        return _fail("Choose a vault zip first.")
    res = vault.restore_zip(io.BytesIO(f.read()), mode="merge")
    return jsonify(res), (200 if res.get("ok") else 400)
