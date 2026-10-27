"""Run vault -- keeps your runs, graveyard, journal and saved strategies across an uninstall/reinstall.

WHY: a packaged (PyInstaller) build keeps its data NEXT TO THE EXECUTABLE (app.data.storage.get_app_base_dir), so uninstalling
the app deleted every report, the trial ledger, the graveyard and the pipeline state with it. The vault copies that state
somewhere an uninstall does not touch, and can also push it to your account on the license server so it follows you across
reinstalls (and machines).

Three layers, all optional and all additive (nothing here ever deletes your data):

  1. LOCAL VAULT   <per-user data dir>/vault/<account>/files/...   an incremental mirror, refreshed in the background.
                   The per-user dir (%LOCALAPPDATA% / ~/Library/Application Support / ~/.local/share) is outside the install
                   folder, so it survives uninstalling. On launch, anything missing from the app folder (or smaller than the
                   vault's copy, e.g. a half-empty graveyard on a fresh install) is restored from here.
  2. ACCOUNT SYNC  the same files as ONE zip, uploaded to the license server keyed by your email (+ checked against your
                   license key and device). Off until you switch it on in Account settings. Needs the server to have
                   /v1/vault endpoints (license_server/app.py) and persistent disk.
  3. EXPORT/IMPORT a zip you download/upload by hand.

What is saved: reports/, data/evolution (graveyard + results), data/discovery, data/journal, data/ai_memory, data/projects,
data/forward_test, strategies/, and the non-secret files of data/config. NEVER saved: license key/state, API keys, broker
credentials, passwords, bot tokens, logs, caches, and data/raw (your market-data files are big and re-importable).
"""
from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import logging
import os
import shutil
import threading
import time
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Relative to the app base dir. Directories are walked recursively.
INCLUDE_DIRS = (
    "reports", "strategies", "data/evolution", "data/discovery", "data/journal", "data/ai_memory",
    "data/projects", "data/forward_test", "data/config",
)
# Matched against the lowercase file NAME (any directory). Secrets and throw-away files stay out of every copy.
EXCLUDE_NAME_PATTERNS = (
    "license_*", "*api_key*", "alpaca_credentials*", "mt5_*", "ollama_settings*", "notification_*", "*password*",
    "*secret*", "*token*", "*.key", "*.pem", "*.env", ".env*", "row_count_cache*", "*.tmp", "*.lock", "*.pyc",
    "*.db-wal", "*.db-shm", "*-wal", "*-shm", ".t58_write_test", "thumbs.db", ".ds_store", "*.log",
)
MAX_FILE_BYTES = 150 * 1024 * 1024         # a single file above this is skipped (and reported)
MAX_ZIP_BYTES = 90 * 1024 * 1024           # account-sync snapshot cap (server enforces its own too)
AUTO_INTERVAL_S = 300                      # background local refresh
CLOUD_INTERVAL_S = 900                     # background account push (when switched on)
_LOCK = threading.RLock()
_thread: threading.Thread | None = None


# --------------------------------------------------------------------------------------------- locations
def _base() -> Path:
    from app.data.storage import get_app_base_dir
    return get_app_base_dir()


def vault_root() -> Path:
    override = os.environ.get("T58_VAULT_DIR")
    if override:
        return Path(override)
    from app.data.storage import user_data_dir
    return user_data_dir() / "vault"


def _credentials() -> tuple[str, str, str]:
    """(email, license_key, device_id) of this install, or blanks."""
    try:
        from app.licensing import client
        st = client.load_state()
        return (st.email or "").strip(), (st.license_key or "").strip(), (st.device_id or "").strip()
    except Exception:  # noqa: BLE001
        return "", "", ""


def account_email() -> str:
    email = _credentials()[0]
    if not email:
        try:
            from app.accounts.settings import load_account_settings
            email = (load_account_settings().email or "").strip()
        except Exception:  # noqa: BLE001
            email = ""
    return email.lower()


def account_id(email: str | None = None) -> str:
    email = account_email() if email is None else email.strip().lower()
    return hashlib.sha256(email.encode("utf-8")).hexdigest()[:20] if email else "anonymous"


def account_dir(email: str | None = None) -> Path:
    return vault_root() / account_id(email)


def files_dir(email: str | None = None) -> Path:
    return account_dir(email) / "files"


# ------------------------------------------------------------------------------------------------ settings
def _settings_path() -> Path:
    # Lives in the vault (not data/config) so the choice itself survives a reinstall.
    return vault_root() / "settings.json"


def load_settings() -> dict:
    out = {"local_auto": True, "cloud_auto": False, "last_local_backup": None, "last_cloud_push": None,
           "last_cloud_pull": None, "last_restore": None, "last_error": ""}
    try:
        out.update(json.loads(_settings_path().read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        pass
    return out


def save_settings(**changes) -> dict:
    cur = load_settings()
    cur.update(changes)
    try:
        p = _settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("vault settings not saved: %s", exc)
    return cur


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ------------------------------------------------------------------------------------------- what to copy
def _excluded(name: str) -> bool:
    low = name.lower()
    return any(fnmatch.fnmatch(low, pat) for pat in EXCLUDE_NAME_PATTERNS)


def iter_state_files(base: Path | None = None):
    """Yield (relative posix path, absolute Path) for every file the vault would carry."""
    base = base or _base()
    for rel_dir in INCLUDE_DIRS:
        root = base / rel_dir
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", "node_modules")]
            for fn in filenames:
                if _excluded(fn):
                    continue
                p = Path(dirpath) / fn
                try:
                    if not p.is_file() or p.is_symlink() or p.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                yield p.relative_to(base).as_posix(), p


def _allowed_rel(rel: str) -> bool:
    """Used on anything coming OUT of a zip: inside an included dir, no traversal, not an excluded name."""
    if not rel or rel.startswith("/") or "\\" in rel or ".." in rel.split("/") or ":" in rel.split("/")[0]:
        return False
    if not any(rel == d or rel.startswith(d + "/") for d in INCLUDE_DIRS):
        return False
    return not _excluded(rel.rsplit("/", 1)[-1])


def _should_restore(src: Path, dest: Path) -> bool:
    """Restore if the app folder lacks the file, or the saved copy is bigger (an append-only file such as the graveyard
    JSONL, or a state file that was reset by the reinstall). Never replaces a larger/equal local file."""
    try:
        if not dest.exists():
            return True
        return src.stat().st_size > dest.stat().st_size
    except OSError:
        return False


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".t58tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)


# ------------------------------------------------------------------------------------------ local vault
def backup_local() -> dict:
    """Incrementally mirror the app's state into the local vault. Returns {copied, skipped, bytes, files}."""
    with _LOCK:
        base = _base()
        dest_root = files_dir()
        copied = unchanged = nbytes = 0
        for rel, src in iter_state_files(base):
            dest = dest_root / rel
            try:
                s = src.stat()
                if dest.exists():
                    d = dest.stat()
                    if d.st_size == s.st_size and int(d.st_mtime) >= int(s.st_mtime):
                        unchanged += 1
                        continue
                _copy(src, dest)
                copied += 1
                nbytes += s.st_size
            except OSError as exc:
                logger.debug("vault: could not copy %s: %s", rel, exc)
        _write_meta(dest_root.parent)
        save_settings(last_local_backup=_now(), last_error="")
        return {"copied": copied, "unchanged": unchanged, "bytes": nbytes, "files": copied + unchanged}


def _write_meta(adir: Path) -> None:
    try:
        (adir / "meta.json").write_text(json.dumps({"email_hash": adir.name, "updated": _now()}), encoding="utf-8")
    except OSError:
        pass


def restore_local(email: str | None = None) -> dict:
    """Copy missing/smaller files from the local vault back into the app folder. Safe to call any time."""
    with _LOCK:
        base = _base()
        src_root = files_dir(email)
        restored = skipped = 0
        if src_root.is_dir():
            for dirpath, _dirs, filenames in os.walk(src_root):
                for fn in filenames:
                    src = Path(dirpath) / fn
                    rel = src.relative_to(src_root).as_posix()
                    if fn.endswith(".t58tmp") or not _allowed_rel(rel):
                        continue
                    dest = base / rel
                    if _should_restore(src, dest):
                        try:
                            _copy(src, dest)
                            restored += 1
                        except OSError as exc:
                            logger.debug("vault: could not restore %s: %s", rel, exc)
                    else:
                        skipped += 1
        if restored:
            save_settings(last_restore=_now())
        return {"restored": restored, "skipped": skipped}


def local_summary() -> dict:
    root = files_dir()
    n = size = 0
    if root.is_dir():
        for dirpath, _d, fns in os.walk(root):
            for fn in fns:
                n += 1
                try:
                    size += (Path(dirpath) / fn).stat().st_size
                except OSError:
                    pass
    return {"files": n, "bytes": size, "path": str(root)}


# ------------------------------------------------------------------------------------- zip (export / sync)
def build_zip(fileobj=None) -> tuple[io.BytesIO | None, dict]:
    """Zip the current state. Returns (buffer, stats); buffer is None when fileobj was supplied."""
    buf = fileobj if fileobj is not None else io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel, p in iter_state_files():
            try:
                zf.write(p, rel)
                n += 1
            except OSError:
                continue
        zf.writestr("t58_vault_manifest.json", json.dumps({"app": "T58", "version": 1, "created": _now(), "files": n}))
    if fileobj is None:
        buf.seek(0)
        return buf, {"files": n, "bytes": buf.getbuffer().nbytes}
    return None, {"files": n}


def restore_zip(fileobj, mode: str = "merge") -> dict:
    """Restore from a vault zip. mode='merge' follows the missing/smaller rule; 'overwrite' replaces everything it contains."""
    with _LOCK:
        base = _base()
        restored = skipped = rejected = 0
        try:
            zf = zipfile.ZipFile(fileobj)
        except zipfile.BadZipFile:
            return {"ok": False, "error": "That isn't a valid vault zip."}
        with zf:
            total = sum(i.file_size for i in zf.infolist())
            if total > 2 * 1024 * 1024 * 1024:
                return {"ok": False, "error": "That archive is too large to restore."}
            for info in zf.infolist():
                rel = info.filename
                if info.is_dir() or rel == "t58_vault_manifest.json":
                    continue
                if not _allowed_rel(rel) or info.file_size > MAX_FILE_BYTES:
                    rejected += 1
                    continue
                dest = base / rel
                if mode != "overwrite":
                    try:
                        if dest.exists() and info.file_size <= dest.stat().st_size:
                            skipped += 1
                            continue
                    except OSError:
                        pass
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".t58tmp")
                with zf.open(info) as src, open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out)
                os.replace(tmp, dest)
                restored += 1
        if restored:
            save_settings(last_restore=_now())
        return {"ok": True, "restored": restored, "skipped": skipped, "rejected": rejected}


# ------------------------------------------------------------------------------------------ account sync
def _server_base() -> str | None:
    try:
        from app.licensing import client
        if client.resolve_sheet_url():
            return None       # the Google-sheet licence mode has no server to hold files
        return client.resolve_license_server_url()
    except Exception:  # noqa: BLE001
        return None


def cloud_available() -> tuple[bool, str]:
    email, key, dev = _credentials()
    if not (email and key and dev):
        return False, "Sign in with your email and license key first."
    try:
        from app.licensing import client
        if client._is_master_key(key):
            return False, "Master keys have no account on the license server, so account sync isn't available."
    except Exception:  # noqa: BLE001
        pass
    if not _server_base():
        return False, "This build has no license server configured, so account sync isn't available."
    return True, ""


def _cloud_call(path: str, payload: dict, timeout: float = 120.0) -> tuple[bool, dict, str]:
    from urllib import request as urllib_request
    from urllib.error import HTTPError, URLError
    base = _server_base()
    if not base:
        return False, {}, "No license server configured."
    req = urllib_request.Request(base.rstrip("/") + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            return True, json.loads(resp.read().decode("utf-8")), ""
    except HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            body = {"ok": False, "error": f"http_{exc.code}"}
        if exc.code == 404 and body.get("error") in (None, "http_404"):
            body = {"ok": False, "error": "server_too_old"}
        return True, body, ""
    except (URLError, TimeoutError, OSError) as exc:
        return False, {}, f"Couldn't reach the license server ({exc})."


_CLOUD_ERRORS = {
    "server_too_old": "Your license server doesn't have account sync yet -- redeploy license_server/ from this update.",
    "too_large": "Your saved runs are too large for account sync.",
    "device_mismatch": "This license is bound to a different device.",
    "email_mismatch": "That email doesn't match the license.",
    "not_found": "License not found.",
    "no_backup": "There's no backup on your account yet.",
}


def _cloud_err(body: dict) -> str:
    code = body.get("error") or "failed"
    return _CLOUD_ERRORS.get(code, body.get("message") or f"The server said: {code}")


def cloud_push() -> dict:
    ok, why = cloud_available()
    if not ok:
        return {"ok": False, "error": why}
    email, key, dev = _credentials()
    import base64
    with _LOCK:
        buf, stats = build_zip()
    raw = buf.getvalue()
    if len(raw) > MAX_ZIP_BYTES:
        return {"ok": False, "error": _CLOUD_ERRORS["too_large"]}
    digest = hashlib.sha256(raw).hexdigest()
    net, body, err = _cloud_call("/v1/vault/put", {
        "email": email, "license_key": key, "device_id": dev, "sha256": digest,
        "files": stats["files"], "data_b64": base64.b64encode(raw).decode("ascii"),
    })
    if not net:
        save_settings(last_error=err)
        return {"ok": False, "error": err}
    if not body.get("ok"):
        msg = _cloud_err(body)
        save_settings(last_error=msg)
        return {"ok": False, "error": msg}
    save_settings(last_cloud_push=_now(), last_error="")
    return {"ok": True, "files": stats["files"], "bytes": len(raw)}


def cloud_info() -> dict:
    ok, why = cloud_available()
    if not ok:
        return {"ok": False, "error": why}
    email, key, dev = _credentials()
    net, body, err = _cloud_call("/v1/vault/info", {"email": email, "license_key": key, "device_id": dev}, timeout=15)
    if not net:
        return {"ok": False, "error": err}
    if not body.get("ok"):
        return {"ok": False, "error": _cloud_err(body)}
    return {"ok": True, "exists": bool(body.get("exists")), "updated_at": body.get("updated_at"),
            "size": body.get("size"), "files": body.get("files")}


def cloud_pull(mode: str = "merge") -> dict:
    ok, why = cloud_available()
    if not ok:
        return {"ok": False, "error": why}
    email, key, dev = _credentials()
    import base64
    net, body, err = _cloud_call("/v1/vault/get", {"email": email, "license_key": key, "device_id": dev})
    if not net:
        return {"ok": False, "error": err}
    if not body.get("ok"):
        return {"ok": False, "error": _cloud_err(body)}
    try:
        raw = base64.b64decode(body.get("data_b64", ""), validate=True)
    except Exception:  # noqa: BLE001
        return {"ok": False, "error": "The server sent a damaged backup."}
    if body.get("sha256") and hashlib.sha256(raw).hexdigest() != body["sha256"]:
        return {"ok": False, "error": "The downloaded backup failed its integrity check."}
    res = restore_zip(io.BytesIO(raw), mode=mode)
    if res.get("ok"):
        save_settings(last_cloud_pull=_now(), last_error="")
    return res


# -------------------------------------------------------------------------------------------- lifecycle
def on_startup() -> dict:
    """Run once when the app starts: bring back anything a reinstall removed, then mirror current state."""
    out: dict = {}
    try:
        out["restore"] = restore_local()
        s = load_settings()
        if s.get("cloud_auto") and cloud_available()[0]:
            local_has = any(True for _ in iter_state_files())
            if not local_has or not out["restore"].get("restored"):
                info = cloud_info()
                if info.get("ok") and info.get("exists"):
                    out["cloud"] = cloud_pull()
        out["backup"] = backup_local()
    except Exception as exc:  # noqa: BLE001
        logger.warning("vault startup failed: %s", exc)
        save_settings(last_error=str(exc))
    return out


def on_activated() -> None:
    """Called right after a successful sign-in on a (possibly fresh) install: restore this account's saved runs."""
    try:
        restore_local()
        if load_settings().get("cloud_auto") and cloud_available()[0]:
            info = cloud_info()
            if info.get("ok") and info.get("exists"):
                cloud_pull()
    except Exception as exc:  # noqa: BLE001
        logger.debug("vault on_activated: %s", exc)


def _loop() -> None:
    last_cloud = 0.0
    while True:
        time.sleep(AUTO_INTERVAL_S)
        try:
            s = load_settings()
            if s.get("local_auto", True):
                backup_local()
            if s.get("cloud_auto") and time.time() - last_cloud >= CLOUD_INTERVAL_S and cloud_available()[0]:
                cloud_push()
                last_cloud = time.time()
        except Exception as exc:  # noqa: BLE001
            logger.debug("vault loop: %s", exc)


def start_background() -> None:
    """Start the restore-then-mirror worker once (daemon thread). Skipped when T58_VAULT_AUTO=0 (tests)."""
    global _thread
    if os.environ.get("T58_VAULT_AUTO", "1") == "0" or _thread is not None:
        return

    def run():
        on_startup()
        _loop()

    _thread = threading.Thread(target=run, name="t58-vault", daemon=True)
    _thread.start()
    import atexit
    atexit.register(lambda: _safe_final_backup())


def _safe_final_backup() -> None:
    try:
        if load_settings().get("local_auto", True):
            backup_local()
    except Exception:  # noqa: BLE001
        pass


def status() -> dict:
    s = load_settings()
    cloud_ok, cloud_why = cloud_available()
    n = size = 0
    for _rel, p in iter_state_files():
        n += 1
        try:
            size += p.stat().st_size
        except OSError:
            pass
    return {"settings": s, "local": local_summary(), "app_files": n, "app_bytes": size,
            "account": account_email(), "cloud_available": cloud_ok, "cloud_reason": cloud_why}
