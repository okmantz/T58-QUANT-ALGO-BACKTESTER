"""Run vault -- keeps your runs, graveyard, journal, saved strategies AND your profile across an uninstall/reinstall.

WHY: a packaged (PyInstaller) build keeps its data NEXT TO THE EXECUTABLE (app.data.storage.get_app_base_dir), so uninstalling
the app deleted every report, the trial ledger, the graveyard and the pipeline state with it. The vault copies that state
somewhere an uninstall does not touch, and can also push it to your account on the license server so it follows you across
reinstalls (and machines).

IDENTITY (v10): everything is filed under your LICENSE KEY (the key is the stable identity; the master key included). The
email you signed in with is a second lookup, so vaults written by older builds (filed by email only) are still found. Nothing
is restored until you sign in, and a sign-in with ANY kind of key (master, sheet, allowlist, server) now triggers the restore --
previously only a license-server sign-in did, which is why a master-key reinstall came back empty.

Layers, all additive (nothing here ever deletes your data):

  1. LOCAL VAULT   <per-user data dir>/vault/<account>/files/...   an incremental mirror, refreshed in the background.
                   The per-user dir (%LOCALAPPDATA% / ~/Library/Application Support / ~/.local/share) is outside the install
                   folder, so it survives uninstalling. After you sign in, anything missing from the app folder (or smaller than
                   the vault's copy, e.g. a half-empty graveyard on a fresh install) is restored from here.
  2. ACCOUNT SYNC  the same files as ONE zip, uploaded to the license server keyed by your email + license key (+ device).
                   On by default whenever the license server can hold it (never for the offline master key / sheet mode).
                   Needs the server to have /v1/vault endpoints (license_server/app.py) and persistent disk.
  3. EXPORT/IMPORT a zip you download/upload by hand.

What is saved:
  * runs and research: reports/, data/run_history.json, data/evolution (graveyard + results), data/discovery,
    data/discovery_engine, data/journal, data/ai_memory, data/projects, data/forward_test, strategies/
  * your profile: account_settings.json (name, username, email, company, local lock password HASH, profile-picture name),
    the profile picture itself, achievements, morning-briefing settings, and data/ui_prefs.json (background, theme, accent
    colour, verse bar, playground scores... everything the browser used to keep only in localStorage)
  * your secrets, ENCRYPTED with a key derived from your license key (PBKDF2 + HMAC): API keys (data providers, Claude, OpenAI,
    X), the Ollama key, Alpaca keys, MT5 login, notification credentials and saved live-broker accounts. They are written
    only to the sealed t58_secrets.enc (never as plain files) and are restored only into fields that are still blank.
NEVER saved: the license state itself, logs, caches, and data/raw (market-data files are big and re-importable).
"""
from __future__ import annotations

import base64
import fnmatch
import hashlib
import hmac
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
    "reports", "strategies", "data/evolution", "data/discovery", "data/discovery_engine", "data/journal", "data/ai_memory",
    "data/projects", "data/forward_test", "data/config",
)
# Single files (relative to the app base dir) that live outside the directories above.
INCLUDE_FILES = (
    "account_settings.json", "data/run_history.json", "data/achievements.json", "data/briefing.json", "data/ui_prefs.json", "data/inbox.json",
    "data/library/paper_state.json", "data/library/focus_sessions.json",
)
# The small "profile" subset: copied far more often than the (potentially large) runs, so a name, a picture, a colour or an
# API key is never more than a few seconds behind.
PROFILE_FILES = INCLUDE_FILES + ("data/config",)
# Where the sealed secrets live inside the vault (outside files/ so they can never be mistaken for app files) and inside a zip.
SECRETS_NAME = "t58_secrets.enc"
ZIP_SECRETS_PATH = "t58_profile/" + SECRETS_NAME
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


def _key_id(license_key: str) -> str:
    return "k" + hashlib.sha256(license_key.strip().upper().encode("utf-8")).hexdigest()[:19]


def _email_id(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:20]


def account_id(email: str | None = None) -> str:
    """The folder name this install files its vault under. The LICENSE KEY is the identity (stable, and the only thing the
    master key has); the email is used only when there is no key (and, via candidate_ids, to find older email-keyed vaults)."""
    if email is None:
        key = _credentials()[1]
        if key:
            return _key_id(key)
        email = account_email()
    email = (email or "").strip().lower()
    return _email_id(email) if email else "anonymous"


def candidate_ids() -> list[str]:
    """Every folder that may hold THIS account's vault, best first: by license key, then by email (older builds)."""
    out: list[str] = []
    email, key, _dev = _credentials()
    if key:
        out.append(_key_id(key))
    for em in (email, account_email()):
        if em and _email_id(em) not in out:
            out.append(_email_id(em))
    return out


def has_identity() -> bool:
    """True once the person has signed in (a license key or email is known). Before that there is nothing to file under."""
    return bool(_credentials()[1] or account_email())


def account_dir(email: str | None = None) -> Path:
    return vault_root() / account_id(email)


def files_dir(email: str | None = None) -> Path:
    return account_dir(email) / "files"


# ------------------------------------------------------------------------------------------------ settings
def _settings_path() -> Path:
    # Lives in the vault (not data/config) so the choice itself survives a reinstall.
    return vault_root() / "settings.json"


def load_settings() -> dict:
    out = {"local_auto": True, "cloud_auto": True, "last_local_backup": None, "last_cloud_push": None,
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


def _walk_dir(base: Path, rel_dir: str):
    root = base / rel_dir
    if not root.is_dir():
        return
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


def _walk_file(base: Path, rel: str):
    p = base / rel
    try:
        if p.is_file() and not p.is_symlink() and p.stat().st_size <= MAX_FILE_BYTES:
            yield rel, p
    except OSError:
        return


def iter_state_files(base: Path | None = None, profile_only: bool = False):
    """Yield (relative posix path, absolute Path) for every file the vault would carry."""
    base = base or _base()
    files = PROFILE_FILES if profile_only else INCLUDE_FILES + INCLUDE_DIRS
    for rel in files:
        if rel in INCLUDE_FILES:
            yield from _walk_file(base, rel)
        else:
            yield from _walk_dir(base, rel)


def _allowed_rel(rel: str) -> bool:
    """Used on anything coming OUT of a zip: an included file or inside an included dir, no traversal, not an excluded name."""
    if not rel or rel.startswith("/") or "\\" in rel or ".." in rel.split("/") or ":" in rel.split("/")[0]:
        return False
    if rel in INCLUDE_FILES:
        return True
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


# ------------------------------------------------------------------------------------------ sealed secrets
_MAGIC = b"T58S1"
_KDF_ITERATIONS = 150_000


def _derive(passphrase: str, salt: bytes) -> tuple[bytes, bytes]:
    km = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, _KDF_ITERATIONS, dklen=64)
    return km[:32], km[32:]


def _keystream_xor(enc_key: bytes, nonce: bytes, data: bytes) -> bytes:
    out = bytearray(len(data))
    for block in range((len(data) + 31) // 32):
        pad = hmac.new(enc_key, nonce + block.to_bytes(8, "big"), hashlib.sha256).digest()
        start = block * 32
        chunk = data[start:start + 32]
        for i, b in enumerate(chunk):
            out[start + i] = b ^ pad[i]
    return bytes(out)


def seal(plain: bytes, passphrase: str) -> bytes:
    """Encrypt-then-MAC with a key stretched from `passphrase` (the license key). Stdlib only."""
    salt, nonce = os.urandom(16), os.urandom(16)
    enc_key, mac_key = _derive(passphrase, salt)
    ct = _keystream_xor(enc_key, nonce, plain)
    tag = hmac.new(mac_key, _MAGIC + salt + nonce + ct, hashlib.sha256).digest()
    return _MAGIC + salt + nonce + tag + ct


def unseal(blob: bytes, passphrase: str) -> bytes | None:
    """Returns the plaintext, or None when the blob is damaged or the passphrase (license key) is wrong."""
    try:
        if not blob.startswith(_MAGIC) or len(blob) < len(_MAGIC) + 16 + 16 + 32:
            return None
        o = len(_MAGIC)
        salt, nonce, tag, ct = blob[o:o + 16], blob[o + 16:o + 32], blob[o + 32:o + 64], blob[o + 64:]
        enc_key, mac_key = _derive(passphrase, salt)
        if not hmac.compare_digest(tag, hmac.new(mac_key, _MAGIC + salt + nonce + ct, hashlib.sha256).digest()):
            return None
        return _keystream_xor(enc_key, nonce, ct)
    except Exception:  # noqa: BLE001
        return None


def _is_set(v) -> bool:
    return v not in (None, "", 0, False, [], {})


def _merge_blank(current: dict, saved: dict, defaults: dict) -> dict:
    """Fields the person has not set on this install (still at their default) take the saved value; set fields are never replaced."""
    out = dict(current)
    for k, v in saved.items():
        if k in defaults and current.get(k, defaults[k]) == defaults[k] and v != defaults[k]:
            out[k] = v
    return out


def collect_secrets() -> dict:
    """Everything secret the person has entered, read through each module's own loader (keyring or fallback file alike)."""
    from dataclasses import asdict
    bundle: dict = {"version": 1, "created": _now()}
    email, _key, _dev = _credentials()
    bundle["email"] = email

    def grab(name, fn):
        try:
            val = fn()
            if val:
                bundle[name] = val
        except Exception as exc:  # noqa: BLE001
            logger.debug("vault: could not read %s secrets: %s", name, exc)

    def api_keys():
        from app.accounts import api_keys as m
        return {k: v for k, v in asdict(m.load_settings()).items() if v}

    def ollama():
        from app.ai import ollama_settings as m
        cur = asdict(m.load_settings())
        return cur if cur != asdict(m.OllamaSettings()) else {}

    def alpaca():
        from app.data import alpaca_credentials as m
        c = m.load_credentials()
        return {"api_key": c.api_key, "secret_key": c.secret_key} if c and c.is_usable else {}

    def mt5():
        from app.forward_test import mt5_settings as m
        cur = asdict(m.load_settings())
        return cur if cur != asdict(m.MT5Settings()) else {}

    def notify():
        from app.web import notifications as m
        cur = asdict(m.load_notification_settings())
        return cur if cur != asdict(m.NotificationSettings()) else {}

    def live():
        from app.live_deploy import live_settings as m
        return [asdict(a) for a in m.load_accounts()]

    grab("api_keys", api_keys)
    grab("ollama", ollama)
    grab("alpaca", alpaca)
    grab("mt5", mt5)
    grab("notifications", notify)
    grab("live_accounts", live)
    return bundle


def apply_secrets(bundle: dict) -> dict:
    """Write a collected bundle back, filling ONLY fields that are still blank on this install."""
    from dataclasses import asdict
    applied: list[str] = []

    def attempt(name, fn):
        if name not in bundle:
            return
        try:
            if fn(bundle[name]):
                applied.append(name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("vault: could not restore %s secrets: %s", name, exc)

    def api_keys(saved):
        from app.accounts import api_keys as m
        cur = m.load_settings()
        changed = False
        for k, v in saved.items():
            if hasattr(cur, k) and v and not getattr(cur, k):
                setattr(cur, k, v)
                changed = True
        if changed:
            m.save_settings(cur)
        return changed

    def ollama(saved):
        from app.ai import ollama_settings as m
        cur = m.load_settings()
        merged = _merge_blank(asdict(cur), saved, asdict(m.OllamaSettings()))
        if merged == asdict(cur):
            return False
        m.save_settings(m.OllamaSettings(**{k: v for k, v in merged.items() if k in asdict(cur)}))
        return True

    def alpaca(saved):
        from app.data import alpaca_credentials as m
        cur = m.load_credentials()
        if cur and cur.is_usable:
            return False
        m.save_credentials(saved.get("api_key", ""), saved.get("secret_key", ""))
        return True

    def mt5(saved):
        from app.forward_test import mt5_settings as m
        cur = m.load_settings()
        merged = _merge_blank(asdict(cur), saved, asdict(m.MT5Settings()))
        if merged == asdict(cur):
            return False
        m.save_settings(m.MT5Settings(**{k: v for k, v in merged.items() if k in asdict(cur)}))
        return True

    def notify(saved):
        from app.web import notifications as m
        cur = m.load_notification_settings()
        merged = _merge_blank(asdict(cur), saved, asdict(m.NotificationSettings()))
        if merged == asdict(cur):
            return False
        m.save_notification_settings(m.NotificationSettings(**{k: v for k, v in merged.items() if k in asdict(cur)}))
        return True

    def live(saved):
        from app.live_deploy import live_settings as m
        have = {a.id for a in m.load_accounts()}
        n = 0
        for raw in saved:
            if raw.get("id") and raw["id"] not in have:
                m.save_account(m.LiveAccount(**{k: v for k, v in raw.items() if k in m.LiveAccount.__dataclass_fields__}))
                n += 1
        return n > 0

    attempt("api_keys", api_keys)
    attempt("ollama", ollama)
    attempt("alpaca", alpaca)
    attempt("mt5", mt5)
    attempt("notifications", notify)
    attempt("live_accounts", live)
    return {"applied": applied}


_LAST_SECRETS_DIGEST: dict[str, str] = {}


def _secrets_passphrase() -> str:
    key = _credentials()[1]
    return key.strip().upper()


def write_sealed_secrets(dest: Path) -> bool:
    """(Re)write the sealed secrets file for the signed-in key, only when something changed. Returns True if written."""
    phrase = _secrets_passphrase()
    if not phrase:
        return False
    bundle = collect_secrets()
    payload = {k: v for k, v in bundle.items() if k not in ("created",)}
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    if len(payload) <= 2:        # nothing but version/email: no secrets to protect yet
        return False
    if dest.exists() and _LAST_SECRETS_DIGEST.get(str(dest)) == digest:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".t58tmp")
    tmp.write_bytes(seal(json.dumps(bundle).encode("utf-8"), phrase))
    os.replace(tmp, dest)
    _LAST_SECRETS_DIGEST[str(dest)] = digest
    return True


def _apply_sealed_blob(blob: bytes) -> dict:
    phrase = _secrets_passphrase()
    if not phrase:
        return {"applied": []}
    plain = unseal(blob, phrase)
    if plain is None:
        return {"applied": [], "error": "wrong_key"}
    try:
        return apply_secrets(json.loads(plain.decode("utf-8")))
    except Exception:  # noqa: BLE001
        return {"applied": []}


# ------------------------------------------------------------------------------------------ local vault
def backup_local(profile_only: bool = False) -> dict:
    """Incrementally mirror the app's state into the local vault. Returns {copied, skipped, bytes, files}.
    Does nothing until the person has signed in (there is nothing to file it under)."""
    with _LOCK:
        if not has_identity():
            return {"copied": 0, "unchanged": 0, "bytes": 0, "files": 0, "skipped": "not_signed_in"}
        base = _base()
        dest_root = files_dir()
        copied = unchanged = nbytes = 0
        for rel, src in iter_state_files(base, profile_only=profile_only):
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
        try:
            write_sealed_secrets(dest_root.parent / SECRETS_NAME)
        except Exception as exc:  # noqa: BLE001
            logger.debug("vault: sealed secrets not written: %s", exc)
        _write_meta(dest_root.parent)
        if profile_only:
            save_settings(last_error="")
        else:
            save_settings(last_local_backup=_now(), last_error="")
        return {"copied": copied, "unchanged": unchanged, "bytes": nbytes, "files": copied + unchanged}


def _write_meta(adir: Path) -> None:
    try:
        adir.mkdir(parents=True, exist_ok=True)
        (adir / "meta.json").write_text(json.dumps({"email_hash": adir.name, "updated": _now()}), encoding="utf-8")
    except OSError:
        pass


def restore_local(email: str | None = None) -> dict:
    """Copy missing/smaller files from the local vault back into the app folder, then re-apply the sealed secrets.
    Safe to call any time; needs a signed-in identity (a license key or email) to know which vault is yours."""
    with _LOCK:
        base = _base()
        restored = skipped = 0
        ids = [account_id(email)] if email else candidate_ids()
        secrets_applied: list[str] = []
        for aid in ids:
            src_root = vault_root() / aid / "files"
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
            sealed = vault_root() / aid / SECRETS_NAME
            if sealed.is_file():
                try:
                    secrets_applied += _apply_sealed_blob(sealed.read_bytes()).get("applied", [])
                except OSError:
                    pass
        if restored or secrets_applied:
            _after_restore()
            save_settings(last_restore=_now())
        return {"restored": restored, "skipped": skipped, "secrets": sorted(set(secrets_applied))}


def _after_restore() -> None:
    """Make an already-running server notice restored settings (caches) without a restart."""
    try:
        from app.ai import ollama_settings
        ollama_settings._invalidate_cache()
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.data.storage import invalidate_dataset_scan_cache
        invalidate_dataset_scan_cache()
    except Exception:  # noqa: BLE001
        pass


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
def build_zip(fileobj=None, profile_only: bool = False) -> tuple[io.BytesIO | None, dict]:
    """Zip the current state (plus the sealed secrets). Returns (buffer, stats); buffer is None when fileobj was supplied."""
    buf = fileobj if fileobj is not None else io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel, p in iter_state_files(profile_only=profile_only):
            try:
                zf.write(p, rel)
                n += 1
            except OSError:
                continue
        phrase = _secrets_passphrase()
        if phrase:
            try:
                bundle = collect_secrets()
                if len(bundle) > 3:
                    zf.writestr(ZIP_SECRETS_PATH, seal(json.dumps(bundle).encode("utf-8"), phrase))
                    n += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("vault: secrets not added to zip: %s", exc)
        zf.writestr("t58_vault_manifest.json", json.dumps(
            {"app": "T58", "version": 2, "created": _now(), "files": n, "profile_only": bool(profile_only)}))
    if fileobj is None:
        buf.seek(0)
        return buf, {"files": n, "bytes": buf.getbuffer().nbytes}
    return None, {"files": n}


def restore_zip(fileobj, mode: str = "merge") -> dict:
    """Restore from a vault zip. mode='merge' follows the missing/smaller rule; 'overwrite' replaces everything it contains."""
    with _LOCK:
        base = _base()
        restored = skipped = rejected = 0
        secrets_applied: list[str] = []
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
                if rel == ZIP_SECRETS_PATH:
                    if info.file_size <= 4 * 1024 * 1024:
                        secrets_applied += _apply_sealed_blob(zf.read(info)).get("applied", [])
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
        if restored or secrets_applied:
            _after_restore()
            save_settings(last_restore=_now())
        return {"ok": True, "restored": restored, "skipped": skipped, "rejected": rejected,
                "secrets": sorted(set(secrets_applied))}


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
    with _LOCK:
        buf, stats = build_zip()
        raw = buf.getvalue()
        profile_only = False
        if len(raw) > MAX_ZIP_BYTES:
            # Too many runs for one snapshot: still keep the account itself (profile, preferences, keys) safe.
            buf, stats = build_zip(profile_only=True)
            raw = buf.getvalue()
            profile_only = True
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
    return {"ok": True, "files": stats["files"], "bytes": len(raw), "profile_only": profile_only}


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
def _restore_everything() -> dict:
    """Bring this account's saved state back: local vault first (instant, offline), then the account on the license server."""
    out: dict = {"restore": restore_local()}
    try:
        if load_settings().get("cloud_auto", True) and cloud_available()[0]:
            info = cloud_info()
            if info.get("ok") and info.get("exists"):
                out["cloud"] = cloud_pull()
    except Exception as exc:  # noqa: BLE001
        logger.debug("vault cloud restore: %s", exc)
    return out


def on_startup() -> dict:
    """Run once when the app starts. A returning, signed-in install restores anything a reinstall removed and mirrors current
    state. A brand-new install has no identity yet, so it waits for on_activated() (called right after sign-in)."""
    out: dict = {}
    try:
        if not has_identity():
            return {"waiting_for_sign_in": True}
        _RESTORED_IDS.add(account_id())
        out.update(_restore_everything())
        out["backup"] = backup_local()
    except Exception as exc:  # noqa: BLE001
        logger.warning("vault startup failed: %s", exc)
        save_settings(last_error=str(exc))
    return out


def on_activated() -> dict:
    """Called right after ANY successful sign-in (master, sheet, allowlist or server key) on a possibly fresh install:
    restore this account's saved runs, profile, preferences and sealed secrets, then start mirroring them."""
    out: dict = {}
    try:
        _RESTORED_IDS.add(account_id())
        out = _restore_everything()
        out["backup"] = backup_local(profile_only=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("vault on_activated: %s", exc)
    return out


_PROFILE_TIMER: threading.Timer | None = None
_RESTORED_IDS: set[str] = set()


def schedule_profile_backup(delay: float = 3.0) -> None:
    """Debounced quick copy of the small profile files + sealed secrets, so a change to a name, picture, colour or API key
    is in the vault seconds later (not at the next 5-minute sweep) and survives a crash or a forced close."""
    global _PROFILE_TIMER
    if os.environ.get("T58_VAULT_AUTO", "1") == "0":
        return

    def run():
        try:
            if load_settings().get("local_auto", True):
                backup_local(profile_only=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("vault profile backup: %s", exc)

    with _LOCK:
        if _PROFILE_TIMER is not None:
            _PROFILE_TIMER.cancel()
        _PROFILE_TIMER = threading.Timer(delay, run)
        _PROFILE_TIMER.daemon = True
        _PROFILE_TIMER.start()


def _loop() -> None:
    last_cloud = 0.0
    last_full = 0.0
    last_seen_identity = ""
    while True:
        time.sleep(30)
        try:
            s = load_settings()
            ident = account_id() if has_identity() else ""
            if ident and ident != last_seen_identity:
                # Signed in (or switched accounts) since the last tick: bring this account's state back first.
                last_seen_identity = ident
                if ident not in _RESTORED_IDS:      # in-memory on purpose: a fresh install must restore again
                    _RESTORED_IDS.add(ident)
                    _restore_everything()
            if not ident:
                continue
            if s.get("local_auto", True):
                backup_local(profile_only=True)
                if time.time() - last_full >= AUTO_INTERVAL_S:
                    backup_local()
                    last_full = time.time()
            if s.get("cloud_auto", True) and time.time() - last_cloud >= CLOUD_INTERVAL_S and cloud_available()[0]:
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
            "account": account_email(), "signed_in": has_identity(), "vault_id": account_id(),
            "cloud_available": cloud_ok, "cloud_reason": cloud_why}
