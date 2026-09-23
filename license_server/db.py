"""
SQLite storage for the T58 license server. One file, no external database
needed -- this is meant to run on the cheapest possible always-on box (a
free-tier Render/Fly/PythonAnywhere instance, or your own always-on
machine), not a managed database service.

Schema is intentionally small: one table, one row per license key. There
is no "users" table -- a license is identified by its key, tied to one
email address and (after first activation) one device.
"""
from __future__ import annotations

import os
import secrets
import sqlite3
import string
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(os.environ.get("T58_LICENSE_DB_PATH", str(Path(__file__).parent / "licenses.db")))

VALID_STATUSES = ("active", "expired", "revoked", "suspended")

SCHEMA = """
CREATE TABLE IF NOT EXISTS licenses (
    license_key TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    device_id TEXT,
    whop_membership_id TEXT,
    plan TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    last_validated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_licenses_email ON licenses(email);
CREATE INDEX IF NOT EXISTS idx_licenses_whop_membership_id ON licenses(whop_membership_id);
"""


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_license_key() -> str:
    """T58-XXXX-XXXX-XXXX-XXXX, uppercase alphanumeric (no 0/O/1/I to avoid
    transcription mistakes when a customer types it in by hand)."""
    alphabet = "".join(c for c in (string.ascii_uppercase + string.digits) if c not in "01OI")
    groups = ["".join(secrets.choice(alphabet) for _ in range(4)) for _ in range(4)]
    return "T58-" + "-".join(groups)


def create_license(email: str, plan: str = "", days: int | None = None,
                    whop_membership_id: str | None = None) -> dict:
    key = generate_license_key()
    expires_at = None
    if days is not None:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO licenses (license_key, email, status, plan, created_at, expires_at, whop_membership_id) "
            "VALUES (?, ?, 'active', ?, ?, ?, ?)",
            (key, email.strip().lower(), plan, _now_iso(), expires_at, whop_membership_id),
        )
    return get_license(key)


def get_license(license_key: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM licenses WHERE license_key = ?", (license_key,)).fetchone()
        return dict(row) if row else None


def find_by_email(email: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM licenses WHERE email = ? ORDER BY created_at DESC", (email.strip().lower(),)
        ).fetchall()
        return [dict(r) for r in rows]


def find_by_whop_membership_id(whop_membership_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM licenses WHERE whop_membership_id = ?", (whop_membership_id,)
        ).fetchone()
        return dict(row) if row else None


def list_licenses(limit: int = 200) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM licenses ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def set_status(license_key: str, status: str) -> dict | None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}.")
    with _connect() as conn:
        conn.execute("UPDATE licenses SET status = ? WHERE license_key = ?", (status, license_key))
    return get_license(license_key)


def extend_license(license_key: str, days: int) -> dict | None:
    lic = get_license(license_key)
    if lic is None:
        return None
    base = datetime.now(timezone.utc)
    if lic["expires_at"]:
        try:
            existing = datetime.fromisoformat(lic["expires_at"])
            if existing > base:
                base = existing
        except ValueError:
            pass
    new_expiry = (base + timedelta(days=days)).isoformat()
    with _connect() as conn:
        conn.execute("UPDATE licenses SET expires_at = ? WHERE license_key = ?", (new_expiry, license_key))
    return get_license(license_key)


def bind_device(license_key: str, device_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE licenses SET device_id = ?, last_validated_at = ? WHERE license_key = ?",
            (device_id, _now_iso(), license_key),
        )


def clear_device(license_key: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE licenses SET device_id = NULL WHERE license_key = ?", (license_key,))


def touch_validated(license_key: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE licenses SET last_validated_at = ? WHERE license_key = ?", (_now_iso(), license_key))


def mark_expired_if_past_due(lic: dict) -> dict:
    """Lazily flips a license to 'expired' the first time anything reads
    it past its expires_at, rather than needing a scheduled job. Safe to
    call on every read -- a no-op for licenses with no fixed expiry
    (expires_at is None, meaning access is controlled entirely by the
    Whop subscription webhook instead) or that aren't currently active."""
    if lic["status"] != "active" or not lic["expires_at"]:
        return lic
    try:
        expires = datetime.fromisoformat(lic["expires_at"])
    except ValueError:
        return lic
    if expires <= datetime.now(timezone.utc):
        return set_status(lic["license_key"], "expired")
    return lic
