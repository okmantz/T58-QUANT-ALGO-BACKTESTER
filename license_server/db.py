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

-- SHARED TRIAL KEYS (2026-09-26): one row per *trial template* -- a
-- single license_key value (e.g. "T58-TRIAL-3DAY") Owen shares with any
-- number of people, unlike the `licenses` table above where one key is
-- permanently tied to one pre-registered email. redeemed_at/expires_at
-- live on the REDEMPTION (trial_redemptions below), not here -- this
-- table only says "this key phrase is a valid trial and grants N days".
CREATE TABLE IF NOT EXISTS trial_keys (
    trial_key TEXT PRIMARY KEY,
    days INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',   -- 'active' | 'disabled' -- Owen can kill a leaked key without touching redemptions
    created_at TEXT NOT NULL
);

-- One row per PERSON who has ever redeemed a trial key -- this is what
-- actually enforces "3 days per person, and you don't get another 3
-- days by changing your email or your device". device_id and email each
-- have a UNIQUE constraint (enforced in redeem_trial below, not just
-- here, so the caller gets a clean "already used" answer instead of a
-- raw IntegrityError): once EITHER value has redeemed once, that same
-- value can never start a second, fresh countdown, even paired with a
-- brand-new value on the other side. ip_address is recorded for Owen's
-- own visibility only (shared/dynamic IPs make it unsafe to hard-block
-- on) -- see redeem_trial's docstring for the exact limitation this
-- does and doesn't cover.
CREATE TABLE IF NOT EXISTS trial_redemptions (
    device_id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    trial_key TEXT NOT NULL,
    ip_address TEXT,
    first_used_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trial_redemptions_email ON trial_redemptions(email);
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


# ---------------------------------------------------------------------------
# Shared trial keys -- see trial_keys/trial_redemptions in SCHEMA above.
# ---------------------------------------------------------------------------

def create_trial_key(trial_key: str, days: int = 3) -> dict:
    """Registers a new shareable trial key phrase, e.g.
    create_trial_key("T58-TRIAL-3DAY", days=3). Raises sqlite3.IntegrityError
    if that exact key phrase is already registered -- catch it and pick a
    different phrase, or use set_trial_key_status to disable the old one
    first if you're deliberately replacing it."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO trial_keys (trial_key, days, status, created_at) VALUES (?, ?, 'active', ?)",
            (trial_key.strip().upper(), days, _now_iso()),
        )
    return get_trial_key(trial_key)


def get_trial_key(trial_key: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM trial_keys WHERE trial_key = ?", (trial_key.strip().upper(),)).fetchone()
        return dict(row) if row else None


def list_trial_keys() -> list[dict]:
    with _connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM trial_keys ORDER BY created_at DESC").fetchall()]


def set_trial_key_status(trial_key: str, status: str) -> dict | None:
    if status not in ("active", "disabled"):
        raise ValueError("status must be 'active' or 'disabled'.")
    with _connect() as conn:
        conn.execute("UPDATE trial_keys SET status = ? WHERE trial_key = ?", (status, trial_key.strip().upper()))
    return get_trial_key(trial_key)


def get_trial_redemption(*, device_id: str | None = None, email: str | None = None) -> dict | None:
    """Look up an existing redemption by EITHER device_id or email (not
    both required) -- used to detect "this device already has a trial
    running" and "this email already has a trial running" as two
    independent checks, since either one alone is enough to say no to a
    fresh countdown."""
    with _connect() as conn:
        if device_id is not None:
            row = conn.execute("SELECT * FROM trial_redemptions WHERE device_id = ?", (device_id,)).fetchone()
            if row:
                return dict(row)
        if email is not None:
            row = conn.execute("SELECT * FROM trial_redemptions WHERE email = ?", (email.strip().lower(),)).fetchone()
            if row:
                return dict(row)
        return None


def list_trial_redemptions(limit: int = 500) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM trial_redemptions ORDER BY first_used_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


class TrialConflict(Exception):
    """Raised by redeem_trial when device_id or email has already been
    used to redeem a DIFFERENT trial pairing. .reason is "device" or
    "email" -- which side of the pairing was the conflict, so the caller
    can return a specific error message."""
    def __init__(self, reason: str, existing: dict):
        self.reason = reason
        self.existing = existing
        super().__init__(f"{reason} already used for a trial")


def redeem_trial(trial_key: str, email: str, device_id: str, ip_address: str | None = None) -> dict:
    """The whole "3-day trial, once per person" rule lives here.

    - Unknown/disabled trial_key -> ValueError.
    - Brand new (device_id, email) pair -> creates a new redemption,
      first_used_at = now, expires_at = now + trial_keys.days. This is
      the ONLY case that starts a fresh countdown.
    - The EXACT SAME (device_id, email) pair returning (the legitimate
      case: the same person checking in again during their own trial, or
      after it's expired) -> returns the EXISTING row unchanged. Never
      extends or resets it -- expiry is judged by the caller against the
      returned expires_at.
    - device_id already redeemed under a DIFFERENT email, or email
      already redeemed under a DIFFERENT device_id -> raises
      TrialConflict. This is what stops "share one key" from becoming
      "unlimited 3-day windows": once either identifier has been used
      once, it can never pair with a different value on the other side
      to start over.

    LIMITATION (matches what Owen already flagged): a person willing to
    use BOTH a fresh device_id AND a fresh email at the same time is not
    caught by this -- device_id is whatever the client sends (a machine-
    generated fingerprint, not something cryptographically hard to
    regenerate on a wiped/reinstalled machine or a second computer), and
    email is self-reported. ip_address is recorded on every redemption
    for Owen's own visibility (e.g. spotting many redemptions from one
    IP) but is deliberately never used to hard-block, since a shared or
    dynamic IP would then lock out legitimate different people.
    """
    tk = get_trial_key(trial_key)
    if tk is None or tk["status"] != "active":
        raise ValueError("Unknown or disabled trial key.")

    email_norm = email.strip().lower()
    existing_by_device = get_trial_redemption(device_id=device_id)
    existing_by_email = get_trial_redemption(email=email_norm)

    if existing_by_device is not None and existing_by_device["email"] == email_norm:
        return existing_by_device  # same person checking in again
    if existing_by_device is not None:
        raise TrialConflict("device", existing_by_device)
    if existing_by_email is not None:
        raise TrialConflict("email", existing_by_email)

    expires_at = (datetime.now(timezone.utc) + timedelta(days=tk["days"])).isoformat()
    now = _now_iso()
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO trial_redemptions (device_id, email, trial_key, ip_address, first_used_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (device_id, email_norm, tk["trial_key"], ip_address, now, expires_at),
            )
    except sqlite3.IntegrityError:
        # Two near-simultaneous redemption attempts for the same brand-new
        # device_id/email raced each other -- re-check rather than treat
        # this as a hard failure. Whichever one actually landed wins;
        # this call just needs to return that same, single row.
        winner = get_trial_redemption(device_id=device_id) or get_trial_redemption(email=email_norm)
        if winner is not None:
            return winner
        raise
    return get_trial_redemption(device_id=device_id)
