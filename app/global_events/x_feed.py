"""X (Twitter) feed for Global Events.

Users add X usernames; this module pulls each account's most recent posts with the
bearer token saved on the API Keys page (X API v2: users/by/username + users/:id/tweets).
Nothing runs unless a token is saved AND at least one username is tracked.

Honest limits: X's API is paid-tier for meaningful read volume, so every failure
(missing token, 401/402/403, 429 rate limit, unknown user) is surfaced per account as a
plain-text reason instead of being swallowed. Results are cached in memory for CACHE_TTL_S
so page polling never burns the user's read quota.
"""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone

from app.data.storage import get_app_base_dir

CACHE_TTL_S = 300.0
MAX_ACCOUNTS = 12
POSTS_PER_ACCOUNT = 5
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")
_LOCK = threading.Lock()
_CACHE: dict = {"t": 0.0, "key": None, "data": None}
_USER_ID_CACHE: dict[str, tuple[str, str]] = {}  # username(lower) -> (id, display name)


def normalize_username(raw: str) -> str | None:
    """'@Foo', 'https://x.com/Foo', 'twitter.com/Foo/status/1' -> 'Foo'; None if invalid."""
    s = (raw or "").strip()
    m = re.match(r"^(?:https?://)?(?:www\.|mobile\.)?(?:x|twitter)\.com/([A-Za-z0-9_]{1,15})(?:[/?#].*)?$", s, re.I)
    if m:
        s = m.group(1)
    s = s.lstrip("@")
    return s if _USERNAME_RE.match(s) else None


def _path():
    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d / "x_accounts.json"


def load_accounts() -> list[str]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        out, seen = [], set()
        for u in data.get("accounts", []):
            n = normalize_username(str(u))
            if n and n.lower() not in seen:
                seen.add(n.lower())
                out.append(n)
        return out[:MAX_ACCOUNTS]
    except Exception:
        return []


def _save(accounts: list[str]) -> None:
    _path().write_text(json.dumps({"accounts": accounts[:MAX_ACCOUNTS]}), encoding="utf-8")
    _invalidate()


def _invalidate() -> None:
    with _LOCK:
        _CACHE.update(t=0.0, key=None, data=None)


def add_account(raw: str) -> tuple[bool, str, list[str]]:
    name = normalize_username(raw)
    accounts = load_accounts()
    if not name:
        return False, "That doesn't look like an X username (letters, numbers, underscore; up to 15).", accounts
    if name.lower() in {a.lower() for a in accounts}:
        return False, f"@{name} is already tracked.", accounts
    if len(accounts) >= MAX_ACCOUNTS:
        return False, f"You can track up to {MAX_ACCOUNTS} accounts.", accounts
    accounts.append(name)
    _save(accounts)
    return True, f"Added @{name}.", accounts


def remove_account(raw: str) -> list[str]:
    name = (normalize_username(raw) or "").lower()
    accounts = [a for a in load_accounts() if a.lower() != name]
    _save(accounts)
    return accounts


def bearer_token() -> str:
    try:
        from app.accounts.api_keys import load_settings
        return (load_settings().x_bearer_token or "").strip()
    except Exception:
        return ""


def _get(path: str, token: str, params: dict | None = None):
    import requests
    return requests.get("https://api.x.com/2" + path, params=params or {}, timeout=(5, 12),
                        headers={"Authorization": f"Bearer {token}", "User-Agent": "T58-GlobalEvents"})


def _explain(resp) -> str:
    code = resp.status_code
    if code == 401:
        return "X rejected the bearer token (401). Re-check it on the API Keys page."
    if code == 402:
        return "Your X API plan has no read credits left (402)."
    if code == 403:
        return "Your X API plan doesn't allow this read (403)."
    if code == 429:
        return "X rate limit reached (429) -- try again in a few minutes."
    return f"X returned HTTP {code}."


def test_token(token: str) -> tuple[bool, str]:
    try:
        r = _get("/users/by/username/X", token)
    except Exception as exc:  # noqa: BLE001
        return False, f"Couldn't reach X: {exc}"
    if r.status_code == 200:
        return True, "X bearer token is valid."
    return False, _explain(r)


def _fetch_account(username: str, token: str) -> tuple[list[dict], str | None]:
    key = username.lower()
    try:
        if key not in _USER_ID_CACHE:
            r = _get(f"/users/by/username/{username}", token)
            if r.status_code != 200:
                return [], _explain(r)
            d = r.json()
            if not d.get("data"):
                return [], f"@{username} wasn't found on X."
            _USER_ID_CACHE[key] = (d["data"]["id"], d["data"].get("name") or username)
        uid, display = _USER_ID_CACHE[key]
        r = _get(f"/users/{uid}/tweets", token, {
            "max_results": 5 if POSTS_PER_ACCOUNT <= 5 else POSTS_PER_ACCOUNT,
            "tweet.fields": "created_at,public_metrics",
            "exclude": "retweets,replies",
        })
        if r.status_code != 200:
            return [], _explain(r)
        posts = []
        for t in (r.json().get("data") or [])[:POSTS_PER_ACCOUNT]:
            posts.append({
                "id": str(t.get("id", "")), "username": username, "name": display,
                "text": str(t.get("text", "")), "created_at": t.get("created_at"),
                "url": f"https://x.com/{username}/status/{t.get('id', '')}",
                "likes": (t.get("public_metrics") or {}).get("like_count"),
            })
        return posts, None
    except Exception as exc:  # noqa: BLE001
        return [], f"Couldn't read @{username}: {exc}"


def feed(force: bool = False) -> dict:
    """{configured, accounts, posts (newest first), errors{username: reason}, fetched_at}."""
    token = bearer_token()
    accounts = load_accounts()
    base = {"configured": bool(token), "accounts": accounts, "posts": [], "errors": {}, "fetched_at": None}
    if not token or not accounts:
        return base
    key = (tuple(a.lower() for a in accounts), bool(token))
    with _LOCK:
        if (not force and _CACHE["data"] is not None and _CACHE["key"] == key
                and time.monotonic() - _CACHE["t"] < CACHE_TTL_S):
            return _CACHE["data"]
    posts, errors = [], {}
    for a in accounts:
        got, err = _fetch_account(a, token)
        posts.extend(got)
        if err:
            errors[a] = err
    posts.sort(key=lambda p: p.get("created_at") or "", reverse=True)
    base.update(posts=posts[:40], errors=errors, fetched_at=datetime.now(timezone.utc).isoformat())
    with _LOCK:
        _CACHE.update(t=time.monotonic(), key=key, data=base)
    return base
