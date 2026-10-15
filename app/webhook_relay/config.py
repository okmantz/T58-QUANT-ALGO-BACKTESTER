"""
Relay definitions: one relay = one webhook URL + its rules.

Stored as plain JSON in data/config/webhook_relays.json. The token is the
credential embedded in the webhook URL, so it is generated with ``secrets``,
compared in constant time, and only ever shown on the (license-gated,
optionally password-locked) setup page. Broker passwords are NOT stored
here -- a relay only references a saved Deploy account by id.
"""
from __future__ import annotations

import hmac
import json
import re
import secrets
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.data.storage import get_app_base_dir

_LOCK = threading.RLock()
SETTINGS_FILE = "webhook_relays.json"
PUBLIC_URL_FILE = "webhook_public_url.json"


@dataclass
class RelayConfig:
    id: str
    name: str
    token: str
    enabled: bool = True
    passphrase: str = ""                 # optional second factor inside the alert body
    default_qty: float = 1.0             # used when the alert has no quantity
    force_default_qty: bool = False      # ignore the alert's quantity, always trade default_qty
    max_qty: float = 10.0                # hard cap per position, whatever the alert says
    allow_short: bool = True
    allowed_tickers: list = field(default_factory=list)   # [] = any; else root symbols, e.g. ["NQ", "ES"]
    symbol_map: dict = field(default_factory=dict)        # {"NQ": "NQZ6"} root -> broker symbol
    dedupe_seconds: float = 3.0          # identical alert inside this window is ignored
    broker_account_id: str = ""          # "" = paper only; else a saved Deploy account id (DEMO only)
    confirm_demo: bool = False           # user attests the broker account is demo/practice (needed when the platform can't tell us)
    created_at: str = ""

    def to_public(self) -> dict:
        d = asdict(self)
        d["has_passphrase"] = bool(self.passphrase)
        return d


def _dir() -> Path:
    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path() -> Path:
    return _dir() / SETTINGS_FILE


def _read() -> list[dict]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001 -- missing/corrupt file = no relays yet
        return []


def _write(rows: list[dict]) -> None:
    path = _path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp.replace(path)


def _from_row(row: dict) -> RelayConfig:
    known = {f for f in RelayConfig.__dataclass_fields__}
    return RelayConfig(**{k: v for k, v in row.items() if k in known})


def list_relays() -> list[RelayConfig]:
    with _LOCK:
        return [_from_row(r) for r in _read() if r.get("id") and r.get("token")]


def get_relay(relay_id: str) -> RelayConfig | None:
    return next((r for r in list_relays() if r.id == relay_id), None)


def find_by_token(token: str) -> RelayConfig | None:
    """Constant-time token lookup (compares against every relay so timing
    does not reveal which prefix matched)."""
    found = None
    for relay in list_relays():
        if hmac.compare_digest(relay.token.encode(), str(token or "").encode()):
            found = relay
    return found


def _clean(values: dict, base: RelayConfig | None = None) -> dict:
    out: dict = {}
    if "name" in values:
        out["name"] = (str(values["name"]).strip() or (base.name if base else "TradingView relay"))[:60]
    if "passphrase" in values:
        out["passphrase"] = str(values["passphrase"]).strip()[:64]
    for key, lo, hi in (("default_qty", 0.0001, 1_000_000), ("max_qty", 0.0001, 1_000_000), ("dedupe_seconds", 0, 3600)):
        if key in values and str(values[key]).strip() != "":
            try:
                out[key] = min(max(float(values[key]), lo), hi)
            except (TypeError, ValueError):
                pass
    for key in ("enabled", "allow_short", "force_default_qty", "confirm_demo"):
        if key in values:
            out[key] = str(values[key]).lower() in ("1", "true", "on", "yes")
    if "allowed_tickers" in values:
        raw = values["allowed_tickers"]
        parts = raw if isinstance(raw, list) else re.split(r"[,\s]+", str(raw))
        out["allowed_tickers"] = [root_symbol(p) for p in parts if str(p).strip()]
    if "symbol_map" in values:
        raw = values["symbol_map"]
        mapping: dict = {}
        if isinstance(raw, dict):
            items = raw.items()
        else:
            items = [tuple(x.split("=", 1)) for x in re.split(r"[\n,;]+", str(raw)) if "=" in x]
        for k, v in items:
            if str(k).strip() and str(v).strip():
                mapping[root_symbol(k)] = str(v).strip()
        out["symbol_map"] = mapping
    if "broker_account_id" in values:
        out["broker_account_id"] = str(values["broker_account_id"]).strip()
    return out


def create_relay(values: dict | None = None) -> RelayConfig:
    with _LOCK:
        relay = RelayConfig(
            id=uuid.uuid4().hex[:12], name="TradingView relay", token=secrets.token_urlsafe(24),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        for k, v in _clean(values or {}, relay).items():
            setattr(relay, k, v)
        rows = _read()
        rows.append(asdict(relay))
        _write(rows)
        return relay


def update_relay(relay_id: str, values: dict) -> RelayConfig | None:
    with _LOCK:
        rows = _read()
        for i, row in enumerate(rows):
            if row.get("id") == relay_id:
                relay = _from_row(row)
                for k, v in _clean(values, relay).items():
                    setattr(relay, k, v)
                rows[i] = asdict(relay)
                _write(rows)
                return relay
    return None


def rotate_token(relay_id: str) -> RelayConfig | None:
    with _LOCK:
        rows = _read()
        for i, row in enumerate(rows):
            if row.get("id") == relay_id:
                row["token"] = secrets.token_urlsafe(24)
                rows[i] = row
                _write(rows)
                return _from_row(row)
    return None


def delete_relay(relay_id: str) -> bool:
    with _LOCK:
        rows = _read()
        kept = [r for r in rows if r.get("id") != relay_id]
        if len(kept) == len(rows):
            return False
        _write(kept)
        return True


def root_symbol(ticker: str) -> str:
    """"CME_MINI:NQ1!" -> "NQ"; "MNQZ2026" -> "MNQ"; "EURUSD" -> "EURUSD"."""
    from app.data.instrument_specs import get_instrument_spec

    text = str(ticker or "").strip().upper()
    if ":" in text:
        text = text.split(":", 1)[1]
    text = text.rstrip("!")
    plain = re.sub(r"\d+$", "", text)                        # NQ1 -> NQ
    if get_instrument_spec(plain) is not None:
        return plain
    m = re.fullmatch(r"([A-Z0-9]{1,5}?)([FGHJKMNQUVXZ])(\d{1,4})", text)   # NQZ26 / ESH6
    if m and get_instrument_spec(m.group(1)) is not None:
        return m.group(1)
    return plain or text


# ---- public base URL (the address TradingView can reach) ---------------------

def get_public_url() -> str:
    import os

    env = os.environ.get("T58_PUBLIC_URL", "").strip().rstrip("/")
    if env:
        return env
    try:
        data = json.loads((_dir() / PUBLIC_URL_FILE).read_text(encoding="utf-8"))
        return str(data.get("url", "")).strip().rstrip("/")
    except Exception:  # noqa: BLE001
        return ""


def _is_private_host(host: str) -> bool:
    """True for addresses TradingView's servers can never reach: localhost,
    LAN ranges, and Tailscale's 100.64.0.0/10 (a private VPN range)."""
    import ipaddress

    h = (host or "").strip().lower().strip("[]")
    if h in ("localhost", "0.0.0.0"):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return h == "localhost" or h.endswith(".localhost") or "." not in h
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_unspecified
                or ip in ipaddress.ip_network("100.64.0.0/10"))


def check_public_url(url: str) -> tuple[str, str]:
    """Validate and normalise the address TradingView will call.

    Returns ``(clean_url, problem)``; ``problem`` is "" when the address is
    usable. TradingView only calls http on port 80 and https on port 443,
    and its servers sit on the public internet, so ``127.0.0.1:5004`` or a
    Tailscale ``100.x.x.x:5004`` address can never work.
    """
    from urllib.parse import urlsplit

    raw = (url or "").strip()
    if not raw:
        return "", "Enter the address first, like https://abc123.ngrok-free.app"
    if re.search(r"\s", raw):
        return "", "That doesn't look like a web address. Paste the https:// address your tunnel printed."
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:
        return "", "That doesn't look like a web address. Paste the https:// address your tunnel printed."
    host = parts.hostname or ""
    if parts.scheme not in ("http", "https") or not host:
        return "", "Paste the full https:// address your tunnel printed, like https://abc123.ngrok-free.app"
    if _is_private_host(host):
        return "", (f"{host} is only reachable from your own network, and TradingView's servers can't call it. "
                    "Use the public https:// address from a tunnel (ngrok, Cloudflare Tunnel or Tailscale Funnel).")
    if port not in (None, 80, 443):
        return "", (f"TradingView only allows port 80 (http) or 443 (https), so :{port} will be rejected. "
                    "Remove the port: a tunnel's https:// address already uses 443 and forwards to this app for you.")
    if parts.scheme == "http" and port not in (None, 80):
        return "", "Use the https:// address from your tunnel."
    return f"{parts.scheme}://{parts.netloc}", ""


def public_url_problem() -> str:
    """Why the currently saved/configured public address won't work with
    TradingView ("" when it is fine, or "Not set" text when absent)."""
    url = get_public_url()
    if not url:
        return "Not set yet."
    return check_public_url(url)[1]


def set_public_url(url: str) -> str:
    url = (url or "").strip()
    if not url:                      # allow clearing it
        (_dir() / PUBLIC_URL_FILE).write_text(json.dumps({"url": ""}), encoding="utf-8")
        return ""
    clean, problem = check_public_url(url)
    if problem:
        raise ValueError(problem)
    (_dir() / PUBLIC_URL_FILE).write_text(json.dumps({"url": clean}), encoding="utf-8")
    return clean
