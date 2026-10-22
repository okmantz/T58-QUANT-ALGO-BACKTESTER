"""Connect-a-prop-firm flow shared by the web and desktop Deploy Live pages.

One place defines, for every connectable trading platform:

  * which credential fields it needs (and which are secret / required),
  * how to turn typed values into a saved ``LiveAccount``,
  * how to TEST the connection and explain a failure in plain language,
  * a pre-deploy checklist (firm automation policy, rules verified,
    strategy validated, demo-vs-live) so a person sees exactly what they are
    about to do before real capital is on the line.

Flask- and Tk-free on purpose: the web page renders ``PLATFORM_FIELDS`` as
JSON and the desktop builds its form from the same spec, so the two apps
cannot drift apart again (the desktop used to ask for credentials as one
semicolon-separated "key=value" text box; the web form was missing the
live/demo switch entirely).
"""
from __future__ import annotations

from typing import Any, Optional

from app.live_deploy.broker_registry import SUPPORTED_PLATFORMS, build_adapter
from app.live_deploy.live_settings import LiveAccount, normalize_platform
from app.live_deploy import prop_firms

# kind: "text" | "secret" | "select"
# store: "core" = a LiveAccount attribute (login/password/server/terminal_path),
#        "extra" = LiveAccount.extra_credentials[key]
PLATFORM_FIELDS: dict[str, list[dict[str, Any]]] = {
    "MT4/MT5": [
        {"key": "login", "label": "Login / account number", "kind": "text", "required": True, "store": "core"},
        {"key": "password", "label": "Password", "kind": "secret", "required": True, "store": "core"},
        {"key": "server", "label": "Server (from your firm's account email)", "kind": "text", "required": True, "store": "core",
         "help": "Exactly as written in the account email, e.g. FTMO-Server3. Server names differ per account type."},
        {"key": "terminal_path", "label": "Terminal path (only if auto-detect fails)", "kind": "text", "required": False, "store": "core"},
    ],
    "cTrader": [
        {"key": "client_id", "label": "Client ID", "kind": "text", "required": True, "store": "extra",
         "help": "From your cTrader Open API application (openapi.ctrader.com)."},
        {"key": "client_secret", "label": "Client secret", "kind": "secret", "required": True, "store": "extra"},
        {"key": "refresh_token", "label": "Refresh token", "kind": "secret", "required": True, "store": "extra"},
        {"key": "ctid_trader_account_id", "label": "Trader account ID (ctidTraderAccountId)", "kind": "text", "required": True, "store": "extra"},
        {"key": "host", "label": "Environment", "kind": "select", "options": ["demo", "live"], "default": "demo", "required": True, "store": "extra"},
        {"key": "login", "label": "Account label / number (optional)", "kind": "text", "required": False, "store": "core"},
    ],
    "Tradovate": [
        {"key": "login", "label": "Tradovate username", "kind": "text", "required": True, "store": "core"},
        {"key": "password", "label": "Tradovate password", "kind": "secret", "required": True, "store": "core"},
        {"key": "app_id", "label": "App ID", "kind": "text", "required": True, "store": "extra"},
        {"key": "app_secret", "label": "App secret", "kind": "secret", "required": True, "store": "extra"},
        {"key": "cid", "label": "CID (client id)", "kind": "text", "required": True, "store": "extra"},
        {"key": "sec", "label": "Sec (API secret)", "kind": "secret", "required": True, "store": "extra"},
        {"key": "is_live", "label": "Environment", "kind": "select", "options": ["false", "true"],
         "option_labels": {"false": "Demo / practice", "true": "Live / funded"}, "default": "false", "required": True, "store": "extra",
         "help": "Prop-firm evaluation accounts usually run on the DEMO environment; funded (PA/Live) accounts are LIVE."},
    ],
    "TradeLocker": [
        {"key": "login", "label": "Email", "kind": "text", "required": True, "store": "core"},
        {"key": "password", "label": "Password", "kind": "secret", "required": True, "store": "core"},
        {"key": "server", "label": "Server name", "kind": "text", "required": True, "store": "core"},
        {"key": "is_live", "label": "Environment", "kind": "select", "options": ["false", "true"],
         "option_labels": {"false": "Demo / practice", "true": "Live / funded"}, "default": "false", "required": True, "store": "extra"},
    ],
    "DXtrade": [
        {"key": "base_url", "label": "DXtrade base URL", "kind": "text", "required": True, "store": "extra",
         "help": "White-labelled per firm -- copy the host from your firm's dashboard."},
        {"key": "login", "label": "Username", "kind": "text", "required": True, "store": "core"},
        {"key": "password", "label": "Password", "kind": "secret", "required": True, "store": "core"},
        {"key": "domain", "label": "Domain (leave 'default' unless your firm says otherwise)", "kind": "text", "default": "default", "required": False, "store": "extra"},
    ],
}

# Platforms firms use that this app cannot connect to (yet) -- surfaced
# honestly in the UI rather than hidden.
UNSUPPORTED_PLATFORM_NOTES: dict[str, str] = {
    "Rithmic": "Rithmic needs a licensed data agreement, not a simple API key, so it cannot be connected here. "
               "If your firm offers Tradovate for the same account, use that instead.",
    "NinjaTrader": "NinjaTrader uses a proprietary desktop plugin model, so it cannot be connected here. "
                   "If your firm offers Tradovate/TradeLocker/MT5 for the same account, use that instead.",
    "TopStepX / ProjectX": "ProjectX/TopstepX is not connectable yet. If your account can also be routed through "
                           "Tradovate, use that.",
    "TradingView": "TradingView is a charting front-end, not an order API this app can drive.",
}


def platform_choices() -> list[str]:
    return list(SUPPORTED_PLATFORMS)


def firms_catalog() -> list[dict]:
    """Every known firm with its platforms split into connectable / not."""
    out = []
    for f in prop_firms.PROP_FIRMS:
        conn = [normalize_platform(p) for p in f.platforms if prop_firms.is_platform_connectable(p)]
        # preserve order, drop duplicates (MT4+MT5 both -> MT4/MT5)
        seen: list[str] = []
        for p in conn:
            if p not in seen:
                seen.append(p)
        other = [p for p in f.platforms if not prop_firms.is_platform_connectable(p)]
        out.append({
            "name": f.name, "asset_focus": f.asset_focus, "notes": f.notes,
            "connectable_platforms": seen if f.name != "Other / not listed" else list(SUPPORTED_PLATFORMS),
            "not_connectable": [{"platform": p, "why": UNSUPPORTED_PLATFORM_NOTES.get(p, "No connector for this platform yet.")}
                                for p in other if p != "Other"],
            "connectable_today": f.connectable_today,
        })
    return out


def fields_for(platform: str) -> list[dict]:
    return PLATFORM_FIELDS.get(normalize_platform(platform), [])


def validate_values(platform: str, values: dict, *, editing: bool = False) -> list[str]:
    """Missing-field messages. When editing an existing account a blank
    secret means "keep the saved one", so it is not an error."""
    errs = []
    for f in fields_for(platform):
        if not f.get("required"):
            continue
        v = (values.get(f["key"]) or "").strip() if isinstance(values.get(f["key"]), str) else values.get(f["key"])
        if v in (None, ""):
            if editing and f["kind"] == "secret":
                continue
            errs.append(f"{f['label']} is required.")
    if normalize_platform(platform) not in PLATFORM_FIELDS:
        errs.append(f"'{platform}' cannot be connected by this app.")
    return errs


def account_from_values(
    *, platform: str, firm_name: str, nickname: str, values: dict, account_id: Optional[str] = None,
) -> LiveAccount:
    platform = normalize_platform(platform)
    core = {"login": "", "password": "", "server": "", "terminal_path": ""}
    extra: dict[str, str] = {}
    for f in fields_for(platform):
        raw = values.get(f["key"])
        val = str(raw).strip() if raw is not None else ""
        if not val and f.get("default") and f["store"] == "extra":
            val = str(f["default"])
        if f["store"] == "core":
            core[f["key"]] = val
        elif val:
            extra[f["key"]] = val
    return LiveAccount(
        id=account_id, nickname=(nickname or "").strip() or f"{firm_name or platform} account",
        firm_name=firm_name or "", platform=platform,
        login=core["login"], server=core["server"], password=core["password"],
        terminal_path=core["terminal_path"], extra_credentials=extra,
    )


def public_values(account: LiveAccount) -> dict:
    """Non-secret field values of a saved account, keyed like PLATFORM_FIELDS
    (for pre-filling the edit form in both apps). Secrets are never returned."""
    out: dict = {}
    for f in fields_for(account.platform):
        if f["kind"] == "secret":
            continue
        if f["store"] == "core":
            out[f["key"]] = getattr(account, f["key"], "") or ""
        else:
            out[f["key"]] = (account.extra_credentials or {}).get(f["key"], "")
    return out


def firm_platforms(firm_name: str) -> list[str]:
    for f in firms_catalog():
        if f["name"] == firm_name:
            return f["connectable_platforms"] or []
    return list(SUPPORTED_PLATFORMS)


def merge_saved_secrets(new: LiveAccount, old: Optional[LiveAccount]) -> LiveAccount:
    """Editing: a blank secret means 'keep the saved one'."""
    if old is None:
        return new
    if not new.password:
        new.password = old.password
    for f in fields_for(new.platform):
        if f["kind"] == "secret" and f["store"] == "extra" and f["key"] not in new.extra_credentials:
            if f["key"] in (old.extra_credentials or {}):
                new.extra_credentials[f["key"]] = old.extra_credentials[f["key"]]
    return new


def account_is_live(account: LiveAccount) -> Optional[bool]:
    """True = real/funded environment, False = demo/practice, None = the
    platform has no environment switch we can read (MT5 / DXtrade)."""
    extra = account.extra_credentials or {}
    plat = normalize_platform(account.platform)
    if plat == "cTrader":
        return str(extra.get("host", "demo")).lower() == "live"
    if plat in ("Tradovate", "TradeLocker"):
        return str(extra.get("is_live", "false")).lower() in ("true", "1", "yes")
    return None


# ---------------------------------------------------------------------------
# Test connection, with a plain-language "what to do" for failures
# ---------------------------------------------------------------------------

def _explain_connection_failure(platform: str, message: str) -> list[str]:
    m = (message or "").lower()
    plat = normalize_platform(platform)
    steps: list[str] = []
    if "not installed" in m or "no module" in m or "pip install" in m or "importerror" in m:
        steps.append("A Python package this platform needs is not installed on this computer -- run the pip install command quoted in the message, then retry.")
    if plat == "MT4/MT5" and ("terminal" in m or "initialize" in m or "metatrader" in m or "not available" in m or "windows" in m or "bridge" in m or "mt5linux" in m):
        steps += ["MetaTrader 5 connections need the MT5 desktop terminal. On Windows the app talks to it directly; on Linux/macOS the terminal runs under Wine and the app connects through the mt5linux bridge (run `python scripts/mt5_bridge_server.py` for the exact steps, or see docs/MT5_CROSS_PLATFORM.md).",
                  "Open the MT5 terminal once, log in manually to confirm the server name works, then keep it running and retry.",
                  "If auto-detect cannot find terminal64.exe (Windows install folders, or your Wine/CrossOver prefix on Linux/macOS), paste its full path into 'Terminal path'.",
                  "On Linux/macOS also check that the bridge server is running and reachable (T58_MT5_BRIDGE_HOST / T58_MT5_BRIDGE_PORT, default localhost:18812)."]
    if any(k in m for k in ("401", "unauthor", "invalid", "incorrect", "denied", "credential", "password", "login failed", "authorization")):
        steps += ["Check the username/login and password for typos (paste them rather than typing).",
                  "Confirm you are using the right environment (demo vs live) -- evaluation accounts and funded accounts use different logins.",
                  "If the platform uses API keys (Tradovate app id/secret/cid/sec, cTrader client id/secret), regenerate them and re-enter."]
    if any(k in m for k in ("timeout", "timed out", "unreachable", "getaddrinfo", "connection refused", "name or service", "network", "ssl")):
        steps += ["The server could not be reached -- check this computer's internet connection and any VPN/firewall.",
                  "For DXtrade, make sure the base URL is the exact host from your firm's dashboard."]
    if not steps:
        steps = ["Re-check every field against your firm's account email/dashboard (the server name and environment are the usual culprits).",
                 "Try again in a minute -- some platforms rate-limit repeated logins.",
                 "If it keeps failing, run the same login on the platform's own website/terminal to confirm the account itself is active."]
    return steps


def test_account(account: LiveAccount) -> dict:
    """Connect, read balance/equity, disconnect. Never raises."""
    try:
        adapter = build_adapter(account)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc), "fix": _explain_connection_failure(account.platform, str(exc))}
    try:
        res = adapter.connect()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"{type(exc).__name__}: {exc}", "fix": _explain_connection_failure(account.platform, str(exc))}
    if not res.ok:
        return {"ok": False, "message": res.message, "fix": _explain_connection_failure(account.platform, res.message)}
    info = {"ok": True, "message": res.message, "balance": res.balance, "equity": res.equity,
            "currency": res.currency, "login": res.account_login, "live": account_is_live(account)}
    try:
        adapter.disconnect()
    except Exception:  # noqa: BLE001
        pass
    return info


# ---------------------------------------------------------------------------
# Pre-deploy checklist
# ---------------------------------------------------------------------------

def _firm_presets(firm_name: str):
    from app.prop.presets import PROP_FIRM_PRESETS
    n = (firm_name or "").strip().lower().replace("%", "")
    if not n:
        return []
    return [p for p in PROP_FIRM_PRESETS
            if p.firm.lower().replace("%", "") == n or n in p.firm.lower() or p.firm.lower() in n]


def preflight_checklist(
    account: Optional[LiveAccount], *, strategy_meta: Optional[dict] = None, preset_key: str = "",
) -> dict:
    """Items: {key, level: ok|warn|ack|block, title, detail}.

    level meanings: ok = fine; warn = proceed with eyes open; ack = the
    person must explicitly acknowledge before deploying (firm automation
    policy); block = cannot deploy until fixed.
    """
    items: list[dict] = []
    if account is None:
        items.append({"key": "account", "level": "block", "title": "No account selected",
                      "detail": "Connect a prop-firm account first."})
        return {"items": items, "can_deploy": False, "needs_ack": False}

    plat = normalize_platform(account.platform)
    if plat not in PLATFORM_FIELDS:
        items.append({"key": "platform", "level": "block", "title": f"{account.platform} cannot be connected",
                      "detail": UNSUPPORTED_PLATFORM_NOTES.get(account.platform, "No connector exists for this platform yet.")})
    elif not account.is_usable:
        items.append({"key": "platform", "level": "block", "title": "Account details incomplete",
                      "detail": "Open the account and fill in the missing credentials."})
    else:
        items.append({"key": "platform", "level": "ok", "title": f"{plat} connector available", "detail": ""})

    live = account_is_live(account)
    if live is True:
        items.append({"key": "env", "level": "warn", "title": "LIVE / funded environment",
                      "detail": "Orders will use real capital. If you have not run this strategy on a demo/practice account first, do that."})
    elif live is False:
        items.append({"key": "env", "level": "ok", "title": "Demo / practice environment",
                      "detail": "Good place to confirm the connector behaves before going live."})
    else:
        items.append({"key": "env", "level": "warn", "title": "Environment not detectable for this platform",
                      "detail": "Confirm in your platform whether this login is a demo or a funded account."})

    # Firm automation policy
    presets = _firm_presets(account.firm_name)
    needs_ack = False
    if presets:
        banned = [p for p in presets if not p.algo_trading_allowed]
        if banned:
            needs_ack = True
            items.append({"key": "policy", "level": "ack",
                          "title": f"{banned[0].firm}: automated trading is NOT clearly permitted",
                          "detail": (banned[0].algo_policy_note or "The firm's published rules do not clearly allow automation.")
                                    + " Violating this can get a funded account terminated regardless of performance. "
                                      "Only continue if you have the firm's written permission for this exact automation."})
        else:
            items.append({"key": "policy", "level": "ok", "title": f"{presets[0].firm}: automation is permitted",
                          "detail": (presets[0].algo_policy_note or "") + " Re-check the firm's current rules before relying on this."})
    else:
        items.append({"key": "policy", "level": "warn", "title": "Firm automation policy unknown",
                      "detail": "No policy is on file for this firm. Read its rules on automated/EA trading before deploying."})

    # Rules verified
    chosen = None
    if preset_key:
        try:
            from app.prop.presets import get_preset
            chosen = get_preset(preset_key)
        except Exception:  # noqa: BLE001
            chosen = None
    if chosen is None and presets:
        chosen = presets[0]
    if chosen is not None:
        if chosen.verified_on and chosen.source_url:
            items.append({"key": "rules", "level": "ok", "title": f"{chosen.label}: rules verified ({chosen.verified_on})",
                          "detail": f"Verification tier: {chosen.verification_tier or 'n/a'}."})
        else:
            items.append({"key": "rules", "level": "warn", "title": f"{chosen.label}: rules NOT verified",
                          "detail": "The loss limits used for the live halt come from numbers nobody has checked against the firm's page."})

    # Strategy validation record
    if strategy_meta is not None:
        try:
            from app.validation.simple_report import CHECK_TO_METHOD, GATING_CHECKS
            on_record = set((strategy_meta.get("validation_methods_run") or {}).keys())
            missing = [c for c in GATING_CHECKS if CHECK_TO_METHOD[c] not in on_record]
            if missing:
                items.append({"key": "validated", "level": "warn", "title": "Strategy has not completed all validation checks",
                              "detail": "Still missing: " + ", ".join(missing) + ". Run them on the Validate page before risking real money."})
            else:
                items.append({"key": "validated", "level": "ok", "title": "All core validation checks are on record", "detail": ""})
        except Exception:  # noqa: BLE001
            pass

    can_deploy = not any(i["level"] == "block" for i in items)
    return {"items": items, "can_deploy": can_deploy, "needs_ack": needs_ack}
