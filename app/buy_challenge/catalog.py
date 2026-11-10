"""
Buy Challenge catalog: every firm and every plan the app models, side by side, with the numbers the firm itself publishes.

Where the data comes from, and what is never invented:
  * rules (target, drawdown, daily loss, consistency, contracts, algo policy) -- ``app.prop.presets`` (each with its own check date and tier)
  * price / reset / profit split -- ``app.prop.presets.FIRM_FEES`` (each with its own check date and source)
  * firm website -- ``FIRM_SITES`` below. Only addresses the app's own firm data cites, or that were confirmed against a public listing, are
    filled in; the rest stay empty so the button asks you for the link instead of sending you somewhere that might be wrong.
  * DISCOUNT CODES -- never shipped. A code only exists once you enter it (Offers editor) and then carries the date you checked it.
    Nothing here can verify a code works: confirm it at the firm's own checkout.
Purchases happen at the firm's official checkout. T58 never takes payment.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import date
from pathlib import Path

from app.data.storage import get_app_base_dir
from app.prop import presets as P

_LOCK = threading.RLock()

# firm display name (as in presets) -> {site, asset, note}. ``site`` empty = not confirmed; the user can set their own in Offers.
FIRM_SITES = {
    "FTMO": {"site": "https://ftmo.com", "asset": "Forex / CFD", "platforms": "MT4, MT5, cTrader, DXtrade"},
    "TopStep": {"site": "https://www.topstep.com", "asset": "Futures", "platforms": "TopstepX, NinjaTrader, Tradovate"},
    "Apex Trader Funding": {"site": "https://apextraderfunding.com", "asset": "Futures", "platforms": "Tradovate, Rithmic, NinjaTrader"},
    "Lucid Trading": {"site": "https://lucidtrading.com", "asset": "Futures", "platforms": "Tradovate, Rithmic"},
    "FundedNext": {"site": "https://fundednext.com", "asset": "Forex / Futures", "platforms": "MT5, cTrader, Tradovate"},
    "The5%ers": {"site": "https://the5ers.com", "asset": "Forex", "platforms": "MT5"},
    "E8 Markets": {"site": "https://e8markets.com", "asset": "Forex / Futures", "platforms": "MT5, Match-Trader, TradeLocker"},
    "The Futures Desk": {"site": "https://thefuturesdesk.com", "asset": "Futures", "platforms": ""},
    "FunderPro": {"site": "", "asset": "Forex / Futures", "platforms": ""},
    "Atlas Funded": {"site": "", "asset": "Forex", "platforms": "TradeLocker"},
}

OFFER_FIELDS = ("code", "discount_pct", "discount_note", "checkout_url", "site_url", "checked_on", "expires")


def offers_path() -> Path:
    d = get_app_base_dir() / "data" / "buy_challenge"
    d.mkdir(parents=True, exist_ok=True)
    return d / "offers.json"


def load_offers() -> dict:
    with _LOCK:
        p = offers_path()
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    return {k: v for k, v in d.items() if isinstance(v, dict)}
            except (OSError, ValueError):
                pass
    return {}


def _safe_url(u) -> str:
    u = str(u or "").strip()
    return u if u.lower().startswith(("https://", "http://")) and " " not in u and len(u) < 500 else ""


def save_offer(firm: str, offer: dict) -> dict:
    """Save one firm's code / discount / links. A code is stored with the date you checked it."""
    firm = str(firm or "").strip()
    if firm not in FIRM_SITES:
        raise ValueError("Unknown firm.")
    if not isinstance(offer, dict):
        raise ValueError("offer must be an object")
    row = {"code": str(offer.get("code") or "").strip()[:40], "discount_note": str(offer.get("discount_note") or "").strip()[:120],
           "checkout_url": _safe_url(offer.get("checkout_url")), "site_url": _safe_url(offer.get("site_url")),
           "expires": str(offer.get("expires") or "")[:10], "checked_on": str(offer.get("checked_on") or "")[:10]}
    pct = offer.get("discount_pct")
    try:
        pct = float(pct) if pct not in (None, "") else None
    except (TypeError, ValueError) as exc:
        raise ValueError("discount_pct must be a number") from exc
    if pct is not None and not (0 <= pct <= 90):
        raise ValueError("discount_pct must be between 0 and 90")
    row["discount_pct"] = pct
    if row["code"] and not row["checked_on"]:
        row["checked_on"] = date.today().isoformat()
    with _LOCK:
        data = load_offers()
        if row["code"] or row["discount_pct"] is not None or any(row[k] for k in ("discount_note", "checkout_url", "site_url", "expires")):
            data[firm] = row
        else:
            data.pop(firm, None)
        p = offers_path()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    return data.get(firm, {})


def _offer_state(offer: dict | None, today: date | None = None) -> dict:
    o = offer or {}
    today = today or date.today()
    expired = False
    if o.get("expires"):
        try:
            expired = date.fromisoformat(o["expires"]) < today
        except ValueError:
            expired = False
    return {"code": o.get("code", ""), "discount_pct": o.get("discount_pct"), "discount_note": o.get("discount_note", ""),
            "checked_on": o.get("checked_on", ""), "expires": o.get("expires", ""), "expired": expired,
            "active": bool(o.get("code")) and not expired}


def _money(x):
    return None if x is None else round(float(x), 2)


def plans(*, today: date | None = None) -> list[dict]:
    """One row per modeled plan, with rules, price and the firm's offer."""
    offers = load_offers()
    sim = _sim_pass_rates()
    rows = []
    for p in P.list_presets():
        fee = P.get_fee_info(p.key)
        site = FIRM_SITES.get(p.firm, {})
        o = offers.get(p.firm, {})
        st = _offer_state(o, today)
        price = fee.challenge_price if fee else None
        disc = price * (1 - st["discount_pct"] / 100.0) if (price is not None and st["active"] and st["discount_pct"]) else None
        a = p.account_size
        pass_rate = sim.get(p.key)
        eff = disc if disc is not None else price
        rows.append({
            "key": p.key, "firm": p.firm, "label": p.label, "size": a,
            "price": _money(price), "price_with_code": _money(disc), "reset": _money(fee.effective_reset_price) if fee else None,
            "activation_fee": _money(fee.activation_fee) if fee else None, "monthly": bool(fee and fee.monthly_subscription),
            "split_pct": fee.profit_split_pct if fee else None,
            "price_checked": fee.date_checked if fee else "", "price_source": fee.source_url if fee else "", "price_tier": fee.source_tier if fee else "",
            "target": round(p.evaluation_profit_target_pct / 100.0 * a, 2), "target_pct": p.evaluation_profit_target_pct,
            "daily_loss": round(p.daily_loss_limit_pct / 100.0 * a, 2) if p.daily_loss_limit_pct else None,
            "max_dd": round(p.max_drawdown_pct / 100.0 * a, 2), "dd_type": p.drawdown_type,
            "consistency_pct": p.consistency_rule_pct, "min_days": p.min_trading_days, "max_contracts": p.max_contracts,
            "algo_allowed": bool(p.algo_trading_allowed), "algo_note": p.algo_policy_note,
            "rules_verified_on": p.verified_on, "rules_tier": p.verification_tier,
            "asset": site.get("asset", ""), "platforms": site.get("platforms", ""),
            "pass_rate": pass_rate, "expected_cost_to_pass": round(eff / pass_rate, 2) if (eff is not None and pass_rate) else None,
            "offer": st,
        })
    return rows


def _sim_pass_rates() -> dict:
    """Pass rate from YOUR latest saved simulation per preset (My Attempts). Empty until you have run one: never a generic figure."""
    try:
        from app.prop import desk_store as ds
        out = {}
        for r in sorted(ds.load()["sim_runs"], key=lambda r: (r["created_at"], r["id"])):
            if r.get("preset_key") and r["summary"].get("pass_rate") is not None:
                out[r["preset_key"]] = float(r["summary"]["pass_rate"])
        return out
    except Exception:  # noqa: BLE001
        return {}


def firms(rows: list[dict] | None = None) -> list[dict]:
    """One card per firm."""
    rows = rows if rows is not None else plans()
    offers = load_offers()
    out = []
    for name, meta in FIRM_SITES.items():
        ps = [r for r in rows if r["firm"] == name]
        if not ps:
            continue
        o = offers.get(name, {})
        priced = [r for r in ps if r["price"] is not None]
        cheapest = min(priced, key=lambda r: r["price_with_code"] or r["price"]) if priced else None
        site = _safe_url(o.get("site_url")) or meta["site"]
        out.append({
            "firm": name, "asset": meta["asset"], "platforms": meta["platforms"], "plans": len(ps),
            "sizes": sorted({r["size"] for r in ps}), "algo_allowed": any(r["algo_allowed"] for r in ps),
            "algo_all": all(r["algo_allowed"] for r in ps), "best_split": max((r["split_pct"] or 0) for r in ps) or None,
            "from_price": (cheapest["price_with_code"] or cheapest["price"]) if cheapest else None,
            "from_label": cheapest["label"] if cheapest else "",
            "offer": _offer_state(o), "site_url": site, "checkout_url": _safe_url(o.get("checkout_url")) or site,
            "link_confirmed": bool(site),
        })
    return out


def catalog() -> dict:
    rows = plans()
    return {"firms": firms(rows), "plans": rows,
            "disclaimer": ("Prices and rules are copied from each firm's published pages on the dates shown and firms change them often. "
                           "Always confirm the price, rules and any code at the firm's own checkout before paying. T58 does not sell "
                           "challenges and does not take payment. Discount codes appear only after you enter and date them."),
            "today": date.today().isoformat()}


def record_purchase(preset_key: str, *, fee: float | None = None, platform: str = "", strategy: str = "") -> dict:
    """"I bought this": adds the account to the Prop Desk (logs the cost, opens an attempt, starts the Mission)."""
    from app.prop import desk_store as ds
    p = P.get_preset(preset_key)
    row = next((r for r in plans() if r["key"] == preset_key), None)
    paid = fee if fee is not None else ((row["price_with_code"] or row["price"]) if row else None)
    return ds.add_account({"firm": p.firm, "preset_key": preset_key, "size": p.account_size, "label": p.label, "fee": paid or 0.0,
                           "platform": platform, "strategy": strategy, "stage": "evaluation"})
