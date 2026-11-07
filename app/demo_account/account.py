"""The T58 demo prop account.

Every user starts in Demo mode. A demo account is a 25k / 50k / 100k T58 Prop Firm simulation with the SAME rule shape the
Rulebook v2 pipeline validates against (``app.prop.lifecycle_v2.V2Rules.tier``): trailing end-of-day loss limit that locks at
start + offset, a daily soft limit, a profit target and a best-day consistency cap.  A strategy "deployed" to it receives the same
TradingView / Pine alerts as a real account (through the webhook relay); every closed trade is booked here as a visible win or
loss, and the account passes or breaches exactly as a real evaluation would.  Live mode never touches this file.

State: one small JSON file under data/config/. Pure stdlib; all money values are dollars.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.data.storage import get_app_base_dir

TIERS = (25_000, 50_000, 100_000)
MODES = ("demo", "live")
MAX_TRADES = 5000
_LOCK = threading.RLock()
_FILE = "demo_account.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path() -> Path:
    d = get_app_base_dir() / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d / _FILE


def _rules(size: float):
    from app.prop.lifecycle_v2 import V2Rules
    return V2Rules().tier(float(size))


@dataclass
class DemoAccount:
    size: float = 50_000.0
    balance: float = 50_000.0
    start_balance: float = 50_000.0
    floor: float = 48_000.0                 # trailing end-of-day loss floor (breach when balance <= floor)
    eod_high: float = 50_000.0              # best end-of-day balance so far
    day: str = ""                           # trading date of ``day_start``
    day_start: float = 50_000.0
    best_day: float = 0.0                   # best single-day profit (consistency)
    status: str = "active"                  # active | passed | breached
    status_reason: str = ""
    soft_limit_hits: int = 0
    day_stopped: bool = False               # today's soft daily limit was hit: the strategy should stand down until tomorrow
    attempt: int = 1
    created_at: str = ""
    relay_ids: list = field(default_factory=list)   # webhook relays that feed this account
    strategy: str = ""                      # name of the deployed strategy (display only)
    trades: list = field(default_factory=list)

    # ---- derived
    @property
    def rules(self):
        return _rules(self.size)

    @property
    def net_pnl(self) -> float:
        return round(self.balance - self.start_balance, 2)

    def to_dict(self) -> dict:
        r = self.rules
        wins = [t["pnl"] for t in self.trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in self.trades if t["pnl"] < 0]
        n = len(self.trades)
        return {
            **{k: v for k, v in asdict(self).items() if k != "trades"},
            "net_pnl": self.net_pnl, "cushion": round(self.balance - self.floor, 2),
            "profit_target": r.profit_target, "loss_limit": r.loss_limit, "soft_daily_limit": r.soft_daily_limit,
            "target_pct": round(100.0 * max(self.net_pnl, 0.0) / r.profit_target, 1) if r.profit_target else 0.0,
            "day_pnl": round(self.balance - self.day_start, 2),
            "n_trades": n, "n_wins": len(wins), "n_losses": len(losses),
            "win_rate": round(100.0 * len(wins) / n, 1) if n else None,
            "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
            "recent_trades": list(reversed(self.trades[-25:])),
        }


def _fresh(size: float, *, attempt: int = 1, relay_ids=None, strategy: str = "") -> DemoAccount:
    size = float(size) if float(size) in TIERS else 50_000.0
    r = _rules(size)
    return DemoAccount(size=size, balance=size, start_balance=size, floor=size - r.loss_limit, eod_high=size, day_start=size,
                       attempt=attempt, created_at=_now(), relay_ids=list(relay_ids or []), strategy=strategy)


def load() -> DemoAccount:
    with _LOCK:
        try:
            d = json.loads(_path().read_text(encoding="utf-8"))
            known = DemoAccount.__dataclass_fields__
            return DemoAccount(**{k: v for k, v in d.items() if k in known})
        except Exception:  # noqa: BLE001 - missing / corrupt file: a fresh 50k demo, never a crash
            return _fresh(50_000.0)


def _save(acct: DemoAccount) -> None:
    acct.trades = acct.trades[-MAX_TRADES:]
    tmp = _path().with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(acct)), encoding="utf-8")
    tmp.replace(_path())


def reset(size: float | None = None, *, keep_deployment: bool = True) -> DemoAccount:
    """New evaluation attempt (after a pass or a breach, or on request). Keeps the tier, relays and strategy unless told otherwise."""
    with _LOCK:
        old = load()
        acct = _fresh(size or old.size, attempt=old.attempt + 1, relay_ids=old.relay_ids if keep_deployment else [],
                      strategy=old.strategy if keep_deployment else "")
        _save(acct)
        return acct


def set_tier(size: float) -> DemoAccount:
    if float(size) not in TIERS:
        raise ValueError(f"Demo tiers are {', '.join(f'${t:,}' for t in TIERS)}.")
    with _LOCK:
        old = load()
        acct = _fresh(size, attempt=1, relay_ids=old.relay_ids, strategy=old.strategy)
        _save(acct)
        return acct


# ---- mode (Demo / Live) -----------------------------------------------------------------------------------------------------
def get_mode() -> str:
    try:
        from app.accounts.settings import load_account_settings
        m = (load_account_settings().account_mode or "demo").lower()
        return m if m in MODES else "demo"
    except Exception:  # noqa: BLE001
        return "demo"


def set_mode(mode: str) -> str:
    mode = (mode or "").strip().lower()
    if mode not in MODES:
        raise ValueError("Mode must be 'demo' or 'live'.")
    from app.accounts.settings import load_account_settings, save_account_settings
    s = load_account_settings()
    s.account_mode = mode
    save_account_settings(s)
    return mode


# ---- booking trades -----------------------------------------------------------------------------------------------------------
def _roll_day(acct: DemoAccount, today: str) -> None:
    """New trading day: bank yesterday's close into the end-of-day high, trail (or lock) the floor, reset the day counters."""
    if acct.day and acct.day != today:
        r = acct.rules
        day_profit = acct.balance - acct.day_start
        acct.best_day = max(acct.best_day, day_profit)
        acct.eod_high = max(acct.eod_high, acct.balance)
        if r.trailing:
            trailed = acct.eod_high - r.loss_limit
            acct.floor = max(acct.floor, min(trailed, acct.start_balance + r.lock_floor_offset))
        acct.day_start = acct.balance
        acct.day_stopped = False
    if not acct.day or acct.day != today:
        acct.day = today
        if not acct.trades:
            acct.day_start = acct.balance


def record_trade(pnl: float, *, root: str = "", side: str = "", qty: float = 1.0, entry=None, exit=None, why: str = "",
                 source: str = "alert", closed_at: str | None = None) -> dict | None:
    """Book one closed trade. Returns the trade row (with running balance) or None when the account is no longer active
    (a passed or breached evaluation stops booking until it is reset, exactly like a real one)."""
    with _LOCK:
        acct = load()
        if acct.status != "active":
            return None
        when = closed_at or _now()
        _roll_day(acct, when[:10])
        r = acct.rules
        acct.balance = round(acct.balance + float(pnl), 2)
        day_pnl = acct.balance - acct.day_start
        row = {"n": len(acct.trades) + 1, "closed_at": when, "root": root, "side": side, "qty": qty, "entry": entry, "exit": exit,
               "pnl": round(float(pnl), 2), "result": "win" if pnl > 0 else "loss" if pnl < 0 else "flat", "why": why, "source": source,
               "balance": acct.balance, "day_pnl": round(day_pnl, 2)}
        acct.trades.append(row)
        if day_pnl <= -r.soft_daily_limit and not acct.day_stopped:
            acct.day_stopped = True
            acct.soft_limit_hits += 1
        profit = acct.balance - acct.start_balance
        best = max(acct.best_day, day_pnl)
        if acct.balance <= acct.floor:
            acct.status, acct.status_reason = "breached", f"Balance ${acct.balance:,.2f} hit the loss floor ${acct.floor:,.2f}."
        elif profit >= r.profit_target:
            share = best / profit if profit > 0 else 1.0
            cap = r.consistency_best_day_pct
            cap = None if cap is None else (cap / 100.0 if cap > 1 else cap)
            if cap is None or share <= cap:
                acct.status, acct.status_reason = "passed", f"Profit target ${r.profit_target:,.0f} reached with the best day at {share:.0%} of profit."
            else:
                acct.status_reason = f"Target reached but best day is {share:.0%} of profit (cap {cap:.0%}): keep trading."
        row["status"] = acct.status
        _save(acct)
        return row


def record_relay_actions(relay_id: str, actions: list) -> int:
    """Called by the webhook relay after each accepted alert: books every CLOSED leg for a relay linked to the demo account."""
    if not actions:
        return 0
    acct = load()
    if relay_id not in acct.relay_ids or get_mode() != "demo":
        return 0
    n = 0
    for a in actions:
        if a.get("action") in ("close", "reduce") and a.get("pnl") is not None:
            if record_trade(a["pnl"], root=a.get("root", ""), side=a.get("side", ""), qty=a.get("qty", 1.0), entry=a.get("entry"),
                            exit=a.get("exit"), why=a.get("why", ""), source="tradingview", closed_at=a.get("closed_at")):
                n += 1
    return n


# ---- deploy a validated strategy to the demo account --------------------------------------------------------------------------
def deploy_strategy(name: str, *, size: float | None = None, tickers: list | None = None, public_url: str = "") -> dict:
    """Create (or reuse) the webhook relay that feeds the demo account and return everything the user pastes into TradingView:
    the webhook URL and the alert message. Identical to what the real-account path uses, so Demo behaves the same as Live."""
    from app.webhook_relay import config as relay_config
    from app.webhook_relay.signals import pine_alert_message
    with _LOCK:
        acct = load()
        if size and float(size) != acct.size:
            acct = set_tier(float(size))
        relay = next((r for r in (relay_config.get_relay(i) for i in acct.relay_ids) if r is not None), None)
        if relay is None:
            relay = relay_config.create_relay({"name": f"T58 Demo {int(acct.size / 1000)}k"})
            acct.relay_ids = [relay.id]
        if tickers is not None:
            relay_config.update_relay(relay.id, {"allowed_tickers": [t.upper() for t in tickers]})
        acct.strategy = (name or acct.strategy or "").strip()[:80]
        _save(acct)
    base = (public_url or relay_config.get_public_url() or "").rstrip("/")
    return {"relay_id": relay.id, "strategy": acct.strategy, "size": acct.size, "alert_message": pine_alert_message(relay.passphrase),
            "webhook_url": f"{base}/webhook/tradingview/{relay.token}" if base else "", "path": f"/webhook/tradingview/{relay.token}",
            "url_ready": bool(base)}


# ---- header pill --------------------------------------------------------------------------------------------------------------------
def header_payload(live_rows: list | None = None) -> dict:
    """What the top-right P&L tracker shows. Demo: the T58 demo account. Live: the running real sessions (passed in)."""
    mode = get_mode()
    if mode == "demo":
        d = load().to_dict()
        return {"mode": "demo", "label": f"T58 Demo {int(d['size'] / 1000)}k", "pnl": d["net_pnl"], "day_pnl": d["day_pnl"],
                "balance": d["balance"], "status": d["status"], "target_pct": d["target_pct"], "n_trades": d["n_trades"],
                "cushion": d["cushion"], "running": bool(d["relay_ids"])}
    rows = live_rows or []
    pnl = sum((r.get("net_pnl") or 0) + (r.get("floating") or 0) for r in rows)
    bal = next((r.get("balance") for r in rows if r.get("balance") is not None), None)
    return {"mode": "live", "label": rows[0].get("label", "Live") if rows else "Live · no account running", "pnl": round(pnl, 2),
            "day_pnl": None, "balance": bal, "status": "active" if rows else "idle", "n_trades": sum(r.get("n_trades_closed") or 0 for r in rows),
            "running": bool(rows)}
