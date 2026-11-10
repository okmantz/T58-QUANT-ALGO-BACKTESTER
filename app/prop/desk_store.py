"""
Prop Desk store: the one saved history behind My Attempts, the Challenge Cost Tracker, the Multi-Account Manager,
the Payout Calendar and the First Payout Mission.

Single-user JSON file at <app dir>/data/prop_desk/desk.json::

    accounts  [{id,label,firm,preset_key,size,stage,status,purchased_on,funded_on,fee,platform,live_account_id,
                strategy,payouts[{id,date,amount}],notes,created_at}]
    costs     [{id,date,kind,firm,account_id,amount,note}]        kind: purchase|reset|activation|subscription|other
    attempts  [{id,source,account_id,firm,preset_key,size,started,ended,outcome,bust_cause,fee,strategy,days,note}]
    sim_runs  [{id,kind,created_at,firm,preset_key,strategy,summary{...}}]   (attempt replay / rolling / survival results)
    settings  {risk_preset, mission{...}}

Money in ``costs`` is the single source of truth for "fees spent"; ``attempts`` is the source of truth for pass rate and
bust causes. Creating an account logs its purchase cost and opens an attempt; marking the account passed / failed closes that
attempt, and a reset closes the old attempt and opens a new one, so the ledger stays correct without double entry.

Nothing here reaches the network and nothing here invents a figure: a field the user did not give stays empty.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from app.data.storage import get_app_base_dir

_LOCK = threading.RLock()

STAGES = ("evaluation", "funded", "closed")
STATUSES = ("active", "passed", "failed", "closed")
COST_KINDS = ("purchase", "reset", "activation", "subscription", "other")
OUTCOMES = ("open", "passed", "failed")
SOURCES = ("real", "manual")

# Canonical reasons an evaluation ends. Keys are stored; labels are shown.
BUST_CAUSES = {
    "daily_loss_limit": "Daily loss limit",
    "max_drawdown": "Max drawdown",
    "consistency": "Consistency rule",
    "inactivity": "Inactivity",
    "time_limit": "Time limit ran out",
    "rule_violation": "Rule violation (news, hedging, size...)",
    "other": "Other",
}


def _uid() -> str:
    return uuid.uuid4().hex[:10]


def desk_path() -> Path:
    d = get_app_base_dir() / "data" / "prop_desk"
    d.mkdir(parents=True, exist_ok=True)
    return d / "desk.json"


def _empty() -> dict:
    return {"accounts": [], "costs": [], "attempts": [], "sim_runs": [], "settings": {}}


def load() -> dict:
    with _LOCK:
        p = desk_path()
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    for k, v in _empty().items():
                        d.setdefault(k, v)
                    return d
            except (OSError, ValueError):
                pass
        return _empty()


def _write(data: dict) -> None:
    p = desk_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, p)


# ------------------------------------------------------------------------------------------------ small helpers
def _num(v: Any, default: float | None = 0.0) -> float | None:
    try:
        f = float(v)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def _day(v: Any, default: str = "") -> str:
    """ISO date (YYYY-MM-DD) or ``default`` when it does not parse."""
    s = str(v or "").strip()[:10]
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError:
        return default


def normalize_bust_cause(text: Any) -> str:
    """Map any free-text / engine failure reason onto a canonical BUST_CAUSES key ('' stays '')."""
    s = str(text or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not s:
        return ""
    if s in BUST_CAUSES:
        return s
    if "daily" in s:
        return "daily_loss_limit"
    if "drawdown" in s or s in ("dd", "mll", "max_loss"):
        return "max_drawdown"
    if "consisten" in s:
        return "consistency"
    if "inactiv" in s:
        return "inactivity"
    if "time" in s or "too_many_days" in s or "target_not_reached" in s or "expired" in s:
        return "time_limit"
    if any(k in s for k in ("news", "hedg", "violat", "contract", "size", "weekend", "banned", "prohibit")):
        return "rule_violation"
    return "other"


# ------------------------------------------------------------------------------------------------ accounts
def _norm_account(d: dict, old: dict | None = None) -> dict:
    old = old or {}
    firm = str(d.get("firm", old.get("firm", "")) or "").strip()[:80]
    if not firm:
        raise ValueError("An account needs a firm name.")
    size = _num(d.get("size", old.get("size", 0)), 0.0)
    if not size or size <= 0:
        raise ValueError("An account needs a positive account size.")
    stage = str(d.get("stage", old.get("stage", "evaluation")) or "evaluation")
    status = str(d.get("status", old.get("status", "active")) or "active")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}.")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}.")
    fee = _num(d.get("fee", old.get("fee", 0)), 0.0)
    if fee is None or fee < 0:
        raise ValueError("fee cannot be negative.")
    return {
        "id": old.get("id") or str(d.get("id") or _uid()),
        "label": str(d.get("label", old.get("label", "")) or f"{firm} {int(size / 1000)}K")[:80],
        "firm": firm,
        "preset_key": str(d.get("preset_key", old.get("preset_key", "")) or ""),
        "size": size,
        "stage": stage,
        "status": status,
        "purchased_on": _day(d.get("purchased_on", old.get("purchased_on")), old.get("purchased_on", "")),
        "funded_on": _day(d.get("funded_on", old.get("funded_on")), old.get("funded_on", "")),
        "fee": fee,
        "platform": str(d.get("platform", old.get("platform", "")) or "")[:40],
        "live_account_id": str(d.get("live_account_id", old.get("live_account_id", "")) or ""),
        "strategy": str(d.get("strategy", old.get("strategy", "")) or "")[:120],
        "payouts": list(old.get("payouts", [])),
        "notes": str(d.get("notes", old.get("notes", "")) or "")[:2000],
        "created_at": old.get("created_at") or date.today().isoformat(),
    }


def get_account(account_id: str, data: dict | None = None) -> dict | None:
    data = data or load()
    return next((a for a in data["accounts"] if a.get("id") == account_id), None)


def add_account(d: dict, *, log_cost: bool = True) -> dict:
    """Create an account. A positive fee logs a 'purchase' cost and an evaluation account opens an attempt."""
    if not isinstance(d, dict):
        raise ValueError("account must be an object")
    with _LOCK:
        data = load()
        acct = _norm_account(d)
        if not acct["purchased_on"]:
            acct["purchased_on"] = date.today().isoformat()
        data["accounts"].append(acct)
        if log_cost and acct["fee"] > 0:
            data["costs"].append({"id": _uid(), "date": acct["purchased_on"], "kind": "purchase", "firm": acct["firm"],
                                  "account_id": acct["id"], "amount": acct["fee"], "note": acct["label"]})
        if acct["stage"] == "evaluation":
            data["attempts"].append(_attempt_for(acct, source="real"))
        _write(data)
        return acct


def _attempt_for(acct: dict, *, source: str) -> dict:
    return {"id": _uid(), "source": source, "account_id": acct["id"], "firm": acct["firm"], "preset_key": acct["preset_key"],
            "size": acct["size"], "started": acct["purchased_on"], "ended": "", "outcome": "open", "bust_cause": "",
            "fee": acct["fee"], "strategy": acct["strategy"], "days": None, "note": ""}


def _close_open_attempt(data: dict, account_id: str, outcome: str, bust_cause: str, ended: str) -> None:
    for a in reversed(data["attempts"]):
        if a.get("account_id") == account_id and a.get("outcome") == "open":
            a["outcome"] = outcome
            a["ended"] = ended or date.today().isoformat()
            a["bust_cause"] = normalize_bust_cause(bust_cause) if outcome == "failed" else ""
            try:
                a["days"] = (date.fromisoformat(a["ended"]) - date.fromisoformat(a["started"])).days if a.get("started") else None
            except ValueError:
                a["days"] = None
            return


def update_account(account_id: str, d: dict) -> dict:
    """Edit an account. Moving it to status passed / failed closes its open attempt (pass ``bust_cause`` for a failure)."""
    with _LOCK:
        data = load()
        old = get_account(account_id, data)
        if old is None:
            raise KeyError(account_id)
        new = _norm_account(d, old)
        idx = data["accounts"].index(old)
        data["accounts"][idx] = new
        if old["status"] == "active" and new["status"] == "passed":
            new["stage"] = "funded" if new["stage"] == "evaluation" else new["stage"]
            new["funded_on"] = new["funded_on"] or _day(d.get("ended"), date.today().isoformat())
            _close_open_attempt(data, account_id, "passed", "", _day(d.get("ended")))
        elif old["status"] == "active" and new["status"] == "failed":
            _close_open_attempt(data, account_id, "failed", str(d.get("bust_cause", "")), _day(d.get("ended")))
        _write(data)
        return new


def reset_account(account_id: str, *, fee: float = 0.0, bust_cause: str = "", when: str = "") -> dict:
    """The account busted and was re-bought: close the open attempt as failed, log the reset fee, open a new attempt."""
    with _LOCK:
        data = load()
        acct = get_account(account_id, data)
        if acct is None:
            raise KeyError(account_id)
        when = _day(when, date.today().isoformat())
        _close_open_attempt(data, account_id, "failed", bust_cause, when)
        fee = _num(fee, 0.0) or 0.0
        if fee < 0:
            raise ValueError("fee cannot be negative.")
        if fee > 0:
            data["costs"].append({"id": _uid(), "date": when, "kind": "reset", "firm": acct["firm"], "account_id": account_id,
                                  "amount": fee, "note": "reset"})
        acct["status"] = "active"
        acct["stage"] = "evaluation"
        acct["purchased_on"] = when
        att = _attempt_for(acct, source="real")
        att["fee"] = fee
        data["attempts"].append(att)
        _write(data)
        return acct


def delete_account(account_id: str) -> bool:
    with _LOCK:
        data = load()
        n = len(data["accounts"])
        data["accounts"] = [a for a in data["accounts"] if a.get("id") != account_id]
        if len(data["accounts"]) == n:
            return False
        _write(data)
        return True


def add_payout(account_id: str, when: str, amount: float) -> dict:
    amount_f = _num(amount, None)
    if amount_f is None or amount_f <= 0:
        raise ValueError("A payout needs a positive amount.")
    with _LOCK:
        data = load()
        acct = get_account(account_id, data)
        if acct is None:
            raise KeyError(account_id)
        p = {"id": _uid(), "date": _day(when, date.today().isoformat()), "amount": amount_f}
        acct["payouts"].append(p)
        acct["payouts"].sort(key=lambda x: x["date"])
        if acct["stage"] == "evaluation":          # a payout only exists on a funded account
            acct["stage"] = "funded"
            acct["status"] = "active" if acct["status"] in ("active", "passed") else acct["status"]
        _write(data)
        return p


def delete_payout(account_id: str, payout_id: str) -> bool:
    with _LOCK:
        data = load()
        acct = get_account(account_id, data)
        if acct is None:
            return False
        n = len(acct["payouts"])
        acct["payouts"] = [p for p in acct["payouts"] if p.get("id") != payout_id]
        if len(acct["payouts"]) == n:
            return False
        _write(data)
        return True


# ------------------------------------------------------------------------------------------------ costs
def add_cost(d: dict) -> dict:
    kind = str(d.get("kind", "purchase") or "purchase")
    if kind not in COST_KINDS:
        raise ValueError(f"kind must be one of {', '.join(COST_KINDS)}.")
    amount = _num(d.get("amount"), None)
    if amount is None or amount <= 0:
        raise ValueError("A cost needs a positive amount.")
    row = {"id": _uid(), "date": _day(d.get("date"), date.today().isoformat()), "kind": kind,
           "firm": str(d.get("firm", "") or "").strip()[:80], "account_id": str(d.get("account_id", "") or ""),
           "amount": amount, "note": str(d.get("note", "") or "")[:200]}
    with _LOCK:
        data = load()
        data["costs"].append(row)
        data["costs"].sort(key=lambda x: (x["date"], x["id"]))
        _write(data)
    return row


def delete_cost(cost_id: str) -> bool:
    with _LOCK:
        data = load()
        n = len(data["costs"])
        data["costs"] = [c for c in data["costs"] if c.get("id") != cost_id]
        if len(data["costs"]) == n:
            return False
        _write(data)
        return True


# ------------------------------------------------------------------------------------------------ attempts
def add_attempt(d: dict, *, log_cost: bool = True) -> dict:
    """Back-fill one attempt you already took (an evaluation you bought before using T58)."""
    outcome = str(d.get("outcome", "open") or "open")
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}.")
    firm = str(d.get("firm", "") or "").strip()[:80]
    if not firm:
        raise ValueError("An attempt needs a firm name.")
    fee = _num(d.get("fee", 0), 0.0) or 0.0
    if fee < 0:
        raise ValueError("fee cannot be negative.")
    started = _day(d.get("started"), date.today().isoformat())
    ended = _day(d.get("ended"), "") if outcome != "open" else ""
    row = {"id": _uid(), "source": "manual", "account_id": str(d.get("account_id", "") or ""), "firm": firm,
           "preset_key": str(d.get("preset_key", "") or ""), "size": _num(d.get("size", 0), 0.0), "started": started,
           "ended": ended, "outcome": outcome,
           "bust_cause": normalize_bust_cause(d.get("bust_cause")) if outcome == "failed" else "",
           "fee": fee, "strategy": str(d.get("strategy", "") or "")[:120], "days": None,
           "note": str(d.get("note", "") or "")[:300]}
    if ended:
        try:
            row["days"] = (date.fromisoformat(ended) - date.fromisoformat(started)).days
        except ValueError:
            row["days"] = None
    with _LOCK:
        data = load()
        data["attempts"].append(row)
        if log_cost and fee > 0:
            data["costs"].append({"id": _uid(), "date": started, "kind": "purchase", "firm": firm, "account_id": row["account_id"],
                                  "amount": fee, "note": "back-filled attempt"})
        _write(data)
    return row


def delete_attempt(attempt_id: str) -> bool:
    with _LOCK:
        data = load()
        n = len(data["attempts"])
        data["attempts"] = [a for a in data["attempts"] if a.get("id") != attempt_id]
        if len(data["attempts"]) == n:
            return False
        _write(data)
        return True


# ------------------------------------------------------------------------------------------------ simulation runs
MAX_SIM_RUNS = 200


def add_sim_run(kind: str, summary: dict, *, firm: str = "", preset_key: str = "", strategy: str = "") -> dict:
    row = {"id": _uid(), "kind": str(kind)[:30], "created_at": date.today().isoformat(), "firm": str(firm)[:80],
           "preset_key": str(preset_key)[:60], "strategy": str(strategy)[:120], "summary": summary}
    with _LOCK:
        data = load()
        data["sim_runs"].append(row)
        data["sim_runs"] = data["sim_runs"][-MAX_SIM_RUNS:]
        _write(data)
    return row


# ------------------------------------------------------------------------------------------------ settings
def get_setting(key: str, default: Any = None) -> Any:
    return load()["settings"].get(key, default)


def set_setting(key: str, value: Any) -> None:
    with _LOCK:
        data = load()
        data["settings"][key] = value
        _write(data)
