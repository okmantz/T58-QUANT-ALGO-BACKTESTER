"""
Trade book: the record of exactly which trades the relay took.

Two sources are merged into one list, newest first:

  * ``paper``  -- the T58 paper account (fills at the alert's price).
  * ``broker`` -- the demo broker account (fills as the broker reported them;
                  when a broker does not return a fill price the alert price is
                  used and the row says so in ``price_source``).

Each row: relay, source, symbol, side, qty, entry, exit, pnl (net of the
instrument's standard commission), opened_at, closed_at, why, price_source.
Broker rows are written by the engine's broker leg; paper rows come from
PaperLedger's closed list. State is a small JSON file per relay.
"""
from __future__ import annotations

import csv
import io
import json
import threading
from datetime import datetime, timezone

from app.webhook_relay.paper import PaperLedger, instrument_economics, relay_dir

_LOCK = threading.RLock()
MAX_ROWS = 5000
CSV_FIELDS = ["source", "symbol", "side", "qty", "entry", "exit", "pnl", "gross", "costs",
              "opened_at", "closed_at", "why", "price_source", "ticket"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BrokerTradeBook:
    def __init__(self, relay_id: str):
        self.path = relay_dir() / f"{relay_id}_broker.json"

    def _load(self) -> dict:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            d.setdefault("open", {})
            d.setdefault("closed", [])
            return d
        except Exception:  # noqa: BLE001
            return {"open": {}, "closed": []}

    def _save(self, d: dict) -> None:
        d["closed"] = d["closed"][-MAX_ROWS:]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d), encoding="utf-8")
        tmp.replace(self.path)

    def reset(self) -> None:
        with _LOCK:
            self.path.unlink(missing_ok=True)

    def record_open(self, *, ticket: str, root: str, symbol: str, side: str, qty: float,
                    entry: float | None, price_source: str) -> None:
        with _LOCK:
            d = self._load()
            d["open"][str(ticket)] = {"ticket": str(ticket), "root": root, "symbol": symbol, "side": side,
                                      "qty": qty, "entry": entry, "price_source": price_source,
                                      "opened_at": _now()}
            self._save(d)

    def record_close(self, *, ticket: str, root: str, symbol: str, side: str, qty: float,
                     broker_entry: float | None, exit_price: float | None, price_source: str,
                     why: str, opened_at: str | None = None) -> dict:
        """Close out a leg. Falls back to the broker-reported entry if we never
        saw the open (e.g. a position that existed before the relay started)."""
        with _LOCK:
            d = self._load()
            leg = d["open"].pop(str(ticket), None) or {}
            entry = leg.get("entry") if leg.get("entry") is not None else broker_entry
            per_point, commission, _ = instrument_economics(root)
            direction = 1 if side == "long" else -1
            if entry is not None and exit_price is not None:
                gross = (exit_price - entry) * direction * qty * per_point
                costs = commission * qty
                pnl = round(gross - costs, 2)
                gross, costs = round(gross, 2), round(costs, 2)
            else:
                gross = costs = pnl = None
            row = {"source": "broker", "ticket": str(ticket), "root": root, "symbol": symbol, "side": side,
                   "qty": qty, "entry": entry, "exit": exit_price, "gross": gross, "costs": costs, "pnl": pnl,
                   "opened_at": leg.get("opened_at") or opened_at, "closed_at": _now(), "why": why,
                   "price_source": price_source}
            d["closed"].append(row)
            self._save(d)
            return row

    def snapshot(self) -> dict:
        with _LOCK:
            return self._load()


def all_trades(relay_id: str) -> list[dict]:
    """Paper + broker closed trades, newest first."""
    rows: list[dict] = []
    paper = PaperLedger(relay_id)._load()
    for t in paper.get("closed", []):
        rows.append({"source": "paper", "symbol": t["root"], "root": t["root"], "side": t["side"],
                     "qty": t["qty"], "entry": t["entry"], "exit": t["exit"], "gross": t.get("gross"),
                     "costs": t.get("costs"), "pnl": t["pnl"], "opened_at": t.get("opened_at"),
                     "closed_at": t.get("closed_at"), "why": t.get("why", ""),
                     "price_source": "alert", "ticket": ""})
    rows.extend(BrokerTradeBook(relay_id).snapshot()["closed"])
    rows.sort(key=lambda r: r.get("closed_at") or "", reverse=True)
    return rows


def open_broker_legs(relay_id: str) -> list[dict]:
    return list(BrokerTradeBook(relay_id).snapshot()["open"].values())


def trades_csv(relay_id: str) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in all_trades(relay_id):
        w.writerow(r)
    return buf.getvalue()
