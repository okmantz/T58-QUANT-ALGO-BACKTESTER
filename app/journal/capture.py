"""
Auto-capture: every closed trade from Forward Test, the Demo/paper relay, MT5 or any connected platform becomes a journal entry.

Each entry carries Strategy, Instrument, entry/exit, P&L, R-multiple, trading session, market regime, setup classification, rule
violations, a screenshot and notes. Entries are keyed by ``externalId`` so capturing the same trade twice updates it instead of
duplicating it (so a backfill and the live hook can both run safely). Anything that cannot be known is left as "unclassified" or
empty -- the journal never invents a regime or a violation.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.web import journal_store as js

ET = ZoneInfo("America/New_York")


# ------------------------------------------------------------------------------------------------ classification
def session_of(ts: datetime | None) -> str:
    """Trading session by US/Eastern clock (naive timestamps are UTC)."""
    if ts is None:
        return "unclassified"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    e = ts.astimezone(ET)
    m = e.hour * 60 + e.minute
    if 570 <= m < 660:
        return "NY open (9:30-11:00)"
    if 660 <= m < 840:
        return "NY midday (11:00-14:00)"
    if 840 <= m < 960:
        return "NY close (14:00-16:00)"
    if 180 <= m < 570:
        return "London / pre-market"
    if m >= 960 or m < 180:
        return "Overnight / Asia"
    return "unclassified"


def regime_from_bars(df, ts=None) -> str:
    """Volatility / trend regime from the bars before ``ts``: high-vol, low-vol, trending or ranging. 'unclassified' if too few bars."""
    try:
        d = df if ts is None else df[df["timestamp"] <= ts]
        d = d.tail(600)
        if len(d) < 60:
            return "unclassified"
        rng = (d["high"] - d["low"]).astype(float)
        recent, base = rng.tail(20).mean(), rng.mean()
        if base > 0 and recent >= 1.5 * base:
            return "high volatility"
        if base > 0 and recent <= 0.6 * base:
            return "low volatility"
        c = d["close"].astype(float).tail(60)
        path = c.diff().abs().sum()
        er = abs(c.iloc[-1] - c.iloc[0]) / path if path > 0 else 0.0
        return "trending" if er >= 0.3 else "ranging"
    except Exception:  # noqa: BLE001
        return "unclassified"


def classify_setup(*, pnl: float, risk: float, why: str = "", held_minutes: float | None = None) -> str:
    """Setup classification from how the trade resolved. It labels the outcome type, not the pattern that triggered it."""
    w = (why or "").lower()
    if "reversed" in w:
        return "reversal exit"
    if "flatten" in w or "guardian" in w:
        return "forced flatten"
    if "stop" in w or "sl" == w:
        return "stopped out"
    if "target" in w or "tp" == w:
        return "target hit"
    if held_minutes is not None and held_minutes <= 15:
        return "scalp"
    if held_minutes is not None and held_minutes >= 240:
        return "swing / long hold"
    return "intraday trend/move"


def detect_violations(*, risk: float, planned_risk: float | None, pnl: float, losses_in_row: int, minutes_since_last_loss: float | None,
                      daily_pnl_before: float, daily_limit: float | None, in_news_window: bool = False) -> list[str]:
    v = []
    if planned_risk and risk > planned_risk * 1.25:
        v.append(f"Oversized: risked ${risk:,.0f} vs ${planned_risk:,.0f} planned")
    if losses_in_row >= 3:
        v.append(f"Traded after {losses_in_row} consecutive losses")
    if minutes_since_last_loss is not None and minutes_since_last_loss < 5 and losses_in_row >= 1:
        v.append("Re-entered within 5 minutes of a loss (revenge pattern)")
    if daily_limit and daily_pnl_before <= -0.8 * daily_limit:
        v.append("Took a trade with over 80% of the daily loss limit used")
    if in_news_window:
        v.append("Traded inside a news restriction window")
    return v


def _r(pnl: float, risk: float):
    return round(pnl / risk, 2) if risk and risk > 0 else None


# ------------------------------------------------------------------------------------------------ building + writing entries
def build_entry(*, external_id: str, source: str, strategy: str, instrument: str, direction: int | str, entry: float | None,
                exit_price: float | None, entry_time: datetime | None, exit_time: datetime | None, pnl: float,
                risk_amount: float = 0.0, volume: float | None = None, regime: str = "unclassified", why: str = "",
                notes: str = "", account: str = "", backtest_avg_r: float | None = None, violations: list | None = None,
                stop: float | None = None, tp: float | None = None) -> dict:
    held = ((exit_time - entry_time).total_seconds() / 60.0) if (entry_time and exit_time) else None
    d = (exit_time or entry_time or datetime.now(timezone.utc))
    side = direction if isinstance(direction, str) else ("long" if direction >= 0 else "short")
    return {
        "id": external_id.replace(":", "-"), "externalId": external_id, "source": source, "strategy": strategy, "instrument": instrument,
        "direction": side, "entry": entry or 0.0, "exitPrice": exit_price, "stop": stop or 0.0, "tp": tp or 0.0,
        "date": d.date().isoformat(), "entryTime": entry_time.isoformat() if entry_time else "", "exitTime": exit_time.isoformat() if exit_time else "",
        "pnl": pnl, "riskAmount": risk_amount, "rMultiple": _r(pnl, risk_amount), "volume": volume,
        "session": session_of(entry_time), "regime": regime, "setup": classify_setup(pnl=pnl, risk=risk_amount, why=why, held_minutes=held),
        "violations": violations or [], "notes": notes or (f"Auto-captured from {source}." + (f" Exit: {why}." if why else "")),
        "backtestAvgR": backtest_avg_r, "account": account, "tags": [], "checklist": {},
    }


def _existing_by_external() -> dict:
    return {t["externalId"]: t for t in js.load()["trades"] if t.get("externalId")}


def _enrich_with_history(entry: dict, prior: list[dict], daily_limit: float | None = None) -> dict:
    """Rule violations that need the trades before this one (loss streak, revenge re-entry, daily-limit use)."""
    day = [t for t in prior if t.get("date") == entry["date"]]
    streak = 0
    for t in sorted(prior, key=lambda x: x.get("exitTime") or x.get("date", ""), reverse=True):
        if (t.get("pnl") or 0) < 0:
            streak += 1
        else:
            break
    since = None
    if prior and entry.get("entryTime"):
        last = sorted(prior, key=lambda x: x.get("exitTime") or "")[-1]
        try:
            if last.get("exitTime") and (last.get("pnl") or 0) < 0:
                since = (datetime.fromisoformat(entry["entryTime"]) - datetime.fromisoformat(last["exitTime"])).total_seconds() / 60.0
        except ValueError:
            since = None
    viol = detect_violations(risk=entry["riskAmount"], planned_risk=None, pnl=entry["pnl"], losses_in_row=streak,
                             minutes_since_last_loss=since, daily_pnl_before=sum(t.get("pnl") or 0 for t in day), daily_limit=daily_limit)
    entry["violations"] = list(dict.fromkeys((entry.get("violations") or []) + viol))
    return entry


def save_entry(entry: dict, *, daily_limit: float | None = None, screenshot: bool = True) -> dict:
    prior = [t for t in js.load()["trades"] if t.get("strategy") == entry.get("strategy") and t.get("externalId") != entry["externalId"]
             and (t.get("exitTime") or "") <= (entry.get("entryTime") or "9999")]
    entry = _enrich_with_history(entry, prior, daily_limit)
    existing = _existing_by_external().get(entry["externalId"])
    if existing:  # keep what the trader typed by hand
        for k in ("notes", "tags", "checklist", "playbookId", "screenshot"):
            if existing.get(k):
                entry[k] = existing[k]
    if screenshot and not entry.get("screenshot") and not entry.get("screenshotData"):
        png = render_trade_png(entry)
        if png:
            entry["screenshotData"] = png
    return js.upsert_trade(entry)


def render_trade_png(entry: dict) -> str | None:
    """A small auto-generated trade card (entry/exit/P&L/R) as a PNG data URL. None if matplotlib is unavailable."""
    try:
        import base64
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 2.2), dpi=110)
        fig.patch.set_facecolor("#0e1116")
        ax.set_facecolor("#0e1116")
        e, x = entry.get("entry") or 0.0, entry.get("exitPrice")
        col = "#3FD1A0" if (entry.get("pnl") or 0) >= 0 else "#E5484D"
        if x is not None:
            ax.plot([0, 1], [e, x], color=col, lw=2.5, marker="o")
            ax.set_ylim(min(e, x) - abs(x - e) * 0.5 - 1e-9, max(e, x) + abs(x - e) * 0.5 + 1e-9)
        ax.set_xticks([0, 1], ["entry", "exit"], color="#9aa4b2", fontsize=8)
        ax.tick_params(axis="y", colors="#9aa4b2", labelsize=7)
        for s in ax.spines.values():
            s.set_color("#2a313c")
        r = entry.get("rMultiple")
        ax.set_title(f"{entry.get('instrument','')} {entry.get('direction','').upper()}   P&L {entry.get('pnl',0):+,.2f}"
                     + (f"   {r:+.2f}R" if r is not None else ""), color="#e6e9ef", fontsize=9, loc="left")
        fig.text(0.99, 0.02, "T58", color="#5b6677", fontsize=7, ha="right")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ sources
def _utc(ts: float | None):
    return datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None


def capture_forward_trade(db_path, trade_id: int, *, backtest_avg_r: float | None = None, regime: str = "unclassified") -> dict | None:
    """One closed Forward Test / live-session trade -> a journal entry."""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        r = con.execute("SELECT t.*, s.strategy_type, s.strategy_filename, s.symbol, s.mt5_login FROM trades t "
                        "JOIN sessions s ON s.id=t.session_id WHERE t.id=? AND t.status='closed'", (trade_id,)).fetchone()
    finally:
        con.close()
    if r is None:
        return None
    risk = abs((r["entry_price"] - r["sl_price"]) * r["volume"]) if r["sl_price"] else 0.0
    why = "stop" if (r["sl_price"] and r["exit_price"] is not None and abs(r["exit_price"] - r["sl_price"]) < 1e-9) else (
        "target" if (r["tp_price"] and r["exit_price"] is not None and abs(r["exit_price"] - r["tp_price"]) < 1e-9) else "")
    e = build_entry(external_id=f"ft:{r['session_id']}:{r['id']}", source="Forward Test", strategy=r["strategy_filename"] or r["strategy_type"],
                    instrument=r["symbol"], direction=r["direction"], entry=r["entry_price"], exit_price=r["exit_price"],
                    entry_time=_utc(r["entry_time"]), exit_time=_utc(r["exit_time"]), pnl=r["pnl"] or 0.0, risk_amount=risk,
                    volume=r["volume"], regime=regime, why=why, account=str(r["mt5_login"] or ""), backtest_avg_r=backtest_avg_r,
                    stop=r["sl_price"], tp=r["tp_price"])
    return save_entry(e)


def sync_forward_test(db_path=None) -> int:
    """Backfill every closed Forward Test trade (idempotent). Returns how many entries were written or refreshed."""
    from app.forward_test.journal import _db_path
    p = db_path or _db_path()
    if not p.exists():
        return 0
    con = sqlite3.connect(str(p))
    try:
        ids = [x[0] for x in con.execute("SELECT id FROM trades WHERE status='closed' ORDER BY id")]
    finally:
        con.close()
    return sum(1 for i in ids if capture_forward_trade(p, i))


def capture_relay_trade(row: dict, *, relay_id: str = "") -> dict | None:
    """One closed trade from the webhook relay's paper or broker book (Demo, connected platform)."""
    if row.get("pnl") is None:
        return None

    def _dt(s):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None
        except ValueError:
            return None
    ext = f"relay:{relay_id}:{row.get('source')}:{row.get('ticket') or ''}:{row.get('closed_at')}"
    e = build_entry(external_id=ext, source="Demo" if row.get("source") == "paper" else "Broker", strategy=relay_id or "Webhook relay",
                    instrument=row.get("root") or row.get("symbol", ""), direction=row.get("side", "long"), entry=row.get("entry"),
                    exit_price=row.get("exit"), entry_time=_dt(row.get("opened_at")), exit_time=_dt(row.get("closed_at")),
                    pnl=float(row["pnl"]), volume=row.get("qty"), why=row.get("why", ""))
    return save_entry(e)


def sync_relay(relay_id: str) -> int:
    from app.webhook_relay.trade_book import all_trades
    return sum(1 for r in all_trades(relay_id) if capture_relay_trade(r, relay_id=relay_id))
