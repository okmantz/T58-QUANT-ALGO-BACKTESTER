"""Portfolio overview: one row per connected account -- firm, live P&L, open and past trades, algo health.

Flask-free so the Deploy page, the Dashboard cards and tests share one source. Trade history comes from the forward-test /
live-deploy journal (``sessions.mt5_login`` ties a session to the account that traded it); the running session's balance and
equity come from its in-memory status. Everything is read-only.
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Optional

_PAST_LIMIT = 8


def _health(status: Optional[dict], n_closed: int) -> dict:
    """Plain-language algo health: ok / watch / halted / idle, with the reason."""
    if not status:
        return {"state": "idle", "label": "Not running", "why": "No strategy is deployed on this account right now."}
    if status.get("halted_reason") or status.get("alert"):
        return {"state": "halted", "label": "Halted",
                "why": str(status.get("alert") or status.get("halted_reason"))}
    if status.get("drift_flag"):
        return {"state": "watch", "label": "Drifting", "why": str(status["drift_flag"])}
    if status.get("running") and not status.get("connected"):
        return {"state": "watch", "label": "Reconnecting", "why": "The session is running but the broker link is down."}
    if status.get("running"):
        return {"state": "ok", "label": "Healthy",
                "why": "Running and behaving like its backtest." if n_closed >= 20 else
                       "Running. Fewer than 20 closed trades so far, so health is not yet measurable."}
    return {"state": "idle", "label": "Stopped", "why": "The last session has ended."}


def _trade_dict(t) -> dict:
    return {"direction": "long" if t.direction > 0 else "short", "volume": t.volume, "entry_price": t.entry_price,
            "entry_time": t.entry_time, "exit_price": t.exit_price, "exit_time": t.exit_time, "pnl": t.pnl,
            "status": t.status}


def build_portfolio(accounts: Iterable[Any], journal, live_status: Optional[dict] = None, live_login: str = "",
                    extra_statuses: Optional[dict] = None) -> dict:
    """``accounts``: LiveAccount-like objects (id, nickname, platform, firm_name, login, is_usable).
    ``live_status``: the running session's status as a dict (or None). ``live_login``: login the session trades.
    ``extra_statuses``: {saved-account id: status dict} for every account in a trade-copier group (leader and followers).
    Copier sessions are journaled under the saved-account id, so several accounts on one broker login stay separate."""
    extra_statuses = extra_statuses or {}
    rows: list[dict] = []
    sessions_by_login: dict[str, list[tuple]] = {}
    try:
        for r in journal._conn.execute(  # noqa: SLF001 - read-only query on our own journal
                "SELECT id, strategy_filename, symbol, mt5_login, started_at, ended_at FROM sessions ORDER BY started_at DESC"):
            sessions_by_login.setdefault(str(r[3] or ""), []).append(r)
    except Exception:  # noqa: BLE001 - no journal yet is a normal first-run state
        pass
    for a in accounts:
        login = str(getattr(a, "login", "") or "")
        acct_id = str(getattr(a, "id", "") or "")
        sess = list(sessions_by_login.get(login, [])) + (list(sessions_by_login.get(acct_id, [])) if acct_id and acct_id != login else [])
        sess.sort(key=lambda r: r[4] or 0, reverse=True)
        open_t: list[dict] = []
        past: list[dict] = []
        net = 0.0
        n_closed = wins = 0
        for s in sess:
            try:
                trades = journal.all_trades(s[0])
            except Exception:  # noqa: BLE001
                trades = []
            for t in trades:
                if t.status == "open":
                    open_t.append({**_trade_dict(t), "strategy": s[1], "symbol": s[2]})
                elif t.pnl is not None:
                    n_closed += 1
                    net += t.pnl
                    wins += 1 if t.pnl > 0 else 0
                    past.append({**_trade_dict(t), "strategy": s[1], "symbol": s[2]})
        past.sort(key=lambda d: d.get("exit_time") or 0, reverse=True)
        is_live_acct = bool(live_status) and login and login == str(live_login or "")
        st = live_status if is_live_acct else None
        if acct_id and acct_id in extra_statuses:
            st = extra_statuses[acct_id]       # copier group member: its own status, not the single-session one
        rows.append({
            "id": getattr(a, "id", ""), "nickname": getattr(a, "nickname", "") or login or "Account",
            "firm": getattr(a, "firm_name", "") or "", "platform": getattr(a, "platform", ""),
            "login": login, "usable": bool(getattr(a, "is_usable", False)),
            "running": bool(st and st.get("running")),
            "balance": (st or {}).get("balance"), "equity": (st or {}).get("equity"),
            "floating": ((st["equity"] - st["balance"]) if st and st.get("equity") is not None and st.get("balance") is not None else None),
            "net_pnl": net, "n_closed": n_closed, "win_rate": (wins / n_closed * 100.0) if n_closed else None,
            "open_trades": open_t, "past_trades": past[:_PAST_LIMIT],
            "strategies": sorted({s[1] for s in sess})[:6],
            "health": _health(st, n_closed),
        })
    total = sum(r["net_pnl"] for r in rows)
    has_usable = any(r["usable"] for r in rows)
    has_history = any(r["n_closed"] or r["open_trades"] or r["running"] for r in rows)
    return {"accounts": rows, "total_net_pnl": total, "generated_at": time.time(),
            "eligible": bool(has_usable and has_history),
            "eligible_reason": ("" if has_usable and has_history else
                                "Connect a trading account on the Deploy page" if not has_usable else
                                "Start a forward test or live deploy on a connected account")}
