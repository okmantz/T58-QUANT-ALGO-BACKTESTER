"""Notification inbox + the Sunday weekly digest.

The inbox is a small local list (data/inbox.json, carried by the account vault) shown behind the bell icon beside the profile
picture. Today it holds one kind of message: the weekly digest, written once per ISO week on (or after) Sunday. Every line in a
digest is computed from results the app saved -- run history, per-strategy reports, the strategy graveyard and the Champion
Board -- so it never pads or invents. If the account has email notifications configured, the same digest is also emailed.

  what you tested     experiments finished in the last 7 days, by tool, with pass / fail counts
  what died           strategies rejected into the graveyard and tests that failed, with the recorded cause
  closest to READY    the Champion Board strategies furthest along (and what each one still needs)
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
MAX_MESSAGES = 120


def _path() -> Path:
    from app.data.storage import get_app_base_dir
    p = get_app_base_dir() / "data" / "inbox.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load() -> dict[str, Any]:
    try:
        d = json.loads(_path().read_text(encoding="utf-8"))
        if isinstance(d, dict):
            d.setdefault("messages", [])
            return d
    except Exception:  # noqa: BLE001
        pass
    return {"messages": []}


def _save(d: dict[str, Any]) -> None:
    p = _path()
    tmp = p.with_suffix(".tmp")
    d["messages"] = d["messages"][:MAX_MESSAGES]
    tmp.write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


# ------------------------------------------------------------------------------------------------------------ digest content
def _week_key(dt: datetime) -> str:
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def _ts(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


def build_weekly_digest(now: float | None = None) -> dict[str, Any]:
    """Structured digest: {title, subtitle, sections:[{id,title,headline,items:[str]}], empty}."""
    from app import briefing
    from app.reports.run_history import load_runs
    from app.scoring import champion_board

    now = now or time.time()
    since = now - 7 * 86400
    entries = [e for e in briefing._index_entries() if since <= float(e.get("recorded_at") or 0) <= now]
    runs = [r for r in load_runs() if since <= _ts(r.get("timestamp")) <= now]
    sections: list[dict[str, Any]] = []

    # ---- tested
    by_tool: dict[str, list] = {}
    for e in entries:
        by_tool.setdefault(briefing._tool_label(e.get("tool")), []).append(e)
    mirrored = sum(1 for e in entries if e.get("tool") in ("full_pipeline", "run_and_report"))
    extra = max(0, len(runs) - mirrored)
    if extra:
        by_tool.setdefault("backtest", []).extend({"passed": r.get("single_run_passed")} for r in runs[:extra])
    total = sum(len(v) for v in by_tool.values())
    if total:
        items = []
        for t, es in sorted(by_tool.items(), key=lambda kv: -len(kv[1])):
            judged = [x for x in es if x.get("passed") is not None]
            p = sum(1 for x in judged if x.get("passed"))
            items.append(f"{briefing._plural(len(es), t)}" + (f" -- {p} passed, {len(judged) - p} failed" if judged else ""))
        names = sorted({str(e.get("strategy_name") or "").split(" -- ")[0] for e in entries if e.get("strategy_name")})
        if names:
            items.append("Strategies touched: " + ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else ""))
        sections.append({"id": "tested", "title": "What you tested", "headline": f"{briefing._plural(total, 'experiment')} this week", "items": items})

    # ---- died
    died: list[str] = []
    n_dead = 0
    causes: dict[str, int] = {}
    d = briefing._base() / "data" / "evolution"
    if d.exists():
        for p in d.glob("strategy_graveyard__*__*.jsonl"):
            try:
                if p.stat().st_mtime < since:
                    continue
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.startswith('{"candidate_id": "seed-'):
                            continue
                        try:
                            rec = json.loads(s)
                        except Exception:  # noqa: BLE001
                            continue
                        at = _ts(rec.get("recorded_at") or rec.get("timestamp") or rec.get("at"))
                        if at and at < since:
                            continue
                        n_dead += 1
                        if rec.get("primary_failure"):
                            causes[str(rec["primary_failure"])] = causes.get(str(rec["primary_failure"]), 0) + 1
            except Exception:  # noqa: BLE001
                continue
    failed = [e for e in entries if e.get("passed") is False]
    if n_dead or failed:
        if causes:
            top = sorted(causes.items(), key=lambda kv: -kv[1])[:3]
            died.append("Most common causes of death: " + "; ".join(f"{c} ({n})" for c, n in top))
        for e in failed[-4:]:
            died.append(f"{str(e.get('strategy_name') or 'strategy')[:50]} failed {briefing._tool_label(e.get('tool'))}: {str(e.get('summary') or '')[:80]}".rstrip(": "))
        head = []
        if n_dead:
            head.append(f"{briefing._plural(n_dead, 'strategy', 'strategies')} buried in the graveyard")
        if failed:
            head.append(f"{briefing._plural(len(failed), 'test')} failed")
        sections.append({"id": "died", "title": "What died", "headline": " and ".join(head), "items": died})

    # ---- closest to READY
    try:
        rows = champion_board.list_board()
    except Exception:  # noqa: BLE001
        rows = []
    live = [r for r in rows if r["promotion"]["next_stage"] is not None]
    if live:
        order = champion_board.PROMOTION_STAGES
        live.sort(key=lambda r: (order.index(r["promotion"]["stage"]), r["strength"] if r.get("strength") is not None else -1.0), reverse=True)
        items = []
        for r in live[:3]:
            bits = [f"stage: {r['promotion']['stage_title']}"]
            if r.get("eval_pct") is not None:
                bits.append(f"eval pass {r['eval_pct']:.0f}%")
            if r.get("payout_pct") is not None:
                bits.append(f"payout {r['payout_pct']:.0f}%")
            if r.get("oos_pct") is not None:
                bits.append(f"out-of-sample {r['oos_pct']:.0f}%")
            items.append(f"{r['display_name'][:50]} -- " + ", ".join(bits))
        sections.append({"id": "ready", "title": "Closest to READY", "headline": f"{live[0]['display_name'][:50]} is furthest along", "items": items})

    start = datetime.fromtimestamp(since)
    end = datetime.fromtimestamp(now)
    return {"title": f"Your week in the lab ({start:%b %d} - {end:%b %d})", "subtitle": "What you tested, what died, and what is closest to READY.",
            "sections": sections, "empty": not sections,
            "empty_line": "A quiet week: nothing new was saved. Run a test or start a Discovery search and next Sunday's digest will have something to say."}


def digest_text(dg: dict[str, Any]) -> str:
    lines = [dg["title"], ""]
    if dg.get("empty"):
        lines.append(dg["empty_line"])
    for s in dg["sections"]:
        lines.append(f"{s['title']}: {s['headline']}")
        lines += [f"  - {i}" for i in s["items"]]
        lines.append("")
    return "\n".join(lines).strip()


# ------------------------------------------------------------------------------------------------------------ inbox API
def add_message(kind: str, title: str, body: dict[str, Any], key: str | None = None) -> dict[str, Any] | None:
    with _LOCK:
        d = _load()
        if key and any(m.get("key") == key for m in d["messages"]):
            return None
        m = {"id": uuid.uuid4().hex[:12], "kind": kind, "key": key, "title": title, "created": time.time(), "read": False, "body": body}
        d["messages"].insert(0, m)
        _save(d)
        return m


def maybe_generate_weekly(now: float | None = None) -> dict[str, Any] | None:
    """Write this week's digest if (a) it is Sunday or later in the ISO week (Mon=1 ... Sun=7 -> Sunday only fires on day 7,
    but a missed Sunday is caught up on the next open) and (b) this ISO week's digest does not exist yet. A fresh install gets
    no empty digest, so the inbox never opens with an empty-week message."""
    now = now or time.time()
    dt = datetime.fromtimestamp(now)
    # The digest for a week is due from Sunday 00:00. If today is Mon-Sat, the most recent Sunday's digest covers the week before.
    last_sunday = dt - timedelta(days=(dt.weekday() + 1) % 7)
    anchor = last_sunday.replace(hour=8, minute=0, second=0, microsecond=0)
    if anchor.timestamp() > now:
        anchor -= timedelta(days=7)
    key = "weekly-" + anchor.strftime("%Y-%m-%d")
    with _LOCK:
        d = _load()
        if any(m.get("key") == key for m in d["messages"]):
            return None
        if not d.get("installed_at"):
            d["installed_at"] = now
            _save(d)
    dg = build_weekly_digest(now)
    if dg["empty"] and not _load()["messages"]:
        return None                                          # nothing to say yet: stay quiet until there is real activity
    msg = add_message("weekly_digest", dg["title"], dg, key=key)
    if msg:
        _email_digest(dg)
    return msg


def _email_digest(dg: dict[str, Any]) -> None:
    try:
        import smtplib
        from email.message import EmailMessage
        from app.web.notifications import load_notification_settings
        s = load_notification_settings()
        if not s.is_usable:
            return
        em = EmailMessage()
        em["Subject"] = "T58 weekly digest -- " + dg["title"]
        em["From"] = s.smtp_from.strip()
        em["To"] = s.notify_email.strip()
        em.set_content(digest_text(dg))

        def _send() -> None:
            try:
                with smtplib.SMTP(s.smtp_host.strip(), int(s.smtp_port or 587), timeout=15) as srv:
                    srv.starttls()
                    if s.smtp_username.strip():
                        srv.login(s.smtp_username.strip(), s.smtp_password)
                    srv.send_message(em)
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=_send, daemon=True).start()
    except Exception:  # noqa: BLE001
        pass


def list_messages() -> dict[str, Any]:
    d = _load()
    msgs = d["messages"]
    return {"messages": msgs, "unread": sum(1 for m in msgs if not m.get("read"))}


def mark_read(ids: list[str] | None = None) -> None:
    with _LOCK:
        d = _load()
        for m in d["messages"]:
            if ids is None or m["id"] in ids:
                m["read"] = True
        _save(d)


def delete_message(mid: str) -> None:
    with _LOCK:
        d = _load()
        d["messages"] = [m for m in d["messages"] if m["id"] != mid]
        _save(d)
