"""
Trading Journal storage + analytics (modelled on okmantz/T58-Edge-Analytics).

Single-user JSON file at <app dir>/data/journal/journal.json:
    settings: displayName, accountBalance, playbooks[{id,title,sections[{id,title,items[{id,text}]}]}],
              tagGroups[{id,name,items[{id,name,color}]}]
    trades:   [{id,date,instrument,direction,playbookId,checklist{itemId:bool},tags[tagId],entry,stop,tp,riskAmount,
                pnl,notes,screenshot,compliancePct}]
    screenshots: <app dir>/data/journal/screenshots/<name>.png|jpg|gif|webp -- one optional chart screenshot per trade;
                the trade only stores the file name (`screenshot`, "" when there is none).

The seed playbook and tag groups are the ones from the Edge Analytics screenshots (M15 Pro Trend Intraday Model;
Setups / Area of Interest / Trend Phase / Structure Flow / Execution Window 0200-1200 + 1800-2200 / Instrument incl. ES1!).
`journal_stats` computes every number the Analytics page shows and returns the inputs next to each result so the
clickable-metric popups can show a real calculation breakdown.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import threading
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from app.data.storage import get_app_base_dir

TAG_COLORS = ["#E8A33D", "#3FD1A0", "#5B9BD5", "#B285E0", "#E07BA6",
              "#E0913D", "#F0616D", "#4FC3D9", "#8FCB4A", "#9AA3B2"]
_LOCK = threading.RLock()


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def journal_path() -> Path:
    d = get_app_base_dir() / "data" / "journal"
    d.mkdir(parents=True, exist_ok=True)
    return d / "journal.json"


# ------------------------------------------------------------------ trade screenshots
MAX_SCREENSHOT_BYTES = 12 * 1024 * 1024
_SHOT_NAME_RE = re.compile(r"^[0-9a-f]{16}\.(png|jpg|gif|webp)$")
_DATA_URL_RE = re.compile(r"^data:image/[a-z+.-]+;base64,", re.I)


def screenshots_dir() -> Path:
    d = journal_path().parent / "screenshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sniff_image_ext(raw: bytes) -> str | None:
    """File extension for the bytes' real image type (never trusts the name or the claimed MIME type)."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:3] == b"\xff\xd8\xff":
        return "jpg"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return None


def save_screenshot_bytes(raw: bytes) -> str:
    """Stores a PNG/JPEG/GIF/WebP and returns its file name. Raises ValueError for anything else or anything too big."""
    if not raw:
        raise ValueError("The screenshot was empty.")
    if len(raw) > MAX_SCREENSHOT_BYTES:
        raise ValueError(f"That screenshot is larger than {MAX_SCREENSHOT_BYTES // (1024 * 1024)} MB.")
    ext = _sniff_image_ext(raw)
    if ext is None:
        raise ValueError("Screenshots must be PNG, JPEG, GIF or WebP images.")
    name = f"{uuid.uuid4().hex[:16]}.{ext}"
    (screenshots_dir() / name).write_bytes(raw)
    return name


def save_screenshot_data_url(data_url: str) -> str:
    """Same as save_screenshot_bytes, for a browser `data:image/...;base64,...` string."""
    text = str(data_url or "").strip()
    if not _DATA_URL_RE.match(text):
        raise ValueError("The screenshot wasn't a valid image.")
    try:
        raw = base64.b64decode(text.split(",", 1)[1], validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The screenshot wasn't a valid image.") from exc
    return save_screenshot_bytes(raw)


def screenshot_path(name: str) -> Path | None:
    """The stored file for `name`, or None if the name is malformed (no path tricks) or the file is gone."""
    name = str(name or "")
    if not _SHOT_NAME_RE.match(name):
        return None
    p = screenshots_dir() / name
    return p if p.is_file() else None


def delete_screenshot(name: str) -> None:
    p = screenshot_path(name)
    if p is not None:
        try:
            p.unlink()
        except OSError:
            pass


def default_playbooks() -> list[dict]:
    def sec(title: str, *items: str) -> dict:
        return {"id": _uid(), "title": title, "items": [{"id": _uid(), "text": t} for t in items]}
    return [{
        "id": _uid(), "title": "M15 Pro Trend Intraday Model",
        "sections": [
            sec("BIAS", "M15 Pro Trend"),
            sec("AREA OF INTEREST", "Strong POI or session liquidity sweep"),
            sec("ENTRY CRITERIA", "M1 ChoCh or BOS aligned with M15", "M1–M5 OB/FVG"),
            sec("ADDITIONAL CONFLUENCES",
                "Supply/demand OB sits inside premium/discount zone of HTF swing structure (50%+ retracement)",
                "Reaction at significant fibonacci level (0.5, 0.618, 0.786)",
                "Price bounces from 20 or 50 EMA (Pro trend aligned with 200 EMA)"),
            sec("MANAGEMENT", "TP must be at least 1:3 for trade to be valid",
                "Move stop to breakeven after break of structural swing high/low I entered from",
                "Close 50% of trade at 50% TP"),
            sec("NEWS", "No trading on session with CPI/NFP/FOMC", "Don't enter before: retail sales / core PCE"),
        ],
    }]


def default_tag_groups() -> list[dict]:
    c = TAG_COLORS

    def grp(name: str, items: list[tuple[str, str]]) -> dict:
        return {"id": _uid(), "name": name, "items": [{"id": _uid(), "name": n, "color": col} for n, col in items]}
    windows = ["0200–0300", "0300–0400", "0400–0500", "0500–0600", "0600–0700", "0700–0800",
               "0800–0900", "0900–1000", "1000–1100", "1100–1200",
               "1800–1900", "1900–2000", "2000–2100", "2100–2200"]
    return [
        grp("Setups", [("NY continuation from LDN POI", c[0]), ("NY sweep LDN", c[1]), ("NY sweep Asia", c[2]),
                       ("Lull sweep LDN, NY continuation", c[3]), ("Lull sweep LDN, NY reversal", c[4]),
                       ("LDN sweep Asia", c[5]), ("Frank sweep LDN continuation", c[6]),
                       ("Frank sweep Asia, LDN continuation", c[7])]),
        grp("Area of Interest", [("Weak (OB without BOS)", c[6]), ("Strong (OB / S&D zone with BOS + impulse)", c[1]),
                                 ("No POI", c[9]), ("Broad (candle cluster)", c[5])]),
        grp("Trend Phase", [("Change of Character", c[2]), ("Trend Confirmation", c[0]), ("Trending", c[1])]),
        grp("Structure Flow", [("Pro Trend Only", c[1]), ("Pro Trend Internal", c[3]), ("Counter Internal", c[6])]),
        grp("Execution Window", [(w, c[i % len(c)]) for i, w in enumerate(windows)]),
        grp("Instrument", [("NAS100", c[2]), ("US30", c[5]), ("EURUSD", c[1]), ("GBPUSD", c[0]),
                           ("XAUUSD", c[6]), ("ES1!", c[0])]),
    ]


def add_tags(group_name: str, names: list[str]) -> list[str]:
    """Add tag names to a journal tag group (created if missing); existing names are skipped. Returns the names actually added."""
    names = [str(n).strip()[:80] for n in names if str(n).strip()]
    if not names:
        return []
    with _LOCK:
        data = load()
        groups = data["settings"].setdefault("tagGroups", [])
        grp = next((g for g in groups if g.get("name") == group_name), None)
        if grp is None:
            grp = {"id": _uid(), "name": group_name, "items": []}
            groups.append(grp)
        have = {i.get("name") for i in grp["items"]}
        added = []
        for n in names:
            if n not in have:
                grp["items"].append({"id": _uid(), "name": n, "color": TAG_COLORS[len(grp["items"]) % len(TAG_COLORS)]})
                have.add(n)
                added.append(n)
        _write(data)
        return added


def default_settings() -> dict:
    return {"displayName": "Owen Mantz", "accountBalance": 10000,
            "playbooks": default_playbooks(), "tagGroups": default_tag_groups()}


def load() -> dict:
    with _LOCK:
        p = journal_path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "settings" in data:
                    data.setdefault("trades", [])
                    return data
            except (OSError, ValueError):
                pass
        data = {"settings": default_settings(), "trades": []}
        _write(data)
        return data


def _write(data: dict) -> None:
    p = journal_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def save_settings(settings: dict) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")
    with _LOCK:
        data = load()
        cur = data["settings"]
        for k in ("displayName", "accountBalance", "playbooks", "tagGroups"):
            if k in settings:
                cur[k] = settings[k]
        try:
            cur["accountBalance"] = float(cur.get("accountBalance") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("accountBalance must be a number") from exc
        _write(data)
        return cur


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def upsert_trade(trade: dict) -> dict:
    """Creates or updates one trade. Screenshot handling: `screenshotData` (a data URL) stores a NEW image and replaces
    any old one; otherwise `screenshot` (the stored file name the page got back earlier) is kept as-is, and an empty
    `screenshot` removes it. The old file is deleted whenever it is replaced or removed."""
    if not isinstance(trade, dict):
        raise ValueError("trade must be an object")
    trade_id = str(trade.get("id") or _uid())
    with _LOCK:
        previous = next((x for x in load()["trades"] if x.get("id") == trade_id), None)
    old_shot = str((previous or {}).get("screenshot") or "")
    if trade.get("screenshotData"):
        shot = save_screenshot_data_url(trade["screenshotData"])
    else:
        asked = str(trade.get("screenshot") or "")
        # a trade can only keep the screenshot it already has -- never a name made up by (or borrowed from) the request
        shot = asked if (asked and asked == old_shot and screenshot_path(asked) is not None) else ""
    t = {
        "id": trade_id,
        "date": str(trade.get("date") or date.today().isoformat())[:10],
        "instrument": str(trade.get("instrument") or "").strip(),
        "direction": "short" if str(trade.get("direction", "long")).lower().startswith("s") else "long",
        "playbookId": trade.get("playbookId") or None,
        "checklist": trade.get("checklist") if isinstance(trade.get("checklist"), dict) else {},
        "tags": [str(x) for x in (trade.get("tags") or [])],
        "entry": _num(trade.get("entry")), "stop": _num(trade.get("stop")), "tp": _num(trade.get("tp")),
        "riskAmount": _num(trade.get("riskAmount")), "pnl": _num(trade.get("pnl")),
        "notes": str(trade.get("notes") or "")[:5000],
        "screenshot": shot,
    }
    cl = t["checklist"]
    t["compliancePct"] = round(100.0 * sum(1 for v in cl.values() if v) / len(cl), 1) if cl else None
    with _LOCK:
        data = load()
        trades = [x for x in data["trades"] if x.get("id") != t["id"]]
        trades.append(t)
        trades.sort(key=lambda x: (x.get("date", ""), x.get("id", "")))
        data["trades"] = trades
        _write(data)
    if old_shot and old_shot != shot:
        delete_screenshot(old_shot)
    return t


def delete_trade(trade_id: str) -> bool:
    with _LOCK:
        data = load()
        n = len(data["trades"])
        gone = [x for x in data["trades"] if x.get("id") == trade_id]
        data["trades"] = [x for x in data["trades"] if x.get("id") != trade_id]
        _write(data)
        removed = len(data["trades"]) != n
    for x in gone:
        delete_screenshot(str(x.get("screenshot") or ""))
    return removed


def journal_stats(trades: list[dict], settings: dict) -> dict:
    """Everything the Analytics page shows, with the inputs behind each number."""
    bal = _num(settings.get("accountBalance"), 0.0)
    ts = sorted(trades, key=lambda x: (x.get("date", ""), x.get("id", "")))
    pnls = [_num(t.get("pnl")) for t in ts]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win, gross_loss = sum(wins), -sum(losses)
    n = len(pnls)
    rs = [(_num(t.get("pnl")) / _num(t.get("riskAmount"))) for t in ts if _num(t.get("riskAmount")) > 0]
    equity, peak, max_dd, max_dd_pct, curve = bal, bal, 0.0, 0.0, []
    for t, p in zip(ts, pnls):
        equity += p
        peak = max(peak, equity)
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = (dd / peak * 100.0) if peak > 0 else 0.0
        curve.append({"date": t.get("date"), "equity": round(equity, 2), "cum_pnl": round(equity - bal, 2)})
    tag_names = {i["id"]: i["name"] for g in settings.get("tagGroups", []) for i in g.get("items", [])}
    by_tag: dict[str, dict] = {}
    for t, p in zip(ts, pnls):
        for tid in t.get("tags", []):
            row = by_tag.setdefault(tid, {"tag": tag_names.get(tid, tid), "pnl": 0.0, "trades": 0})
            row["pnl"] += p
            row["trades"] += 1
    comp = [t["compliancePct"] for t in ts if t.get("compliancePct") is not None]
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (None if gross_win == 0 else float("inf"))
    win_rate = (len(wins) / n * 100.0) if n else 0.0
    avg_win = (gross_win / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    return {
        "trades": n,
        "total_pnl": round(sum(pnls), 2),
        "win_rate": round(win_rate, 1),
        "wins": len(wins), "losses": len(losses), "breakeven": n - len(wins) - len(losses),
        "gross_profit": round(gross_win, 2), "gross_loss": round(gross_loss, 2),
        "profit_factor": None if profit_factor is None else (None if profit_factor == float("inf") else round(profit_factor, 2)),
        "profit_factor_infinite": profit_factor == float("inf"),
        "avg_r": round(sum(rs) / len(rs), 2) if rs else None, "r_trades": len(rs),
        "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "expectancy": round(sum(pnls) / n, 2) if n else 0.0,
        "max_drawdown": round(max_dd, 2), "max_drawdown_pct": round(max_dd_pct, 1),
        "starting_balance": bal, "ending_equity": round(equity, 2),
        "avg_compliance": round(sum(comp) / len(comp), 1) if comp else None,
        "curve": curve,
        "pnl_by_tag": sorted(({**v, "pnl": round(v["pnl"], 2)} for v in by_tag.values()), key=lambda r: -r["pnl"]),
    }
