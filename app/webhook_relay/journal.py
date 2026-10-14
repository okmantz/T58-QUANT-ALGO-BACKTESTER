"""
Per-relay signal journal: one JSON line per alert that reached the relay,
newest last, capped so it can never grow without bound. The broker leg runs
on a background thread (TradingView gives up on a webhook after ~3 seconds),
so its result is patched into the same record afterwards.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone

from app.webhook_relay.paper import relay_dir

_LOCK = threading.RLock()
MAX_RECORDS = 1000


def _path(relay_id: str):
    return relay_dir() / f"{relay_id}_signals.jsonl"


def _read(relay_id: str) -> list[dict]:
    path = _path(relay_id)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write(relay_id: str, rows: list[dict]) -> None:
    path = _path(relay_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows[-MAX_RECORDS:]) + "\n", encoding="utf-8")
    tmp.replace(path)


def record(relay_id: str, **fields) -> str:
    rec_id = uuid.uuid4().hex[:10]
    rec = {"id": rec_id, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
    with _LOCK:
        rows = _read(relay_id)
        rows.append(rec)
        _write(relay_id, rows)
    return rec_id


def patch(relay_id: str, rec_id: str, **fields) -> None:
    with _LOCK:
        rows = _read(relay_id)
        for row in rows:
            if row.get("id") == rec_id:
                row.update(fields)
                break
        _write(relay_id, rows)


def recent(relay_id: str, limit: int = 50) -> list[dict]:
    with _LOCK:
        return list(reversed(_read(relay_id)[-limit:]))


def clear(relay_id: str) -> None:
    with _LOCK:
        _path(relay_id).unlink(missing_ok=True)
