"""Retire a Strategy Library strategy to the Strategy Graveyard.

"Retire" is the deliberate version of a delete: the person has decided a strategy is not worth more time, so we
  1. write it into the graveyard (with the reason they gave and the numbers its last run recorded), so the
     discovery tools stop re-proposing the same idea, and
  2. keep a recoverable copy of its source under ``data/retired_strategies/`` (nothing is lost by accident), then
  3. remove it from the library so the library only shows live candidates.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from app.data.storage import get_app_base_dir
from app.search.graveyard import GraveyardEntry, graveyard_path_for, record_rejection
from app.strategy import library as lib


def retired_dir() -> Path:
    d = get_app_base_dir() / "data" / "retired_strategies"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _num(d: dict, *keys):
    for k in keys:
        v = (d or {}).get(k)
        if isinstance(v, (int, float)) and v == v:
            return float(v)
    return None


def retire_to_graveyard(strategy_type: str, filename: str, reason: str = "") -> dict:
    """Move one library strategy to the graveyard. Returns {"graveyard_path", "backup_path", "entry_id"}.
    Raises ValueError / FileNotFoundError for an unknown strategy (same as delete_saved_strategy)."""
    path = lib.resolve_saved_strategy_path(strategy_type, filename)
    meta = lib.load_strategy_metadata(strategy_type, filename) or {}
    last = meta.get("last_run") or {}
    stats = last.get("statistics") if isinstance(last.get("statistics"), dict) else last

    instrument = str(meta.get("market") or meta.get("instrument") or "").strip() or "unknown"
    timeframe = str(meta.get("timeframe") or last.get("timeframe") or "unknown").strip() or "unknown"
    reason = (reason or "").strip() or "Retired by hand from the Strategy Library."
    stamp = time.strftime("%Y%m%d-%H%M%S")
    entry_id = f"{Path(filename).stem}-retired-{stamp}"

    backup = retired_dir() / strategy_type
    backup.mkdir(parents=True, exist_ok=True)
    backup_path = backup / f"{stamp}__{path.name}"
    shutil.copy2(path, backup_path)
    meta_path = lib._metadata_path(path.parent, path.name)
    if meta_path.exists():
        shutil.copy2(meta_path, backup / f"{stamp}__{meta_path.name}")

    entry = GraveyardEntry(
        candidate_id=entry_id,
        family=str(meta.get("family") or Path(filename).stem),
        generation=None,
        stage_died="retired",
        reason=reason,
        notes=[f"Retired from the {strategy_type} library at status '{meta.get('status', 'draft')}'.",
               f"Source kept at {backup_path}"],
        n_trades=int(_num(stats, "trades", "total_trades", "n_trades") or 0) or None,
        net_profit=_num(stats, "net_profit"),
        profit_factor=_num(stats, "profit_factor"),
        instrument=instrument,
        timeframe=timeframe,
    )
    gpath = graveyard_path_for(instrument, timeframe)
    record_rejection(entry, gpath)
    lib.delete_saved_strategy(strategy_type, filename)
    return {"graveyard_path": str(gpath), "backup_path": str(backup_path), "entry_id": entry_id}
