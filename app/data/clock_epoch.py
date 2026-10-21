"""One-time reset of everything learned under the misaligned session clock (report 2026-10-20, P0 step 5).

Until the exchange-clock fix, session / gap / opening-range / event / VWAP families read UTC stamps as exchange time, so every
verdict those families ever got was measured at the wrong hours. Memory built on those verdicts is not evidence:

* Family Health marks a family a "dead end" after 30 tests with zero Stage-3 passes (``app/search/family_health.py``);
* the Strategy Graveyard blocks regeneration of rejected parameter neighbourhoods;
* the pre-registered runner imported files with no offset, so any NOT SUPPORTED it printed is not a result.

``apply_clock_epoch()`` archives (renames with a suffix -- recoverable, never deleted) the first two, sets aside the stale
pre-registered results, and KEEPS the trial ledger (search effort is real, and the deflated-Sharpe penalty must still see it) while
recording that earlier trials ran on a misaligned clock. It is idempotent: a marker file records that this epoch was applied, so it
runs once per install. ``ensure_clock_epoch()`` is the cheap startup hook; set ``T58_SKIP_CLOCK_EPOCH=1`` to disable it (tests).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

CLOCK_EPOCH_ID = "exchange-clock-2026-10-20"
STALE_SUFFIX = ".pre-clock-fix"

# Builders / families that read a clock time or a day boundary (report P0; the list is informational -- the reset archives ALL
# history because a dead-end verdict for any family may have been decided on cross-family search budget that was mis-timed).
CLOCK_DEPENDENT_FAMILIES = (
    "session_time_effect", "prev_day_range_breakout", "overnight_gap_fade", "range_midpoint_fade",
    "opening_range_retest_confirmation", "gap_and_go", "session_gated_liquidity_sweep", "session_extreme_fade",
    "ib_contraction_breakout", "session_range_breakout", "opening_momentum_burst", "overnight_gap_fade_atr_gated",
    "ny_lunch_vwap_band_fade", "overnight_fade", "overnight_trend", "daily_pullback", "daily_mom", "tom", "opex",
    "preholiday", "vix_dip",
)


def _root() -> Path:
    from app.data.storage import get_app_base_dir
    return get_app_base_dir()


def marker_path(root: Path | None = None) -> Path:
    return (root or _root()) / "reports" / "clock_epoch.json"


def status(root: Path | None = None) -> dict:
    p = marker_path(root)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"applied": False, "epoch": CLOCK_EPOCH_ID}
    return {"applied": data.get("epoch") == CLOCK_EPOCH_ID, "epoch": CLOCK_EPOCH_ID, **data}


def _archive(path: Path) -> bool:
    try:
        target = path.with_name(path.name + STALE_SUFFIX)
        if target.exists():
            target.unlink()
        path.rename(target)
        return True
    except Exception:  # noqa: BLE001 -- a locked / missing file must not abort the whole reset
        return False


def apply_clock_epoch(*, root: Path | str | None = None, force: bool = False) -> dict:
    """Archive clock-contaminated memory. Returns a summary dict (also written to ``reports/clock_epoch.json``)."""
    base = Path(root) if root else _root()
    if not force and status(base).get("applied"):
        return {"applied_now": False, **status(base)}

    summary: dict = {"epoch": CLOCK_EPOCH_ID, "applied_at": time.time(), "families_flagged": list(CLOCK_DEPENDENT_FAMILIES)}

    # 1. family health: archives search_*.db and tested_candidates.jsonl (existing, recoverable '.pre-reset')
    try:
        from app.search.family_health import reset_family_health
        summary["family_health"] = reset_family_health(base / "reports" / "search", base / "data" / "evolution")
    except Exception as exc:  # noqa: BLE001
        summary["family_health"] = {"error": str(exc)}

    # 2. graveyard (every instrument-scoped file + the default one)
    n_grave = 0
    evo = base / "data" / "evolution"
    if evo.exists():
        for p in list(evo.glob("strategy_graveyard*.jsonl")):
            if _archive(p):
                n_grave += 1
    summary["graveyard_files_archived"] = n_grave

    # 3. pre-registered runner output: results are not evidence; the registry (fingerprints, timestamps) stays
    n_prereg = 0
    results = base / "reports" / "preregistered_results.json"
    if results.exists() and _archive(results):
        n_prereg += 1
    reg = base / "reports" / "preregistrations.json"
    try:
        if reg.exists():
            data = json.loads(reg.read_text(encoding="utf-8"))
            data.setdefault("clock_epochs", []).append({
                "epoch": CLOCK_EPOCH_ID, "t": time.time(),
                "note": "Results produced before this epoch ran on a misaligned (UTC-as-exchange) session clock; "
                        "re-run every registered test. Registrations and fingerprints are unchanged."})
            reg.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    summary["preregistered_results_set_aside"] = n_prereg

    # 4. trial ledger: KEEP, annotate
    try:
        from app.validation import trial_ledger
        trial_ledger.mark_clock_epoch(CLOCK_EPOCH_ID,
                                      "Trials recorded before this point were evaluated on a misaligned (UTC-as-exchange) session clock. "
                                      "They stay in the count (search effort is real) but their verdicts are not evidence.")
        summary["trial_ledger"] = "kept, epoch recorded"
    except Exception as exc:  # noqa: BLE001
        summary["trial_ledger"] = f"not annotated: {exc}"

    try:
        p = marker_path(base)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return {"applied_now": True, **summary}


def ensure_clock_epoch() -> dict | None:
    """Startup hook: apply the epoch once. Never raises."""
    if os.environ.get("T58_SKIP_CLOCK_EPOCH") == "1":
        return None
    try:
        return apply_clock_epoch()
    except Exception:  # noqa: BLE001
        return None
