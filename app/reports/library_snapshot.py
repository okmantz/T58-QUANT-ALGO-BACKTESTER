"""Compact, durable "what did the last Full Pipeline say about this strategy" snapshot, stored on the strategy's own
metadata sidecar as ``pipeline_snapshot``.

Why: the Strategy Library detail panel must show the equity curve, quick stats and the worst-time findings the moment
you open a strategy that has already been through the pipeline -- without re-running anything and without depending on
the in-memory job (which is pruned) or on matching a decorated display name against run_history. The full report files
stay the source of truth; this is the small, fast copy the UI reads.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

EQUITY_POINTS = 160
_STAT_FIELDS = ("net_profit", "total_trades", "win_rate", "profit_factor", "sharpe_ratio", "max_drawdown_pct",
                "expectancy", "average_winner", "average_loser", "average_r")


def _num(v: Any, nd: int = 2):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def _downsample(values, n: int = EQUITY_POINTS) -> list[float]:
    values = [float(v) for v in values]
    if len(values) <= n:
        return [round(v, 2) for v in values]
    step = len(values) / n
    out = [round(values[min(int(i * step), len(values) - 1)], 2) for i in range(n)]
    out[-1] = round(values[-1], 2)
    return out


def build_pipeline_snapshot(result, *, instrument: str = "", report_urls: dict | None = None) -> dict:
    """From a FullPipelineResult (duck-typed: everything is getattr-guarded so a partial result still snapshots)."""
    from app.optimize.time_filter import grid_from_trades
    bt = getattr(result, "final_bt", None) or getattr(result, "baseline_bt", None)
    st = getattr(bt, "statistics", None)
    mc = getattr(result, "final_mc", None)
    sc = getattr(result, "scorecard", None)
    snap: dict[str, Any] = {
        "recorded_at": time.time(),
        "instrument": instrument,
        "verdict": getattr(result, "verdict", None),
        "reasons": [str(r)[:300] for r in (getattr(result, "verdict_reasons", None) or [])[:8]],
        "t58_score": _num(getattr(sc, "score", None), 1),
        "t58_tier": getattr(sc, "tier", None),
        "stats": {k: _num(getattr(st, k, None)) for k in _STAT_FIELDS},
        "eval_pass_probability": _num(getattr(mc, "headline_evaluation_pass_probability", None), 1),
        "first_payout_probability": _num(getattr(mc, "headline_first_payout_probability", None), 1),
        "risk_of_ruin_pct": _num(getattr(mc, "risk_of_ruin_pct", None), 1),
        "reports": dict(report_urls or {}),
    }
    try:
        eq = bt.equity_curve["equity"].astype(float).tolist()
        snap["equity"] = _downsample(eq)
        try:
            ts = bt.equity_curve["timestamp"]
            snap["period"] = [str(ts.iloc[0])[:10], str(ts.iloc[-1])[:10]]
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        snap["equity"] = []
    try:
        pnl, cnt = grid_from_trades(bt.trades)
        snap["heatmap"] = [[round(v, 2) for v in row] for row in pnl]
        snap["heatmap_counts"] = cnt
    except Exception:  # noqa: BLE001
        snap["heatmap"], snap["heatmap_counts"] = None, None
    return snap


def stamp_pipeline_snapshot(library_ref, result, *, instrument: str = "", report_urls: dict | None = None,
                            strategy_name: str = "") -> None:
    """Best-effort: write the snapshot to the library entry and mirror the report files into the strategy's
    consolidated report folder under THIS library identity (run_history only does that when the display name matches)."""
    if not library_ref:
        return
    try:
        from app.strategy.library import save_strategy_metadata
        save_strategy_metadata(library_ref[0], library_ref[1],
                               {"pipeline_snapshot": build_pipeline_snapshot(result, instrument=instrument, report_urls=report_urls)},
                               merge=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.reports import strategy_folder
        rp = getattr(result, "report_paths", None) or {}
        existing = strategy_folder.list_reports(strategy_name, instrument, library_type=library_ref[0], library_filename=library_ref[1])
        now = time.time()
        recent = [e for e in existing if e.get("tool") == "full_pipeline" and now - float(e.get("recorded_at", 0)) < 300]
        if rp and not recent:
            strategy_folder.record_report(
                strategy_name or Path(library_ref[1]).stem, instrument, "full_pipeline",
                library_type=library_ref[0], library_filename=library_ref[1],
                passed=(getattr(result, "verdict", None) == "READY"),
                summary=f"Verdict {getattr(result, 'verdict', '?')}",
                source_files={k: v for k, v in rp.items() if k in ("html", "json")},
            )
    except Exception:  # noqa: BLE001
        pass
