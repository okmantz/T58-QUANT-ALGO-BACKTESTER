"""Which stretch of history a number was measured on, and why two pages can honestly show different numbers.

The Full Pipeline tunes and scores a strategy on the DEVELOPMENT window (the first part of the history) and seals the last
part as a holdout so the verdict never sees it. It then re-runs the same final strategy over the FULL history for the report
and the charts. Both are real backtests of the same strategy; they cover different dates, so trades, profit, win rate and
Sharpe differ (for example 390 development trades + 73 holdout trades = 463 on the full history).

Before this module nothing said which window a page's numbers came from: the Library and Champion read the development
window, the Dashboard and the run report read the full history, and the user saw "different numbers for the same strategy".
Now every stored result carries a ``window_info`` block, every surface labels its window, and ``reconcile`` builds the
side-by-side table with a plain-English explanation of each difference.

Pure functions, no I/O.
"""
from __future__ import annotations

import math
from typing import Any, Optional

STAT_KEYS = ("total_trades", "net_profit", "win_rate", "profit_factor", "sharpe_ratio", "max_drawdown_pct")
WINDOW_LABEL = {
    "dev": "Development window",
    "full": "Full history",
}
WINDOW_HELP = {
    "dev": "The part of the history the strategy was tuned and scored on. The verdict, the T58 score and the pass chances are measured here.",
    "full": "The whole history: the development window plus the sealed holdout the verdict never saw. Charts and the run report use this.",
}
# Which pages read which window (kept next to the data so the explanation can never drift from the code that follows it).
SURFACES = {
    "dev": ["Library > Backtest tab", "Champion page", "Strategy Library cards", "Verdict and pass chances"],
    "full": ["Dashboard cards", "Dashboard best Sharpe", "Test page / run report"],
}


def _n(v: Any, nd: int = 2) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def stats_from_statistics(st: Any) -> dict:
    """Plain numbers from a backtest ``statistics`` object (or dict). Missing values stay None, never 0."""
    get = (lambda k: st.get(k)) if isinstance(st, dict) else (lambda k: getattr(st, k, None))
    out = {k: _n(get(k)) for k in STAT_KEYS}
    if out["total_trades"] is not None:
        out["total_trades"] = int(out["total_trades"])
    return out


def _period(bt: Any) -> Optional[list]:
    try:
        ts = bt.equity_curve["timestamp"]
        return [str(ts.iloc[0])[:10], str(ts.iloc[-1])[:10]]
    except Exception:  # noqa: BLE001
        return None


def window_info(dev_bt: Any, full_bt: Any = None) -> dict:
    """The ``window_info`` block stored on a result: development stats, full-history stats, and the holdout trade count."""
    dev = {"stats": stats_from_statistics(getattr(dev_bt, "statistics", None)), "period": _period(dev_bt)}
    full_bt = full_bt if full_bt is not None else dev_bt
    full = {"stats": stats_from_statistics(getattr(full_bt, "statistics", None)), "period": _period(full_bt)}
    d, f = dev["stats"].get("total_trades"), full["stats"].get("total_trades")
    return {"scored_on": "dev", "dev": dev, "full": full, "same_window": full_bt is dev_bt,
            "holdout_trades": (f - d) if (d is not None and f is not None and f >= d) else None}


def _row(key: str, stats: dict, period: Any, *, source: str) -> dict:
    return {"key": key, "label": WINDOW_LABEL[key], "help": WINDOW_HELP[key], "surfaces": SURFACES[key], "source": source,
            "period": period, "trades": stats.get("total_trades"), "net_profit": stats.get("net_profit"), "win_rate": stats.get("win_rate"),
            "profit_factor": stats.get("profit_factor"), "sharpe_ratio": stats.get("sharpe_ratio"),
            "max_drawdown_pct": stats.get("max_drawdown_pct")}


def _from_run_row(run: dict) -> dict:
    return {"total_trades": run.get("trades"), "net_profit": run.get("net_profit"), "win_rate": run.get("win_rate"),
            "profit_factor": run.get("profit_factor"), "sharpe_ratio": run.get("sharpe_ratio"), "max_drawdown_pct": run.get("max_drawdown_pct")}


def reconcile(meta: Optional[dict], run_row: Optional[dict] = None) -> dict:
    """Side-by-side development vs full-history numbers for one strategy, and an explanation of any difference.

    ``meta`` is the strategy's metadata sidecar; ``run_row`` an optional run-history row (what the Dashboard reads), used when the
    result predates ``window_info``. Returns ``{"has": False}`` when there is nothing to compare.
    """
    meta = meta or {}
    snap = meta.get("pipeline_snapshot") or {}
    wi = snap.get("window_info") or (meta.get("last_run") or {}).get("window_info") or {}
    rows: list[dict] = []
    if wi.get("dev"):
        rows.append(_row("dev", wi["dev"].get("stats") or {}, wi["dev"].get("period"), source="this strategy's saved Full Pipeline result"))
        if not wi.get("same_window") and wi.get("full"):
            rows.append(_row("full", wi["full"].get("stats") or {}, wi["full"].get("period"), source="this strategy's saved Full Pipeline result"))
    else:
        dev_stats = snap.get("stats") or {}
        lr = meta.get("last_run") or {}
        if not dev_stats and lr:
            dev_stats = {"total_trades": lr.get("trades"), "net_profit": lr.get("net_profit"), "win_rate": lr.get("win_rate"), "max_drawdown_pct": lr.get("max_dd")}
        if dev_stats:
            rows.append(_row("dev", dev_stats, snap.get("period"), source="saved result (older run: window not recorded)"))
        if run_row:
            rows.append(_row("full", _from_run_row(run_row), None, source="run history (what the Dashboard shows)"))
    if not rows:
        return {"has": False}
    out: dict = {"has": True, "rows": rows, "lines": [], "agree": True}
    by = {r["key"]: r for r in rows}
    d, f = by.get("dev"), by.get("full")
    if d and f:
        dt, ft = d.get("trades"), f.get("trades")
        same = (dt is not None and dt == ft)
        out["agree"] = same
        if not same:
            hold = (ft - dt) if (dt is not None and ft is not None and ft >= dt) else None
            out["lines"].append(
                f"The two windows are different stretches of the same history, so the numbers differ by design: {dt} trades on the development window"
                + (f" plus {hold} on the sealed holdout = {ft} on the full history." if hold is not None else f" versus {ft} on the full history."))
            out["lines"].append("Win rate, profit factor and Sharpe move because the holdout trades are added; none of them is an error. "
                                "A big gap (for example Sharpe falling sharply once the holdout is included) would mean the edge is weaker on data the strategy never saw.")
            if d.get("sharpe_ratio") is not None and f.get("sharpe_ratio") is not None:
                gap = f["sharpe_ratio"] - d["sharpe_ratio"]
                out["lines"].append(f"Sharpe: {d['sharpe_ratio']:.2f} on the development window, {f['sharpe_ratio']:.2f} on the full history ({gap:+.2f}).")
            out["lines"].append("The verdict, T58 score and pass chances always use the development window, so the holdout stays unseen. "
                                "Dashboard cards and the run report show the full history.")
    else:
        out["lines"].append("Only one window is on record for this strategy. Re-run the Full Pipeline once to record both.")
    return out
