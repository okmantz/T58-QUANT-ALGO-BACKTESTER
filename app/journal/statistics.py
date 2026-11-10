"""Trading Statistics: compare months and years in the journal side by side."""
from __future__ import annotations

from collections import defaultdict


def _row(label: str, ts: list[dict]) -> dict:
    pnls = [t.get("pnl") or 0.0 for t in ts]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    rs = [t["rMultiple"] for t in ts if t.get("rMultiple") is not None]
    gl = -sum(losses)
    eq = peak = dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return {"period": label, "trades": len(ts), "pnl": round(sum(pnls), 2), "win_rate": round(len(wins) / len(ts) * 100, 1) if ts else 0.0,
            "avg_r": round(sum(rs) / len(rs), 2) if rs else None,
            "profit_factor": round(sum(wins) / gl, 2) if gl > 0 else None, "best_trade": round(max(pnls), 2) if pnls else 0.0,
            "worst_trade": round(min(pnls), 2) if pnls else 0.0, "max_drawdown": round(dd, 2),
            "violations": sum(1 for t in ts if t.get("violations")), "days_traded": len({t.get("date") for t in ts})}


def compare(trades: list[dict], by: str = "month") -> dict:
    """{rows: one per period oldest->newest, with change vs the previous period}. ``by`` is 'month' or 'year'."""
    n = 7 if by == "month" else 4
    groups: dict[str, list] = defaultdict(list)
    for t in sorted(trades, key=lambda x: (x.get("date", ""), x.get("id", ""))):
        if t.get("date"):
            groups[t["date"][:n]].append(t)
    rows = [_row(k, v) for k, v in sorted(groups.items())]
    for i, r in enumerate(rows):
        prev = rows[i - 1] if i else None
        r["pnl_change"] = round(r["pnl"] - prev["pnl"], 2) if prev else None
        r["win_rate_change"] = round(r["win_rate"] - prev["win_rate"], 1) if prev else None
    best = max(rows, key=lambda r: r["pnl"])["period"] if rows else None
    worst = min(rows, key=lambda r: r["pnl"])["period"] if rows else None
    return {"by": by, "rows": rows, "best": best, "worst": worst}
