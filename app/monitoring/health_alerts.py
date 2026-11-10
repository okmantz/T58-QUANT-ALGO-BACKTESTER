"""
Strategy Health Alerts: plain-English alerts when a strategy that is trading (forward test, demo or funded) stops behaving like its backtest.

Works from the journal, so it covers every source the journal captures. Each alert says what it compared and why it matters; when the
inputs needed for a comparison are missing (no backtest figure saved) that check is skipped, never guessed. The Monte-Carlo drift check in
``app.monitoring.strategy_health`` feeds in through ``from_health_result``.
"""
from __future__ import annotations

MIN_TRADES = 10
LEVELS = {"ok": 0, "watch": 1, "warning": 2, "critical": 3}


def _alert(level: str, key: str, title: str, detail: str) -> dict:
    return {"level": level, "key": key, "title": title, "detail": detail}


def from_journal(trades: list[dict], *, backtest_avg_r: float | None = None, backtest_max_dd: float | None = None,
                 backtest_win_rate: float | None = None, backtest_max_losing_streak: int | None = None) -> list[dict]:
    ts = sorted(trades, key=lambda t: (t.get("exitTime") or t.get("date", ""), t.get("id", "")))
    out: list[dict] = []
    if len(ts) < MIN_TRADES:
        return [_alert("ok", "thin", "Too early to judge", f"{len(ts)} trades so far; health checks start at {MIN_TRADES}.")]
    pnls = [t.get("pnl") or 0.0 for t in ts]
    eq = peak = dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    cur = (peak - eq)
    if backtest_max_dd:
        if dd > backtest_max_dd:
            out.append(_alert("critical", "drawdown", "Drawdown is beyond the backtest's worst case",
                              f"Worst drawdown so far ${dd:,.0f} versus ${backtest_max_dd:,.0f} in the backtest. The edge may have decayed, or this is an unusual stretch: stop adding size until you know which."))
        elif dd > 0.8 * backtest_max_dd:
            out.append(_alert("warning", "drawdown", "Drawdown is close to the backtest's worst case", f"${dd:,.0f} of ${backtest_max_dd:,.0f}."))
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / len(pnls) * 100
    if backtest_win_rate and wr < backtest_win_rate - 15:
        out.append(_alert("warning", "win_rate", "Win rate has dropped", f"{wr:.0f}% over {len(pnls)} trades versus {backtest_win_rate:.0f}% in the backtest."))
    streak = 0
    for p in reversed(pnls):
        if p < 0:
            streak += 1
        else:
            break
    if backtest_max_losing_streak and streak > backtest_max_losing_streak:
        out.append(_alert("critical", "streak", "Losing streak is longer than the backtest's worst", f"{streak} losses in a row versus a backtest maximum of {backtest_max_losing_streak}."))
    elif streak >= 5:
        out.append(_alert("watch", "streak", f"{streak} losses in a row", "Not outside the backtest on its own, but check the journal for rule breaks."))
    from app.journal.behavior import realized_vs_backtest
    gap = realized_vs_backtest(ts)
    if backtest_avg_r and not gap:
        rs = [t["rMultiple"] for t in ts if t.get("rMultiple") is not None]
        if len(rs) >= MIN_TRADES:
            real = sum(rs) / len(rs)
            gap = {"n": len(rs), "realized_r": real, "backtest_r": backtest_avg_r, "gap_pct": (backtest_avg_r - real) / abs(backtest_avg_r) * 100}
    if gap and gap.get("gap_pct") is not None and gap["gap_pct"] >= 20:
        out.append(_alert("warning" if gap["gap_pct"] < 50 else "critical", "r_gap", f"Realized R is {gap['gap_pct']:.0f}% below the backtest",
                          f"{gap['realized_r']:+.2f}R over {gap['n']} trades versus {gap['backtest_r']:+.2f}R expected."))
    rule_breaks = sum(1 for t in ts[-20:] if t.get("violations"))
    if rule_breaks >= 4:
        out.append(_alert("warning", "violations", "Rule breaks are piling up", f"{rule_breaks} of the last {min(20, len(ts))} trades broke a rule."))
    return sorted(out, key=lambda a: -LEVELS[a["level"]]) or [_alert("ok", "healthy", "In line with the backtest", f"{len(ts)} trades, current drawdown ${cur:,.0f}.")]


def from_health_result(result) -> list[dict]:
    """Convert a ``StrategyHealthResult`` (Monte-Carlo drift check) into alerts."""
    lvl = {"ok": "ok", "watch": "watch", "warning": "warning", "critical": "critical"}
    return [_alert(lvl.get(f.severity, "watch"), f"mc_{f.metric}", f"{f.metric.replace('_', ' ').title()} drift", f.message) for f in result.flags
            if f.severity != "ok"]


def worst(alerts: list[dict]) -> str:
    return max((a["level"] for a in alerts), key=lambda l: LEVELS[l], default="ok")
