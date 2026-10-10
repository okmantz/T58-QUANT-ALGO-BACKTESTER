"""
Baseline trade explorer (v9.13).

Full Pipeline's Step 1 baseline backtest finishes in seconds, but until
now its trades sat invisible inside the final result while Steps 2-7
ground on for many minutes. This module turns the completed baseline's
trade list into the aggregates the run page shows live during the run:

  * an hour-of-day x day-of-week P&L heatmap (2D -- Owen was explicit:
    no 3D weekly P&L),
  * hour-of-day and day-of-week P&L profiles,
  * where the losses cluster: worst hour buckets, worst weekdays,
    longest losing streaks, and the biggest contiguous loss clusters
    with their time windows,
  * where the wins cluster (v9.14): the exact mirror -- best hour
    buckets and weekdays with % of gross profit, longest winning
    streaks, and the biggest contiguous win clusters with windows,
  * 1-4 concrete, honest suggestions derived from those clusters, with
    real numbers ("entries at 02:00-04:00 lost $X -- Y% of all baseline
    losses"; "entries at 14:00 made $Y -- Z% of gross profit").

Everything here is a pure function of the trade list (no engine calls,
no globals), so the web endpoint serves it mid-run and tests pin the
aggregation math directly. Buckets key off each trade's ENTRY time in
the clock the data itself carries (the engine's timestamps); nothing
here re-derives or re-times a trade.

"Feeds the next run": Full Pipeline's form has no entry-time/session
filter parameter today (checked -- the manual builder's JSON supports
a time_of_day operand, but no form field or guided-prefill key carries
one), so the suggestions are deliberately TEXT with the numbers, and
the run page says plainly that the filter isn't wired yet. Nothing is
ever auto-applied to a strategy or a later run from here.
"""
from __future__ import annotations

import math
from datetime import datetime

WEEKDAY_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def trades_to_payload(trades) -> list[dict]:
    """Serializes engine Trade objects for the job store / JSON
    endpoint: entry/exit instants (ISO), direction (+1 long / -1
    short), net P&L, and duration in minutes. Best-effort per trade --
    a trade missing a field degrades to None/0 rather than sinking the
    whole payload, because this runs on a hook that must never break
    the pipeline it observes."""
    out: list[dict] = []
    for t in trades or []:
        try:
            entry = getattr(t, "entry_time", None)
            exit_ = getattr(t, "exit_time", None)
            duration_min = None
            if entry is not None and exit_ is not None:
                duration_min = (exit_ - entry).total_seconds() / 60.0
            out.append({
                "entry_time": entry.isoformat() if entry is not None else None,
                "exit_time": exit_.isoformat() if exit_ is not None else None,
                "direction": int(getattr(t, "direction", 0) or 0),
                "pnl": float(getattr(t, "pnl", 0.0) or 0.0),
                "duration_minutes": duration_min,
            })
        except Exception:  # noqa: BLE001 -- one odd trade must not sink the payload
            continue
    return out


def _parse_ts(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _losing_streaks(ordered: list[dict]) -> list[dict]:
    """Contiguous runs of losing trades (pnl < 0) in chronological
    order. Each: trade count, total P&L (negative), start/end entry
    times. Breakeven trades (pnl == 0) do not extend a losing streak --
    this panel is about where losses cluster, and the streak definition
    is stated on the page so the number is never mysterious."""
    streaks: list[dict] = []
    cur: list[dict] = []
    for tr in ordered:
        if tr["pnl"] < 0:
            cur.append(tr)
        elif cur:
            streaks.append(cur)
            cur = []
    if cur:
        streaks.append(cur)
    return [
        {
            "trades": len(s),
            "total_pnl": round(sum(t["pnl"] for t in s), 2),
            "start": s[0]["entry_time"],
            "end": s[-1]["entry_time"],
        }
        for s in streaks
    ]


def _winning_streaks(ordered: list[dict]) -> list[dict]:
    """Mirror of _losing_streaks for the win side: contiguous runs of
    winning trades (pnl > 0) in chronological order. Breakeven trades
    (pnl == 0) break a winning streak exactly like they break a losing
    one -- the definition is symmetric so the two panels are always
    comparable."""
    streaks: list[dict] = []
    cur: list[dict] = []
    for tr in ordered:
        if tr["pnl"] > 0:
            cur.append(tr)
        elif cur:
            streaks.append(cur)
            cur = []
    if cur:
        streaks.append(cur)
    return [
        {
            "trades": len(s),
            "total_pnl": round(sum(t["pnl"] for t in s), 2),
            "start": s[0]["entry_time"],
            "end": s[-1]["entry_time"],
        }
        for s in streaks
    ]


def summarize_baseline_trades(trades: list[dict]) -> dict:
    """Aggregates a baseline trade payload (see trades_to_payload) into
    the explorer panel's numbers. Pure arithmetic over the given
    trades; with zero trades it returns the same shape with empty/zero
    values and `has_trades` False, so the panel can say "the baseline
    produced no trades" instead of dividing by zero."""
    parsed = []
    for t in trades or []:
        ts = _parse_ts(t.get("entry_time"))
        if ts is None:
            continue
        parsed.append({
            "entry_time": t.get("entry_time"),
            "pnl": float(t.get("pnl", 0.0) or 0.0),
            "hour": ts.hour,
            "weekday": ts.weekday(),
        })
    parsed.sort(key=lambda r: r["entry_time"] or "")

    matrix = [[0.0] * 24 for _ in range(7)]        # [weekday][hour] net P&L
    matrix_counts = [[0] * 24 for _ in range(7)]
    hour_profile = [0.0] * 24
    hour_counts = [0] * 24
    weekday_profile = [0.0] * 7
    weekday_counts = [0] * 7
    gross_loss = 0.0
    gross_profit = 0.0
    for r in parsed:
        matrix[r["weekday"]][r["hour"]] += r["pnl"]
        matrix_counts[r["weekday"]][r["hour"]] += 1
        hour_profile[r["hour"]] += r["pnl"]
        hour_counts[r["hour"]] += 1
        weekday_profile[r["weekday"]] += r["pnl"]
        weekday_counts[r["weekday"]] += 1
        if r["pnl"] < 0:
            gross_loss += -r["pnl"]
        elif r["pnl"] > 0:
            gross_profit += r["pnl"]
    net_total = sum(r["pnl"] for r in parsed)

    def _share(loss: float) -> float:
        return round(100.0 * loss / gross_loss, 1) if gross_loss > 0 else 0.0

    worst_hours = sorted(
        (
            {"hour": h, "pnl": round(hour_profile[h], 2), "trades": hour_counts[h],
             "share_of_gross_losses_pct": _share(-hour_profile[h])}
            for h in range(24) if hour_profile[h] < 0
        ),
        key=lambda r: r["pnl"],
    )[:5]
    worst_weekdays = sorted(
        (
            {"weekday": d, "label": WEEKDAY_LABELS[d], "pnl": round(weekday_profile[d], 2),
             "trades": weekday_counts[d], "share_of_gross_losses_pct": _share(-weekday_profile[d])}
            for d in range(7) if weekday_profile[d] < 0
        ),
        key=lambda r: r["pnl"],
    )[:4]

    streaks = _losing_streaks(parsed)
    longest_streaks = sorted(streaks, key=lambda s: (-s["trades"], s["total_pnl"]))[:3]
    biggest_clusters = sorted(streaks, key=lambda s: s["total_pnl"])[:3]

    def _pshare(profit: float) -> float:
        return round(100.0 * profit / gross_profit, 1) if gross_profit > 0 else 0.0

    best_hours = sorted(
        (
            {"hour": h, "pnl": round(hour_profile[h], 2), "trades": hour_counts[h],
             "share_of_gross_profits_pct": _pshare(hour_profile[h])}
            for h in range(24) if hour_profile[h] > 0
        ),
        key=lambda r: -r["pnl"],
    )[:5]
    best_weekdays = sorted(
        (
            {"weekday": d, "label": WEEKDAY_LABELS[d], "pnl": round(weekday_profile[d], 2),
             "trades": weekday_counts[d], "share_of_gross_profits_pct": _pshare(weekday_profile[d])}
            for d in range(7) if weekday_profile[d] > 0
        ),
        key=lambda r: -r["pnl"],
    )[:4]

    win_streaks = _winning_streaks(parsed)
    longest_win_streaks = sorted(win_streaks, key=lambda s: (-s["trades"], -s["total_pnl"]))[:3]
    biggest_win_clusters = sorted(win_streaks, key=lambda s: -s["total_pnl"])[:3]

    suggestions = _build_suggestions(
        parsed, worst_hours, worst_weekdays, streaks,
        gross_loss=gross_loss, net_total=net_total,
        best_hours=best_hours, gross_profit=gross_profit,
    )

    return {
        "has_trades": bool(parsed),
        "n_trades": len(parsed),
        "net_pnl": round(net_total, 2),
        "gross_loss": round(gross_loss, 2),
        "matrix": [[round(v, 2) for v in row] for row in matrix],
        "matrix_counts": matrix_counts,
        "hour_profile": [round(v, 2) for v in hour_profile],
        "hour_counts": hour_counts,
        "weekday_profile": [round(v, 2) for v in weekday_profile],
        "weekday_counts": weekday_counts,
        "weekday_labels": list(WEEKDAY_LABELS),
        "worst_hours": worst_hours,
        "worst_weekdays": worst_weekdays,
        "longest_losing_streaks": longest_streaks,
        "biggest_loss_clusters": biggest_clusters,
        "gross_profit": round(gross_profit, 2),
        "best_hours": best_hours,
        "best_weekdays": best_weekdays,
        "longest_winning_streaks": longest_win_streaks,
        "biggest_win_clusters": biggest_win_clusters,
        "suggestions": suggestions,
    }


def _fmt_money(v: float) -> str:
    return f"${v:,.0f}"


def _build_suggestions(parsed, worst_hours, worst_weekdays, streaks, *, gross_loss, net_total,
                       best_hours=None, gross_profit: float = 0.0) -> list[str]:
    """1-3 honest, numbers-first suggestions. Every figure is computed
    from the baseline trades themselves; the hypothetical ("excluding
    those entries would have changed net by $W") is arithmetic on the
    recorded trades only -- it does NOT model re-sizing or knock-on
    effects, and the text never pretends otherwise."""
    out: list[str] = []
    if not parsed:
        return out

    # Worst entry hours as one contiguous-ish block (the worst hours,
    # up to three, reported individually and combined).
    if worst_hours:
        hours = sorted(h["hour"] for h in worst_hours[:3])
        in_hours = [r for r in parsed if r["hour"] in hours]
        block_net = sum(r["pnl"] for r in in_hours)
        losers = sum(1 for r in in_hours if r["pnl"] < 0)
        if block_net < 0 and gross_loss > 0:
            hour_txt = ", ".join(f"{h:02d}:00" for h in hours)
            out.append(
                f"Entries at {hour_txt} lost {_fmt_money(-block_net)} across {len(in_hours)} trades "
                f"({losers} losers) -- {_share_txt(-block_net, gross_loss)} of all baseline losses. "
                f"Leaving those entry hours out entirely would have changed the baseline net by "
                f"{_fmt_money(-block_net)} (to {_fmt_money(net_total - block_net)}), before any "
                "re-sizing or knock-on effects. Full Pipeline has no entry-time filter field yet, "
                "so this can't be carried into the next run automatically -- it is the first thing "
                "to test by hand (a time_of_day condition in the strategy config)."
            )
    # Worst weekday.
    if worst_weekdays:
        w = worst_weekdays[0]
        if w["pnl"] < 0 and gross_loss > 0:
            out.append(
                f"{w['label']} entries lost {_fmt_money(-w['pnl'])} over {w['trades']} trades "
                f"({_share_txt(-w['pnl'], gross_loss)} of baseline losses). A day-of-week "
                "exclusion is the same manual test as the hour filter above -- not wired into "
                "the pipeline form yet."
            )
    # Worst contiguous loss cluster.
    if streaks:
        worst = min(streaks, key=lambda s: s["total_pnl"])
        if worst["trades"] >= 3 and worst["total_pnl"] < 0:
            out.append(
                f"Worst loss cluster: {worst['trades']} straight losers totaling "
                f"{_fmt_money(-worst['total_pnl'])} from {worst['start']} to {worst['end']} -- "
                "check what the market (and the strategy's own state) was doing in that window "
                "before trusting the averages."
            )
    # Win window worth favoring (v9.14: the win-side mirror). Same honesty
    # rule as the loss suggestions: arithmetic on recorded trades only.
    if best_hours and gross_profit > 0:
        hours = sorted(h["hour"] for h in best_hours[:3])
        in_hours = [r for r in parsed if r["hour"] in hours]
        block_net = sum(r["pnl"] for r in in_hours)
        winners = sum(1 for r in in_hours if r["pnl"] > 0)
        if block_net > 0:
            hour_txt = ", ".join(f"{h:02d}:00" for h in hours)
            out.append(
                f"Entries at {hour_txt} made {_fmt_money(block_net)} across {len(in_hours)} trades "
                f"({winners} winners) -- {_share_txt(block_net, gross_profit)} of all baseline "
                "gross profit. If a next run concentrates size or session filters anywhere, "
                "these are the entry hours the baseline says to favor (recorded trades only -- "
                "not a promise about the next run's fills)."
            )
    return out[:4]


def _share_txt(loss: float, gross_loss: float) -> str:
    if gross_loss <= 0 or not math.isfinite(loss):
        return "0%"
    return f"{100.0 * loss / gross_loss:.0f}%"
