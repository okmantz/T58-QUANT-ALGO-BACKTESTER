"""Turn a strategy's day/hour P&L heatmap into a "don't trade then" filter.

The Full Pipeline / Run & Report heatmap is a 7 x 24 grid (Mon..Sun x hour 0..23) of net P&L by trade ENTRY time.
`suggest_blocked_slots` picks the few cells (or whole weekdays) that lose the most and returns them in the
RiskConfig.blocked_entry_slots text format (see app/backtest/time_slots.py), plus an honest estimate of what the
filter would have recovered.

Honest limit: the slots are chosen on the same trades they are scored on, so the recovered figure is in-sample and
flatters the filter. The optimizer therefore re-runs the backtest WITH the filter (so the real effect, including
trades that shift, is measured) and the Validate stage should confirm it out-of-sample.
"""
from __future__ import annotations

from app.backtest.time_slots import DAY_NAMES, format_slots, slot_label

CAVEAT = ("Chosen on the same trades it is scored on, so the saving is optimistic. Re-run Full Pipeline / Validate "
          "after applying it to confirm it holds on data the filter has not seen.")


def suggest_blocked_slots(grid, counts=None, *, max_slots: int = 3, min_trades: int = 5,
                          min_loss_share_pct: float = 8.0, weekday_loss_share_pct: float = 30.0) -> dict:
    """grid[weekday][hour] = net P&L (Mon=0). counts[weekday][hour] = trade count (optional).

    Returns {"slots": [(weekday|None, (h0,h1)|None), ...], "text": "THU 11\\nFRI *", "items": [...], "recovered_pnl": float,
             "caveat": str}. `items` carries a plain-English line for each suggestion."""
    try:
        g = [[float(grid[d][h]) for h in range(24)] for d in range(7)]
    except Exception:  # noqa: BLE001 -- malformed/missing grid: no suggestion
        return {"slots": [], "text": "", "items": [], "recovered_pnl": 0.0, "caveat": CAVEAT}
    c = None
    if counts:
        try:
            c = [[int(counts[d][h]) for h in range(24)] for d in range(7)]
        except Exception:  # noqa: BLE001
            c = None
    gross_loss = sum(-v for row in g for v in row if v < 0)
    if gross_loss <= 0:
        return {"slots": [], "text": "", "items": [], "recovered_pnl": 0.0, "caveat": CAVEAT}

    slots, items, covered_days, recovered = [], [], set(), 0.0

    # whole weekdays first (a day that is net-negative and carries a big share of the losses)
    day_tot = [sum(row) for row in g]
    for d in sorted(range(7), key=lambda i: day_tot[i]):
        if len(slots) >= max_slots:
            break
        # a whole day only when the loss is spread across it; one bad hour gets blocked as that hour, not the day
        worst_cell = min(g[d])
        if (day_tot[d] < 0 and 100.0 * (-day_tot[d]) / gross_loss >= weekday_loss_share_pct
                and (-worst_cell) < 0.6 * (-day_tot[d])):
            n = sum(c[d]) if c else None
            if n is not None and n < min_trades:
                continue
            slots.append((d, None))
            covered_days.add(d)
            recovered += -day_tot[d]
            items.append({"slot": f"{DAY_NAMES[d]} *", "label": slot_label(d, None), "pnl": round(day_tot[d], 2),
                          "trades": n, "why": f"{slot_label(d, None)} is net negative and makes up "
                                              f"{100.0 * (-day_tot[d]) / gross_loss:.0f}% of the losses."})

    cells = sorted(((g[d][h], d, h) for d in range(7) for h in range(24) if g[d][h] < 0 and d not in covered_days))
    for pnl, d, h in cells:
        if len(slots) >= max_slots:
            break
        n = c[d][h] if c else None
        if n is not None and n < min_trades:
            continue
        if 100.0 * (-pnl) / gross_loss < min_loss_share_pct:
            break  # sorted worst-first: everything after is smaller
        slots.append((d, (h, h)))
        recovered += -pnl
        items.append({"slot": f"{DAY_NAMES[d]} {h:02d}", "label": slot_label(d, (h, h)), "pnl": round(pnl, 2), "trades": n,
                      "why": f"{slot_label(d, (h, h))} lost ${-pnl:,.0f}"
                             + (f" over {n} trades" if n else "") + f" ({100.0 * (-pnl) / gross_loss:.0f}% of the losses)."})
    return {"slots": slots, "text": format_slots(slots), "items": items,
            "recovered_pnl": round(recovered, 2), "caveat": CAVEAT}


def grid_from_trades(trades) -> tuple[list, list]:
    """(pnl grid, count grid), 7 x 24, bucketed by trade ENTRY time on the data's own clock (same as the heatmap)."""
    import pandas as pd
    pnl = [[0.0] * 24 for _ in range(7)]
    cnt = [[0] * 24 for _ in range(7)]
    for t in trades or []:
        try:
            ts = pd.Timestamp(t.entry_time)
            pnl[int(ts.dayofweek)][int(ts.hour)] += float(t.pnl)
            cnt[int(ts.dayofweek)][int(ts.hour)] += 1
        except Exception:  # noqa: BLE001
            continue
    return pnl, cnt


def run_time_filter_optimization(df, strategy, risk, *, slots_text: str = "", library_ref=None, save: bool = True,
                                 max_slots: int = 3, progress_cb=None) -> dict:
    """Backtest as-is, pick the worst weekday/hour slots (or use the ones given), backtest again WITH the filter, and report
    the measured before/after. Saves the slots onto the library strategy only when the filter actually helps."""
    from dataclasses import replace
    from app.backtest.engine import run_backtest
    from app.backtest.time_slots import format_slots, merge_slot_text, parse_slots
    log = progress_cb or (lambda m: None)

    def _stats(bt):
        st = getattr(bt, "statistics", None)
        return {"trades": len(bt.trades), "net_profit": round(float(getattr(st, "net_profit", 0.0) or 0.0), 2),
                "max_drawdown_pct": round(float(getattr(st, "max_drawdown_pct", 0.0) or 0.0), 2),
                "profit_factor": round(float(getattr(st, "profit_factor", 0.0) or 0.0), 3)}

    log("Time filter: baseline run...")
    base = run_backtest(df, strategy, risk)
    out = {"ran": True, "baseline": _stats(base), "suggestion": None, "slots": "", "saved": False}
    given = format_slots(parse_slots(slots_text)) if slots_text else ""
    if given:
        chosen, items = given, [{"slot": s, "why": "chosen by you"} for s in given.splitlines()]
    else:
        pnl, cnt = grid_from_trades(base.trades)
        sug = suggest_blocked_slots(pnl, cnt, max_slots=max_slots)
        chosen, items = sug["text"], sug["items"]
        out["suggestion"] = sug
    out["items"] = items
    if not chosen:
        out["note"] = "No weekday/hour stands out as a consistent loser, so nothing was restricted."
        out["improved"] = False
        return out
    log(f"Time filter: re-running with no entries at {', '.join(chosen.splitlines())}...")
    r2 = replace(risk, blocked_entry_slots=merge_slot_text(getattr(risk, "blocked_entry_slots", None), chosen))
    filt = run_backtest(df, strategy, r2)
    out["filtered"] = _stats(filt)
    out["slots"] = chosen
    out["improved"] = out["filtered"]["net_profit"] > out["baseline"]["net_profit"]
    out["profit_change"] = round(out["filtered"]["net_profit"] - out["baseline"]["net_profit"], 2)
    out["caveat"] = CAVEAT
    if out["improved"] and save and library_ref:
        try:
            from app.strategy.library import set_blocked_entry_slots
            set_blocked_entry_slots(library_ref[0], library_ref[1], chosen)
            out["saved"] = True
        except Exception as exc:  # noqa: BLE001
            out["save_error"] = str(exc)
    return out
