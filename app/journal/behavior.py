"""
Automatic behavioral analytics over the journal, and the question every trader needs answered: did the STRATEGY fail, or did I?

Every insight states its evidence (counts, dollars) and is only shown when there are enough trades behind it (``MIN_N``). Nothing
here predicts anything; it describes what already happened in the journal.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

MIN_N = 5            # trades needed behind a bucket before it can be called a pattern
GAP_R = 0.20         # realized R this far (fraction) below the backtest's R counts as a gap


def _r(t):
    return t.get("rMultiple")


def _sorted(trades):
    return sorted(trades, key=lambda t: (t.get("exitTime") or t.get("entryTime") or t.get("date", ""), t.get("id", "")))


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _hour_bucket(t):
    from app.journal.capture import ET
    try:
        d = datetime.fromisoformat(t["entryTime"])
    except (KeyError, ValueError, TypeError):
        return None
    if d.tzinfo is None:
        from datetime import timezone
        d = d.replace(tzinfo=timezone.utc)
    e = d.astimezone(ET)
    return e.hour, e.minute


def insights(trades: list[dict]) -> list[dict]:
    """[{key, title, detail, severity(info|warn|bad), evidence}] sorted most important first."""
    ts = _sorted(trades)
    out: list[dict] = []
    if len(ts) < MIN_N:
        return [{"key": "thin", "severity": "info", "title": "Not enough trades yet",
                 "detail": f"{len(ts)} trade(s) so far. Behavioral patterns need at least {MIN_N} trades behind them before they are shown.", "evidence": {"trades": len(ts)}}]

    # 1. best time window (NY open 9:30-11:00 vs the rest)
    win, rest = [], []
    for t in ts:
        hm = _hour_bucket(t)
        if hm is None:
            continue
        (win if (9 * 60 + 30) <= hm[0] * 60 + hm[1] < 11 * 60 else rest).append(t)
    if len(win) >= MIN_N and len(rest) >= MIN_N:
        wp, rp = sum(t["pnl"] for t in win), sum(t["pnl"] for t in rest)
        wa, ra = sum(t["pnl"] for t in win) / len(win), sum(t["pnl"] for t in rest) / len(rest)
        if wa > ra:
            out.append({"key": "best_window", "severity": "info", "title": "Your best window is 9:30-11:00 ET",
                        "detail": f"${wa:,.2f} per trade in that window ({len(win)} trades) versus ${ra:,.2f} outside it ({len(rest)} trades).",
                        "evidence": {"window_pnl": round(wp, 2), "rest_pnl": round(rp, 2)}})
        else:
            out.append({"key": "worst_window", "severity": "warn", "title": "9:30-11:00 ET is not your best window",
                        "detail": f"${wa:,.2f} per trade in that window ({len(win)}) versus ${ra:,.2f} outside it ({len(rest)}).", "evidence": {}})
    else:  # best session generally
        by = defaultdict(list)
        for t in ts:
            by[t.get("session") or "unclassified"].append(t["pnl"])
        ok = {k: v for k, v in by.items() if len(v) >= MIN_N and k != "unclassified"}
        if ok:
            k = max(ok, key=lambda x: sum(ok[x]) / len(ok[x]))
            out.append({"key": "best_session", "severity": "info", "title": f"Your best session is {k}",
                        "detail": f"${sum(ok[k]) / len(ok[k]):,.2f} per trade over {len(ok[k])} trades.", "evidence": {}})

    # 2. deterioration after 3 consecutive losses
    after, streak = [], 0
    for t in ts:
        if streak >= 3:
            after.append(t)
        streak = streak + 1 if t["pnl"] < 0 else 0
    if len(after) >= 3:
        base = [t for t in ts if t not in after]
        a, b = _avg([_r(t) for t in after]), _avg([_r(t) for t in base])
        ap, bp = sum(t["pnl"] for t in after) / len(after), sum(t["pnl"] for t in base) / max(1, len(base))
        if ap < bp:
            out.append({"key": "tilt", "severity": "bad", "title": "You deteriorate after 3 consecutive losses",
                        "detail": f"The {len(after)} trades taken right after a 3-loss streak averaged ${ap:,.2f}"
                                  + (f" ({a:+.2f}R)" if a is not None else "") + f" versus ${bp:,.2f} on all other trades. A hard stop after 3 losses would have avoided ${-sum(t['pnl'] for t in after if t['pnl'] < 0):,.0f} of losses.",
                        "evidence": {"n_after_streak": len(after)}})

    # 3. drawdowns on high-volatility days
    hv = [t for t in ts if "high vol" in (t.get("regime") or "")]
    if len(hv) >= MIN_N:
        lo = [t for t in ts if t not in hv and (t.get("regime") or "unclassified") != "unclassified"]
        hp = sum(t["pnl"] for t in hv) / len(hv)
        lp = sum(t["pnl"] for t in lo) / len(lo) if lo else None
        if hp < 0 and (lp is None or hp < lp):
            out.append({"key": "high_vol", "severity": "warn", "title": "Your worst losses come on high-volatility days",
                        "detail": f"{len(hv)} trades in high-volatility conditions averaged ${hp:,.2f}" + (f" versus ${lp:,.2f} otherwise." if lp is not None else "."),
                        "evidence": {"n": len(hv)}})

    # 4. weekday losses
    byd = defaultdict(list)
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for t in ts:
        try:
            byd[datetime.fromisoformat(t["date"]).weekday()].append(t["pnl"])
        except (KeyError, ValueError):
            pass
    cands = {d: v for d, v in byd.items() if len(v) >= MIN_N and sum(v) < 0}
    if cands:
        d = min(cands, key=lambda k: sum(cands[k]))
        out.append({"key": "weekday", "severity": "warn", "title": f"{names[d]} is a losing day for you",
                    "detail": f"{len(cands[d])} trades on {names[d]}s lost ${-sum(cands[d]):,.0f} in total.", "evidence": {"day": names[d]}})

    # 5. realized R versus the backtest
    gap = realized_vs_backtest(ts)
    if gap and gap["gap_pct"] is not None and gap["gap_pct"] >= GAP_R * 100:
        out.append({"key": "r_gap", "severity": "bad", "title": f"Your average realized R is {gap['gap_pct']:.0f}% below the backtest",
                    "detail": f"{gap['realized_r']:+.2f}R realized over {gap['n']} trades versus {gap['backtest_r']:+.2f}R in the backtest.", "evidence": gap})

    # 6. rule violations
    v = [t for t in ts if t.get("violations")]
    if len(v) >= 3:
        vp = sum(t["pnl"] for t in v)
        out.append({"key": "violations", "severity": "bad" if vp < 0 else "warn", "title": f"{len(v)} trades broke your rules",
                    "detail": f"Those trades netted ${vp:,.2f}. Most common: {_top_violation(v)}.", "evidence": {"n": len(v), "pnl": round(vp, 2)}})

    rank = {"bad": 0, "warn": 1, "info": 2}
    return sorted(out, key=lambda i: rank[i["severity"]])


def _top_violation(v):
    c = defaultdict(int)
    for t in v:
        for x in t["violations"]:
            c[x.split(":")[0].split(" vs ")[0]] += 1
    return max(c, key=c.get) if c else "none"


def realized_vs_backtest(trades: list[dict]) -> dict | None:
    rs = [_r(t) for t in trades if _r(t) is not None]
    bt = [t["backtestAvgR"] for t in trades if t.get("backtestAvgR") not in (None, 0)]
    if len(rs) < MIN_N or not bt:
        return None
    btr = sum(bt) / len(bt)
    real = sum(rs) / len(rs)
    return {"n": len(rs), "realized_r": round(real, 3), "backtest_r": round(btr, 3),
            "gap_pct": round((btr - real) / abs(btr) * 100.0, 1) if btr else None}


def diagnose(trades: list[dict], *, backtest_avg_r: float | None = None, backtest_win_rate: float | None = None) -> dict:
    """Strategy failure versus trader failure.

    Rule-following trades (no recorded violation) measure the STRATEGY as it was meant to run; trades with violations measure the
    TRADER. If the clean trades also lose, or fall short of the backtest, the strategy is the problem. If the clean trades match
    the backtest and the violating ones drag the total down, the trader is. Needs ``MIN_N`` trades in each bucket it relies on.
    """
    ts = _sorted(trades)
    clean = [t for t in ts if not t.get("violations")]
    dirty = [t for t in ts if t.get("violations")]
    if backtest_avg_r is None:
        bt = [t["backtestAvgR"] for t in ts if t.get("backtestAvgR") not in (None, 0)]
        backtest_avg_r = sum(bt) / len(bt) if bt else None
    cr, dr, ar = _avg([_r(t) for t in clean]), _avg([_r(t) for t in dirty]), _avg([_r(t) for t in ts])
    base = {"clean_trades": len(clean), "violating_trades": len(dirty), "clean_avg_r": cr, "violating_avg_r": dr, "all_avg_r": ar,
            "clean_pnl": round(sum(t["pnl"] for t in clean), 2), "violating_pnl": round(sum(t["pnl"] for t in dirty), 2), "backtest_avg_r": backtest_avg_r}
    if len(clean) < MIN_N:
        return {**base, "verdict": "unclear", "headline": "Not enough rule-following trades to tell yet",
                "detail": f"{len(clean)} clean trade(s) so far; at least {MIN_N} are needed to judge the strategy on its own."}
    if backtest_avg_r is None:
        strat_bad = cr is not None and cr < 0
    else:
        strat_bad = cr is not None and cr < backtest_avg_r * (1 - GAP_R) - 1e-9
    trader_bad = len(dirty) >= 3 and dr is not None and cr is not None and dr < cr and sum(t["pnl"] for t in dirty) < 0
    if strat_bad and trader_bad:
        verdict, head = "both", "Both: the strategy is under-delivering and rule breaks are costing more"
    elif strat_bad:
        verdict, head = "strategy", "The strategy is the problem, not you"
    elif trader_bad:
        verdict, head = "trader", "The strategy is working; the rule-breaking trades are what is costing you"
    else:
        verdict, head = "ok", "No failure detected: the strategy and your execution are both in line"
    bits = [f"Rule-following trades: {len(clean)} trades, {'%+.2fR' % cr if cr is not None else 'no R data'}, ${base['clean_pnl']:,.0f}."]
    if dirty:
        bits.append(f"Trades with a rule violation: {len(dirty)}, {'%+.2fR' % dr if dr is not None else 'no R data'}, ${base['violating_pnl']:,.0f}.")
    if backtest_avg_r is not None:
        bits.append(f"Backtest: {backtest_avg_r:+.2f}R per trade.")
    return {**base, "verdict": verdict, "headline": head, "detail": " ".join(bits)}
