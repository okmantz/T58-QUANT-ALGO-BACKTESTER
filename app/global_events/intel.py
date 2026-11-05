"""Market intelligence for the Global Events page: sessions, central banks, macro releases, futures stats, volatility regime,
correlations, cross-asset monitor, geopolitics and per-instrument Market Regime Intelligence.

Everything here is a PURE function of (daily bars, the event calendar, headlines, the clock) so it is testable offline and every
label can be explained by the numbers beside it. Nothing is a prediction or a trade signal: it is context, with the rule that
produced each label printed next to it (``RULES``). Daily bars are fetched by the poller (Stooq CSV) into ``ge_daily``.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

# --------------------------------------------------------------------------- transparent regime rules (shown on the page)
RULES = {
    "window_days": 20,
    "trending_er_min": 0.35,       # Kaufman efficiency ratio over the window: |net move| / sum of daily moves
    "range_er_max": 0.20,
    "high_vol_pct_min": 80.0,      # realised-vol percentile within its own trailing year
    "range_vol_pct_max": 60.0,
    "event_window_hours": 24,      # a high-impact release for the instrument this close => Event-driven
    "corr_days": 60,
    "corr_min_overlap": 30,
    "unusual_move_z": 2.0,
}
RULE_TEXT = [
    "Event-driven: a high-impact release for this instrument is due within the next %(event_window_hours)d hours (or just printed). It outranks every other label because the calendar, not the chart, is in charge." % RULES,
    "High-vol: 20-day realised volatility sits at or above the %(high_vol_pct_min).0fth percentile of its own past year." % RULES,
    "Trending: efficiency ratio over 20 days is at least %(trending_er_min).2f (price made real net progress, not just noise); direction is the sign of the 20-day move." % RULES,
    "Range-bound: efficiency ratio is at most %(range_er_max).2f and volatility is at or below the %(range_vol_pct_max).0fth percentile (lots of movement, no progress, nothing violent)." % RULES,
    "Mixed: none of the above is clearly true. Fewer than 25 daily bars: Unknown.",
]
FUTURES = ["ES", "NQ", "YM", "RTY", "GC", "CL"]
ALL_SYMBOLS = ["ES", "NQ", "YM", "RTY", "DXY", "GC", "CL", "BTC", "VIX"]
# which calendar currencies / names move which instruments (the calendar rows already carry an instruments list; this is a fallback)
_MACRO = re.compile(r"\b(cpi|ppi|non[- ]?farm|nfp|payrolls?|gdp|pce|retail sales|ism|unemployment|jolts|fomc|federal funds|interest rate|rate decision|"
                    r"powell|lagarde|bailey|ueda)\b", re.I)
_GROUPS = (("Inflation", re.compile(r"\b(cpi|ppi|pce|inflation)\b", re.I)),
           ("Jobs", re.compile(r"non[- ]?farm|nfp|payrolls?|unemployment|jobless|jolts|employment", re.I)),
           ("Growth", re.compile(r"\b(gdp|retail sales|ism|pmi|industrial production|durable)\b", re.I)),
           ("Central banks", re.compile(r"fomc|federal funds|interest rate|rate decision|rate statement|powell|lagarde|bailey|ueda|minutes", re.I)))

# Central-bank meeting calendar as published by each bank for 2026. These are decision DAYS (second day of two-day meetings).
# Confirm on the bank's own page before relying on a date; the live economic calendar below overrides them when it has the event.
CENTRAL_BANKS = (
    {"id": "fed", "name": "Federal Reserve (FOMC)", "ccy": "USD", "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
     "dates": ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"], "hm_utc": "19:00",
     "moves": ["ES", "NQ", "YM", "RTY", "DXY", "GC", "ZN"]},
    {"id": "ecb", "name": "European Central Bank", "ccy": "EUR", "url": "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html",
     "dates": ["2026-02-05", "2026-03-19", "2026-04-30", "2026-06-11", "2026-07-23", "2026-09-10", "2026-10-29", "2026-12-17"], "hm_utc": "13:15",
     "moves": ["DXY", "GC"]},
    {"id": "boe", "name": "Bank of England", "ccy": "GBP", "url": "https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates",
     "dates": ["2026-02-05", "2026-03-19", "2026-04-30", "2026-06-18", "2026-07-30", "2026-09-17", "2026-11-05", "2026-12-17"], "hm_utc": "12:00",
     "moves": ["DXY"]},
    {"id": "boj", "name": "Bank of Japan", "ccy": "JPY", "url": "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm",
     "dates": ["2026-01-23", "2026-03-19", "2026-04-28", "2026-06-16", "2026-07-31", "2026-09-18", "2026-10-30", "2026-12-18"], "hm_utc": "03:00",
     "moves": ["DXY", "ES", "NQ"]},
)


def _dt(s: str, hm: str = "00:00") -> datetime:
    return datetime.fromisoformat(f"{s}T{hm}:00+00:00")


# --------------------------------------------------------------------------- sessions
def sessions(now: datetime) -> list[dict]:
    """Open/closed state of the sessions that matter for futures, in each market's own local time (DST-correct)."""
    out = []
    try:
        from zoneinfo import ZoneInfo
    except Exception:  # noqa: BLE001
        return out
    spec = (("Tokyo", "Asia/Tokyo", (9, 0), (15, 30)), ("London", "Europe/London", (8, 0), (16, 30)),
            ("New York (cash)", "America/New_York", (9, 30), (16, 0)))
    for name, tz, (oh, om), (ch, cm) in spec:
        try:
            z = ZoneInfo(tz)
        except Exception:  # noqa: BLE001 - tzdata missing (some Windows installs)
            continue
        loc = now.astimezone(z)
        o = loc.replace(hour=oh, minute=om, second=0, microsecond=0)
        c = loc.replace(hour=ch, minute=cm, second=0, microsecond=0)
        weekday = loc.weekday() < 5
        is_open = weekday and o <= loc < c
        out.append({"name": name, "tz": tz, "open": bool(is_open), "local_time": loc.strftime("%a %H:%M"),
                    "hours": f"{oh:02d}:{om:02d}-{ch:02d}:{cm:02d} local", "until": _session_until(loc, o, c, weekday, is_open)})
    try:
        ct = now.astimezone(ZoneInfo("America/Chicago"))
        # CME Globex equity/energy/metals: Sunday 17:00 CT to Friday 16:00 CT, daily maintenance break 16:00-17:00 CT.
        wd, hr = ct.weekday(), ct.hour + ct.minute / 60
        if wd == 5 or (wd == 6 and hr < 17) or (wd == 4 and hr >= 16):
            is_open, note = False, "weekend close (reopens Sunday 17:00 CT)"
        elif 16 <= hr < 17:
            is_open, note = False, "daily maintenance break (reopens 17:00 CT)"
        else:
            is_open, note = True, "closes 16:00 CT" if hr < 16 else "closes tomorrow 16:00 CT"
        out.append({"name": "CME Globex futures", "tz": "America/Chicago", "open": is_open, "local_time": ct.strftime("%a %H:%M"),
                    "hours": "Sun 17:00 - Fri 16:00 CT, break 16:00-17:00", "until": note})
    except Exception:  # noqa: BLE001
        pass
    return out


def _session_until(loc, o, c, weekday, is_open) -> str:
    if is_open:
        m = int((c - loc).total_seconds() // 60)
        return f"closes in {m // 60}h {m % 60:02d}m"
    if weekday and loc < o:
        m = int((o - loc).total_seconds() // 60)
        return f"opens in {m // 60}h {m % 60:02d}m"
    return "opens next trading day"


# --------------------------------------------------------------------------- calendar views
def macro_releases(events: Iterable[dict], now: datetime) -> list[dict]:
    """High/medium-impact releases grouped (Inflation / Jobs / Growth / Central banks) from the live calendar rows."""
    out = []
    for e in events:
        name = e.get("name") or ""
        if not _MACRO.search(name) and (e.get("impact") or 0) < 3:
            continue
        group = next((g for g, rx in _GROUPS if rx.search(name)), "Other high impact")
        when = e.get("when") or e.get("scheduled_utc")
        out.append({"name": name, "ccy": e.get("country"), "group": group, "impact": e.get("impact"), "when": when,
                    "forecast": e.get("forecast"), "previous": e.get("previous"), "actual": e.get("actual"),
                    "instruments": e.get("instruments") or []})
    out.sort(key=lambda r: r["when"] or "")
    return out


def central_banks(now: datetime, calendar_events: Iterable[dict] = ()) -> list[dict]:
    """Next decision for each bank, days away, and whether the live calendar has a matching event with forecast/previous."""
    live = list(calendar_events)
    res = []
    for b in CENTRAL_BANKS:
        upcoming = [d for d in b["dates"] if _dt(d, b["hm_utc"]) >= now - timedelta(hours=6)]
        nxt = upcoming[0] if upcoming else None
        row = {"id": b["id"], "name": b["name"], "ccy": b["ccy"], "url": b["url"], "next": nxt, "moves": b["moves"],
               "days": None if not nxt else round((_dt(nxt, b["hm_utc"]) - now).total_seconds() / 86400, 1),
               "schedule_note": "Published schedule - confirm on the bank's page." if nxt else "Schedule list ended: check the bank's page for the new calendar."}
        if nxt:
            match = [e for e in live if (e.get("country") == b["ccy"]) and re.search(r"rate (decision|statement)|interest rate|fomc|federal funds", e.get("name") or "", re.I)
                     and (e.get("when") or "")[:10] == nxt]
            if match:
                m = match[0]
                row.update(forecast=m.get("forecast"), previous=m.get("previous"), live_event=m.get("name"))
        res.append(row)
    return res


# --------------------------------------------------------------------------- daily-bar statistics
def _closes(rows: list[dict]) -> list[float]:
    return [float(r["c"]) for r in rows if r.get("c") not in (None, 0)]


def _rets(closes: list[float]) -> list[float]:
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]


def _std(x: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    m = sum(x) / len(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def efficiency_ratio(closes: list[float], n: int = 20) -> Optional[float]:
    if len(closes) < n + 1:
        return None
    w = closes[-(n + 1):]
    path = sum(abs(w[i] - w[i - 1]) for i in range(1, len(w)))
    return None if path == 0 else abs(w[-1] - w[0]) / path


def realised_vol_series(closes: list[float], n: int = 20) -> list[float]:
    r = _rets(closes)
    return [_std(r[i - n:i]) * math.sqrt(252) * 100 for i in range(n, len(r) + 1)]


def _pct_rank(series: list[float], v: float) -> Optional[float]:
    s = [x for x in series if x == x]
    return None if not s else 100.0 * sum(1 for x in s if x <= v) / len(s)


def atr(rows: list[dict], n: int = 14) -> Optional[float]:
    if len(rows) < n + 1:
        return None
    trs = []
    for i in range(len(rows) - n, len(rows)):
        h, l, pc = rows[i].get("h"), rows[i].get("l"), rows[i - 1].get("c")
        if None in (h, l, pc):
            continue
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs) if trs else None


def futures_stats(daily: dict[str, list[dict]]) -> list[dict]:
    out = []
    for sym in FUTURES:
        rows = daily.get(sym) or []
        c = _closes(rows)
        if len(c) < 6:
            out.append({"symbol": sym, "available": False})
            continue
        last = c[-1]
        n = min(20, len(c) - 1)
        hi, lo = max(c[-(n + 1):]), min(c[-(n + 1):])
        rv = realised_vol_series(c)
        a = atr(rows)
        out.append({"symbol": sym, "available": True, "last": last, "chg_1d": (last / c[-2] - 1) * 100, "chg_5d": (last / c[-6] - 1) * 100,
                    "chg_20d": (last / c[-(n + 1)] - 1) * 100, "range_pos_20d": None if hi == lo else (last - lo) / (hi - lo) * 100,
                    "rv20": rv[-1] if rv else None, "atr14": a, "atr14_pct": (a / last * 100) if a else None})
    return out


def regime_for(symbol: str, rows: list[dict], events: Iterable[dict], now: datetime) -> dict:
    """One instrument's regime with every number that produced the label."""
    c = _closes(rows)
    if len(c) < 25:
        return {"symbol": symbol, "regime": "Unknown", "why": "Fewer than 25 daily bars stored yet.", "metrics": {}}
    er = efficiency_ratio(c, RULES["window_days"])
    rv = realised_vol_series(c)
    rv_now = rv[-1] if rv else None
    pct = _pct_rank(rv[-252:], rv_now) if rv_now is not None else None
    ret20 = (c[-1] / c[-(RULES["window_days"] + 1)] - 1) * 100
    horizon = timedelta(hours=RULES["event_window_hours"])
    nearest = None
    for e in events:
        if (e.get("impact") or 0) < 3 or symbol not in (e.get("instruments") or []):
            continue
        t = e.get("when") or e.get("scheduled_utc")
        try:
            dt = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            continue
        if -timedelta(hours=3) <= dt - now <= horizon and (nearest is None or abs(dt - now) < abs(nearest[0] - now)):
            nearest = (dt, e.get("name"))
    m = {"efficiency_ratio_20d": er, "realised_vol_20d_pct": rv_now, "vol_percentile_1y": pct, "return_20d_pct": ret20}
    if nearest:
        hrs = (nearest[0] - now).total_seconds() / 3600
        return {"symbol": symbol, "regime": "Event-driven", "why": f"{nearest[1]} {'in %.0fh' % hrs if hrs >= 0 else 'just printed'} (high impact for {symbol}).",
                "metrics": m, "event": nearest[1]}
    if pct is not None and pct >= RULES["high_vol_pct_min"]:
        return {"symbol": symbol, "regime": "High-vol", "why": f"20d realised vol {rv_now:.1f}% is the {pct:.0f}th percentile of its past year (rule: >= {RULES['high_vol_pct_min']:.0f}th).", "metrics": m}
    if er is not None and er >= RULES["trending_er_min"]:
        d = "up" if ret20 > 0 else "down"
        return {"symbol": symbol, "regime": f"Trending {d}", "why": f"Efficiency ratio {er:.2f} (rule: >= {RULES['trending_er_min']:.2f}); 20d move {ret20:+.1f}%.", "metrics": m}
    if er is not None and er <= RULES["range_er_max"] and (pct is None or pct <= RULES["range_vol_pct_max"]):
        return {"symbol": symbol, "regime": "Range-bound", "why": f"Efficiency ratio {er:.2f} (rule: <= {RULES['range_er_max']:.2f}) with vol at the {pct or 0:.0f}th percentile.", "metrics": m}
    return {"symbol": symbol, "regime": "Mixed", "why": f"Efficiency ratio {er:.2f} and vol percentile {pct or 0:.0f} sit between the rules.", "metrics": m}


def regimes(daily: dict[str, list[dict]], events: list[dict], now: datetime) -> list[dict]:
    return [regime_for(s, daily.get(s) or [], events, now) for s in FUTURES + ["DXY"]]


def vol_regime(daily: dict[str, list[dict]]) -> dict:
    """VIX level bucket + VIX year percentile + ES realised vol, in plain language."""
    v = _closes(daily.get("VIX") or [])
    es = _closes(daily.get("ES") or [])
    out = {"available": bool(v)}
    if v:
        lvl = v[-1]
        out.update(vix=lvl, vix_pct_1y=_pct_rank(v[-252:], lvl),
                   bucket=("Calm (<15)" if lvl < 15 else "Normal (15-20)" if lvl < 20 else "Elevated (20-30)" if lvl < 30 else "Stress (30+)"),
                   vix_chg_5d=(lvl / v[-6] - 1) * 100 if len(v) > 5 else None)
    rv = realised_vol_series(es)
    if rv:
        out.update(es_rv20=rv[-1], es_rv_pct_1y=_pct_rank(rv[-252:], rv[-1]))
        if v:
            out["vix_minus_rv"] = out["vix"] - rv[-1]
            out["premium_note"] = ("Implied vol above realised: options are pricing more movement than has happened." if out["vix_minus_rv"] > 3
                                   else "Implied vol below realised: the market is under-pricing the recent movement." if out["vix_minus_rv"] < -3
                                   else "Implied and realised vol are in line.")
    return out


def correlations(daily: dict[str, list[dict]]) -> dict:
    n = RULES["corr_days"]
    series = {}
    for s in ALL_SYMBOLS:
        rows = daily.get(s) or []
        series[s] = {r["d"]: r["c"] for r in rows if r.get("c") and r.get("d")}
    syms = [s for s in ALL_SYMBOLS if len(series[s]) > 25]
    ret = {}
    for s in syms:
        ds = sorted(series[s])[-(n + 1):]
        ret[s] = {ds[i]: math.log(series[s][ds[i]] / series[s][ds[i - 1]]) for i in range(1, len(ds)) if series[s][ds[i - 1]] > 0 and series[s][ds[i]] > 0}
    matrix = []
    for a in syms:
        row = []
        for b in syms:
            common = sorted(set(ret[a]) & set(ret[b]))
            if len(common) < RULES["corr_min_overlap"]:
                row.append(None)
                continue
            x, y = [ret[a][d] for d in common], [ret[b][d] for d in common]
            sx, sy = _std(x), _std(y)
            if not sx or not sy:
                row.append(None)
                continue
            mx, my = sum(x) / len(x), sum(y) / len(y)
            row.append(round(sum((p - mx) * (q - my) for p, q in zip(x, y)) / (len(x) - 1) / (sx * sy), 2))
        matrix.append(row)
    return {"symbols": syms, "matrix": matrix, "days": n, "note": f"Pearson correlation of daily log returns over the last {n} sessions; blank = fewer than {RULES['corr_min_overlap']} shared days."}


def cross_asset(daily: dict[str, list[dict]]) -> dict:
    rows = []
    for s in ALL_SYMBOLS:
        c = _closes(daily.get(s) or [])
        if len(c) < 22:
            rows.append({"symbol": s, "available": False})
            continue
        r = _rets(c)
        sd = _std(r[-61:-1]) if len(r) > 30 else float("nan")
        z = (r[-1] / sd) if sd and sd == sd else None
        rows.append({"symbol": s, "available": True, "chg_1d": (c[-1] / c[-2] - 1) * 100, "chg_5d": (c[-1] / c[-6] - 1) * 100,
                     "chg_20d": (c[-1] / c[-21] - 1) * 100, "z_1d": z, "unusual": bool(z is not None and abs(z) >= RULES["unusual_move_z"])})
    by = {r["symbol"]: r for r in rows if r.get("available")}
    score, parts = 0, []
    for sym, sign, label in (("ES", 1, "S&P up"), ("NQ", 1, "Nasdaq up"), ("RTY", 1, "Russell up"), ("VIX", -1, "VIX down"), ("DXY", -1, "Dollar down")):
        r = by.get(sym)
        if r:
            v = 1 if r["chg_5d"] * sign > 0 else -1
            score += v
            parts.append(f"{label}: {'yes' if v > 0 else 'no'}")
    tone = None if not parts else ("Risk-on" if score >= 3 else "Risk-off" if score <= -3 else "Mixed")
    return {"rows": rows, "tone": tone, "tone_rule": "Over 5 days, one point each for S&P up, Nasdaq up, Russell up, VIX down, Dollar down (minus one when not). +3 or more = Risk-on, -3 or less = Risk-off.",
            "tone_parts": parts, "unusual_rule": f"Unusual = today's move is at least {RULES['unusual_move_z']:.0f} standard deviations of the last 60 days."}


def geopolitics(theaters: list[dict], headlines: list[dict]) -> list[dict]:
    """Top theaters by heat with the instruments they touch and their lead headline."""
    out = []
    for t in sorted(theaters, key=lambda x: -(x.get("score") if x.get("score") is not None else -1))[:5]:
        if t.get("score") is None:
            continue
        out.append({"id": t["id"], "name": t["name"], "score": t["score"], "delta": t.get("delta"), "instruments": t.get("instruments") or [], "top": t.get("top")})
    return out


def build(daily: dict[str, list[dict]], events: list[dict], theaters: list[dict], headlines: list[dict], now: datetime) -> dict:
    return {"server_time": now.isoformat(), "rules": {"values": RULES, "text": RULE_TEXT}, "sessions": sessions(now),
            "central_banks": central_banks(now, events), "macro": macro_releases(events, now), "futures": futures_stats(daily),
            "vol": vol_regime(daily), "correlations": correlations(daily), "cross_asset": cross_asset(daily),
            "regimes": regimes(daily, events, now), "geopolitics": geopolitics(theaters, headlines),
            "data_note": "Daily bars from Stooq (delayed). Context only - not a signal and not tick-level."}
