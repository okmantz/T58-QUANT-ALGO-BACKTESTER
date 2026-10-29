"""Session brief: "Ollama interprets, the app calculates."

The prompt is built from stored rows only; every number comes from the store. A deterministic text brief is always
produced first, and an Ollama narrative is appended only if Ollama is enabled and reachable.
"""
from __future__ import annotations

from datetime import timedelta

from . import store

BRIEF_SYSTEM = (
    "You are a concise market-session analyst for a futures prop trader. Use ONLY the facts provided. Do not invent numbers, "
    "prices, dates or sources. Write one short paragraph followed by exactly three bullet points: what could move markets next, "
    "what is already moving them, and one risk to respect. If data is missing or stale, say so plainly."
)


def _fmt_age(sec: int | None) -> str:
    if sec is None:
        return "no data"
    if sec < 90:
        return f"{sec}s old"
    if sec < 5400:
        return f"{sec // 60}m old"
    return f"{sec // 3600}h old"


def build_facts(now=None) -> dict:
    now = now or store.utcnow()
    return {
        "events": store.events(now - timedelta(minutes=15), now + timedelta(hours=24), min_impact=2, limit=8),
        "headlines": store.headlines(range_h=12, min_impact=2, limit=8, now=now),
        "theaters": [t for t in store.theater_scores() if t["score"] is not None],
        "quotes": [q for q in store.quotes_latest(["ES", "NQ", "CL", "GC", "DXY", "VIX"]) if q["price"] is not None],
        "health": store.health_summary(),
    }


def deterministic_brief(facts: dict) -> str:
    lines = []
    ev = facts["events"]
    if ev:
        lines.append("Next high-impact events (UTC): " + "; ".join(f"{e['when'][11:16]} {e['country'] or ''} {e['name']}" for e in ev[:4]))
    else:
        lines.append("No high/medium-impact scheduled events in the next 24h in the local store.")
    hot = sorted(facts["theaters"], key=lambda t: t["score"], reverse=True)[:3]
    if hot:
        lines.append("Hottest theaters: " + ", ".join(f"{t['short']} {t['score']:.0f}" for t in hot) + " (experimental score).")
    if facts["headlines"]:
        lines.append("Top headlines: " + " | ".join(f"{h['title']} ({h['publisher']})" for h in facts["headlines"][:3]))
    else:
        lines.append("No impact>=2 headlines in the last 12h in the local store.")
    if facts["quotes"]:
        lines.append("Quotes vs open: " + ", ".join(f"{q['symbol']} {q['chg_pct']:+.2f}%" for q in facts["quotes"] if q["chg_pct"] is not None))
    h = facts["health"]
    lines.append(f"Data health: {h['overall']}; freshest source {_fmt_age(h['freshest_age_sec'])}.")
    return "\n".join(lines)


def structured_brief(facts: dict) -> dict:
    """The same facts as deterministic_brief(), shaped for a readable card layout: a one-line takeaway plus typed sections
    (events / theaters / headlines / markets) that the page renders as lists, bars and chips instead of one text block."""
    ev, hl = facts["events"], facts["headlines"]
    hot = sorted(facts["theaters"], key=lambda t: t["score"], reverse=True)[:4]
    quotes = [q for q in facts["quotes"] if q.get("chg_pct") is not None]
    h = facts["health"]
    bits = []
    if ev:
        bits.append(f"{len(ev)} scheduled event{'s' if len(ev) != 1 else ''} in the next 24h")
    if hot:
        bits.append(f"hottest theater: {hot[0]['short']}")
    if quotes:
        mv = max(quotes, key=lambda q: abs(q["chg_pct"]))
        bits.append(f"biggest mover: {mv['symbol']} {mv['chg_pct']:+.2f}%")
    takeaway = ("; ".join(bits).capitalize() + ".") if bits else "Quiet: nothing notable in the local store right now."
    sections = []
    sections.append({"type": "events", "title": "Coming up (next 24h)", "empty": "No high or medium impact events scheduled in the next 24 hours.",
                     "items": [{"when": e["when"], "country": e.get("country") or "", "name": e["name"], "impact": e.get("impact")} for e in ev[:5]]})
    sections.append({"type": "theaters", "title": "Hottest theaters", "empty": "No theater scores yet.",
                     "items": [{"name": t.get("short") or t.get("name"), "score": round(float(t["score"]))} for t in hot],
                     "note": "Experimental score, 0-100."})
    sections.append({"type": "headlines", "title": "Top headlines", "empty": "No impact 2+ headlines in the last 12 hours.",
                     "items": [{"title": x["title"], "publisher": x.get("publisher") or ""} for x in hl[:4]]})
    sections.append({"type": "markets", "title": "Markets vs open", "empty": "No quotes available.",
                     "items": [{"symbol": q["symbol"], "chg_pct": round(float(q["chg_pct"]), 2)} for q in quotes]})
    return {"takeaway": takeaway, "sections": sections,
            "health": {"overall": h.get("overall"), "age": _fmt_age(h.get("freshest_age_sec"))}}


def build_prompt(facts: dict) -> str:
    return "FACTS (all figures come from the app's local store):\n" + deterministic_brief(facts) + "\n\nWrite the session brief now."


def make_brief() -> dict:
    facts = build_facts()
    text = deterministic_brief(facts)
    out = {"text": text, "brief": structured_brief(facts), "ai": None, "error": None, "label": "Summary of the listed headlines and figures"}
    try:
        from app.ai import ollama_transport
        from app.ai.ollama_settings import load_settings

        settings = load_settings()
        if not settings.is_usable:
            out["error"] = "Ollama is off -- showing the deterministic summary only."
            return out
        out["ai"] = ollama_transport.generate(settings, build_prompt(facts), system=BRIEF_SYSTEM,
                                              total_timeout=90, first_token_timeout=45, stall_timeout=30).strip()
    except Exception as exc:  # noqa: BLE001 -- the brief must always reach a terminal state
        out["error"] = f"Ollama narrative unavailable: {str(exc)[:160]}"
    return out
