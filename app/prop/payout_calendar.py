"""
Payout Calendar: when each funded account can next ask for a payout, from the firm's own payout rule and the payouts you have logged.

``payout_frequency_days`` in the presets counts TRADE days; the calendar converts it to weekdays and says so, so a date is an estimate to
plan around, never a promise. Confirm the date in the firm's own payout dashboard.
"""
from __future__ import annotations

from datetime import date, timedelta

from app.prop import desk_store as ds
from app.prop import presets as P


def add_weekdays(start: date, n: int) -> date:
    d = start
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def events(data: dict | None = None, today: date | None = None) -> list[dict]:
    data = data or ds.load()
    today = today or date.today()
    out = []
    for a in data["accounts"]:
        if a["stage"] != "funded" or a["status"] in ("failed", "closed"):
            continue
        freq, note = 14, "Default 14 trade days (no firm rule on file)."
        try:
            p = P.get_preset(a["preset_key"]) if a.get("preset_key") else None
            if p:
                freq, note = int(p.payout_frequency_days), f"{p.firm}: one payout per {p.payout_frequency_days} trade days (counted here as weekdays)."
        except KeyError:
            pass
        pays = sorted(a.get("payouts", []), key=lambda x: x["date"])
        base = date.fromisoformat(pays[-1]["date"]) if pays else (date.fromisoformat(a["funded_on"]) if a.get("funded_on") else None)
        if base is None:
            out.append({"account_id": a["id"], "label": a["label"], "date": None, "kind": "unknown", "days_away": None,
                        "note": "Add the date this account was funded to see its payout window."})
            continue
        due = add_weekdays(base, freq)
        out.append({"account_id": a["id"], "label": a["label"], "firm": a["firm"], "date": due.isoformat(), "kind": "eligible" if due <= today else "upcoming",
                    "days_away": (due - today).days, "note": note + (" Since the last payout." if pays else " Since funding."),
                    "last_payout": pays[-1] if pays else None})
    return sorted(out, key=lambda e: (e["date"] is None, e["date"] or ""))


def ics(evts: list[dict]) -> str:
    """An iCalendar file of the payout windows, for any calendar app."""
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//T58//Payout Calendar//EN", "CALSCALE:GREGORIAN"]
    for e in evts:
        if not e.get("date"):
            continue
        d = e["date"].replace("-", "")
        nxt = (date.fromisoformat(e["date"]) + timedelta(days=1)).isoformat().replace("-", "")
        lines += ["BEGIN:VEVENT", f"UID:t58-payout-{e['account_id']}-{d}@t58", f"DTSTAMP:{date.today().isoformat().replace('-', '')}T000000Z",
                  f"DTSTART;VALUE=DATE:{d}", f"DTEND;VALUE=DATE:{nxt}", f"SUMMARY:Payout window: {e['label']}",
                  "DESCRIPTION:" + e["note"].replace("\n", " ") + " Estimate only; confirm with the firm.", "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"
