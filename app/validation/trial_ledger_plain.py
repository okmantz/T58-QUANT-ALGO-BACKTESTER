"""The trial ledger in plain language: how many things were tried, and how many the Deflated Sharpe check corrects for.

A strategy that looks great after 10,000 attempts is much less impressive than one that looked great on its first. The Deflated Sharpe
Ratio (and the Bonferroni gate) therefore ask "how many ideas did you try before this one?" and demand a stronger result the bigger the
number is. This module reads the same ledger those checks read (``app.validation.trial_ledger``) and says it in words.

Pure reading: nothing is written or recomputed here, so what the page says is what the verdict used.
"""
from __future__ import annotations

import time

from app.validation import trial_ledger as tl


def _label_of(key: str) -> tuple[str, str]:
    """('family'|'strategy', readable name) from a ledger key such as ``family_ema_cross`` or ``my_strategy``."""
    if key.startswith("family_"):
        return "family", key[len("family_"):].replace("_", " ")
    return "strategy", key.replace("_", " ")


def _age_text(t: float | None, now: float) -> str:
    if not t:
        return "date unknown"
    d = max(0.0, (now - float(t)) / 86400.0)
    if d < 1:
        return "today"
    if d < 2:
        return "yesterday"
    if d < 60:
        return f"{int(d)} days ago"
    if d < 730:
        return f"{int(d // 30)} months ago"
    return f"{d / 365:.1f} years ago"


def plain_summary(now: float | None = None, top: int = 40) -> dict:
    now = time.time() if now is None else float(now)
    data = tl._load()                      # noqa: SLF001 -- read-only view of the same file the verdict reads
    rows = []
    for key, value in data.items():
        if key.startswith("_"):
            continue
        ents = tl._entries(value)          # noqa: SLF001
        raw = int(sum(int(e.get("n", 0)) for e in ents))
        if raw <= 0:
            continue
        counted = tl.total(key)            # the aged count the Deflated Sharpe actually uses (default half-life)
        last = max((float(e["t"]) for e in ents if e.get("t")), default=None)
        kind, name = _label_of(key)
        rows.append({"key": key, "kind": kind, "name": name, "tried": raw, "counted": int(counted), "last": _age_text(last, now), "last_t": last})
    rows.sort(key=lambda r: -r["tried"])
    fam = [r for r in rows if r["kind"] == "family"]
    strat = [r for r in rows if r["kind"] == "strategy"]
    kinds_total: dict = {}
    kd = data.get(tl.KINDS_KEY)
    if isinstance(kd, dict):
        for run in kd.values():
            for k, n in (run or {}).items():
                kinds_total[k] = kinds_total.get(k, 0) + int(n)
    total_tried = int(sum(r["tried"] for r in fam)) or int(sum(r["tried"] for r in rows))
    total_counted = int(sum(r["counted"] for r in fam)) or int(sum(r["counted"] for r in rows))
    plain = {
        "prefilter_kills": "ideas thrown out by a cheap pre-screen",
        "stage1": "ideas scored in the first (fast) round",
        "ga_inner_loop": "settings tried by the optimizer's search",
        "grammar_mutations": "rule variations the search invented",
        "walk_forward": "settings picked inside walk-forward windows",
        "seed_evals": "starting ideas",
        "other": "other evaluations",
    }
    breakdown = [{"label": plain.get(k, k), "n": n} for k, n in sorted(kinds_total.items(), key=lambda kv: -kv[1]) if n]
    if not rows:
        headline = "Nothing has been tried yet, so the Deflated Sharpe check is not correcting for anything."
    else:
        headline = (f"Across all runs, {total_tried:,} ideas have been tried. Older tries on different data count for less over time, so the "
                    f"Deflated Sharpe check currently corrects for about {total_counted:,} of them.")
    return {
        "headline": headline, "total_tried": total_tried, "total_counted": total_counted, "families": fam[:top], "strategies": strat[:top],
        "n_families": len(fam), "n_strategies": len(strat), "breakdown": breakdown,
        "explain": [
            "Every time a search, optimizer or discovery run scores a candidate, one trial is written to this ledger.",
            "The Deflated Sharpe check asks: given this many tries, how good would the BEST one look by pure luck? Your strategy has to beat that.",
            "A try on the exact same data never fades, because it adds the same luck. A try on older, different data fades with a two-year half-life.",
            "Similar tries are grouped when their daily results move together, so ten near-identical variants do not count as ten separate ideas.",
        ],
        "half_life_days": tl.DEFAULT_HALF_LIFE_DAYS,
    }
