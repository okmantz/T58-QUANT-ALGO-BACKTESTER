"""Bridge from a validated research finding to the Trading Journal's tag system.

Only findings that SURVIVED the lab's own acceptance logic (multiple-testing correction + out-of-sample agreement) become
journal tags, so the tags you review your trades with are the conditions the research actually supports -- not every
pattern someone once noticed. Tags land in a journal tag group called "Research findings"."""
from __future__ import annotations

from typing import Any

GROUP_NAME = "Research findings"
LAB_LABEL = {"volume_profile": "VP", "order_flow": "Flow", "order_book": "Book", "lead_lag": "Lead-lag", "stat_arb": "Spread",
             "event_study": "Event", "seasonality": "Season", "hypothesis_lab": "Hypothesis"}


def _walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def tags_from_record(rec: dict) -> list[str]:
    lab = rec.get("lab", "")
    prefix = LAB_LABEL.get(lab, lab)
    res = rec.get("results") or {}
    tags: list[str] = []
    if lab == "hypothesis_lab":
        if res.get("accepted"):
            h = (res.get("spec") or {}).get("hypothesis")
            if h:
                tags.append(f"{prefix}: {str(h)[:70]}")
        return tags
    if lab == "stat_arb":
        status = (res.get("stability_monitor") or {}).get("status")
        if status in ("structurally broken", "stretched (relationship intact)"):
            mk = " / ".join(res.get("markets") or [])
            tags.append(f"{prefix}: {mk} {status}")
        return tags
    for d in _walk(res):
        verdict = str(d.get("verdict", ""))
        if verdict.startswith("promising"):
            name = d.get("group") or d.get("name")
            if name:
                tags.append(f"{prefix}: {name}"[:80])
        elif verdict == "robust lead" and d.get("leader") and d.get("follower"):
            tags.append(f"{prefix}: {d['leader']} leads {d['follower']} (lag {d.get('best_lag')})")
    seen, out = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
