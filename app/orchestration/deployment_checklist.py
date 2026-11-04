"""Deployment checklist (2026-11-04 audit, P1-3).

The verdict answers "is there an edge worth paying to test?". Whether it is safe to BUY an evaluation today is a different question that
depends on facts that age or live outside the backtest: how old the firm-rules stamp is, whether the firm's automation policy was
confirmed in writing, whether the data covers enough history, and whether the fill / latency assumptions have been checked against a
real account. Those used to be folded into the verdict (a stale rules stamp demoted READY); they are now explicit checklist items so a
good strategy is not marked down for the calendar, and a purchase is not made on stale facts.

``build_checklist`` returns a list of items {id, label, status ("ok" | "blocker" | "warn"), detail}; ``blockers`` filters them.
Pure function over the audit evidence dict; never raises.
"""
from __future__ import annotations


def build_checklist(audit_evidence: dict | None, *, fee=None, ladder_stage: str | None = None) -> list:
    ev = audit_evidence or {}
    items: list = []

    def add(i, label, status, detail=""):
        items.append({"id": i, "label": label, "status": status, "detail": str(detail)})

    rs = ev.get("rules_stamp") or {}
    st = rs.get("state")
    if st == "ok":
        add("rules_stamp", "Firm rules verified recently", "ok")
    elif st == "stale":
        add("rules_stamp", "Firm rules verified recently", "blocker", rs.get("note", "stamp older than the age limit: re-check the firm's page before buying"))
    elif st == "unverified":
        add("rules_stamp", "Firm rules verified against the firm's own page", "blocker", rs.get("note", ""))
    else:
        add("rules_stamp", "Firm rules verified recently", "warn", "no stamp information in this run")

    if fee is not None:
        if getattr(fee, "written_confirmation", False):
            add("automation_policy", "Automation / algo trading allowed in writing", "ok")
        else:
            add("automation_policy", "Automation / algo trading allowed in writing", "blocker",
                "the firm's automation policy has not been confirmed in writing; many firms ban or restrict bots")
    else:
        add("automation_policy", "Automation / algo trading allowed in writing", "warn", "no fee row for this preset")

    dg = ev.get("data_gate") or {}
    if dg.get("ok_for_ready", True):
        add("data_manifest", "Data covers enough clean history", "ok", "" if not dg.get("cleared_by_transfer") else "cleared by the cross-market transfer test")
    else:
        add("data_manifest", "Data covers enough clean history", "blocker" if not dg.get("short_history") else "warn", "; ".join(dg.get("reasons") or []))

    bk = ev.get("bankroll") or {}
    if bk.get("ruin_pct") is not None:
        add("bankroll_ruin", "Bankroll ruin before first payout within the cap", "ok" if float(bk["ruin_pct"]) <= 20.0 else "warn", f"{float(bk['ruin_pct']):.1f}%")

    add("execution_assumptions", "Fills, spread and latency checked against a real or demo account", "warn",
        "not verifiable from a backtest: confirm on the firm's platform before relying on the simulated pass odds")
    if ladder_stage in ("EVALUATE", "FUNDED"):
        add("forward_test", "Forward test accepted", "ok")
    else:
        add("forward_test", "Forward test (SPRT) accepted before buying", "blocker" if ladder_stage != "REJECT" else "warn",
            "paper-trade under the SPRT monitor first; the backtest alone never reaches EVALUATE")
    return items


def blockers(items: list) -> list:
    return [i for i in items if i.get("status") == "blocker"]
