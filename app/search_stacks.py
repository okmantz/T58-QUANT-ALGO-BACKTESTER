"""The three search stacks and the ONE job each is allowed to do (2026-11-12 audit P2-14).

Stack A  app.search.batch_runner.run_search (Search Lab funnel; Speed Run / Forge / cross-instrument / Autopilot wrap it)
         -> FEEDER ONLY. Its best survivor is a *finalist* (and a *champion* only after its own Stage-4 DSR/pool evidence); neither
         word means certified. Finalists are routed to Stack C.
Stack B  app.discovery (idea compiler, hypotheses, experiment runner, break battery)
         -> its ORACLE role is retired. The break battery (app.discovery.break_battery) is kept and is run INSIDE Stack C's promotion
         sequence; Stack B no longer produces a verdict of its own.
Stack C  app.discovery_engine (SealedLedger, HoldoutBudget, promotion sequence)
         -> THE single certification door: the only place a strategy earns the right to be called certified.
Full Pipeline (app.orchestration.full_pipeline) READY additionally requires the Rulebook gates (see ``rulebook_waive_empty_register``).
"""
from __future__ import annotations

STACK_ROLES = {
    "A": {"module": "app.search.batch_runner", "role": "feeder", "may_certify": False},
    "B": {"module": "app.discovery", "role": "break_battery_provider", "may_certify": False},
    "C": {"module": "app.discovery_engine", "role": "certification_door", "may_certify": True},
}


def certification_door() -> str:
    return next(v["module"] for v in STACK_ROLES.values() if v["may_certify"])


def may_certify(stack: str) -> bool:
    return bool(STACK_ROLES.get(stack, {}).get("may_certify", False))
