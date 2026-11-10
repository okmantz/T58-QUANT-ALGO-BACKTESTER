"""
My Attempts ledger: your own prop-firm economics, stated plainly.

Three questions, answered from what you actually bought and what happened to it:

    How much have I spent on fees?          (every purchase / reset / activation / subscription cost)
    What is my pass rate?                   (attempts that resolved: passed / (passed + failed))
    What most often ends my attempts?       (the most common bust cause)

and next to them, what the simulators said you should expect, so a gap between the two is visible:

    app.prop.attempt_replay      -> ``record_replay``     (fresh account at every start date, real engine)
    app.prop.rolling_evaluation  -> ``record_rolling``    (every real historical start day)
    app.prop.survival_engine     -> ``record_survival``   (Monte Carlo evaluation + funded survival)

Each ``record_*`` stores ONE summary row per run in the Prop Desk store (app.prop.desk_store), never one row per simulated
attempt: the ledger is a history of decisions, not a dump of a simulator. Real attempts and simulated expectations are never
mixed into the same number -- the "Simulated" figures are labelled as such everywhere.
"""
from __future__ import annotations

from collections import Counter
from datetime import date

from app.prop import desk_store as ds

MIN_RESOLVED_FOR_RATE = 5      # below this many resolved attempts the pass rate is shown but flagged as thin evidence


def _clean_counts(raw: dict | None) -> dict:
    out: Counter = Counter()
    for k, v in (raw or {}).items():
        try:
            n = float(v)
        except (TypeError, ValueError):
            continue
        if n > 0:
            out[ds.normalize_bust_cause(k) or "other"] += n
    return {k: (int(v) if float(v).is_integer() else round(v, 2)) for k, v in out.most_common()}


# ------------------------------------------------------------------------------------------------ wiring to the simulators
def record_replay(result, *, firm: str = "", preset_key: str = "", strategy: str = "", fee: float | None = None) -> dict:
    """Save an ``AttemptReplayResult``. ``fee`` (one evaluation purchase) is stored so cost per pass can be re-derived."""
    d = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    summary = {
        "n": int(d.get("n_attempts", 0)), "pass_rate": float(d.get("pass_rate", 0.0)), "bust_rate": float(d.get("bust_rate", 0.0)),
        "bust_causes": _clean_counts(d.get("failure_breakdown")),
        "expected_attempts_to_pass": d.get("expected_attempts_to_pass"), "expected_cost_per_pass": d.get("expected_cost_per_pass"),
        "median_days_to_pass": d.get("median_days_to_pass"), "median_days_to_payout": d.get("median_days_to_payout"),
        "eval_fee": fee, "sample_ok": bool(d.get("sample_ok", True)),
    }
    return ds.add_sim_run("attempt_replay", summary, firm=firm, preset_key=preset_key, strategy=strategy)


def record_rolling(result, *, firm: str = "", preset_key: str = "", strategy: str = "", fee: float | None = None) -> dict:
    """Save a ``RollingEvaluationResult``."""
    d = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    n = int(d.get("n_windows", 0))
    pr = float(d.get("pass_rate_pct", 0.0)) / 100.0
    passes = pr * n
    summary = {
        "n": n, "pass_rate": pr, "bust_rate": (1.0 - pr) if n else 0.0,
        "bust_causes": _clean_counts(d.get("failure_breakdown")),
        "expected_attempts_to_pass": (1.0 / pr) if pr > 0 else None,
        "expected_cost_per_pass": (fee / pr) if (fee and pr > 0) else None,
        "median_days_to_pass": d.get("median_days_to_pass"), "median_days_to_payout": None,
        "first_payout_rate": float(d.get("first_payout_rate_pct", 0.0)) / 100.0, "eval_fee": fee,
        "sample_ok": passes >= 0 and n >= 30,
    }
    return ds.add_sim_run("rolling_evaluation", summary, firm=firm, preset_key=preset_key, strategy=strategy)


def record_survival(result, *, firm: str = "", preset_key: str = "", strategy: str = "", fee: float | None = None) -> dict:
    """Save a ``PropSurvivalResult`` (Monte Carlo)."""
    d = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    ev, fu = d.get("evaluation", {}), d.get("funded", {})
    n = int(d.get("n_simulations", 0))
    p_pass = float(ev.get("probability_pass_evaluation", 0.0))
    causes = {"daily_loss_limit": float(ev.get("probability_hit_daily_loss", 0.0)) * n,
              "max_drawdown": float(ev.get("probability_hit_max_drawdown", 0.0)) * n}
    summary = {
        "n": n, "pass_rate": p_pass, "bust_rate": max(0.0, 1.0 - p_pass), "bust_causes": _clean_counts(causes),
        "expected_attempts_to_pass": (1.0 / p_pass) if p_pass > 0 else None,
        "expected_cost_per_pass": (fee / p_pass) if (fee and p_pass > 0) else None,
        "median_days_to_pass": ev.get("median_days_required"), "median_days_to_payout": fu.get("median_days_to_first_payout"),
        "first_payout_rate": fu.get("probability_first_payout"), "eval_fee": fee, "sample_ok": n >= 1000,
    }
    return ds.add_sim_run("survival", summary, firm=firm, preset_key=preset_key, strategy=strategy)


def try_record(kind: str, result, **kw) -> None:
    """Best-effort wrapper for the callers inside the pipelines: saving history must never break a run."""
    try:
        {"replay": record_replay, "rolling": record_rolling, "survival": record_survival}[kind](result, **kw)
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------------------------------------ the ledger numbers
def economics(data: dict | None = None) -> dict:
    """Everything the My Attempts tab shows, from the saved history."""
    data = data or ds.load()
    attempts = [a for a in data["attempts"] if a.get("source") in ds.SOURCES]
    passed = [a for a in attempts if a["outcome"] == "passed"]
    failed = [a for a in attempts if a["outcome"] == "failed"]
    open_ = [a for a in attempts if a["outcome"] == "open"]
    resolved = len(passed) + len(failed)

    costs = data["costs"]
    by_kind: Counter = Counter()
    by_firm: Counter = Counter()
    for c in costs:
        by_kind[c["kind"]] += float(c["amount"])
        by_firm[c.get("firm") or "Unassigned"] += float(c["amount"])
    fees_spent = round(sum(by_kind.values()), 2)
    payouts = round(sum(float(p["amount"]) for a in data["accounts"] for p in a.get("payouts", [])), 2)

    causes = Counter(a["bust_cause"] or "other" for a in failed)
    top = causes.most_common(1)
    pass_rate = (len(passed) / resolved) if resolved else None
    days = sorted(a["days"] for a in passed if isinstance(a.get("days"), int))
    median_days = (days[len(days) // 2] if len(days) % 2 else (days[len(days) // 2 - 1] + days[len(days) // 2]) / 2) if days else None

    sims = sorted(data["sim_runs"], key=lambda r: (r["created_at"], r["id"]))
    latest_sim = sims[-1] if sims else None
    comparison = None
    if latest_sim and pass_rate is not None:
        sim_rate = float(latest_sim["summary"].get("pass_rate", 0.0))
        comparison = {"real_pass_rate": pass_rate, "simulated_pass_rate": sim_rate, "gap_points": round((pass_rate - sim_rate) * 100.0, 1),
                      "sim_kind": latest_sim["kind"], "sim_strategy": latest_sim.get("strategy", ""), "sim_date": latest_sim["created_at"]}

    headline = _headline(len(attempts), resolved, pass_rate, fees_spent, payouts, top[0] if top else None)
    return {
        "attempts_total": len(attempts), "passed": len(passed), "failed": len(failed), "open": len(open_), "resolved": resolved,
        "pass_rate": pass_rate, "thin_evidence": resolved < MIN_RESOLVED_FOR_RATE,
        "fees_spent": fees_spent, "fees_by_kind": {k: round(v, 2) for k, v in by_kind.most_common()},
        "fees_by_firm": {k: round(v, 2) for k, v in by_firm.most_common()},
        "payouts_received": payouts, "net": round(payouts - fees_spent, 2),
        "cost_per_pass": round(fees_spent / len(passed), 2) if passed else None,
        "most_common_bust_cause": top[0][0] if top else None,
        "most_common_bust_label": ds.BUST_CAUSES.get(top[0][0], top[0][0]) if top else None,
        "bust_causes": [{"cause": k, "label": ds.BUST_CAUSES.get(k, k), "count": v} for k, v in causes.most_common()],
        "median_days_to_pass": median_days,
        "simulated": {"latest": latest_sim, "runs": len(sims)}, "comparison": comparison, "headline": headline,
        "attempts": sorted(attempts, key=lambda a: (a.get("started", ""), a["id"]), reverse=True),
        "costs": sorted(costs, key=lambda c: (c["date"], c["id"]), reverse=True),
    }


def _headline(n: int, resolved: int, pass_rate, fees: float, payouts: float, top) -> str:
    if n == 0:
        return "No attempts saved yet. Log a purchase on Buy Challenge or the Accounts page and this ledger fills itself in."
    parts = [f"You have spent ${fees:,.0f} on fees"]
    if resolved:
        parts.append(f"passed {pass_rate * 100:.0f}% of the {resolved} attempt{'s' if resolved != 1 else ''} that finished")
    if top:
        parts.append(f"and most often lose them to {ds.BUST_CAUSES.get(top[0], top[0]).lower()}")
    line = ", ".join(parts[:2]) + (" " + parts[2] if len(parts) > 2 else "") + "."
    if payouts > 0:
        line += f" Payouts received so far: ${payouts:,.0f} (net {'+' if payouts - fees >= 0 else '-'}${abs(payouts - fees):,.0f})."
    return line


def cost_tracker(data: dict | None = None) -> dict:
    """The Challenge Cost Tracker view: spend by month / firm / kind against payouts, with a break-even line."""
    data = data or ds.load()
    months: dict[str, dict] = {}
    for c in data["costs"]:
        m = c["date"][:7]
        row = months.setdefault(m, {"month": m, "spent": 0.0, "payouts": 0.0})
        row["spent"] += float(c["amount"])
    for a in data["accounts"]:
        for p in a.get("payouts", []):
            row = months.setdefault(p["date"][:7], {"month": p["date"][:7], "spent": 0.0, "payouts": 0.0})
            row["payouts"] += float(p["amount"])
    rows = [dict(r, spent=round(r["spent"], 2), payouts=round(r["payouts"], 2), net=round(r["payouts"] - r["spent"], 2))
            for _, r in sorted(months.items())]
    cum = 0.0
    for r in rows:
        cum += r["net"]
        r["cumulative_net"] = round(cum, 2)
    eco = economics(data)
    subs = [c for c in data["costs"] if c["kind"] == "subscription"]
    return {"months": rows, "total_spent": eco["fees_spent"], "total_payouts": eco["payouts_received"], "net": eco["net"],
            "by_kind": eco["fees_by_kind"], "by_firm": eco["fees_by_firm"], "subscriptions_total": round(sum(float(c["amount"]) for c in subs), 2),
            "to_break_even": round(max(0.0, eco["fees_spent"] - eco["payouts_received"]), 2), "generated_on": date.today().isoformat()}
