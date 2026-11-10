"""
Strategy Family Intelligence: a map of which KINDS of strategy have worked, built from what the app has actually tested.

Every family (trend, mean reversion, breakout, ...) gets six measured facts, each with the number of candidates behind it:

    profitable            share with a positive net average R at modelled costs
    significant           share whose own significance test passed (random-entry null gate when it ran, else the day-block bootstrap, labelled)
    survives higher cost  share still profitable with round-trip friction doubled
    redundant             share of candidates that sit in a parameter neighbourhood already tested (many near-copies = one idea)
    across instruments    number of different instruments where the family produced a profitable candidate
    prop-rule fit         mean evaluation pass probability the candidates showed under the prop rules in force when they were tested

Source: the Strategy Graveyard (every rejected candidate, with its edge diagnostics) plus the saved Strategy Library (survivors, counted per
family). The graveyard holds rejections by construction, so "profitable" here means "profitable among the candidates that were tested and
dropped", and a family with many profitable-but-rejected candidates is the interesting kind. Candidates with a missing figure are left out of
that figure's denominator, never counted as a pass or a fail. The prop-rule fit is NOT per firm: the rows do not record which firm's rules
were used, so it is a mean over the rules each run used.
"""
from __future__ import annotations

from collections import defaultdict

MIN_N = 5     # a family needs this many candidates with a figure before that figure is shown as a rate


def _share(num: int, den: int):
    return (num / den) if den >= MIN_N else None


def family_of(row: dict) -> str:
    try:
        from app.strategy.family_taxonomy import classify_record
        return classify_record({"family": row.get("family"), "source_type": "manual", "config": {}}) or "uncategorized"
    except Exception:  # noqa: BLE001
        return str(row.get("family") or "uncategorized")


def summarize(rows: list[dict], library_counts: dict | None = None) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[family_of(r)].append(r)
    for fam in (library_counts or {}):
        groups.setdefault(fam, [])
    fams = []
    for fam, rs in groups.items():
        with_r = [r for r in rs if r.get("avg_r_net") is not None]
        profitable = [r for r in with_r if r["avg_r_net"] > 0]
        sig_rows = [r for r in rs if r.get("null_gate_p") is not None or r.get("bootstrap_p") is not None]
        sig = [r for r in sig_rows if (r["null_gate_p"] if r.get("null_gate_p") is not None else r["bootstrap_p"]) < 0.05]
        gate_used = sum(1 for r in sig_rows if r.get("null_gate_p") is not None)
        cost_rows = [r for r in rs if r.get("doubled_cost_net_profit") is not None]
        cost_ok = [r for r in cost_rows if r["doubled_cost_net_profit"] > 0]
        sigs = [r.get("param_signature") for r in rs if r.get("param_signature")]
        redundancy = (1.0 - len(set(sigs)) / len(sigs)) if len(sigs) >= MIN_N else None
        inst = sorted({r["instrument"] for r in profitable if r.get("instrument")})
        all_inst = sorted({r["instrument"] for r in rs if r.get("instrument")})
        prop = [r["prop_sim_pass_pct"] for r in rs if r.get("prop_sim_pass_pct") is not None]
        survivors = int((library_counts or {}).get(fam, 0))
        fams.append({
            "family": fam, "label": fam.replace("_", " ").title(), "candidates": len(rs), "survivors": survivors,
            "profitable": _share(len(profitable), len(with_r)), "profitable_n": len(with_r),
            "significant": _share(len(sig), len(sig_rows)), "significant_n": len(sig_rows),
            "significance_basis": ("null gate" if gate_used * 2 >= len(sig_rows) and sig_rows else "bootstrap screen") if sig_rows else None,
            "survives_cost": _share(len(cost_ok), len(cost_rows)), "survives_cost_n": len(cost_rows),
            "redundancy": redundancy, "redundancy_n": len(sigs),
            "instruments_profitable": inst, "instruments_tested": all_inst,
            "prop_fit_pct": (sum(prop) / len(prop)) if len(prop) >= MIN_N else None, "prop_fit_n": len(prop),
        })
    for f in fams:
        f["tags"] = _tags(f)
        f["verdict"] = _verdict(f)
    fams.sort(key=lambda f: (-(f["profitable"] or 0), -f["candidates"]))
    return {"families": fams, "n_candidates": len(rows), "min_n": MIN_N,
            "note": ("Built from the Strategy Graveyard (rejected candidates) and the Library (survivors). 'Profitable' means net-positive average R at modelled "
                     "costs among tested candidates. A figure appears only when at least %d candidates carry it. Prop-rule fit is a mean over the rules each run used, not per firm." % MIN_N)}


def _tags(f: dict) -> list[str]:
    t = []
    if f["profitable"] is not None and f["profitable"] >= 0.5:
        t.append("profitable")
    if f["significant"] is not None and f["significant"] >= 0.2:
        t.append("statistically significant")
    if f["survives_cost"] is not None and f["survives_cost"] >= 0.5:
        t.append("survives higher costs")
    if f["redundancy"] is not None and f["redundancy"] >= 0.6:
        t.append("redundant")
    if len(f["instruments_profitable"]) >= 3:
        t.append("works across instruments")
    if f["prop_fit_pct"] is not None and f["prop_fit_pct"] >= 50:
        t.append("fits prop rules")
    return t


def _verdict(f: dict) -> str:
    if f["candidates"] < MIN_N:
        return "Too few candidates to judge."
    good = [x for x in f["tags"] if x != "redundant"]
    if len(good) >= 3:
        return "A family worth more search time."
    if f["profitable"] is not None and f["profitable"] < 0.1:
        return "Almost nothing here makes money after costs."
    if "redundant" in f["tags"]:
        return "Many near-copies of the same idea: search somewhere new."
    return "Mixed: some signal, nothing convincing yet."


def load() -> dict:
    """Summary of everything on disk right now."""
    from app.search.graveyard import load_graveyard
    rows = load_graveyard()
    counts: dict[str, int] = defaultdict(int)
    try:
        from app.strategy.family_taxonomy import classify_family
        from app.strategy.library import list_saved_strategies
        for s in list_saved_strategies():
            tags = [t.lower() for t in s.tags]
            fam = next((t[len("family:"):] for t in tags if t.startswith("family:")), None) or classify_family(raw_text=s.name)
            counts[fam] += 1
    except Exception:  # noqa: BLE001
        pass
    return summarize(rows, dict(counts))
