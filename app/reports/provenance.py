"""Report provenance lines (2026-11-12 audit P0-3 / P0-5).

* ``gate_provenance()``: the state of every environment switch that can loosen a certification run, printed in every
  report header next to the data stamp so a number is never read without knowing which gates produced it.
* ``risk_cap_audit()``: post-hoc check of the per-trade risk cap -- the largest risk any trade actually carried at its
  stop versus the 1%-of-account cap, regardless of what the configuration said.
"""
from __future__ import annotations

import os

PROP_RISK_CAP_PCT = 1.0
_FALSY = ("0", "false", "no", "off")


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return None if v is None else v.strip()


def gate_provenance() -> dict:
    skip_raw = _env("T58_SKIP_EXTRA_GATES")
    fam_raw = _env("T58_USE_ALL_FAMILIES")
    skip_active = skip_raw == "1"
    fam_active = fam_raw is None or fam_raw.lower() not in _FALSY
    # Certification is clean only when no gate is bypassed. The family switch widens WHAT is searched (every family still
    # faces the same gates), so it is reported but does not by itself void certification; the gate bypass does.
    return {
        "T58_SKIP_EXTRA_GATES": skip_raw if skip_raw is not None else "unset",
        "T58_USE_ALL_FAMILIES": fam_raw if fam_raw is not None else "unset",
        "extra_gates_bypassed": skip_active,
        "all_families_search": fam_active,
        "certification_clean": not skip_active,
    }


def gate_provenance_line(prov: dict | None = None) -> str:
    p = prov or gate_provenance()
    state = "CLEAN" if p.get("certification_clean") else "NOT CERTIFIABLE (extra gates bypassed)"
    return (f"Gate provenance: {state} | T58_SKIP_EXTRA_GATES={p.get('T58_SKIP_EXTRA_GATES')} | "
            f"T58_USE_ALL_FAMILIES={p.get('T58_USE_ALL_FAMILIES')} "
            f"({'all families searched' if p.get('all_families_search') else 'narrowed family set'})")


def cost_basis_line() -> str:
    """P2-17: the cost basis stated in every report (Alpha's schedule, conservative vs Lucid's cheaper commission)."""
    try:
        from app.prop.firm_registry import cost_basis_line as _c
        return _c()
    except Exception:  # noqa: BLE001
        return ""


def risk_cap_audit(trades, risk_config=None, prop_rules=None) -> dict | None:
    """Max realized per-trade risk (dollars at the stop, costs included) vs the 1%-of-account cap. None with no trades."""
    try:
        acct = float(getattr(prop_rules, "account_size", None) or getattr(risk_config, "initial_balance", 0) or 0)
        if acct <= 0 or not trades:
            return None
        vals = []
        for t in trades:
            r = getattr(t, "risk_at_stop_dollars", None)
            if not r:
                ir, sz = getattr(t, "initial_risk", None), getattr(t, "size", None)
                r = (ir * sz) if (ir and sz) else getattr(t, "intended_risk_dollars", None)
            if r:
                vals.append(float(r))
        if not vals:
            return None
        cap = acct * PROP_RISK_CAP_PCT / 100.0
        mx = max(vals)
        return {"max_realized_risk_dollars": mx, "cap_dollars": cap, "cap_pct": PROP_RISK_CAP_PCT,
                "over_cap_trades": sum(1 for v in vals if v > cap * 1.001), "n_trades": len(vals),
                "within_cap": mx <= cap * 1.001,
                "config_note": getattr(risk_config, "risk_cap_note", "") or ""}
    except Exception:  # noqa: BLE001 -- an audit line must never break a report
        return None


def risk_cap_audit_line(a: dict | None) -> str:
    if not a:
        return ""
    s = (f"Risk-cap audit: max realized risk per trade ${a['max_realized_risk_dollars']:,.2f} vs cap ${a['cap_dollars']:,.2f} "
         f"({a['cap_pct']:g}% of account) -- {'WITHIN CAP' if a['within_cap'] else 'OVER CAP on ' + str(a['over_cap_trades']) + ' of ' + str(a['n_trades']) + ' trades'}")
    return s + (f" [{a['config_note']}]" if a.get("config_note") else "")
