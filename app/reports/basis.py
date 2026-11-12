"""Evaluation basis: WHAT a pass-rate / payout number was measured on.

Why this exists (Owen, 2026-11): the same strategy showed 94% eval pass on the Dashboard, 39% in Validate and
100% in Optimize. Each number was correct for the setup it was measured on -- but the setups were different
(risk per trade, sizing mode, data window, prop preset) and nothing told the user. A pass rate is only comparable
between two engines when both used the same basis, so every engine now records its basis next to its numbers and
this module turns any difference into plain English ("Full Pipeline risked 0.19% per trade, Validate risked 1%").

Pure functions, no I/O. A basis is a small dict::

    {"source": "full_pipeline" | "validate" | "optimize", "prop_preset": str, "risk_pct": float | None,
     "sizing_mode": str, "timeframe": str, "window": "full" | "dev_80", "trades": int,
     "reset_on_breach": bool | None}
"""
from __future__ import annotations

from typing import Any, Optional

SOURCE_LABEL = {"full_pipeline": "Full Pipeline", "validate": "Validate", "optimize": "Optimize"}
WINDOW_LABEL = {"full": "the full history", "dev_80": "the first 80% of the history (the last 20% is a sealed holdout)"}
SIZING_LABEL = {"fit_stop": "fixed % risk per trade", "rr_planned": "sized to the planned stop/target in dollars"}

_KEYS = ("prop_preset", "risk_pct", "sizing_mode", "timeframe", "window", "reset_on_breach")


def _f(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def make_basis(source: str, *, prop_preset: str = "", risk_pct: Any = None, sizing_mode: str = "",
               timeframe: str = "", window: str = "full", trades: Any = None,
               reset_on_breach: Optional[bool] = None, engine: Optional[dict] = None) -> dict:
    r = _f(risk_pct)
    try:
        n = int(trades) if trades is not None else None
    except (TypeError, ValueError):
        n = None
    out = {"source": source, "prop_preset": str(prop_preset or ""), "risk_pct": None if r is None else round(r, 3),
           "sizing_mode": str(sizing_mode or ""), "timeframe": str(timeframe or ""), "window": window or "full",
           "trades": n, "reset_on_breach": reset_on_breach}
    if engine:
        # Everything needed to re-run the SAME measurement elsewhere (risk model, adaptive throttle, Monte Carlo
        # settings, holdout split). Validate's "same setup as the Full Pipeline" mode reads this.
        out["engine"] = engine
    return out


def fingerprint(basis: Optional[dict]) -> str:
    """Stable short string; equal fingerprints mean the numbers are directly comparable."""
    b = basis or {}
    return "|".join(str(b.get(k) if b.get(k) not in (None, "") else "-") for k in _KEYS)


def differences(a: Optional[dict], b: Optional[dict]) -> list[str]:
    """Plain-English reasons two bases are not comparable (empty list == they are)."""
    if not a or not b:
        return []
    sa, sb = SOURCE_LABEL.get(a.get("source"), "The first run"), SOURCE_LABEL.get(b.get("source"), "the second run")
    out: list[str] = []
    ra, rb = _f(a.get("risk_pct")), _f(b.get("risk_pct"))
    if ra is not None and rb is not None and abs(ra - rb) > 0.02:
        out.append(f"{sa} risked {ra:g}% of the account per trade; {sb} risked {rb:g}%. "
                   "Risk size is the biggest lever on pass rate: smaller risk survives the drawdown limit, bigger risk reaches the target faster.")
    if a.get("sizing_mode") and b.get("sizing_mode") and a["sizing_mode"] != b["sizing_mode"]:
        out.append(f"Position sizing differs: {sa} used {SIZING_LABEL.get(a['sizing_mode'], a['sizing_mode'])}, "
                   f"{sb} used {SIZING_LABEL.get(b['sizing_mode'], b['sizing_mode'])}.")
    if a.get("window") != b.get("window") and a.get("window") and b.get("window"):
        out.append(f"{sa} measured {WINDOW_LABEL.get(a['window'], a['window'])}; {sb} measured {WINDOW_LABEL.get(b['window'], b['window'])}.")
    if a.get("prop_preset") and b.get("prop_preset") and a["prop_preset"] != b["prop_preset"]:
        out.append(f"Different prop-firm rules: {a['prop_preset']} vs {b['prop_preset']}.")
    if a.get("timeframe") and b.get("timeframe") and a["timeframe"] != b["timeframe"]:
        out.append(f"Different timeframes: {a['timeframe']} vs {b['timeframe']}.")
    ta, tb = a.get("trades"), b.get("trades")
    if ta and tb and (ta > 1.5 * tb or tb > 1.5 * ta):
        out.append(f"Different trade samples: {ta} trades vs {tb} trades.")
    return out


def saved_basis_from_last_run(last_run: Optional[dict]) -> Optional[dict]:
    lr = last_run or {}
    b = lr.get("basis")
    return b if isinstance(b, dict) and b else None


def explain_gap(this_basis: dict, saved_last_run: Optional[dict], this_pct: Optional[float],
                label: str = "eval pass chance") -> Optional[dict]:
    """If another engine already measured this strategy, say whether the gap is a real disagreement or just a
    different setup. Returns {"same_basis": bool, "saved_pct": float|None, "lines": [...]} or None."""
    lr = saved_last_run or {}
    saved = saved_basis_from_last_run(lr)
    saved_pct = _f(lr.get("eval_pass_probability"))
    if saved is None and saved_pct is None:
        return None
    if saved is None:
        # An older run with no basis recorded: we can still show the number and say it can't be compared.
        return {"same_basis": False, "saved_pct": saved_pct, "lines": [
            "The saved Full Pipeline result predates basis tracking, so its exact setup is unknown. "
            "Re-run the Full Pipeline once and the two will be compared automatically."]}
    diffs = differences(saved, this_basis)
    return {"same_basis": not diffs, "saved_pct": saved_pct, "lines": diffs}


def annotate_mc_item(item: dict, this_basis: dict, saved_last_run: Optional[dict]) -> dict:
    """Append the 'why this differs from the pipeline number' explanation to a Validate Monte Carlo item."""
    try:
        ev = None
        for m in (item.get("metrics") or []):
            k, v = (m.get("label"), m.get("value")) if isinstance(m, dict) else m
            if str(k).lower().startswith("eval pass"):
                ev = _f(str(v).replace("%", ""))
        gap = explain_gap(this_basis, saved_last_run, ev)
        if not gap:
            return item
        saved_pct = gap["saved_pct"]
        head = ""
        if saved_pct is not None:
            head = f"Full Pipeline saved {saved_pct:.0f}% eval pass for this strategy; this check says {ev:.0f}%. " if ev is not None else ""
        if gap["same_basis"] and ev is not None and saved_pct is not None and abs(ev - saved_pct) <= 3.0:
            note = (f"Matches the Full Pipeline's {saved_pct:.0f}% -- same setup, same simulation settings, so the engines agree.")
        elif gap["same_basis"]:
            note = head + "Both were measured on the same setup, so this is a real disagreement -- trust the lower one until a re-run explains it."
        else:
            note = head + "They are NOT contradicting each other -- they measured different setups: " + " ".join(gap["lines"])
        item["detail"] = ((item.get("detail") or "") + " " + note).strip()
        g = item.get("guidance")
        if isinstance(g, dict):
            g["basis_note"] = note
            steps = list(g.get("steps") or [])
            if not gap["same_basis"] and gap["lines"]:
                steps.insert(0, "Run Validate with the same risk size, sizing and data window the Full Pipeline used "
                                "(its Settings Used panel lists them) to get a like-for-like number.")
            g["steps"] = steps
    except Exception:  # noqa: BLE001 -- an explanation must never break the check
        pass
    return item


def engine_from_run(risk: Any, *, adaptive: Optional[dict] = None, mc: Optional[dict] = None,
                    holdout_frac: Optional[float] = None) -> dict:
    """The reproducible part of a basis: plain JSON, no objects."""
    keys = ("risk_mode", "risk_value", "sizing_mode", "planned_target_dollars", "initial_balance")
    out = {k: getattr(risk, k, None) for k in keys}
    out["adaptive"] = adaptive or {"enabled": False}
    out["mc"] = mc or {}
    out["holdout_frac"] = holdout_frac
    return out


def reproducible_engine(saved_last_run: Optional[dict], prop_preset: str = "") -> Optional[dict]:
    """The saved engine settings if a Full Pipeline run recorded them for this preset, else None."""
    b = saved_basis_from_last_run(saved_last_run)
    if not b or b.get("source") != "full_pipeline" or not isinstance(b.get("engine"), dict):
        return None
    if prop_preset and b.get("prop_preset") and b["prop_preset"] != prop_preset:
        return None
    return b
