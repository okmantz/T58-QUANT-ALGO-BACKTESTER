"""Failure diagnosis: WHY a strategy failed and EXACTLY what to change, in order.

The Champion Check saves its verdict reasons as long technical sentences. ``app.scoring.verdict_explain`` turns the ones it
knows into one-line bullets; this module goes further for the gates that most often block a strategy that otherwise looks
good (high pass chance, positive profit) and used to leave the user on a dead end ("Already optimized -- re-run the Full
Pipeline"):

* the pooled out-of-sample test is UNPROVEN (positive, but too few out-of-sample trades to separate it from zero),
* checks that could not be measured (parameter stability, supporting CPCV, deflated Sharpe on unseen trades),
* the expected-value test (NOT WORTH A FEE) and the specific blocker behind it,
* the ladder rung (REJECT / LEAD / ...) that follows from the two above.

Every cause carries evidence (the numbers that failed), plain-English steps, and ONE concrete action: an optimizer link
pre-set to the right goal, a Validate link pre-ticked with the right checks, or an outside-the-app step. The module also says
whether the optimizer can help at all and whether re-running the Full Pipeline would change anything -- it only does when
an input (data, risk, rules, settings) changed, because the pipeline is deterministic for the same inputs.

Pure functions, no I/O, so they are cheap to call while rendering a page and trivially testable.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Optional
from urllib.parse import urlencode

Z95 = 1.96

# check names the Champion Check reports as "not available" -> (what it is, how to get it measured, validate check key or "")
_UNAVAILABLE: dict[str, tuple[str, str, str]] = {
    "parameter_stability": (
        "Parameter stability: whether nearby settings are about as good as the chosen ones.",
        "Run Sensitivity and Parameter Robustness in Validate; they sweep every parameter and record the result.",
        "sensitivity,robustness"),
    "cpcv_supporting": (
        "Supporting CPCV: a second out-of-sample opinion from many train/test splits.",
        "Run CPCV in Validate (or tick 'CPCV supporting' in the Full Pipeline's advanced options).",
        "cpcv"),
    "dsr_unseen": (
        "Deflated Sharpe on unseen trades: whether the out-of-sample Sharpe survives the number of variants tried.",
        "It needs enough out-of-sample trades to compute. It becomes available once the out-of-sample sample is large enough "
        "(see the out-of-sample cause above).",
        ""),
}

_RE_UNPROVEN = re.compile(
    r"UNPROVEN \(caps at MARGINAL\):\s*pooled OOS mean R\s*([+-]?\d+(?:\.\d+)?)\s*\(95% day-block bootstrap lower bound\s*([+-]?\d+(?:\.\d+)?)\)"
    r"\s*over\s*(\d+)\s*trades(?:,\s*(\d+)% of merged folds positive)?", re.I)
_RE_POOLED_FAIL = re.compile(r"pooled purged out-of-sample test FAILED:\s*pooled OOS mean R\s*([+-]?\d+(?:\.\d+)?)"
                             r"(?:\s*\(95% day-block bootstrap lower bound\s*([+-]?\d+(?:\.\d+)?)\))?\s*over\s*(\d+)\s*trades", re.I)
_RE_UNAVAIL = re.compile(r"(\d+)\s+of\s+(\d+)\s+checks weren't available\s*\(([^)]*)\)", re.I)
_RE_EV_NONPOS = re.compile(r"EV per attempt is \$(-?[\d,]+(?:\.\d+)?)\s*\(<= 0\)\s*at the ([\w -]+?) pass rate\s*([\d.]+)%\s*\(break-even\s*([\d.]+)%\)", re.I)
_RE_EV_MARGIN = re.compile(r"([\w -]+?) pass rate\s*([\d.]+)% is above break-even\s*([\d.]+)% but short of the required margin\s*\(([\d.]+)%\)", re.I)
_RE_RUIN = re.compile(r"bankroll ruin before first payout is\s*([\d.]+)%\s*\(cap\s*([\d.]+)%\)", re.I)
_RE_FEE = re.compile(r"q x V does not cover the activation fee(?:\s*\(\$([\d,]+(?:\.\d+)?)\))?", re.I)


def _f(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _q(path: str, **kw: str) -> str:
    kw = {k: v for k, v in kw.items() if v}
    return path + ("?" + urlencode(kw) if kw else "")


def trades_needed_for_lower_bound(mean_r: float, lower_bound: float, n: int) -> Optional[dict]:
    """How many out-of-sample trades the lower bound needs to clear zero, extrapolated from the interval already measured.

    The bootstrap interval half-width is mean - lb ~ 1.96 * sd / sqrt(n), so sd ~ (mean - lb) * sqrt(n) / 1.96 and the bound
    clears zero once 1.96 * sd / sqrt(N) < mean, i.e. N > (1.96 * sd / mean)^2 = n * ((mean - lb) / mean)^2. This assumes the average R
    and its spread stay the same as the sample grows -- an estimate, labelled as one wherever it is shown.
    """
    if n <= 0 or mean_r is None or lower_bound is None or mean_r <= 0 or lower_bound > mean_r:
        return None
    need = int(math.ceil(n * ((mean_r - lower_bound) / mean_r) ** 2))
    need = max(need, n)
    return {"have": n, "need": need, "more": max(0, need - n), "factor": round(need / n, 2) if n else None}


class _Ctx:
    def __init__(self, pick: str, optimized: Optional[dict], lookahead_clean: bool = True):
        self.pick = pick
        self.optimized = optimized or {}
        self.lookahead_clean = lookahead_clean

    def optimize(self, goal: str, why: str = "") -> dict:
        return {"kind": "optimize", "goal": goal, "label": {
            "robust": "Optimize for a reliable edge", "eval": "Optimize for the evaluation pass", "payout": "Optimize for payouts",
            "risk": "Right-size the risk", "exits": "Fix the exits", "timefilter": "Cut the worst times"}.get(goal, "Open Optimize"),
            "href": _q("/optimize-simple", strategy_pick=self.pick, goal=goal, why=why[:160])}

    def validate(self, checks: str) -> dict:
        n = len([c for c in checks.split(",") if c])
        return {"kind": "validate", "label": f"Run {n} check{'s' if n != 1 else ''} in Validate", "href": _q("/validate-simple", strategy_pick=self.pick, run=checks)}

    def pipeline(self, label: str = "Re-run Full Pipeline with the changes") -> dict:
        ftype, _, fname = self.pick.partition("::")
        return {"kind": "pipeline", "label": label,
                "href": _q("/full-pipeline", load_strategy_type=ftype, load_strategy_name=fname) if fname else "/full-pipeline"}

    @staticmethod
    def data() -> dict:
        return {"kind": "data", "label": "Open the Data Center", "href": "/data-center"}

    @staticmethod
    def recommender() -> dict:
        return {"kind": "settings", "label": "Compare prop firms", "href": "/prop-firm-recommender"}

    def already_tried(self, goal: str) -> bool:
        lo = self.optimized
        if not lo:
            return False
        g = str(lo.get("goal") or "")
        return bool(lo) and (not g or g == goal) and lo.get("improved") is False


def _cause(cid: str, severity: str, title: str, why: str, evidence: list[str], steps: list[str], action: Optional[dict],
           *, optimizer: str = "no", also: Optional[list[dict]] = None, rerun_changes_it: bool = False) -> dict:
    return {"id": cid, "severity": severity, "title": title, "why": why, "evidence": evidence, "steps": steps,
            "action": action, "also": also or [], "optimizer": optimizer, "rerun_changes_it": rerun_changes_it}


def _oos_unproven(m: re.Match, c: _Ctx) -> dict:
    mean, lb, n = float(m.group(1)), float(m.group(2)), int(m.group(3))
    folds = m.group(4)
    need = trades_needed_for_lower_bound(mean, lb, n)
    ev = [f"Average result per out-of-sample trade: {mean:+.3f}R (positive).",
          f"Lower end of the 95% interval: {lb:+.3f}R (needs to be above 0).",
          f"Out-of-sample trades measured: {n}." + (f" {folds}% of merged folds were positive." if folds else "")]
    steps = [f"This is a sample-size problem, not a broken strategy: the edge is positive but {n} trades cannot rule out luck."
             if n < 100 else "The edge is positive but the out-of-sample sample is still too small to rule out luck."]
    if need:
        ev.append(f"Estimate: about {need['need']} out-of-sample trades are needed ({need['more']} more, ~{need['factor']}x what you have), "
                  "assuming the average R and its spread stay the same.")
        steps.append(f"Get roughly {need['need']} out-of-sample trades. The quickest ways: load a longer history for this instrument, "
                     "or test a faster timeframe, or loosen an entry filter WITHOUT lowering the average R.")
    steps.append("Optionally run Optimize with 'Make the edge more reliable': it scores settings on this exact lower bound, "
                 "so it favours settings with a steadier edge. It cannot create more trades on its own.")
    steps.append("Re-run the Full Pipeline only AFTER the data or settings changed. The same data and settings give the same 51 trades "
                 "and the same verdict, so re-running unchanged cannot fix this.")
    act = c.data() if need and need["factor"] and need["factor"] > 1.4 else c.optimize("robust", "Out-of-sample edge is positive but unproven")
    also = [c.optimize("robust", "Out-of-sample edge is positive but unproven")] if act["kind"] == "data" else [c.data()]
    return _cause("oos_unproven", "cap", "The out-of-sample edge is positive but unproven",
                  "The strategy earns money on data it was not tuned on, but there are too few out-of-sample trades to be statistically sure "
                  "the average is above zero. The Champion Check caps this at MARGINAL; it is not a rejection.",
                  ev, steps, act, optimizer="partly", also=also, rerun_changes_it=False)


def _oos_fail(m: re.Match, c: _Ctx) -> dict:
    mean = float(m.group(1))
    lb = _f(m.group(2))
    n = int(m.group(3))
    return _cause("oos_fail", "blocker", "The edge did not carry to unseen data",
                  "The pooled out-of-sample test failed: the lower bound of the average R is not above zero, or fewer than half of the folds were positive.",
                  [f"Pooled out-of-sample mean: {mean:+.3f}R over {n} trades" + (f" (lower bound {lb:+.3f}R)." if lb is not None else ".")],
                  ["Cut the number of tuned parameters and widen their ranges so the settings are not fitted to the development window.",
                   "Run Optimize with 'Make the edge more reliable', then re-validate with Walk-Forward.",
                   "If it still fails, the entry idea has no edge outside the data it was built on: scrap it and run Discovery."],
                  c.optimize("robust", "Out-of-sample test failed"), optimizer="partly", also=[{"kind": "search", "label": "Find a new strategy", "href": "/search"}])


def _unavailable(m: re.Match, c: _Ctx) -> Optional[dict]:
    n_missing, n_total = int(m.group(1)), int(m.group(2))
    names = [x.strip() for x in m.group(3).split(",") if x.strip()]
    ev, steps, checks = [f"{n_missing} of {n_total} checks could not be measured: {', '.join(names)}."], [], []
    for nm in names:
        what, how, chk = _UNAVAILABLE.get(nm, (nm.replace("_", " ").capitalize() + ".", "Run the matching check in Validate.", ""))
        ev.append(what)
        steps.append(how)
        if chk:
            checks.extend(x for x in chk.split(",") if x not in checks)
    steps.append("The score is re-normalised over what WAS measured, so these are not counted against you, but READY needs the core checks on record.")
    act = c.validate(",".join(checks)) if checks else c.data()
    return _cause("checks_missing", "cap", "Some checks have not been measured yet", "A check that cannot run cannot pass. These are caps, not failures.",
                  ev, steps, act, optimizer="no", rerun_changes_it=False)


def _ev_causes(text: str, c: _Ctx, lookahead_text: bool) -> list[dict]:
    out: list[dict] = []
    m = _RE_EV_NONPOS.search(text)
    if m:
        ev_val, what, p, be = _f(m.group(1)), m.group(2).strip(), float(m.group(3)), float(m.group(4))
        out.append(_cause(
            "ev_negative", "blocker", "One evaluation attempt loses money on average",
            f"At the {what} pass rate of {p:.1f}%, the expected payouts do not cover the evaluation fee and resets (break-even needs {be:.1f}%). "
            "The pass chance can look high on the Test page and still lose money here: the Test page shows the pass chance of the development window, "
            "this test uses a deliberately conservative lower bound measured on data the strategy was not tuned on, then subtracts the fee.",
            [f"Expected value per attempt: {'-' if (ev_val or 0) < 0 else ''}${abs(ev_val or 0):,.2f} (needs to be above $0, with margin).", f"Conservative pass rate used: {p:.1f}% versus break-even {be:.1f}%."],
            ["Raise the conservative pass rate: run Optimize with 'Pass the evaluation and get paid', or lower the risk per trade so fewer accounts breach.",
             "Lower the cost side: compare prop firms for a cheaper evaluation, a bigger profit split or free resets.",
             "A pass rate measured on more out-of-sample trades also rises, because the lower bound tightens (see the out-of-sample cause)."],
            c.optimize("eval", "Expected value per attempt is not positive"), optimizer="partly", also=[c.recommender()]))
    m = _RE_EV_MARGIN.search(text)
    if m and not out:
        what, p, be, need = m.group(1).strip(), float(m.group(2)), float(m.group(3)), float(m.group(4))
        out.append(_cause(
            "ev_margin", "blocker", "Positive expected value, but not by enough margin",
            f"The {what} pass rate {p:.1f}% beats break-even ({be:.1f}%), but the app requires {need:.1f}% so a small estimation error cannot turn it negative.",
            [f"Pass rate used: {p:.1f}%. Break-even: {be:.1f}%. Required: {need:.1f}%."],
            ["Run Optimize for the evaluation pass goal to push the pass rate above the required margin.", "Or choose a prop firm with a lower break-even (cheaper fee, higher split)."],
            c.optimize("eval", "Expected value margin too thin"), optimizer="yes", also=[c.recommender()]))
    m = _RE_RUIN.search(text)
    if m:
        r, cap = float(m.group(1)), float(m.group(2))
        out.append(_cause("ev_bankroll_ruin", "blocker", "Too many accounts bust before the first payout",
                          f"{r:.1f}% of simulated attempts lose the whole bankroll before getting paid (cap {cap:.0f}%).",
                          [f"Bankroll ruin before first payout: {r:.1f}% (cap {cap:.0f}%)."],
                          ["Halve the risk per trade and re-run: risk size is the lever that moves this.", "Run Optimize with 'Just right-size my risk'."],
                          c.optimize("risk", "Bankroll ruin above cap"), optimizer="yes"))
    if re.search(r"bankroll ruin was not computed", text, re.I):
        out.append(_cause("ev_ruin_missing", "cap", "Bankroll ruin was not computed", "The expected-value test needs it and could not measure it.",
                          [], ["Re-run the Full Pipeline with Monte Carlo enabled and a prop-firm preset selected."], c.pipeline("Re-run with a prop preset"), optimizer="no"))
    if re.search(r"rules are not verified", text, re.I):
        out.append(_cause("ev_rules_unverified", "blocker", "The prop firm's rules are not verified in the app",
                          "READY and the expected-value test only trust rule sets that were checked against the firm's own page.",
                          ["The selected preset is not marked verified."],
                          ["Pick a verified preset in the prop-firm picker, or verify this preset against the firm's page (app/prop/presets.py) and re-run."],
                          c.recommender(), optimizer="no"))
    if lookahead_text or re.search(r"lookahead check failed or crashed", text, re.I):
        out.append(_cause("lookahead", "blocker", "The lookahead check failed or crashed",
                          "If the strategy can see bars that had not closed yet, its backtest is not real.", [],
                          ["Fix the signal so it only uses closed bars (no future bars, no shift(-1), no same-bar close as the entry price). Optimizing cannot fix this."],
                          c.pipeline("Re-run after fixing the signal"), optimizer="no"))
    m = _RE_FEE.search(text)
    if m:
        out.append(_cause("ev_fee", "blocker", "The fee can never be earned back",
                          "Even a 100% pass rate would not pay for the activation fee at this payout size.",
                          [f"Activation fee: ${m.group(1)}." if m.group(1) else "Activation fee exceeds the achievable payout."],
                          ["Choose a prop firm with a smaller activation fee or a larger profit split."], c.recommender(), optimizer="no"))
    if re.search(r"has not confirmed in writing", text, re.I):
        out.append(_cause("written_confirmation", "advisory", "Get the firm's written OK for automation (a to-do, not the failure)",
                          "The firm has not confirmed in writing that this kind of automation is allowed. This is a pre-purchase reminder; it does not change the score or cause the verdict.",
                          ["written_confirmation = False for this preset."],
                          ["Ask the firm's support IN WRITING whether your execution method is allowed, keep the reply, then set written_confirmation=True for the preset in app/prop/presets.py."],
                          None, optimizer="no"))
    return out


def diagnose(reasons: Optional[Iterable[Any]], *, pick: str = "", verdict: str = "", ladder_stage: str = "",
             stats: Optional[dict] = None, optimized: Optional[dict] = None, eval_pct: Optional[float] = None,
             validate_failed: Optional[Iterable[str]] = None) -> dict:
    """Return the structured diagnosis for one strategy. Always returns a dict (``causes`` may be empty)."""
    c = _Ctx(pick, optimized)
    items = [str(r) for r in (reasons or []) if r]
    text = " ".join(items)
    causes: list[dict] = []

    m = _RE_UNPROVEN.search(text)
    if m:
        causes.append(_oos_unproven(m, c))
    else:
        m = _RE_POOLED_FAIL.search(text)
        if m:
            causes.append(_oos_fail(m, c))
    m = _RE_UNAVAIL.search(text)
    if m:
        cu = _unavailable(m, c)
        if cu:
            causes.append(cu)
    causes.extend(_ev_causes(text, c, "LOOKAHEAD CHECK CRASHED" in text.upper() or "HARD SAFETY GATE FAILED: the lookahead" in text))

    # Gates the plain-language explainer already knows: carry them over with a concrete action.
    known = (("HARD SAFETY GATE FAILED: Monte Carlo risk of ruin", "ruin", "blocker", "Too many simulated accounts bust", "risk"),
             ("HOLDOUT", "holdout", "blocker", "It lost money on the sealed holdout", "robust"),
             ("CPCV", "cpcv_gate", "blocker", "It failed cross-validation", "robust"),
             ("PBO", "pbo_gate", "blocker", "High chance the winning settings were just lucky", "robust"),
             ("DSR", "dsr_gate", "blocker", "The edge is not convincing after counting every variant tried", "robust"),
             ("COST STRESS", "cost_stress", "blocker", "Too little profit survives doubled trading costs", "exits"),
             ("PASS-AND-PAID TOO LOW", "pass_and_paid", "blocker", "Accounts pass but do not reach a first payout", "payout"),
             ("30-DAY PAYOUT", "payout_30d", "blocker", "Too few fresh accounts get paid within 30 days", "payout"),
             ("SAMPLE FLOOR", "sample_floor", "blocker", "Too few trades to trust", "robust"),
             ("RANDOM-ENTRY NULL NOT BEATEN", "null_test", "blocker", "Random entries did about as well", ""))
    from app.scoring.verdict_explain import _fix_for, _plain_for  # local import: avoid a cycle at module load
    seen_ids = {x["id"] for x in causes}
    for raw in items:
        for prefix, cid, sev, title, goal in known:
            if raw.startswith(prefix) and cid not in seen_ids:
                seen_ids.add(cid)
                act = c.optimize(goal, title) if goal else {"kind": "search", "label": "Find a new strategy", "href": "/search"}
                causes.append(_cause(cid, sev, title, _plain_for(raw) or title, [raw[:220]], [_fix_for(raw) or "See the report for the failing number."],
                                     act, optimizer="partly" if goal else "no"))
                break
    for chk in (validate_failed or []):
        cid = f"validate_{chk}"
        if cid not in seen_ids and chk:
            seen_ids.add(cid)
            causes.append(_cause(cid, "blocker", f"The {chk.replace('_', ' ')} validation check failed",
                                 "A validation check on record failed.", [], ["Open the Validate report: the failing check lists its own bar and fix."],
                                 c.validate(chk), optimizer="partly"))

    order = {"blocker": 0, "cap": 1, "advisory": 2}
    causes.sort(key=lambda x: order.get(x["severity"], 3))
    first_actionable = next((x for x in causes if x.get("action") and x["severity"] != "advisory"), None)
    blockers = [x for x in causes if x["severity"] == "blocker"]
    opt_help = "no"
    if any(x["optimizer"] == "yes" for x in causes):
        opt_help = "yes"
    elif any(x["optimizer"] == "partly" for x in causes):
        opt_help = "partly"
    tried = bool(optimized) and optimized.get("improved") is False
    if tried and first_actionable and first_actionable["action"]["kind"] == "optimize":
        # Optimize already ran and found nothing: do not send the user back to the same button.
        alt = next((a for x in causes for a in x.get("also", []) if a["kind"] != "optimize"), None)
        if alt:
            first_actionable = dict(first_actionable, action=alt)
    summary = ""
    if causes:
        if blockers:
            summary = f"{len(blockers)} thing{'s' if len(blockers) != 1 else ''} blocking READY" + (
                f" and {len(causes) - len(blockers)} cap{'s' if len(causes) - len(blockers) != 1 else ''}." if len(causes) > len(blockers) else ".")
        else:
            summary = (f"Nothing is rejecting it; {len(causes)} items cap it below READY." if len(causes) != 1
                       else "Nothing is rejecting it; 1 item caps it below READY.")
    return {
        "has": bool(causes), "verdict": verdict, "ladder_stage": ladder_stage, "summary": summary, "causes": causes,
        "primary": (first_actionable or {}).get("action"),
        "primary_cause": (first_actionable or {}).get("title"),
        "optimizer": {
            "can_help": opt_help,
            "tried_without_gain": tried,
            "explain": {
                "yes": "Optimize targets this directly.",
                "partly": ("Optimize can nudge this, but only a change to the data, risk size or rules clears it fully."
                           + (" It already ran once without improving, so running it again unchanged will not help." if tried else "")),
                "no": "Optimize cannot fix this: it tunes settings, and this needs a different input (data, a check, a firm rule or the signal itself).",
            }[opt_help],
        },
        "rerun_note": ("Re-running the Full Pipeline on the same data and settings reproduces the same result. It only helps after you change "
                       "something listed above."),
    }
