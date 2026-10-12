"""Plain-language interpretation of the Validate page's checks.

The Validate page (``/validate-simple``) runs up to eight checks against one
hardened baseline. Each check returns a raw ``summary`` dict (see
``app.web.server._dispatch_validation_checks``). This module turns those raw
numbers into what a trader actually needs once the run finishes:

  * a per-check status (PASS / CAUTION / FAIL / ERROR) with a one-line
    headline and the few numbers behind it,
  * an overall verdict for the run, and
  * ONE concrete next step chosen from the actual results -- not a generic
    "go to Champion" button.

It is deliberately free of Flask / pandas imports so it can be unit-tested in
isolation and reused by the Champion page.

Thresholds are intentionally simple and stated in each headline, so a person
can always see *why* a check came back the way it did.
"""
from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlencode

# check key (as posted by the Validate form) -> human label
CHECK_LABELS: dict[str, str] = {
    "monte_carlo": "Monte Carlo",
    "cpcv": "CPCV",
    "pbo": "PBO (overfitting)",
    "walk_forward": "Walk-Forward",
    "wfga": "Walk-Forward GA",
    "sensitivity": "Sensitivity (1-D)",
    "robustness": "Parameter Robustness",
    "regime": "Regime Matrix",
}

# Order checks are executed / displayed in (cheapest first, the long GA-based
# walk-forward checks last so a failure in the quick ones shows up early).
CHECK_ORDER: tuple[str, ...] = (
    "monte_carlo", "cpcv", "pbo", "sensitivity", "robustness", "regime", "walk_forward", "wfga",
)

# Rough relative cost of each check -- used only to weight the percent bar so
# it moves at a believable pace (a Walk-Forward GA is far slower than a Monte
# Carlo resample). Not a time estimate.
CHECK_WEIGHTS: dict[str, float] = {
    "monte_carlo": 1.0,
    "cpcv": 3.0,
    "pbo": 3.0,
    "sensitivity": 2.0,
    "robustness": 3.0,
    "regime": 1.0,
    "walk_forward": 4.0,
    "wfga": 8.0,
}
BASELINE_WEIGHT = 1.0

# Validate-page check key -> the canonical Strategy Library validation method
# the Champion gate counts (app.strategy.library.VALIDATION_STEP_METHODS), or
# a non-gating key that is still saved on the strategy for the record.
CHECK_TO_METHOD: dict[str, str] = {
    "monte_carlo": "monte_carlo",
    "cpcv": "cpcv",
    "pbo": "pbo",
    "walk_forward": "wfo",
    "wfga": "wfga",
    "sensitivity": "sensitivity",
    "robustness": "parameter_robustness",
    "regime": "regime_matrix",
}

# What the Champion gate needs before a strategy can be promoted to
# "Validated" (mirrors app.strategy.library.VALIDATION_STEP_METHODS).
GATING_CHECKS: tuple[str, ...] = ("cpcv", "walk_forward", "wfga", "sensitivity", "regime")

STATUS_ORDER = {"pass": 0, "info": 0, "caution": 1, "fail": 2, "error": 3}


def _f(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def _pct(v: Any, digits: int = 1) -> str:
    x = _f(v)
    return "--" if x is None else f"{x:.{digits}f}%"


def _num(v: Any, digits: int = 2) -> str:
    x = _f(v)
    return "--" if x is None else f"{x:.{digits}f}"


def _item(check: str, status: str, headline: str, detail: str = "",
          metrics: Optional[list[tuple[str, str]]] = None) -> dict[str, Any]:
    return {
        "check": check,
        "label": CHECK_LABELS.get(check, check),
        "method": CHECK_TO_METHOD.get(check),
        "status": status,
        "headline": headline,
        "detail": detail,
        "metrics": [{"label": k, "value": v} for k, v in (metrics or [])],
    }


# ---------------------------------------------------------------------------
# Per-check interpretation
# ---------------------------------------------------------------------------

def _monte_carlo(s: dict) -> dict:
    ev = _f(s.get("evaluation_pass_probability"))
    po = _f(s.get("first_payout_probability"))
    ruin = _f(s.get("risk_of_ruin_pct"))
    metrics = [("Eval pass chance", _pct(ev)), ("First payout chance", _pct(po)),
               ("Risk of ruin", _pct(ruin)), ("Simulations", str(s.get("n_simulations", "--")))]
    if ev is None:
        return _item("monte_carlo", "error", "Monte Carlo returned no pass probability.", "", metrics)
    if ev >= 70:
        return _item("monte_carlo", "pass", f"{ev:.0f}% chance to pass the evaluation.",
                     "Resampled trades clear the prop-firm rules comfortably (bar: 70%).", metrics)
    if ev >= 50:
        return _item("monte_carlo", "caution", f"{ev:.0f}% chance to pass the evaluation.",
                     "Better than a coin flip but under the 70% READY bar -- an edge exists, the odds are not strong enough yet.", metrics)
    return _item("monte_carlo", "fail", f"Only a {ev:.0f}% chance to pass the evaluation.",
                 "Under 50% against these rules. This is a performance problem, not a robustness one.", metrics)


def _cpcv(s: dict) -> dict:
    oos, ins = _f(s.get("mean_oos_metric")), _f(s.get("mean_is_metric"))
    robust = bool(s.get("is_robust"))
    metric = str(s.get("metric") or "metric")
    metrics = [(f"In-sample {metric}", _num(ins)), (f"Out-of-sample {metric}", _num(oos)),
               ("Paths tested", str(s.get("n_paths", "--")))]
    if robust:
        return _item("cpcv", "pass", "Held up on out-of-sample paths.",
                     "Out-of-sample results kept enough of the in-sample edge across every train/test split.", metrics)
    return _item("cpcv", "fail", "Edge did not survive out-of-sample.",
                 "Out-of-sample performance fell well below in-sample -- the classic sign of curve-fitting.", metrics)


def _pbo(s: dict) -> dict:
    p = _f(s.get("pbo"))
    metrics = [("Probability of overfitting", "--" if p is None else _pct(p * 100 if p <= 1 else p)),
               ("Candidates compared", str(s.get("n_candidates", "--")))]
    if p is None:
        return _item("pbo", "error", "PBO returned no value.", "", metrics)
    frac = p if p <= 1 else p / 100.0
    if frac < 0.35:
        return _item("pbo", "pass", f"{frac * 100:.0f}% chance the winner is an overfit pick.",
                     "Low: the best in-sample variant usually stayed above median out-of-sample.", metrics)
    if frac < 0.5:
        return _item("pbo", "caution", f"{frac * 100:.0f}% chance the winner is an overfit pick.",
                     "Borderline. Treat the exact parameter values as soft.", metrics)
    return _item("pbo", "fail", f"{frac * 100:.0f}% chance the winner is an overfit pick.",
                 "At or above 50%: picking the best in-sample variant is no better than picking at random.", metrics)


def _walk_forward(s: dict) -> dict:
    eff = _f(s.get("out_of_sample_efficiency"))
    folds = s.get("n_folds_completed")
    metrics = [("OOS efficiency", _num(eff)), ("Folds completed", str(folds if folds is not None else "--"))]
    for w in (s.get("warnings") or [])[:2]:
        metrics.append(("Warning", str(w)))
    if eff is None:
        return _item("walk_forward", "caution", "Walk-forward ran but produced no efficiency figure.",
                     "Usually too few trades in the test windows to measure.", metrics)
    if eff >= 0.5:
        return _item("walk_forward", "pass", f"Kept {eff * 100:.0f}% of its in-sample edge out-of-sample.",
                     "Re-optimised on each fold's training window only, then judged on unseen data (bar: 50%).", metrics)
    if eff >= 0.25:
        return _item("walk_forward", "caution", f"Kept only {eff * 100:.0f}% of its in-sample edge out-of-sample.",
                     "Some edge survives but a lot of the in-sample result was fit to noise.", metrics)
    return _item("walk_forward", "fail", f"Kept only {max(eff, 0) * 100:.0f}% of its in-sample edge out-of-sample.",
                 "The edge mostly disappears on data the optimiser never saw.", metrics)


def _wfga(s: dict) -> dict:
    ins, oos, gap = _f(s.get("in_sample_fitness")), _f(s.get("oos_fitness")), _f(s.get("overfitting_gap"))
    metrics = [("In-sample fitness", _num(ins)), ("Out-of-sample fitness", _num(oos)), ("Overfitting gap", _num(gap)),
               ("OOS trades", str(s.get("oos_trade_count", "--")))]
    if gap is None or ins is None:
        return _item("wfga", "caution", "Walk-Forward GA finished without a gap figure.",
                     "Usually means no usable out-of-sample trades.", metrics)
    scale = abs(ins) or 1.0
    ratio = gap / scale
    if ratio <= 0.15:
        return _item("wfga", "pass", "Out-of-sample fitness is close to in-sample.",
                     "A GA searching for the best out-of-sample fitness found no large gap between fit and reality.", metrics)
    if ratio <= 0.3:
        return _item("wfga", "caution", "Noticeable gap between in-sample and out-of-sample.",
                     "The tuned parameters look somewhat over-fit.", metrics)
    return _item("wfga", "fail", "Large gap between in-sample and out-of-sample.",
                 "More than 30% of the in-sample fitness vanished out-of-sample -- likely over-fit.", metrics)


def _sensitivity(s: dict) -> dict:
    n, cliffs = int(s.get("n_parameters") or 0), int(s.get("n_cliffs") or 0)
    metrics = [("Parameters swept", str(n)), ("Cliff edges found", str(cliffs))]
    named = [p.get("label") for p in (s.get("parameters") or []) if p.get("cliff_detected")]
    if named:
        metrics.append(("Fragile parameters", ", ".join(str(x) for x in named[:4])))
    if n == 0:
        return _item("sensitivity", "caution", "No tunable parameters were found to sweep.", "", metrics)
    if cliffs == 0:
        return _item("sensitivity", "pass", "No cliff edges: nudging any parameter degrades results gently.",
                     "A real edge sits on a plateau, not a spike.", metrics)
    if cliffs / n <= 0.34:
        return _item("sensitivity", "caution", f"{cliffs} of {n} parameters sit on a cliff edge.",
                     "A small change to those values changes results sharply.", metrics)
    return _item("sensitivity", "fail", f"{cliffs} of {n} parameters sit on a cliff edge.",
                 "Results depend on exact values -- a typical sign of curve-fitting.", metrics)


def _robustness(s: dict) -> dict:
    score = _f(s.get("parameter_robustness_score"))
    metrics = [("Robustness score", "--" if score is None else f"{score:.0f}/100"),
               ("Parameters checked", str(s.get("n_parameters_checked", "--"))),
               ("Cliffs detected", str(s.get("n_cliffs_detected", "--")))]
    if score is None:
        return _item("robustness", "error", "No robustness score returned.", "", metrics)
    if score >= 70:
        return _item("robustness", "pass", f"Robustness score {score:.0f}/100.", "Wide plateau around the chosen parameters.", metrics)
    if score >= 50:
        return _item("robustness", "caution", f"Robustness score {score:.0f}/100.", "Moderate plateau; some parameters are touchy.", metrics)
    return _item("robustness", "fail", f"Robustness score {score:.0f}/100.", "Narrow plateau -- results are fragile to small parameter changes.", metrics)


def _regime(s: dict) -> dict:
    cells, off = int(s.get("n_cells") or 0), int(s.get("n_disable_recommended") or 0)
    metrics = [("Regime cells analysed", str(cells)), ("Cells to switch off", str(off))]
    if off == 0:
        return _item("regime", "pass", "No market regime is hurting the strategy.", "", metrics)
    if off <= 1:
        return _item("regime", "caution", f"{off} regime loses money and should probably be switched off.",
                     "Use the Regime Matrix page to see which one and test the exclusion.", metrics)
    return _item("regime", "fail", f"{off} regimes lose money and should be switched off.",
                 "The edge only exists in part of the market. Filtering those regimes is the likely fix.", metrics)


_INTERPRETERS = {
    "monte_carlo": _monte_carlo, "cpcv": _cpcv, "pbo": _pbo, "walk_forward": _walk_forward,
    "wfga": _wfga, "sensitivity": _sensitivity, "robustness": _robustness, "regime": _regime,
}


def interpret_check(result: dict) -> dict:
    """One check result ({check, ok, summary, error}) -> display item."""
    check = result.get("check")
    if not result.get("ok"):
        return _item(check, "error", "This check could not run.", str(result.get("error") or "Unknown error."))
    fn = _INTERPRETERS.get(check)
    if fn is None:
        return _item(check, "info", "Ran.", "")
    try:
        return fn(result.get("summary") or {})
    except Exception as exc:  # noqa: BLE001 -- never let a display bug hide a finished check
        return _item(check, "info", "Ran, but the result could not be summarised.", str(exc))


# ---------------------------------------------------------------------------
# Next step
# ---------------------------------------------------------------------------

def _href(path: str, ctx: dict) -> str:
    q = {k: v for k, v in (ctx or {}).items() if v}
    return path + ("?" + urlencode(q) if q else "")


def _next_step(items: list[dict], ctx: dict, gating_missing: list[str]) -> dict:
    """Choose ONE next step from what actually happened."""
    by = {i["check"]: i for i in items}
    failed = [i for i in items if i["status"] == "fail"]
    errored = [i for i in items if i["status"] == "error"]

    def st(c: str) -> str:
        return by[c]["status"] if c in by else ""

    # 1. Something could not run -- nothing else is trustworthy until it does.
    if errored:
        names = ", ".join(i["label"] for i in errored)
        run = ",".join(i["check"] for i in errored)
        return {
            "key": "rerun_errors", "tone": "warn",
            "title": f"Re-run the check{'s' if len(errored) > 1 else ''} that failed to run",
            "why": f"{names} could not complete, so this run is not a full picture. The error is shown on its card.",
            "label": "Re-run those checks", "href": _href("/validate-simple", {**ctx, "run": run}),
        }

    # 2. The edge itself is too weak.
    if st("monte_carlo") == "fail":
        return {
            "key": "optimize_performance", "tone": "bad",
            "title": "Improve the strategy's odds before validating further",
            "why": "The Monte Carlo pass chance is under 50%, so robustness checks are not the bottleneck -- the edge is. Tune it, then validate again.",
            "label": "Go to Optimize", "href": _href("/optimize-simple", ctx),
        }

    # 3. Overfitting signals.
    overfit = [c for c in ("cpcv", "pbo", "walk_forward", "wfga") if st(c) == "fail"]
    if overfit:
        names = ", ".join(by[c]["label"] for c in overfit)
        return {
            "key": "reduce_overfit", "tone": "bad",
            "title": "This looks over-fit -- simplify it, don't polish it",
            "why": f"{names} failed: the edge shrinks on data the strategy was not tuned on. Re-optimise with fewer free parameters or a larger out-of-sample window, or retire it.",
            "label": "Go to Optimize", "href": _href("/optimize-simple", ctx),
        }

    # 4. Fragile parameters.
    fragile = [c for c in ("sensitivity", "robustness") if st(c) == "fail"]
    if fragile:
        names = ", ".join(by[c]["label"] for c in fragile)
        return {
            "key": "stabilise_params", "tone": "bad",
            "title": "Move the parameters onto a plateau",
            "why": f"{names} found cliff edges. Re-optimise and prefer values in the middle of a flat region over the single best spike.",
            "label": "Go to Optimize", "href": _href("/optimize-simple", ctx),
        }

    # 5. A regime is bleeding.
    if st("regime") == "fail":
        return {
            "key": "filter_regimes", "tone": "bad",
            "title": "Switch off the losing market regimes",
            "why": "More than one regime loses money. The Regime Matrix page shows which and lets you test the exclusion.",
            "label": "Open Regime Matrix", "href": _href("/regime-matrix", ctx),
        }

    # 6. No failures, but gating checks still missing -- finish the set.
    if gating_missing:
        labels = ", ".join(CHECK_LABELS[c] for c in gating_missing)
        return {
            "key": "finish_checks", "tone": "info",
            "title": f"Run the remaining required check{'s' if len(gating_missing) > 1 else ''}",
            "why": f"Nothing failed, but promotion needs all five core checks on record. Still missing: {labels}.",
            "label": "Run the missing checks", "href": _href("/validate-simple", {**ctx, "run": ",".join(gating_missing)}),
        }

    # 7. Cautions only.
    cautions = [i for i in items if i["status"] == "caution"]
    if cautions:
        names = ", ".join(i["label"] for i in cautions)
        return {
            "key": "promote_with_care", "tone": "warn",
            "title": "Promote it, but treat the amber checks as homework",
            "why": f"No check failed. {names} came back borderline -- fine for a Champion Candidate, worth re-checking before real money.",
            "label": "Go to Champion", "href": _href("/champion-simple", ctx),
        }

    # 8. Clean sweep.
    return {
        "key": "promote", "tone": "good",
        "title": "Everything passed -- promote it",
        "why": "All five core checks are on record and none raised a flag. The Champion page will move this strategy up a stage.",
        "label": "Promote on Champion page", "href": _href("/champion-simple", ctx),
    }


def build_report(results: list[dict], *, context: Optional[dict] = None,
                 methods_on_record: Optional[set] = None) -> dict:
    """Everything the finished-run report needs.

    results           -- the list _dispatch_validation_checks returned.
    context           -- lifecycle handoff params (candidate / strategy_pick,
                         existing_dataset, timeframe, prop_preset) so every
                         link lands on the next page already filled in.
    methods_on_record -- canonical validation method keys already saved for
                         this strategy (this run + earlier ones), used to
                         work out which of the five core checks are missing.
    """
    ctx = dict(context or {})
    items = [interpret_check(r) for r in results]
    items.sort(key=lambda i: CHECK_ORDER.index(i["check"]) if i["check"] in CHECK_ORDER else 99)

    on_record = set(methods_on_record or set())
    for i in items:
        if i["status"] != "error" and i.get("method"):
            on_record.add(i["method"])
    gating_missing = [c for c in GATING_CHECKS if CHECK_TO_METHOD[c] not in on_record]

    counts = {k: sum(1 for i in items if i["status"] == k) for k in ("pass", "caution", "fail", "error")}
    worst = max((STATUS_ORDER.get(i["status"], 0) for i in items), default=0)
    if worst >= 3:
        overall, label = "incomplete", "Incomplete -- a check failed to run"
    elif worst == 2:
        overall, label = "needs_work", "Needs work"
    elif worst == 1:
        overall, label = "mixed", "Mostly solid, with caveats"
    else:
        overall, label = "strong", "Held up"

    n = len(items)
    sentence = (f"{counts['pass']} of {n} passed" + (f", {counts['caution']} borderline" if counts["caution"] else "")
                + (f", {counts['fail']} failed" if counts["fail"] else "")
                + (f", {counts['error']} could not run" if counts["error"] else "") + ".")

    return {
        "items": items,
        "counts": counts,
        "overall": overall,
        "overall_label": label,
        "summary_sentence": sentence,
        "gating_missing": gating_missing,
        "gating_missing_labels": [CHECK_LABELS[c] for c in gating_missing],
        "next_step": _next_step(items, ctx, gating_missing),
    }
