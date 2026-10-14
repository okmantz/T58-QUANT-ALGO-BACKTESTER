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

# What each check needs to PASS -- shown in the downloadable report for EVERY check (pass or not),
# so a reader can always see the bar a result was judged against.
CHECK_BARS: dict[str, str] = {
    "monte_carlo": "Eval pass chance of 70% or more (50-70% is borderline, under 50% fails).",
    "cpcv": "Out-of-sample results must keep most of the in-sample edge on every train/test split.",
    "pbo": "Probability of overfitting under 35% (35-50% is borderline, 50% or more fails).",
    "walk_forward": "Keep at least 50% of the in-sample edge out-of-sample (25-50% is borderline, under 25% fails).",
    "wfga": "Out-of-sample fitness within 15% of in-sample (15-30% gap is borderline, over 30% fails).",
    "sensitivity": "No cliff edges (up to one in three parameters on a cliff is borderline, more fails).",
    "robustness": "Robustness score of 70/100 or more (50-70 is borderline, under 50 fails).",
    "regime": "No market regime that loses money (one losing regime is borderline, two or more fail).",
}

# Which kind of Auto-Fix can act on a flagged check (see app.validation.auto_fix). Monte Carlo is a
# performance problem rather than a robustness one, so it has no automatic fix -- it goes to Optimize.
FIX_KIND_BY_CHECK: dict[str, str] = {
    "regime": "regime",
    "cpcv": "overfit", "pbo": "overfit", "walk_forward": "overfit", "wfga": "overfit",
    "sensitivity": "overfit", "robustness": "overfit",
}


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


# ---------------------------------------------------------------------------
# Guidance: for every check that did not pass, say exactly WHY and exactly
# WHAT TO DO to make it pass, plus a button to the right page.
# ---------------------------------------------------------------------------

def _go(label: str, path: str, ctx: Optional[dict], extra: Optional[dict] = None) -> dict:
    return {"label": label, "href": _href(path, {**(ctx or {}), **(extra or {})})}


def _diagnose_error(check: str, error: str, ctx: Optional[dict]) -> dict:
    """Turn a raw exception message from a check that could not RUN into
    plain-language cause + fix steps + a place to go. Matching is by
    message content so every check (and any future one) gets useful help."""
    e = (error or "").lower()
    rerun = _go("Re-run this check", "/validate-simple", ctx, {"run": check})
    test_page = _go("Test the strategy on this data first", "/full-pipeline", ctx)
    data_page = _go("Open Data Center", "/data-center", ctx)
    optimize = _go("Open Optimize", "/optimize-simple", ctx)

    if "zero trades" in e or "produced zero" in e or "no trades" in e or "zero-trade" in e:
        return {
            "why": "The strategy never opened a single trade on this dataset and timeframe, so there is nothing to validate. "
                   "This is a strategy/data mismatch, not a bug in the check.",
            "steps": [
                "Run a plain backtest of this strategy on the same dataset and timeframe (the Test step). If that also shows 0 trades, the problem is the strategy or the data.",
                "If the strategy filters by hour or session (London/NY open, opening range, etc.), it needs INTRADAY bars. Daily bars, or data with a constant time of day, exclude every bar.",
                "Check the dataset has enough bars for the slowest indicator to warm up (a 200-period average needs at least a few hundred bars).",
                "Check the strategy was written for this instrument's price scale and symbol, and that the timeframe picked on the Validate page matches the one it was built on.",
                "Loosen the entry conditions, or switch to a dataset/timeframe the strategy was designed for, then run Validate again.",
            ],
            "go": test_page, "go2": data_page,
        }
    if "tunable" in e or "no parameters" in e or "no numeric" in e or "at least 2 candidates" in e:
        return {
            "why": "This check works by nudging the strategy's numeric settings up and down. This strategy has no numeric settings it can safely change, so the check has nothing to vary.",
            "steps": [
                "Open the strategy and expose its key numbers (periods, thresholds, stop/target multiples) as plain numeric inputs/constants near the top of the file.",
                "Save it back to your Strategy Library and run Validate again.",
                "If the strategy is intentionally parameter-free, you can skip this check -- the other checks still count.",
            ],
            "go": _go("Open Strategy Library", "/library", ctx), "go2": None,
        }
    if "not enough bars" in e or "too few bars" in e or "insufficient" in e or "too short" in e:
        return {
            "why": "The dataset is too short to split into the training/test windows this check needs.",
            "steps": [
                "Import a longer history (more months/years) for this instrument in the Data Center.",
                "Or lower the number of folds/groups in the Validate page's advanced settings so each window is bigger.",
                "Run Validate again.",
            ],
            "go": data_page, "go2": None,
        }
    if "nonetype" in e or "mkdir" in e or "permission" in e or "no such file" in e or "winerror" in e or "read-only" in e:
        return {
            "why": "The app could not create or read a temporary working file this check needs. This is an environment problem on this machine, not a problem with your strategy.",
            "steps": [
                "Run the check again -- the app now creates its own scratch folder automatically.",
                "If it fails again, make sure your user account can write to its temp folder and the app's data folder, and that antivirus is not quarantining the app's temp files.",
                "Close other copies of the app that may be holding the same files, then retry.",
            ],
            "go": rerun, "go2": None,
        }
    if "memory" in e or "killed" in e:
        return {
            "why": "The machine ran out of memory while running this check.",
            "steps": ["Close other heavy programs.", "Use a shorter date range or a coarser timeframe.", "Lower the number of simulations/paths in advanced settings and re-run."],
            "go": rerun, "go2": None,
        }
    if "timeout" in e or "timed out" in e:
        return {
            "why": "This check took longer than allowed.",
            "steps": ["Use a shorter date range or a coarser timeframe.", "Lower folds/paths/generations in advanced settings.", "Re-run it on its own rather than with every other check."],
            "go": rerun, "go2": None,
        }
    return {
        "why": "The check stopped on an unexpected error. The exact message is shown above so it can be diagnosed.",
        "steps": [
            "Re-run just this check on its own.",
            "If it fails with the same message, run a plain backtest of this strategy on the same data (the Test step) to confirm the strategy itself runs.",
            "Still stuck? Copy the error message above into the AI Assistant and ask it to fix the strategy or settings.",
        ],
        "go": rerun, "go2": _go("Ask the AI Assistant", "/assistant/", ctx),
    }


# What a check needs to PASS, and what to do when it doesn't (fail/caution).
def _fix_guidance(check: str, status: str, ctx: Optional[dict]) -> Optional[dict]:
    optimize = _go("Go to Optimize", "/optimize-simple", ctx)
    if check == "monte_carlo":
        return {"bar": "Eval pass chance of 70% or more.",
                "why": "The strategy's trades, shuffled thousands of ways, do not clear the prop-firm rules often enough.",
                "steps": ["Optimize for eval pass probability (not just net profit).",
                          "Reduce position size or risk per trade so the daily/max loss limits are hit less often.",
                          "Pick a prop-firm preset whose drawdown rules suit this strategy's style."],
                "go": optimize}
    if check == "cpcv":
        return {"bar": "Out-of-sample results must keep most of the in-sample edge on every split.",
                "why": "The strategy was tuned to history it has already seen, so it fades on unseen data.",
                "steps": ["Remove parameters until only the few that matter remain.",
                          "Re-optimize on a longer history so the fit has more to generalize from.",
                          "Prefer parameter values in the middle of a flat region over the single best spike."],
                "go": optimize}
    if check == "pbo":
        return {"bar": "Probability of overfitting under 35% (50% or more is a fail).",
                "why": "Choosing the best in-sample variant is not reliably choosing the best out-of-sample one.",
                "steps": ["Test fewer, more distinct ideas instead of many near-identical variants.",
                          "Use a longer history.", "Lock in simpler logic with fewer free parameters."],
                "go": optimize}
    if check == "walk_forward":
        return {"bar": "Keep at least 50% of the in-sample edge out-of-sample (25% is the floor).",
                "why": "Re-optimizing on each training window and trading the next unseen window loses most of the edge.",
                "steps": ["Use fewer optimizable parameters or narrower ranges.",
                          "Increase the number of trades per window (longer data or a faster timeframe).",
                          "Try an anchored window if rolling windows are too short."],
                "go": optimize}
    if check == "wfga":
        return {"bar": "Out-of-sample fitness within 15% of in-sample (30% is the floor).",
                "why": "The tuned parameters fit the training folds much better than the folds they were tested on.",
                "steps": ["Reduce the parameter ranges the GA can explore.", "Lower generations/population so it cannot memorize noise.",
                          "Make sure each test fold has enough trades to be meaningful."],
                "go": optimize}
    if check == "sensitivity":
        return {"bar": "No cliff edges (at most one in three parameters on a cliff for a caution).",
                "why": "A small nudge to some parameter changes the results sharply, so the edge depends on exact values.",
                "steps": ["Open Sensitivity to see which parameter is fragile.", "Re-optimize and choose the middle of the flat region, not the peak.",
                          "Widen or smooth that parameter's logic (e.g. use a threshold band instead of an exact level)."],
                "go": _go("Open Sensitivity", "/sensitivity", ctx)}
    if check == "robustness":
        return {"bar": "Robustness score of 70/100 or more.",
                "why": "The area around the chosen parameters is narrow -- the strategy only works at a pinpoint setting.",
                "steps": ["Re-optimize preferring broad plateaus.", "Drop parameters that do not add edge.", "Re-run this check after changing."],
                "go": _go("Open Parameter Robustness", "/parameter-robustness", ctx)}
    if check == "regime":
        return {"bar": "No market regime that loses money (one is a caution, more than one is a fail).",
                "why": "The edge only exists in part of the market; the other regimes cost you money.",
                "steps": ["Press Auto-fix: it finds the losing regimes on the first 70% of the data, adds a filter that stands aside in them, "
                          "and only keeps it if the last 30% (data it never looked at) holds up.",
                          "Or open the Regime Matrix to inspect the regimes yourself and test an exclusion by hand.",
                          "Re-validate the fixed copy to confirm it now passes."],
                "go": _go("Open Regime Matrix", "/regime-matrix", ctx)}
    return None


def _attach_guidance(item: dict, result: dict, ctx: Optional[dict]) -> dict:
    status = item.get("status")
    if status == "error":
        g = _diagnose_error(item["check"], str(result.get("error") or ""), ctx)
        g["bar"] = ""
    elif status in ("fail", "caution"):
        g = _fix_guidance(item["check"], status, ctx)
        if g is None:
            return item
        g.setdefault("go2", None)
    else:
        return item
    kind = FIX_KIND_BY_CHECK.get(item["check"])
    if kind and status in ("fail", "caution"):
        g["fix_kind"] = kind
    item["guidance"] = g
    return item


_INTERPRETERS = {
    "monte_carlo": _monte_carlo, "cpcv": _cpcv, "pbo": _pbo, "walk_forward": _walk_forward,
    "wfga": _wfga, "sensitivity": _sensitivity, "robustness": _robustness, "regime": _regime,
}


def interpret_check(result: dict, ctx: Optional[dict] = None) -> dict:
    """One check result ({check, ok, summary, error}) -> display item.
    Items that did not pass carry a ``guidance`` dict (why / steps / go)."""
    check = result.get("check")
    if not result.get("ok"):
        raw_err = str(result.get("error") or "Unknown error.").strip()
        # Long multi-paragraph messages (e.g. the zero-trade preflight) are
        # summarised by the guidance block below; keep just the first
        # paragraph here so the card stays readable.
        short = raw_err.split("\n\n", 1)[0].strip()
        item = _item(check, "error", "This check could not run.", short)
        return _attach_guidance(item, result, ctx)
    fn = _INTERPRETERS.get(check)
    if fn is None:
        return _item(check, "info", "Ran.", "")
    try:
        item = fn(result.get("summary") or {})
    except Exception as exc:  # noqa: BLE001 -- never let a display bug hide a finished check
        return _item(check, "info", "Ran, but the result could not be summarised.", str(exc))
    return _attach_guidance(item, result, ctx)


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
        zero = [i for i in errored if str((i.get("guidance") or {}).get("why", "")).startswith("The strategy never opened")]
        if len(zero) == len(errored):
            return {
                "key": "fix_zero_trades", "tone": "bad",
                "title": "This strategy opens no trades on this data -- fix that first",
                "why": f"{names} could not run because the strategy never fired on this dataset/timeframe. "
                       "Run a plain test on the same data to confirm, then check the strategy's hour/session filters, "
                       "the bar timeframe, and indicator warm-up (steps are listed on the card).",
                "label": "Test it on this data", "href": _href("/full-pipeline", ctx),
            }
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
            "key": "reduce_overfit", "tone": "bad", "fix_kind": "overfit",
            "title": "This looks over-fit -- simplify it, don't polish it",
            "why": f"{names} failed: the edge shrinks on data the strategy was not tuned on. Re-optimise with fewer free parameters or a larger out-of-sample window, or retire it.",
            "label": "Go to Optimize", "href": _href("/optimize-simple", ctx),
        }

    # 4. Fragile parameters.
    fragile = [c for c in ("sensitivity", "robustness") if st(c) == "fail"]
    if fragile:
        names = ", ".join(by[c]["label"] for c in fragile)
        return {
            "key": "stabilise_params", "tone": "bad", "fix_kind": "overfit",
            "title": "Move the parameters onto a plateau",
            "why": f"{names} found cliff edges. Re-optimise and prefer values in the middle of a flat region over the single best spike.",
            "label": "Go to Optimize", "href": _href("/optimize-simple", ctx),
        }

    # 5. A regime is bleeding.
    if st("regime") == "fail":
        return {
            "key": "filter_regimes", "tone": "bad", "fix_kind": "regime",
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
    items = [interpret_check(r, ctx) for r in results]
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


# ---------------------------------------------------------------------------
# Plain-text rendering (desktop app / CLI / logs) -- same content the web
# report shows, so both apps tell the user the same thing.
# ---------------------------------------------------------------------------

_STATUS_TEXT = {"pass": "PASS", "caution": "CAUTION", "fail": "FAIL", "error": "DID NOT RUN", "info": "INFO"}


def report_to_text(report: dict, baseline: Optional[dict] = None) -> str:
    """Render a ``build_report`` dict as readable plain text, including the
    "why / what to do / where to go" guidance for every check that did not
    pass. Mirrors the web report card for card."""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append(f"VALIDATION REPORT -- {report.get('overall_label', '')}")
    lines.append(report.get("summary_sentence", ""))
    lines.append("=" * 64)
    if baseline:
        lines.append(
            f"Baseline backtest: {baseline.get('trades', 0)} trades, net {baseline.get('net_profit', 0):.2f}, "
            f"win rate {baseline.get('win_rate', 0):.1f}%, max drawdown {baseline.get('max_drawdown_pct', 0):.1f}%."
        )
        lines.append("")
    for it in report.get("items", []):
        lines.append(f"[{_STATUS_TEXT.get(it['status'], it['status'].upper())}] {it['label']} -- {it['headline']}")
        if it.get("detail"):
            lines.append(f"    {it['detail']}")
        for m in it.get("metrics", []):
            lines.append(f"    - {m['label']}: {m['value']}")
        g = it.get("guidance")
        if g:
            lines.append("    WHY: " + g.get("why", ""))
            if g.get("bar"):
                lines.append("    TO PASS: " + g["bar"])
            for n, step in enumerate(g.get("steps", []), 1):
                lines.append(f"    {n}. {step}")
            go = g.get("go")
            if go:
                lines.append(f"    GO TO: {go['label']}  ({go['href'].split('?')[0]})")
        lines.append("")
    n = report.get("next_step") or {}
    if n:
        lines.append("YOUR NEXT STEP: " + n.get("title", ""))
        lines.append("    " + n.get("why", ""))
        if n.get("href"):
            lines.append(f"    -> {n.get('label', 'Continue')}  ({n['href'].split('?')[0]})")
    return "\n".join(lines)
