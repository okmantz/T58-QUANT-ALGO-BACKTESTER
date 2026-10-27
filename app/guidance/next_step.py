"""One place that decides "what should I do with THIS strategy next?".

Every page that shows a strategy (Dashboard cards, Full Pipeline result, Validate, Champion, Deploy) calls
``resolve()`` so the advice is identical everywhere and always comes from that strategy's own saved numbers --
never a generic "run validation". The result is a plain dict, safe to pass to a template or ``jsonify``:

    {
      "stage":    "untested" | "failed" | "weak" | "needs_validation" | "ready_to_promote" |
                  "forward_test" | "demo" | "live_ready",
      "tone":     "bad" | "warn" | "info" | "good",
      "headline": short imperative ("Optimize drawdown, then re-test"),
      "why":      one sentence quoting the actual scores,
      "steps":    ["1. ...", "2. ..."]   # at most 4, specific to the scores,
      "primary":  {"label": ..., "href": ...},
      "secondary": [{"label": ..., "href": ...}, ...],
    }

Inputs are deliberately loose: a Champion Board row (``champion_board.list_board()``), a run-history row, or both.
"""
from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlencode

# Thresholds the advice keys off. They mirror the Champion Board gates (eval/payout/oos) so the advice never
# contradicts the gate the strategy will actually be held to.
EVAL_OK, PAYOUT_OK, OOS_OK = 60.0, 50.0, 50.0
DD_WARN = 12.0         # % max drawdown in a backtest that usually breaks a prop-firm daily/trailing limit
MIN_TRADES = 100
SHARPE_OK = 0.8


def _f(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _pick(row: Optional[dict]) -> str:
    return f"{row['strategy_type']}::{row['filename']}" if row and row.get("filename") else ""


def _link(path: str, **q: str) -> str:
    q = {k: v for k, v in q.items() if v}
    return path + ("?" + urlencode(q) if q else "")


def _board_facts(row: dict) -> dict:
    promo = row.get("promotion") or {}
    reqs = promo.get("requirements") or []
    return {
        "stage": promo.get("stage"),
        "next_stage_title": promo.get("next_stage_title"),
        "can_promote": bool(promo.get("can_promote")),
        "unmet": [r for r in reqs if not r.get("met")],
        "eval": _f(row.get("eval_pct")),
        "payout": _f(row.get("payout_pct")),
        "oos": _f(row.get("oos_pct")),
        "verdict": row.get("verdict"),
        "score": _f(row.get("t58_score")),
        "val_ran": (row.get("validation") or {}).get("ran", 0),
        "val_total": (row.get("validation") or {}).get("total", 5),
        "val_methods": (row.get("validation") or {}).get("methods") or [],
    }


def resolve(board_row: Optional[dict] = None, run: Optional[dict] = None) -> dict:
    """Return the next-step plan for one strategy. ``board_row`` and/or ``run`` may be None."""
    pick = _pick(board_row)
    ftype = (board_row or {}).get("strategy_type", "")
    fname = (board_row or {}).get("filename", "")
    load_q = {"load_strategy_type": ftype, "load_strategy_name": fname}
    name = (board_row or {}).get("display_name") or (run or {}).get("strategy_name") or "This strategy"

    sharpe = _f((run or {}).get("sharpe_ratio"))
    dd = _f((run or {}).get("max_drawdown_pct"))
    trades = _f((run or {}).get("trades"))
    win = _f((run or {}).get("win_rate"))
    net = _f((run or {}).get("net_profit"))

    run_href = _link("/full-pipeline", **load_q) if fname else "/full-pipeline"
    def opt(goal: str, why: str = "") -> str:
        return _link("/optimize-simple", strategy_pick=pick, goal=goal, why=why)
    opt_href = opt("eval")
    val_href = _link("/validate-simple", strategy_pick=pick)
    secondary: list[dict] = []

    # ---- never tested -------------------------------------------------------------------------------------
    has_verdict = bool(board_row and board_row.get("verdict"))
    if board_row is None and run is None:
        return _out("untested", "info", "Create or load a strategy", "Nothing on record yet.",
                    ["1. Open Create and build or import a strategy.", "2. Run it through the Full Pipeline."],
                    "Open Create", "/start-here/create")
    if not has_verdict and (board_row is None or not (board_row.get("promotion") or {}).get("requirements")
                            or any(r["key"] == "backtested" and not r["met"]
                                   for r in board_row["promotion"]["requirements"])):
        why = "It has never been through the Full Pipeline, so there is no verdict to act on."
        steps = ["1. Run the Full Pipeline on your prop-firm and instrument.",
                 "2. Read the verdict, then come back here for the exact next move."]
        if sharpe is not None:
            why = f"Single backtest: Sharpe {sharpe:.2f}" + (f", max drawdown {dd:.1f}%" if dd is not None else "") + \
                  " -- but no verdict yet."
        return _out("untested", "info", "Run the Full Pipeline", why, steps, "Run Full Pipeline", run_href)

    f = _board_facts(board_row) if board_row else {}
    verdict = (f.get("verdict") or "").upper()
    ev, pay, oos = f.get("eval"), f.get("payout"), f.get("oos")

    # ---- hard fail ----------------------------------------------------------------------------------------
    if verdict == "NOT READY":
        problems = []
        if dd is not None and dd > DD_WARN:
            problems.append(f"drawdown {dd:.1f}%")
        if trades is not None and trades < MIN_TRADES:
            problems.append(f"only {int(trades)} trades")
        if ev is not None and ev < EVAL_OK:
            problems.append(f"eval pass {ev:.0f}%")
        if sharpe is not None and sharpe < SHARPE_OK:
            problems.append(f"Sharpe {sharpe:.2f}")
        why = "Failed the Champion Check" + (": " + ", ".join(problems) + "." if problems else " (a hard safety gate).")
        if (net is not None and net <= 0) or (sharpe is not None and sharpe < 0.3):
            return _out("failed", "bad", "Scrap it -- the edge isn't there",
                        why + " Tuning a strategy with no edge only curve-fits it.",
                        ["1. Send it to the Graveyard (keep the notes).",
                         "2. Use Discovery to find a replacement on the same instrument."],
                        "Find a replacement", "/search",
                        [{"label": "Auto-optimize anyway", "href": opt_href}])
        goal = "risk" if (dd is not None and dd > DD_WARN) else ("eval" if (ev is not None and ev < EVAL_OK) else "robust")
        opt_href = opt(goal, why)
        steps = []
        if dd is not None and dd > DD_WARN:
            steps.append("1. Optimize for drawdown: tighten the stop / ATR multiple and cap risk per trade.")
        if trades is not None and trades < MIN_TRADES:
            steps.append(f"{len(steps)+1}. Loosen entry filters or test a lower timeframe -- {int(trades)} trades is too few to trust.")
        if ev is not None and ev < EVAL_OK:
            steps.append(f"{len(steps)+1}. Re-run Optimize with the prop-firm rules on so it targets eval pass rate.")
        steps.append(f"{len(steps)+1}. Re-run the Full Pipeline. If it still fails, scrap it.")
        return _out("failed", "bad", "Rework once, then decide", why, steps[:4], "Open Optimize", opt_href,
                    [{"label": "Scrap & find new", "href": "/search"}])

    # ---- weak but not dead: marginal numbers ---------------------------------------------------------------
    weak = [(k, v, t) for k, v, t in (("Eval pass", ev, EVAL_OK), ("Payout", pay, PAYOUT_OK), ("OOS", oos, OOS_OK))
            if v is not None and v < t]
    unmet = f.get("unmet") or []

    if f.get("can_promote"):
        nxt = f.get("next_stage_title") or "the next stage"
        facts = ", ".join(f"{k} {v:.0f}%" for k, v in (("eval", ev), ("payout", pay), ("OOS", oos)) if v is not None)
        return _out("ready_to_promote", "good", f"Promote to {nxt}",
                    f"Every requirement is met ({facts})." if facts else "Every requirement is met.",
                    [f"1. Press Promote on the Champion page -> {nxt}.",
                     "2. Then forward-test it on live data before any real money."],
                    "Open Champion", _link("/champion-simple"))

    if weak and not [r for r in unmet if str(r.get("key", "")).startswith("validated_")]:
        k, v, t = min(weak, key=lambda x: x[1] - x[2])
        opt_href = opt({"Eval pass": "eval", "Payout": "payout", "OOS": "robust"}[k], f"{k} is {v:.0f}% (needs {t:.0f}%+)")
        fix = {"Eval pass": "Optimize with the prop-firm rules on: it targets the profit target and daily-loss limit directly.",
               "Payout": "Optimize for consistency -- smaller risk per trade and a tighter stop raise payout odds.",
               "OOS": "That's overfitting. Re-optimize with fewer parameters (or widen the ranges) and re-validate walk-forward."}[k]
        return _out("weak", "warn", f"Optimize {k.lower()}, then re-validate",
                    f"{k} is {v:.0f}% (needs {t:.0f}%+).",
                    ["1. " + fix, "2. Re-run the failed validation checks.", "3. Promote when all gates are green."],
                    "Open Optimize", opt_href, [{"label": "Validate again", "href": val_href}])

    # ---- missing validation checks --------------------------------------------------------------------------
    missing = [m for m in f.get("val_methods", []) if not m.get("ran")]
    if missing:
        keymap = {"cpcv": "cpcv", "wfo": "walk_forward", "wfga": "wfga", "sensitivity": "sensitivity", "regime_matrix": "regime"}
        checks = ",".join(keymap[m["key"]] for m in missing if m["key"] in keymap)
        names = ", ".join(m["label"] for m in missing[:3]) + ("..." if len(missing) > 3 else "")
        return _out("needs_validation", "info", f"Run {len(missing)} missing check{'s' if len(missing) != 1 else ''}",
                    f"{f['val_ran']}/{f['val_total']} validation checks on record. Missing: {names}.",
                    ["1. Open Validate -- the missing checks are pre-ticked.", "2. Run them (one click).",
                     "3. Come back: this card will tell you whether to promote or optimize."],
                    f"Run {len(missing)} check{'s' if len(missing) != 1 else ''}",
                    _link("/validate-simple", strategy_pick=pick, run=checks))

    # ---- past validation: forward / demo -------------------------------------------------------------------
    stage = f.get("stage")
    if stage in ("champion_candidate",):
        return _out("forward_test", "info", "Forward-test it for 2+ weeks",
                    "Validated on history; now it has to survive data it has never seen.",
                    ["1. Start a forward test on the same instrument.", "2. Don't touch the parameters while it runs.",
                     "3. Compare forward vs backtest drawdown at the end."],
                    "Open Forward Test", "/forward-test", [{"label": "Optimize first", "href": opt("robust")}])
    if stage == "forward_testing":
        return _out("demo", "info", "Demo-trade it on a prop-firm sim",
                    "Forward test is running/complete -- the last gate is real execution (slippage, fills, rules).",
                    ["1. Run it on a demo account with the exact prop-firm rules.",
                     "2. If it behaves like the backtest, promote to Production Ready.",
                     "3. If it drifts, return to Optimize."],
                    "Open Deploy", "/deploy-live", [{"label": "Back to Optimize", "href": opt_href}])
    if stage == "production_ready":
        return _out("live_ready", "good", "Go live small, monitor daily",
                    "Passed every gate on the board.",
                    ["1. Start at the smallest size the firm allows.", "2. Watch Live Market for drift vs forward test."],
                    "Open Deploy", "/deploy-live", [{"label": "Monitor", "href": "/live-market"}])

    return _out("needs_validation", "info", "Open Validate", "Validation not complete.",
                ["1. Pick your checks and run them."], "Open Validate", val_href)


def _out(stage, tone, headline, why, steps, label, href, secondary=None) -> dict:
    return {"stage": stage, "tone": tone, "headline": headline, "why": why, "steps": steps,
            "primary": {"label": label, "href": href}, "secondary": secondary or []}


def match_run_to_row(run: dict, rows: list[dict]) -> Optional[dict]:
    """Best-effort: the Champion Board row a run-history row belongs to (by display/metadata name)."""
    n = str(run.get("strategy_name") or "").strip().lower()
    if not n:
        return None
    for r in rows:
        if n in (str(r.get("display_name", "")).lower(), str(r.get("metadata_name", "")).lower()):
            return r
    return None
