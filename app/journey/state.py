"""
First Payout Mission state.

Seven stages, each with the specific requirements that must be met before moving on:

    1 Test a strategy          2 Optimize or validate a strategy   3 Promote the strategy to champion
    4 Complete forward testing 5 Purchase the evaluation           6 Pass the evaluation   7 Receive the first payout

Everything is derived from what the app already records (current strategy + its validation checklist, the forward-test and journal
trades, the Prop Desk accounts, the latest live Guardian report). A requirement the app cannot measure is reported as ``met: None``
("not tracked yet") and never as met. The dashboard's day / expense estimates come from YOUR saved simulations and the firm's published
price; with neither, they are shown as an illustrative example and labelled as one.
"""
from __future__ import annotations

from datetime import date

from app.prop import desk_store as ds
from app.prop import presets as P

STAGES = [
    ("test", "Test a strategy", "/full-pipeline"),
    ("refine", "Optimize or validate a strategy", "/validate-simple"),
    ("champion", "Promote strategy to champion", "/champion-simple"),
    ("forward", "Complete forward testing", "/forward-test"),
    ("purchase", "Purchase evaluation", "/buy-challenge"),
    ("pass", "Pass the evaluation", "/guardian"),
    ("payout", "Receive the first payout", "/accounts"),
]
CHAMPION_STAGES = ("champion_candidate", "forward_testing", "production_ready")

DEFAULT_SETTINGS = {"preset_key": "", "strategy": "", "forward_trades_required": 30, "forward_days_required": 14,
                    "backtest_max_drawdown": None}
ILLUSTRATIVE = {"days": 35, "expenses": 225, "text": "Example only: 35 days and $225. Pick a plan and run a simulation to replace this with your own estimate."}
MIN_ROBUSTNESS_RUN = 3
OPTIONAL_REQ = "A discount code you have checked was used"


def settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    s.update(ds.get_setting("mission", {}) or {})
    return s


def save_settings(d: dict) -> dict:
    cur = settings()
    if "preset_key" in d:
        key = str(d["preset_key"] or "")
        if key:
            P.get_preset(key)           # raises KeyError for an unknown plan
        cur["preset_key"] = key
    if "strategy" in d:
        cur["strategy"] = str(d["strategy"] or "")[:120]
    for k, lo, hi in (("forward_trades_required", 5, 1000), ("forward_days_required", 0, 365)):
        if k in d:
            v = int(float(d[k]))
            if not lo <= v <= hi:
                raise ValueError(f"{k} must be between {lo} and {hi}")
            cur[k] = v
    if "backtest_max_drawdown" in d:
        v = d["backtest_max_drawdown"]
        cur["backtest_max_drawdown"] = None if v in (None, "") else float(v)
        if cur["backtest_max_drawdown"] is not None and cur["backtest_max_drawdown"] <= 0:
            raise ValueError("backtest_max_drawdown must be a positive dollar amount")
    ds.set_setting("mission", cur)
    return cur


def _req(label: str, met, detail: str = "") -> dict:
    return {"label": label, "met": met, "detail": detail}


def _money(x) -> str:
    return f"${x:,.0f}"


# ------------------------------------------------------------------------------------------------ inputs
def _strategy_inputs(cfg: dict) -> dict:
    out = {"name": cfg.get("strategy") or "", "ran": 0, "passed": 0, "decided": 0, "failed": 0, "set": False}
    try:
        from app.reports import strategy_state as ss
        cur = ss.get_current_strategy()
        if cur:
            out.update(set=True, name=out["name"] or cur["strategy_name"], instrument=cur.get("instrument", ""))
            r = ss.robustness_score(cur["strategy_name"], cur.get("instrument", ""), library_type=cur.get("library_type", ""),
                                    library_filename=cur.get("library_filename", ""))
            out.update(ran=r["ran_count"], passed=r["passed_count"], decided=r["decided_count"], failed=r["decided_count"] - r["passed_count"])
    except Exception:  # noqa: BLE001
        pass
    return out


def _champion_inputs(st: dict) -> dict:
    """Where the current strategy sits on the Champion board: has it been tested at all, and has it been promoted to champion."""
    out = {"found": False, "tested": bool(st.get("ran")), "promoted": False, "stage_title": ""}
    try:
        from app.reports import strategy_state as ss
        from app.scoring import champion_board as cb
        cur = ss.get_current_strategy()
        if cur:
            row = cb.match_current_to_row(cur, cb.list_board())
            if row:
                promo = row.get("promotion") or {}
                out.update(found=True, tested=out["tested"] or bool(row.get("verdict")), stage_title=promo.get("stage_title", ""),
                           promoted=promo.get("stage") in CHAMPION_STAGES)
    except Exception:  # noqa: BLE001
        pass
    return out


def _forward_inputs() -> dict:
    out = {"trades": 0, "qualifying": 0, "first": None, "last": None, "net": 0.0, "max_dd": 0.0}
    try:
        from app.web import journal_store
        try:
            # a missing/broken capture module must not zero the whole forward record: the journal store is read regardless
            from app.journal import capture
            capture.sync_forward_test()
        except Exception:  # noqa: BLE001
            pass
        ts = [t for t in journal_store.load()["trades"] if t.get("source") in ("Forward Test", "Demo", "Broker")]
        ts.sort(key=lambda t: (t.get("exitTime") or t.get("date", ""), t.get("id", "")))
        out["trades"] = len(ts)
        out["qualifying"] = sum(1 for t in ts if not t.get("violations"))
        if ts:
            out["first"], out["last"] = ts[0]["date"], ts[-1]["date"]
        eq = peak = dd = 0.0
        for t in ts:
            eq += t.get("pnl") or 0.0
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        out["net"], out["max_dd"] = round(eq, 2), round(dd, 2)
    except Exception:  # noqa: BLE001
        pass
    return out


def _span_days(first, last) -> int:
    try:
        return (date.fromisoformat(last) - date.fromisoformat(first)).days + 1
    except (TypeError, ValueError):
        return 0


def _focus_account(data: dict, cfg: dict):
    accts = [a for a in data["accounts"] if a["stage"] != "closed" and a["status"] != "failed"]
    if cfg.get("preset_key"):
        m = [a for a in accts if a.get("preset_key") == cfg["preset_key"]]
        accts = m or accts
    # the furthest-along account leads the journey
    rank = lambda a: (a["stage"] == "funded", len(a.get("payouts", [])), a["status"] == "passed", a.get("purchased_on", ""))  # noqa: E731
    return max(accts, key=rank) if accts else None


# ------------------------------------------------------------------------------------------------ the stages
def compute(today: date | None = None) -> dict:
    today = today or date.today()
    cfg = settings()
    data = ds.load()
    st = _strategy_inputs(cfg)
    fw = _forward_inputs()
    acct = _focus_account(data, cfg)
    preset_key = (acct or {}).get("preset_key") or cfg.get("preset_key") or ""
    preset = None
    if preset_key:
        try:
            preset = P.get_preset(preset_key)
        except KeyError:
            preset = None
    fee = P.get_fee_info(preset_key) if preset_key else None
    firm = (acct or {}).get("firm") or (preset.firm if preset else "")
    try:
        from app.guardian.engine import load_latest
        live = load_latest()
    except Exception:  # noqa: BLE001
        live = None

    need_trades, need_days = int(cfg["forward_trades_required"]), int(cfg["forward_days_required"])
    span = _span_days(fw["first"], fw["last"])
    stages: dict[str, dict] = {}

    ch = _champion_inputs(st)
    stages["test"] = {"requirements": [
        _req("A strategy is selected", st["set"], st["name"] or "Open a strategy from the Library and set it as current."),
        _req("A full test has been run on it", ch["tested"] if st["set"] else None, "Run the Full Pipeline on the Test page." if not ch["tested"] else "A result is on file."),
    ]}
    stages["refine"] = {"requirements": [
        _req(f"At least {MIN_ROBUSTNESS_RUN} of 5 robustness checks run", st["ran"] >= MIN_ROBUSTNESS_RUN, f"{st['ran']} of 5 run (walk-forward, CPCV, sensitivity, WFGA, regime matrix). Run them on the Validate page, or tune the strategy on Optimize."),
        _req("No robustness check failed", (st["failed"] == 0) if st["ran"] else None, f"{st['failed']} failed of {st['decided']} decided." if st["ran"] else "Run the checks first."),
    ]}
    stages["champion"] = {"requirements": [
        _req("Strategy promoted to champion", ch["promoted"] if ch["found"] else (False if st["set"] else None),
             f"Currently: {ch['stage_title']}. Promote it on the Champion page." if ch["found"] and not ch["promoted"] else ("Promoted." if ch["promoted"] else "Save the strategy to the library, then promote it on the Champion page.")),
    ]}
    stages["forward"] = {"requirements": [
        _req(f"{need_trades} qualifying forward trades", fw["qualifying"] >= need_trades, f"{fw['qualifying']} of {need_trades} (trades without a rule violation)."),
        _req(f"Forward tested for {need_days}+ days", span >= need_days, f"{span} day(s) so far."),
        _req("Forward results are not losing", fw["net"] >= 0 if fw["trades"] else None, f"Net {_money(fw['net'])} over {fw['trades']} trades." if fw["trades"] else "No forward trades yet."),
    ]}
    offer_active = False
    try:
        from app.buy_challenge import catalog
        offer_active = bool(firm and catalog._offer_state(catalog.load_offers().get(firm))["active"])
    except Exception:  # noqa: BLE001
        pass
    stages["purchase"] = {"requirements": [
        _req("A plan is chosen", bool(preset), preset.label if preset else "Compare firms on Buy Challenge and pick a plan."),
        _req("The firm allows algorithmic trading", preset.algo_trading_allowed if preset else None, (preset.algo_policy_note[:160] + "...") if preset and preset.algo_policy_note else "Choose a plan to check its policy."),
        _req("Evaluation purchased at the firm's checkout", bool(acct), f"{acct['label']} bought {acct['purchased_on']}." if acct else "Buy at the firm's official checkout, then press 'I bought this'."),
        _req(OPTIONAL_REQ, offer_active if firm else None, "A dated code is on file for this firm." if offer_active else "No active code on file; add one on Buy Challenge if you have it."),
        _req("Fee logged in My Attempts", bool(acct) and acct["fee"] > 0 if acct else False, _money(acct["fee"]) if acct and acct["fee"] else ""),
    ]}
    reqs = []
    if preset:
        tgt = preset.evaluation_profit_target_pct / 100.0 * preset.account_size
        reqs.append(_req(f"Reach the {_money(tgt)} profit target", (acct["stage"] == "funded" or acct["status"] == "passed") if acct else False,
                         (f"{_money(live['metrics']['profit_target_remaining'])} to go (live)." if live and live["metrics"].get("profit_target_remaining") is not None else "Tracked live by the Guardian once deployment is running.")))
        reqs.append(_req(f"Trade at least {preset.min_trading_days} day(s)", (acct["stage"] == "funded" or acct["status"] == "passed") if acct and preset.min_trading_days else (None if not acct else True), ""))
        reqs.append(_req("Stay inside the daily loss and drawdown limits", (live["level"] in ("ok", "warn")) if live else None, live["headline"] if live else "No live account is reporting right now."))
    else:
        reqs.append(_req("Choose a plan first", False, ""))
    stages["pass"] = {"requirements": reqs}
    funded = bool(acct and acct["stage"] == "funded")
    n_pay = len((acct or {}).get("payouts", []))
    prow = [_req("Account funded (activation completed with the firm)", funded, "Mark the account funded on the Accounts page once the firm activates it.")]
    if preset:
        prow.append(_req(f"Minimum {preset.winning_days_for_payout} winning day(s)" if preset.winning_days_for_payout else "Firm's payout conditions met", None, "Not tracked live yet; confirm in the firm's payout dashboard."))
        if preset.consistency_rule_pct or preset.funded_consistency_rule_pct:
            prow.append(_req(f"Consistency rule ({preset.funded_consistency_rule_pct or preset.consistency_rule_pct:g}%)", None, "Checked by the Guardian on every payout window."))
    prow.append(_req("First payout received", n_pay >= 1, f"{n_pay} payout(s) logged." if n_pay else "Log the payout on the Accounts page when it lands."))
    stages["payout"] = {"requirements": prow}

    # a stage is done when nothing that can be measured is unmet and something is met (the discount code is a nice-to-have, never a gate)
    for key, label, href in STAGES:
        rs = [r for r in stages[key]["requirements"] if r["label"] != OPTIONAL_REQ]
        unmet = [r for r in rs if r["met"] is False]
        stages[key]["unmet"] = [r["label"] for r in unmet]
        stages[key]["done"] = not unmet and any(r["met"] is True for r in rs)
    cur = None
    out = []
    for key, label, href in STAGES:
        s = stages[key]
        if s["done"] and cur is None:
            state = "done"
        elif cur is None:
            cur, state = key, "current"
        else:
            state = "upcoming"
            if s["done"]:
                state = "done"
        out.append({"key": key, "label": label, "href": href, "state": state, "requirements": s["requirements"], "unmet": s["unmet"]})
    if cur is None:
        cur = "payout"
    done_n = sum(1 for s in out if s["state"] == "done")
    messages = _messages(cur, out, fw, need_trades, st, preset, acct, live, cfg)
    return {"stages": out, "current": cur, "done": done_n, "total": len(out), "progress_pct": round(done_n / len(out) * 100),
            "next_actions": messages["next"], "alerts": messages["alerts"], "dashboard": _dashboard(data, preset, fee, acct, cfg),
            "settings": cfg, "firm": firm, "preset_key": preset_key, "plan": preset.label if preset else ""}


def _messages(cur, stages, fw, need_trades, st, preset, acct, live, cfg) -> dict:
    nxt, alerts = [], []
    if cur == "test":
        if not st["set"]:
            nxt.append("Pick a strategy and run the full test on it (Test page).")
        else:
            nxt.append(f"Run the full test on {st['name']} (Test page) to see where it stands.")
    elif cur == "refine":
        if st["ran"] < MIN_ROBUSTNESS_RUN:
            nxt.append(f"Run {MIN_ROBUSTNESS_RUN - st['ran']} more robustness check(s) on {st['name']} (Validate page), or tune it on Optimize first.")
        elif st["failed"]:
            nxt.append(f"{st['failed']} robustness check(s) failed. Optimize the strategy or replace it before promoting it.")
    elif cur == "champion":
        nxt.append("Promote the strategy on the Champion page once its checks are in.")
    elif cur == "forward":
        left = max(0, need_trades - fw["qualifying"])
        if left:
            nxt.append(f"You need {left} more qualifying trades.")
        span_left = [r for r in next(x for x in stages if x["key"] == "forward")["requirements"] if r["label"].startswith("Forward tested") and r["met"] is False]
        if span_left:
            nxt.append("Keep forward testing: the minimum number of days is not reached yet.")
        if fw["trades"] and fw["net"] < 0:
            nxt.append("Forward results are negative. Review the Journal's Behavior tab to see whether the strategy or the execution is the cause.")
    elif cur == "purchase":
        if not preset:
            nxt.append("Compare plans on Buy Challenge and pick one that allows algorithmic trading.")
        nxt.append("Buy the evaluation at the firm's official checkout (use a code you have checked), then press 'I bought this'.")
    elif cur == "pass":
        nxt.append("Deploy with the Guardian on and let the strategy work toward the profit target.")
    else:
        if not (acct and acct["stage"] == "funded"):
            nxt.append("Mark the evaluation passed on the Accounts page and complete activation with the firm.")
        nxt.append("Keep risk small and the cushion intact until the payout conditions are met, then log the payout.")
    bd = cfg.get("backtest_max_drawdown")
    if bd and fw["max_dd"] > bd:
        alerts.append({"level": "bad", "text": f"Your strategy has exceeded its historical drawdown: {_money(fw['max_dd'])} forward versus {_money(bd)} in the backtest."})
    if live:
        for c in live.get("checks", []):
            if c["key"] == "daily_loss" and c["level"] in ("warn", "reduce", "block", "flatten"):
                alerts.append({"level": "warn", "text": f"You are approaching the daily loss limit ({_money(c['remaining'])} left)."})
            if c["key"] == "profit_target" and c.get("remaining") == 0:
                alerts.append({"level": "ok", "text": "You have satisfied the evaluation target."})
            if c["key"] == "max_drawdown" and c["level"] in ("block", "flatten"):
                alerts.append({"level": "bad", "text": "The drawdown cushion is nearly gone."})
    if acct and acct["stage"] == "funded" and preset and not acct.get("payouts"):
        pass
    return {"next": nxt, "alerts": alerts}


def _dashboard(data: dict, preset, fee, acct, cfg) -> dict:
    """Days and money to a first payout. Real inputs only; otherwise a labelled example."""
    sims = [r for r in sorted(data["sim_runs"], key=lambda r: (r["created_at"], r["id"])) if (not cfg.get("preset_key") and not preset) or r.get("preset_key") == getattr(preset, "key", "")]
    sim = sims[-1]["summary"] if sims else None
    spent = round(sum(float(c["amount"]) for c in data["costs"] if not acct or c.get("account_id") == acct["id"]), 2) if acct else 0.0
    days = expenses = None
    basis = []
    if sim and sim.get("median_days_to_pass") is not None:
        days = float(sim["median_days_to_pass"]) + float(sim.get("median_days_to_payout") or 0)
        basis.append("your latest saved simulation")
    if fee and fee.challenge_price is not None:
        pr = float(sim["pass_rate"]) if sim and sim.get("pass_rate") else None
        exp = (fee.challenge_price / pr if pr else fee.challenge_price) + (fee.activation_fee or 0.0)
        expenses = round(exp, 2)
        basis.append("the firm's listed price" + (" divided by your simulated pass rate" if pr else " for a single attempt"))
    illustrative = days is None and expenses is None
    return {"estimated_days": days if days is not None else (ILLUSTRATIVE["days"] if illustrative else None),
            "estimated_expenses": expenses if expenses is not None else (ILLUSTRATIVE["expenses"] if illustrative else None),
            "spent_so_far": spent, "illustrative": illustrative, "basis": basis,
            "note": ILLUSTRATIVE["text"] if illustrative else "Estimates use " + " and ".join(basis) + ". All figures are illustrative: they are not a promise of results.",
            "partial": (days is None) != (expenses is None)}
