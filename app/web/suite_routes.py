"""
Routes for the prop-firm journey suite: journal behavior / statistics / My Attempts, Evaluation Guardian, Buy Challenge,
First Payout Mission, Research Integrity, Strategy Family Map, Multi-Account Manager. One Blueprint (`suite_bp`), registered in
app/web/server.py.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.web import journal_store

suite_bp = Blueprint("suite", __name__)


def _err(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


def _strategy_filter(trades):
    s = request.args.get("strategy", "").strip()
    return [t for t in trades if t.get("strategy") == s] if s else trades


# ----------------------------------------------------------------- journal: sync / behavior / statistics / attempts
@suite_bp.route("/api/journal/sync", methods=["POST"])
def journal_sync():
    from app.journal import capture
    n = 0
    try:
        n += capture.sync_forward_test()
    except Exception as exc:  # noqa: BLE001
        return _err(f"Forward Test sync failed: {exc}", 500)
    return jsonify({"ok": True, "synced": n})


@suite_bp.route("/api/journal/behavior")
def journal_behavior():
    from app.journal import behavior
    trades = _strategy_filter(journal_store.load()["trades"])
    return jsonify({"ok": True, "trades": len(trades), "insights": behavior.insights(trades), "diagnosis": behavior.diagnose(trades),
                    "realized_vs_backtest": behavior.realized_vs_backtest(trades),
                    "unclassified_regime": sum(1 for t in trades if (t.get("regime") or "unclassified") == "unclassified")})


@suite_bp.route("/api/journal/statistics")
def journal_statistics():
    from app.journal import statistics
    by = "year" if request.args.get("by") == "year" else "month"
    return jsonify({"ok": True, **statistics.compare(_strategy_filter(journal_store.load()["trades"]), by)})


@suite_bp.route("/api/journal/attempts")
def journal_attempts():
    from app.prop import attempt_ledger
    return jsonify({"ok": True, **attempt_ledger.economics()})


# ----------------------------------------------------------------- Buy Challenge
@suite_bp.route("/buy-challenge")
def buy_challenge_page():
    from flask import render_template
    return render_template("buy_challenge.html", active_page="buy_challenge")


@suite_bp.route("/api/buy-challenge/catalog")
def buy_challenge_catalog():
    from app.buy_challenge import catalog
    return jsonify({"ok": True, **catalog.catalog()})


@suite_bp.route("/api/buy-challenge/offers/<path:firm>", methods=["PUT", "POST"])
def buy_challenge_offer_save(firm: str):
    from app.buy_challenge import catalog
    try:
        row = catalog.save_offer(firm, request.get_json(force=True, silent=True) or {})
    except ValueError as exc:
        return _err(str(exc))
    return jsonify({"ok": True, "offer": row})


@suite_bp.route("/api/buy-challenge/purchase", methods=["POST"])
def buy_challenge_purchase():
    from app.buy_challenge import catalog
    d = request.get_json(force=True, silent=True) or {}
    try:
        acct = catalog.record_purchase(str(d.get("preset_key", "")), fee=d.get("fee"), platform=str(d.get("platform", "")),
                                       strategy=str(d.get("strategy", "")))
    except (KeyError, ValueError) as exc:
        return _err(str(exc))
    return jsonify({"ok": True, "account": acct})


# ----------------------------------------------------------------- Payout optimizer (Deploy page)
@suite_bp.route("/api/payout-optimizer/stages")
def payout_optimizer_stages():
    from app.live_deploy.guardrails import futures_prop_safe_rules
    from app.prop import payout_optimizer as po
    try:
        size = float(request.args.get("account_size", 100000))
        rules = futures_prop_safe_rules(account_size=size, daily_loss_limit_pct=float(request.args.get("daily_loss", 5)) or None,
                                        max_drawdown_pct=float(request.args.get("max_dd", 10)))
        return jsonify({"ok": True, "preset": po.preset_name(request.args.get("preset")), "presets": po.PRESETS,
                        "stages": po.all_stage_plans(rules, request.args.get("preset"))})
    except (ValueError, TypeError) as exc:
        return _err(str(exc))


@suite_bp.route("/api/prop-presets/keys")
def prop_preset_keys():
    from app.prop import presets as P
    return jsonify({"ok": True, "presets": [{"key": p.key, "label": p.label} for p in P.list_presets()]})


# ----------------------------------------------------------------- Prop Desk: accounts, payouts, costs, calendar, guardian, mission
def _desk_call(fn, *a, **k):
    from app.prop import desk_store as ds  # noqa: F401
    try:
        return jsonify({"ok": True, **fn(*a, **k)})
    except KeyError:
        return _err("Not found.", 404)
    except (ValueError, TypeError) as exc:
        return _err(str(exc))


@suite_bp.route("/accounts")
def accounts_page():
    from flask import render_template
    return render_template("accounts.html", active_page="accounts")


@suite_bp.route("/api/desk")
def desk_overview():
    from app.monitoring import health_alerts
    from app.prop import attempt_ledger, desk_store as ds, payout_calendar
    data = ds.load()
    trades = journal_store.load()["trades"]
    from app.journey import state as jstate
    cfg = jstate.settings()
    alerts = health_alerts.from_journal(trades, backtest_max_dd=cfg.get("backtest_max_drawdown"))
    return jsonify({"ok": True, "accounts": data["accounts"], "economics": attempt_ledger.economics(data), "costs": attempt_ledger.cost_tracker(data),
                    "calendar": payout_calendar.events(data), "health_alerts": alerts, "health_level": health_alerts.worst(alerts),
                    "bust_causes": ds.BUST_CAUSES})


@suite_bp.route("/api/desk/accounts", methods=["POST"])
def desk_account_add():
    from app.prop import desk_store as ds
    return _desk_call(lambda: {"account": ds.add_account(request.get_json(force=True, silent=True) or {})})


@suite_bp.route("/api/desk/accounts/<account_id>", methods=["PUT", "DELETE"])
def desk_account_edit(account_id: str):
    from app.prop import desk_store as ds
    if request.method == "DELETE":
        return jsonify({"ok": ds.delete_account(account_id)})
    return _desk_call(lambda: {"account": ds.update_account(account_id, request.get_json(force=True, silent=True) or {})})


@suite_bp.route("/api/desk/accounts/<account_id>/reset", methods=["POST"])
def desk_account_reset(account_id: str):
    from app.prop import desk_store as ds
    d = request.get_json(force=True, silent=True) or {}
    return _desk_call(lambda: {"account": ds.reset_account(account_id, fee=float(d.get("fee") or 0), bust_cause=str(d.get("bust_cause") or ""))})


@suite_bp.route("/api/desk/accounts/<account_id>/payouts", methods=["POST"])
def desk_payout_add(account_id: str):
    from app.prop import desk_store as ds
    d = request.get_json(force=True, silent=True) or {}
    return _desk_call(lambda: {"account": ds.add_payout(account_id, str(d.get("date") or ""), float(d.get("amount") or 0))})


@suite_bp.route("/api/desk/costs", methods=["POST"])
def desk_cost_add():
    from app.prop import desk_store as ds
    return _desk_call(lambda: {"cost": ds.add_cost(request.get_json(force=True, silent=True) or {})})


@suite_bp.route("/api/desk/costs/<cost_id>", methods=["DELETE"])
def desk_cost_delete(cost_id: str):
    from app.prop import desk_store as ds
    return jsonify({"ok": ds.delete_cost(cost_id)})


@suite_bp.route("/api/desk/payout-calendar.ics")
def desk_calendar_ics():
    from flask import Response
    from app.prop import payout_calendar
    return Response(payout_calendar.ics(payout_calendar.events()), mimetype="text/calendar",
                    headers={"Content-Disposition": "attachment; filename=t58-payouts.ics"})


@suite_bp.route("/api/health-alerts")
def health_alerts_api():
    from app.journey import state as jstate
    from app.monitoring import health_alerts
    cfg = jstate.settings()
    trades = _strategy_filter(journal_store.load()["trades"])
    alerts = health_alerts.from_journal(trades, backtest_max_dd=cfg.get("backtest_max_drawdown"))
    return jsonify({"ok": True, "alerts": alerts, "level": health_alerts.worst(alerts)})


@suite_bp.route("/api/journey")
def journey_api():
    from app.journey import state as jstate
    return jsonify({"ok": True, **jstate.compute()})


@suite_bp.route("/api/mission/settings", methods=["PUT", "POST"])
def mission_settings():
    from app.journey import state as jstate
    try:
        return jsonify({"ok": True, "settings": jstate.save_settings(request.get_json(force=True, silent=True) or {})})
    except KeyError:
        return _err("Unknown plan.")
    except (ValueError, TypeError) as exc:
        return _err(str(exc))


@suite_bp.route("/mission")
def mission_page():
    from flask import render_template
    return render_template("mission.html", active_page="mission")


@suite_bp.route("/guardian")
def guardian_page():
    from flask import render_template
    return render_template("guardian.html", active_page="guardian")


@suite_bp.route("/api/guardian/evaluate", methods=["POST"])
def guardian_evaluate():
    """What-if / live check: rules (a firm preset or typed values) plus an account state -> the Guardian's report."""
    from datetime import datetime
    from app.guardian import engine
    d = request.get_json(force=True, silent=True) or {}
    try:
        stage = str(d.get("stage") or "evaluation")
        news = bool(d.get("news_restricted", False))
        if d.get("preset_key"):
            rules = engine.GuardianRules.from_preset(str(d["preset_key"]), stage=stage, news_restricted=news)
        else:
            f = lambda k: (float(d[k]) if d.get(k) not in (None, "") else None)  # noqa: E731
            rules = engine.GuardianRules(account_size=float(d.get("account_size") or 50000), daily_loss_limit=f("daily_loss_limit"), max_drawdown=f("max_drawdown"),
                                         drawdown_type=str(d.get("drawdown_type") or "trailing"), profit_target=f("profit_target"), consistency_pct=f("consistency_pct"),
                                         min_trading_days=int(d.get("min_trading_days") or 0), max_contracts=int(d["max_contracts"]) if d.get("max_contracts") else None,
                                         news_restricted=news, stage=stage)
        s = d.get("state") or {}
        n = lambda k, default=None: (float(s[k]) if s.get(k) not in (None, "") else default)  # noqa: E731
        bal = n("balance", rules.account_size)
        now = datetime.fromisoformat(str(d["now"])) if d.get("now") else None
        state = engine.AccountState(balance=bal, equity=n("equity"), start_of_day_balance=n("start_of_day_balance"), peak_equity=n("peak_equity"),
                                    open_contracts=n("open_contracts", 0.0), open_risk=n("open_risk", 0.0), trading_days=int(n("trading_days", 0) or 0),
                                    best_day_profit=n("best_day_profit", 0.0), now=now)
        rep = engine.evaluate(rules, state, preset=d.get("preset"), now=now)
    except (KeyError, ValueError, TypeError) as exc:
        return _err(f"Could not evaluate: {exc}")
    return jsonify({"ok": True, "report": rep.to_dict()})


@suite_bp.route("/api/guardian/latest")
def guardian_latest():
    from app.guardian import engine, news
    return jsonify({"ok": True, "report": engine.load_latest(), "news_coverage": news.coverage()})


# ----------------------------------------------------------------- Research Integrity
_BENCH_JOB = {"running": False, "error": ""}


@suite_bp.route("/research-integrity")
def research_integrity_page():
    from flask import render_template
    return render_template("research_integrity.html", active_page="research_integrity")


@suite_bp.route("/api/research-integrity")
def research_integrity_api():
    from app.validation import research_integrity as ri
    return jsonify({"ok": True, **ri.dashboard(), "benchmark_job": dict(_BENCH_JOB)})


@suite_bp.route("/api/research-integrity/benchmark", methods=["POST"])
def research_integrity_benchmark():
    import threading
    from app.validation import research_integrity as ri
    if _BENCH_JOB["running"]:
        return _err("The benchmark is already running.", 409)

    def work():
        try:
            ri.save_benchmark(ri.run_benchmark())
            _BENCH_JOB["error"] = ""
        except Exception as exc:  # noqa: BLE001
            _BENCH_JOB["error"] = str(exc)
        finally:
            _BENCH_JOB["running"] = False

    _BENCH_JOB["running"] = True
    threading.Thread(target=work, daemon=True).start()
    return jsonify({"ok": True, "started": True})


# ----------------------------------------------------------------- Strategy Family Map
@suite_bp.route("/family-map")
def family_map_page():
    from flask import render_template
    return render_template("family_map.html", active_page="graveyard")


@suite_bp.route("/api/family-map")
def family_map_api():
    from app.search import family_intelligence
    return jsonify({"ok": True, **family_intelligence.load()})
