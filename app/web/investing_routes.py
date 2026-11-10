"""Investing tab (under Hedge Fund Manager): search, SEC-powered research, manual + automated investing, paper by default.

Blueprint pattern identical to hedge_fund_routes (no import of server.py)."""
from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.investing import algos, backtest as bt, market_data as md, profile as prof, research
from app.investing.paper import PaperAccount
from app.investing.sec_edgar import SecClient, SecError, set_user_agent, user_agent

investing_bp = Blueprint("investing", __name__)
_sec = SecClient()


def _err(e, code=400):
    return jsonify({"ok": False, "error": str(e)}), code


def _broker(mode: str, confirm_live: bool = False):
    if mode == "paper":
        return PaperAccount()
    from app.investing.broker_alpaca import AlpacaBroker
    if mode == "live" and not confirm_live:
        raise ValueError("Real-money trading needs the confirmation box ticked.")
    return AlpacaBroker(live=(mode == "live"))


@investing_bp.route("/investing")
def investing_page():
    return render_template("investing.html", active_page="investing", questions=prof.QUESTIONS, algos=algos.ALGOS,
                           sec_ready=bool(user_agent()), indices=research.INDICES, etfs=research.ETFS)


@investing_bp.route("/investing/api/sec-agent", methods=["POST"])
def api_sec_agent():
    ua = (request.get_json(silent=True) or {}).get("user_agent", "").strip()
    if len(ua) < 6 or "@" not in ua:
        return _err("Enter a contact like 'Your Name you@example.com' (SEC requires it; it is sent only to sec.gov).")
    set_user_agent(ua)
    return jsonify({"ok": True})


@investing_bp.route("/investing/api/search")
def api_search():
    try:
        return jsonify({"ok": True, "results": research.search(request.args.get("q", ""), _sec)})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@investing_bp.route("/investing/api/asset/<symbol>")
def api_asset(symbol):
    try:
        return jsonify({"ok": True, **research.asset(symbol, _sec)})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@investing_bp.route("/investing/api/history/<symbol>")
def api_history(symbol):
    try:
        df = md.history(symbol, request.args.get("range", "2y"))
        return jsonify({"ok": True, "points": [[str(i.date()), round(float(v), 4)] for i, v in df["close"].iloc[:: max(1, len(df) // 300)].items()]})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@investing_bp.route("/investing/api/profile", methods=["POST"])
def api_profile():
    b = request.get_json(silent=True) or {}
    p = prof.build_profile(b.get("answers", {}), float(b.get("starting_cash") or 10000))
    return jsonify({"ok": True, "profile": p.to_dict(), "config": prof.algo_config(p)})


@investing_bp.route("/investing/api/discover", methods=["POST"])
def api_discover():
    """Undervaluation screen over the user's allowed universe (SEC fundamentals + price)."""
    b = request.get_json(silent=True) or {}
    syms = b.get("symbols") or research.universe(b.get("universe") or ["stocks"])
    syms = [s for s in syms if s not in research.CRYPTO and s not in research.ETFS and not s.startswith("^")]
    try:
        return jsonify({"ok": True, "rows": research.screen(syms, _sec, top=int(b.get("top") or 15))})
    except SecError as e:
        return _err(e)


@investing_bp.route("/investing/api/plan", methods=["POST"])
def api_plan():
    """Current recommended portfolio for the profile (what the automated mode would hold), with reasons."""
    b = request.get_json(silent=True) or {}
    p = prof.build_profile(b.get("answers", {}), float(b.get("starting_cash") or 10000))
    cfg = {**prof.algo_config(p), **{k: v for k, v in (b.get("overrides") or {}).items() if k in prof.algo_config(p)}}
    algo = b.get("algo") or cfg["algo"]
    syms = research.universe(p.universe)
    try:
        import pandas as pd
        prices = pd.concat({s: md.history(s, "2y")["close"] for s in syms if _ok(s)}, axis=1)
        value = {}
        if "stocks" in p.universe and user_agent() and algo in ("value", "t58_composite"):
            value = {s: research.sec_value_factor(s, _sec) for s in syms if s in research.DEFAULT_STOCKS}
            value = {k: v for k, v in value.items() if v is not None}
        w = algos.target_weights(algo, prices, cfg, value, research.CRYPTO)
        return jsonify({"ok": True, "algo": algo, "config": cfg, "weights": w, "value_scores": value,
                        "prices": {s: float(prices[s].dropna().iloc[-1]) for s in w}})
    except Exception as e:  # noqa: BLE001
        return _err(e)


def _ok(s):
    try:
        md.history(s, "2y"); return True
    except Exception:  # noqa: BLE001
        return False


@investing_bp.route("/investing/api/backtest", methods=["POST"])
def api_backtest():
    b = request.get_json(silent=True) or {}
    p = prof.build_profile(b.get("answers", {}), float(b.get("starting_cash") or 10000))
    cfg = prof.algo_config(p)
    algo = b.get("algo") or "momentum"
    if algo == "value":
        return _err("Value uses today's SEC data and cannot be back-tested without look-ahead. Test momentum / low-vol / composite instead.")
    syms = research.universe(p.universe)
    try:
        import pandas as pd
        prices = pd.concat({s: md.history(s, "10y")["close"] for s in syms if _ok(s)}, axis=1).dropna(how="all")
        return jsonify({"ok": True, "result": bt.backtest(prices, algo, cfg, research.CRYPTO)})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@investing_bp.route("/investing/api/account")
def api_account():
    mode = request.args.get("mode", "paper")
    try:
        br = _broker(mode, request.args.get("confirm") == "1")
        st = br.state()
        px = {}
        for p in st["positions"]:
            try: px[p["symbol"]] = md.quote(p["symbol"])["price"]
            except Exception: pass  # noqa: BLE001
        return jsonify({"ok": True, "account": br.state(px) if mode == "paper" else st})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@investing_bp.route("/investing/api/order", methods=["POST"])
def api_order():
    b = request.get_json(silent=True) or {}
    try:
        mode = b.get("mode", "paper")
        br = _broker(mode, bool(b.get("confirm_live")))
        sym = b["symbol"].upper()
        px = md.quote(sym)["price"]
        qty = float(b["qty"]) if b.get("qty") else float(b["dollars"]) / px
        return jsonify({"ok": True, "order": br.order(sym, b["side"], qty, px, "manual")})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@investing_bp.route("/investing/api/close", methods=["POST"])
def api_close():
    """Sell all (or a fraction) of one holding at the latest price: the one-click exit used by the position drawer."""
    b = request.get_json(silent=True) or {}
    try:
        mode = b.get("mode", "paper")
        br = _broker(mode, bool(b.get("confirm_live")))
        sym = str(b.get("symbol") or "").upper().strip()
        fraction = float(b.get("fraction") or 1.0)
        if not sym or not 0 < fraction <= 1:
            raise ValueError("Pick a holding and a share between 0% and 100% to sell.")
        held = next((p for p in br.state()["positions"] if p["symbol"].upper() == sym), None)
        if not held or held["qty"] <= 0:
            raise ValueError(f"You do not hold {sym}.")
        try:
            px = md.quote(sym)["price"]
        except Exception:  # noqa: BLE001 -- fall back to the position's own last price
            px = held["price"]
        qty = held["qty"] if fraction >= 0.999 else held["qty"] * fraction
        return jsonify({"ok": True, "order": br.order(sym, "sell", qty, px, "close position" if fraction >= 0.999 else f"sell {fraction:.0%}")})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@investing_bp.route("/investing/api/auto/run", methods=["POST"])
def api_auto_run():
    """Apply the automated plan to the account (paper by default). Hybrid style: client sends approve=true after the user reviews."""
    b = request.get_json(silent=True) or {}
    try:
        mode = b.get("mode", "paper")
        br = _broker(mode, bool(b.get("confirm_live")))
        weights, prices = b.get("weights") or {}, b.get("prices") or {}
        if not weights:
            return _err("No plan to apply.")
        if mode == "paper":
            placed = br.rebalance(weights, prices)
            br.set_settings(last_auto=dict(algo=b.get("algo"), weights=weights))
        else:
            st = br.state(); eq = st["equity"]; held = {p["symbol"]: p for p in st["positions"]}; placed = []
            for s, p in held.items():
                if s not in weights: placed.append(br.order(s, "sell", p["qty"], p["price"], "auto"))
            for s, w in weights.items():
                diff = w * eq - held.get(s, {}).get("market_value", 0)
                if diff > 25: placed.append(br.order(s, "buy", diff / prices[s], prices[s], "auto"))
        return jsonify({"ok": True, "orders": placed})
    except Exception as e:  # noqa: BLE001
        return _err(e)


@investing_bp.route("/investing/api/paper/reset", methods=["POST"])
def api_reset():
    return jsonify({"ok": True, "account": PaperAccount().reset(float((request.get_json(silent=True) or {}).get("cash") or 10000))})
