"""Options Outlook -- web routes.

Deterministic call/put candidate generation (app.ai.options_outlook,
Black-Scholes under the hood -- see that module's docstring) plus an
optional Ollama-ranked outlook (app.ai.trading_assistant.options_outlook)
for a chosen symbol/horizon. Same "app computes facts, Ollama explains
them" split as app/web/ai_assistant_routes.py.

Registered as a Flask Blueprint, same pattern as quant_lab_routes.py /
ai_assistant_routes.py -- kept out of the already-large server.py.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.ai import options_outlook as options_outlook_module
from app.ai import trading_assistant
from app.ai.ollama_settings import load_settings as load_ollama_settings

options_outlook_bp = Blueprint("options_outlook", __name__, url_prefix="/options-outlook")


@options_outlook_bp.route("/")
def dashboard():
    return render_template("options_outlook.html", active_page="options_outlook")


@options_outlook_bp.route("/api/candidates", methods=["POST"])
def api_candidates():
    """Pure Black-Scholes candidates -- no Ollama call, so this works even
    with the AI assistant disabled/unconfigured."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        spot = float(data.get("spot"))
        iv = float(data.get("iv"))
        horizon = (data.get("horizon") or "today").strip()
        rate = float(data.get("rate", 0.045))
        dte_days = options_outlook_module.HORIZON_PRESETS.get(horizon)
        if dte_days is None:
            dte_days = int(data.get("dte_days") or 1)
    except (TypeError, ValueError):
        return jsonify({"error": "spot and iv must be numbers."}), 400

    try:
        candidates = options_outlook_module.build_candidates(spot=spot, iv=iv, dte_days=dte_days, r=rate)
    except options_outlook_module.OptionsOutlookError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "candidates": options_outlook_module.candidates_to_dicts(candidates),
        "expected_move": round(options_outlook_module.expected_move(spot, iv, dte_days), 4),
        "dte_days": dte_days,
    })


@options_outlook_bp.route("/api/outlook", methods=["POST"])
def api_outlook():
    """Candidates + an Ollama-ranked "Options Outlook" write-up. Requires
    Ollama enabled/configured (AI Assistant settings) -- returns the
    candidates either way so the table still renders if Ollama is off."""
    data = request.get_json(force=True, silent=True) or {}
    symbol = (data.get("symbol") or "SYMBOL").strip()
    notes = (data.get("notes") or "").strip()
    try:
        spot = float(data.get("spot"))
        iv = float(data.get("iv"))
        horizon = (data.get("horizon") or "today").strip()
        rate = float(data.get("rate", 0.045))
        dte_days = options_outlook_module.HORIZON_PRESETS.get(horizon)
        if dte_days is None:
            dte_days = int(data.get("dte_days") or 1)
    except (TypeError, ValueError):
        return jsonify({"error": "spot and iv must be numbers."}), 400

    try:
        candidates = options_outlook_module.build_candidates(spot=spot, iv=iv, dte_days=dte_days, r=rate)
    except options_outlook_module.OptionsOutlookError as exc:
        return jsonify({"error": str(exc)}), 400

    candidate_dicts = options_outlook_module.candidates_to_dicts(candidates)
    client = trading_assistant.TradingAssistantClient(load_ollama_settings())
    reply, error = client.options_outlook(symbol, horizon, candidate_dicts, notes=notes)

    return jsonify({
        "candidates": candidate_dicts,
        "expected_move": round(options_outlook_module.expected_move(spot, iv, dte_days), 4),
        "dte_days": dte_days,
        "reply": reply,
        "error": error,
    })
