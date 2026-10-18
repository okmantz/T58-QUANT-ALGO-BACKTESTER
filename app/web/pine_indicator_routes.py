"""Pine Indicator Creator (``/pine-indicator``): TradingView indicators from an idea or from pasted code."""
from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.strategy import pine_indicator as P

pine_indicator_bp = Blueprint("pine_indicator", __name__, url_prefix="/pine-indicator")
MAX_CHARS = 60_000


def _ai_hint() -> str:
    try:
        from app.ai.llm_client import preferred_client, NullClient

        return "off" if isinstance(preferred_client(), NullClient) else "on"
    except Exception:  # noqa: BLE001
        return "off"


@pine_indicator_bp.route("/")
def page():
    return render_template("pine_indicator.html", active_page="pine_indicator", catalog=[{"id": c.id, "title": c.title, "overlay": c.overlay, "keywords": list(c.keywords[:3])} for c in P.CATALOG],
                           ai_state=_ai_hint())


@pine_indicator_bp.route("/api/create", methods=["POST"])
def api_create():
    body = request.get_json(silent=True) or {}
    mode = str(body.get("mode") or "idea")
    text = str(body.get("text") or "")
    if not text.strip():
        return jsonify({"ok": False, "error": "Type an idea or paste some code first."}), 400
    if len(text) > MAX_CHARS:
        return jsonify({"ok": False, "error": f"That is too long ({len(text):,} characters; the limit is {MAX_CHARS:,})."}), 400
    try:
        if mode == "code":
            res = P.check_and_fix(text)
            if not res.ok and body.get("use_ai"):
                # one AI repair pass over the user's own code
                from app.ai.llm_client import complete_text

                reply, err = complete_text(P._repair_prompt("Fix this user-supplied indicator without changing its behaviour.", res.code,
                                                             [P.Issue(i["level"], i["line"], i["message"], i.get("fix", "")) for i in res.issues]), system=P.SYSTEM_PROMPT)
                if reply:
                    fixed = P.check_and_fix(P._extract_code(reply))
                    if fixed.ok:
                        fixed.notes.insert(0, "Your code had compile problems; the AI repaired them. Review the diff against your original before using it.")
                        res = fixed
                elif err:
                    res.notes.append(f"AI repair unavailable: {err}")
        elif body.get("use_ai"):
            res = P.generate_with_llm(text)
        else:
            res = P.build_from_idea(text)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
    return jsonify({"ok": True, "result": res.to_dict()})
