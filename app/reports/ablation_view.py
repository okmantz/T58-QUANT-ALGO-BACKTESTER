"""Ablation in plain words: for every rule, what the strategy does WITH it compared with WITHOUT it."""
from __future__ import annotations


def rule_effects(ablation: dict) -> list[dict]:
    """One row per removed rule from research.director.ablation_test()'s output.

    effect_pct   = how much better (+) or worse (-) the net result is with the rule than without it
    trades_delta = trades the rule removes (positive) or adds (negative) compared with running without it
    """
    rows = ablation.get("rows") or []
    if not rows:
        return []
    full = rows[0]
    out = []
    for row in rows[1:]:
        name = str(row.get("label", "")).removeprefix("Remove ")
        f_net, w_net = full.get("net_profit"), row.get("net_profit")
        f_n, w_n = full.get("total_trades") or 0, row.get("total_trades") or 0
        if f_net is None or w_net is None:
            out.append({"rule": name, "effect_pct": None, "trades_delta": w_n - f_n,
                        "sentence": f"{name}: without it the strategy made no trades, so the rule cannot be judged on profit.", "kind": "unknown"})
            continue
        effect = None if w_net == 0 else (f_net - w_net) / abs(w_net) * 100.0
        delta = w_n - f_n
        trades = (f"{abs(delta):,} trades removed" if delta > 0 else f"{abs(delta):,} trades added" if delta < 0 else "same number of trades")
        if effect is None:
            sentence = f"With this rule: {trades}; without it the result was flat, so no percentage applies."
            kind = "neutral"
        else:
            sentence = f"With this rule: {effect:+.1f}% result, {trades}."
            kind = "helps" if effect > 5 else "hurts" if effect < -5 else "neutral"
        out.append({"rule": name, "effect_pct": None if effect is None else round(effect, 1), "trades_delta": delta,
                    "sentence": sentence, "kind": kind})
    return sorted(out, key=lambda r: (r["effect_pct"] is None, r["effect_pct"] if r["effect_pct"] is not None else 0))
