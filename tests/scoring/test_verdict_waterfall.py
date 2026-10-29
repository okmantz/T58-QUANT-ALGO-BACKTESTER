from app.scoring.verdict_waterfall import build_waterfall


def test_waterfall_orders_score_then_blockers_then_final():
    w = build_waterfall("MARGINAL", [
        "CAPPED AT MARGINAL: the T58 Score alone would have called this READY, but the final trade count is 41 (needs 100).",
        "For reference, T58 Score: 81/100 -- Strong",
        "COST STRESS: at 2x spread the same strategy keeps 12% of profit (needs 40%).",
    ], {"score": 81.0, "tier": "Strong", "weight_coverage": 0.9, "missing_core": []})
    kinds = [s["kind"] for s in w["steps"]]
    assert kinds[0] == "score" and kinds[-1] == "final" and "gate" in kinds and "cap" in kinds
    assert w["single_change"]["label"] == "Profit does not survive doubled costs"        # a hard gate outranks a cap
    assert all("58" not in s["measured"].split(", ") for s in w["steps"])                 # 'T58' is a name, not a measurement


def test_ready_has_no_blockers_and_unknown_reason_is_still_shown():
    w = build_waterfall("READY", [], {"score": 88.0, "tier": "Elite"})
    assert w["single_change"] is None and w["steps"][-1]["label"] == "READY"
    w2 = build_waterfall("NOT READY", ["SOMETHING NEW: a brand new gate fired. More text."], None)
    assert any("brand new gate" in s["detail"] for s in w2["steps"])
