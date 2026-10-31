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


_DSR = ("HARD VALIDATION GATE FAILED: REJECTED by the Deflated Sharpe gate: probabilistic Sharpe 0.908 < 0.95 -- after deflating for 13 "
        "evaluated configuration(s) (expected best-by-chance Sharpe 0.071), this champion's Sharpe of 1.234 is indistinguishable from what the "
        "search would produce by luck alone. This strategy is NOT READY -- a strong backtest that does not generalize is exactly what these gates exist to catch.")


def test_long_reason_text_is_never_cut_off():
    """Regression (2026-10-31): the label was cut at 90 characters and the detail at 220, ending in '...'. Every character must reach the page."""
    w = build_waterfall("NOT READY", [_DSR], {"score": 42.0, "tier": "Reject", "weight_coverage": 0.87})
    gate = next(s for s in w["steps"] if s["kind"] == "gate")
    assert "…" not in gate["label"] and "…" not in gate["detail"]
    assert gate["detail"].endswith("luck alone.") and "this champion's Sharpe of 1.234" in gate["detail"]
    assert gate["label"] == "Edge not convincing after counting every idea tried"      # recognised, so it is a hard gate with a plain-English fix
    assert gate["fix"]


def test_unrecognised_gate_keeps_full_text_and_is_not_merged():
    a = "HARD VALIDATION GATE FAILED: some brand new gate with a very long explanation " + "word " * 80 + "end."
    b = "HARD VALIDATION GATE FAILED: another brand new gate. Second sentence."
    w = build_waterfall("NOT READY", [a, b], None)
    gates = [s for s in w["steps"] if s["kind"] == "gate"]
    assert len(gates) == 2 and all("…" not in s["label"] + s["detail"] for s in gates)
    assert gates[0]["detail"].endswith("end.") and len(gates[0]["label"]) <= 80
