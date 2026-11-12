"""Failure diagnosis: the screenshot case (NQ Ind Rule 15m) must say exactly why it failed and what to change."""
from __future__ import annotations

import math

from app.guidance import failure_diagnosis as fd
from app.guidance.next_step import resolve
from app.scoring.verdict_explain import explain_verdict

UNPROVEN = ("UNPROVEN (caps at MARGINAL): pooled OOS mean R +0.155 (95% day-block bootstrap lower bound -0.060) over 51 trades, "
            "100% of merged folds positive. Positive but the lower bound has not cleared zero yet -- more trades are needed. Not a rejection.")
UNAVAIL = "3 of 14 checks weren't available (parameter_stability, cpcv_supporting, dsr_unseen) -- score is re-normalised over what WAS checked"
EV = ("EV TEST FAILED: EV per attempt is $-41.20 (<= 0) at the holdout pass rate 31.0% (break-even 44.0%). "
      "The strategy does not earn back the fee at its measured, deflated pass rate.")
NOTE = "NOTE: the firm has not confirmed in writing that this automation is allowed (written_confirmation=False). Get it before paying."
LADDER = "LADDER: REJECT | EV: NOT WORTH A FEE"
PICK = "python::NQ1! Ind Rule 15m.py"


def _diag(**kw):
    return fd.diagnose([UNPROVEN, UNAVAIL, NOTE, EV, LADDER], pick=PICK, verdict="NOT READY", **kw)


def test_trades_needed_matches_the_interval_maths():
    n = fd.trades_needed_for_lower_bound(0.155, -0.060, 51)
    assert n["need"] == math.ceil(51 * (0.215 / 0.155) ** 2) == 99
    assert n["more"] == 48
    assert fd.trades_needed_for_lower_bound(-0.1, -0.3, 40) is None
    assert fd.trades_needed_for_lower_bound(0.1, 0.2, 40) is None


def test_names_every_cause_with_numbers():
    d = _diag()
    ids = [c["id"] for c in d["causes"]]
    assert {"oos_unproven", "checks_missing", "ev_negative", "written_confirmation"} <= set(ids)
    oos = next(c for c in d["causes"] if c["id"] == "oos_unproven")
    assert any("51" in e for e in oos["evidence"]) and any("99" in e for e in oos["evidence"])
    ev = next(c for c in d["causes"] if c["id"] == "ev_negative")
    assert any("-$41.20" in e for e in ev["evidence"]) and any("44.0%" in e for e in ev["evidence"])


def test_written_confirmation_is_advisory_not_the_cause():
    d = _diag()
    wc = next(c for c in d["causes"] if c["id"] == "written_confirmation")
    assert wc["severity"] == "advisory" and "not the failure" in wc["title"]
    assert d["causes"][-1]["id"] == "written_confirmation"          # advisory sorts last
    assert d["primary_cause"] != wc["title"]


def test_missing_checks_link_to_prefilled_validate():
    d = _diag()
    c = next(x for x in d["causes"] if x["id"] == "checks_missing")
    assert c["action"]["kind"] == "validate"
    assert "run=sensitivity%2Crobustness%2Ccpcv" in c["action"]["href"] or "run=sensitivity,robustness,cpcv" in c["action"]["href"]


def test_unproven_points_at_data_not_rerun():
    d = fd.diagnose([UNPROVEN], pick=PICK, verdict="NOT READY")
    c = d["causes"][0]
    assert c["rerun_changes_it"] is False
    assert c["action"]["kind"] == "data"                              # ~1.9x more trades are needed -> more history
    assert any("same data and settings" in s for s in c["steps"])
    assert "cannot create more trades" in " ".join(c["steps"])


def test_optimizer_answer_and_loop_breaker():
    fresh = _diag()
    assert fresh["optimizer"]["can_help"] == "partly" and not fresh["optimizer"]["tried_without_gain"]
    tried = _diag(optimized={"improved": False})
    assert tried["optimizer"]["tried_without_gain"] is True
    assert "already ran once" in tried["optimizer"]["explain"]
    assert tried["primary"] and tried["primary"]["href"]
    only_data = fd.diagnose([UNAVAIL], pick=PICK, verdict="MARGINAL")
    assert only_data["optimizer"]["can_help"] == "no"


def test_ev_blockers_each_get_an_action():
    txt = ["EV TEST FAILED: bankroll ruin before first payout is 31.0% (cap 20%). this firm preset's rules are not verified against a firm page."]
    ids = {c["id"]: c for c in fd.diagnose(txt, pick=PICK)["causes"]}
    assert ids["ev_bankroll_ruin"]["action"]["kind"] == "optimize" and "goal=risk" in ids["ev_bankroll_ruin"]["action"]["href"]
    assert ids["ev_rules_unverified"]["action"]["href"] == "/prop-firm-recommender"


def test_no_reasons_is_not_a_diagnosis():
    d = fd.diagnose([], pick=PICK)
    assert d["has"] is False and d["causes"] == [] and d["primary"] is None


def test_next_step_uses_the_diagnosis_instead_of_the_dead_end():
    diag = _diag(optimized={"improved": False})
    row = {"strategy_type": "python", "filename": "NQ1! Ind Rule 15m.py", "verdict": "NOT READY", "optimized": True, "diagnosis": diag,
           "promotion": {"requirements": [{"key": "backtested", "met": True}]}}
    plan = resolve(row, {"sharpe_ratio": 0.9, "trades": 463, "net_profit": 13761.0})
    assert plan["headline"].startswith("Fix: ")
    assert "Already optimized" not in plan["headline"]
    assert plan["primary"]["label"] != "Re-run Full Pipeline"
    assert any("same data and settings" in s for s in plan["steps"])
    assert plan["diagnosis"]["has"]


def test_next_step_without_diagnosis_is_unchanged():
    row = {"strategy_type": "python", "filename": "x.py", "verdict": "NOT READY", "optimized": True,
           "promotion": {"requirements": [{"key": "backtested", "met": True}]}}
    assert resolve(row, {"sharpe_ratio": 0.9, "trades": 463, "net_profit": 5.0})["headline"] == "Already optimized -- change the setup or retire it"


def test_verdict_explain_leads_with_diagnosed_causes():
    ve = explain_verdict("NOT READY", [UNPROVEN, UNAVAIL, NOTE, EV, LADDER], score=66, tier="Promising")
    assert ve["points"][0].startswith("One evaluation attempt loses money") or "out-of-sample" in " ".join(ve["points"])
    joined = " ".join(ve["points"])
    assert "unproven" in joined.lower() and "-$41.20" in joined
    assert ve["fixes"] and ve["diagnosis"]["has"]
    assert not any(p.startswith(("UNPROVEN", "LADDER", "EV TEST FAILED")) for p in ve["points"])
