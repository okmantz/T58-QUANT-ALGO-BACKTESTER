"""One basis per number, real failed gates in the guidance, and no circular Optimize advice (Owen, 2026-11)."""
from datetime import date, timedelta

from app.guidance import next_step
from app.prop.account import PropAccount
from app.prop.presets import get_preset
from app.reports import basis
from app.scoring.verdict_explain import explain_verdict


def _b(source, **kw):
    d = dict(prop_preset="topstep_100k", risk_pct=1.0, sizing_mode="fit_stop", timeframe="15m", window="full", trades=641,
             reset_on_breach=True)
    d.update(kw)
    return basis.make_basis(source, **d)


def test_same_basis_has_no_differences_and_equal_fingerprint():
    a, b = _b("full_pipeline"), _b("validate")
    assert basis.differences(a, b) == []
    assert basis.fingerprint(a) == basis.fingerprint(b)


def test_different_basis_names_each_difference():
    a = _b("full_pipeline", risk_pct=0.19, sizing_mode="rr_planned", window="dev_80", trades=177)
    b = _b("validate")
    text = " ".join(basis.differences(a, b))
    assert "0.19%" in text and "1%" in text
    assert "sizing" in text.lower() and "holdout" in text and "177" in text
    assert basis.fingerprint(a) != basis.fingerprint(b)


def test_annotate_mc_item_explains_gap_instead_of_contradicting():
    saved = {"eval_pass_probability": 94.0, "basis": _b("full_pipeline", risk_pct=0.19, window="dev_80", trades=177)}
    item = {"check": "monte_carlo", "detail": "x", "metrics": [("Eval pass chance", "39%")], "guidance": {"steps": ["a"]}}
    basis.annotate_mc_item(item, _b("validate"), saved)
    assert "94%" in item["detail"] and "NOT contradicting" in item["detail"]
    assert item["guidance"]["steps"][0].startswith("Run Validate with the same risk")


def test_annotate_same_basis_flags_real_disagreement():
    saved = {"eval_pass_probability": 94.0, "basis": _b("full_pipeline")}
    item = {"check": "monte_carlo", "detail": "", "metrics": [("Eval pass chance", "39%")], "guidance": {"steps": []}}
    basis.annotate_mc_item(item, _b("validate"), saved)
    assert "real disagreement" in item["detail"]


def test_champion_check_reasons_and_fixes_are_surfaced():
    ve = explain_verdict("NOT READY", ["HARD SAFETY GATE FAILED: Monte Carlo risk of ruin 41.9% > 25%"], score=50)
    assert ve["fixes"] and "risk" in ve["fixes"][0].lower()
    assert "hard safety gates" in ve["what_is_it"]


def _row(optimized):
    ve = explain_verdict("NOT READY", ["HARD SAFETY GATE FAILED: Monte Carlo risk of ruin 41.9% > 25%"], score=50)
    return {"strategy_type": "python", "filename": "x.py", "display_name": "x", "verdict": "NOT READY",
            "verdict_explain": ve, "optimized": optimized, "promotion": {"stage": "candidate", "requirements": [
                {"key": "backtested", "met": True}]}}


def test_not_ready_guidance_quotes_gate_and_breaks_the_optimize_loop():
    run = {"net_profit": 5000.0, "sharpe_ratio": 1.5, "trades": 300, "max_drawdown_pct": 5.0}
    first = next_step.resolve(_row(False), run)
    assert "blew up" in first["why"] and first["primary"]["label"] == "Open Optimize"
    again = next_step.resolve(_row(True), run)
    assert again["primary"]["label"] != "Open Optimize"
    assert "already" in again["headline"].lower()
    assert any("will not change the verdict" in s for s in again["steps"])


def test_prop_account_start_index_past_end_does_not_raise():
    rules = get_preset("apex_50k_eod").to_prop_rules()
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(3)]
    acct = PropAccount(rules, start_day_index=len(days), day_dates=days)
    acct.on_trade_close(10.0, len(days) - 1)  # used to raise IndexError (list index out of range)


def test_library_row_percent_is_not_rescaled():
    from app.strategy import library_control as lc
    assert lc._pct(1.0) == 1.0 and lc._pct(0.5) == 0.5 and lc._pct(94.0) == 94.0
