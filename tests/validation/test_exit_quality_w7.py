"""v5 bundle (w7): exit-quality analysis + sizing-halt banner + avg_give_back_r
refinement objective.

- Part C: red "TRADING STOPPED" banner renders at the top of the HTML
  report when equity_df.attrs["sizing_halt"]["halted"] is truthy, and is
  absent otherwise.
- Part A port #1: exit-quality math (give_back_r, exit_efficiency_pct,
  per-exit-mechanism stats, MIN_SAMPLE=8 insufficient labeling).
- refinement.py: "avg_give_back_r" present in FITNESS_METRICS and wired
  through compute_fitness.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.analysis.exit_quality import (
    MIN_SAMPLE,
    analyze_exit_quality,
    classify_row,
)
from app.optimize.refinement import (
    FITNESS_METRICS,
    RefinementError,
    compute_fitness,
)
from app.reports.generator import (
    _sizing_halt_banner,
    export_html,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _trade(*, direction=1, entry=100.0, exit_px=100.5, mfe=102.0,
           worst=None, stop=99.0, exit_reason="take_profit",
           exit_cause=None, pnl=50.0, inst="MGC"):
    t = {
        "direction": direction,
        "entry_price": entry,
        "exit_price": exit_px,
        "mfe_price": mfe,
        "worst_price": worst,
        "initial_risk": abs(entry - stop) if stop else None,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "instrument": inst,
    }
    if exit_cause is not None:
        t["exit_cause"] = exit_cause
    return t


def _halted_report():
    return {
        "generated_at": "2026-10-04T00:00:00+00:00",
        "strategy": {
            "name": "T", "source_type": "manual", "instrument": "MGC",
            "timeframe": "1m",
            "backtest_period_start": "2023-01-01",
            "backtest_period_end": "2024-01-01",
        },
        "verdict": None,
        "verdict_reasons": None,
        "final_parameters": None,
        "baseline_parameters": None,
        "risk_config": None,
        "execution_warnings": [],
        "headline_warnings": [],
        "sizing_halt": {
            "halted": True,
            "skipped": 8530,
            "skip_ratio": 0.957,
            "last_trade_exit": "2025-08-29",
            "risk_value": 0.25,
            "contract_size": 10,
        },
        "exit_quality": None,
        "historical_backtest": {"statistics": {"net_profit": 0.0}},
        "concentration_check": {},
        "cost_ladder": [],
        "holdout_comparison": None,
        "prop_firm_rules": {},
        "prop_firm_single_run": {},
        "monte_carlo": {
            "return_distribution": [],
            "return_percentiles": {5: 0.0, 50: 0.0, 95: 0.0},
            "drawdown_distribution": [],
            "drawdown_percentiles": {50: 0.0, 95: 0.0},
            "evaluation_pass_probability": 0.0,
            "first_payout_probability": 0.0,
            "failure_before_payout_probability": 0.0,
            "median_days_to_first_payout": None,
            "expected_payout": 0.0,
            "risk_of_ruin_pct": 0.0,
            "methodology_note": "",
            "n_simulations": 0,
        },
    }


# --------------------------------------------------------------------------
# Part C: sizing-halt banner
# --------------------------------------------------------------------------

def test_sizing_halt_banner_renders_when_halted(tmp_path):
    report = _halted_report()
    out = tmp_path / "report.html"
    export_html(report, out)
    page = out.read_text(encoding="utf-8")
    assert '<div class="sizing-halt-banner">' in page
    # banner sits at the top of the report body, before the Overview tab
    assert page.index("sizing-halt-banner") < page.index('data-tab="overview"')
    # exact spec copy, with the placeholder values filled in
    assert ("TRADING STOPPED — 8,530 signals (96%) were skipped because "
            "risk_value (0.25%) cannot afford 1 MGC contract with this "
            "strategy's stop width. Last trade: 2025-08-29. "
            "This is a configuration problem, not a strategy problem — "
            "raise risk_value, switch to the micro contract, increase "
            "account size, or tighten stops.") in page


def test_sizing_halt_banner_absent_when_not_halted(tmp_path):
    report = _halted_report()
    report["sizing_halt"] = None  # no attrs from the engine
    out = tmp_path / "report.html"
    export_html(report, out)
    page = out.read_text(encoding="utf-8")
    assert '<div class="sizing-halt-banner">' not in page
    assert "TRADING STOPPED" not in page


def test_sizing_halt_banner_never_crashes_on_partial_attrs():
    # defensive .get() contract: partial/malformed dicts must not raise
    assert _sizing_halt_banner(None, "MGC") == ""
    assert _sizing_halt_banner({}, "MGC") == ""
    assert _sizing_halt_banner({"halted": False}, "MGC") == ""
    assert _sizing_halt_banner("not-a-dict", "MGC") == ""
    partial = {"halted": True}  # engine set halted, nothing else
    page = _sizing_halt_banner(partial, "MGC")
    assert "TRADING STOPPED" in page


# --------------------------------------------------------------------------
# Part A port #1: exit-quality math
# --------------------------------------------------------------------------

def test_classify_row_give_back_math():
    # long: entry 100, stop 99 (1R = 1.0), peak 102 (+2.0R), exit 100.5 (+0.5R)
    s = classify_row(_trade())
    assert s["mfe_r"] == pytest.approx(2.0)
    assert s["realized_r"] == pytest.approx(0.5)
    assert s["give_back_r"] == pytest.approx(1.5)   # mfe_r - realized_r
    assert s["give_back_pct"] == pytest.approx(1.5)
    assert s["exit_efficiency_pct"] == pytest.approx(25.0)  # 0.5/2.0
    assert s["exit_cause"] == "take_profit"
    assert s["exit_reason_source"] == "inferred"


def test_classify_row_machine_verified_exit_cause():
    s = classify_row(_trade(exit_cause="trailing_stop"))
    assert s["exit_cause"] == "trailing_stop"
    assert s["exit_reason_source"] == "mechanism"


def test_classify_row_short_math():
    # short: entry 100, stop 101 (1R = 1.0), trough 98 (+2.0R favorable),
    # exit 99.5 (+0.5R realized) -> give_back_r = 1.5
    s = classify_row(_trade(direction=-1, exit_px=99.5, mfe=98.0, stop=101.0,
                            exit_reason="stop_loss"))
    assert s["side"] == "short"
    assert s["give_back_r"] == pytest.approx(1.5)
    assert s["exit_efficiency_pct"] == pytest.approx(25.0)


def test_classify_row_no_mfe_evidence_returns_none():
    t = _trade()
    del t["mfe_price"]
    t["worst_price"] = None
    assert classify_row(t) is None


def test_min_sample_constant_is_eight():
    assert MIN_SAMPLE == 8


def test_insufficient_labeling_for_thin_mechanism_sample():
    # 7 trailing-stop trades: below MIN_SAMPLE -> thin-sample flag
    trades = [_trade(exit_cause="trailing_stop", pnl=float(i)) for i in range(7)]
    result = analyze_exit_quality(trades)
    mech = [m for m in result["by_mechanism"] if m["exit_cause"] == "trailing_stop"]
    assert len(mech) == 1 and mech[0]["n"] == 7
    assert any("trailing stop" in note.lower() or "trailing_stop" in note
               for note in result["insufficient"])
    assert result["min_sample"] == 8


def test_inferred_rows_excluded_from_mechanism_table():
    trades = [_trade(exit_reason="take_profit", pnl=float(i)) for i in range(10)]
    result = analyze_exit_quality(trades)
    assert result["evidence_rows"] == 10
    assert result["by_mechanism"] == []  # inferred -> overall only
    assert any("mechanism" in note for note in result["insufficient"])


def test_no_evidence_is_honestly_flagged():
    result = analyze_exit_quality([{"entry_price": 1.0, "exit_price": 1.1}])
    assert result["evidence_rows"] == 0
    assert result["overall"] is None
    assert len(result["insufficient"]) == 1


def test_overall_avg_give_back_r_for_optimizer():
    trades = [_trade(pnl=float(i)) for i in range(10)]  # each give_back_r = 1.5
    result = analyze_exit_quality(trades)
    assert result["overall"]["avg_give_back_r"] == pytest.approx(1.5)


# --------------------------------------------------------------------------
# refinement.py: avg_give_back_r objective
# --------------------------------------------------------------------------

def test_avg_give_back_r_is_a_registered_objective():
    assert "avg_give_back_r" in FITNESS_METRICS
    assert "EXIT" in FITNESS_METRICS["avg_give_back_r"].upper() or \
           "exit" in FITNESS_METRICS["avg_give_back_r"].lower()


def test_compute_fitness_negates_avg_give_back_r():
    trades = [_trade(pnl=float(i)) for i in range(10)]  # avg give_back_r = 1.5
    mc = SimpleNamespace()
    fitness = compute_fitness({}, None, mc, "avg_give_back_r", trades=trades)
    assert fitness == pytest.approx(-1.5)  # maximize fitness == minimize give-back


def test_compute_fitness_give_back_without_trades_is_negative_inf():
    mc = SimpleNamespace()
    assert compute_fitness({}, None, mc, "avg_give_back_r", trades=None) == float("-inf")
    assert compute_fitness({}, None, mc, "avg_give_back_r", trades=[]) == float("-inf")


def test_compute_fitness_give_back_prefers_better_exits():
    mc = SimpleNamespace()
    good = [_trade(exit_px=101.9, pnl=float(i)) for i in range(10)]  # give_back 0.1R
    bad = [_trade(exit_px=100.5, pnl=float(i)) for i in range(10)]   # give_back 1.5R
    assert (compute_fitness({}, None, mc, "avg_give_back_r", trades=good)
            > compute_fitness({}, None, mc, "avg_give_back_r", trades=bad))


def test_unknown_metric_still_raises():
    with pytest.raises(RefinementError):
        compute_fitness({}, None, SimpleNamespace(), "not_a_metric")
