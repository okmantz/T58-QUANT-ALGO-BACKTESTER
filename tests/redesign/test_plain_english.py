"""The plain-English card is derived only from numbers in the report and never breaks it."""
from __future__ import annotations

from app.reports import plain_english as pe

REPORT = {
    "verdict": "NOT READY", "verdict_reasons": ["Out-of-sample holdout failed."],
    "strategy": {"name": "S", "backtest_period_start": "2021-01-01", "backtest_period_end": "2024-12-31"},
    "historical_backtest": {"statistics": {"total_trades": 412, "net_profit": 18250.0, "win_rate": 44.2, "profit_factor": 1.31, "max_drawdown_pct": 6.4}},
    "monte_carlo": {"risk_of_ruin_pct": 31.0, "evaluation_pass_probability": 38.0, "per_attempt_pass_probability": 38.0, "total_independent_attempts": 500},
    "concentration_check": {"best_trade_pct_of_gross_profit": 41.0, "net_profit_excluding_best_trade": -200.0},
    "holdout_comparison": {"in_sample_statistics": {"net_profit": 20000.0}, "holdout_statistics": {"net_profit": -900.0}},
    "headline_warnings": ["w1"],
}


def test_summary_reads_the_numbers_and_flags_the_real_concerns():
    s = pe.summarize(REPORT)
    text = " ".join(s["bullets"] + s["concerns"])
    assert "412 trades" in text and "$18,250" in text and "44%" in text and "1.31x" in text and "6.4%" in text
    assert "38 of every 100" in text and "weak rate" in text
    assert any("single best trade" in c or "best single trade" in c for c in s["concerns"]) and any("never saw" in c for c in s["concerns"])
    assert s["stage"] in ("LEAD", "REJECT") and s["next_step"]


def test_missing_numbers_are_skipped_not_invented():
    s = pe.summarize({"verdict": "READY", "strategy": {}, "historical_backtest": {"statistics": {}}, "monte_carlo": {}})
    assert s["bullets"] == [] and s["concerns"] == []
    assert pe.summary_html({"verdict": "READY"}) == ""


def test_html_escapes_and_is_a_single_section():
    r = dict(REPORT, verdict_reasons=["<script>x</script> failed"])
    h = pe.summary_html(r)
    assert h.startswith('<section class="sec plain"') and "<script>" not in h and "In plain English" in h


def test_validation_reports_open_with_a_plain_english_card():
    from app.reports.validation_reports import _page
    out = _page("CPCV Report", "<p>body</p>")
    assert "How stable is the result" in out and "<p>body</p>" in out
