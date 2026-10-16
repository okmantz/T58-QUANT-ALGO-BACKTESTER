from app.reports.charts import svg_attempt_chart
from app.reports.generator import _reliability_header


def test_attempt_chart_draws_one_polyline_per_attempt_plus_cumulative():
    segs = [dict(attempt_id=0, values=[50000, 50500, 48000], start_index=0, outcome="failed", start_balance=50000),
            dict(attempt_id=1, values=[50000, 51000, 53000], start_index=3, outcome="passed", start_balance=50000)]
    svg = svg_attempt_chart(segs)
    assert svg.count("<polyline") == 3 and "#F05B63" in svg and "#1e9e5a" in svg
    # attempt 1 starts at the account size on its own line: no teleport inside a single polyline
    assert "cumulative" in svg


def test_report_header_flags_thin_sample():
    rep = {"historical_backtest": {"statistics": {"total_trades": 14}}, "sizing_halt": {"skip_ratio": 0.99}}
    h = _reliability_header(rep)
    assert "14 trades" in h and "99%" in h and "Only 14 trades" in h
