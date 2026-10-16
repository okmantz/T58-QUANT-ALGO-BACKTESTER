"""Regression tests for the Dashboard's report links.

Bug: app.reports.run_history.record_run() (the shared function every
tool's report generation funnels through) stores report_html as a raw
ABSOLUTE FILESYSTEM PATH for some callers (e.g. Full Pipeline) instead of
a ready-to-use "/xxx_reports/<file>" URL. The Dashboard used to naively
build every report link as "/reports/<basename>" regardless of which tool
actually produced the file, which 404'd for anything not stored directly
under the plain reports/ root -- including after restarting the server,
since this is read back from the persistent run_history.json file. See
app.web.server._dashboard_report_url and the /dashboard route.
"""
from app.web.server import (
    FULL_PIPELINE_DIR, REPORTS_DIR, SEARCH_DIR, _dashboard_report_url, app,
)


def test_resolves_full_pipeline_absolute_path_to_its_own_route():
    raw = str(FULL_PIPELINE_DIR / "full_pipeline_003_my_strategy.html")
    assert _dashboard_report_url(raw) == "/full_pipeline_reports/full_pipeline_003_my_strategy.html"


def test_resolves_search_absolute_path_to_its_own_route():
    raw = str(SEARCH_DIR / "search_report.html")
    assert _dashboard_report_url(raw) == "/search_reports/search_report.html"


def test_resolves_plain_reports_dir_absolute_path():
    raw = str(REPORTS_DIR / "report.html")
    assert _dashboard_report_url(raw) == "/reports/report.html"


def test_leaves_an_already_correct_relative_url_unchanged():
    # The plain "Run & Report" tool and most others already store a
    # proper relative URL -- this must pass through unmodified.
    assert _dashboard_report_url("/search_reports/x.html") == "/search_reports/x.html"
    assert _dashboard_report_url("/refinement_reports/y.html") == "/refinement_reports/y.html"


def test_none_and_empty_are_safe():
    assert _dashboard_report_url(None) is None
    assert _dashboard_report_url("") is None


def test_dashboard_route_rewrites_full_pipeline_report_links(monkeypatch):
    """End-to-end: a run_history entry with a raw Full-Pipeline filesystem
    path should render a working /full_pipeline_reports/... link on the
    Dashboard page, not the old broken /reports/... guess."""
    raw_path = str(FULL_PIPELINE_DIR / "full_pipeline_042_demo.html")
    fake_row = {
        "strategy_name": "demo_strategy", "instrument": "ES1!", "timeframe": "1m",
        "timestamp": "2026-01-01T00:00:00+00:00", "single_run_passed": True,
        "sharpe_ratio": 1.2, "net_profit": 100.0, "equity_curve": [], "heatmap": [],
        "report_html": raw_path, "eval_pass_probability": 50.0, "expected_payout": 0.0,
        "risk_of_ruin_pct": 0.0, "win_rate": 55.0, "max_drawdown_pct": 5.0, "run_count": 1,
    }
    monkeypatch.setattr(
        "app.web.server.run_history.dashboard_data",
        lambda: {
            "total_strategies": 1, "total_runs": 1, "pass_rate": 100.0,
            "best": dict(fake_row), "strategies": [dict(fake_row)],
            "graph": {"nodes": [], "edges": [], "instruments": []},
            "heatmap": [[0.0] * 24 for _ in range(7)], "equity_series": [],
        },
    )
    client = app.test_client()
    r = client.get("/dashboard")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "/full_pipeline_reports/full_pipeline_042_demo.html" in body
    assert 'href="/reports/full_pipeline_042_demo.html"' not in body
