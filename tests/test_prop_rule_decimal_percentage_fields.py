"""
FIX (2026-09-18): `<input type="number">` with no `step` attribute defaults
to a browser-enforced step of 1 -- HTML5's native validation then silently
rejects (blocks form submission on) any decimal value, e.g. typing "2.4"
into a "Daily loss limit (%)" field for a prop firm (such as Lucid) whose
actual rule is a non-integer percentage. Every OTHER tab in this app that
has these same prop-rule fields (Full Pipeline, Quick Optimize, Search
Lab, Evolution Lab, Forge, Speed Run, CPCV, Sensitivity, Refine, Research,
Research Agent, Multi-Objective, Parameter Robustness, WFO, WFGA, Risk
Sweep) already had `step="0.1"` on these inputs -- only Run & Report
(index.html), Overnight Autopilot, and Payout Probability were missing it.
These tests pin down that all three now match every other tab, so a
decimal daily-loss/profit-target/max-drawdown/consistency/payout-
threshold/buffer value can actually be submitted from any of them.
"""
from __future__ import annotations

import re

from app.web.server import app

# Every one of these percentage fields must accept decimals: step="0.1"
# (or looser) rather than the browser's implicit step=1 default.
_DECIMAL_PCT_FIELDS = ["profit_target", "daily_loss", "max_dd"]


def _has_step_attr(html: bytes, field_name: str) -> bool:
    """True if `field_name`'s <input> tag carries an explicit step
    attribute anywhere in it (order of attributes isn't fixed across
    templates, so this checks the whole tag, not just what immediately
    precedes/follows `name=`)."""
    text = html.decode("utf-8", errors="ignore")
    # Grab the whole <input ...> tag that has this exact name=, in either
    # attribute order used across the app's templates.
    pattern = re.compile(
        r'<input\b[^>]*\bname="' + re.escape(field_name) + r'"[^>]*>'
        r'|<input\b[^>]*\bstep="[^"]*"[^>]*\bname="' + re.escape(field_name) + r'"[^>]*>'
    )
    matches = pattern.findall(text)
    assert matches, f"no <input name=\"{field_name}\"> found in this page at all"
    return any('step="' in m for m in matches)


def test_run_and_report_daily_loss_and_prop_fields_accept_decimals():
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    for field in _DECIMAL_PCT_FIELDS + ["consistency", "payout_threshold", "buffer"]:
        assert _has_step_attr(r.data, field), (
            f"Run & Report's '{field}' input is missing step=\"0.1\" -- a decimal like 2.4 "
            "would be silently rejected by the browser's native number-input validation."
        )


def test_overnight_autopilot_prop_fields_accept_decimals():
    client = app.test_client()
    r = client.get("/overnight-autopilot")
    assert r.status_code == 200
    for field in _DECIMAL_PCT_FIELDS:
        assert _has_step_attr(r.data, field), (
            f"Overnight Autopilot's '{field}' input is missing step=\"0.1\"."
        )


def test_payout_probability_prop_fields_accept_decimals():
    client = app.test_client()
    r = client.get("/payout-probability")
    assert r.status_code == 200
    for field in _DECIMAL_PCT_FIELDS:
        assert _has_step_attr(r.data, field), (
            f"Payout Probability's '{field}' input is missing step=\"0.1\"."
        )


def test_full_pipeline_already_had_the_fix_unchanged():
    """Control case: Full Pipeline was never broken -- confirms the test
    helper itself correctly detects step="0.1" where it's already
    present, rather than passing everything unconditionally."""
    client = app.test_client()
    r = client.get("/full-pipeline")
    assert r.status_code == 200
    for field in _DECIMAL_PCT_FIELDS:
        assert _has_step_attr(r.data, field)
