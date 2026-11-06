"""Legacy READY / MARGINAL / NOT READY are internal only; users see REJECT / LEAD / FORWARD-TEST CANDIDATE (EVALUATE / FUNDED from forward tests)."""
from __future__ import annotations

import re

from app.orchestration.verdict_labels import headline, relabel_list, relabel_text, tone

LEGACY = re.compile(r"\bNOT READY\b|\bMARGINAL\b|\bREADY\b")


def test_headline_never_exceeds_forward_test_candidate_for_a_backtest():
    assert headline("READY") == "FORWARD-TEST CANDIDATE"
    assert headline("MARGINAL") == "LEAD"
    assert headline("NOT READY") == "LEAD"
    assert headline("NOT READY", reasons=["EV TEST FAILED: fee not earned back"]) == "REJECT"
    assert headline("READY", "LEAD") == "LEAD"                       # a stored ladder stage always wins
    assert headline("READY") not in ("EVALUATE", "FUNDED")


def test_relabel_text_removes_every_legacy_word_but_keeps_search_markers():
    src = ["CAPPED AT MARGINAL: the score", "This strategy is NOT READY regardless", "READY needs >= 70%", "UNPROVEN (caps at MARGINAL): x",
           "read the READY / MARGINAL / NOT READY verdict"]
    for t in relabel_list(src):
        assert not LEGACY.search(t), t
    assert relabel_text("CAPPED AT MARGINAL: x").startswith("CAPPED AT")        # audit code finds caps by this marker
    assert relabel_text("EV TEST FAILED: x") == "EV TEST FAILED: x"
    assert relabel_text(None) is None and relabel_text(5) == 5


def test_tone():
    assert tone("FORWARD-TEST CANDIDATE") == "good" and tone("LEAD") == "warn" and tone("REJECT") == "bad" and tone("READY") == "good"


def test_report_shows_only_ladder_names():
    from app.reports.generator import _verdict_section
    html = _verdict_section("NOT READY", ["CAPPED AT MARGINAL: thin data", "This strategy is NOT READY: EV TEST FAILED x"], None, None)
    assert "LEAD" in html or "REJECT" in html
    assert not LEGACY.search(re.sub(r"<[^>]+>", " ", html))
    ok = _verdict_section("READY", ["T58 Score 80"], {"label": "HIGH CONFIDENCE"}, {"stage": "FORWARD-TEST CANDIDATE"})
    assert "FORWARD-TEST CANDIDATE" in ok and not LEGACY.search(re.sub(r"<[^>]+>", " ", ok))


def test_web_filters(tmp_path):
    from app.web.server import _ladder_filter, _relabel_filter
    assert _ladder_filter("READY") == "FORWARD-TEST CANDIDATE" and _ladder_filter("MARGINAL") == "LEAD"
    assert not LEGACY.search(_relabel_filter("NOT READY and READY and MARGINAL"))
