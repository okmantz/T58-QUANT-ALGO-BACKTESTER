"""Plain-words ablation, and the interactive breakdown section that ships inside every report."""
from __future__ import annotations

from types import SimpleNamespace

from app.reports import ablation_view, breakdowns


def test_rule_effects_say_what_the_rule_does_in_numbers():
    ab = {"rows": [{"label": "Full strategy", "net_profit": 8180.0, "total_trades": 400},
                   {"label": "Remove RSI filter (filter, long entry)", "net_profit": 10000.0, "total_trades": 777},
                   {"label": "Remove trend gate (filter, long entry)", "net_profit": 4000.0, "total_trades": 450}]}
    rows = ablation_view.rule_effects(ab)
    worst = rows[0]
    assert worst["rule"].startswith("RSI filter") and worst["effect_pct"] == -18.2 and worst["trades_delta"] == 377
    assert "With this rule: -18.2% result, 377 trades removed" in worst["sentence"] and worst["kind"] == "hurts"
    assert rows[1]["kind"] == "helps"


def test_rule_effects_handle_a_variant_with_no_trades():
    ab = {"rows": [{"label": "Full strategy", "net_profit": 100.0, "total_trades": 10}, {"label": "Remove X", "net_profit": None, "total_trades": 0}]}
    assert ablation_view.rule_effects(ab)[0]["kind"] == "unknown"


def test_compact_trades_accepts_objects_and_dicts():
    objs = [SimpleNamespace(entry_time="2024-03-05 09:35:00", pnl=12.345), {"entry_time": "2024-03-06 10:00:00", "pnl": -5}]
    assert breakdowns.compact_trades(objs) == [["2024-03-05T09:35", 12.35], ["2024-03-06T10:00", -5.0]]


def test_breakdown_section_embeds_the_payload_and_the_click_to_exclude_ui():
    html = breakdowns.breakdown_section([SimpleNamespace(entry_time="2024-03-05 09:35:00", pnl=-20.0)])
    assert 'id="bd-root"' in html and "What if you left this out?" in html and "2024-03-05T09:35" in html
    assert "<script>" in html and "Month by year" in html


def test_empty_trades_degrade_gracefully():
    assert "No trades" in breakdowns.breakdown_section([])
