from datetime import datetime, timedelta, timezone

from app.global_events import classify as c
from app.global_events import theaters as th


def test_golden_headlines():
    g = c.classify("Fed's Powell signals patience as inflation data looms")
    assert g["category"] == "fed" and g["impact_raw"] == 2 and "ES" in g["instruments"]
    g = c.classify("US announces new tariffs on Chinese chips")
    assert g["category"] == "tariff" and g["impact_raw"] == 2 and "trade" in g["theaters"]
    g = c.classify("Emergency rate cut announced after bank run")
    assert g["impact_raw"] == 3
    assert c.classify("Local bakery wins award")["impact_raw"] == 0


def test_word_boundaries_not_substrings():
    # "ppi" must not match inside "shipping", "oil" not inside "boil", "cpi" not inside "scrapin"
    g = c.classify("Red Sea shipping rates climb as carriers reroute")
    assert g["category"] != "inflation"
    assert "redsea" in g["theaters"]
    assert c.classify("How to boil water")["category"] == "other"
    assert c.classify("Tariffs hit")["category"] == "tariff"      # plural still matches the stem


def test_past_reference_downgrades_one_tier():
    assert c.classify("CPI surprise rattles markets")["impact_raw"] == 2
    assert c.classify("Anniversary of the CPI surprise of 1979")["impact_raw"] == 1


def test_theater_hit_gets_low_tag_without_macro_keyword():
    g = c.classify("Tanker struck near Hormuz; Houthi claim attack")
    assert "mideast" in g["theaters"] and g["impact_raw"] >= 1


def test_tier_cap_until_corroborated():
    assert c.effective_impact(3, 1, False) == 3 and c.effective_impact(2, 2, False) == 2
    assert c.effective_impact(3, 3, False) == 1 and c.effective_impact(2, None, False) == 1
    assert c.effective_impact(3, 3, True) == 3


def test_jaccard_clustering_window_and_publisher():
    now = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    a = c.tokens("Iran tanker strike raises Hormuz risk, crude jumps")
    b = c.tokens("Iran tanker strike raises Hormuz risk, oil climbs")
    existing = [(7, a, "Reuters", now)]
    cid, corroborated = c.cluster_into(existing, b, "Bloomberg", now + timedelta(minutes=5))
    assert cid == 7 and corroborated
    cid, corroborated = c.cluster_into(existing, b, "Reuters", now + timedelta(minutes=5))
    assert cid == 7 and not corroborated                      # same publisher never corroborates itself
    assert c.cluster_into(existing, b, "Bloomberg", now + timedelta(hours=9))[0] is None   # outside the 6h window
    assert c.cluster_into(existing, c.tokens("Bitcoin ETF flows turn positive"), "CoinDesk", now)[0] is None


def test_theater_for_point_prefers_smallest_box():
    assert th.theater_for_point(38.0, 20.0) == "redsea"      # Red Sea box is the smallest containing it
    assert th.theater_for_point(121.0, 24.0) == "taiwan"
    assert th.theater_for_point(0.0, -40.0) is None
    assert th.theater_for_point(None, 10.0) is None


def test_heat_is_clamped_and_warmup_has_no_zscore():
    s, comp = th.heat(50.0, 1.0, 1.0, 10, 10)
    assert s == 100.0 and comp["hi_impact_6h"] == 5 and comp["hazard_24h"] == 3
    s, comp = th.heat(0.0, 20.0, 1.0, 0, 0)
    assert s == 4.0 and comp["vol_z"] == -3.0               # z clamped at -3 -> 40 - 36
    s, comp = th.heat(9.0, 0.0, 0.0, 0, 0, calibrating=True)
    assert comp["vol_z"] == 0.0 and s == 40.0 and comp["calibrating"]
