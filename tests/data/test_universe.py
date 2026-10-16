"""Tests for app.orchestration.universe -- dots, families and layout."""
from __future__ import annotations

import math
import sqlite3
from types import SimpleNamespace

from app.orchestration import universe as un
from app.search.results_db import ResultsDB


def _row(cid, family="trend_breakout", stage="stage1", **kw):
    base = {"candidate_id": cid, "family": family, "source_type": "manual", "config": {}, "stage": stage,
            "statistics": {"net_profit": 500.0, "max_drawdown_pct": 3.2}, "composite_score": 1.0}
    base.update(kw)
    return base


def test_search_row_stages():
    rows = [
        _row("dead", passed_stage1=0),
        _row("tested", passed_stage1=1),
        _row("validated", stage="stage3", passed_stage1=1, passed_stage2=1, passed_stage3_gate=0),
        _row("winner", stage="stage3", passed_stage3_gate=1),
    ]
    stages = {d.id: d.stage for d in un.dots_from_search_rows(rows, "EURUSD", "5M")}
    assert stages == {"dead": "rejected", "tested": "tested", "validated": "validated", "winner": "survivor"}


def test_search_row_carries_symbol_timeframe_profit_drawdown_and_family():
    (d,) = un.dots_from_search_rows([_row("a")], "EURUSD", "5M")
    assert (d.symbol, d.timeframe, d.profit, d.drawdown_pct) == ("EURUSD", "5M", 500.0, 3.2)
    assert d.family == "breakout" or d.family == "trend_following"  # canonical group, not the skeleton name
    assert d.to_dict()["family_label"]


def test_missing_and_bad_numbers_become_none_not_errors():
    (d,) = un.dots_from_search_rows([_row("a", statistics={"net_profit": "n/a", "max_drawdown_pct": float("nan")},
                                          composite_score=None, fitness=None, quick_score=None)])
    assert d.profit is None and d.drawdown_pct is None and d.score == 0.0


def test_a_hostile_record_never_breaks_the_map():
    dots = un.dots_from_search_rows([{"candidate_id": "x", "config": "not a dict", "source_type": 5}])
    assert len(dots) == 1 and dots[0].family


def test_evolution_rows():
    rows = [
        {"candidate_id": "a", "family": "trend_breakout", "passed": False, "stage": "prefilter", "net_profit": -5},
        {"candidate_id": "b", "family": "trend_breakout", "passed": True, "stage": "prefilter"},
        {"candidate_id": "c", "family": "vwap_reversion", "passed": True, "stage": "full_eval",
         "net_profit": 900, "max_drawdown_pct": 2.5, "fitness_score": 70},
    ]
    dots = {d.id: d for d in un.dots_from_evolution_rows(rows, "ES", "15M")}
    assert [dots[k].stage for k in "abc"] == ["rejected", "tested", "validated"]
    assert dots["c"].profit == 900 and dots["c"].symbol == "ES" and dots["c"].source == "evolution"


def test_library_strategies_map_status_to_stage():
    def s(name, status, **meta):
        return SimpleNamespace(name=name, status=status, metadata=meta)

    dots = {d.id: d for d in un.dots_from_library([
        s("a.py", "tested_failed"), s("b.py", "ready_for_live", market="XAUUSD", timeframe="1H",
                                      last_run={"net_profit": 1200, "max_drawdown_pct": 4}),
        s("c.py", "weird_new_status"),
    ])}
    assert dots["a.py"].stage == "rejected" and dots["b.py"].stage == "survivor"
    assert dots["c.py"].stage == "tested"          # unknown status degrades, never raises
    assert (dots["b.py"].symbol, dots["b.py"].timeframe, dots["b.py"].profit) == ("XAUUSD", "1H", 1200)


def _dots(n_by_family, survivors_per=1):
    dots = []
    for fam, n in n_by_family.items():
        for i in range(n):
            dots.append(un.Dot(f"{fam}-{i}", fam, "ES", "5M", 10.0 * i, 2.0, "survivor" if i < survivors_per else "tested",
                               "search", score=float(n - i)))
    return dots


def test_layout_is_deterministic_and_inside_the_unit_square():
    dots = _dots({"trend_following": 40, "breakout": 25, "mean_reversion": 9})
    a, b = un.layout_universe(dots), un.layout_universe(list(reversed(dots)))
    assert a["dots"] == b["dots"] and a["clusters"] == b["clusters"]
    assert all(-1.0 <= d["x"] <= 1.0 and -1.0 <= d["y"] <= 1.0 for d in a["dots"])
    assert a["shown"] == 74 and a["truncated"] is False


def test_clusters_report_counts_survivors_and_are_ordered_by_size():
    out = un.layout_universe(_dots({"a": 3, "b": 8}, survivors_per=2))
    assert [c["family"] for c in out["clusters"]] == ["b", "a"]
    assert {c["family"]: (c["count"], c["survivors"]) for c in out["clusters"]} == {"a": (3, 2), "b": (8, 2)}


def test_dots_in_different_clusters_do_not_overlap_their_neighbours_centres():
    out = un.layout_universe(_dots({"a": 30, "b": 30, "c": 30}))
    cl = out["clusters"]
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            assert math.dist((cl[i]["cx"], cl[i]["cy"]), (cl[j]["cx"], cl[j]["cy"])) > cl[i]["r"] + cl[j]["r"]


def test_truncation_keeps_survivors_first():
    dots = _dots({"a": 100}, survivors_per=5)
    out = un.layout_universe(dots, max_dots=20)
    assert out["truncated"] and out["shown"] == 20 and out["total"] == 100
    assert sum(1 for d in out["dots"] if d["stage"] == "survivor") == 5


def test_empty_and_single_family_layouts():
    assert un.layout_universe([])["dots"] == []
    one = un.layout_universe(_dots({"solo": 5}))
    assert one["clusters"][0]["cx"] == 0.0 and len(one["dots"]) == 5


def test_load_search_dots_reads_result_dbs_and_skips_corrupt_ones(tmp_path):
    db_path = tmp_path / "search_aaa.db"
    with ResultsDB(db_path) as db:
        db.create_run("run1", "family", "trend_breakout", "EURUSD", "5M", 2, {})
        db.insert_candidate("run1", "c1", "stage1", {"family": "trend_breakout", "source_type": "manual",
                            "config": {}, "passed_stage1": True, "quick_score": 1.0,
                            "statistics": {"net_profit": 10.0, "max_drawdown_pct": 1.0}})
    (tmp_path / "search_bad.db").write_bytes(b"this is not sqlite")
    dots = un.load_search_dots(tmp_path)
    assert len(dots) == 1 and dots[0].symbol == "EURUSD" and dots[0].stage == "tested"


def test_load_universe_honours_source_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(un, "load_library_dots", lambda: _dots({"lib": 2}))
    monkeypatch.setattr(un, "load_evolution_dots", lambda limit=1500: _dots({"evo": 3}))
    only_lib = un.load_universe(tmp_path, ["library"])
    assert only_lib["total"] == 2 and only_lib["sources"] == ["library"]
    assert un.load_universe(tmp_path, ["library", "evolution"])["total"] == 5
