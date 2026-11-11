"""Champion page v2: per-strategy promotion assessment, 1st/2nd/3rd podium and same-market replacement."""
import shutil

import pytest

from app.scoring import champion_board as cb
from app.strategy import library


@pytest.fixture(autouse=True)
def clean_library_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    base = library.get_strategy_library_dir()
    yield
    shutil.rmtree(base, ignore_errors=True)


def _seed(name, *, instrument="GC", score=70.0, stage="champion", eval_pct=75.0):
    library.save_strategy_bytes(b"print('x')\n", name, "python")
    methods = {m: {"recorded_at": 1, "efficiency": 0.7} for m in library.VALIDATION_STEP_METHODS}
    library.save_strategy_metadata("python", name, {
        "last_run": {"trades": 420, "win_rate": 48.0, "max_dd": 5.0, "eval_pass_probability": eval_pct, "first_payout_probability": 65.0,
                     "instrument": instrument},
        "last_validation": {"efficiency": 0.7}, "validation_methods_run": methods,
        "last_champion_check": {"verdict": "READY", "t58_score": score, "ladder_stage": "FORWARD-TEST CANDIDATE"},
        "promotion": {"stage": stage, "history": {}},
    })
    return "python", name


def _row(name):
    return next(r for r in cb.list_board() if r["filename"] == name)


def test_assessment_eligible_has_score_and_four_checks():
    _seed("a.py", stage="candidate")
    a = cb.assessment_for_row(_row("a.py"))
    assert a["eligible"] and a["headline"] == "Eligible" and a["score"] == 70.0
    assert [c["label"] for c in a["checks"]] == ["Validation completed", "READY requirements satisfied",
                                                  "Strategy version recorded", "No unresolved critical issues"]
    assert all(c["ok"] for c in a["checks"])


def test_open_lookahead_issue_blocks_even_when_gates_pass():
    _seed("b.py", stage="candidate")
    library.save_strategy_metadata("python", "b.py", {"lookahead": {"clean": False}})
    a = cb.assessment_for_row(_row("b.py"))
    assert not a["eligible"] and a["tone"] == "bad" and not a["checks"][3]["ok"]


def test_podium_award_shifts_and_caps_at_three():
    for i, n in enumerate(["p1.py", "p2.py", "p3.py", "p4.py"]):
        _seed(n, instrument=["ES", "NQ", "CL", "YM"][i])
    for n in ["p1.py", "p2.py", "p3.py"]:
        ok, msg = cb.award_place("python", n, 1)
        assert ok, msg
    holders = cb.podium_holders(cb.list_board())
    assert [holders[p]["filename"] for p in (1, 2, 3)] == ["p3.py", "p2.py", "p1.py"]
    ok, _ = cb.award_place("python", "p4.py", 1)
    holders = cb.podium_holders(cb.list_board())
    assert ok and [holders[p]["filename"] for p in (1, 2, 3)] == ["p4.py", "p3.py", "p2.py"]
    assert _row("p1.py")["podium_place"] is None


def test_same_market_needs_explicit_replace_and_comparison_names_a_leader():
    _seed("old.py", instrument="GC", score=60.0)
    _seed("new.py", instrument="GC", score=80.0)
    assert cb.award_place("python", "old.py", 1)[0]
    ok, msg = cb.award_place("python", "new.py", 1)
    assert not ok and "already holds" in msg
    rows = cb.list_board()
    clash = cb.overlapping_holders(next(r for r in rows if r["filename"] == "new.py"), rows)
    assert [c["filename"] for c in clash] == ["old.py"]
    cmp = cb.compare_rows(next(r for r in rows if r["filename"] == "new.py"), clash[0])
    assert cmp["lead"] == "new" and "replacing it is reasonable" in cmp["advice"]
    ok, msg = cb.award_place("python", "new.py", 1, replace_same_market=True)
    assert ok and "replaced" in msg
    assert _row("old.py")["podium_place"] is None and _row("new.py")["podium_place"] == 1


def test_unready_strategy_cannot_take_a_place():
    _seed("c.py", stage="candidate")
    ok, msg = cb.award_place("python", "c.py", 1)
    assert not ok and "Champion Candidate" in msg


def test_champion_page_renders_with_assessment_and_without_rerun(tmp_path):
    import app.web.server as server
    _seed("pg.py", stage="candidate")
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        r = c.get("/champion-simple")
        body = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Promotion assessment" in body and "ch-card" in body and "Re-run checks" not in body
