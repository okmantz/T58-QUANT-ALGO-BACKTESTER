import shutil

import pytest

from app.scoring import champion_board as cb
from app.strategy import library


@pytest.fixture(autouse=True)
def clean_library_dir(tmp_path, monkeypatch):
    """Isolate every test in this file from the real strategies/
    directory, mirroring tests/test_strategy_library.py's fixture."""
    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    base_dir = library.get_strategy_library_dir()
    yield
    shutil.rmtree(base_dir, ignore_errors=True)


def _seed(filename="s1.py", strategy_type="python"):
    library.save_strategy_bytes(b"print('hi')\n", filename, strategy_type)
    return strategy_type, filename


# ---------------------------------------------------------------------------
# oos_pct / status_from_verdict
# ---------------------------------------------------------------------------

def test_oos_pct_from_efficiency_fraction():
    assert cb.oos_pct({"efficiency": 0.62}) == 62.0


def test_oos_pct_from_pbo_inverted():
    assert cb.oos_pct({"pbo": 20.0}) == 80.0


def test_oos_pct_none_when_nothing_recorded():
    assert cb.oos_pct({}) is None
    assert cb.oos_pct({"method": "cpcv"}) is None


def test_status_from_verdict():
    assert cb.status_from_verdict("READY") == "READY"
    assert cb.status_from_verdict("MARGINAL") == "MARGINAL"
    assert cb.status_from_verdict("NOT READY") == "NOT READY"
    assert cb.status_from_verdict(None) == "DEVELOPING"
    assert cb.status_from_verdict("") == "DEVELOPING"


# ---------------------------------------------------------------------------
# Promotion gating -- the core "not automatic" requirement
# ---------------------------------------------------------------------------

def test_fresh_strategy_starts_at_candidate_and_cannot_promote():
    state = cb.evaluate_promotion({})
    assert state.stage == "candidate"
    assert state.next_stage == "validated"
    assert state.can_promote is False
    assert any(not r.met for r in state.requirements)


def test_candidate_to_validated_requires_backtest_and_validation():
    md = {"last_run": {"eval_pass_probability": 80}}
    state = cb.evaluate_promotion(md)
    # Backtested, but no deeper validation run yet -> still blocked.
    assert state.can_promote is False
    req_by_key = {r.key: r for r in state.requirements}
    assert req_by_key["backtested"].met is True
    assert req_by_key["validated_run"].met is False


def test_promote_strategy_blocked_reports_specific_reason():
    stype, fname = _seed()
    ok, message, new_stage = cb.promote_strategy(stype, fname)
    assert ok is False
    assert new_stage is None
    assert "backtest" in message.lower() or "Has at least one recorded backtest" in message


def test_promote_strategy_succeeds_once_requirements_met():
    stype, fname = _seed()
    library.save_strategy_metadata(stype, fname, {
        "last_run": {"eval_pass_probability": 80},
        "last_validation": {"method": "cpcv", "efficiency": 0.6},
    }, merge=True)
    ok, message, new_stage = cb.promote_strategy(stype, fname)
    assert ok is True
    assert new_stage == "validated"

    # Never automatic: re-reading metadata directly must show the same
    # stage this call just wrote -- nothing else could have set it.
    meta = library.load_strategy_metadata(stype, fname)
    assert meta["promotion"]["stage"] == "validated"
    assert "validated" in meta["promotion"]["history"]


def test_cannot_skip_a_stage():
    stype, fname = _seed()
    library.save_strategy_metadata(stype, fname, {"promotion": {"stage": "candidate", "history": {}}}, merge=True)
    # Even with strong metrics, jumping straight to champion_candidate
    # without first passing through validated is not offered -- promote
    # only ever advances one stage at a time.
    library.save_strategy_metadata(stype, fname, {
        "last_run": {"eval_pass_probability": 95, "first_payout_probability": 90, "verdict": "READY"},
        "last_validation": {"method": "cpcv", "efficiency": 0.9},
        "last_champion_check": {"verdict": "READY", "t58_score": 95},
    }, merge=True)
    ok, _msg, new_stage = cb.promote_strategy(stype, fname)
    assert ok is True
    assert new_stage == "validated"  # one stage, not champion_candidate


def test_validated_to_champion_candidate_requires_thresholds():
    stype, fname = _seed()
    library.save_strategy_metadata(stype, fname, {
        "promotion": {"stage": "validated", "history": {"validated": 1.0}},
        "last_run": {"eval_pass_probability": 40, "first_payout_probability": 40, "verdict": "MARGINAL"},
        "last_validation": {"method": "cpcv", "efficiency": 0.3},
        "last_champion_check": {"verdict": "MARGINAL"},
    }, merge=True)
    ok, message, new_stage = cb.promote_strategy(stype, fname)
    assert ok is False
    assert new_stage is None
    assert "60%" in message  # eval threshold called out specifically


def test_champion_candidate_requires_ready_verdict():
    stype, fname = _seed()
    library.save_strategy_metadata(stype, fname, {
        "promotion": {"stage": "champion_candidate", "history": {"champion_candidate": 1.0}},
        "last_run": {"eval_pass_probability": 85, "verdict": "MARGINAL"},
        "last_champion_check": {"verdict": "MARGINAL"},
    }, merge=True)
    ok, _msg, new_stage = cb.promote_strategy(stype, fname)
    assert ok is False
    assert new_stage is None


def test_forward_testing_requires_minimum_elapsed_days():
    stype, fname = _seed()
    library.save_strategy_metadata(stype, fname, {
        "promotion": {"stage": "forward_testing", "history": {"forward_testing": __import__("time").time()}},
    }, merge=True)
    ok, _msg, new_stage = cb.promote_strategy(stype, fname)
    assert ok is False  # just entered -- 0 days elapsed
    assert new_stage is None


def test_production_ready_is_terminal():
    stype, fname = _seed()
    library.save_strategy_metadata(stype, fname, {"promotion": {"stage": "production_ready", "history": {}}}, merge=True)
    ok, message, new_stage = cb.promote_strategy(stype, fname)
    assert ok is False
    assert new_stage is None
    assert "final stage" in message.lower()


def test_demote_strategy_always_allowed():
    stype, fname = _seed()
    library.save_strategy_metadata(stype, fname, {"promotion": {"stage": "champion_candidate", "history": {}}}, merge=True)
    ok, message = cb.demote_strategy(stype, fname, "validated")
    assert ok is True
    meta = library.load_strategy_metadata(stype, fname)
    assert meta["promotion"]["stage"] == "validated"


# ---------------------------------------------------------------------------
# Board listing + "strongest validated candidate" (replaces highest-Sharpe)
# ---------------------------------------------------------------------------

def test_list_board_includes_every_saved_strategy():
    _seed("a.py")
    _seed("b.py")
    rows = cb.list_board("python")
    names = {r["filename"] for r in rows}
    assert names == {"a.py", "b.py"}
    for r in rows:
        assert r["promotion"]["stage"] == "candidate"  # untouched strategies


def test_strongest_validated_candidate_ignores_unvalidated_high_metrics():
    # A candidate-stage strategy with a spectacular raw metric...
    stype, fname_hot = _seed("hot_but_unvalidated.py")
    library.save_strategy_metadata(stype, fname_hot, {
        "last_run": {"eval_pass_probability": 99, "first_payout_probability": 99},
    }, merge=True)

    # ...and a modest but actually-validated strategy.
    _stype2, fname_mod = _seed("modest_but_validated.py")
    library.save_strategy_metadata(stype, fname_mod, {
        "promotion": {"stage": "validated", "history": {"validated": 1.0}},
        "last_run": {"eval_pass_probability": 65, "first_payout_probability": 60},
        "last_validation": {"method": "cpcv", "efficiency": 0.55},
    }, merge=True)

    rows = cb.list_board("python")
    winner = cb.strongest_validated_candidate(rows)
    assert winner is not None
    assert winner["filename"] == fname_mod  # NOT the unvalidated hot one


def test_strongest_validated_candidate_none_when_nothing_validated():
    _seed("only_a_candidate.py")
    rows = cb.list_board("python")
    assert cb.strongest_validated_candidate(rows) is None


def test_five_question_snapshot_prefers_current_strategy():
    stype, fname = _seed("tracked.py")
    library.save_strategy_metadata(stype, fname, {
        "promotion": {"stage": "validated", "history": {"validated": 1.0}},
        "last_run": {"eval_pass_probability": 55, "first_payout_probability": 50, "verdict": "MARGINAL"},
        "last_validation": {"method": "cpcv", "efficiency": 0.3},
        "last_champion_check": {"verdict": "MARGINAL"},
    }, merge=True)
    _seed("untracked.py")

    rows = cb.list_board("python")
    snap = cb.five_question_snapshot({"strategy_name": "tracked", "instrument": "ES"}, rows)
    assert snap is not None
    assert snap["what"] == "tracked"
    assert snap["where"] == "Validated"
    assert snap["is_it_working"] == "MARGINAL"
    assert "OOS" in snap["why"] or "degradation" in snap["why"].lower()


def test_five_question_snapshot_none_on_empty_library():
    assert cb.five_question_snapshot(None, []) is None
