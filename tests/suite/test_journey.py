from datetime import date, timedelta

import pytest

from app.journey import state
from app.prop import desk_store as ds
from app.web import journal_store


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "desk_path", lambda: tmp_path / "desk.json")
    monkeypatch.setattr(journal_store, "journal_path", lambda: tmp_path / "journal.json")
    from app.forward_test import journal as ftj
    monkeypatch.setattr(ftj, "_db_path", lambda: tmp_path / "ft.db")
    from app.guardian import engine
    monkeypatch.setattr(engine, "_latest_path", lambda: tmp_path / "latest.json")
    return tmp_path


def _promoted(monkeypatch):
    """The current strategy has been tested and promoted to champion (the board lookup is its own, separately tested, module)."""
    monkeypatch.setattr(state, "_champion_inputs", lambda st: {"found": True, "tested": True, "promoted": True, "stage_title": "Champion"})


def test_timeline_is_the_seven_requested_stages():
    assert [label for _k, label, _h in state.STAGES] == [
        "Test a strategy", "Optimize or validate a strategy", "Promote strategy to champion", "Complete forward testing",
        "Purchase evaluation", "Pass the evaluation", "Receive the first payout"]


def _recent(i):
    """A date inside the journey's rolling window whatever day the suite runs (hard-coded 2026-10-xx dates aged out of it)."""
    return (date.today() - timedelta(days=i % 27 + 1)).isoformat()


def trades(n, pnl=10.0, **kw):
    for i in range(n):
        journal_store.upsert_trade({"id": f"t{i}", "date": _recent(i), "instrument": "MES", "pnl": pnl, "source": "Forward Test",
                                    "exitTime": f"{_recent(i)}T15:00:00+00:00", **kw})


def test_fresh_user_starts_at_strategy_with_labelled_example(iso):
    r = state.compute()
    assert r["current"] == "test" and r["done"] == 0 and len(r["stages"]) == 7
    assert r["dashboard"]["illustrative"] and "Example only" in r["dashboard"]["note"]
    assert any("strategy" in a.lower() for a in r["next_actions"])


def test_forward_requirement_counts_down(iso, monkeypatch):
    from app.reports import strategy_state as ss
    monkeypatch.setattr(ss, "get_current_strategy", lambda: {"strategy_name": "alpha", "instrument": "MES", "library_type": "", "library_filename": ""})
    monkeypatch.setattr(ss, "robustness_score", lambda *a, **k: {"ran_count": 4, "passed_count": 4, "decided_count": 4})
    _promoted(monkeypatch)
    trades(15)
    r = state.compute()
    assert r["current"] == "forward" and [x["state"] for x in r["stages"][:3]] == ["done"] * 3
    assert "You need 15 more qualifying trades." in r["next_actions"]


def test_champion_stage_waits_for_the_promotion(iso, monkeypatch):
    from app.reports import strategy_state as ss
    monkeypatch.setattr(ss, "get_current_strategy", lambda: {"strategy_name": "alpha", "instrument": "MES", "library_type": "", "library_filename": ""})
    monkeypatch.setattr(ss, "robustness_score", lambda *a, **k: {"ran_count": 4, "passed_count": 4, "decided_count": 4})
    monkeypatch.setattr(state, "_champion_inputs", lambda st: {"found": True, "tested": True, "promoted": False, "stage_title": "Candidate"})
    r = state.compute()
    assert r["current"] == "champion" and [x["state"] for x in r["stages"][:2]] == ["done", "done"]
    assert any("Champion page" in a for a in r["next_actions"])


def test_violating_trades_do_not_qualify(iso):
    trades(5)
    journal_store.upsert_trade({"id": "bad", "date": _recent(1), "pnl": 5, "source": "Forward Test", "violations": ["Oversized"]})
    from app.journey.state import _forward_inputs
    f = _forward_inputs()
    assert f["trades"] == 6 and f["qualifying"] == 5


def test_drawdown_alert_when_exceeding_backtest(iso):
    trades(3, pnl=-200.0)
    state.save_settings({"backtest_max_drawdown": 300})
    r = state.compute()
    assert any("exceeded its historical drawdown" in a["text"] for a in r["alerts"])


def test_full_journey_to_first_payout(iso, monkeypatch):
    from app.reports import strategy_state as ss
    monkeypatch.setattr(ss, "get_current_strategy", lambda: {"strategy_name": "alpha", "instrument": "MES"})
    monkeypatch.setattr(ss, "robustness_score", lambda *a, **k: {"ran_count": 5, "passed_count": 5, "decided_count": 5})
    _promoted(monkeypatch)
    trades(31)
    state.save_settings({"preset_key": "lucid_50k", "forward_days_required": 1})
    a = ds.add_account({"firm": "Lucid Trading", "preset_key": "lucid_50k", "size": 50000, "fee": 172})
    assert state.compute()["current"] == "pass"
    ds.update_account(a["id"], {"status": "passed"})
    assert state.compute()["current"] == "payout"           # passed, still to be activated and paid
    ds.update_account(a["id"], {"stage": "funded", "status": "active", "live_account_id": "x"})
    assert state.compute()["current"] == "payout"
    ds.add_payout(a["id"], "2026-11-01", 500)
    r = state.compute()
    assert r["done"] == 7 and r["progress_pct"] == 100


def test_estimates_come_from_real_inputs(iso):
    state.save_settings({"preset_key": "lucid_50k"})
    ds.add_sim_run("survival", {"pass_rate": 0.5, "median_days_to_pass": 20, "median_days_to_payout": 15}, preset_key="lucid_50k")
    d = state.compute()["dashboard"]
    assert not d["illustrative"] and d["estimated_days"] == 35 and d["estimated_expenses"] == 344.0 and "All figures are illustrative" in d["note"]


def test_live_guardian_alerts(iso):
    from datetime import datetime, timezone
    from app.guardian.engine import AccountState, GuardianRules, evaluate, save_latest
    rep = evaluate(GuardianRules(account_size=50000, daily_loss_limit=1200, max_drawdown=2000), AccountState(balance=49000, start_of_day_balance=50000, peak_equity=50000, now=datetime(2026, 11, 10, 18, tzinfo=timezone.utc)))
    save_latest(rep)
    assert any("daily loss limit" in a["text"] for a in state.compute()["alerts"])


def test_settings_validation(iso):
    with pytest.raises(KeyError):
        state.save_settings({"preset_key": "nope"})
    with pytest.raises(ValueError):
        state.save_settings({"forward_trades_required": 2})
    with pytest.raises(ValueError):
        state.save_settings({"backtest_max_drawdown": -5})


def test_a_promoted_strategy_advances_the_mission_even_with_no_current_strategy(iso, monkeypatch):
    """2026-11-11 bug: tested + validated + promoted on the Champion page, yet the mission still said "Test a strategy"."""
    row = {"display_name": "orb", "funnel_rank": 4, "strength": 70.0, "verdict": "READY", "eval_pct": 80.0,
           "promotion": {"stage": "champion", "stage_title": "Champion"}, "validation": {"ran": 5}}
    from app.scoring import champion_board as cb
    monkeypatch.setattr(cb, "list_board", lambda *a, **k: [row])
    from app.reports import strategy_state as ss
    monkeypatch.setattr(ss, "get_current_strategy", lambda: None)
    r = state.compute()
    by = {s["key"]: s for s in r["stages"]}
    assert by["test"]["state"] == "done"
    assert r["current"] != "test"
    assert r["done"] >= 2


def test_an_untouched_library_strategy_is_not_progress(iso, monkeypatch):
    row = {"display_name": "fresh", "funnel_rank": 0, "strength": None, "verdict": None, "eval_pct": None,
           "promotion": {"stage": "candidate", "stage_title": "Candidate"}, "validation": {"ran": 0}}
    from app.scoring import champion_board as cb
    monkeypatch.setattr(cb, "list_board", lambda *a, **k: [row])
    from app.reports import strategy_state as ss
    monkeypatch.setattr(ss, "get_current_strategy", lambda: None)
    assert state.compute()["current"] == "test"
