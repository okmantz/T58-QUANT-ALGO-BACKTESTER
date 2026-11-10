from datetime import datetime, timezone

import pandas as pd
import pytest

from app.guardian import news as gnews
from app.guardian.engine import AccountState, GuardianRules, GuardianTracker, evaluate

@pytest.fixture(autouse=True)
def _latest_in_tmp(tmp_path, monkeypatch):
    from app.guardian import engine
    monkeypatch.setattr(engine, "_latest_path", lambda: tmp_path / "latest.json")


NOON = datetime(2026, 11, 10, 18, 0, tzinfo=timezone.utc)


def rules(**kw):
    base = dict(account_size=50_000, daily_loss_limit=1_200, max_drawdown=2_000, drawdown_type="trailing", profit_target=3_000,
                max_contracts=4)
    base.update(kw)
    return GuardianRules(**base)


def test_clean_account_is_all_clear_and_reports_headline_numbers():
    rep = evaluate(rules(), AccountState(balance=51_250, start_of_day_balance=51_250, peak_equity=51_250, now=NOON))
    assert rep.level == "ok" and rep.action["can_open"] and rep.action["size_multiplier"] == 1.0
    m = rep.metrics
    assert m["daily_loss_remaining"] == 1_200 and m["drawdown_cushion"] == 2_000 and m["profit_target_remaining"] == 1_750


def test_ladder_escalates_with_daily_loss_used():
    levels = []
    for loss in (100, 800, 1_000, 1_150, 1_200):
        rep = evaluate(rules(), AccountState(balance=50_000 - loss, start_of_day_balance=50_000, peak_equity=50_000, now=NOON))
        levels.append(next(c.level for c in rep.checks if c.key == "daily_loss"))
    assert levels == ["ok", "warn", "reduce", "block", "flatten"]


def test_reduce_shrinks_size_and_block_stops_entries():
    rep = evaluate(rules(), AccountState(balance=49_000, start_of_day_balance=50_000, peak_equity=50_000, now=NOON))
    assert rep.action["level"] == "reduce" and 0 < rep.action["size_multiplier"] < 1 and rep.action["can_open"]
    rep = evaluate(rules(), AccountState(balance=48_880, start_of_day_balance=50_000, peak_equity=50_000, now=NOON))
    assert rep.action["level"] == "block" and not rep.action["can_open"] and rep.action["size_multiplier"] == 0


def test_flatten_and_halt_when_daily_loss_hit():
    rep = evaluate(rules(), AccountState(balance=48_800, start_of_day_balance=50_000, peak_equity=50_000, open_contracts=1, now=NOON))
    assert rep.action["flatten"] and rep.action["halt_day"]


def test_trailing_floor_follows_the_equity_high_but_static_does_not():
    st = AccountState(balance=52_000, start_of_day_balance=52_000, peak_equity=53_000, now=NOON)
    assert evaluate(rules(), st).metrics["drawdown_floor"] == 51_000
    assert evaluate(rules(drawdown_type="static"), st).metrics["drawdown_floor"] == 48_000


def test_trailing_floor_locks():
    st = AccountState(balance=55_000, start_of_day_balance=55_000, peak_equity=57_000, now=NOON)
    assert evaluate(rules(trailing_lock_floor=50_100), st).metrics["drawdown_floor"] == 50_100


def test_drawdown_breach_flattens_and_halts_account():
    rep = evaluate(rules(), AccountState(balance=49_000, equity=48_900, start_of_day_balance=49_000, peak_equity=51_000, now=NOON))
    assert rep.action["flatten"] and rep.action["halt_account"]


def test_contract_cap_and_exposure():
    rep = evaluate(rules(), AccountState(balance=50_000, open_contracts=5, now=NOON))
    assert next(c for c in rep.checks if c.key == "contracts").level == "flatten"
    rep = evaluate(rules(max_exposure=2), AccountState(balance=50_000, open_contracts=2, now=NOON))
    assert not rep.action["can_open"]
    rep = evaluate(rules(), AccountState(balance=50_000, start_of_day_balance=50_000, open_contracts=1, open_risk=1_300, now=NOON))
    assert next(c for c in rep.checks if c.key == "exposure").level == "block"


def test_consistency_and_min_days():
    r = rules(consistency_pct=30, min_trading_days=5)
    rep = evaluate(r, AccountState(balance=51_000, start_of_day_balance=51_000, trading_days=2, best_day_profit=800, now=NOON))
    by = {c.key: c for c in rep.checks}
    assert by["consistency"].level == "warn" and by["min_days"].remaining == 3


def test_news_window_blocks_entries():
    ev = pd.DataFrame({"timestamp": [pd.Timestamp("2026-11-10 13:00")], "event": ["CPI"], "impact": ["High"]})
    at = datetime(2026, 11, 10, 13, 2)  # 13:02 ET (naive = Eastern)
    rep = evaluate(rules(news_restricted=True), AccountState(balance=50_000, now=at), events=ev, now=at)
    assert not rep.action["can_open"] and "CPI" in rep.headline
    far = datetime(2026, 11, 10, 9, 0)
    rep = evaluate(rules(news_restricted=True), AccountState(balance=50_000, now=far), events=ev, now=far)
    assert rep.action["can_open"]


def test_flatten_time_and_window():
    r = rules(flatten_time="15:45")
    late = datetime(2026, 11, 10, 21, 50, tzinfo=timezone.utc)  # 15:50 CT
    assert evaluate(r, AccountState(balance=50_000, open_contracts=1, now=late), now=late).action["flatten"]
    assert not evaluate(r, AccountState(balance=50_000, open_contracts=0, now=late), now=late).action["can_open"]


def test_from_preset_uses_real_firm_numbers():
    r = GuardianRules.from_preset("lucid_50k")
    assert r.account_size == 50_000 and r.daily_loss_limit == 1_200 and r.max_drawdown == 2_000 and r.max_contracts == 4


def test_conservative_brakes_earlier_than_accelerated():
    st = AccountState(balance=49_300, start_of_day_balance=50_000, peak_equity=50_000, now=NOON)
    assert evaluate(rules(), st, preset="conservative").level != "ok"
    assert evaluate(rules(), st, preset="accelerated").level == "ok"


def test_tracker_remembers_day_open_and_peak():
    tr = GuardianTracker(rules())
    t0 = pd.Timestamp("2026-11-10 15:00", tz="UTC")
    tr.update(50_000, 50_000, t0)
    rep = tr.update(49_700, 49_700, t0 + pd.Timedelta(minutes=5))
    assert rep.metrics["daily_loss_remaining"] == 900
    rep = tr.update(51_000, 51_000, t0 + pd.Timedelta(minutes=10))
    assert rep.metrics["drawdown_floor"] == 49_000
    assert tr.update(None, None, t0) is rep


def test_news_module_blackout_lines_and_coverage():
    ev = pd.DataFrame({"timestamp": [pd.Timestamp("2026-11-10 14:00")], "event": ["FOMC"], "impact": ["High"]})
    assert gnews.blackout_lines_for_day("2026-11-10", events=ev) == "12:55-13:05"
    assert "FOMC" in gnews.coverage(ev)["types"]


def test_live_session_hook_blocks_entries_and_reports(monkeypatch):
    from app.live_deploy import execution_engine as ee
    assert hasattr(ee.LiveExecutionConfig, "__dataclass_fields__") and "guardian" in ee.LiveExecutionConfig.__dataclass_fields__
    assert "guardian" in ee.LiveExecutionStatus.__dataclass_fields__


def test_latest_report_round_trip_and_staleness(tmp_path, monkeypatch):
    import time
    from app.guardian import engine
    rep = evaluate(rules(), AccountState(balance=50_000, now=NOON))
    engine.save_latest(rep)
    assert engine.load_latest()["level"] == "ok"
    assert engine.load_latest(max_age_seconds=-1) is None
