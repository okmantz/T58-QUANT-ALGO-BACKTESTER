"""T58 Prop Firm Rulebook v2: the published numbers, reproduced by app.validation.rulebook."""
import statistics
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.validation import rulebook as R


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_register_path", lambda: tmp_path / "trial_register.json")
    monkeypatch.setattr(R, "_sheet_path", lambda: tmp_path / "rules_sheet.json")
    return tmp_path


def _trades(n=900, edge=0.12, b=2.0, per_day=3, seed=3, years=3.5, commission=0.02):
    """Binary +bR / -1R trades on distinct session days, planned risk $150 (initial_risk 1.0 x size 150)."""
    rng = np.random.default_rng(seed)
    p = (edge + 1.0) / (b + 1.0)
    days = pd.bdate_range("2023-01-02", periods=int(years * 252))
    out = []
    i = 0
    for d in days:
        for k in range(per_day):
            if i >= n:
                break
            win = rng.random() < p
            r = b if win else -1.0
            t0 = d + pd.Timedelta(hours=9 + 2 * k)
            out.append(SimpleNamespace(entry_time=t0, exit_time=t0 + pd.Timedelta(minutes=30), direction=1 if k % 2 == 0 else -1,
                                       pnl=r * 150.0 - commission * 150.0, initial_risk=1.0, size=150.0, commission=commission * 150.0))
            i += 1
    return out


# ---- core math, against the Rulebook's published tables ---------------------------------------------------------------------
def test_g3_required_edge_table_matches_rulebook():
    book = {400: (0.050, 0.112, 0.137, 0.165, 0.196), 600: (0.050, 0.101, 0.121, 0.144, 0.169),
            1000: (0.050, 0.089, 0.105, 0.123, 0.142), 1500: (0.050, 0.082, 0.095, 0.109, 0.125)}
    for row in R.g3_table():
        want = book[row["trades"]]
        got = tuple(row[f"N={k}"] for k in (1, 3, 5, 10, 25))
        assert all(abs(g - w) <= 0.001 for g, w in zip(got, want)), (row["trades"], got, want)


def test_deflation_example_is_plus_0_014r():
    assert R.deflation_example(0.15, 600, 50) == pytest.approx(0.014, abs=0.001)


def test_trade_floor_is_derived_and_never_below_400():
    assert R.trade_floor(1, 0.10) == 400
    assert R.trade_floor(10, 0.15) > 400                     # more looks need more trades
    assert R.trade_floor(10, 0.04) >= 10 ** 9                # an edge under the floor can never clear it


def test_z_bar_and_expected_max_are_monotone():
    assert [round(R.z_bar(n), 2) for n in (1, 5, 10, 25, 50)] == [1.96, 2.58, 2.81, 3.09, 3.29]
    assert R.expected_max_z(1) == 0.0 and R.expected_max_z(10) < R.expected_max_z(50)


def test_edge_and_breakeven():
    assert R.edge_per_trade(0.40, 2, 0.05) == pytest.approx(0.15)
    assert R.breakeven_win_rate(2, 0.05) == pytest.approx(0.35)


def test_effective_bets_examples():
    assert round(R.effective_bets(5, 0.3), 1) == 2.3
    assert round(R.effective_bets(4, 0.8), 1) == 1.2


# ---- layer 2: the recentred kill line --------------------------------------------------------------------------------------
def _kill_rate(edge, n=1500, trials=300, seed=5):
    rng = np.random.default_rng(seed)
    w, l = 1.95, -1.05
    p = (edge - l) / (w - l)
    hits = 0
    for _ in range(trials):
        k = R.LLRKillSwitch()
        for v in np.where(rng.random(n) < p, w, l):
            k.add(v)
            if k.killed_at_trade:
                hits += 1
                break
    return hits / trials


def test_v2_kill_line_is_centred_at_plus_0_05r():
    assert R.LLRKillSwitch().status()["centre_r"] == pytest.approx(0.05)
    assert _kill_rate(0.10) < 0.15                    # Rulebook: 8% (v1 killed a +0.10R edge 35% of the time)
    assert 0.25 < _kill_rate(0.05) < 0.55             # Rulebook: 38%
    assert _kill_rate(0.0) > 0.75                     # Rulebook: 82% of dead-edge paths


def test_sprt_monitor_surfaces_rulebook_kill():
    from app.forward_test.sprt_monitor import SprtMonitor
    m = SprtMonitor(h1_r=0.10)
    st = None
    for _ in range(700):
        st = m.add(-0.4)
    assert st["rulebook_llr"]["killed"] and st["halted"] and st["ladder_state"] == "REJECT"


def test_sprt_monitor_evaluate_after_20_live_sim_days():
    from app.forward_test.sprt_monitor import SprtMonitor
    m = SprtMonitor(h1_r=0.30)
    st = None
    for i in range(25):
        st = m.add(0.05, day=f"2026-11-{i + 1:02d}")
    assert st["live_sim_days"] == 25 and st["ladder_state"] == "EVALUATE"
    assert SprtMonitor(h1_r=0.30).status()["ladder_state"] == "FORWARD-TEST CANDIDATE"


# ---- G7: the attempt score ---------------------------------------------------------------------------------------------------
def test_attempt_score_scale_retire_and_cap():
    assert R.g7_capped_attempts([True, True]).data["decision"] == "SCALE"             # +3.0
    assert R.g7_capped_attempts([False, False]).data["decision"] == "CONTINUE"         # 2 x -0.9 = -1.8, above the -2.2 line
    assert R.g7_capped_attempts([False, False, False]).data["decision"] == "RETIRE"    # 3 x -0.9 = -2.7
    a = R.AttemptScore()
    for _ in range(3):
        a.add(False)
    assert a.decision() == "RETIRE" and a.score == pytest.approx(-2.7)
    mixed = R.AttemptScore()
    for paid in (True, False, False, True, False, False):                              # drifts without ever reaching a bar
        mixed.add(paid)
    st = mixed.status()
    assert st["attempts"] <= R.ATTEMPT_MAX
    both = R.g7_capped_attempts([True, False, False, True, False, False])
    assert both.data["decision"] in ("RETIRE", "SCALE", "CONTINUE")
    assert R.g7_capped_attempts([]).data["fees_cap"] == pytest.approx(6 * 172.0)


# ---- G0: rules sheet ---------------------------------------------------------------------------------------------------------
def _full_sheet(today):
    cell = lambda v: {"value": v, "source": "firm help centre", "checked_on": today.isoformat()}  # noqa: E731
    sheet = {k: cell("x") for k, _ in R.RULES_SHEET_FIELDS}
    sheet["product"] = cell("pro")
    sheet.update({k: cell(True) for k, _ in R.RULES_SHEET_CONFIRMATIONS})
    return sheet


def test_g0_one_blank_stops_everything(store):
    today = date(2026, 11, 7)
    assert R.g0_rules_sheet(_full_sheet(today), today).passed
    sheet = _full_sheet(today)
    del sheet["payout_cap"]
    g = R.g0_rules_sheet(sheet, today)
    assert not g.passed and "payout_cap" not in g.data["missing"] and "Payout: cap per payout ($)" in g.data["missing"]


def test_g0_stale_unsourced_and_unconfirmed_fail(store):
    today = date(2026, 11, 7)
    sheet = _full_sheet(today)
    sheet["fee"]["checked_on"] = "2026-09-01"
    assert not R.g0_rules_sheet(sheet, today).passed
    sheet = _full_sheet(today)
    sheet["automation_confirmed"]["value"] = False
    assert not R.g0_rules_sheet(sheet, today).passed
    sheet = _full_sheet(today)
    sheet["product"]["value"] = ""
    assert not R.g0_rules_sheet(sheet, today).passed


def test_rules_sheet_from_preset_leaves_payout_terms_blank(store):
    sheet = R.rules_sheet_from_preset("lucid_50k")
    assert sheet["max_loss"]["value"] == pytest.approx(2000.0) and sheet["profit_target"]["value"] == pytest.approx(3000.0)
    assert "payout_cap" not in sheet and "fee" not in sheet and "automation_confirmed" not in sheet


# ---- G2: the trial register --------------------------------------------------------------------------------------------------
def test_register_enforces_three_looks_per_family_and_breadth(store):
    kw = dict(family="opening-drive", mechanism="m", counterparty="late fade traders", sign="long", instruments=["MNQ", "MES", "M2K"],
              sealed={"MNQ": "2026-03-01", "MES": "2026-03-01", "M2K": "2026-03-01"}, today=date(2026, 11, 7))
    assert R.register_trial(variants=2, **kw)["ok"]
    assert R.register_trial(variants=1, **kw)["ok"]
    r = R.register_trial(variants=1, **kw)
    assert not r["ok"] and any("cap" in p for p in r["problems"])
    assert R.total_looks() == 3
    thin = R.register_trial(**{**kw, "family": "other", "instruments": ["MNQ"], "mechanisms": 1})
    assert not thin["ok"] and any("Breadth" in p for p in thin["problems"])
    noseal = R.register_trial(**{**kw, "family": "third", "sealed": {}, "instruments": ["MGC", "MCL", "MNQ"]})
    assert not noseal["ok"] and any("sealed window" in p for p in noseal["problems"])
    assert not R.register_trial(**{**kw, "family": "x", "counterparty": ""})["ok"]


def test_archive_keeps_the_looks_and_a_revival_adds_more(store):
    kw = dict(family="f", mechanism="m", counterparty="c", sign="long", instruments=["MNQ", "MES", "M2K"],
              sealed={"MNQ": "2026-03-01", "MES": "2026-03-01", "M2K": "2026-03-01"})
    r1 = R.register_trial(**kw)
    assert R.archive_trial(r1["row"]["id"], "G3", "E_plan +0.012R")
    assert R.total_looks() == 1 and R.load_register()["rows"][0]["archived"]
    assert R.register_trial(**kw)["ok"] and R.total_looks() == 2           # a revival counts as new looks
    g2 = R.g2_preregistration(family="f")
    assert g2.passed


# ---- G3 / G4 / G5 on trades --------------------------------------------------------------------------------------------------
def test_g3_passes_a_real_edge_and_fails_noise(store):
    good = R.g3_development_evidence(_trades(n=1500, edge=0.14, per_day=1, years=6), n_looks=3, neighbours=[0.9, 0.85, 0.95], lookahead_ok=True)
    assert good.data["e_plan"] > 0.05, good.data
    assert good.passed, good.failing_number()
    noise = R.g3_development_evidence(_trades(n=1500, edge=0.0, seed=9, per_day=1, years=6), n_looks=10, neighbours=[1, 1], lookahead_ok=True)
    assert not noise.passed and any(c.name.startswith("Deflated edge") for c in noise.failing)


def test_g3_pending_checks_cannot_pass_a_gate(store):
    g = R.g3_development_evidence(_trades(n=1500, edge=0.14, per_day=1, years=6), n_looks=1)          # no neighbours, no lookahead result
    assert not g.passed and any(c.ok is None for c in g.checks)


def test_g3_needs_three_calendar_years_and_400_trades(store):
    short = R.g3_development_evidence(_trades(n=200, edge=0.3, years=0.6), n_looks=1, neighbours=[1], lookahead_ok=True)
    assert not short.passed
    names = {c.name: c.ok for c in short.checks}
    assert any(k.startswith("Sample") and v is False for k, v in names.items())
    assert any(k.startswith("No year above") and v is False for k, v in names.items())


def test_g3_detects_edge_that_lives_in_three_trades(store):
    ts = _trades(n=900, edge=-0.02, seed=4)
    for t in ts[:3]:
        t.pnl = 150.0 * 40
    g = R.g3_development_evidence(ts, n_looks=1, neighbours=[1], lookahead_ok=True)
    assert any(c.name == "Positive with the top 3 trades removed" and c.ok is False for c in g.checks)


def test_g4_day_behaviour_reads_real_days(store):
    from app.prop.lifecycle_v2 import V2Rules
    g = R.g4_day_behaviour(_trades(n=900, edge=0.12), V2Rules())
    assert g.data["median_trades_per_day"] == 3 and g.data["p_day_loss_pct"] <= 1.0
    assert any(c.name.startswith("Median 2 to 4") and c.ok for c in g.checks)


def test_g5_lifecycle_uses_v2_sizing_and_wilson_bars(store):
    from app.prop.lifecycle_v2 import V2Rules
    strong = R.g5_lifecycle(_trades(n=2000, edge=0.20, per_day=3), V2Rules(), n_paths=4000, n_looks=1)
    assert strong.data["sim"]["sizing"] == "v2" and strong.passed, strong.failing_number()
    weak = R.g5_lifecycle(_trades(n=2000, edge=0.0, per_day=3), V2Rules(), n_paths=4000, n_looks=1)
    assert not weak.passed


def test_pipeline_stops_at_first_failure_and_never_reaches_evaluate_from_a_backtest(store):
    from app.prop.lifecycle_v2 import V2Rules
    r = R.evaluate_pipeline(_trades(n=1500, edge=0.20), V2Rules(), n_looks=1, neighbours=[1, 1], lookahead_ok=True, n_paths=4000,
                            skip_gates=("G0", "G2", "G6", "G7"))
    assert r["stage"] in (R.VERDICT_CANDIDATE, R.VERDICT_LEAD)               # never EVALUATE / FUNDED from backtest evidence alone
    bad = R.evaluate_pipeline(_trades(n=1500, edge=0.0, seed=8), V2Rules(), n_looks=20, neighbours=[1], lookahead_ok=True, n_paths=2000,
                              skip_gates=("G0", "G2", "G6", "G7"))
    assert bad["stopped_at"] == "G3" and bad["gates"]["G4"]["skipped"] and bad["gates"]["G5"]["skipped"] and bad["failing_number"]
    assert bad["stage"] in (R.VERDICT_REJECT, R.VERDICT_LEAD)


def test_ladder_stage_requires_all_gates_for_each_rung():
    ok = lambda g: R.GateResult(g, g, checks=[R.Check("c", True)])  # noqa: E731
    all_ok = {g: ok(g) for g in ("G0", "G1", "G2", "G3", "G4", "G5", "G6")}
    assert R.ladder_stage(all_ok) == R.VERDICT_EVALUATE
    assert R.ladder_stage({k: v for k, v in all_ok.items() if k != "G6"}) == R.VERDICT_CANDIDATE
    assert R.ladder_stage({k: all_ok[k] for k in ("G0", "G1", "G2")}) == R.VERDICT_LEAD
    assert R.ladder_stage({}) == R.VERDICT_REJECT


def test_full_ladder_demotes_on_failed_rulebook_gate():
    from app.orchestration.verdict_ladder import ladder_stage
    rb = {"gates": {"G3": {"passed": False, "skipped": False, "name": "Development evidence", "failing_number": "E_plan +0.01R"}}}
    res = ladder_stage("READY", "HIGH CONFIDENCE", rulebook=rb)
    assert res.stage == "LEAD" and any("RULEBOOK v2 G3" in r for r in res.reasons)
    assert ladder_stage("READY", "HIGH CONFIDENCE").stage == "FORWARD-TEST CANDIDATE"
