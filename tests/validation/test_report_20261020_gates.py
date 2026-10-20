"""Acceptance tests for the 2026-10-20 report: statistics, gates, ladder, costs, two-stage optimizer, edge portfolio."""
from __future__ import annotations

import inspect
import math
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from app.validation import edge_stats as es


def _trades(r_values, start="2022-01-03", direction=None, risk=10.0, size=1.0, seed=0):
    ts = pd.bdate_range(start, periods=len(r_values))
    rng = np.random.default_rng(seed)
    out = []
    for t, r in zip(ts, r_values):
        d = int(direction) if direction is not None else int(rng.choice([-1, 1]))
        entry = 100.0
        out.append(NS(entry_time=t, exit_time=t + pd.Timedelta(hours=3), direction=d, entry_price=entry, exit_price=entry + d * r * risk,
                      initial_risk=risk, size=size, pnl=float(r) * risk * size, commission=0.0))
    return out


# ------------------------------------------------------------------------------------------- power and SPRT planning
def test_report_power_numbers():
    assert es.trades_needed(0.21, 1.3) in range(236, 240)
    assert es.trades_needed(0.30, 1.3) in range(115, 120)
    assert es.trades_needed(0.10, 1.3) in range(1040, 1050)
    assert es.trades_needed(-0.1) is None


def test_sprt_expected_stopping_points_and_months():
    b = es.sprt_boundaries(0.21, 1.3)
    assert 100 <= b.expected_trades_h0 <= 106
    assert 143 <= b.expected_trades_h1 <= 149
    one = es.months_to_decide(b.expected_trades_h1, 6.7, 1)
    four = es.months_to_decide(b.expected_trades_h1, 6.7, 4)
    assert 15 <= one <= 25 and 4 <= four <= 6.5


def test_sprt_error_rates_by_simulation():
    rng = np.random.default_rng(2)

    def decide(mu):
        x = rng.normal(mu, 1.3, 400)
        for k in range(10, 401, 5):
            d = es.sprt_decision(x[:k], 0.21)["decision"]
            if d != "CONTINUE":
                return d
        return "CONTINUE"

    h0 = [decide(0.0) for _ in range(150)]
    h1 = [decide(0.30) for _ in range(150)]
    assert h0.count("ACCEPT_H1") / 150 <= 0.08            # 5% false-accept (with sampling noise)
    assert h1.count("ACCEPT_H1") / 150 >= 0.80


# --------------------------------------------------------------------------------------------- pooled OOS gate power
def _worlds(mu, n_worlds=60, folds=5, per_fold=60, seed=5):
    rng = np.random.default_rng(seed)
    passed = 0
    for _ in range(n_worlds):
        fr = [list(rng.normal(mu, 1.3, per_fold)) for _ in range(folds)]
        fd = [list(pd.bdate_range("2023-01-02", periods=per_fold) + pd.Timedelta(days=k * 120)) for k in range(folds)]
        passed += es.pooled_oos_edge(fr, fd, n_boot=300).passed
    return passed / n_worlds


def test_pooled_gate_passes_real_edge_and_not_noise():
    assert _worlds(0.25) >= 0.90          # +0.25R, 300 pooled trades
    assert _worlds(0.0) <= 0.10
    assert _worlds(-0.2) == 0.0


def test_pooled_gate_not_proven_is_not_a_fail():
    rng = np.random.default_rng(3)
    fr = [list(rng.normal(0.12, 1.3, 40)) for _ in range(3)]
    res = es.pooled_oos_edge(fr, None, n_boot=300)
    assert res.status in ("NOT_PROVEN", "PASS")
    if res.status == "NOT_PROVEN":
        assert res.trades_needed_more and res.trades_needed_more > 0


def test_small_folds_are_merged():
    fr = [[0.1] * 10, [0.2] * 10, [0.3] * 40]
    merged, _ = es.merge_small_folds(fr, None, 30)
    assert all(len(g) >= 30 for g in merged) and sum(len(g) for g in merged) == 60


# ---------------------------------------------------------------------------------------------------- ICIR harness
def test_edge_ic_gate_symmetric_edge_passes_and_losing_shorts_fail():
    from app.validation.icir import edge_ic_gate
    rng = np.random.default_rng(9)
    ok = bad = 0
    for _ in range(60):
        good = _trades(rng.normal(0.30, 1.3, 300), seed=int(rng.integers(1e6)))
        ok += edge_ic_gate(good).ok
        # long side earns, short side loses -> negative IC on the raw move, must never pass
        r = rng.normal(0.0, 1.3, 300)
        dirs = rng.choice([-1, 1], 300)
        r = np.where(dirs == 1, r + 0.3, r - 0.5)
        ts = _trades(r, seed=1)
        for t, d, rr in zip(ts, dirs, r):
            t.direction = int(d)
            t.exit_price = t.entry_price + int(d) * rr * t.initial_risk
        bad += edge_ic_gate(ts).ok
    assert ok / 60 >= 0.80
    assert bad / 60 <= 0.05


# ------------------------------------------------------------------------------------------------------ PBO on noise
def test_pbo_is_about_half_on_pure_noise_with_mean_r():
    from app.validation.cpcv import compute_pbo
    from app.backtest.risk import RiskConfig
    idx = pd.date_range("2022-01-03", periods=4000, freq="30min")
    df = pd.DataFrame({"timestamp": idx, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1})
    rng = np.random.default_rng(4)
    cand = []
    for _ in range(12):
        n = 800
        sel = np.sort(rng.choice(len(idx) - 5, n, replace=False))
        cand.append([NS(entry_time=idx[i], exit_time=idx[i + 2], direction=1, entry_price=100, exit_price=100, size=1.0, pnl=float(rng.normal(0, 13)),
                        initial_risk=10.0, commission=0.0) for i in sel])
    res = compute_pbo(df, [{"source_type": "manual", "config": {}}] * 12, RiskConfig(), n_groups=8, n_test_groups=2, max_paths=None,
                      metric="mean_r", precomputed_trades=cand)
    assert 0.25 <= res.pbo <= 0.75


# ------------------------------------------------------------------------------------------------------ edge decay
def test_decay_flags_a_fading_edge_but_not_a_stable_one():
    n = 600
    ts = list(pd.bdate_range("2019-01-01", periods=n))
    rng = np.random.default_rng(6)
    fading = list(np.linspace(0.8, -0.4, n) + rng.normal(0, 0.5, n))
    stable = list(rng.normal(0.25, 0.8, n))
    assert es.rolling_mean_r_slope(fading, ts)["flag"] is True
    assert es.rolling_mean_r_slope(stable, ts)["flag"] is False
    years = es.edge_by_year(fading, ts)
    assert years[0]["mean_r"] > years[-1]["mean_r"]


# ------------------------------------------------------------------------------------------ shrinkage and sizing
def test_shrinkage_pulls_small_samples_toward_family_and_sizes_from_lower_bound():
    from app.scoring.edge_bounds import shrunk_expectancy_bound
    r = list(np.random.default_rng(1).normal(0.4, 1.3, 100))
    out = shrunk_expectancy_bound(r, 0.0, 0.15)
    assert out["shrunk_mean"] < out["mean"]
    assert out["shrunk_lb"] < out["lb"] + 0.35
    assert es.size_from_lower_bound(-0.01) == 0.0
    assert es.size_from_lower_bound(0.2) == pytest.approx(0.5)
    assert es.size_from_lower_bound(5.0) == 1.5


# ----------------------------------------------------------------------------------------------- effective trials
def test_effective_trials_collapse_near_duplicates():
    from app.validation.effective_trials import effective_trial_count, scale_trial_count
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=300)
    base = [pd.Series(rng.normal(0, 1, 300), index=idx) for _ in range(3)]
    streams = [b + rng.normal(0, 0.05, 300) for b in base for _ in range(10)]
    assert effective_trial_count(streams)["n_clusters"] == 3
    assert scale_trial_count(3000, streams)["effective_trials"] == 300


def test_trial_ledger_ages_old_entries(tmp_path, monkeypatch):
    from app.validation import trial_ledger as tl
    monkeypatch.setattr(tl, "_PATH", tmp_path / "ledger.json")
    now = 2_000_000_000.0
    tl.record("idea", 1000, now=now - 730 * 86400)       # two years old = one half-life
    tl.record("idea", 100, now=now)
    assert tl.raw_total("idea") == 1100
    assert 590 <= tl.total("idea", now=now) <= 610
    assert tl.total("idea", half_life_days=None) == 1100


# ---------------------------------------------------------------------------------------------------------- costs
def test_cost_model_ticks_not_pips():
    from app.data.cost_model import compute_tick_cost_ladder, resolve_cost
    nq = resolve_cost("NQ_5m.csv")
    assert nq.spread_ticks == 1 and nq.slippage_ticks == 1
    assert nq.spread_pips == pytest.approx(0.25)                         # 1 tick = 0.25 points, NOT the legacy 1.0 pip
    assert nq.round_trip_dollars() == pytest.approx(2 * 2 * 5.0 + 4.20)
    assert nq.stress(2.0).round_trip_dollars() == pytest.approx(2 * 4 * 5.0 + 4.20)   # commission not scaled
    tr = _trades([0.3] * 50, risk=10.0, size=2.0)
    lad = compute_tick_cost_ladder(tr, nq, (1.0, 2.0))
    assert lad[1]["net_profit"] < lad[0]["net_profit"]
    assert lad[0]["net_profit"] - lad[1]["net_profit"] == pytest.approx(50 * 2 * 2 * 5.0 * 2)


def test_web_forms_no_longer_default_to_legacy_pips():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "app" / "web"
    src = (root / "server.py").read_text(encoding="utf-8")
    assert not re.search(r'spread_pips=float\(form\.get\("spread_pips", 1\.0\)', src)
    assert not re.search(r'slippage_pips=float\(form\.get\("slippage_pips", 0\.5\)', src)
    assert 'name="spread_pips" value="1.0"' not in (root / "templates" / "index.html").read_text(encoding="utf-8")


def test_new_instrument_specs_present():
    from app.data.instrument_specs import get_any_instrument_spec, guess_any_instrument_symbol, micro_equivalent
    for s in ("AUDUSD", "USDJPY", "XAGUSD", "BRENT", "ETHUSD", "US30", "SPX500", "US2000", "XCUUSD", "XPTUSD", "SI", "SIL"):
        assert get_any_instrument_spec(s) is not None, s
    assert guess_any_instrument_symbol("UKOIL_15m") == "BRENT"
    assert micro_equivalent("SI").symbol == "SIL"


# ------------------------------------------------------------------------------------- sizing preflight and lint
def test_sizing_skip_limit_is_25_percent():
    from app.validation.preflight import baseline_preflight
    assert inspect.signature(baseline_preflight).parameters["max_skip_ratio"].default == 0.25


def test_orb_equality_on_float_time_of_day_is_flagged():
    import json
    from pathlib import Path
    from app.strategy.manual import IMPOSSIBLE_CONDITION_MARKER, validate_bounded_conditions
    cfg = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "nq_orb_manual.json").read_text(encoding="utf-8"))
    cfg = cfg.get("config", cfg)
    warns = validate_bounded_conditions(cfg)
    assert any(IMPOSSIBLE_CONDITION_MARKER in w and "0.627" in w for w in warns)
    price_eq = {"entry_conditions": {"long": [{"left": {"type": "close"}, "operator": "==", "right": {"type": "value", "value": 18432.37}}]}}
    assert validate_bounded_conditions(price_eq)


# -------------------------------------------------------------------------------------------------- ladder / verdict
def test_ladder_stages():
    from app.orchestration.verdict_ladder import ladder_stage
    assert ladder_stage("READY", "HIGH CONFIDENCE").stage == "FORWARD-TEST CANDIDATE"
    assert ladder_stage("MARGINAL", "FORWARD-TEST CANDIDATE").stage == "FORWARD-TEST CANDIDATE"
    assert ladder_stage("MARGINAL", "INSUFFICIENT EVIDENCE").stage == "LEAD"
    assert ladder_stage("NOT READY", "NOT WORTH A FEE").stage == "REJECT"
    assert ladder_stage("READY", "HIGH CONFIDENCE", pooled_status="FAIL").stage == "REJECT"
    assert ladder_stage("READY", "HIGH CONFIDENCE", forward_state="EVALUATE").stage == "EVALUATE"
    assert ladder_stage("READY", "HIGH CONFIDENCE", forward_state="FUNDED").stage == "FUNDED"


def test_pipeline_defaults_use_new_metric_and_cap_mode():
    from app.orchestration.full_pipeline import FullPipelineConfig
    c = FullPipelineConfig()
    assert c.oos_check_metric == "mean_r" and c.per_attempt_pass_mode == "cap" and c.ev_hard_bar and c.dsr_use_effective_trials


# ----------------------------------------------------------------------------------------------------- two-stage
def test_stage_a_fitness_ranks_real_edge_above_noise_and_rejects_thin_samples():
    from app.optimize.two_stage import edge_fitness_from_trades, stage_a_summary
    rng = np.random.default_rng(3)
    good, noise = _trades(rng.normal(0.3, 1.2, 400)), _trades(rng.normal(0.0, 1.2, 400))
    assert edge_fitness_from_trades(good) > edge_fitness_from_trades(noise)
    assert edge_fitness_from_trades(good[:10]) == float("-inf")
    assert stage_a_summary(good).survives and not stage_a_summary(noise).survives


def test_stage_a_metric_registered_in_refinement():
    from app.optimize.refinement import FITNESS_METRICS
    assert "edge_lb_r" in FITNESS_METRICS


def test_stage_b_rescale_keeps_r_multiples():
    from app.optimize.two_stage import rescale_trades_to_risk
    from app.backtest.execution import Trade
    t = Trade(entry_time=pd.Timestamp("2024-01-02"), exit_time=pd.Timestamp("2024-01-02 01:00"), direction=1, entry_price=100, exit_price=110,
              size=3.0, pnl=300.0, pnl_pct=0, exit_reason="signal", commission=3.0, equity_after=0, initial_risk=10.0)
    out = rescale_trades_to_risk([t], 100000, 0.5)[0]
    assert out.pnl / (out.initial_risk * out.size) == pytest.approx(10.0)    # R unchanged
    assert out.initial_risk * out.size == pytest.approx(500.0)               # 0.5% of 100k


# --------------------------------------------------------------------------------------------------- edge portfolio
def test_edge_portfolio_admission_rules():
    from app.portfolio.edge_portfolio import EdgeCandidate, evaluate_book
    mk = lambda mu, seed, n=300: _trades(np.random.default_rng(seed).normal(mu, 1.2, n), seed=seed)  # noqa: E731
    cands = [EdgeCandidate("A", "NQ", mk(0.35, 1), null_beaten=True), EdgeCandidate("A_clone", "ES", mk(0.35, 1), null_beaten=True),
             EdgeCandidate("B", "GC", mk(0.35, 2), null_beaten=True), EdgeCandidate("C", "CL", mk(0.35, 3), null_beaten=None),
             EdgeCandidate("D", "RTY", mk(-0.2, 4), null_beaten=True)]
    v = evaluate_book(cands, n_boot=300)
    names = {a.name for a in v.admitted}
    assert {"A", "B"} <= names
    assert "A_clone" not in names and "C" not in names and "D" not in names
    assert v.stage in ("FORWARD-TEST CANDIDATE", "LEAD")
    assert v.sprt and v.sprt["months_to_decide_h1"] < v.sprt["months_to_decide_single_edge"]


# ---------------------------------------------------------------------------------------------- scorecard wiring
def test_scorecard_prefers_generalization_score():
    from app.scoring import t58_scorecard as sc
    src = inspect.getsource(sc)
    assert 'getattr(walk_forward_result, "generalization_score"' in src
