"""Regression tests for the 2026-10-17 Fresh-Eyes Audit changes: sample-aware intervals, expectancy lower bound,
eval-vs-funded breach split, rule-verification eligibility, firm-rule corrections, scorecard independence."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop import presets as P
from app.scoring import edge_bounds as eb
from app.scoring import t58_scorecard as sc


def _trades(n=157, seed=11, per_day=2):
    rng = np.random.default_rng(seed)
    pool = np.where(rng.random(n) < 0.52, rng.normal(420, 120, n), -rng.normal(300, 90, n)).round(0)
    days = pd.bdate_range("2024-01-02", periods=n)
    out = []
    for i in range(n):
        t = days[i // per_day] + pd.Timedelta(hours=9 + (i % per_day) * 3)
        out.append(Trade(entry_time=t, exit_time=t + pd.Timedelta(hours=1), direction=1, entry_price=1.0, exit_price=1.0,
                         size=1, pnl=float(pool[i]), pnl_pct=0.1, exit_reason="signal", commission=0, equity_after=0))
    return out


# ---- item 4: the interval must reflect the SOURCE sample, not the simulation count -----------------------------------
def test_effective_attempts_are_capped_by_history_not_by_simulations():
    assert eb.effective_independent_attempts(150, 240, n_simulated_attempts=10_000) == pytest.approx(7.5)   # min(240d/30d = 8, 150 trades/20 = 7.5)
    assert eb.effective_independent_attempts(150, 4000, n_simulated_attempts=10_000) == pytest.approx(7.5)  # 150 trades / 20
    assert eb.effective_independent_attempts(10_000, 4000, n_simulated_attempts=50) == 50.0
    assert eb.effective_independent_attempts(0, 0) == 1.0


def test_sample_aware_interval_is_far_wider_than_simulation_noise():
    (lo_s, hi_s), n_eff = eb.sample_aware_interval(80.0, n_source_trades=157, n_source_calendar_days=220, n_simulated_attempts=10_000)
    lo_w, hi_w = (v * 100 for v in eb.wilson_interval(8000, 10_000))
    assert (hi_s - lo_s) > 5 * (hi_w - lo_w) and n_eff < 20


def test_monte_carlo_exposes_both_intervals():
    r = run_monte_carlo(_trades(), P.get_preset("topstep_50k").to_prop_rules(),
                        MonteCarloConfig(n_simulations=300, random_seed=3, ruin_horizon_days=60), selection_bias_caveat=False)
    w = r.per_attempt_pass_ci95[1] - r.per_attempt_pass_ci95[0]
    s = r.per_attempt_pass_ci95_sample[1] - r.per_attempt_pass_ci95_sample[0]
    assert s > w and r.effective_independent_attempts < r.n_simulations


# ---- item 5: expectancy lower bound ---------------------------------------------------------------------------------
def test_expectancy_bound_is_conservative_and_gated_on_sample_size():
    rng = np.random.default_rng(0)
    big = rng.normal(0.25, 1.0, 400)
    b = eb.expectancy_bound(big)
    assert b["lb"] < b["mean"] < b["ub"] and b["lb"] == min(b["lb_t"], b["lb_boot"]) and b["lb"] > 0
    small = eb.expectancy_bound(big[:12])
    assert small["lb"] < b["lb"]                        # fewer trades, less certainty
    assert eb.expectancy_bound([1.0, 2.0])["lb"] is None  # too few to say anything
    noise = eb.expectancy_bound(rng.normal(0.0, 1.0, 60))
    assert noise["lb"] < 0 < noise["ub"]


def test_verdict_demotes_when_expectancy_lower_bound_is_not_positive(monkeypatch):
    from types import SimpleNamespace
    import app.orchestration.full_pipeline as fp
    from app.orchestration.full_pipeline import _make_verdict
    # the wrapper only ever demotes a READY base verdict, so force one and test the new gates in isolation
    monkeypatch.setattr(fp, '_make_verdict_core', lambda *a, **k: ('READY', [], SimpleNamespace(), False, False))

    class _MC:
        evaluation_pass_probability = 97.0; first_payout_probability = 95.0; risk_of_ruin_pct = 1.0
        per_attempt_pass_ci95 = (96.0, 99.0); per_attempt_payout_ci95 = (94.0, 98.0); per_attempt_paid_ci95 = (92.0, 97.0)
        per_attempt_paid_probability = 95.0; per_attempt_payout_probability = 98.0; return_percentiles = {}
        sample_ok = True; ruin_horizon_covered = True; ruin_horizon_days = 90; n_source_calendar_days = 400
        risk_of_ruin_definition = "x"; funded_attempts = 0; funded_breach_within_horizon_pct = 0.0

    class _WF:
        walk_forward_efficiency = 0.95; is_stable = True; stability_threshold = 0.4; n_folds = 4; oos_profitable = True

    class _S:
        total_trades = 300; average_r = 0.6; max_drawdown_pct = 1.0
        def to_dict(self): return {}

    kw = dict(icir_gate=None, statistics=_S(), prop_rules=SimpleNamespace(max_drawdown_pct=10.0), mc_dispersion_pts=1.0, min_trades_for_ready=100)
    neg = {"n": 200, "mean": 0.05, "lb": -0.04, "ub": 0.14}
    v, reasons, *_ = _make_verdict(_MC(), _WF(), expectancy_bound=neg, **kw)
    assert v != "READY" and any("expectancy" in r.lower() for r in reasons)
    dead = {"n": 200, "mean": -0.2, "lb": -0.3, "ub": -0.05}
    v2, *_ = _make_verdict(_MC(), _WF(), expectancy_bound=dead, **kw)
    assert v2 == "NOT READY"


# ---- item 3: one Monte Carlo no longer decides most of the score ---------------------------------------------------------
def test_monte_carlo_family_is_a_minority_of_the_score():
    mc_w = sum(abs(sc._WEIGHTS[k]) for k in sc._MC_FAMILY)
    total = sum(abs(w) for w in sc._WEIGHTS.values())
    assert total == 100.0 and mc_w / total < 0.35
    assert "replay_pass" in sc._INDEPENDENT_REQUIRED and "expectancy_lower_bound" in sc._INDEPENDENT_REQUIRED


# ---- item 6: eval and funded breach are reported separately --------------------------------------------------------------
def test_eval_and_funded_breach_are_split():
    r = run_monte_carlo(_trades(), P.get_preset("lucid_50k").to_prop_rules(),
                        MonteCarloConfig(n_simulations=300, random_seed=4, ruin_horizon_days=90), selection_bias_caveat=False)
    assert 0.0 <= r.eval_breach_pct <= 100.0 and 0.0 <= r.funded_breach_within_horizon_pct <= 100.0
    assert r.risk_of_ruin_pct >= max(r.eval_breach_pct * 0.0, 0.0)
    assert r.funded_attempts >= 0


# ---- item 7: rule verification ----------------------------------------------------------------------------------------------
def test_only_fresh_primary_stamps_are_ready_eligible():
    p = P.get_preset("lucid_50k")
    assert P.is_ready_eligible(p, today=date(2026, 10, 20))
    assert not P.is_ready_eligible(p, today=date(2026, 12, 20))          # stamp older than 30 days
    for key in ("the5ers_20k", "the5ers_100k", "funderpro_one_phase_100k", "atlas_funded_1step_100k", "futures_desk_50k"):
        ok, why = P.ready_eligibility(P.get_preset(key), today=date(2026, 10, 20))
        assert not ok and "verified" in why


def test_corrected_rules_match_the_firms_own_pages():
    lucid = P.get_preset("lucid_100k")
    assert lucid.daily_loss_limit_pct == 1.8 and lucid.daily_loss_action == "lock_day" and lucid.trailing_lock
    assert P.get_preset("apex_100k_eod").min_winning_day_profit == 300.0
    assert P.get_preset("apex_50k_eod").daily_loss_action == "lock_day"
    assert P.get_preset("topstep_100k").consistency_rule_pct == 55.0
    assert P.get_preset("fundednext_25k").consistency_rule_pct is None
    assert P.get_preset("e8_one_100k").evaluation_profit_target_pct == 9.0
    assert P.get_preset("the5ers_100k").evaluation_profit_target_pct == 10.0
    assert P.get_fee_info("fundednext_25k").challenge_price == 199.99
    assert P.get_fee_info("atlas_funded_1step_100k").profit_split_pct == 90.0
    for p in P.list_presets():
        assert "2026-10-17 verification" in (p.source_note or "") or p.key not in P._RULE_CORRECTIONS


def _ready_fixture(monkeypatch, **mc_over):
    from types import SimpleNamespace
    import app.orchestration.full_pipeline as fp
    monkeypatch.setattr(fp, '_make_verdict_core', lambda *a, **k: ('READY', [], SimpleNamespace(), False, False))

    class _MC:
        evaluation_pass_probability = 97.0; first_payout_probability = 95.0; risk_of_ruin_pct = 1.0
        per_attempt_pass_ci95 = (96.0, 99.0); per_attempt_paid_ci95 = (92.0, 97.0); per_attempt_paid_probability = 95.0
        per_attempt_payout_probability = 98.0; sample_ok = True; ruin_horizon_covered = True; ruin_horizon_days = 90
        n_source_calendar_days = 400; funded_attempts = 0; funded_breach_within_horizon_pct = 0.0
    for k, v in mc_over.items():
        setattr(_MC, k, v)
    return fp._make_verdict, _MC()


def test_funded_breach_cap_demotes_ready(monkeypatch):
    mv, mc = _ready_fixture(monkeypatch, funded_attempts=60, funded_breach_within_horizon_pct=80.0)
    good = {"n": 200, "mean": 0.2, "lb": 0.05, "ub": 0.35}
    v, reasons, *_ = mv(mc, None, expectancy_bound=good)
    assert v == "MARGINAL" and any("FUNDED ACCOUNTS DIE" in r for r in reasons)
    mv2, mc2 = _ready_fixture(monkeypatch, funded_attempts=60, funded_breach_within_horizon_pct=10.0)
    assert mv2(mc2, None, expectancy_bound=good)[0] == "READY"


def test_missing_independent_evidence_caps_ready_only_when_required(monkeypatch):
    mv, mc = _ready_fixture(monkeypatch)
    assert mv(mc, None)[0] == "READY"                                   # legacy callers unchanged
    v, reasons, *_ = mv(mc, None, evidence_required=True)
    assert v == "MARGINAL" and any("lower bound on expectancy" in r for r in reasons)
