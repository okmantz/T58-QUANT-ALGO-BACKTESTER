"""v9.15 (Owen's 2026-10-10 work order) regression tests for WS-A/B/C.

WS-A: "Quick Optimize for a lower-drawdown parameter set" on a finished
Full Pipeline crashed with "must be called with a dataclass type or
instance" -- app.web.server's module-level ``RiskConfig`` FACTORY
function shadows the imported dataclass, and /recovery/quick-optimize
passed that factory to ``_dataclass_from_dict``. These tests drive the
real endpoint the way the finished-pipeline page does (plus the sibling
guided pages and the risk-fidelity of the recovery actions).

WS-B: the job page's "EVAL PASS PROBABILITY" tile read 90.0% (the
any-attempt CHAIN figure) while the report's per-attempt gate metric
was 8.67%. Tiles/report/dashboard/job JSON now all read the per-attempt
headline from the ONE canonical derivation
(app.reports.mc_headline.monte_carlo_headline); the chain figure may
appear only labeled "any attempt in chain". The report also labels the
historical-backtest scope (development vs holdout vs full window).

WS-C: Optimize and Validate strategy dropdowns now group by
Manual/Python/PineScript/MQL5 exactly like the Test page.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig as RealRiskConfig
from app.monte_carlo.engine import MonteCarloResult
from app.prop.simulator import PropRules, simulate_account
from app.reports.generator import build_report
from app.reports.mc_headline import monte_carlo_headline
from app.strategy.manual import ManualStrategy
from app.web import server
from app.web.job_manager import JOB_MANAGER
from app.web.server import app

OWEN_REPORT = (
    "/home/hatch/workspace/user/files/full_pipeline_91c1a13b87f4_1_pcmb.json"
)


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------

def _owen_mc(mc_dd) -> MonteCarloResult:
    """Monte Carlo result carrying Owen's actual NQ-run shape: chain
    (any-attempt) 90.05% vs per-attempt gate metric 8.67%."""
    return MonteCarloResult(
        n_simulations=10000,
        evaluation_pass_probability=float(mc_dd["evaluation_pass_probability"]),
        first_payout_probability=float(mc_dd["first_payout_probability"]),
        failure_before_payout_probability=30.0,
        multiple_payout_probability=10.0,
        median_days_to_pass=5.0,
        median_days_to_first_payout=7.0,
        average_days_to_first_payout=8.0,
        median_return_pct=1.0,
        mean_return_pct=0.5,
        expected_payout=120.0,
        median_payout=100.0,
        total_simulated_withdrawals=100,
        median_drawdown_pct=9.0,
        p95_drawdown_pct=15.0,
        worst_drawdown_pct=22.0,
        risk_of_ruin_pct=float(mc_dd.get("risk_of_ruin_pct", 95.77)),
        median_max_losing_streak=4,
        worst_max_losing_streak=9,
        per_attempt_pass_probability=float(mc_dd["per_attempt_pass_probability"]),
        per_attempt_payout_probability=float(
            mc_dd["per_attempt_payout_probability"]),
        per_attempt_pass_ci95=tuple(mc_dd["per_attempt_pass_ci95"]),
        per_attempt_payout_ci95=(
            tuple(mc_dd["per_attempt_payout_ci95"])
            if mc_dd.get("per_attempt_payout_ci95")
            else None),
        total_independent_attempts=int(mc_dd["total_independent_attempts"]),
    )


def _fake_fp_result(mc: MonteCarloResult) -> SimpleNamespace:
    stats = SimpleNamespace(
        net_profit=-21270.27, max_drawdown_pct=31.4, win_rate=38.2)
    return SimpleNamespace(
        verdict="NOT READY",
        verdict_reasons=["Risk of ruin too high."],
        warnings=[],
        elapsed_seconds=1977.0,
        refinement_ran=True,
        refinement_skip_reason="",
        saved_library_note="",
        oos_validation_skip_reason="",
        baseline_bt=SimpleNamespace(trades=[object()] * 639, statistics=stats),
        final_bt=SimpleNamespace(trades=[object()] * 639, statistics=stats),
        final_mc=mc,
        # Owen's run: ruin 95.77% vs the 20% cap -- the diagnosis branch
        # that emits the "Quick Optimize for a lower-drawdown parameter
        # set" action he clicked.
        risk_of_ruin_hard_fail=True,
    )


def _seed_completed_fp_job(mc: MonteCarloResult, *, with_risk: bool) -> str:
    job_id = JOB_MANAGER.create(
        tool="Full Pipeline", instrument="EURUSD_5M_sample (28).csv")
    fields: dict = {"done": True, "result": _fake_fp_result(mc)}
    if with_risk:
        fields["risk"] = RealRiskConfig(risk_value=0.42)
        fields["rules"] = PropRules()
    JOB_MANAGER.update(job_id, **fields)
    return job_id


# ----------------------------------------------------------------------
# WS-A -- Quick Optimize launch + sibling next steps
# ----------------------------------------------------------------------

def test_dataclass_from_dict_rejects_the_factory_function_clearly():
    """Regression pin for the exact crash: the module-level RiskConfig
    FACTORY (not the dataclass) must produce a clear error, not
    dataclasses' \"must be called with a dataclass type or instance\"."""
    with pytest.raises(TypeError) as excinfo:
        server._dataclass_from_dict(server.RiskConfig, {})
    assert "_BaseRiskConfig" in str(excinfo.value)

    ok = server._dataclass_from_dict(
        server._BaseRiskConfig, {"risk_value": 0.5})
    assert ok.risk_value == 0.5


def test_recovery_quick_optimize_from_completed_pipeline(monkeypatch):
    """Owen's exact click: 'Quick Optimize for a lower-drawdown
    parameter set' on the finished Full Pipeline (his final_parameters
    + empty risk/rules dicts). The route must create a Quick Optimize
    job, and the job must inherit the risk values from the payload."""
    report = json.load(open(OWEN_REPORT))
    captured: dict = {}

    def _stub_runner(job_id, df, strategy, risk, rules, cfg,
                     cancel_event, dataset_label):
        captured.update(job_id=job_id, risk=risk, rules=rules, cfg=cfg)
        JOB_MANAGER.finish(job_id, result=None)
        server.HEAVY_JOB_GUARD.release(server.JOB_QUICK_OPTIMIZE)

    monkeypatch.setattr(server, "_run_quickopt_job", _stub_runner)

    client = app.test_client()
    body = {
        "strategy_ref": {
            "source": "pipeline",
            "display_name": "NQ1! Opening Range Breakout (v9.14)",
            "final_source_type": "manual",
            "final_config": report["final_parameters"],
            "final_code_text": None,
            "final_code_extension": None,
        },
        "dataset_label": "EURUSD_5M_sample (28).csv",
        "risk": {"risk_value": 0.42},
        "rules": {},
        "qo_cfg": {"fitness_metric": "eval_pass_probability",
                   "ga_population": 32, "ga_generations": 12, "n_folds": 4},
    }
    resp = client.post("/recovery/quick-optimize", json=body)
    assert resp.status_code == 200, resp.get_data(as_text=True)
    payload = resp.get_json()
    assert payload["ok"] is True, payload
    job_id = payload["job_id"]
    assert JOB_MANAGER.get(job_id) is not None
    # the real route rebuilt strategy + dataset and converted the
    # risk/rules dicts into real dataclass instances carrying the
    # payload's values (0.42 -- the pre-fix code silently discarded
    # these into dataclass defaults AND crashed first)
    assert captured["risk"].risk_value == 0.42
    assert captured["risk"].__class__ is server._BaseRiskConfig
    # the Quick Optimize job page renders
    assert client.get(f"/quick-optimize/job/{job_id}").status_code == 200

def test_pipeline_status_summary_is_per_attempt():
    """Job page JSON: eval/first-payout are the per-attempt gate
    metric (8.67%, not the 90.05% chain), chain rides along labeled."""
    mc = _owen_mc(json.load(open(OWEN_REPORT))["monte_carlo"])
    job_id = _seed_completed_fp_job(mc, with_risk=False)
    client = app.test_client()
    body = client.get(f"/full-pipeline/job/{job_id}/status.json").get_json()
    s = body["summary"]
    assert s["eval_pass_probability"] == pytest.approx(8.67, abs=0.01)
    assert s["any_attempt_eval_pass_probability"] == pytest.approx(
        90.05, abs=0.01)
    assert s["eval_pass_ci95"][0] == pytest.approx(8.56, abs=0.01)


def test_pipeline_recovery_actions_inherit_run_risk():
    """The Quick Optimize action on a completed pipeline carries the
    run's OWN risk setup (0.42), not dataclass defaults."""
    mc = _owen_mc(json.load(open(OWEN_REPORT))["monte_carlo"])
    job_id = _seed_completed_fp_job(mc, with_risk=True)
    client = app.test_client()
    body = client.get(f"/full-pipeline/job/{job_id}/status.json").get_json()
    rec = body["recovery"]
    assert rec is not None and rec.get("actions")
    qo = [a for a in rec["actions"] if a.get("kind") == "quick_optimize"]
    assert qo, rec
    assert qo[0]["params"]["risk"]["risk_value"] == pytest.approx(0.42)


@pytest.mark.parametrize("path", [
    "/search", "/evolution", "/full-pipeline", "/validate-simple", "/champion-simple",
])
def test_guided_next_step_pages_render(path):
    """Sibling next-step landings out of a completed pipeline (Validate
    hub send, Champion send, re-run) must serve cleanly. (The Deploy
    landing /deploy-live sits behind the license gate -- it 302s to
    /activate in an unlicensed test client, by design, not a crash.)"""
    client = app.test_client()
    assert client.get(path).status_code == 200


# ----------------------------------------------------------------------
# WS-B -- headline derivation + report scopes
# ----------------------------------------------------------------------

def test_monte_carlo_headline_uses_per_attempt_as_gate_metric():
    mc = json.load(open(OWEN_REPORT))["monte_carlo"]
    head = monte_carlo_headline(mc)
    assert head["eval_pass_probability"] == pytest.approx(8.67, abs=0.01)
    assert head["eval_pass_ci95"][0] == pytest.approx(8.56, abs=0.01)
    assert head["eval_pass_ci95"][1] == pytest.approx(8.79, abs=0.01)
    assert head["any_attempt_eval_pass_probability"] == pytest.approx(
        90.05, abs=0.01)
    assert head["risk_of_ruin_pct"] == pytest.approx(95.77, abs=0.01)
    assert head["eval_pass_basis"] == "per_attempt"


def test_monte_carlo_headline_legacy_fallback():
    """Pre-attempt-split results (no independent attempts) fall back
    to the chain figure -- the only number they ever had."""
    head = monte_carlo_headline({
        "evaluation_pass_probability": 42.0,
        "first_payout_probability": 30.0,
        "risk_of_ruin_pct": 12.0,
        "total_independent_attempts": 0,
        "per_attempt_pass_probability": 0.0,
    })
    assert head["eval_pass_probability"] == 42.0
    assert head["eval_pass_basis"] == "chain_fallback"


def _sma_strategy():
    return ManualStrategy({
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 15,
        "take_profit_pips": 30,
    })


def _flat_fx_df(n=240, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.10 + np.cumsum(rng.normal(0, 0.00005, n))
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.0003,
        "low": price - 0.0003, "close": price, "volume": 100.0,
    })


def test_report_scope_labels_and_headline():
    """Report JSON: full-window backtest is labeled as such; the
    development/holdout split rides in `headline`; the MC headline is
    per-attempt with the chain figure under its explicit name."""
    risk = RealRiskConfig()
    bt = run_backtest(_flat_fx_df(), _sma_strategy(), risk)
    rules = PropRules()
    pnls = [t.pnl for t in bt.trades]
    dates = [t.entry_time for t in bt.trades]
    single_run = simulate_account(pnls, dates, rules)
    mc = _owen_mc(json.load(open(OWEN_REPORT))["monte_carlo"])
    holdout = {
        "in_sample_statistics": {"net_profit": -16724.34, "total_trades": 639,
                                 "win_rate": 38.0, "max_drawdown_pct": 30.0,
                                 "profit_factor": 0.9},
        "holdout_statistics": {"net_profit": -4545.93, "total_trades": 222,
                               "win_rate": 36.0, "max_drawdown_pct": 12.0,
                               "profit_factor": 0.85},
        "in_sample_trades": 639,
        "holdout_trades": 222,
        "holdout_frac": 0.25,
        "in_sample_period": ("2023-01-01", "2024-06-30"),
        "holdout_period": ("2024-07-01", "2024-12-31"),
    }
    report = build_report(
        "NQ1! Opening Range Breakout (v9.14)", "manual", "NQ1!", "5m",
        ("2023-01-01", "2024-12-31"), bt, rules, single_run, mc,
        holdout_comparison=holdout, risk_config=risk,
        verdict="NOT READY", verdict_reasons=["Risk of ruin too high."],
    )
    hb = report["historical_backtest"]
    assert hb["scope"] == "full_window"
    assert "development + holdout" in hb["scope_label"]
    head = report["headline"]
    assert head["development"]["scope"] == "development_window"
    assert head["development"]["trades"] == 639
    assert head["development"]["net_profit"] == pytest.approx(-16724.34)
    assert head["holdout"]["trades"] == 222
    assert head["monte_carlo"]["eval_pass_probability"] == pytest.approx(
        8.67, abs=0.01)
    assert head["monte_carlo"]["any_attempt_eval_pass_probability"] == (
        pytest.approx(90.05, abs=0.01))

    # no-holdout variant: label must say so, holdout block absent
    report2 = build_report(
        "s", "manual", "ES", "15m", ("2024-01-01", "2024-12-31"),
        bt, rules, single_run, mc, risk_config=risk)
    assert "no holdout reserved" in report2["historical_backtest"]["scope_label"]
    assert report2["headline"]["holdout"] is None


# ----------------------------------------------------------------------
# WS-C -- grouped strategy dropdowns
# ----------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/optimize-simple", "/validate-simple"])
def test_strategy_dropdowns_grouped_like_test_page(path):
    client = app.test_client()
    body = client.get(path).get_data(as_text=True)
    assert client.get(path).status_code == 200
    labels = ["<optgroup label=\"Manual\">", "<optgroup label=\"Python\">",
              "<optgroup label=\"PineScript\">", "<optgroup label=\"MQL5\">"]
    positions = [body.find(lbl) for lbl in labels]
    assert all(p >= 0 for p in positions), body[:2000]
    assert positions == sorted(positions)  # Manual, Python, PineScript, MQL5
