"""v9.6 web redesign: lifecycle sidebar, the three simplified pages
(Optimize / Validate / Champion pickers), validate dispatch, and the
Risk Sweep dead-field wiring (A3a). Light + headless: full runs are
stubbed at the import sites server.py actually calls.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.web.server as server


@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


# ---------------------------------------------------------------- (a) pages

def test_optimize_simple_page_has_goal_picker(client):
    r = client.get("/optimize-simple")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert 'name="goal"' in body
    for goal in ("eval", "payout", "robust"):
        assert f'value="{goal}"' in body, goal


def test_validate_simple_page_has_check_checkboxes(client):
    r = client.get("/validate-simple")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    for name in ("check_monte_carlo", "check_cpcv", "check_pbo", "check_walk_forward",
                 "check_sensitivity", "check_robustness", "check_regime"):
        assert f'name="{name}"' in body, name


def test_champion_simple_page_has_board_and_actions(client):
    r = client.get("/champion-simple")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Champion board" in body
    assert 'formaction="/champion-simple/promote"' in body
    assert 'formaction="/champion-simple/rerun"' in body
    for name in ("check_monte_carlo", "check_cpcv", "check_pbo"):
        assert f'name="{name}"' in body, name


# ------------------------------------------------------------- (b) sidebar

def test_sidebar_lifecycle_order_and_account_order(client):
    r = client.get("/full-pipeline")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    labels = [f'<span class="label">{n}</span>' for n in ("Create", "Test", "Optimize", "Validate", "Champion", "Deploy", "Strategy Graveyard")]
    positions = [body.index(lbl) for lbl in labels]
    assert positions == sorted(positions), list(zip(labels, positions))
    # Notification settings now live inside Account settings (no separate sidebar entry)
    assert "/settings/notifications" not in body
    assert "/settings/api-keys" in body
    # New simplified pages are linked from the lifecycle groups
    for href in ("/optimize-simple", "/validate-simple", "/champion-simple"):
        assert href in body, href


# ------------------------------------------------- (c) validate dispatch

def test_validate_simple_dispatches_only_ticked_checks(client, monkeypatch):
    import pandas as pd

    df = pd.DataFrame({"open": [1.0] * 30, "high": [1.0] * 30, "low": [1.0] * 30,
                       "close": [1.0] * 30, "volume": [1.0] * 30})
    monkeypatch.setattr(server, "_resolve_dataset",
                        lambda form, files: (df, "tiny.csv", "", None))
    monkeypatch.setattr(server, "load_strategy_text", lambda t, f: "code")
    monkeypatch.setattr(server, "build_strategy_from_code",
                        lambda t, code, **k: SimpleNamespace(name="Stub"))
    stats = SimpleNamespace(net_profit=10.0, win_rate=50.0, max_drawdown_pct=1.0)
    monkeypatch.setattr(server, "run_backtest",
                        lambda df_, s, risk: SimpleNamespace(trades=[], statistics=stats))

    called = {}

    def fake_cpcv(*args, **kwargs):
        called["cpcv"] = True
        return SimpleNamespace(n_paths=3, metric="profit_factor", mean_oos_metric=1.2,
                               median_oos_metric=1.1, mean_is_metric=1.3, is_robust=True)

    def fake_mc(*args, **kwargs):  # must NOT be called: not ticked
        called["mc"] = True
        raise AssertionError("monte_carlo dispatched but not ticked")

    import app.validation.runner as _vrunner  # the dispatcher now lives here
    monkeypatch.setattr(_vrunner, "run_cpcv", fake_cpcv)
    monkeypatch.setattr(_vrunner, "run_monte_carlo", fake_mc)

    r = client.post("/validate-simple/start", data={
        "strategy_pick": "python::stub.py",
        "existing_dataset": "tiny.csv",
        "check_cpcv": "on",
        "initial_balance": "100000", "account_size": "100000",
    })
    from tests.validate_job_helper import wait_for_validate_job
    job = wait_for_validate_job(client, r)
    assert not job["error"], job["error"]
    assert called.get("cpcv") is True
    assert "mc" not in called
    assert list(job.get("checks") or []) == ["CPCV"], job.get("checks")
    # The "how this was simulated" explanation travels with the job
    assert job.get("simulation_text")


# --------------------------------------------- (d) risk sweep reads sizing

def test_risk_sweep_form_sizing_mode_survives():
    """A3a: the Risk Sweep form renders sizing_mode but the handler used to
    drop it ('skip' default). The construction the handler now performs
    must keep the typed fit_stop."""
    from app.backtest.risk import RiskConfig
    from app.prop.simulator import PropRules
    from app.web.accuracy_form import accuracy_kwargs, harden_risk_config

    form = {"sizing_mode": "fit_stop", "max_stop_dollars": "450",
            "account_model": "prop", "intrabar_replay": "on"}
    with server.app.test_request_context("/risk-sweep/start", method="POST", data=form):
        from flask import request as _req
        kw = accuracy_kwargs(_req.form)
    risk = RiskConfig(initial_balance=100_000.0, **kw)
    assert risk.sizing_mode == "fit_stop"
    hardened = harden_risk_config(risk, prop_rules=PropRules(account_size=100_000.0), form=form)
    assert hardened.sizing_mode == "fit_stop"
    assert hardened.max_stop_dollars == 450.0
    assert hardened.account_model == "prop"
