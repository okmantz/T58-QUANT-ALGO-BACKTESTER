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

def test_optimize_simple_page_has_engine_checkboxes(client):
    r = client.get("/optimize-simple")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    for name in ("engine_quick_optimize", "engine_refine", "engine_risk_sweep",
                 "engine_search", "engine_evolution", "engine_multi_objective"):
        assert f'name="{name}"' in body, name


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
    assert "Promotion board" in body
    assert 'formaction="/champion-simple/promote"' in body
    assert 'formaction="/champion-simple/rerun"' in body
    for name in ("check_monte_carlo", "check_cpcv", "check_pbo"):
        assert f'name="{name}"' in body, name


# ------------------------------------------------------------- (b) sidebar

def test_sidebar_lifecycle_order_and_account_order(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    labels = ["&#9312;</span> Create", "&#9313;</span> Test", "&#9314;</span> Optimize",
              "&#9315;</span> Validate", "&#9316;</span> Champion", "&#9317;</span> Deploy",
              "&#9318;</span> Strategy Graveyard"]
    positions = [body.index(lbl) for lbl in labels]
    assert positions == sorted(positions), list(zip(labels, positions))
    # ACCOUNT group: notifications BEFORE api keys (v9.6 swap)
    assert body.index("/settings/notifications") < body.index("/settings/api-keys")
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
                        lambda t, code: SimpleNamespace(name="Stub"))
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

    monkeypatch.setattr(server, "run_cpcv", fake_cpcv)
    monkeypatch.setattr(server, "run_monte_carlo", fake_mc)

    r = client.post("/validate-simple/start", data={
        "strategy_pick": "python::stub.py",
        "existing_dataset": "tiny.csv",
        "check_cpcv": "on",
        "initial_balance": "100000", "account_size": "100000",
    })
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert called.get("cpcv") is True
    assert "mc" not in called
    assert "cpcv" in body and "mean_oos_metric" in body
    # The report box renders on the results page too
    assert "How this was simulated" in body


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
