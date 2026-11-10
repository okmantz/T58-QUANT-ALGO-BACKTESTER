"""/payout-planner: the page renders, validates its three steps, and shows decisions built from (stubbed) Monte Carlo results."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from app.web import payout_planner_routes as routes
from app.web.server import app


@pytest.fixture
def client():
    return app.test_client()


def _fake_mc(p=60.0, q=70.0):
    return SimpleNamespace(per_attempt_pass_probability=p, per_attempt_payout_probability=q,
                           evaluation_pass_probability=p, first_payout_probability=p * q / 100.0,
                           median_days_to_pass=9.0, median_days_to_first_payout=14.0, median_payout=2500.0)


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(routes, "import_csv", lambda path: SimpleNamespace(is_valid=True, errors=[], dataframe=pd.DataFrame({"x": [1]})))
    monkeypatch.setattr(routes, "load_strategy_text", lambda t, n: "code")
    monkeypatch.setattr(routes, "build_strategy_from_spec", lambda spec, tmp_dir=None: object())
    monkeypatch.setattr(routes, "run_backtest", lambda df, strat, risk: SimpleNamespace(trades=[object()]))
    monkeypatch.setattr(routes, "run_monte_carlo", lambda trades, rules, cfg: _fake_mc())
    monkeypatch.setattr(routes, "harden_risk_config", lambda risk, prop_rules=None, form=None, **k: risk)


def test_the_page_loads_with_lucid_50k_and_100k_ticked_by_default(client):
    html = client.get("/payout-planner").get_data(as_text=True)
    assert "Payout planner" in html
    assert 'value="lucid_50k"' in html and 'value="lucid_100k"' in html
    assert html.index('value="lucid_50k"') < html.index('value="lucid_25k"')       # the two defaults come first
    assert html.count("checked") >= 2


def test_the_planner_is_linked_from_the_test_section(client):
    assert "/payout-planner" in client.get("/full-pipeline").get_data(as_text=True)


def test_each_missing_step_gets_a_plain_message(client):
    assert "Step 1" in client.post("/payout-planner/run", data={}).get_data(as_text=True)
    assert "Step 2" in client.post("/payout-planner/run", data={"strategy_pick": "python::x.py"}).get_data(as_text=True)
    assert "Step 3" in client.post("/payout-planner/run", data={"strategy_pick": "python::x.py", "existing_dataset": "a.csv"}).get_data(as_text=True)


def test_a_full_run_shows_the_decision_numbers_for_both_accounts(client, stubbed):
    r = client.post("/payout-planner/run", data={
        "strategy_pick": "python::x.py", "existing_dataset": "ES1!/futures_ES.F_1m.parquet",
        "accounts": ["lucid_50k", "lucid_100k"], "price_lucid_100k": "250", "lanes": "2", "n_sims": "300", "risk_value": "0.5"})
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "The short answer" in html
    assert "Lucid Trading - $50k Evaluation" in html and "Lucid Trading - $100k Evaluation" in html
    assert "Evaluations to a first payout" in html and "2.4" in html            # 1 / (0.6 * 0.7) = 2.38
    assert "What if you run 2 evaluations at once?" in html
    assert "days faster" in html


def test_an_account_without_a_known_price_asks_for_one_instead_of_guessing(client, stubbed):
    html = client.post("/payout-planner/run", data={
        "strategy_pick": "python::x.py", "existing_dataset": "a.csv", "accounts": ["lucid_100k"], "lanes": "1"}).get_data(as_text=True)
    assert "enter price" in html and "NEEDS A PRICE" in html


def test_a_negative_price_is_refused(client, stubbed):
    html = client.post("/payout-planner/run", data={
        "strategy_pick": "python::x.py", "existing_dataset": "a.csv", "accounts": ["lucid_50k"], "price_lucid_50k": "-5"}).get_data(as_text=True)
    assert "cannot be negative" in html


def test_one_failing_account_does_not_sink_the_others(client, stubbed, monkeypatch):
    calls = {"n": 0}

    def flaky(trades, rules, cfg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        return _fake_mc()

    monkeypatch.setattr(routes, "run_monte_carlo", flaky)
    html = client.post("/payout-planner/run", data={
        "strategy_pick": "python::x.py", "existing_dataset": "a.csv", "accounts": ["lucid_50k", "lucid_100k"], "lanes": "1"}).get_data(as_text=True)
    assert "Left out" in html and "boom" in html and "The short answer" in html
