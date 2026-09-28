"""
Tests for:
  1. Cross-Instrument Search (app.search.cross_instrument + /cross-instrument routes)
  2. Project Chat widget being collapsible (root cause: an explicit CSS `display:flex`
     overrode the `hidden` attribute, so the panel could never actually hide)
  3. Market-data-library card removed from the dashboard
  4. Evolution Lab checklists converted to the dropdown widget
"""
from __future__ import annotations

import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig, with_prop_safety_defaults
from app.data import storage
from app.monte_carlo.engine import MonteCarloConfig
from app.optimize.parameter_space import RefinementError
from app.orchestration.resource_guard import HEAVY_JOB_GUARD
from app.prop.simulator import PropRules
from app.search import cross_instrument as ci
from app.web.job_manager import JOB_MANAGER
from app.web.server import app

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "app" / "web" / "templates"
STATIC = REPO / "app" / "web" / "static"


# ---------------------------------------------------------------------------
# Synthetic multi-instrument data
# ---------------------------------------------------------------------------

def _make_market(seed: int, n: int = 2500, start: str = "2024-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="5min")
    steps = rng.normal(0, 0.0004, n) + 0.00002 * np.sin(np.arange(n) / 60.0)
    close = 1.10 + np.cumsum(steps)
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.0003, n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({
        "timestamp": idx, "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(100, 1000, n),
    })


@pytest.fixture(scope="module")
def three_markets() -> dict:
    return {"AAA": _make_market(1), "BBB": _make_market(2), "CCC": _make_market(3)}


def _account():
    rules = PropRules(account_size=100000)
    risk = with_prop_safety_defaults(RiskConfig(initial_balance=100000, pip_size=0.0001, reset_on_breach=True), rules)
    return risk, rules


FAMILIES_UNDER_TEST = ["trend_breakout", "rsi_extreme_reversion", "macd_cross_trend"]


# ---------------------------------------------------------------------------
# 1a. Search module
# ---------------------------------------------------------------------------

def test_search_scores_every_candidate_on_every_market(three_markets):
    risk, rules = _account()
    logs: list[str] = []
    result = ci.run_cross_instrument_search(
        three_markets, risk, rules, MonteCarloConfig(n_simulations=100, reset_on_breach=True),
        families=FAMILIES_UNDER_TEST, max_candidates=6, screen_mc_sims=40, finalists=3,
        progress_cb=logs.append,
    )
    assert result.markets == ["AAA", "BBB", "CCC"]
    assert result.candidates_generated == 6
    assert result.leaderboard, "expected at least one finalist"
    for cand in result.leaderboard:
        assert [m.market for m in cand.per_market] == ["AAA", "BBB", "CCC"]
        assert all(m.evaluated for m in cand.per_market)
    # finalists come back best-first on the combined score
    scores = [c.robustness_score for c in result.leaderboard]
    assert scores == sorted(scores, reverse=True)
    assert result.best is result.leaderboard[0]
    assert any("Screened" in line for line in logs)


def test_holdout_is_never_seen_by_the_search_and_is_reported(three_markets):
    risk, rules = _account()
    result = ci.run_cross_instrument_search(
        three_markets, risk, rules, MonteCarloConfig(n_simulations=100, reset_on_breach=True),
        families=["trend_breakout"], max_candidates=4, screen_mc_sims=40, finalists=2, holdout_frac=0.25,
    )
    assert result.holdout_frac == 0.25
    for cand in result.leaderboard:
        assert cand.holdout_per_market is not None
        assert [m.market for m in cand.holdout_per_market] == ["AAA", "BBB", "CCC"]
        assert cand.holdout_verdict


def test_split_is_chronological_and_disjoint(three_markets):
    warnings: list[str] = []
    dev, hold = ci._split_dev_holdout(three_markets, 0.2, warnings)
    for label, df in three_markets.items():
        assert len(dev[label]) + len(hold[label]) == len(df)
        assert dev[label]["timestamp"].max() < hold[label]["timestamp"].min()
    assert not warnings


def test_holdout_disabled_when_data_too_short():
    short = {"A": _make_market(1, n=250), "B": _make_market(2, n=250)}
    warnings: list[str] = []
    dev, hold = ci._split_dev_holdout(short, 0.2, warnings)
    assert hold == {}
    assert dev.keys() == short.keys()
    assert warnings and "Holdout disabled" in warnings[0]


def test_candidate_dead_on_one_market_is_disqualified():
    """A config that never trades on one instrument must not be averaged into a good score."""
    scores = [ci.MarketScore("A", 80.0, 40), ci.MarketScore("B", float("-inf"), 0)]
    mean, worst, disp, robust = ci._aggregate(scores, "mean_minus_dispersion")
    assert robust == float("-inf")
    cand = ci.CrossInstrumentCandidate(
        candidate_id="x", family="f", params={}, config={}, per_market=scores,
        mean_fitness=mean, worst_case_fitness=worst, dispersion=disp, robustness_score=robust,
    )
    assert not cand.is_viable
    assert ci._pick_finalists([cand], 3, 2) == []


def _cand(cid, family, score):
    return ci.CrossInstrumentCandidate(
        candidate_id=cid, family=family, params={}, config={}, per_market=[],
        mean_fitness=score, worst_case_fitness=score, dispersion=0.0, robustness_score=score,
    )


def test_finalists_are_family_diverse_but_still_filled():
    pool = [_cand("a1", "A", 9), _cand("a2", "A", 8), _cand("a3", "A", 7), _cand("b1", "B", 6), _cand("c1", "C", 5)]
    picked = ci._pick_finalists(pool, 4, max_per_family=2)
    assert [c.candidate_id for c in picked] == ["a1", "a2", "b1", "c1"]
    # not enough distinct families -> the cap relaxes rather than returning fewer finalists
    only_a = ci._pick_finalists([_cand("a1", "A", 3), _cand("a2", "A", 2), _cand("a3", "A", 1)], 3, max_per_family=1)
    assert len(only_a) == 3


def test_input_validation(three_markets):
    risk, rules = _account()
    mc = MonteCarloConfig(n_simulations=50)
    with pytest.raises(RefinementError):
        ci.run_cross_instrument_search({"only": three_markets["AAA"]}, risk, rules, mc)
    with pytest.raises(RefinementError):
        ci.run_cross_instrument_search(three_markets, risk, rules, mc, families=["not_a_family"])
    with pytest.raises(RefinementError):
        ci.run_cross_instrument_search(three_markets, risk, rules, mc, aggregation="bogus")
    with pytest.raises(RefinementError):
        ci.run_cross_instrument_search(three_markets, risk, rules, mc, fitness_metric="bogus")


def test_cancel_check_stops_the_search(three_markets):
    risk, rules = _account()
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(ci.CrossInstrumentCancelled):
        ci.run_cross_instrument_search(
            three_markets, risk, rules, MonteCarloConfig(n_simulations=50),
            families=FAMILIES_UNDER_TEST, max_candidates=8, screen_mc_sims=30, cancel_check=cancel,
        )


def test_a_market_that_raises_scores_dead_instead_of_crashing(three_markets, monkeypatch):
    risk, rules = _account()
    real = ci._evaluate

    def flaky(df, strategy, *a, **k):
        if df is three_markets["CCC"] or (len(df) and df.equals(three_markets["CCC"].iloc[: len(df)])):
            raise RuntimeError("boom")
        return real(df, strategy, *a, **k)

    monkeypatch.setattr(ci, "_evaluate", flaky)
    result = ci.run_cross_instrument_search(
        three_markets, risk, rules, MonteCarloConfig(n_simulations=50),
        families=["trend_breakout"], max_candidates=3, screen_mc_sims=30, finalists=2,
    )
    assert result.candidates_viable == 0
    assert result.best is None
    assert any("raised errors" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 1b. Routes (real Flask app, real importer, real background job)
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "get_app_base_dir", lambda: tmp_path)
    raw = storage.get_raw_data_dir()
    for instrument, seed in (("AAA", 1), ("BBB", 2), ("CCC", 3)):
        folder = raw / instrument
        folder.mkdir(parents=True, exist_ok=True)
        _make_market(seed).to_csv(folder / f"{instrument}_5M.csv", index=False)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
    shutil.rmtree(raw, ignore_errors=True)


def test_form_page_renders_with_dropdowns_and_sidebar_link(client):
    resp = client.get("/cross-instrument")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'id="ci-datasets-list"' in body and "data-t58-multiselect" in body
    assert "AAA_5M.csv" in body and "BBB_5M.csv" in body
    assert 'name="families"' in body
    assert "t58-multiselect.js" in body
    # reachable from every page's sidebar
    assert 'href="/cross-instrument"' in client.get("/dashboard").get_data(as_text=True)


def test_start_requires_two_instruments(client):
    resp = client.post("/cross-instrument/start", data={"datasets": ["AAA/AAA_5M.csv"]})
    assert resp.status_code == 400
    assert "at least 2" in resp.get_data(as_text=True)
    assert HEAVY_JOB_GUARD.active_name is None, "the heavy-job slot must be released on a rejected request"


def test_full_job_runs_end_to_end_and_result_is_valid_json(client):
    resp = client.post("/cross-instrument/start", data={
        "datasets": ["AAA/AAA_5M.csv", "BBB/BBB_5M.csv", "CCC/CCC_5M.csv"],
        "families": ["trend_breakout", "rsi_extreme_reversion"],
        "max_candidates": "4", "finalists": "2", "screen_sims": "40", "n_sims": "60", "holdout_pct": "20",
        "pip_size": "0.0001",
    })
    assert resp.status_code == 302, resp.get_data(as_text=True)[:400]
    job_id = resp.headers["Location"].rstrip("/").split("/")[-1]

    payload = None
    deadline = time.time() + 240
    while time.time() < deadline:
        r = client.get(f"/cross-instrument/job/{job_id}/status.json")
        assert r.status_code == 200
        payload = r.get_json(silent=False)  # must parse: no Infinity/NaN tokens
        if payload["done"]:
            break
        time.sleep(0.5)
    assert payload and payload["done"], "job did not finish in time"
    assert not payload.get("error"), payload.get("error")

    result = payload["result"]
    assert result["markets"] == ["AAA_5M", "BBB_5M", "CCC_5M"]
    assert result["candidates_generated"] == 4
    assert result["leaderboard"], result["warnings"]
    top = result["leaderboard"][0]
    assert top["rank"] == 1
    assert {m["market"] for m in top["per_market"]} == {"AAA_5M", "BBB_5M", "CCC_5M"}
    assert top["holdout_verdict"]

    page = client.get(f"/cross-instrument/job/{job_id}")
    assert page.status_code == 200
    assert HEAVY_JOB_GUARD.active_name is None, "slot must be released when the job ends"

    # save the winner to the Strategy Library
    saved = client.post(f"/cross-instrument/job/{job_id}/save/1")
    assert saved.status_code == 200, saved.get_data(as_text=True)
    filename = saved.get_json()["filename"]
    assert filename.endswith(".json")
    from app.strategy import library
    try:
        assert library.strategy_exists("manual", filename)
        again = client.post(f"/cross-instrument/job/{job_id}/save/1")
        assert again.status_code == 409
    finally:
        try:
            library.delete_saved_strategy("manual", filename)
        except Exception:
            pass

    assert client.post(f"/cross-instrument/job/{job_id}/save/99").status_code == 404


def test_unknown_job_is_a_clean_404(client):
    assert client.get("/cross-instrument/job/nope/status.json").status_code == 404
    assert client.post("/cross-instrument/job/nope/cancel").status_code == 404
    assert "was not found" in client.get("/cross-instrument/job/nope").get_data(as_text=True)


def test_cancel_marks_the_job(client):
    job_id = JOB_MANAGER.create(log=[], cancelled=False)
    assert client.post(f"/cross-instrument/job/{job_id}/cancel").status_code == 200
    assert JOB_MANAGER.get(job_id)["cancelled"] is True


# ---------------------------------------------------------------------------
# 2. Chat widget is collapsible
# ---------------------------------------------------------------------------

def test_widget_hidden_attribute_is_not_overridden_by_display_flex():
    css = (TEMPLATES / "_project_chat_widget.html").read_text(encoding="utf-8")
    # the original bug: `.t58-pc-panel { display:flex }` beat the browser's [hidden] rule
    assert re.search(r"\.t58-pc-panel\s*\{[^}]*display:\s*flex", css)
    assert re.search(r"#t58-pc-panel\[hidden\][^{]*\{\s*display:\s*none\s*!important", css)


def test_launcher_toggles_and_remembers_state():
    js = (STATIC / "project_chat.js").read_text(encoding="utf-8")
    assert 'addEventListener("click", togglePanel)' in js
    assert "function togglePanel" in js and "closePanel()" in js
    assert "localStorage" in js
    # starts collapsed unless it was left open
    assert "if (wasOpen()) { openPanel(); } else { closePanel(); }" in js


def test_widget_is_present_on_pages_that_include_the_sidebar(client):
    body = client.get("/dashboard").get_data(as_text=True)
    assert 'id="t58-pc-launcher"' in body and 'aria-expanded="false"' in body


# ---------------------------------------------------------------------------
# 3. Dashboard no longer shows the market data library
# ---------------------------------------------------------------------------

def test_dashboard_has_no_market_data_library(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Market data library" not in body
    assert "grouped by instrument" not in body


# ---------------------------------------------------------------------------
# 4. Evolution Lab lists are dropdowns (with Select all) and still submit the same fields
# ---------------------------------------------------------------------------

def test_evolution_families_are_a_dropdown(client):
    body = client.get("/evolution").get_data(as_text=True)
    assert 'id="evo-families-list" data-t58-multiselect' in body
    assert body.count('name="families"') > 5, "every family checkbox must still be submitted with the form"
    assert "t58SetCheckboxes('evo-families-list'" not in body, "old always-open Select All buttons should be gone"


def test_multi_instrument_evolution_lists_are_dropdowns(client):
    body = client.get("/evolution/multi-instrument").get_data(as_text=True)
    assert 'id="mi-evo-datasets-list"' in body
    assert 'id="mi-evo-families-list" data-t58-multiselect' in body
    assert 'name="datasets"' in body and 'name="families"' in body
    assert "AAA_5M.csv" in body


def test_multiselect_component_has_select_all_clear_and_search():
    js = (STATIC / "t58-multiselect.js").read_text(encoding="utf-8")
    assert '"Select all"' in js and '"Clear"' in js
    assert "Search " in js and "dispatchEvent(new Event(\"change\"" in js
    # the old helper other pages may still call must remain untouched
    assert "function t58SetCheckboxes" in (STATIC / "t58-chrome.js").read_text(encoding="utf-8")
