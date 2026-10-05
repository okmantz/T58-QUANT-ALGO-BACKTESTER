"""Per-instrument pip size / contract size in Cross-Instrument Search, the
job page no longer hanging silently, and the Full Pipeline verdict/guidance
fixes (empty holdout, misleading NOT READY advice)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig, with_prop_safety_defaults
from app.data.instrument_specs import resolve_risk_per_market
from app.monte_carlo.engine import MonteCarloConfig
from app.orchestration import pipeline_guide
from app.orchestration.full_pipeline import _holdout_untestable_note
from app.prop.simulator import PropRules
from app.search import cross_instrument as ci

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "app" / "web" / "templates"


def _market(price: float, seed: int, n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="5min")
    close = price + np.cumsum(rng.normal(0, price * 0.0004, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, price * 0.0003, n))
    return pd.DataFrame({
        "timestamp": idx, "open": open_, "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread, "close": close, "volume": rng.integers(100, 1000, n),
    })


def test_each_market_gets_its_own_risk():
    dfs = {"MGC_1m_2023-2026": _market(2500, 1), "MES_5m": _market(5000, 2), "EURUSD_5m": _market(1.10, 3),
           "XYZSTOCK": _market(150, 4)}
    base = RiskConfig(pip_size=0.0001, contract_size=99.0, commission_per_trade=0.0)
    risks, report = resolve_risk_per_market(dfs, base, auto_detect=True)
    assert (risks["MGC_1m_2023-2026"].pip_size, risks["MGC_1m_2023-2026"].contract_size) == (1.0, 10.0)
    assert (risks["MES_5m"].pip_size, risks["MES_5m"].contract_size) == (1.0, 5.0)
    # unknown names: pip detected from price, the stray shared contract size is NOT carried over
    assert risks["EURUSD_5m"].pip_size == 0.0001 and risks["EURUSD_5m"].contract_size is None
    assert risks["XYZSTOCK"].pip_size == 0.01 and risks["XYZSTOCK"].contract_size is None
    # B2-4 (2026-10-04): the spec's round-turn rate now lands on
    # commission_per_contract (per-contract at settle), not the old flat
    # commission_per_trade fill.
    assert risks["MGC_1m_2023-2026"].commission_per_contract == 1.60   # filled from spec because shared commission was 0
    assert risks["MGC_1m_2023-2026"].commission_per_trade == 0.0       # no longer auto-filled (would double-charge)
    assert len(report) == 4 and all(r.describe() for r in report)


def test_explicit_commission_is_never_overwritten():
    risks, _ = resolve_risk_per_market({"MGC_1m": _market(2500, 1)}, RiskConfig(commission_per_trade=2.5))
    assert risks["MGC_1m"].commission_per_trade == 2.5


def test_auto_detect_off_keeps_shared_risk():
    base = RiskConfig(pip_size=0.5, contract_size=7.0)
    risks, report = resolve_risk_per_market({"MGC_1m": _market(2500, 1)}, base, auto_detect=False)
    assert risks["MGC_1m"] is base and report[0].pip_source == "shared setting"


def test_search_uses_the_market_specific_risk(monkeypatch):
    seen = {}

    def fake_evaluate(df, strategy, risk, *a, **k):
        seen[float(df["close"].median() // 1)] = (risk.pip_size, risk.contract_size)
        return 1.0, {"total_trades": 5}, None, None, None, None, None

    monkeypatch.setattr(ci, "_evaluate", fake_evaluate)
    dfs = {"MGC": _market(2500, 1), "EURUSD": _market(1.10, 2)}
    risks, _ = resolve_risk_per_market(dfs, RiskConfig(), auto_detect=True)
    ci._score_across_markets({}, dfs, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=50), "eval_pass_probability",
                             "mean_minus_dispersion", early_exit=False, risk_by_market=risks)
    scales = sorted(seen.values())
    assert (0.0001, None) in scales and (1.0, 10.0) in scales


def test_cancel_is_checked_between_markets(monkeypatch):
    calls = {"n": 0}

    def fake_evaluate(*a, **k):
        calls["n"] += 1
        return 1.0, {"total_trades": 5}, None, None, None, None, None

    monkeypatch.setattr(ci, "_evaluate", fake_evaluate)

    def check():
        if calls["n"] >= 1:
            raise ci.CrossInstrumentCancelled("stop")

    dfs = {"A": _market(100, 1), "B": _market(100, 2), "C": _market(100, 3)}
    with pytest.raises(ci.CrossInstrumentCancelled):
        ci._score_across_markets({}, dfs, RiskConfig(), PropRules(), MonteCarloConfig(n_simulations=50), "eval_pass_probability",
                                 "mean_minus_dispersion", early_exit=False, check_cancel=check)
    assert calls["n"] == 1   # stopped before scoring markets B and C


def test_job_page_no_longer_hangs_when_the_job_is_lost():
    html = (TEMPLATES / "cross_instrument_job.html").read_text(encoding="utf-8")
    assert "Search lost" in html and "Lost contact with the server" in html
    assert "if (data.not_found) return;" not in html


def test_form_has_detect_button_and_auto_toggle():
    html = (TEMPLATES / "cross_instrument.html").read_text(encoding="utf-8")
    assert "Detect pip size from data" in html and 'name="auto_pip"' in html


def test_detect_endpoint_and_full_run_log(monkeypatch, tmp_path):
    from app.web.server import app
    from app.data import storage
    monkeypatch.setattr(storage, "get_raw_data_dir", lambda: tmp_path, raising=False)
    import app.web.extra_routes as er
    monkeypatch.setattr(er, "get_raw_data_dir", lambda: tmp_path)
    _market(2500, 1).to_csv(tmp_path / "MGC_5m.csv", index=False)
    _market(1.10, 2).to_csv(tmp_path / "EURUSD_5m.csv", index=False)
    client = app.test_client()
    r = client.post("/cross-instrument/detect", data={"datasets": ["MGC_5m.csv", "EURUSD_5m.csv"]})
    assert r.status_code == 200, r.get_data(as_text=True)
    by = {m["label"]: m for m in r.get_json()["markets"]}
    assert by["MGC_5m"]["pip_size"] == 1.0 and by["MGC_5m"]["contract_size"] == 10.0
    assert by["EURUSD_5m"]["pip_size"] == 0.0001
    assert client.post("/cross-instrument/detect", data={}).status_code == 400


def test_empty_holdout_is_called_out():
    hold = {"holdout_period": ["2025-08-07", "2026-03-23"],
            "in_sample_statistics": {"total_trades": 44}, "holdout_statistics": {"total_trades": 0}}
    assert "HOLDOUT UNTESTED" in _holdout_untestable_note(hold)
    hold["holdout_statistics"]["total_trades"] = 6
    assert _holdout_untestable_note(hold) is None
    assert _holdout_untestable_note(None) is None


class _MC:
    risk_of_ruin_pct = 48.8
    reset_on_breach = True


class _Res:
    risk_of_ruin_hard_fail = True
    lookahead_hard_fail = False
    risk_of_ruin_cap = 20.0
    final_mc = _MC()
    scorecard = None
    warnings = ["9387 potential entries (100%) were skipped because position sizing rounded DOWN to 0 whole contracts"]
    final_bt = None
    final_holdout = {"holdout_statistics": {"total_trades": 0}, "in_sample_statistics": {"total_trades": 44}}


def test_guidance_for_sizing_starved_ruin_fail_does_not_say_lower_risk():
    text = pipeline_guide.after_full_pipeline("NOT READY", False, _Res())
    assert "does not lock anything" in text and "Validate hub" in text
    assert "Do NOT lower risk-per-trade" in text
    assert "Turn on reset" not in text          # it was already on
    assert "already ON" in text
