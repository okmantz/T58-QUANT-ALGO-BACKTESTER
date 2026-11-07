"""Payout forecast (2026-11-07): ranks firms by conservative expected payout, not by the T58 score."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.prop.presets import get_preset
from app.validation import payout_forecast as pf


def _trades(n=400, win=0.45, payoff=1.8, risk=400.0, seed=1, start="2022-01-03 10:00"):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    out, eq = [], 100_000.0
    for i in range(n):
        pnl = float(rng.choice([payoff, -1.0], p=[win, 1 - win])) * risk
        eq += pnl
        et = t0 + pd.Timedelta(days=i // 2 + (i // 10) * 2, hours=i % 2)
        out.append(Trade(entry_time=et, exit_time=et + pd.Timedelta(minutes=30), direction=1, entry_price=100.0, exit_price=101.0, size=1.0,
                         pnl=pnl, pnl_pct=pnl / eq * 100, exit_reason="tp", commission=0.0, equity_after=eq, initial_risk=risk))
    return out


def test_rescale_keeps_risk_percent_constant():
    tr = _trades(10)
    half = pf.rescale_trades(tr, 0.5)
    assert [t.pnl for t in half] == pytest.approx([t.pnl * 0.5 for t in tr])
    assert [t.initial_risk for t in half] == pytest.approx([200.0] * 10)
    assert pf.rescale_trades(tr, 1.0)[0] is tr[0]


def test_losing_strategy_does_not_pay_and_winning_one_does():
    preset = get_preset("topstep_50k")
    good = pf.forecast_firm(_trades(400, win=0.5), preset, source_account_size=100_000, sims=600, calibrate=False)
    bad = pf.forecast_firm(_trades(400, win=0.30, payoff=1.0, seed=3), preset, source_account_size=100_000, sims=600, calibrate=False)
    assert good.p_pass > bad.p_pass
    assert bad.verdict == "DOES NOT PAY"
    assert good.verdict in ("PAYS", "MARGINAL")
    assert good.p_pass_lcb <= good.p_pass            # decisions use the conservative number
    assert good.grade.startswith("IN-SAMPLE")


def test_ranking_is_by_conservative_ev_per_month():
    fs = pf.forecast_across_firms(_trades(400), sims=400, max_firms=4, source_account_size=100_000)
    keyed = [f.ev_per_month_lcb for f in fs if f.ev_per_month_lcb is not None]
    assert keyed == sorted(keyed, reverse=True)
    assert pf.summarize(fs)["n_firms"] == len(fs)


def test_too_few_trades_is_not_a_forecast():
    f = pf.forecast_firm(_trades(10), get_preset("topstep_50k"), sims=200)
    assert f.verdict == "MARGINAL" and f.p_pass is None


def test_score_vs_payout_flags_a_useless_score():
    rows = [{"name": str(i), "score": s, "ev": e} for i, (s, e) in enumerate([(90, -50), (85, -20), (70, 50), (65, 100), (40, 120)])]
    r = pf.score_vs_payout(rows)
    assert r["spearman"] < 0 and "does NOT predict" in r["message"]
    good = pf.score_vs_payout([{"name": str(i), "score": i, "ev": i * 10} for i in range(6)])
    assert good["spearman"] == pytest.approx(1.0)


def test_ledger_calibration_shrinks_but_never_raises(tmp_path):
    p = tmp_path / "ledger.jsonl"
    for i in range(10):
        pf.record_outcome(f"s{i}", "k", 0.8, passed=(i < 4), path=p)       # forecast 80%, reality 40%
    s = pf.calibration_summary(p)
    assert s["n"] == 10 and s["realised_rate"] == pytest.approx(0.4)
    assert pf.shrink_factor(p) == pytest.approx(0.5)
    for i in range(10):
        pf.record_outcome(f"t{i}", "k", 0.2, passed=True, path=tmp_path / "other.jsonl")
    assert pf.shrink_factor(tmp_path / "other.jsonl") == 1.0
    assert pf.shrink_factor(tmp_path / "missing.jsonl") == 1.0
    pf.record_outcome("s9", "k", 0.8, passed=True, path=p)                  # later record replaces the earlier one
    assert pf.calibration_summary(p)["realised_rate"] == pytest.approx(0.5)
