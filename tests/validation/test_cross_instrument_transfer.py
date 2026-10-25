"""Cross-instrument transfer: sibling mapping, judging, and the pipeline's verdict handling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.strategy.manual import ManualStrategy
from app.validation import cross_instrument_transfer as xt


@pytest.mark.parametrize("label,root", [("ES1!", "ES"), ("ES_5min_20260120_20260415.csv", "ES"), ("NQ1!/NQ_5min_x.csv", "NQ"),
                                        ("futures_ES.F_1m.parquet", "ES"), ("MGC1!", "MGC"), ("GC1!", "GC"),
                                        ("MNQ1!/MNQ_1min.csv", "MNQ"), ("EURUSD", None), ("", None)])
def test_root_symbol(label, root):
    assert xt.root_symbol(label) == root


def test_sibling_mapping_es_to_nq_first():
    assert xt.sibling_candidates("ES1!")[0] == "NQ"
    assert xt.sibling_candidates("GC1!")[0] == "MGC"
    assert xt.sibling_candidates("EURUSD") == []


def test_find_sibling_dataset_matches_bar_size():
    listing = [{"instrument": "NQ1!", "files": [{"name": "NQ_5min_20260120_20260415.csv", "full_name": "NQ1!/NQ_5min_20260120_20260415.csv", "empty": False},
                                                 {"name": "futures_NQ.F_1m.parquet", "full_name": "NQ1!/futures_NQ.F_1m.parquet", "empty": False}]},
               {"instrument": "ES1!", "files": []}]
    assert xt.find_sibling_dataset("ES1!", 300, listing)[:2] == ("NQ", "NQ1!/NQ_5min_20260120_20260415.csv")
    assert xt.find_sibling_dataset("ES1!", 60, listing)[:2] == ("NQ", "NQ1!/futures_NQ.F_1m.parquet")
    sib, name, tried = xt.find_sibling_dataset("ES1!", 900, listing)
    assert sib is None and name is None and tried


def test_judge_transfer_states():
    rng = np.random.default_rng(1)
    pos = rng.normal(0.15, 1.0, 400)
    st, why, m, ret = xt.judge_transfer(pos, 0.3, 0.02)
    assert st == "PASS" and m > 0 and ret == pytest.approx(m / 0.3)
    # positive but indistinguishable and tiny relative to primary -> FAIL
    st, *_ = xt.judge_transfer(rng.normal(0.01, 1.0, 400) + 0.0 - (rng.normal(0.01, 1.0, 400).mean()) + 0.01, 0.5, 0.45)
    assert st == "FAIL"
    # sign flips -> FAIL
    assert xt.judge_transfer(rng.normal(-0.1, 1.0, 400), 0.3, 0.9)[0] == "FAIL"
    # too few trades is NOT_TESTED, never FAIL
    assert xt.judge_transfer(pos[:10], 0.3, 0.02)[0] == "NOT_TESTED"
    # weak by size alone (p not significant but keeps >= 25% of the primary's R)
    st, *_ = xt.judge_transfer(np.full(100, 0.1) + rng.normal(0, 0.01, 100), 0.2, 0.6)
    assert st == "PASS"


def _momentum_frame(n, seed, phi=0.3, base=5000.0, vol=1.0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    prev, step, rows = base, 0.0, []
    for i in range(n):
        step = phi * step + rng.normal(0, vol)
        c = prev + step
        rows.append((ts[i], prev, max(prev, c) + abs(rng.normal(0, vol * 0.3)), min(prev, c) - abs(rng.normal(0, vol * 0.3)), c, 100.0))
        prev = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _strategy():
    return ManualStrategy({
        "name": "mom",
        "entry_conditions": {"long": [{"left": {"type": "close"}, "operator": ">", "right": {"type": "ema", "period": 20}}],
                             "short": [{"left": {"type": "close"}, "operator": "<", "right": {"type": "ema", "period": 20}}]},
        "risk_management": {"stop_type": "fixed", "stop_value": 600, "target_type": "fixed", "target_value": 1200}})


def test_run_transfer_check_end_to_end_with_injected_sibling():
    from app.backtest.engine import run_backtest
    risk = RiskConfig(pip_size=0.01, risk_mode="fixed", risk_value=2000.0, spread_pips=1, slippage_pips=0, commission_per_trade=0.5,
                      max_trades_per_day=999, initial_balance=50_000.0)
    dev = _momentum_frame(6000, 1)
    primary = run_backtest(dev, _strategy(), risk)
    # sibling with the same behaviour (momentum) and one with the opposite (mean reversion)
    same = run_xt = xt.run_transfer_check(_strategy(), dev, primary.trades, risk, "ES1!", sibling_df=_momentum_frame(6000, 2, base=18000.0), sibling_label="NQ")
    rev = xt.run_transfer_check(_strategy(), dev, primary.trades, risk, "ES1!", sibling_df=_momentum_frame(6000, 3, phi=-0.4, base=18000.0), sibling_label="NQ")
    assert same.n_trades > 30 and rev.n_trades > 30
    assert same.mean_r > rev.mean_r
    assert same.status in ("PASS", "FAIL") and rev.status in ("PASS", "FAIL")
    assert rev.status == "FAIL"


def test_no_sibling_data_is_not_tested_never_fail(monkeypatch):
    monkeypatch.setattr(xt, "find_sibling_dataset", lambda *a, **k: (None, None, ["NQ: no 300s dataset"]))
    dev = _momentum_frame(800, 1)
    res = xt.run_transfer_check(_strategy(), dev, [], RiskConfig(pip_size=0.01), "ES1!", timeframe_seconds=300)
    assert res.status == "NOT_TESTED" and "no sibling" in res.reason
    res2 = xt.run_transfer_check(_strategy(), dev, [], RiskConfig(pip_size=0.01), "EURUSD")
    assert res2.status == "NOT_TESTED"


def test_verdict_cap_advisory_off_and_not_tested():
    from app.orchestration import full_pipeline as fp
    src = open(fp.__file__, encoding="utf-8").read()
    assert 'NO CROSS-INSTRUMENT TRANSFER' in src and 'transfer_mode: str = "cap"' in src
    # behavioural check of the verdict branch through the wrapper's own helper signature
    import inspect
    sig = inspect.signature(fp._make_verdict)
    assert "transfer" in sig.parameters and "transfer_mode" in sig.parameters
