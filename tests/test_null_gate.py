from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig
from app.research.director import random_entry_distribution

HOLD = 6


def _df(n=700, seed=0):
    rng = np.random.default_rng(seed)
    px = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(dict(timestamp=pd.date_range("2024-01-02", periods=n, freq="15min"), open=px, high=px + 0.3, low=px - 0.3, close=px, volume=1))


def _risk():
    return RiskConfig(initial_balance=50_000.0, risk_mode="fixed", risk_value=200.0, pip_size=0.01, contract_size=1.0,
                      spread_pips=1.0, slippage_pips=1.0, commission_per_contract=0.0, max_trades_per_day=50)


def _observed(df, sig):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr, _ = run_execution(df, pd.Series(sig, index=df.index), _risk(), stop_loss_pips=150, take_profit_pips=150)
    return float(sum(t.pnl for t in tr)), len(tr), sum(t.direction == 1 for t in tr) / max(1, len(tr))


def _random_strategy(n, rng):
    sig = np.zeros(n, dtype=int)
    for st in rng.choice(np.arange(0, n - HOLD - 1, HOLD + 1), size=25, replace=False):
        sig[st:st + HOLD] = 1 if rng.random() < 0.5 else -1
    return sig


def _null(df, sig, seeds=60, seed=0):
    obs, k, lf = _observed(df, sig)
    return random_entry_distribution(df, _risk(), obs, k, lf, HOLD, stop_loss_pips=150, take_profit_pips=150, n_seeds=seeds, seed=seed)


def test_planted_edge_has_small_p_and_coin_flip_does_not():
    df = _df()
    nxt = np.sign(np.r_[df["close"].to_numpy()[HOLD:], np.zeros(HOLD)] - df["close"].to_numpy()).astype(int)   # planted: knows the future
    sig = np.zeros(len(df), dtype=int)
    for st in range(0, len(df) - HOLD - 1, 2 * (HOLD + 1)):
        sig[st:st + HOLD] = nxt[st] if nxt[st] != 0 else 1
    assert _null(df, sig)["p_value"] < 0.05
    rng = np.random.default_rng(1)
    flip = _random_strategy(len(df), rng)
    assert _null(df, flip)["p_value"] > 0.05


def test_p_values_of_random_strategies_are_roughly_uniform():
    ps = []
    for i in range(30):
        df = _df(seed=100 + i)
        sig = _random_strategy(len(df), np.random.default_rng(i))
        ps.append(_null(df, sig, seeds=40, seed=i)["p_value"])
    ps = np.array(ps)
    assert 0.3 < ps.mean() < 0.7
    assert (ps < 0.10).mean() < 0.25


def test_verdict_wrapper_demotes_ready_when_null_not_beaten(monkeypatch):
    import app.orchestration.full_pipeline as fp
    monkeypatch.setattr(fp, "_make_verdict_core", lambda *a, **k: ("READY", ["ok"], {}, False, False))
    mc = SimpleNamespace(sample_ok=True, per_attempt_pass_probability=60.0)
    v, reasons, *_ = fp._make_verdict(mc, null_result={"p_value": 0.4, "n_seeds": 100, "metric": "net_profit"})
    assert v == "MARGINAL" and any("NULL NOT BEATEN" in r for r in reasons)
    v2, *_ = fp._make_verdict(mc, null_result={"p_value": 0.02, "n_seeds": 100, "metric": "net_profit"})
    assert v2 == "READY"
