from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.validation.preflight import baseline_preflight, enforce_preflight


def _bt(n_long, n_short, skip_ratio=0.0):
    trades = [SimpleNamespace(direction=1) for _ in range(n_long)] + [SimpleNamespace(direction=-1) for _ in range(n_short)]
    eq = pd.DataFrame({"timestamp": [], "equity": []})
    eq.attrs["sizing_halt"] = {"skip_ratio": skip_ratio, "skipped": 5, "skip_reasons": {"budget": 5}}
    return SimpleNamespace(trades=trades, equity_curve=eq)


def _risk():
    return RiskConfig(initial_balance=50_000.0, risk_mode="fixed", risk_value=500.0, pip_size=0.25, contract_size=50.0,
                      spread_pips=1.0, slippage_pips=1.0, commission_per_contract=2.4)


DF = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=500, freq="1h")})


def test_fourteen_trades_stops_at_preflight_with_the_count():
    rep = baseline_preflight(_bt(14, 0), DF, _risk())
    assert rep.blocked and "14 trades" in rep.render()
    with pytest.raises(Exception, match="14 trades"):
        enforce_preflight(rep, "Full Pipeline")


def test_sizing_skips_block_and_one_sided_warns():
    assert baseline_preflight(_bt(200, 0, skip_ratio=0.5), DF, _risk()).blocked
    ok = baseline_preflight(_bt(200, 0), DF, _risk())
    assert not ok.blocked and any(i.code == "one_sided" for i in ok.issues)


def test_continuous_holdout_never_compares_nothing():
    from app.backtest.engine import run_holdout_comparison
    from app.strategy.base import Strategy, StrategyResult

    class Flip(Strategy):
        name = "flip"
        def generate(self, df):
            return StrategyResult(name='flip', source_type='python', signals=pd.Series(np.where(np.arange(len(df)) % 10 < 5, 1, -1), index=df.index), stop_loss_pips=40, take_profit_pips=80)

    rng = np.random.default_rng(0)
    n = 3000
    px = 4000 + np.cumsum(rng.normal(0, 2, n))
    df = pd.DataFrame(dict(timestamp=pd.date_range("2024-01-02", periods=n, freq="15min"), open=px, high=px + 1, low=px - 1, close=px, volume=1))
    risk = RiskConfig(initial_balance=50_000.0, risk_mode="fixed", risk_value=500.0, pip_size=0.25, contract_size=5.0,
                      spread_pips=1.0, slippage_pips=1.0, commission_per_contract=0.6, max_trades_per_day=50)
    out = run_holdout_comparison(df, Flip(), risk, holdout_frac=0.2)
    assert out["mode"] == "continuous_account"
    assert out["holdout_trades"] > 0 and out["in_sample_trades"] > 0
    assert out["holdout_ok"] == (out["holdout_trades"] >= out["min_holdout_trades"])


def test_lucid_preset_carries_verified_account_model_and_mismatch_detection():
    from app.prop.presets import get_preset
    from app.validation.preflight import compare_rules_to_preset
    p = get_preset("lucid_50k")
    assert (p.dd_basis, p.trailing_lock, p.max_contracts, p.daily_loss_action) == ("eod", True, 4, "lock_day")
    r = p.to_prop_rules()
    assert r.dd_basis == "eod" and r.trailing_lock and r.max_contracts == 4
    assert compare_rules_to_preset(r, "lucid_50k") == []
    # the user's earlier run: 4% trailing, 30% consistency, 5 minimum days
    import dataclasses
    mine = dataclasses.replace(r, consistency_rule_pct=30.0, min_trading_days=5, dd_basis="floating")
    fields = {i.message.split(":")[0] for i in compare_rules_to_preset(mine, "lucid_50k")}
    assert {"consistency_rule_pct", "min_trading_days", "dd_basis"} <= fields


def test_lookahead_check_runs_once_per_structure(monkeypatch):
    import app.backtest.engine as eng
    import app.strategy.lookahead_check as lc
    from app.strategy.manual import ManualStrategy
    calls = []
    real = lc.check_for_lookahead
    monkeypatch.setattr(lc, "check_for_lookahead", lambda s, d: (calls.append(1), real(s, d))[1])
    eng._LOOKAHEAD_CACHE.clear()
    rng = np.random.default_rng(0)
    n = 400
    px = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(dict(timestamp=pd.date_range("2024-01-02", periods=n, freq="1h"), open=px, high=px + 1, low=px - 1, close=px, volume=1))
    def cfg(p):
        return {"name": "t", "indicators": [{"type": "rsi", "period": p, "column": "close", "as": "r"}],
                "long_entry": "r < 40", "long_exit": "r > 60", "short_entry": "r > 60", "short_exit": "r < 40",
                "stop_loss_pips": 100, "take_profit_pips": 150}
    risk = RiskConfig(initial_balance=50_000.0, risk_mode="fixed", risk_value=100.0, pip_size=0.01, contract_size=1.0,
                      spread_pips=1.0, slippage_pips=1.0, commission_per_contract=0.1)
    for p in (10, 20, 30):
        eng.run_backtest(df, ManualStrategy(cfg(p)), risk)
    assert len(calls) == 1
    eng._LOOKAHEAD_CACHE.clear(); calls.clear()
    for p in (10, 20):
        eng.run_backtest(df, ManualStrategy(cfg(p)), risk, lookahead_check="always")
    assert len(calls) == 2
