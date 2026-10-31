"""A Discovery Lab winner must be one Full Pipeline's preflight accepts (2026-10-30: winners were failing at preflight with a sizing error)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from app.backtest.risk import RiskConfig
from app.discovery_lab import families as F
from app.discovery_lab.codegen import render
from app.discovery_lab.verify import _pipeline_ready
from app.prop.simulator import PropRules
from app.strategy.python import PythonStrategy


def _bars(n=6000, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-02 14:30", periods=n, freq="5min")
    c = 4500 + np.cumsum(rng.normal(0, 1.2, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0, 1, n)
    lo = np.minimum(o, c) - rng.uniform(0, 1, n)
    return pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": lo, "close": c, "volume": 100.0})


def _strategy(stop_atr):
    fam = next(iter(F.FAMILIES.values()))
    c = {"family": fam.name, "params": next(iter(F.all_param_sets(fam))), "flt": {}, "target_r": 2.0, "stop_atr": stop_atr,
         "hold": 100, "label": "t", "id": "x", "train": {}, "val": {}, "test": {}}
    p = Path(tempfile.mkdtemp()) / "s.py"
    p.write_text(render(c, "ES", 5, "s"))
    return PythonStrategy(p)


def test_pipeline_ready_returns_a_verdict_and_never_raises():
    risk = RiskConfig(initial_balance=100_000.0, risk_mode="percent", risk_value=1.0, contract_size=50.0)
    out = _pipeline_ready(_strategy(1.0), _bars(), None, risk, PropRules(), "ES")
    assert set(out) >= {"ok"} and isinstance(out["ok"], bool)
    if not out["ok"]:
        assert out["reason"]


def test_unreadable_strategy_is_not_ready():
    out = _pipeline_ready(object(), _bars(), None, RiskConfig(), PropRules(), "ES")
    assert out["ok"] is False and out["reason"]
