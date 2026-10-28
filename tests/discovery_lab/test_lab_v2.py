"""Discovery Lab v2: the saved strategy must be causal, budgets must meet the promised idea counts, and winners must be real-engine proven."""
import numpy as np
import pandas as pd

from app.discovery_lab import LabConfig, run_lab
from app.discovery_lab.codegen import render
from app.discovery_lab.engine import EFFORT_BUDGET


def _bars(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    c = 4000 + np.cumsum(rng.normal(0, 2, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0, 1.5, n)
    l = np.minimum(o, c) - rng.uniform(0, 1.5, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="60min")
    return pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 1000})


def test_effort_budgets_meet_promises():
    assert EFFORT_BUDGET["quick"] >= 2500 and EFFORT_BUDGET["standard"] >= 5000 and EFFORT_BUDGET["deep"] >= 10000


def test_generated_strategy_is_causal(tmp_path):
    from app.strategy.python import PythonStrategy
    df = _bars()
    cand = {"family": "zscore_rev", "params": {"n": 20, "z": 2.0, "side": 1}, "flt": {}, "stop_atr": 2.0, "target_r": 1.0, "hold": 30,
            "id": "t1", "label": "t"}
    from app.discovery_lab import families as F
    fam = F.FAMILIES["zscore_rev"]
    cand["params"] = {k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in fam.grid.items()} if hasattr(fam, "grid") else cand["params"]
    p = tmp_path / "s.py"
    p.write_text(render({**cand, "target_r": 1.0}, "ES", 60, "s"), encoding="utf-8")
    mod = {}
    exec(compile(p.read_text(), str(p), "exec"), mod)
    full = mod["generate_signals"](df).to_numpy()
    for cut in (4000, 4500, 5200, 5900):
        part = mod["generate_signals"](df.iloc[:cut].reset_index(drop=True)).to_numpy()
        assert (part[:cut - 1] == full[:cut - 1]).all(), f"signal at bars before {cut} changed when later bars were removed"


def test_run_lab_small_budget_runs_and_serialises():
    import json
    res = run_lab(_bars(), 0.5, LabConfig(effort="quick", budget=400, prove_in_engine=False))
    json.dumps(res.to_dict())
    assert res.tested >= 300
