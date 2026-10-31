"""Discovery Lab strategy files must pass the behavioural lookahead test (2026-10-31).

Regression: every generated strategy was BLOCKED in Full Pipeline with "LOOKAHEAD BIAS DETECTED ... 1 of N bars before the cutoff produced a
different signal". The lookahead test truncates the data right after a signal bar; the saved strategy used to drop a setup that completes on the
LAST available bar (it wanted the next bar's open to price the entry), so that one bar's signal differed between the truncated and the full run.
"""
from __future__ import annotations

import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.discovery_lab import families as F
from app.discovery_lab.codegen import render
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.python import PythonStrategy


def _bars(n=4000, seed=11, freq="1h"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-03", periods=n, freq=freq, tz="UTC")
    c = 100 + np.cumsum(rng.normal(0, 0.3, n))
    o = np.r_[c[0], c[:-1]] + rng.normal(0, 0.05, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.15, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.15, n))
    return pd.DataFrame({"timestamp": ts, "open": o, "high": h, "low": lo, "close": c, "volume": 1000.0})


def _cand(family, params, hold, stop=1.5, target=1.0):
    return {"family": family, "params": params, "flt": {}, "stop_atr": stop, "target_r": target, "hold": hold, "label": "t", "id": "t",
            "train": {}, "val": {}, "test": {}}


def _strategy(cand, minutes=60):
    p = Path(tempfile.mkdtemp()) / "s.py"
    p.write_text(render(cand, "TEST", minutes, "s"), encoding="utf-8")
    return PythonStrategy(p)


def _module(cand, minutes=60):
    mod = types.ModuleType("gen")
    exec(compile(render(cand, "TEST", minutes, "s"), "gen", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("hold", [6, 48, 400])
@pytest.mark.parametrize("family,params", [
    ("ema_cross", {"fast": 8, "slow": 40, "adx_min": 0}),
    ("rsi_rev", {"n": 3, "lo": 20, "trend": 0}),
    ("donchian", {"n": 20, "trend": 0}),
])
def test_generated_strategy_passes_the_lookahead_test(family, params, hold):
    res = check_for_lookahead(_strategy(_cand(family, params, hold)), _bars())
    assert res.checked and not res.bug_detected, res.summary()
    assert res.checkpoints_tested, "the test must actually have truncated somewhere"


def test_signal_on_the_last_bar_is_kept():
    """The exact leak: a setup completing on the final bar must be emitted, not dropped, so it matches the same bar when more data follows."""
    mod = _module(_cand("ema_cross", {"fast": 8, "slow": 40, "adx_min": 0}, 12))
    n = 600
    o = h = l = c = np.linspace(100, 110, n)
    atr = np.full(n, 1.0)
    lg = np.zeros(n, bool)
    sh = np.zeros(n, bool)
    lg[n - 1] = True
    level = mod._levels(lg, sh, o, h, l, c, atr)
    assert level[n - 1] == 1


def test_levels_are_prefix_invariant_for_every_cut():
    """level[t] depends only on bars 0..t: cutting the data anywhere must not change the bars that remain."""
    mod = _module(_cand("ema_cross", {"fast": 8, "slow": 40, "adx_min": 0}, 7, stop=1.0, target=1.0))
    rng = np.random.default_rng(2)
    n = 700
    c = 100 + np.cumsum(rng.normal(0, 0.4, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0, 0.3, n)
    l = np.minimum(o, c) - rng.uniform(0, 0.3, n)
    atr = np.full(n, 0.6)
    lg = rng.random(n) < 0.05
    sh = (rng.random(n) < 0.05) & ~lg
    full = mod._levels(lg, sh, o, h, l, c, atr)
    for cut in range(mod.WARMUP + 1, n):
        part = mod._levels(lg[:cut], sh[:cut], o[:cut], h[:cut], l[:cut], c[:cut], atr[:cut])
        assert np.array_equal(full[:cut], part), f"signal changed when bars after {cut} were removed"


def test_every_family_is_causal_on_real_codegen_output():
    """One random-ish setting from each strategy family (pool indicators have their own test) through the same checker Full Pipeline uses."""
    df = _bars(n=2500, seed=4)
    rng = np.random.default_rng(1)
    for name, fam in F.FAMILIES.items():
        if name == "ind_rule" or fam.group == "session":
            continue
        params = {k: v[int(rng.integers(len(v)))] for k, v in fam.grid.items()}
        res = check_for_lookahead(_strategy(_cand(name, params, 24, stop=2.0, target=1.5)), df)
        assert not res.bug_detected, f"{name} {params}: {res.summary()}"
