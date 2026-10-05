"""v7 (2026-10-05, worker B) -- Workstream B: new strategy families.

For every family in app.search.families_v7.V7_FAMILIES:
  * it is registered in app.search.strategy_space.FAMILIES (so BOTH
    Search Lab's generate_search_space and Evolution Lab's immigrant draw
    can use it),
  * it generates valid candidates on synthetic ES-scale OHLC,
  * a built candidate passes app.strategy.lookahead_check (no lookahead),
  * its risk management is ATR-relative (no fixed-pip / FX-scale trap),
  * the data it is tested on suggests an ES-scale pip_size (instrument-
    aware: suggest_pip_size(df) == 1.0, not the 0.0001 FX default).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import suggest_pip_size
from app.search import strategy_space as sps
from app.search.families_v7 import V7_FAMILIES, V7_HYPOTHESIS_QUESTIONS
from app.search.strategy_space import generate_search_space
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.manual import ManualStrategy

FAMILY_NAMES = sorted(V7_FAMILIES.keys())


def make_synthetic_es(n_bars=3000, seed=7, start_price=6000.0):
    """Synthetic ES-scale 5-min OHLCV: trend legs + abrupt volatility
    regimes (so atr_contraction/atr_expansion terminals fire), tz-aware
    timestamps (for time_of_day / session_* / VWAP operands)."""
    rng = np.random.default_rng(seed)
    sig = np.ones(n_bars) * 0.00045
    i, hi = 0, False
    while i < n_bars:
        seg = int(rng.integers(100, 300))
        hi = not hi
        sig[i:i + seg] = 0.0011 if hi else 0.00022
        i += seg
    drift = np.zeros(n_bars)
    i = 0
    while i < n_bars:
        seg = int(rng.integers(60, 240))
        if rng.random() < 0.45:
            drift[i:i + seg] = rng.normal(0.35, 0.15)
        i += seg
    rets = rng.normal(0, 1, n_bars) * sig + drift / start_price
    close = start_price * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0.6, 0.25, n_bars)) + 0.25
    open_ = np.concatenate([[close[0]], close[:-1]]) + rng.normal(0, 0.3, n_bars)
    high = np.maximum(open_, close) + spread * rng.random(n_bars)
    low = np.minimum(open_, close) - spread * rng.random(n_bars)
    volume = rng.integers(200, 4000, n_bars)
    ts = pd.date_range("2026-09-01", periods=n_bars, freq="5min", tz="America/Chicago")
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


@pytest.fixture(scope="module")
def es_df():
    return make_synthetic_es()


def test_v7_families_registered_for_both_labs():
    # Search Lab draws via generate_search_space; Evolution Lab draws its
    # immigrant families from list_families() -- both read FAMILIES.
    for name in FAMILY_NAMES:
        assert name in sps.FAMILIES, f"{name} not registered"
        assert name in sps.list_families(), f"{name} missing from list_families()"
    # every family is reachable through the family grid entry point
    space = generate_search_space(mode="family", family="all", max_candidates=50, seed=1)
    fams = {m["family"] for m in space.meta.values()}
    assert set(FAMILY_NAMES) & fams, "no v7 family drawn in an 'all' sample"


def test_v7_hypothesis_questions_present():
    for name in FAMILY_NAMES:
        assert V7_HYPOTHESIS_QUESTIONS.get(name), f"{name} has no hypothesis question"


def test_v7_test_data_is_es_scale():
    df = make_synthetic_es()
    assert suggest_pip_size(df) == 1.0


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_v7_family_builds_and_trades(family, es_df):
    space = generate_search_space(mode="family", family=family, max_candidates=4, seed=42)
    assert len(space.candidates) > 0, f"{family}: no candidates generated"
    fired = 0
    for _cid, spec in space.candidates.items():
        strat = ManualStrategy(spec["config"])
        res = strat.generate(es_df.copy())
        fired += int(res.signals.abs().sum())
    assert fired > 0, f"{family}: no signals across sampled combos (dead family?)"


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_v7_family_passes_lookahead(family, es_df):
    space = generate_search_space(mode="family", family=family, max_candidates=2, seed=7)
    _cid, spec = next(iter(space.candidates.items()))
    strat = ManualStrategy(spec["config"])
    result = check_for_lookahead(strat, es_df.copy())
    assert result.checked, f"{family}: lookahead check could not run ({result.skip_reason})"
    assert not result.bug_detected, f"{family}: LOOKAHEAD BUG: {result.summary()}"


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_v7_family_risk_is_atr_relative(family):
    """No FX-scale pip trap: stops/targets are ATR multiples, never fixed
    pips or absolute price levels."""
    space = generate_search_space(mode="family", family=family, max_candidates=1, seed=3)
    _cid, spec = next(iter(space.candidates.items()))
    rm = spec["config"]["risk_management"]
    assert rm.get("stop_type") == "atr", f"{family}: stop_type={rm.get('stop_type')}"
    assert rm.get("target_type") == "atr", f"{family}: target_type={rm.get('target_type')}"
