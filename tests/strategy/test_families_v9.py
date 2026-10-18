"""v9 gamma-exposure families + indicators: registration, signals, lookahead, ATR-relative risk."""
from __future__ import annotations

import pytest

from app.search import strategy_space as sps
from app.search.families_v9 import V9_FAMILIES, V9_HYPOTHESIS_QUESTIONS
from app.search.strategy_space import generate_search_space
from app.strategy.indicators import build_indicator_series
from app.strategy.indicators_v9 import V9_INDICATOR_KINDS
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.manual import ManualStrategy
from tests.strategy.test_families_v8 import make_synthetic_es

FAMILY_NAMES = sorted(V9_FAMILIES)


@pytest.fixture(scope="module")
def es_df():
    return make_synthetic_es()


def test_family_count_and_registration():
    assert len(V9_FAMILIES) == 8
    from app.strategy.family_taxonomy import _SKELETON_TO_GROUP
    for n in FAMILY_NAMES:
        assert n in sps.FAMILIES and n in sps.list_families() and V9_HYPOTHESIS_QUESTIONS[n]
        assert n in _SKELETON_TO_GROUP


@pytest.mark.parametrize("kind", V9_INDICATOR_KINDS)
def test_indicator_length_and_no_nan(kind, es_df):
    s = build_indicator_series(es_df, kind, period=9)
    assert len(s) == len(es_df) and not s.isna().any()


@pytest.mark.parametrize("kind", V9_INDICATOR_KINDS)
def test_indicator_prefix_invariance(kind, es_df):
    """Appending future bars must not change any earlier value (no lookahead)."""
    full = build_indicator_series(es_df, kind, period=9)
    cut = 2000
    part = build_indicator_series(es_df.iloc[:cut].copy(), kind, period=9)
    assert (full.iloc[:cut].reset_index(drop=True) - part.reset_index(drop=True)).abs().max() < 1e-9


def test_real_gamma_levels_override_proxy(es_df):
    df = es_df.copy()
    df["gex_flip"] = df["close"] - 10.0
    df["gex_call_wall"] = df["close"] + 20.0
    df["gex_put_wall"] = df["close"] - 20.0
    s = build_indicator_series(df, "gex_flip_distance", period=14)
    assert (s.iloc[50:] > 0).all()


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_family_builds_and_generates(family, es_df):
    space = generate_search_space(mode="family", family=family, max_candidates=6, seed=42)
    assert space.candidates
    fired = sum(int(ManualStrategy(s["config"]).generate(es_df.copy()).signals.abs().sum()) for s in space.candidates.values())
    assert fired > 0, f"{family}: dead family"



@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_family_passes_lookahead(family, es_df):
    space = generate_search_space(mode="family", family=family, max_candidates=2, seed=7)
    spec = next(iter(space.candidates.values()))
    res = check_for_lookahead(ManualStrategy(spec["config"]), es_df.copy())
    assert res.checked, res.skip_reason
    assert not res.bug_detected, res.summary()


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_family_risk_is_atr_relative(family):
    spec = next(iter(generate_search_space(mode="family", family=family, max_candidates=1, seed=3).candidates.values()))
    rm = spec["config"]["risk_management"]
    assert rm["stop_type"] == "atr" and rm["target_type"] == "atr"
