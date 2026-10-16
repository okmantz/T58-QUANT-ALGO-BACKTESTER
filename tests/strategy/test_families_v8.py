"""v8 (2026-10-05) -- new strategy families + new search-space tunables.

For every family in app.search.families_v8.V8_FAMILIES:
  * it is registered in app.search.strategy_space.FAMILIES (so BOTH
    Search Lab's generate_search_space and Evolution Lab's immigrant draw
    can use it),
  * it generates valid candidates on synthetic ES-scale OHLC,
  * a built candidate passes app.strategy.lookahead_check (no lookahead),
  * its risk management is ATR-relative (no fixed-pip / FX-scale trap),
  * the data it is tested on suggests an ES-scale pip_size (instrument-
    aware: suggest_pip_size(df) == 1.0, not the 0.0001 FX default).

Also covers the four new generate_search_space() tunables
(max_hold_bars, stop_mult_scale, session_filter, family_budget_caps):
each must actually change candidate generation/evaluation, and all four
must be default-neutral (existing callers see byte-identical spaces).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import suggest_pip_size
from app.search import strategy_space as sps
from app.search.families_v8 import V8_FAMILIES, V8_HYPOTHESIS_QUESTIONS
from app.search.strategy_space import StrategySpaceError, generate_search_space
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.manual import ManualStrategy

FAMILY_NAMES = sorted(V8_FAMILIES.keys())


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


def test_v8_families_registered_for_both_labs():
    # Search Lab draws via generate_search_space; Evolution Lab draws its
    # immigrant families from list_families() -- both read FAMILIES.
    for name in FAMILY_NAMES:
        assert name in sps.FAMILIES, f"{name} not registered"
        assert name in sps.list_families(), f"{name} missing from list_families()"
        assert name in sps.HYPOTHESIS_QUESTIONS, f"{name} missing hypothesis question"
    # every family is reachable through the family grid entry point
    space = generate_search_space(mode="family", family="all", max_candidates=200, seed=1)
    fams = {m["family"] for m in space.meta.values()}
    assert set(FAMILY_NAMES) & fams, "no v8 family drawn in an 'all' sample"
    # taxonomy map covers the new names (test_family_taxonomy pins the
    # key-sets equal, this is the belt-and-braces half).
    from app.strategy.family_taxonomy import _SKELETON_TO_GROUP
    for name in FAMILY_NAMES:
        assert name in _SKELETON_TO_GROUP, f"{name} missing from taxonomy map"


def test_v8_hypothesis_questions_present():
    for name in FAMILY_NAMES:
        assert V8_HYPOTHESIS_QUESTIONS.get(name), f"{name} has no hypothesis question"


def test_v8_test_data_is_es_scale():
    df = make_synthetic_es()
    assert suggest_pip_size(df) == 1.0


def test_v8_family_count():
    assert len(V8_FAMILIES) == 8


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_v8_family_builds_and_trades(family, es_df):
    space = generate_search_space(mode="family", family=family, max_candidates=6, seed=42)
    assert len(space.candidates) > 0, f"{family}: no candidates generated"
    fired = 0
    for _cid, spec in space.candidates.items():
        strat = ManualStrategy(spec["config"])
        res = strat.generate(es_df.copy())
        fired += int(res.signals.abs().sum())
    assert fired > 0, f"{family}: no signals across sampled combos (dead family?)"


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_v8_family_passes_lookahead(family, es_df):
    space = generate_search_space(mode="family", family=family, max_candidates=2, seed=7)
    _cid, spec = next(iter(space.candidates.items()))
    strat = ManualStrategy(spec["config"])
    result = check_for_lookahead(strat, es_df.copy())
    assert result.checked, f"{family}: lookahead check could not run ({result.skip_reason})"
    assert not result.bug_detected, f"{family}: LOOKAHEAD BUG: {result.summary()}"


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_v8_family_risk_is_atr_relative(family):
    """No FX-scale pip trap: stops/targets are ATR multiples, never fixed
    pips or absolute price levels."""
    space = generate_search_space(mode="family", family=family, max_candidates=1, seed=3)
    _cid, spec = next(iter(space.candidates.items()))
    rm = spec["config"]["risk_management"]
    assert rm.get("stop_type") == "atr", f"{family}: stop_type={rm.get('stop_type')}"
    assert rm.get("target_type") == "atr", f"{family}: target_type={rm.get('target_type')}"


# ---------------------------------------------------------------------------
# v8 generate_search_space() tunables: each must actually change candidate
# generation/evaluation, and all must be default-neutral.
# ---------------------------------------------------------------------------

def _first_rms(family, seed=42, **kwargs):
    space = generate_search_space(mode="family", family=family, seed=seed, **kwargs)
    return [spec["config"]["risk_management"] for spec in space.candidates.values()]


def test_tunables_are_default_neutral():
    base = generate_search_space(mode="family", family="linreg_confirmed_trend_continuation", seed=42)
    same = generate_search_space(
        mode="family", family="linreg_confirmed_trend_continuation", seed=42,
        max_hold_bars=None, stop_mult_scale=1.0, session_filter=None, family_budget_caps=None,
    )
    assert set(base.candidates) == set(same.candidates)
    for cid in base.candidates:
        assert base.candidates[cid] == same.candidates[cid], cid


def test_max_hold_bars_overrides_time_exits():
    # rsi2_volume_confirmed_reversion's grid exposes max_bars [None, 24]:
    # combos that HAVE the key get overridden, combos without it keep
    # their designed behavior (no key is added).
    space = generate_search_space(
        mode="family", family="rsi2_volume_confirmed_reversion",
        max_candidates=200, seed=42, max_hold_bars=8)
    assert len(space.candidates) > 0
    seen_with, seen_without = 0, 0
    for cid, spec in space.candidates.items():
        rm = spec["config"]["risk_management"]
        fam_params = space.meta[cid]["params"]
        if fam_params["max_bars"] is None:
            assert "max_bars_in_trade" not in rm, "override must not invent a time exit"
            seen_without += 1
        else:
            assert rm["max_bars_in_trade"] == 8, rm
            seen_with += 1
    assert seen_with > 0 and seen_without > 0


def test_stop_mult_scale_scales_atr_stops():
    base = _first_rms("linreg_confirmed_trend_continuation")
    scaled = _first_rms("linreg_confirmed_trend_continuation", stop_mult_scale=2.0)
    assert len(base) == len(scaled) > 0
    for b, s in zip(base, scaled):
        assert s["stop_value"] == pytest.approx(b["stop_value"] * 2.0), (b, s)
        # target untouched -- the knob is stop-only
        assert s["target_value"] == pytest.approx(b["target_value"])


def test_stop_mult_scale_rejects_nonpositive():
    with pytest.raises(StrategySpaceError):
        generate_search_space(mode="family", family="trend_breakout", stop_mult_scale=0)
    with pytest.raises(StrategySpaceError):
        generate_search_space(mode="family", family="trend_breakout", stop_mult_scale=-1.5)


def test_max_hold_bars_rejects_nonpositive():
    with pytest.raises(StrategySpaceError):
        generate_search_space(mode="family", family="trend_breakout", max_hold_bars=0)


def _has_time_of_day_gate(config, start, end) -> bool:
    for side in ("long", "short"):
        for cond in config["entry_conditions"].get(side, []):
            left = cond.get("left", {})
            if (left.get("type") == "time_of_day"
                    and left.get("session_start") == start
                    and left.get("session_end") == end):
                return True
    return False


def test_session_filter_gates_entries():
    base = generate_search_space(mode="family", family="trend_breakout", max_candidates=3, seed=42)
    gated = generate_search_space(
        mode="family", family="trend_breakout", max_candidates=3, seed=42,
        session_filter=("09:30", "11:00"))
    assert len(base.candidates) == len(gated.candidates) > 0
    for cid, spec in gated.candidates.items():
        assert _has_time_of_day_gate(spec["config"], "09:30", "11:00"), cid
    for _cid, spec in base.candidates.items():
        assert not _has_time_of_day_gate(spec["config"], "09:30", "11:00")


def test_session_filter_rejects_inverted_window():
    with pytest.raises(StrategySpaceError):
        generate_search_space(mode="family", family="trend_breakout",
                              session_filter=("15:00", "09:30"))


def test_family_budget_caps_clamp_allocation():
    space = generate_search_space(
        mode="family", family="all", max_candidates=400, seed=1,
        family_budget_caps={"trend_breakout": 2, "mean_reversion_band": 0})
    counts: dict[str, int] = {}
    for m in space.meta.values():
        counts[m["family"]] = counts.get(m["family"], 0) + 1
    assert counts.get("trend_breakout", 0) <= 2, counts.get("trend_breakout")
    assert counts.get("mean_reversion_band", 0) == 0, "cap of 0 must drop the family"
    # other families still fill the budget
    assert sum(counts.values()) > 10


def test_family_budget_caps_never_empty_the_space():
    # Capping EVERYTHING at 0 must fall back to searching everything
    # (same safety rule as exclude_families), not raise.
    all_fams = list(sps.FAMILIES.keys())
    caps = {f: 0 for f in all_fams}
    space = generate_search_space(mode="family", family="all", max_candidates=50,
                                  seed=1, family_budget_caps=caps)
    assert len(space.candidates) > 0
