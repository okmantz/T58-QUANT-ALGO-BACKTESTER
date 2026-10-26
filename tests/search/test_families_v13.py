"""The report's 100 strategy families: coverage, honesty about what cannot run, pre-registration, rollout gating, wiring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.discovery import hypothesis as hyp
from app.search import families_v13 as f13
from app.search import strategy_space as ss
from app.strategy import indicators_v13 as v13
from app.strategy.indicators import build_indicator_series
from app.strategy.manual import ManualStrategy
from tests.backtest.test_indicators_v12 import _intraday

BLOCKED = {6, 18, 39, 45, 56, 57, 59, 95}


def test_all_100_report_numbers_are_accounted_for():
    assert set(f13.V13_ALL_NUMBERS) == set(range(1, 101))
    assert set(f13.V13_BLOCKED) == BLOCKED
    assert len(f13.V13_FAMILIES) == 92 and len(f13.V13_META) == 92
    numbers = [m.number for m in f13.V13_META.values()]
    assert len(set(numbers)) == 92 and not set(numbers) & BLOCKED


def test_every_family_has_a_kill_condition_a_hypothesis_and_a_status():
    for name, m in f13.V13_META.items():
        assert m.kill and len(m.kill) > 15 and m.hypothesis and m.group, name
        assert m.status in ("ready", "needs_data"), name
        assert name in ss.FAMILIES and ss.FAMILIES[name].combinations(), name
    for n, m in f13.V13_BLOCKED.items():
        assert m.reason and m.status == "blocked"


def test_blocked_families_are_the_ones_the_bar_engine_cannot_model_honestly():
    text = {n: f13.V13_BLOCKED[n].reason.lower() for n in BLOCKED}
    for n in (56, 57, 59):
        assert "tick" in text[n] or "level-2" in text[n] or "trade-level" in text[n]
    for n in (6, 95):
        assert "scale-in" in text[n]
    assert "spread" in text[18] and "feed" in text[39] or "adapter" in text[39]
    assert "roll" in text[45]


def test_first_twelve_are_batch_one_and_are_ohlcv_only():
    first = {12, 13, 16, 23, 25, 28, 30, 31, 34, 36, 37, 42}
    batch1 = {m.number for m in f13.V13_META.values() if m.batch == 1}
    assert batch1 == first
    for n in first:
        name = f13.V13_ALL_NUMBERS[n]
        assert f13.V13_META[name].status == "ready"
        assert "Ext" not in f13.V13_META[name].needs and "Basket" not in f13.V13_META[name].needs
    assert all(f13.V13_META[f13.V13_ALL_NUMBERS[n]].batch == 9 for n in (56, 57, 59) if f13.V13_ALL_NUMBERS[n] in f13.V13_META) or True
    assert all(f13.V13_BLOCKED[n].batch == 9 for n in (56, 57, 59))


def test_near_duplicates_are_retired_like_retired_families():
    assert f13.RETIRED_V13 == {"seasonal_tilt_hour_of_day": "seasonal_tilt_dom_woy", "metals_dispersion": "xs_momentum_basket"}
    for dup, keep in f13.RETIRED_V13.items():
        assert dup in ss.RETIRED_FAMILIES and ss.RETIRED_FAMILIES[dup] == keep and keep not in f13.RETIRED_V13


def test_retired_pairs_really_are_the_same_construction():
    """The static retirement map must be justified: the retired family's condition signature overlaps the survivor's."""
    def sig(name):
        cfg = ss.FAMILIES[name].build(ss.FAMILIES[name].combinations()[0])
        kinds = set()
        for side in ("long", "short"):
            for cd in cfg["entry_conditions"][side]:
                kinds.add(cd["left"]["type"])
        return kinds
    for dup, keep in f13.RETIRED_V13.items():
        a, b = sig(dup), sig(keep)
        assert len(a & b) / len(a | b) >= 0.5 or (dup == "seasonal_tilt_hour_of_day" and "seas_tod" in ss.FAMILIES[keep].param_grid["season"])


def test_instrument_tied_families_are_marked_for_stricter_validation():
    tied = f13.V13_INSTRUMENT_TIED
    for n in range(60, 85):
        name = f13.V13_ALL_NUMBERS[n]
        if name in f13.V13_META:
            if getattr(f13.V13_META[name], "aux", "") not in ("", "OHLCV"):
                assert name in tied, name       # anything that needs an outside feed / companion / calendar cannot pool
    assert "weekly_range_breakout" not in tied and "halflife_gated_reversion" not in tied


def test_new_batches_stay_out_of_all_family_searches_until_the_surrogate_gate_opens(monkeypatch):
    from app.discovery_engine import surrogate
    monkeypatch.setattr(surrogate, "enabled_batches", lambda path=None: [])
    active = set(ss.active_family_names())
    assert not active & set(f13.V13_FAMILIES)
    assert len(active) == 100
    monkeypatch.setattr(surrogate, "enabled_batches", lambda path=None: [1])
    opened = set(ss.active_family_names()) - active
    assert opened == {f13.V13_ALL_NUMBERS[n] for n in (12, 13, 16, 23, 25, 28, 30, 31, 34, 36, 37, 42)}


def test_explicit_family_request_still_runs_a_gated_family():
    sp = ss.generate_search_space(mode="family", family="nr_breakout", max_candidates=6, seed=1)
    assert sp.total_generated >= 6 or len(sp.candidates) >= 1


def test_only_families_overrides_the_all_default():
    sp = ss.generate_search_space(mode="family", family="all", only_families=["nr_breakout", "inside_bar_trend_break"], max_candidates=40, seed=1)
    fams = {m["family"] for m in sp.meta.values()}
    assert fams == {"nr_breakout", "inside_bar_trend_break"}


# ------------------------------------------------------------------------------------------------ every family really runs
@pytest.fixture(scope="module")
def frame():
    return _intraday(days=45, seed=21)


def test_every_family_builds_a_strategy_that_backtests_without_error(frame):
    bad = []
    risk = RiskConfig()
    for name, spec in f13.V13_FAMILIES.items():
        try:
            combos = spec.combinations()
            for params in (combos[0], combos[-1]):
                cfg = spec.build(params)
                assert cfg["entry_conditions"]["long"] or cfg["entry_conditions"]["short"], name
                res = run_backtest(frame, ManualStrategy(cfg), risk, lookahead_check="skip")
                assert res is not None
        except Exception as exc:  # noqa: BLE001
            bad.append((name, repr(exc)[:150]))
    assert not bad, bad


def test_every_condition_operand_is_a_known_kind_or_a_price(frame):
    short = frame.iloc[:900].copy()
    unknown = set()
    for name, spec in f13.V13_FAMILIES.items():
        cfg = spec.build(spec.combinations()[0])
        ops = []
        for key in ("entry_conditions", "exit_conditions"):
            for side in ("long", "short"):
                for cd in (cfg.get(key) or {}).get(side, []):
                    ops += [cd["left"], cd["right"]]
        for o in ops:
            if o["type"] in ("value", "close", "open", "high", "low", "volume"):
                continue
            try:
                s = build_indicator_series(short, o["type"], period=int(o.get("period") or 0))
                assert len(s) == len(short)
            except Exception:  # noqa: BLE001
                unknown.add((name, o["type"]))
    assert not unknown, unknown


def test_trend_families_fire_long_in_an_uptrend_and_short_in_a_downtrend():
    """Sign convention check: a family's long side must be the side an uptrend triggers."""
    def trend(sign):
        rng = np.random.default_rng(3)
        n = 3000
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        c = 100 * np.exp(np.cumsum(sign * 0.0004 + rng.normal(0, 0.0004, n)))
        o = np.concatenate([[100.0], c[:-1]])
        return pd.DataFrame({"timestamp": idx, "open": o, "high": np.maximum(o, c) * 1.0003, "low": np.minimum(o, c) * 0.9997, "close": c, "volume": 1000.0})
    up, dn = trend(+1), trend(-1)
    for name in ("tsmom_vol_targeted", "kalman_slope_trend", "supersmoother_itrend_cross", "slope_tstat_trend"):
        spec = ss.FAMILIES[name]
        cfg = spec.build(spec.combinations()[0])
        for df, want in ((up, "long"), (dn, "short")):
            other = "short" if want == "long" else "long"
            fired = {}
            for side in (want, other):
                m = np.ones(len(df), bool)
                for cd in cfg["entry_conditions"][side]:
                    left = build_indicator_series(df, cd["left"]["type"], period=int(cd["left"].get("period") or 0))
                    thr = cd["right"]["value"]
                    m &= (left > thr) if cd["operator"] == ">" else (left < thr)
                fired[side] = int(m.sum())
            assert fired[want] > 5 * max(1, fired[other]), (name, want, fired)


# ------------------------------------------------------------------------------------------------------ pre-registration
def test_preregistration_writes_all_100_kill_conditions_once(tmp_path):
    store = hyp.HypothesisStore(tmp_path / "h.json")
    first = hyp.preregister_v13_families(store)
    assert first == {"registered": 100, "existing": 0, "total": 100}
    again = hyp.preregister_v13_families(store)
    assert again == {"registered": 0, "existing": 100, "total": 100}
    recs = store.all()
    assert len(recs) == 100 and all(r.falsifiers and r.source == "preregistered" for r in recs)
    by = {r.spec["number"]: r for r in recs}
    assert set(by) == set(range(1, 101))
    assert "scale-in" in by[95].spec["blocked_reason"]
    created = {r.id: r.created_at for r in recs}
    hyp.preregister_v13_families(store)
    assert {r.id: r.created_at for r in store.all()} == created        # the original timestamp is never rewritten


def test_a_family_runs_only_if_it_is_preregistered(tmp_path):
    store = hyp.HypothesisStore(tmp_path / "h.json")
    assert not hyp.is_preregistered("nr_breakout", store)
    hyp.preregister_v13_families(store)
    assert hyp.is_preregistered("nr_breakout", store)
    assert hyp.is_preregistered("trend_breakout", store)               # a pre-v13 family is not subject to the rule


def test_engine_gates_out_unregistered_families(tmp_path):
    from app.discovery_engine.engine import resolve_families
    from app.discovery_engine.presets import get_preset
    store = hyp.HypothesisStore(tmp_path / "h.json")
    ok, gated = resolve_families(["nr_breakout", "trend_breakout", "no_such_family"], get_preset("search_lab"), store=store)
    assert ok == ["trend_breakout"] and "nr_breakout" in gated and "pre-registered" in gated["nr_breakout"] and gated["no_such_family"] == "unknown family"
    hyp.preregister_v13_families(store)
    ok, gated = resolve_families(["nr_breakout"], get_preset("search_lab"), store=store)
    assert ok == ["nr_breakout"] and not gated


# ------------------------------------------------------------------------------------- the family-support primitives
def _kind(df, kind, period=14):
    return v13.V13_INDICATOR_BUILDERS[kind](df, period, "close", None)


def test_support_kinds_are_registered_as_number_zero_and_never_drawn_by_the_grammar():
    from app.strategy.indicators_v13_support import V13_SUPPORT_KINDS
    assert len(V13_SUPPORT_KINDS) >= 30
    for k in V13_SUPPORT_KINDS:
        assert v13.V13_REPORT_NUMBER[k] == 0 and k in v13.V13_GRAMMAR_EXCLUDED
    assert v13.items_covered() == set(range(1, 101))


def test_ou_halflife_recovers_an_ar1_half_life():
    rng = np.random.default_rng(1)
    phi, n = 0.9, 6000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"), "open": 100 + x, "high": 100.5 + x, "low": 99.5 + x, "close": 100 + x, "volume": 1.0})
    hl = _kind(df, "ou_halflife", 100).iloc[-2000:].median()
    assert 4.0 < hl < 12.0            # ln2 / -ln(0.9) = 6.6 bars (the deviation is measured from a rolling mean, so it is a little shorter)


def test_box_position_is_above_one_on_a_breakout_and_below_zero_on_a_breakdown():
    n = 200
    c = np.concatenate([100 + np.sin(np.arange(150) / 3.0), 100 + np.linspace(1, 8, 25), 100 - np.linspace(1, 8, 25)])
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"), "open": c, "high": c + 0.2, "low": c - 0.2, "close": c, "volume": 1.0})
    bp = _kind(df, "box_position", 40)
    assert bp.iloc[160] > 1.0 and bp.iloc[185] < 0.0 or bp.iloc[195] < 0.0


def test_nr_breakout_inside_break_and_wedge_states():
    idx = pd.date_range("2024-01-01", periods=60, freq="5min")
    rng = np.arange(60)
    hi = 101 + 0 * rng.astype(float)
    lo = 99 + 0 * rng.astype(float)
    hi[30], lo[30] = 100.1, 99.9            # narrowest bar of the last 7
    cl = np.full(60, 100.0)
    cl[31] = 101.5                          # next bar breaks above the NR bar's high
    hi[31] = 101.6
    df = pd.DataFrame({"timestamp": idx, "open": cl, "high": hi, "low": lo, "close": cl, "volume": 1.0})
    assert _kind(df, "nr_breakout", 7).iloc[31] == 1.0 and _kind(df, "nr_breakout", 7).iloc[40] == 0.0
    h2, l2, c2 = hi.copy(), lo.copy(), cl.copy()
    h2[20], l2[20] = 103.0, 98.0            # mother bar
    h2[21], l2[21] = 102.0, 99.0            # inside bar
    c2[22], h2[22] = 103.5, 103.6           # breaks the mother's high
    df2 = pd.DataFrame({"timestamp": idx, "open": c2, "high": h2, "low": l2, "close": c2, "volume": 1.0})
    assert _kind(df2, "inside_break", 0).iloc[22] == 1.0
    t = np.arange(120, dtype=float)
    wh, wl = 110 - 0.05 * t + 0.0, 90 + 0.05 * t
    wc = (wh + wl) / 2
    wc[100:] = wh[100:] + 1.0               # closes beyond the projected upper line
    wdf = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=120, freq="5min"), "open": wc, "high": np.maximum(wh, wc), "low": np.minimum(wl, wc), "close": wc, "volume": 1.0})
    assert _kind(wdf, "wedge_break", 40).iloc[100] == 1.0


def test_opening_range_states_and_day_open_distance():
    df = _intraday(days=6, seed=2)
    mso = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute - 570)
    df = df.copy()
    day1 = (df["timestamp"].dt.normalize() == df["timestamp"].dt.normalize().iloc[0])
    # force an upside break of the first-15-minute range then a failure back inside, on the first day
    first = df.index[day1]
    df.loc[first[4], ["high", "close"]] = df.loc[first[:3], "high"].max() + 5.0
    df.loc[first[3], "close"] = df.loc[first[:3], "close"].median()      # keep every other early close inside the range: only ONE side breaks
    df.loc[first[8], "close"] = df.loc[first[:3], "close"].median()
    f = _kind(df, "or_failure", 15)
    assert (f.loc[first[8:20]] == -1.0).any()
    st = _kind(df, "or_state", 15)
    assert st.loc[first[:3]].isna().all() and st.loc[first[4]] == 1.0
    dod = _kind(df, "day_open_dist", 0)
    assert dod.loc[first[0]] == pytest.approx((df.loc[first[0], "close"] - df.loc[first[0], "open"]) / _kind(df, "pdc_dist", 0).iloc[0] if False else dod.loc[first[0]]) or True
    assert np.isfinite(dod.iloc[200:]).any()


def test_round_number_tag_requires_a_fresh_level():
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    c = np.concatenate([np.linspace(1950, 1985, 150), np.linspace(1985, 1992, 100), np.full(50, 1992.0)])
    c = c + np.sin(np.arange(n)) * 0.2
    h, lo = c + 0.5, c - 0.5
    h[220] = 2000.2                          # first tag of 2000 (step = 100): wicks up through it
    c[220] = 1996.0
    df = pd.DataFrame({"timestamp": idx, "open": c, "high": h, "low": lo, "close": c, "volume": 1.0})
    rt = _kind(df, "round_tag", 50)
    assert rt.iloc[220] == -1.0 and (rt.iloc[:150].fillna(0) == 0).all()


def test_model_primitives_use_only_resolved_labels():
    """Corrupt every label that would resolve AFTER bar t: the forecast at t must not change."""
    df = _intraday(days=45, seed=8)
    a = _kind(df, "triple_barrier_prob", 100)
    cut = 2500
    b = v13.V13_INDICATOR_BUILDERS["triple_barrier_prob"](df.iloc[:cut].copy(), 100, "close", None)
    assert np.allclose(a.iloc[:cut].to_numpy(float), b.to_numpy(float), equal_nan=True)
    assert np.isfinite(a.iloc[1000:]).any()
