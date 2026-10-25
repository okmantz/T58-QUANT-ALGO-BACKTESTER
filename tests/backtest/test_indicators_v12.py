"""v12 indicator tests (2026-10-24): time-of-day range, Parkinson/Garman-Klass, variance ratio / autocorr,
round-number distance (ES/NQ/GC), NR4/NR7 -- reference values, prefix-invariance (no lookahead), dispatch,
manual-strategy behavioural lookahead check, and the parameter_space range registration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.optimize import parameter_space as ps
from app.strategy import indicators_v12 as v12
from app.strategy.indicators import BOUNDED_OSCILLATOR_RANGES, build_indicator_series
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.manual import ManualStrategy


def _intraday(days=30, bars_per_day=78, seed=5, start_price=5000.0):
    rng = np.random.default_rng(seed)
    idx = []
    for d in pd.bdate_range("2024-01-02", periods=days):
        idx.extend(pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=bars_per_day, freq="5min"))
    n = len(idx)
    # U-shaped intraday volatility so the same-minute baseline is meaningful
    minute = np.array([t.hour * 60 + t.minute for t in idx])
    shape = 1.0 + 1.5 * np.exp(-((minute - 570) / 25.0) ** 2) + 0.8 * np.exp(-((minute - 945) / 25.0) ** 2)
    ret = rng.normal(0, 0.0006, n) * shape
    close = start_price * np.exp(np.cumsum(ret))
    op = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0.0006, 0.0002, n)) * shape * close
    return pd.DataFrame({
        "timestamp": idx, "open": op,
        "high": np.maximum(op, close) + spread, "low": np.minimum(op, close) - spread,
        "close": close, "volume": np.abs(rng.normal(1000, 100, n)),
    })


def _prefix_invariant(fn, df, cut):
    a = fn(df).iloc[:cut].to_numpy(dtype=float)
    b = fn(df.iloc[:cut].copy()).to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True), "series changed when future bars were removed"


def test_dispatch_resolves_every_kind():
    df = _intraday()
    for kind in v12.V12_INDICATOR_KINDS:
        s = build_indicator_series(df, kind, period=14)
        assert len(s) == len(df), kind
        assert s.notna().any(), kind


def test_parkinson_matches_textbook_reference():
    df = _intraday(days=5)
    n = 20
    out = v12.parkinson_vol(df, n, "close", None)
    hl2 = np.log(df["high"] / df["low"]) ** 2
    ref = np.sqrt(hl2.rolling(n).mean() / (4 * np.log(2))) * 100
    assert np.allclose(out.dropna(), ref.dropna())


def test_garman_klass_matches_textbook_reference():
    df = _intraday(days=5)
    n = 20
    out = v12.garman_klass_vol(df, n, "close", None)
    per = 0.5 * np.log(df["high"] / df["low"]) ** 2 - (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2
    ref = np.sqrt(per.rolling(n).mean().clip(lower=0)) * 100
    assert np.allclose(out.dropna(), ref.dropna())


def test_variance_ratio_random_walk_near_one_and_trend_above_one():
    rng = np.random.default_rng(3)
    n = 4000
    def frame(ret):
        c = 100 * np.exp(np.cumsum(ret))
        return pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=n, freq="5min"), "open": c, "high": c * 1.001,
                             "low": c * 0.999, "close": c, "volume": 1.0})
    rw = v12.variance_ratio(frame(rng.normal(0, 0.001, n)), 5, "close", None).dropna()
    assert abs(rw.mean() - 1.0) < 0.15
    # AR(1) with positive phi = trending -> VR > 1; negative phi = reverting -> VR < 1
    def ar(phi):
        e = rng.normal(0, 0.001, n); r = np.zeros(n)
        for i in range(1, n):
            r[i] = phi * r[i - 1] + e[i]
        return r
    assert v12.variance_ratio(frame(ar(0.4)), 5, "close", None).dropna().mean() > 1.3
    assert v12.variance_ratio(frame(ar(-0.4)), 5, "close", None).dropna().mean() < 0.8


def test_autocorr_lag1_sign_and_bounds():
    rng = np.random.default_rng(4)
    n = 3000
    e = rng.normal(0, 0.001, n); r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.5 * r[i - 1] + e[i]
    c = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=n, freq="5min"), "open": c, "high": c, "low": c, "close": c, "volume": 1.0})
    ac = v12.autocorr_lag1(df, 100, "close", None).dropna()
    assert 0.3 < ac.mean() < 0.7
    assert ac.min() >= -1.0 and ac.max() <= 1.0


def test_round_number_distance_values_and_grids():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=5, freq="5min"),
                       "open": 0.0, "high": 0.0, "low": 0.0, "close": [5000.0, 5012.5, 5025.0, 5037.5, 5049.0], "volume": 1.0})
    es = v12.round_dist_es(df, 14, "close", None)
    assert es.iloc[0] == pytest.approx(0.0)
    assert es.iloc[1] == pytest.approx(0.25)      # 12.5 above 5000 on a 50 grid
    assert abs(es.iloc[2]) == pytest.approx(0.5)  # exactly half way
    assert es.iloc[3] == pytest.approx(-0.25)     # 12.5 below 5050
    assert es.between(-0.5, 0.5).all()
    nq = v12.round_dist_nq(pd.DataFrame({"close": [18050.0, 18000.0, 18010.0]}), 14, "close", None)
    assert list(np.round(nq.abs(), 3))[:2] == [0.5, 0.0] and nq.iloc[2] == pytest.approx(0.1)
    gc = v12.round_dist_gc(pd.DataFrame({"close": [2350.0, 2352.5, 2347.0]}), 14, "close", None)
    assert gc.iloc[0] == pytest.approx(0.0) and gc.iloc[1] == pytest.approx(0.25) and gc.iloc[2] == pytest.approx(-0.3)


def test_round_dist_auto_uses_instrument_attr_then_price_level():
    f = pd.DataFrame({"close": [18030.0, 18070.0]})
    f.attrs["instrument"] = "NQ1!"
    assert v12.round_dist_auto(f, 14, "close", None).iloc[0] == pytest.approx(0.3)
    g = pd.DataFrame({"close": [5030.0, 5070.0]})   # no attrs: decade rule -> grid 100 for ~5000
    assert v12.round_dist_auto(g, 14, "close", None).iloc[0] == pytest.approx(0.3)


def test_nr4_nr7_flag_exactly_the_narrowest_bars():
    rng_vals = [5, 4, 3, 2, 6, 7, 1, 8, 9, 3, 4, 5]
    df = pd.DataFrame({"high": [100 + r for r in rng_vals], "low": [100.0] * len(rng_vals)})
    df["open"] = df["close"] = 100.0
    nr4 = v12.nr4(df, 0, "close", None)
    nr7 = v12.nr7(df, 0, "close", None)
    # warmup is NaN
    assert nr4.iloc[:3].isna().all() and nr7.iloc[:6].isna().all()
    # bar 3 (range 2) is the narrowest of bars 0-3; bar 6 (range 1) narrowest of any window
    assert nr4.iloc[3] == 1.0 and nr4.iloc[4] == 0.0 and nr4.iloc[6] == 1.0
    assert nr7.iloc[6] == 1.0 and nr7.iloc[7] == 0.0
    # reference: range <= min over trailing window
    ref = pd.Series(rng_vals, dtype=float).rolling(4).apply(lambda w: float(w[-1] <= w.min()), raw=True)
    assert np.array_equal(nr4.dropna().to_numpy(), ref.dropna().to_numpy())


def test_daily_nr_uses_only_prior_completed_day():
    df = _intraday(days=12, bars_per_day=20)
    d = v12.daily_nr7(df, 0, "close", None)
    day = pd.to_datetime(df["timestamp"]).dt.normalize()
    # constant within a day (it is a prior-day property) ...
    assert (d.groupby(day).nunique(dropna=True) <= 1).all()
    # ... and unchanged when the LAST day is removed or its bars are altered (no same-day information)
    last = day.iloc[-1]
    mutated = df.copy()
    mutated.loc[day == last, ["high", "low", "close"]] *= 1.5
    assert np.allclose(v12.daily_nr7(mutated, 0, "close", None).to_numpy(dtype=float),
                       d.to_numpy(dtype=float), equal_nan=True)


def test_time_of_day_ratio_reference_and_shape():
    df = _intraday(days=25)
    out = v12.range_tod_ratio(df, 20, "close", None)
    rng = df["high"] - df["low"]
    key = pd.to_datetime(df["timestamp"]).dt.hour * 60 + pd.to_datetime(df["timestamp"]).dt.minute
    # reference for one specific bar: mean of the previous 20 same-minute ranges
    i = len(df) - 1
    prev = rng[(key == key.iloc[i]) & (df.index < i)].tail(20)
    assert out.iloc[i] == pytest.approx(rng.iloc[i] / prev.mean())
    # the first session has no history -> NaN; a typical later bar sits near 1
    assert out.iloc[:78].isna().all()
    assert 0.6 < out.dropna().median() < 1.5
    z = v12.range_tod_zscore(df, 20, "close", None)
    assert z.dropna().abs().median() < 3


def test_prefix_invariance_all_kinds():
    df = _intraday(days=25)
    cut = 1200
    for kind, fn in v12.V12_INDICATOR_BUILDERS.items():
        _prefix_invariant(lambda f, fn=fn: fn(f, 14, "close", None), df, cut)


@pytest.mark.parametrize("kind,thr", [("range_tod_ratio", 1.5), ("variance_ratio", 1.0), ("autocorr_lag1", 0.0),
                                       ("parkinson_vol", 0.05), ("round_dist_es", 0.1), ("nr7", 0.5)])
def test_behavioral_lookahead_check(kind, thr):
    df = _intraday(days=25)
    strat = ManualStrategy({
        "name": f"v12-{kind}",
        "entry_conditions": {"long": [{"left": {"type": kind, "period": 14}, "operator": ">",
                                       "right": {"type": "value", "value": thr}}], "short": []},
        "risk_management": {},
    })
    res = check_for_lookahead(strat, df)
    assert not res.bug_detected


def test_bounded_ranges_registered_for_validation():
    for k in ("autocorr_lag1", "round_dist_es", "round_dist_nq", "round_dist_gc", "round_dist_auto"):
        assert k in BOUNDED_OSCILLATOR_RANGES
    assert BOUNDED_OSCILLATOR_RANGES["autocorr_lag1"] == (-1.0, 1.0)
    assert BOUNDED_OSCILLATOR_RANGES["round_dist_es"] == (-0.5, 0.5)


def _genes(kind, threshold, period=14):
    cfg = {"entry_conditions": {"long": [{"left": {"type": kind, "period": period}, "operator": ">",
                                          "right": {"type": "value", "value": threshold}}]}}
    return {g.kind + "@" + g.label.split(".")[-1]: g for g in ps.extract_genome(cfg)}, ps.extract_genome(cfg)


def test_parameter_space_clamps_thresholds_into_registered_ranges():
    for kind, thr, (lo, hi) in [("autocorr_lag1", 0.2, (-1.0, 1.0)), ("round_dist_es", 0.2, (-0.5, 0.5)),
                                ("variance_ratio", 1.1, ps.INDICATOR_VALUE_RANGES["variance_ratio"]),
                                ("range_tod_ratio", 1.5, ps.INDICATOR_VALUE_RANGES["range_tod_ratio"]),
                                ("parkinson_vol", 0.1, ps.INDICATOR_VALUE_RANGES["parkinson_vol"])]:
        _, genes = _genes(kind, thr)
        value_genes = [g for g in genes if g.kind == "value"]
        assert value_genes, kind
        for g in value_genes:
            assert g.lo >= lo - 1e-9 and g.hi <= hi + 1e-9, (kind, g.lo, g.hi)


def test_parameter_space_period_ranges_and_dead_dimensions():
    _, genes = _genes("variance_ratio", 1.1, period=5)
    p = [g for g in genes if g.kind == "period"][0]
    assert 2 <= p.lo and p.hi <= 30
    _, genes = _genes("range_tod_ratio", 1.5, period=20)
    p = [g for g in genes if g.kind == "period"][0]
    assert 5 <= p.lo and p.hi <= 60
    # round_dist_* and NR flags ignore period -> no period gene; NR flag comparison constant is frozen
    for kind in ("round_dist_es", "round_dist_nq", "round_dist_gc", "nr4", "nr7", "daily_nr4", "daily_nr7"):
        _, genes = _genes(kind, 1.0)
        assert not [g for g in genes if g.kind == "period"], kind
    _, genes = _genes("nr7", 1.0)
    assert not [g for g in genes if g.kind == "value"]
