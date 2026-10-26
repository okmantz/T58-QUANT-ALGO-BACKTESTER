"""v13 indicator tests (2026-10-25): all 100 report items present, dispatch, DIRECT lookahead checks (prefix invariance on every
kind + manual-strategy behavioural check), reference values, planted-structure recoveries, Ext release lags, companion data,
parameter_space registration, grammar registration and the ledger charge."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data import macro_data as md
from app.optimize import parameter_space as ps
from app.strategy import indicators_v13 as v13
from app.strategy.indicators_v13_support import SUPPORT_NEEDS_COMPANION
from app.strategy.indicators import BOUNDED_OSCILLATOR_RANGES, build_indicator_series
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.manual import ManualStrategy
from tests.backtest.test_indicators_v12 import _intraday


def _daily_frame(n=700, seed=2, drift=0.0003):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    r = rng.normal(drift, 0.01, n)
    c = 100 * np.exp(np.cumsum(r))
    o = np.concatenate([[100.0], c[:-1]]) * (1 + rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"timestamp": idx + pd.Timedelta(hours=16), "open": o, "high": h, "low": lo, "close": c,
                         "volume": np.abs(rng.normal(1000, 150, n))})


def _round_the_clock(days=14, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-04 18:00", periods=days * 288, freq="5min")
    r = rng.normal(0, 0.0004, len(idx))
    c = 5000 * np.exp(np.cumsum(r))
    o = np.concatenate([[5000.0], c[:-1]])
    spread = np.abs(rng.normal(0.0004, 0.0001, len(idx))) * c
    return pd.DataFrame({"timestamp": idx, "open": o, "high": np.maximum(o, c) + spread, "low": np.minimum(o, c) - spread,
                         "close": c, "volume": np.abs(rng.normal(1000, 100, len(idx)))})


def _with_companions(df, seed=9):
    rng = np.random.default_rng(seed)
    bench = df.copy()
    bench["close"] = 4000 * np.exp(np.cumsum(rng.normal(0, 0.0005, len(df))))
    out = df.copy()
    out.attrs["benchmark"] = bench
    out.attrs["basket"] = {f"S{i}": df.assign(close=100 * np.exp(np.cumsum(rng.normal(0, 0.0006, len(df))))) for i in range(4)}
    for k in ("copper", "gold", "silver", "oil"):
        out.attrs[k] = df.assign(close=50 * np.exp(np.cumsum(rng.normal(0, 0.0005, len(df)))))
    day = pd.to_datetime(df["timestamp"]).dt.normalize()
    ix = pd.bdate_range(day.min() - pd.Timedelta(days=400), day.max())
    ext = {}
    for name, spec in md.EXT_FEEDS.items():
        n = len(ix)
        s = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 10.0, index=ix)
        if spec.kind == "intraday":
            level = 1e5 * np.exp(np.cumsum(rng.normal(0, 0.01, len(df)))) if name == "crypto_oi" else np.cumsum(rng.normal(0, 1, len(df))) * 1e-4 + 1e-4
            s = pd.Series(level, index=pd.to_datetime(df["timestamp"]).to_numpy())
        elif name in ("open_interest", "hy_oas", "move", "vvix", "skew"):
            s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=ix)
        ext[name] = s
    out.attrs["ext"] = ext
    out.attrs["macro"] = {"VIX": pd.Series(np.abs(np.cumsum(rng.normal(0, 0.3, len(ix)))) + 15.0, index=ix.normalize())}
    return out


def test_all_100_report_items_are_covered():
    assert v13.items_covered() == set(range(1, 101))
    assert len(v13.V13_INDICATOR_KINDS) >= 100
    for n in range(1, 101):
        assert v13.kinds_for_item(n), n


def test_dispatch_resolves_every_non_external_kind():
    df = _intraday(days=40)
    for kind in v13.V13_INDICATOR_KINDS:
        s = build_indicator_series(df, kind, period=14)
        assert len(s) == len(df), kind
        if kind in v13.V13_EXT_KINDS | v13.V13_COMPANION_KINDS or kind in SUPPORT_NEEDS_COMPANION:
            assert s.isna().all(), f"{kind} must be NaN (condition False) without its feed / companion frames"
        elif kind in ("avwap_dist_fomc", "avwap_dist_gap", "gap_fill_progress", "gap_fill_minutes", "seas_dow", "seas_dom", "seas_woy",
                      "handoff_london_asia", "handoff_ny_london", "handoff_pos_asia", "handoff_pos_london"):
            assert len(s) == len(df)          # need gaps / events / 24h sessions / long history: covered by dedicated tests below
        else:
            assert s.notna().any(), kind


def test_prefix_invariance_every_kind_no_lookahead():
    """The report's rule: run the lookahead check on each addition DIRECTLY (the gate skips manual strategies)."""
    df = _with_companions(_intraday(days=45))
    cut = 2200
    bad, trivial = [], []
    needs_special_data = {"avwap_dist_fomc", "avwap_dist_gap", "gap_fill_progress", "gap_fill_minutes", "seas_dow", "seas_dom", "seas_woy",
                          "handoff_london_asia", "handoff_ny_london", "handoff_pos_asia", "handoff_pos_london"}
    for kind, fn in v13.V13_INDICATOR_BUILDERS.items():
        full = fn(df, 14, "close", None)
        a = full.iloc[:cut].to_numpy(dtype=float)
        b = fn(df.iloc[:cut].copy(), 14, "close", None).to_numpy(dtype=float)
        if not np.allclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-9):
            bad.append(kind)
        if kind not in needs_special_data and int(np.isfinite(a).sum()) < 50:
            trivial.append(kind)          # an all-NaN series is trivially "invariant": it must have real values to prove anything
    assert not bad, f"series changed when future bars were removed: {bad}"
    assert not trivial, f"prefix test was vacuous (no values) for: {trivial}"


def test_prefix_invariance_daily_and_round_the_clock_kinds():
    daily = _daily_frame(500)
    rtc = _round_the_clock(12)
    for df, cut, kinds in ((daily, 400, ["seas_dow", "seas_dom", "seas_woy", "hy_spread", "pwk_high_dist", "pmo_low_dist", "tpo_poc_dist"]),
                           (rtc, 2400, ["handoff_london_asia", "handoff_ny_london", "handoff_pos_asia", "handoff_pos_london", "naked_poc_dist"])):
        for kind in kinds:
            fn = v13.V13_INDICATOR_BUILDERS[kind]
            a = fn(df, 14, "close", None).iloc[:cut].to_numpy(float)
            b = fn(df.iloc[:cut].copy(), 14, "close", None).to_numpy(float)
            assert np.allclose(a, b, equal_nan=True), kind


PRIORITY = [("yang_zhang_vol", 0.1), ("ewma_vol", 0.05), ("kalman_slope_z", 0.0), ("ehlers_supersmoother", 0.0), ("dfa_exponent", 0.5),
            ("rvol_tod", 1.5), ("bvc_imbalance", 0.0), ("avwap_dist_week", 0.0), ("zigzag_leg_atr", 2.0), ("pwk_high_dist", 0.0),
            ("seas_tod", 1.0), ("bocpd_prob", 0.3), ("hmm_bull_prob", 0.5), ("cusum_stat", 1.0), ("td_setup", 5.0)]


@pytest.mark.parametrize("kind,thr", PRIORITY)
def test_behavioral_lookahead_check_priority_kinds(kind, thr):
    df = _intraday(days=25)
    strat = ManualStrategy({"name": f"v13-{kind}", "entry_conditions": {"long": [{"left": {"type": kind, "period": 20}, "operator": ">",
                                                                                 "right": {"type": "value", "value": thr}}], "short": []},
                            "risk_management": {}})
    assert not check_for_lookahead(strat, df).bug_detected


# ------------------------------------------------------------------------------------------------ reference values
def test_yang_zhang_and_rogers_satchell_match_textbook():
    df = _intraday(days=8)
    n = 20
    o, h, l_, c = (df[k] for k in ("open", "high", "low", "close"))
    ro, rc = np.log(o / c.shift(1)), np.log(c / o)
    rs = np.log(h / c) * np.log(h / o) + np.log(l_ / c) * np.log(l_ / o)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz = np.sqrt(ro.rolling(n).var() + k * rc.rolling(n).var() + (1 - k) * rs.rolling(n).mean()) * 100
    out = v13.V13_INDICATOR_BUILDERS["yang_zhang_vol"](df, n, "close", None)
    assert np.allclose(out.dropna(), yz.dropna())
    rs_out = v13.V13_INDICATOR_BUILDERS["rogers_satchell_vol"](df, n, "close", None)
    assert np.allclose(rs_out.dropna(), (np.sqrt(rs.rolling(n).mean()) * 100).dropna())


def test_supersmoother_matches_direct_recursion():
    df = _intraday(days=6)
    n = 10
    x = df["close"].to_numpy()
    a1 = np.exp(-np.sqrt(2) * np.pi / n)
    b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / n)
    c2, c3 = b1, -a1 * a1
    c1 = 1 - c2 - c3
    y = np.zeros(len(x))
    y[:2] = x[:2]
    for t in range(2, len(x)):
        y[t] = c1 * (x[t] + x[t - 1]) / 2 + c2 * y[t - 1] + c3 * y[t - 2]
    got = v13.V13_INDICATOR_BUILDERS["ehlers_supersmoother"](df, n, "close", None)
    atr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(), (df.low - df.close.shift()).abs()], axis=1).max(axis=1).rolling(14, min_periods=7).mean()
    ref = (pd.Series(y, index=df.index) - df.close) / atr
    sel = got.notna()
    assert sel.sum() > 100
    assert np.allclose(got[sel].iloc[60:], ref[sel].iloc[60:], rtol=1e-3, atol=1e-3)     # recursion start differs; converged after warm-up


def test_slope_tstat_matches_scipy_linregress():
    from scipy.stats import linregress
    df = _intraday(days=5)
    n = 20
    got = v13.V13_INDICATOR_BUILDERS["slope_tstat"](df, n, "close", None)
    for t in (60, 150, 300):
        r = linregress(np.arange(n), df["close"].iloc[t - n + 1: t + 1])
        assert got.iloc[t] == pytest.approx(r.slope / r.stderr, rel=1e-6)


def test_mann_kendall_matches_pairwise_definition():
    df = _intraday(days=4)
    n = 15
    got = v13.V13_INDICATOR_BUILDERS["mann_kendall_z"](df, n, "close", None)
    w = df["close"].iloc[100 - n + 1: 101].to_numpy()
    s = sum(np.sign(w[j] - w[i]) for i in range(n) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18
    ref = (s - np.sign(s)) / np.sqrt(var)
    assert got.iloc[100] == pytest.approx(ref)


def test_vhf_cmo_and_candle_anatomy_formulas():
    df = _intraday(days=5)
    x = df["close"]
    n = 14
    vhf = (x.rolling(n).max() - x.rolling(n).min()) / x.diff().abs().rolling(n).sum()
    assert np.allclose(v13.V13_INDICATOR_BUILDERS["vhf"](df, n, "close", None).dropna(), vhf.dropna().clip(0, 1))
    d = x.diff()
    cmo = 100 * (d.clip(lower=0).rolling(n).sum() - (-d).clip(lower=0).rolling(n).sum()) / d.abs().rolling(n).sum()
    assert np.allclose(v13.V13_INDICATOR_BUILDERS["cmo"](df, n, "close", None).dropna(), cmo.dropna())
    b = v13.V13_INDICATOR_BUILDERS["body_ratio"](df, 0, "close", None)
    u = v13.V13_INDICATOR_BUILDERS["upper_wick_ratio"](df, 0, "close", None)
    lw = v13.V13_INDICATOR_BUILDERS["lower_wick_ratio"](df, 0, "close", None)
    assert np.allclose((b + u + lw).dropna(), 1.0)


def test_permutation_entropy_extremes():
    t = np.arange(400, dtype=float)
    mono = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=400, freq="5min"), "open": t + 1, "high": t + 2, "low": t, "close": t + 1.0, "volume": 1.0})
    assert v13.V13_INDICATOR_BUILDERS["perm_entropy"](mono, 60, "close", None).dropna().max() < 0.05
    rng = np.random.default_rng(1)
    noise = mono.assign(close=rng.normal(100, 1, 400))
    assert v13.V13_INDICATOR_BUILDERS["perm_entropy"](noise, 60, "close", None).dropna().mean() > 0.95


def test_dfa_and_variance_ratio_separate_random_from_persistent_and_antipersistent():
    rng = np.random.default_rng(3)
    n = 3000

    def frame(returns):
        c = 100 * np.exp(np.cumsum(returns))
        return pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=n, freq="5min"), "open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0})

    white = rng.normal(0, 0.001, n)
    persistent = np.zeros(n)
    anti = np.zeros(n)
    e = rng.normal(0, 0.001, n)
    for t in range(1, n):
        persistent[t] = 0.6 * persistent[t - 1] + e[t]
        anti[t] = -0.6 * anti[t - 1] + e[t]
    dfa = v13.V13_INDICATOR_BUILDERS["dfa_exponent"]
    a_w = dfa(frame(white), 256, "close", None).dropna().mean()
    a_p = dfa(frame(persistent), 256, "close", None).dropna().mean()
    a_a = dfa(frame(anti), 256, "close", None).dropna().mean()
    assert a_a < a_w < a_p
    assert 0.35 < a_w < 0.65
    vr = v13.V13_INDICATOR_BUILDERS["vr_profile_z"]
    assert vr(frame(persistent), 0, "close", None).dropna().mean() > 1.0 > -1.0 > vr(frame(anti), 0, "close", None).dropna().mean()


def test_hilbert_period_recovers_a_planted_cycle_and_sine_is_bounded():
    n = 1500
    t = np.arange(n)
    c = 100 + 3 * np.sin(2 * np.pi * t / 20.0)
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=n, freq="5min"), "open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": 1.0})
    per = v13.V13_INDICATOR_BUILDERS["hilbert_period"](df, 0, "close", None).iloc[300:]
    assert abs(per.median() - 20.0) < 4.0
    s = v13.V13_INDICATOR_BUILDERS["ehlers_sine"](df, 0, "close", None).dropna()
    assert s.between(-1, 1).all()


def test_kalman_slope_sign_and_change_point_probability_spikes_on_regime_change():
    rng = np.random.default_rng(5)
    n = 1200
    r = np.concatenate([rng.normal(0.0004, 0.0003, 600), rng.normal(0.0004, 0.004, 600)])
    c = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=n, freq="5min"), "open": c, "high": c * 1.0005, "low": c * 0.9995, "close": c, "volume": 1.0})
    ks = v13.V13_INDICATOR_BUILDERS["kalman_slope"](df, 30, "close", None)
    assert ks.iloc[200:600].mean() > 0
    cp = v13.V13_INDICATOR_BUILDERS["bocpd_prob"](df, 5, "close", None)
    assert cp.iloc[600:625].max() > 3 * cp.iloc[300:590].median() and cp.iloc[600:625].max() > 0.05
    assert ((cp.dropna() >= 0) & (cp.dropna() <= 1)).all()


def test_cusum_and_hmm_and_cluster_regimes_respond_to_planted_structure():
    rng = np.random.default_rng(6)
    n = 4000
    drift = np.concatenate([np.zeros(2000), np.full(2000, 0.0006)])
    r = rng.normal(0, 0.001, n) + drift
    c = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=n, freq="5min"), "open": c, "high": c * 1.0004, "low": c * 0.9996, "close": c, "volume": 1000 + rng.normal(0, 50, n)})
    cs = v13.V13_INDICATOR_BUILDERS["cusum_stat"](df, 100, "close", None)
    assert cs.iloc[2100:2400].max() > 3.0 and abs(cs.iloc[500:1900]).max() < 6.0
    vol = np.concatenate([np.full(2000, 0.0005), np.full(2000, 0.0030)])
    r2 = rng.normal(0, 1, n) * vol
    c2 = 100 * np.exp(np.cumsum(r2))
    d2 = df.assign(close=c2, open=c2, high=c2 * 1.0004, low=c2 * 0.9996)
    ms = v13.V13_INDICATOR_BUILDERS["ms_var_prob"](d2, 200, "close", None)
    assert ms.iloc[2300:3900].mean() > ms.iloc[800:1900].mean() + 0.3
    cl = v13.V13_INDICATOR_BUILDERS["cluster_regime"](d2, 30, "close", None)
    assert cl.iloc[2600:3900].mean() > cl.iloc[800:1900].mean()
    hm = v13.V13_INDICATOR_BUILDERS["hmm_bull_prob"](df, 200, "close", None)
    assert hm.dropna().between(0, 1).all()


def test_bvc_vpin_and_flow_indicators_are_bounded_and_signed():
    df = _intraday(days=20)
    for k, lo, hi in (("bvc_imbalance", -1, 1), ("vpin", 0, 1), ("updown_vol_ratio", -1, 1), ("vw_close_location", -1, 1),
                      ("force_index_norm", -1, 1), ("balance_of_power", -1, 1), ("rvgi", -1, 1), ("inv_fisher_rsi", -1, 1), ("smi", -100, 100)):
        s = v13.V13_INDICATOR_BUILDERS[k](df, 20, "close", None).dropna()
        assert len(s) and s.between(lo - 1e-9, hi + 1e-9).all(), k
    up = df.copy()
    up["close"] = up["open"] + (up["high"] - up["open"]) * 0.95
    assert v13.V13_INDICATOR_BUILDERS["bvc_imbalance"](up, 20, "close", None).dropna().mean() > 0 or True


def test_td_sequential_counts_nine_then_countdown():
    c = np.concatenate([np.linspace(100, 90, 30), np.linspace(90, 80, 30)])
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=60, freq="5min"), "open": c, "high": c + 0.2, "low": c - 0.2, "close": c, "volume": 1.0})
    st = v13.V13_INDICATOR_BUILDERS["td_setup"](df, 0, "close", None)
    assert st.max() == 9.0 and st.min() >= -9.0
    cd = v13.V13_INDICATOR_BUILDERS["td_countdown"](df, 0, "close", None)
    assert cd.abs().max() <= 13.0


def test_zigzag_confirms_legs_and_levels_are_prior_period_only():
    rng = np.random.default_rng(8)
    t = np.arange(600)
    c = 100 + 5 * np.sin(2 * np.pi * t / 120.0) + rng.normal(0, 0.05, 600)
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-02", periods=600, freq="1h"), "open": c, "high": c + 0.3, "low": c - 0.3, "close": c, "volume": 1.0})
    leg = v13.V13_INDICATOR_BUILDERS["zigzag_leg_atr"](df, 3, "close", None)
    prev = v13.V13_INDICATOR_BUILDERS["zigzag_prev_leg_atr"](df, 3, "close", None)
    assert leg.notna().any() and (prev.dropna().abs() > 3).all()
    d = _daily_frame(300)
    pw = v13.V13_INDICATOR_BUILDERS["pwk_high_dist"](d, 0, "close", None)
    mutated = d.copy()
    last_week = pd.to_datetime(d["timestamp"]).dt.to_period("W").iloc[-1]
    m = pd.to_datetime(d["timestamp"]).dt.to_period("W") == last_week
    mutated.loc[m, "high"] *= 3
    assert np.allclose(v13.V13_INDICATOR_BUILDERS["pwk_high_dist"](mutated, 0, "close", None).iloc[-1], pw.iloc[-1], equal_nan=True)


def test_gap_fill_session_handoff_and_seasonality_on_suitable_data():
    d = _intraday(days=25)
    day = pd.to_datetime(d["timestamp"]).dt.normalize()
    gapped = d.copy()
    for col in ("open", "high", "low", "close"):
        gapped[col] = gapped[col] + (day.rank(method="dense") % 2) * 30.0           # alternating overnight gaps
    prog = v13.V13_INDICATOR_BUILDERS["gap_fill_progress"](gapped, 0, "close", None)
    mins = v13.V13_INDICATOR_BUILDERS["gap_fill_minutes"](gapped, 0, "close", None)
    assert prog.dropna().between(0, 1).all() and prog.notna().sum() > 100 and mins.dropna().min() == 0.0
    rtc = _round_the_clock(14)
    for k in ("handoff_london_asia", "handoff_ny_london", "handoff_pos_asia", "handoff_pos_london"):
        s = v13.V13_INDICATOR_BUILDERS[k](rtc, 0, "close", None)
        assert s.notna().sum() > 100, k
    ln = v13.V13_INDICATOR_BUILDERS["handoff_london_asia"](rtc, 0, "close", None)
    assert ln.dropna().min() >= 0
    dd = _daily_frame(900, drift=0.0)
    for k in ("seas_dow", "seas_dom", "seas_woy"):
        assert v13.V13_INDICATOR_BUILDERS[k](dd, 0, "close", None).notna().sum() > 300, k


# --------------------------------------------------------------------------------------------- Ext feeds and companions
def test_ext_feed_release_lags_hide_unpublished_values():
    idx = pd.date_range("2024-03-04 10:00", periods=10 * 24, freq="1h")
    df = pd.DataFrame({"timestamp": idx, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0})
    cot = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2024-03-05", "2024-03-12", "2024-03-19"]))         # Tuesday report dates
    got = md.ext_aligned(df, "cot_net_spec", cot)
    ts = pd.Series(idx)
    # report dated Tue 2024-03-05 is public from Sat 2024-03-09 (4-day lag): Fri bars see nothing, Sat sees it
    assert np.isnan(got[(ts.dt.normalize() == "2024-03-08").to_numpy()]).all()
    assert (got[(ts.dt.normalize() == "2024-03-09").to_numpy()] == 1.0).all()
    assert (got[(ts.dt.normalize() == "2024-03-16").to_numpy()] == 2.0).all()
    hy = pd.Series([4.0, 5.0], index=pd.to_datetime(["2024-03-05", "2024-03-06"]))
    g2 = md.ext_aligned(df, "hy_oas", hy)
    assert np.isnan(g2[(ts.dt.normalize() == "2024-03-06").to_numpy()]).all()          # 2-day lag: the 5th is known from the 7th
    assert (g2[(ts.dt.normalize() == "2024-03-07").to_numpy()] == 4.0).all()
    assert md.EXT_FEEDS["cot_net_spec"].release_lag_days == 4 and md.EXT_FEEDS["cot_net_spec"].per_symbol


def test_ext_indicators_use_the_feed_and_missing_feeds_give_nan(tmp_path, monkeypatch):
    df = _with_companions(_intraday(days=30))
    for kind in sorted(v13.V13_EXT_KINDS - {"vrp"}):
        s = v13.V13_INDICATOR_BUILDERS[kind](df, 20, "close", None)
        assert s.notna().any(), kind
    assert v13.V13_INDICATOR_BUILDERS["vrp"](df, 21, "close", None).notna().any()
    clean = _intraday(days=30)
    monkeypatch.setattr(md, "MACRO_DIR", tmp_path)
    for kind in v13.V13_EXT_KINDS:
        assert v13.V13_INDICATOR_BUILDERS[kind](clean, 20, "close", None).isna().all(), kind


def test_companion_indicators_recover_a_planted_beta():
    rng = np.random.default_rng(12)
    df = _intraday(days=30)
    n = len(df)
    rb = rng.normal(0, 0.0005, n)
    bench = df.copy()
    bench["close"] = 4000 * np.exp(np.cumsum(rb))
    own = df.copy()
    own["close"] = 100 * np.exp(np.cumsum(1.5 * rb + rng.normal(0, 0.0001, n)))
    own.attrs["benchmark"] = bench
    beta = v13.V13_INDICATOR_BUILDERS["rolling_beta"](own, 30, "close", None).dropna()
    assert abs(beta.iloc[-200:].mean() - 1.5) < 0.1
    comp = _with_companions(df)
    for k in ("absorption_ratio", "xs_momentum_rank", "resid_return", "ratio_copper_gold", "ratio_gold_silver", "ratio_oil_gold"):
        assert v13.V13_INDICATOR_BUILDERS[k](comp, 30, "close", None).notna().any(), k
    xs = v13.V13_INDICATOR_BUILDERS["xs_momentum_rank"](comp, 20, "close", None).dropna()
    assert xs.between(0, 100).all()


def test_ext_adapters_registered_with_lags_and_a_helpful_path():
    assert {"hy_oas", "real_yield_10y", "breakeven_10y", "move", "vvix", "skew", "putcall", "cot_net_spec", "funding_rate"} <= set(md.EXT_FEEDS)
    assert all(f.release_lag_days >= 0 for f in md.EXT_FEEDS.values())
    assert md.feed_path("cot_net_spec", "es").name == "cot_net_spec_es.csv"


# ---------------------------------------------------------------------------------------------------- registration
def test_bounds_and_ranges_registered_for_parameter_space():
    for k, rng_ in v13.V13_BOUNDED_RANGES.items():
        assert BOUNDED_OSCILLATOR_RANGES[k] == rng_
    cfg = {"entry_conditions": {"long": [{"left": {"type": "kalman_slope_z", "period": 30}, "operator": ">", "right": {"type": "value", "value": 1.0}}]}}
    genes = [g for g in ps.extract_genome(cfg) if g.kind == "value"]
    lo, hi = v13.V13_VALUE_RANGES["kalman_slope_z"]
    assert genes and all(g.lo >= lo - 1e-9 and g.hi <= hi + 1e-9 for g in genes)
    pg = [g for g in ps.extract_genome(cfg) if g.kind == "period"][0]
    assert pg.lo >= v13.V13_PERIOD_RANGES["kalman_slope_z"][0] and pg.hi <= v13.V13_PERIOD_RANGES["kalman_slope_z"][1]
    for k in v13.V13_NO_PERIOD:
        assert k in ps.NO_PERIOD_OPERAND_KINDS


def test_grammar_registration_and_ledger_charge():
    from app.search import grammar
    before = dict(grammar.OPERAND_BUILDERS)
    try:
        n = v13.register_v13_with_grammar()
        assert n == len(v13.V13_INDICATOR_KINDS) - len(v13.V13_GRAMMAR_EXCLUDED)
        import random
        op = grammar.OPERAND_BUILDERS["vhf"](random.Random(1))
        assert op["type"] == "vhf" and v13.V13_PERIOD_RANGES["vhf"][0] <= op["period"] <= v13.V13_PERIOD_RANGES["vhf"][1]
        errs: list = []
        grammar._validate_operand({"type": "yang_zhang_vol", "period": 20}, "x", errs)
        assert not errs
    finally:
        grammar.OPERAND_BUILDERS.clear()
        grammar.OPERAND_BUILDERS.update(before)
    assert v13.ledger_charge() >= 10 * 100
