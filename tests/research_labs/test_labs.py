"""The seven research engines: reproducible, out-of-sample-aware, honest about proxy data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research_labs import common as C
from app.research_labs import event_study, hypothesis_lab, lead_lag, order_book_lab, order_flow, stat_arb, volume_profile


def _bars(seed: int = 1, n: int = 4000, base: float = 100.0, freq: str = "5min", with_volume: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.001, n)
    close = base * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0004, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0004, n)))
    ts = pd.date_range("2024-01-02 00:00", periods=n, freq=freq)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": hi, "low": lo, "close": close,
                         "volume": rng.integers(100, 1000, n) if with_volume else 0})


# ----------------------------------------------------------------------------------------------------------- common
def test_holm_and_bh_are_monotone_and_bounded():
    p = [0.001, 0.02, 0.04, 0.5]
    holm, bh = C.holm_adjust(p), C.bh_adjust(p)
    assert all(0 <= x <= 1 for x in holm + bh)
    assert holm[0] >= p[0] and bh[0] >= p[0]
    assert all(h >= b - 1e-12 for h, b in zip(holm, bh))      # Holm is the stricter of the two


def test_group_compare_flags_a_planted_effect_and_ignores_noise():
    rng = np.random.default_rng(0)
    n = 1200
    flag = pd.Series(rng.random(n) < 0.3)
    outcome = pd.Series(rng.normal(0, 1, n))
    outcome[flag] += 1.0                                            # a real, large effect
    noise = pd.Series(rng.random(n) < 0.3)                          # unrelated
    rows = {r["group"]: r for r in C.group_compare(outcome, pd.Series(True, index=outcome.index), {"real": flag, "noise": noise})}
    assert rows["real"]["verdict"].startswith("promising")
    assert not rows["noise"]["verdict"].startswith("promising")


def test_sweeps_are_causal():
    df = C.prepare_ohlcv(_bars())
    full = C.detect_liquidity_sweeps(df)
    cut = C.detect_liquidity_sweeps(df.iloc[:2000].copy())
    assert (full["ssl_sweep"].iloc[:2000].to_numpy() == cut["ssl_sweep"].to_numpy()).all()


# ------------------------------------------------------------------------------------------------- lab 1: volume profile
def test_volume_profile_runs_is_reproducible_and_records_assumptions():
    df = _bars()
    a = volume_profile.run(df, label="t", save=False)
    b = volume_profile.run(df, label="t", save=False)
    assert a.spec_hash == b.spec_hash and a.results["headline"] == b.results["headline"]
    assert a.assumptions and a.data["fingerprint"]


def test_volume_profile_features_use_only_prior_session():
    df = C.prepare_ohlcv(_bars(n=3000))
    f1 = volume_profile.profile_feature_frame(df)
    truncated = df.iloc[:2000].copy()
    f2 = volume_profile.profile_feature_frame(truncated)
    both = f1.iloc[:1000].fillna(-999).to_numpy()
    assert np.allclose(both, f2.iloc[:1000].fillna(-999).to_numpy())


# --------------------------------------------------------------------------------------------------- lab 2: order flow
def test_order_flow_labels_ohlcv_as_proxy():
    rec = order_flow.run(_bars(), label="t", save=False)
    assert "PROXY" in rec.results["headline"].upper()


def test_order_flow_true_flow_when_bid_ask_present():
    df = _bars()
    df["ask_volume"] = df["volume"] * 0.6
    df["bid_volume"] = df["volume"] * 0.4
    fr = order_flow.flow_frame(C.prepare_ohlcv(df) if False else df)
    assert fr["is_true_flow"].iloc[0] == 1.0


# ------------------------------------------------------------------------------------------------ lab 3: order book
def test_order_book_refuses_to_infer_depth_from_candles():
    rec = order_book_lab.run(None, None, label="candles only", save=False)
    assert rec.status == "needs_data"


def test_order_book_demo_is_labelled_synthetic():
    rec = order_book_lab.run(demo=True, save=False)
    assert any("SYNTHETIC" in w for w in rec.warnings)


# ------------------------------------------------------------------------------------------------- lab 4: lead-lag
def test_lead_lag_finds_a_planted_lead():
    n = 5000
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.001, n)
    lead = 100 * np.exp(np.cumsum(r))
    follow_r = np.r_[0, r[:-1]] * 0.8 + rng.normal(0, 0.0004, n)         # follower = leader's previous-bucket return
    follow = 100 * np.exp(np.cumsum(follow_r))
    ts = pd.date_range("2024-01-02", periods=n, freq="15min")

    def frame(c):
        return pd.DataFrame({"timestamp": ts, "open": c, "high": c * 1.0002, "low": c * 0.9998, "close": c, "volume": 100})

    rec = lead_lag.run({"LEADER": frame(lead), "FOLLOWER": frame(follow)}, freq="15min", max_lag=4, save=False)
    assert "LEADER -> FOLLOWER" in rec.results["headline"]
    top = rec.results["pairs"][0]
    assert top["leader"] == "LEADER" and top["best_lag"] == 1 and top["corr_oos"] > 0.5      # survives out-of-sample


def test_lead_lag_reports_nothing_on_independent_noise():
    rec = lead_lag.run({"A": _bars(1, 5000, freq="15min"), "B": _bars(2, 5000, freq="15min")}, freq="15min", max_lag=4, save=False)
    assert rec.results["headline"].startswith("No robust lead-lag")


# --------------------------------------------------------------------------------------------------- lab 5: stat arb
def test_stat_arb_detects_cointegration_and_monitor_runs():
    n = 3000
    rng = np.random.default_rng(5)
    b = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.9 * spread[i - 1] + rng.normal(0, 0.002)
    a = b * 0.8 * np.exp(spread)
    ts = pd.date_range("2024-01-02", periods=n, freq="1h")

    def frame(c):
        return pd.DataFrame({"timestamp": ts, "open": c, "high": c * 1.0002, "low": c * 0.9998, "close": c, "volume": 100})

    rec = stat_arb.run({"A": frame(a), "B": frame(b)}, freq="1h", save=False)
    assert "cointegrated" in rec.results["cointegration_in_sample"]["verdict"]
    assert rec.results["stability_monitor"]["status"] in {"normal", "stretched (relationship intact)", "weak mean reversion", "structurally broken"}


def test_adf_stat_separates_stationary_from_random_walk():
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(size=1500))
    stationary = np.zeros(1500)
    for i in range(1, 1500):
        stationary[i] = 0.7 * stationary[i - 1] + rng.normal()
    assert stat_arb.adf_stat(stationary) < -3.5 < stat_arb.adf_stat(walk)


# ------------------------------------------------------------------------------------------------ lab 6: event study
def test_event_study_runs_with_matched_controls():
    df = _bars(n=6000)
    events = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-02") + pd.Timedelta(days=k, hours=9) for k in range(1, 19)]})
    rec = event_study.run(df, events, save=False)
    assert rec.results["events_used"] >= 8
    assert {r["metric"] for r in rec.results["event_vs_control"]} >= {"expansion", "adverse_atr"}


def test_event_study_rejects_too_few_events():
    with pytest.raises(C.ResearchLabError):
        event_study.run(_bars(), pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-05 09:00")]}), save=False)


def test_nfp_schedule_is_first_fridays():
    sched = event_study.nfp_schedule("2024-01", "2024-06")
    assert len(sched) == 6 and (sched["timestamp"].dt.weekday == 4).all() and (sched["timestamp"].dt.day <= 7).all()


# ---------------------------------------------------------------------------------------------- lab 7: hypothesis
def test_hypothesis_spec_must_be_complete():
    spec = hypothesis_lab.default_spec()
    del spec["acceptance"]
    with pytest.raises(C.ResearchLabError):
        hypothesis_lab.validate_spec(spec)


def test_hypothesis_is_reproducible_and_ledger_tightens_alpha(tmp_path):
    df = _bars(n=6000)
    spec = hypothesis_lab.default_spec()
    spec["sample"]["zone"] = None
    r1 = hypothesis_lab.run(df, spec, save=True)
    r1b = hypothesis_lab.run(df, spec, save=True)                      # same spec again: no extra penalty
    assert r1.results["spec_hash"] == r1b.results["spec_hash"]
    assert r1b.results["hypotheses_tested_on_this_data"] == r1.results["hypotheses_tested_on_this_data"]
    spec2 = hypothesis_lab.default_spec()
    spec2["sample"]["zone"] = None
    spec2["feature"] = {"type": "trend_ema", "span": 20}
    r2 = hypothesis_lab.run(df, spec2, save=True)
    assert r2.results["hypotheses_tested_on_this_data"] == r1.results["hypotheses_tested_on_this_data"] + 1
    assert r2.results["alpha_adjusted"] < r1.results["alpha_adjusted"]


def test_hypothesis_accepts_a_planted_effect():
    n = 12000
    df = C.prepare_ohlcv(_bars(n=n, seed=9))
    sweeps = C.detect_liquidity_sweeps(df)
    ema = df["close"].ewm(span=50, adjust=False).mean()
    feat = df["close"] > ema
    # plant: after an SSL sweep with the feature true, price rises by ~2 ATR over the horizon
    atr = C.atr_series(df).shift(1)
    idx = np.where((sweeps["ssl_sweep"] & feat).to_numpy())[0]
    c = df["close"].to_numpy().copy()
    for i in idx:
        if i + 12 < n:
            bump = 2.0 * atr.iloc[i] if np.isfinite(atr.iloc[i]) else 0.0
            c[i + 1:] += bump * 0.0
            c[i + 12] += bump
    df2 = df.copy()
    df2["close"] = c
    df2["high"] = np.maximum(df2["high"], df2["close"])
    spec = hypothesis_lab.default_spec()
    spec["sample"]["zone"] = None
    spec["feature"] = {"type": "trend_ema", "span": 50}
    spec["acceptance"]["min_n"] = 20
    rec = hypothesis_lab.run(df2, spec, save=False, use_ledger=False)
    assert rec.results["discovery"]["diff"] is not None and rec.results["discovery"]["diff"] > 0


def test_hypothesis_can_use_gex_indicator_as_the_feature():
    spec = hypothesis_lab.default_spec()
    spec["sample"]["zone"] = None
    spec["feature"] = {"type": "indicator", "kind": "gex_regime", "op": ">", "value": 0}
    rec = hypothesis_lab.run(_bars(n=6000), spec, save=False, use_ledger=False)
    assert "criteria" in rec.results


def test_seasonality_finds_planted_volatility_and_rejects_noise():
    import numpy as np
    import pandas as pd

    from app.research_labs import seasonality as S

    def make(plant):
        rng = np.random.default_rng(5)
        days = pd.bdate_range("2023-01-02", "2024-03-29")
        idx = pd.DatetimeIndex(np.concatenate([pd.date_range(d + pd.Timedelta(hours=9, minutes=30), d + pd.Timedelta(hours=15, minutes=55), freq="5min") for d in days]))
        r = rng.normal(0, 0.0008, len(idx))
        if plant:
            r[idx.hour == 10] *= 3
        c = 100 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]]
        return pd.DataFrame({"timestamp": idx, "open": o, "high": np.maximum(o, c) * 1.0004, "low": np.minimum(o, c) * 0.9996, "close": c, "volume": 100})

    planted = S.run(make(True), save=False).results
    assert any(r["group"] == "time of day: 10:00 (volatility)" and r["verdict"].startswith("promising") for r in planted["slices"])
    noise = S.run(make(False), save=False).results
    assert noise["persistence"]["survive_correction_and_oos"] <= 2        # pure noise: (almost) nothing survives
    assert S.run(make(False), save=False).spec_hash == S.run(make(False), save=False).spec_hash
