"""v7 indicator tests (worker D, 2026-10-05).

Covers the eight new indicator kinds in app.strategy.indicators_v7
(StochRSI k/d, Wilder +DI/-DI, linear-regression slope, Hurst exponent,
KST, Coppock), each with:

  1. reference-value agreement -- an independent plain-loop reference
     implementation (written from the textbook definition, not by calling
     the module) must match the module's vectorized output;
  2. dispatch -- build_indicator_series(df, kind) resolves the kind;
  3. sane warmup/NaN behavior (leading NaNs, no crash on flat data);
  4. zero lookahead -- the series computed on a truncated frame must
     equal the full-frame prefix (a series that peeked at future bars
     would diverge), plus the repo's behavioral lookahead check
     (app.strategy.lookahead_check) run through a real Manual strategy
     trading on the kind.

Also covers the grammar registration (app.search.grammar): every new kind
is a registered terminal with sane random thresholds, and validate()
round-trips a hand-built config per kind through the real Manual builder
-- including one config with a v7 "timeframe" (MTF) operand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy import indicators_v7 as v7
from app.strategy.indicators import build_indicator_series
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.manual import ManualStrategy


def _ohlcv(n=600, seed=11):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-02", periods=n, freq="15min")
    drift = np.linspace(0, 40, n)
    close = 2000.0 + drift + np.cumsum(rng.normal(0, 0.6, n))
    spread = np.abs(rng.normal(0.5, 0.2, n))
    return pd.DataFrame({
        "timestamp": ts,
        "open": close - rng.normal(0, 0.2, n),
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": np.abs(rng.normal(200, 30, n)),
    })


def _flat(n=300, price=100.0):
    ts = pd.date_range("2023-01-02", periods=n, freq="15min")
    c = pd.Series([price] * n)
    return pd.DataFrame({
        "timestamp": ts, "open": c, "high": c + 0.01, "low": c - 0.01,
        "close": c, "volume": pd.Series([10.0] * n),
    })


# ---------------------------------------------------------------------------
# Independent reference implementations (plain loops, textbook definitions)
# ---------------------------------------------------------------------------

def _ref_sma(x: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(p - 1, len(x)):
        out[i] = x[i - p + 1:i + 1].mean()
    return out


def _ref_sma_pandas(x: np.ndarray, p: int) -> np.ndarray:
    """pandas rolling(p, min_periods=p).mean() semantics: NaNs are skipped,
    and a window needs p non-NaN values to produce a value."""
    out = np.full(len(x), np.nan)
    for i in range(p - 1, len(x)):
        w = x[i - p + 1:i + 1]
        v = w[~np.isnan(w)]
        if len(v) >= p:
            out[i] = v.mean()
    return out


def _ref_ewm_wilder(x: np.ndarray, p: int) -> np.ndarray:
    # ewm(alpha=1/p, adjust=False, min_periods=p): y[0]=x[0]; NaN below p.
    out = np.full(len(x), np.nan)
    a = 1.0 / p
    y = x[0]
    for i in range(1, len(x)):
        y = (1 - a) * y + a * x[i]
        if i >= p - 1:
            out[i] = y
    return out


def _ref_stochrsi_kd_from_rsi(rsi: np.ndarray, p: int):
    """Reference for the StochRSI *layer* (stochastic-of-RSI + %K/%D
    smoothing), taking the RSI series itself as given -- the repo's rsi()
    (Wilder smoothing, NaN filled with 50) is pre-existing, separately
    tested code; what is new here is everything below it."""
    raw = np.full(len(rsi), np.nan)
    for i in range(p - 1, len(rsi)):
        w = rsi[i - p + 1:i + 1]
        lo, hi = w.min(), w.max()
        raw[i] = 50.0 if hi == lo else 100 * (rsi[i] - lo) / (hi - lo)
    # Module does raw.fillna(50) then pandas-semantics SMA(3) twice.
    k = _ref_sma_pandas(np.where(np.isnan(raw), 50.0, raw), 3)
    d = _ref_sma_pandas(k, 3)
    return k, d


def _ref_dm(df: pd.DataFrame, p: int):
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
    tr = np.zeros(n)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - df["close"].to_numpy()[i - 1]),
                        abs(low[i] - df["close"].to_numpy()[i - 1]))
    s_tr = _ref_ewm_wilder(tr, p)
    s_plus = _ref_ewm_wilder(plus_dm, p)
    s_minus = _ref_ewm_wilder(minus_dm, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(s_tr == 0, np.nan, 100 * s_plus / s_tr)
        mdi = np.where(s_tr == 0, np.nan, 100 * s_minus / s_tr)
    return pdi, mdi


def _ref_linreg_slope(close: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    x = np.arange(p, dtype=float)
    for i in range(p - 1, len(close)):
        out[i] = np.polyfit(x, close[i - p + 1:i + 1], 1)[0]
    return out


def _ref_hurst(close: np.ndarray, p: int) -> np.ndarray:
    """Reference R/S Hurst on first differences (matches the module's
    corrected construction -- see its docstring for why levels are wrong)."""
    diffs = np.diff(close, prepend=np.nan)
    out = np.full(len(close), np.nan)
    for i in range(p, len(close)):
        w = diffs[i - p + 1:i + 1]
        if np.isnan(w).any():
            continue
        dev = w - w.mean()
        s = w.std(ddof=1)
        if s == 0 or not np.isfinite(s):
            continue
        z = np.cumsum(dev)
        r = z.max() - z.min()
        if r <= 0 or not np.isfinite(r):
            continue
        out[i] = np.log(r / s) / np.log(p)
    return out


def _ref_roc(close: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    for i in range(p, len(close)):
        out[i] = 100 * (close[i] - close[i - p]) / close[i - p] if close[i - p] != 0 else np.nan
    return out


def _ref_roc_filled(close: np.ndarray, p: int) -> np.ndarray:
    """Mirrors app.strategy.indicators.roc: percentage change, NaN filled
    with 0.0 (so the warmup is zeros, not NaNs)."""
    return np.where(np.isnan(_ref_roc(close, p)), 0.0, _ref_roc(close, p))


def _ref_wma(x: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    w = np.arange(1, p + 1, dtype=float)
    for i in range(p - 1, len(x)):
        seg = x[i - p + 1:i + 1]
        out[i] = np.dot(seg, w) / w.sum() if not np.isnan(seg).any() else np.nan
    return out


# ---------------------------------------------------------------------------
# 1. reference-value agreement
# ---------------------------------------------------------------------------

def test_stochrsi_matches_reference():
    from app.strategy.indicators import rsi as repo_rsi
    df = _ohlcv()
    k, d = v7.stochrsi(df, 14)
    rsi_vals = repo_rsi(df["close"], 14).to_numpy()
    rk, rd = _ref_stochrsi_kd_from_rsi(rsi_vals, 14)
    mask = ~np.isnan(rk)
    assert mask.sum() > 400
    assert np.allclose(k.to_numpy()[mask], rk[mask], atol=1e-6)
    mask_d = ~np.isnan(rd)
    assert mask_d.sum() > 400
    assert np.allclose(d.to_numpy()[mask_d], rd[mask_d], atol=1e-6)


def test_plus_minus_di_match_reference_and_adx_math():
    df = _ohlcv()
    rp, rm = _ref_dm(df, 14)
    pdi = v7.plus_di(df, 14).to_numpy()
    mdi = v7.minus_di(df, 14).to_numpy()
    mask = ~np.isnan(rp)
    assert mask.sum() > 400
    assert np.allclose(pdi[mask], rp[mask], atol=1e-8)
    assert np.allclose(mdi[mask], rm[mask], atol=1e-8)
    # +DI/-DI must live in [0, 100] wherever defined.
    assert np.nanmin(pdi) >= 0 and np.nanmax(pdi) <= 100
    assert np.nanmin(mdi) >= 0 and np.nanmax(mdi) <= 100


def test_linreg_slope_matches_polyfit():
    df = _ohlcv()
    got = v7.linreg_slope(df["close"], 14).to_numpy()
    ref = _ref_linreg_slope(df["close"].to_numpy(), 14)
    mask = ~np.isnan(ref)
    assert mask.sum() > 400
    assert np.allclose(got[mask], ref[mask], atol=1e-9)
    # Sign check on a clean trend: positive slope on rising data.
    rising = pd.Series(np.arange(200, dtype=float))
    assert v7.linreg_slope(rising, 20).iloc[-1] > 0
    falling = pd.Series(np.arange(200, dtype=float)[::-1])
    assert v7.linreg_slope(falling, 20).iloc[-1] < 0


def test_hurst_matches_reference():
    df = _ohlcv()
    got = v7.hurst_exponent(df["close"], 100).to_numpy()
    ref = _ref_hurst(df["close"].to_numpy(), 100)
    mask = ~np.isnan(ref)
    assert mask.sum() > 300
    assert np.allclose(got[mask], ref[mask], atol=1e-9)


def _ar1(n, phi, seed, sigma=1.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, sigma)
    return pd.Series(100 + np.cumsum(x))


def test_hurst_centers_random_walk_at_half():
    # A random walk's increments are white noise: H must center ~0.5.
    rw = pd.Series(100 + np.cumsum(np.random.default_rng(3).normal(0, 1, 3000)))
    h = v7.hurst_exponent(rw, 100).dropna()
    assert 0.45 < h.median() < 0.62, f"median={h.median():.3f}"


def test_hurst_distinguishes_persistent_from_antipersistent():
    # Positively autocorrelated increments -> H > 0.5; negatively
    # autocorrelated (mean-reverting) -> H < 0.5.
    pers = _ar1(3000, 0.4, seed=5)
    anti = _ar1(3000, -0.4, seed=6)
    h_pers = v7.hurst_exponent(pers, 100).dropna().median()
    h_anti = v7.hurst_exponent(anti, 100).dropna().median()
    assert h_pers > 0.55, f"persistent median={h_pers:.3f}"
    assert h_anti < 0.48, f"anti-persistent median={h_anti:.3f}"
    assert h_anti < h_pers


def test_kst_matches_reference():
    df = _ohlcv()
    close = df["close"].to_numpy()
    ref = (_ref_sma(_ref_roc_filled(close, 10), 10) + 2 * _ref_sma(_ref_roc_filled(close, 15), 10)
           + 3 * _ref_sma(_ref_roc_filled(close, 20), 10) + 4 * _ref_sma(_ref_roc_filled(close, 30), 15))
    got = v7.kst(df).to_numpy()
    mask = ~np.isnan(ref)
    assert mask.sum() > 400
    assert np.allclose(got[mask], ref[mask], atol=1e-8)


def test_coppock_matches_reference():
    df = _ohlcv()
    close = df["close"].to_numpy()
    ref = _ref_wma(_ref_roc_filled(close, 14) + _ref_roc_filled(close, 11), 10)
    got = v7.coppock(df).to_numpy()
    mask = ~np.isnan(ref)
    assert mask.sum() > 400
    assert np.allclose(got[mask], ref[mask], atol=1e-8)


# ---------------------------------------------------------------------------
# 2. dispatch through build_indicator_series (+ manual.py routing)
# ---------------------------------------------------------------------------

V7_KINDS = ["stochrsi_k", "stochrsi_d", "plus_di", "minus_di",
            "linreg_slope", "hurst_exponent", "kst", "coppock"]


def test_all_v7_kinds_dispatch_through_build_indicator_series():
    df = _ohlcv()
    for kind in V7_KINDS:
        s = build_indicator_series(df, kind, period=14)
        assert isinstance(s, pd.Series), kind
        assert s.notna().sum() > 100, kind  # produces real values, not all-NaN


def test_all_v7_kinds_dispatchable_from_manual_operand():
    # ManualStrategy._series_from_operand is what strategies actually call;
    # an unknown kind raises StrategyError here.
    df = _ohlcv()
    for kind in V7_KINDS:
        strat = ManualStrategy({"name": "t", "entry_conditions": {"long": [], "short": []}})
        s = strat._series_from_operand(df, {"type": kind, "period": 14}, "left")
        assert s.notna().sum() > 100, kind


# ---------------------------------------------------------------------------
# 3. warmup / NaN behavior
# ---------------------------------------------------------------------------

def test_warmup_lengths_are_sane():
    df = _ohlcv(n=600)
    # linreg_slope: exactly period-1 leading NaNs (pure rolling).
    assert v7.linreg_slope(df["close"], 20).isna().sum() == 19
    # hurst: period leading NaNs -- the first difference is NaN, so the
    # first full window of differences completes one bar later.
    assert v7.hurst_exponent(df["close"], 100).isna().sum() == 100
    # plus_di/minus_di: ewm min_periods -> period-1 leading NaNs.
    assert v7.plus_di(df, 14).isna().sum() == 13
    assert v7.minus_di(df, 14).isna().sum() == 13
    # kst: the repo's roc() fills warmup with 0.0, so the longest leg is
    # just SMA(15) -> 14 leading NaNs. Same for coppock: WMA(10) -> 9.
    assert v7.kst(df).isna().sum() == 14
    assert v7.coppock(df).isna().sum() == 9
    # stochrsi: raw.fillna(50) then pandas-semantics SMA(3) twice ->
    # %K valid from bar 2, %D from bar 4.
    assert v7.stochrsi(df, 14)[0].isna().sum() == 2
    assert v7.stochrsi(df, 14)[1].isna().sum() == 4


def test_flat_series_does_not_crash_and_hurst_is_nan():
    df = _flat()
    # Nothing may raise on degenerate input.
    v7.plus_di(df, 14)
    v7.minus_di(df, 14)
    v7.stochrsi(df, 14)[0]
    v7.stochrsi(df, 14)[1]
    v7.linreg_slope(df["close"], 14)
    v7.kst(df)
    v7.coppock(df)
    v7.hurst_exponent(df["close"], 50)
    # Flat slope is exactly 0; Hurst is undefined (NaN), never fabricated.
    assert (v7.linreg_slope(df["close"], 20).dropna() == 0).all()
    assert v7.hurst_exponent(df["close"], 50).isna().all()


def test_stochrsi_stays_bounded():
    df = _ohlcv()
    k, d = v7.stochrsi(df, 14)
    assert k.dropna().between(0, 100).all()
    assert d.dropna().between(0, 100).all()


# ---------------------------------------------------------------------------
# 4. zero lookahead: truncated-frame prefix invariance + behavioral check
# ---------------------------------------------------------------------------

def _assert_prefix_invariant(fn, df, cut=400):
    full = fn(df)
    trunc = fn(df.iloc[:cut].copy())
    a = full.iloc[:cut].to_numpy(dtype=float)
    b = trunc.to_numpy(dtype=float)
    assert len(a) == len(b) == cut
    assert np.allclose(a, b, equal_nan=True), "series changed when future bars were removed"


def test_no_lookahead_prefix_invariance():
    df = _ohlcv(n=600)
    _assert_prefix_invariant(lambda f: v7.stochrsi(f, 14)[0], df)
    _assert_prefix_invariant(lambda f: v7.stochrsi(f, 14)[1], df)
    _assert_prefix_invariant(lambda f: v7.plus_di(f, 14), df)
    _assert_prefix_invariant(lambda f: v7.minus_di(f, 14), df)
    _assert_prefix_invariant(lambda f: v7.linreg_slope(f["close"], 14), df)
    _assert_prefix_invariant(lambda f: v7.hurst_exponent(f["close"], 100), df)
    _assert_prefix_invariant(lambda f: v7.kst(f), df)
    _assert_prefix_invariant(lambda f: v7.coppock(f), df)


def _strategy_on_kind(kind, threshold, op=">"):
    return ManualStrategy({
        "name": f"v7-lookahead-{kind}",
        "entry_conditions": {
            "long": [{"left": {"type": kind, "period": 14},
                      "operator": op,
                      "right": {"type": "value", "value": threshold}}],
            "short": [],
        },
        "risk_management": {},
    })


def test_behavioral_lookahead_check_passes_for_each_kind():
    # The repo's own truncate-and-compare detector, run through a real
    # strategy that trades on each new kind.
    df = _ohlcv(n=600)
    cases = [("stochrsi_k", 70, ">"), ("stochrsi_d", 30, "<"),
             ("plus_di", 25, ">"), ("minus_di", 25, ">"),
             ("linreg_slope", 0, ">"), ("hurst_exponent", 0.55, ">"),
             ("kst", 0, ">"), ("coppock", 0, "<")]
    for kind, thr, op in cases:
        strat = _strategy_on_kind(kind, thr, op)
        result = check_for_lookahead(strat, df)
        assert result.checked, f"{kind}: check could not run"
        assert not result.bug_detected, f"{kind}: {result.summary()}"


# ---------------------------------------------------------------------------
# 5. grammar registration
# ---------------------------------------------------------------------------

def test_grammar_registers_all_v7_kinds():
    from app.search import grammar
    for kind in V7_KINDS:
        assert kind in grammar.INDICATOR_KINDS, kind
        assert kind in grammar.OPERAND_BUILDERS, kind
        assert kind in grammar.NUMERIC_KINDS, kind
    # Bounded oscillators get in-range random thresholds.
    for kind in ("stochrsi_k", "stochrsi_d", "plus_di", "minus_di"):
        assert grammar.THRESHOLD_BOUNDS[kind] == (0.0, 100.0)


def test_grammar_validate_round_trips_each_v7_kind():
    import random as _random
    from app.search import grammar
    for kind in V7_KINDS:
        rng = _random.Random(1234)
        left = grammar.OPERAND_BUILDERS[kind](rng)
        cfg = {
            "name": f"v7-validate-{kind}",
            "entry_conditions": {
                "long": [{"left": left, "operator": ">",
                          "right": grammar.random_threshold(rng, kind)}],
                "short": [],
            },
            "risk_management": {},
        }
        errors = grammar.validate(cfg)
        assert errors == [], f"{kind}: {errors[:2]}"


def test_grammar_validate_round_trips_mtf_operand():
    # A v7 "timeframe" operand must survive validate(): the round-trip
    # runs the same HTF data-prep the production backtest path runs.
    from app.search import grammar
    cfg = {
        "name": "v7-validate-mtf",
        "entry_conditions": {
            "long": [{"left": {"type": "rsi", "period": 14, "timeframe": "1h"},
                      "operator": ">",
                      "right": {"type": "value", "value": 55}}],
            "short": [],
        },
        "risk_management": {},
    }
    assert grammar._config_declares_timeframe(cfg) is True
    assert grammar.validate(cfg) == []


def test_generate_random_with_mtf_stays_valid():
    import random as _random
    from app.search import grammar
    prev = grammar.set_mtf_probability(1.0)  # force MTF on every indicator operand
    try:
        rng = _random.Random(99)
        cfgs = [grammar.generate_random(rng=rng) for _ in range(10)]
    finally:
        grammar.set_mtf_probability(prev)
    mtf_seen = sum(1 for c in cfgs if grammar._config_declares_timeframe(c))
    assert mtf_seen > 0, "forced MTF probability produced no timeframe operands"
    # generate_random already asserts validity internally; belt-and-braces:
    for c in cfgs:
        assert grammar.validate(c) == []


def test_generate_random_allow_mtf_false_emits_no_timeframes():
    import random as _random
    from app.search import grammar
    prev = grammar.set_mtf_probability(1.0)
    try:
        rng = _random.Random(99)
        cfgs = [grammar.generate_random(rng=rng, allow_mtf=False) for _ in range(10)]
    finally:
        grammar.set_mtf_probability(prev)
    assert all(not grammar._config_declares_timeframe(c) for c in cfgs)
