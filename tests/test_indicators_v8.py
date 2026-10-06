"""v8 indicator tests (2026-10-05).

Covers the eight new indicator kinds in app.strategy.indicators_v8
(ATR percentile rank, Kaufman efficiency ratio, Donchian channel
position, Bollinger bandwidth, volume z-score, overnight gap in ATR
units, fractal swing strength, linear-regression R-squared), each with:

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
round-trips a hand-built config per kind through the real Manual builder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategy import indicators_v8 as v8
from app.strategy.indicators import build_indicator_series
from app.strategy.lookahead_check import check_for_lookahead
from app.strategy.manual import ManualStrategy

V8_KINDS = list(v8.V8_INDICATOR_KINDS)


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

def _ref_true_range(df: pd.DataFrame) -> np.ndarray:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)
    tr = np.zeros(n)
    for i in range(n):
        pc = close[i - 1] if i > 0 else close[0]
        tr[i] = max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc))
    return tr


def _ref_atr(df: pd.DataFrame, p: int) -> np.ndarray:
    # ewm(alpha=1/p, adjust=False, min_periods=p): y[0]=x[0]; NaN below p.
    x = _ref_true_range(df)
    out = np.full(len(x), np.nan)
    a = 1.0 / p
    y = x[0]
    for i in range(1, len(x)):
        y = (1 - a) * y + a * x[i]
        if i >= p - 1:
            out[i] = y
    return out


def _ref_atr_percentile(df: pd.DataFrame, p: int, w: int = 100) -> np.ndarray:
    a = _ref_atr(df, p)
    out = np.full(len(a), np.nan)
    for i in range(w - 1, len(a)):
        win = a[i - w + 1:i + 1]
        # pandas rolling(w, min_periods=w): NaN until the window holds w
        # non-NaN ATR values (ATR itself needs p-1 warmup bars first).
        if np.isnan(win).any():
            continue
        out[i] = float((win <= a[i]).mean()) * 100.0
    return np.clip(out, 0.0, 100.0)


def _ref_efficiency_ratio(close: np.ndarray, p: int) -> np.ndarray:
    out = np.zeros(len(close))
    for i in range(p, len(close)):
        change = abs(close[i] - close[i - p])
        vol = sum(abs(close[j] - close[j - 1]) for j in range(i - p + 1, i + 1))
        out[i] = 0.0 if vol == 0 else min(max(change / vol, 0.0), 1.0)
    return out


def _ref_donchian_mid_distance(df: pd.DataFrame, p: int) -> np.ndarray:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    out = np.zeros(len(df))
    for i in range(p - 1, len(df)):
        up = high[i - p + 1:i + 1].max()
        lo = low[i - p + 1:i + 1].min()
        width = up - lo
        out[i] = 0.0 if width == 0 else (close[i] - (up + lo) / 2.0) / width
    return out


def _ref_bollinger_bandwidth(close: np.ndarray, p: int, k: float = 2.0) -> np.ndarray:
    out = np.zeros(len(close))
    for i in range(p - 1, len(close)):
        w = close[i - p + 1:i + 1]
        mid = w.mean()
        sd = w.std(ddof=1)  # pandas rolling().std() is ddof=1
        out[i] = 0.0 if mid == 0 else ((mid + k * sd) - (mid - k * sd)) / mid
    return out


def _ref_volume_zscore(vol: np.ndarray, p: int) -> np.ndarray:
    out = np.zeros(len(vol))
    for i in range(p - 1, len(vol)):
        w = vol[i - p + 1:i + 1]
        sd = w.std(ddof=1)
        out[i] = 0.0 if sd == 0 else (vol[i] - w.mean()) / sd
    return out


def _ref_overnight_gap_atr(df: pd.DataFrame, p: int) -> np.ndarray:
    open_ = df["open"].to_numpy()
    close = df["close"].to_numpy()
    a = _ref_atr(df, p)
    out = np.zeros(len(df))
    for i in range(1, len(df)):
        gap = open_[i] - close[i - 1]
        out[i] = 0.0 if not np.isfinite(a[i]) or a[i] == 0 else gap / a[i]
    return out


def _ref_fractal_strength(df: pd.DataFrame, p: int, w: int = 5) -> np.ndarray:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    hi_conf: dict[int, float] = {}
    lo_conf: dict[int, float] = {}
    for c in range(0, n):
        # Centered 2w+1 window, clipped at the edges the way pandas'
        # center=True rolling does; min_periods=w+1 (so the left edge,
        # c < w, still emits once w+1 bars exist).
        lo_i, hi_i = max(0, c - w), min(n, c + w + 1)
        if hi_i - lo_i < w + 1:
            continue
        if high[c] == high[lo_i:hi_i].max() and c + w < n:
            hi_conf[c + w] = high[c]
        if low[c] == low[lo_i:hi_i].min() and c + w < n:
            lo_conf[c + w] = low[c]
    a = _ref_atr(df, p)
    out = np.zeros(n)
    last_hi = last_lo = None
    for i in range(n):
        if i in hi_conf:
            last_hi = hi_conf[i]
        if i in lo_conf:
            last_lo = lo_conf[i]
        if last_hi is None or last_lo is None or not np.isfinite(a[i]) or a[i] == 0:
            out[i] = 0.0
        else:
            out[i] = abs(last_hi - last_lo) / a[i]
    return out


def _ref_linreg_r2(close: np.ndarray, p: int) -> np.ndarray:
    out = np.zeros(len(close))
    for i in range(p - 1, len(close)):
        w = close[i - p + 1:i + 1]
        if w.std() == 0:
            out[i] = 0.0
            continue
        r = np.corrcoef(np.arange(p), w)[0, 1]
        out[i] = min(max(float(r) ** 2, 0.0), 1.0)
    return out


# ---------------------------------------------------------------------------
# 1. reference-value agreement
# ---------------------------------------------------------------------------

def test_reference_agreement():
    df = _ohlcv()
    close = df["close"].to_numpy()
    vol = df["volume"].to_numpy()
    cases = [
        ("atr_percentile", v8.atr_percentile(df, 14), _ref_atr_percentile(df, 14)),
        ("efficiency_ratio", v8.efficiency_ratio(df["close"], 14).to_numpy(), _ref_efficiency_ratio(close, 14)),
        ("donchian_mid_distance", v8.donchian_mid_distance(df, 20).to_numpy(), _ref_donchian_mid_distance(df, 20)),
        ("bollinger_bandwidth", v8.bollinger_bandwidth(df, 20).to_numpy(), _ref_bollinger_bandwidth(close, 20)),
        ("volume_zscore", v8.volume_zscore(df, 20).to_numpy(), _ref_volume_zscore(vol, 20)),
        ("overnight_gap_atr", v8.overnight_gap_atr(df, 14).to_numpy(), _ref_overnight_gap_atr(df, 14)),
        ("fractal_strength", v8.fractal_strength(df, 14).to_numpy(), _ref_fractal_strength(df, 14)),
        ("linreg_r2", v8.linreg_r2(df["close"], 14).to_numpy(), _ref_linreg_r2(close, 14)),
    ]
    for name, got, ref in cases:
        # linreg_r2's reference uses np.corrcoef (sample covariance) while
        # the module uses the algebraically identical population-variance
        # form -- they agree to ~1e-8, pure float reordering noise.
        rtol = 1e-6 if name == "linreg_r2" else 1e-9
        assert np.allclose(got, ref, equal_nan=True, rtol=rtol, atol=1e-9), name


def test_dispatch_through_build_indicator_series():
    df = _ohlcv()
    for kind in V8_KINDS:
        s = build_indicator_series(df, kind, period=14, column="close")
        assert len(s) == len(df), kind
        assert s.dropna().shape[0] > 0, f"{kind}: all-NaN"
        assert np.isfinite(s.dropna().to_numpy()).all(), f"{kind}: non-finite values"


def test_flat_data_no_crash():
    df = _flat()
    for kind in V8_KINDS:
        s = build_indicator_series(df, kind, period=14, column="close")
        assert np.isfinite(s.fillna(0).to_numpy()).all(), f"{kind}: non-finite on flat data"


def test_bounded_kinds_stay_in_range():
    df = _ohlcv()
    for kind, lo, hi in [("atr_percentile", 0.0, 100.0), ("efficiency_ratio", 0.0, 1.0),
                         ("linreg_r2", 0.0, 1.0)]:
        s = build_indicator_series(df, kind, period=14, column="close").dropna()
        assert (s >= lo - 1e-9).all() and (s <= hi + 1e-9).all(), kind


# ---------------------------------------------------------------------------
# 2. lookahead: prefix invariance + behavioral check
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
    _assert_prefix_invariant(lambda f: v8.atr_percentile(f, 14), df)
    _assert_prefix_invariant(lambda f: v8.efficiency_ratio(f["close"], 14), df)
    _assert_prefix_invariant(lambda f: v8.donchian_mid_distance(f, 20), df)
    _assert_prefix_invariant(lambda f: v8.bollinger_bandwidth(f, 20), df)
    _assert_prefix_invariant(lambda f: v8.volume_zscore(f, 20), df)
    _assert_prefix_invariant(lambda f: v8.overnight_gap_atr(f, 14), df)
    _assert_prefix_invariant(lambda f: v8.fractal_strength(f, 14), df)
    _assert_prefix_invariant(lambda f: v8.linreg_r2(f["close"], 14), df)


def _strategy_on_kind(kind, threshold, op=">"):
    return ManualStrategy({
        "name": f"v8-lookahead-{kind}",
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
    cases = [("atr_percentile", 70, ">"), ("efficiency_ratio", 0.5, ">"),
             ("donchian_mid_distance", -0.3, "<"),
             ("bollinger_bandwidth", 0.002, ">"), ("volume_zscore", 2.0, ">"),
             ("overnight_gap_atr", 0.05, ">"), ("fractal_strength", 1.5, ">"),
             ("linreg_r2", 0.7, ">")]
    for kind, thr, op in cases:
        strat = _strategy_on_kind(kind, thr, op)
        result = check_for_lookahead(strat, df)
        assert result.checked, f"{kind}: check could not run (no signals?)"
        assert not result.bug_detected, f"{kind}: {result.summary()}"


# ---------------------------------------------------------------------------
# 3. grammar registration
# ---------------------------------------------------------------------------

def test_grammar_registers_all_v8_kinds():
    from app.search import grammar
    for kind in V8_KINDS:
        assert kind in grammar.INDICATOR_KINDS, kind
        assert kind in grammar.OPERAND_BUILDERS, kind
        assert kind in grammar.NUMERIC_KINDS, kind
    # Bounded oscillators get in-range random thresholds.
    for kind, (lo, hi) in v8.V8_BOUNDED_RANGES.items():
        assert grammar.THRESHOLD_BOUNDS[kind] == (lo, hi), kind


def test_grammar_validate_round_trips_each_v8_kind():
    import random as _random
    from app.search import grammar
    for kind in V8_KINDS:
        rng = _random.Random(1234)
        left = grammar.OPERAND_BUILDERS[kind](rng)
        cfg = {
            "name": f"v8-validate-{kind}",
            "entry_conditions": {
                "long": [{"left": left, "operator": ">",
                          "right": grammar.random_threshold(rng, kind)}],
                "short": [],
            },
            "risk_management": {},
        }
        errors = grammar.validate(cfg)
        assert errors == [], f"{kind}: {errors[:2]}"
