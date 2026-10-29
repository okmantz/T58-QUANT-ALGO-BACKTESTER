"""2026-10 lookahead audit: EVERY indicator kind (base + v7..v12) must give the same value on bar i whether or not later bars exist.

A Discovery winner was blocked in Full Pipeline because `adr` stamped each day's FULL range onto its earlier intraday bars; the
per-version tests did not cover the base kinds. This test covers all of them in one place so a future leak cannot slip in."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from app.strategy import (indicators_v7, indicators_v8, indicators_v9, indicators_v10, indicators_v11, indicators_v12)
from app.strategy.indicators import average_daily_range, build_indicator_series
from tests.backtest.test_indicators_v12 import _intraday

NEEDS_COMPANION = {"correlation", "pair_ratio", "pair_zscore", "news_minutes_since_high_impact", "news_minutes_until_high_impact"}


def _base_kinds():
    import app.strategy.indicators as I
    src = Path(I.__file__).read_text()
    a = src.index("def _build_indicator_series_uncached")
    b = src.index("INDICATOR_FUNCS = {")
    return sorted(set(re.findall(r'kind == "([a-z0-9_]+)"', src[a:b])) - NEEDS_COMPANION)


def _builder_kinds():
    out = []
    for m in (indicators_v7, indicators_v8, indicators_v9, indicators_v10, indicators_v11, indicators_v12):
        name = [x for x in dir(m) if x.endswith("_INDICATOR_BUILDERS")][0]
        out += list(getattr(m, name))
    return out


def test_every_base_and_v7_to_v12_kind_is_prefix_invariant():
    df = _intraday(days=45)
    cuts = [1500, 2200, int(len(df) * 0.7)]
    bad = []
    for kind in _base_kinds() + _builder_kinds():
        for cut in cuts:
            a = build_indicator_series(df, kind, 14).iloc[:cut].to_numpy(float)
            b = build_indicator_series(df.iloc[:cut].copy(), kind, 14).to_numpy(float)
            if not np.allclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-9):
                bad.append(kind)
                break
    assert not bad, f"these kinds look into the future: {bad}"


def test_adr_uses_only_completed_days():
    df = _intraday(days=30)
    full = average_daily_range(df, 5)
    cut = 1000
    part = average_daily_range(df.iloc[:cut].copy(), 5)
    assert np.allclose(full.iloc[:cut].to_numpy(float), part.to_numpy(float), equal_nan=True)
