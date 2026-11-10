"""Data-health cards: each problem is named in plain words, and each fix removes exactly that problem."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.health_cards import FIX_BAD_BARS, FIX_DUPES, FIX_ROLLS, FIX_SORT, apply_fix, build_cards


def _frame(splice=0.0):
    rng = np.random.default_rng(1)
    days = [pd.date_range(d + pd.Timedelta(hours=1), periods=92, freq="15min") for d in pd.bdate_range("2024-02-01", periods=30)]
    ts = pd.DatetimeIndex(np.concatenate([x.values for x in days]))
    c = 4000 + np.cumsum(rng.normal(0, 1, len(ts)))
    df = pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1})
    if splice:
        df.loc[(len(ts) // 2 // 92) * 92:, ["open", "high", "low", "close"]] += splice
    return df


def _ids(df):
    return [c["id"] for c in build_cards(df)]


def test_clean_data_gets_a_single_all_clear_card():
    cards = build_cards(_frame())
    assert [c["id"] for c in cards] == ["clean"] and cards[0]["fix"] is None


def test_roll_splice_card_quotes_points_and_the_fix_clears_it():
    df = _frame(splice=60.0)
    card = next(c for c in build_cards(df) if c["id"] == "rolls")
    assert "points of contract-roll splice" in card["title"] and card["fix"]["id"] == FIX_ROLLS
    fixed, _ = apply_fix(df, FIX_ROLLS)
    assert _ids(fixed) == ["clean"] and len(fixed) == len(df)


def test_duplicates_and_order_are_flagged_and_fixed():
    df = _frame()
    messy = pd.concat([df.iloc[100:], df.iloc[:100], df.iloc[[5]]], ignore_index=True)
    assert {"order", "dupes"} <= set(_ids(messy))
    assert "order" not in _ids(apply_fix(messy, FIX_SORT)[0])
    assert "dupes" not in _ids(apply_fix(messy, FIX_DUPES)[0])


def test_impossible_bars_are_flagged_and_removed():
    df = _frame()
    df.loc[10, "high"] = df.loc[10, "low"] - 5
    df.loc[20, "close"] = float("nan")
    assert "bad_bars" in _ids(df)
    fixed, msg = apply_fix(df, FIX_BAD_BARS)
    assert len(fixed) == len(df) - 2 and "2" in msg and "bad_bars" not in _ids(fixed)


def test_unknown_fix_is_rejected():
    with pytest.raises(ValueError):
        apply_fix(_frame(), "rm -rf")
