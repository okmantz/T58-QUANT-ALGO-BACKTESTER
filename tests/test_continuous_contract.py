import numpy as np
import pandas as pd

from app.data.continuous_contract import back_adjust, detect_rolls


def _series():
    rng = np.random.default_rng(1)
    ts = pd.date_range("2024-02-26 00:00", periods=24 * 20, freq="1h")  # crosses Mar 8-15
    px = 4000 + np.cumsum(rng.normal(0, 1, len(ts)))
    df = pd.DataFrame(dict(timestamp=ts, open=px, high=px + 1, low=px - 1, close=px, volume=1))
    # roll on 2024-03-11 18:00 after a 3h halt: +40 point jump persisting afterwards
    k = int(np.argmax(ts >= pd.Timestamp("2024-03-11 18:00")))
    df = df.drop(index=[k - 1, k - 2, k - 3]).reset_index(drop=True)
    k -= 3
    for col in ("open", "high", "low", "close"):
        df.loc[k:, col] += 40.0
    return df, k


def test_roll_detected_and_removed_without_touching_recent_prices():
    df, k = _series()
    rolls = detect_rolls(df)
    assert [r.index for r in rolls] == [k] and abs(rolls[0].gap - 40) < 5
    adj, _ = back_adjust(df)
    assert abs(adj.loc[k, "open"] - adj.loc[k - 1, "close"]) < 5      # gap closed
    assert adj["close"].iloc[-1] == df["close"].iloc[-1]               # latest price untouched
    assert adj["roll_bar"].sum() == 1


def test_no_false_rolls_on_clean_series():
    df, _ = _series()
    clean = df.copy()
    px = 4000 + np.cumsum(np.random.default_rng(2).normal(0, 1, len(df)))
    for col in ("open", "close"):
        clean[col] = px
    clean["high"], clean["low"] = px + 1, px - 1
    assert detect_rolls(clean) == []
