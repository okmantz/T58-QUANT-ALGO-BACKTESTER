"""
VWAP Trend Continuation Strategy
---------------------------------
Long:  EMA(294) > EMA(58)  AND  Close > RollingVWAP(4)  AND  RSI(18) < 57.6102629644959
Short: EMA(35)  < EMA(159) AND  Close < RollingVWAP(3)  AND  RSI(42) > 66.008585331847

Exit (signal-based, checked in addition to stop/target):
Long:  RSI(5) > 66.68737207001313
Short: RSI(5) < 64.42442560822603

Risk management: ATR-based stop and target, opposite-signal exit, max bars in trade.

Note: "RollingVWAP(n)" here is a non-anchored, n-bar rolling volume-weighted
average price (typical price * volume, summed over the last n bars) since the
spec calls for a fixed lookback "period" rather than a session anchor.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Strategy parameters (verbatim from spec)
# ---------------------------------------------------------------------------
PARAMS = dict(
    ema_long_slow=294,
    ema_long_fast=58,
    ema_short_fast=35,
    ema_short_slow=159,
    rsi_long_period=18,
    rsi_long_thresh=57.6102629644959,
    rsi_short_period=42,
    rsi_short_thresh=66.008585331847,
    rsi_exit_period=5,
    rsi_exit_long_thresh=66.68737207001313,
    rsi_exit_short_thresh=64.42442560822603,
    vwap_long_period=4,
    vwap_short_period=3,
    atr_stop_period=16,
    atr_stop_mult=4.806786275425379,
    atr_target_period=11,
    atr_target_mult=5.054720923946611,
    max_bars_in_trade=63,
    opposite_signal_exit=True,
)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100)  # avg_loss == 0 -> RSI saturates at 100


def rolling_vwap(df: pd.DataFrame, period: int) -> pd.Series:
    """Non-anchored, `period`-bar rolling volume-weighted average price."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    return pv.rolling(period).sum() / df["volume"].rolling(period).sum()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def build_indicators(df: pd.DataFrame, p: dict = PARAMS) -> pd.DataFrame:
    df = df.copy()
    df["ema_long_slow"] = ema(df["close"], p["ema_long_slow"])
    df["ema_long_fast"] = ema(df["close"], p["ema_long_fast"])
    df["ema_short_fast"] = ema(df["close"], p["ema_short_fast"])
    df["ema_short_slow"] = ema(df["close"], p["ema_short_slow"])

    df["rsi_long"] = rsi(df["close"], p["rsi_long_period"])
    df["rsi_short"] = rsi(df["close"], p["rsi_short_period"])
    df["rsi_exit"] = rsi(df["close"], p["rsi_exit_period"])

    df["vwap_long"] = rolling_vwap(df, p["vwap_long_period"])
    df["vwap_short"] = rolling_vwap(df, p["vwap_short_period"])

    df["atr_stop"] = atr(df, p["atr_stop_period"])
    df["atr_target"] = atr(df, p["atr_target_period"])
    return df


def generate_signals(df: pd.DataFrame, p: dict = PARAMS) -> pd.DataFrame:
    df = build_indicators(df, p)

    df["long_entry"] = (
        (df["ema_long_slow"] > df["ema_long_fast"])
        & (df["close"] > df["vwap_long"])
        & (df["rsi_long"] < p["rsi_long_thresh"])
    )
    df["short_entry"] = (
        (df["ema_short_fast"] < df["ema_short_slow"])
        & (df["close"] < df["vwap_short"])
        & (df["rsi_short"] > p["rsi_short_thresh"])
    )
    df["long_exit_signal"] = df["rsi_exit"] > p["rsi_exit_long_thresh"]
    df["short_exit_signal"] = df["rsi_exit"] < p["rsi_exit_short_thresh"]
    return df


# ---------------------------------------------------------------------------
# Bar-by-bar backtest
# ---------------------------------------------------------------------------
def backtest(df: pd.DataFrame, p: dict = PARAMS, starting_equity: float = 10000.0) -> pd.DataFrame:
    """
    df must have columns: open, high, low, close, volume (chronological order).

    Signals are evaluated on the *previous, fully closed* bar and filled at the
    current bar's open (no look-ahead). Stops/targets are checked intrabar via
    high/low. Only one position is held at a time.
    """
    df = generate_signals(df, p).reset_index(drop=True)

    trades = []
    position = None  # dict: side, entry_bar, entry_price, stop, target
    equity = starting_equity

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if position is not None:
            bars_held = i - position["entry_bar"]
            exit_price, exit_reason = None, None

            if position["side"] == "long":
                if row["low"] <= position["stop"]:
                    exit_price, exit_reason = position["stop"], "stop"
                elif row["high"] >= position["target"]:
                    exit_price, exit_reason = position["target"], "target"
                elif prev["long_exit_signal"]:
                    exit_price, exit_reason = row["open"], "rsi_exit"
                elif p["opposite_signal_exit"] and prev["short_entry"]:
                    exit_price, exit_reason = row["open"], "opposite_signal"
                elif bars_held >= p["max_bars_in_trade"]:
                    exit_price, exit_reason = row["open"], "max_bars"
            else:  # short
                if row["high"] >= position["stop"]:
                    exit_price, exit_reason = position["stop"], "stop"
                elif row["low"] <= position["target"]:
                    exit_price, exit_reason = position["target"], "target"
                elif prev["short_exit_signal"]:
                    exit_price, exit_reason = row["open"], "rsi_exit"
                elif p["opposite_signal_exit"] and prev["long_entry"]:
                    exit_price, exit_reason = row["open"], "opposite_signal"
                elif bars_held >= p["max_bars_in_trade"]:
                    exit_price, exit_reason = row["open"], "max_bars"

            if exit_price is not None:
                pnl = (
                    (exit_price - position["entry_price"])
                    if position["side"] == "long"
                    else (position["entry_price"] - exit_price)
                )
                equity += pnl
                trades.append(
                    {
                        **position,
                        "exit_bar": i,
                        "exit_price": exit_price,
                        "exit_reason": exit_reason,
                        "pnl": pnl,
                        "equity_after": equity,
                    }
                )
                position = None

        if position is None:
            if prev["long_entry"]:
                entry_price = row["open"]
                position = dict(
                    side="long",
                    entry_bar=i,
                    entry_price=entry_price,
                    stop=entry_price - p["atr_stop_mult"] * prev["atr_stop"],
                    target=entry_price + p["atr_target_mult"] * prev["atr_target"],
                )
            elif prev["short_entry"]:
                entry_price = row["open"]
                position = dict(
                    side="short",
                    entry_bar=i,
                    entry_price=entry_price,
                    stop=entry_price + p["atr_stop_mult"] * prev["atr_stop"],
                    target=entry_price - p["atr_target_mult"] * prev["atr_target"],
                )

    return pd.DataFrame(trades)


if __name__ == "__main__":
    # Example usage:
    # df = pd.read_csv("your_ohlcv.csv", parse_dates=["timestamp"])
    # df = df.rename(columns=str.lower)  # ensure open/high/low/close/volume
    # trades = backtest(df)
    # print(trades)
    # print("Win rate:", (trades["pnl"] > 0).mean())
    print("Import this module and call backtest(df) with an OHLCV DataFrame "
          "(columns: open, high, low, close, volume).")
