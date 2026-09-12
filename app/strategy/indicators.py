"""Technical indicators and deterministic market-derived series for T58."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _period(period: int) -> int:
    return max(int(period), 1)


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=_period(period), min_periods=_period(period)).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    p = _period(period)
    return series.ewm(span=p, adjust=False, min_periods=p).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    p = _period(period)
    weights = np.arange(1, p + 1, dtype=float)
    return series.rolling(p, min_periods=p).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    p = _period(period)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / p, adjust=False, min_periods=p).mean()
    avg_loss = loss.ewm(alpha=1 / p, adjust=False, min_periods=p).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss.ne(0), 100)
    return result.fillna(50)


def true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    return pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - prev_close).abs(),
        (frame["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    p = _period(period)
    return true_range(frame).ewm(alpha=1 / p, adjust=False, min_periods=p).mean()


def vwap(frame: pd.DataFrame) -> pd.Series:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = frame["volume"] if "volume" in frame.columns else pd.Series(1.0, index=frame.index)
    ts = pd.to_datetime(frame["timestamp"])
    day = ts.dt.normalize()
    pv = typical * volume
    return pv.groupby(day).cumsum() / volume.groupby(day).cumsum().replace(0, np.nan)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    line = fast_ema - slow_ema
    signal_line = line.ewm(span=_period(signal), adjust=False, min_periods=_period(signal)).mean()
    histogram = line - signal_line
    return line, signal_line, histogram


def stdev(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling (population, ddof=0) standard deviation -- Pine's ta.stdev()
    and a common building block for custom volatility filters/normalized
    thresholds that aren't already one of the named indicators above."""
    return series.rolling(_period(period), min_periods=_period(period)).std(ddof=0)


def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    p = _period(period)
    mid = sma(series, p)
    std = series.rolling(p, min_periods=p).std(ddof=0)
    return mid, mid + std_mult * std, mid - std_mult * std


def highest_high(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(_period(period), min_periods=_period(period)).max()


def lowest_low(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(_period(period), min_periods=_period(period)).min()


def average_volume(series: pd.Series, period: int = 20) -> pd.Series:
    return sma(series, period)


def candle_range(frame: pd.DataFrame) -> pd.Series:
    return frame["high"] - frame["low"]


def percentage_change(series: pd.Series, period: int = 1) -> pd.Series:
    return series.pct_change(_period(period)) * 100.0


def relative_volume(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current bar's volume divided by its own trailing average -- >1 means
    this bar traded on above-average participation. Falls back to a
    constant 1.0 series when the data has no volume column (e.g. some FX
    feeds), so a strategy that references it simply never fires rather
    than crashing."""
    if "volume" not in frame.columns:
        return pd.Series(1.0, index=frame.index)
    volume = frame["volume"]
    avg = average_volume(volume, period)
    return (volume / avg.replace(0, np.nan)).fillna(1.0)


def volume_delta(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling sum of signed volume (candle-direction-signed: up-close bars
    contribute +volume, down-close bars contribute -volume) over `period`
    bars, normalized by the rolling sum of total volume so the result is a
    dimensionless imbalance ratio in roughly [-1, 1] regardless of the
    instrument's absolute volume scale. A simple, transparent proxy for
    order-flow / buying-vs-selling pressure imbalance -- not a true
    tick-level bid/ask delta (this app only has OHLCV bars, not trade-by-
    trade prints), but directionally meaningful and, crucially, computed
    only from already-closed bars (no lookahead)."""
    if "volume" not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    signed = frame["volume"].where(frame["close"] >= frame["open"], -frame["volume"])
    p = _period(period)
    signed_sum = signed.rolling(p, min_periods=p).sum()
    total_sum = frame["volume"].rolling(p, min_periods=p).sum()
    return (signed_sum / total_sum.replace(0, np.nan)).fillna(0.0)


def pair_ratio(frame: pd.DataFrame, pair_column: str = "pair_close") -> pd.Series:
    """Price ratio of this instrument's close to a second instrument's close
    that has already been merged into `frame` as `pair_column` (see
    app.data.pairs.merge_pair_series). Requires the merge step to have
    happened first -- raises KeyError otherwise so a mis-set-up pairs
    strategy fails loudly instead of silently trading on garbage."""
    if pair_column not in frame.columns:
        raise KeyError(
            f"'{pair_column}' not found in market data -- a pairs/relative-value "
            "strategy requires the second instrument's close to be merged in first "
            "via app.data.pairs.merge_pair_series()."
        )
    return frame["close"] / frame[pair_column].replace(0, np.nan)


def pair_zscore(frame: pd.DataFrame, period: int = 50, pair_column: str = "pair_close") -> pd.Series:
    """Rolling z-score of the two-instrument price ratio -- the standard
    statistical-arbitrage signal: how many standard deviations the current
    ratio sits from its own trailing mean. A large positive/negative
    z-score is the classic 'spread has stretched, bet on reversion' entry
    trigger for a pairs strategy."""
    ratio = pair_ratio(frame, pair_column)
    p = _period(period)
    mean = ratio.rolling(p, min_periods=p).mean()
    std = ratio.rolling(p, min_periods=p).std(ddof=0)
    return ((ratio - mean) / std.replace(0, np.nan)).fillna(0.0)


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (equivalent to an EMA with alpha=1/period) -- the
    specific averaging method ADX/+DI/-DI and Wilder's own RSI are defined
    with, kept as a separate helper so adx() below reads as a direct
    transcription of the standard definition rather than reusing rsi()'s
    EMA (which already hard-codes the 100/(1+rs) RSI-specific finish)."""
    p = _period(period)
    return series.ewm(alpha=1 / p, adjust=False, min_periods=p).mean()


def adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder) -- a trend-STRENGTH filter (0-100,
    no direction), used to gate entries so a breakout/pullback/trend family
    only fires when the market is actually trending rather than chopping.
    Standard definition: +DM/-DM from consecutive high/low deltas (each
    zeroed out unless it's both positive and larger than the other side),
    Wilder-smoothed and normalized by smoothed True Range into +DI/-DI, then
    ADX is the Wilder-smoothed |+DI - -DI| / (+DI + -DI) * 100."""
    p = _period(period)
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=frame.index)
    tr_smooth = _wilder_smooth(true_range(frame), p)
    plus_di = 100 * _wilder_smooth(plus_dm, p) / tr_smooth.replace(0, np.nan)
    minus_di = 100 * _wilder_smooth(minus_dm, p) / tr_smooth.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder_smooth(dx.fillna(0.0), p)


def stochastic(frame: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator (%K, %D) -- current close's position within the
    trailing high/low range, smoothed. Bounded 0-100; standard oversold/
    overbought reversal reads are <20 / >80."""
    p = _period(period)
    hh = highest_high(frame["high"], p)
    ll = lowest_low(frame["low"], p)
    raw_k = 100 * (frame["close"] - ll) / (hh - ll).replace(0, np.nan)
    k = raw_k.rolling(_period(smooth_k), min_periods=_period(smooth_k)).mean()
    d = k.rolling(_period(smooth_d), min_periods=_period(smooth_d)).mean()
    return k.fillna(50.0), d.fillna(50.0)


def cci(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index -- typical price's deviation from its rolling
    mean, normalized by mean absolute deviation (the standard 0.015 constant
    makes ~+/-100 the typical range). An extreme reading (>+100 / <-100) is
    the classic CCI reversion/breakout trigger, distinct from RSI in that it
    is unbounded rather than clamped to 0-100."""
    p = _period(period)
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    sma_tp = sma(typical, p)
    mad = typical.rolling(p, min_periods=p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return ((typical - sma_tp) / (0.015 * mad.replace(0, np.nan))).fillna(0.0)


def obv(frame: pd.DataFrame) -> pd.Series:
    """On-Balance Volume -- cumulative volume, added on up-close bars and
    subtracted on down-close bars. Used for divergence: price makes a new
    extreme that OBV doesn't confirm. Falls back to an all-zero series when
    the data has no volume column, same convention as volume_delta()."""
    if "volume" not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    direction = np.sign(frame["close"].diff().fillna(0.0))
    return (direction * frame["volume"]).cumsum()


def obv_ema(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    """EMA of On-Balance Volume -- OBV's own short-term TREND (not its raw
    cumulative level, which is unbounded and non-stationary across a long
    dataset and therefore useless to compare directly against a threshold
    or another period's EMA on raw levels would still work, but is kept as
    its own named indicator here for clarity at the call site)."""
    return ema(obv(frame), period)


def keltner(frame: pd.DataFrame, period: int = 20, atr_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channel (EMA midline +/- ATR multiple) -- an ATR-based
    volatility band, distinct from Bollinger's stdev-based band: it widens
    with directional range expansion rather than close-to-close dispersion,
    so a Keltner squeeze/breakout can disagree with a Bollinger one on the
    same bar. Returns (mid, upper, lower)."""
    p = _period(period)
    mid = ema(frame["close"], p)
    band = atr(frame, p) * atr_mult
    return mid, mid + band, mid - band


def donchian(frame: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channel (rolling high/low envelope + midline) -- the
    original turtle-trader breakout band. Distinct from the app's existing
    `_breakout_flag`/`bos` primitive (which is a one-shot "did price just
    clear the prior N-bar extreme" boolean): this exposes the band's actual
    levels as continuous series, e.g. for a midline-fade or band-width
    filter rather than only a breakout trigger.

    Upper/lower are computed over the PRIOR `period` bars (high/low shifted
    by 1 before the rolling max/min), same non-lookahead convention as
    `_breakout_flag`'s own prior_high/prior_low -- a window that includes
    the current bar's own high/low would make "close > upper" structurally
    always-false (a bar's close can never exceed its own bar's high).
    Returns (mid, upper, lower)."""
    p = _period(period)
    upper = highest_high(frame["high"].shift(1), p)
    lower = lowest_low(frame["low"].shift(1), p)
    return (upper + lower) / 2.0, upper, lower


def supertrend(frame: pd.DataFrame, period: int = 10, atr_mult: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """SuperTrend -- a flip-based trend-following band: the trailing stop
    line ratchets toward price and only flips side when price closes through
    it. Returns (line, direction) where direction is +1.0 while price is
    above the line (uptrend) and -1.0 while below (downtrend). Implemented
    as a straightforward sequential ratchet (each bar's line depends on the
    prior bar's line AND prior direction, which is not expressible as a
    single vectorized rolling op) -- consistent with wma()'s own use of a
    rolling .apply for the same reason; cost is negligible next to a single
    backtest's own bar-by-bar simulation loop.

    Bars before ATR has warmed up (the first `period` bars) get NaN/neutral
    output, same convention as every other indicator here -- the ratchet
    only starts once ATR itself is defined, since comparing against a NaN
    band with plain `<`/`>` (both False for NaN) would otherwise freeze the
    band at NaN forever once it first went undefined.
    """
    p = _period(period)
    atr_series = atr(frame, p)
    hl2 = (frame["high"] + frame["low"]) / 2.0
    basic_upper = (hl2 + atr_mult * atr_series).to_numpy()
    basic_lower = (hl2 - atr_mult * atr_series).to_numpy()
    atr_values = atr_series.to_numpy()
    close = frame["close"].to_numpy()
    n = len(frame)
    line = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    started = False
    for i in range(n):
        if np.isnan(atr_values[i]):
            continue
        if not started:
            # First bar with a defined ATR -- initialize the ratchet fresh
            # from this bar's own basic bands rather than carrying forward
            # whatever (NaN) value sat in final_upper/final_lower during
            # warmup.
            direction[i] = 1.0
            line[i] = basic_lower[i]
            started = True
            continue
        if not (basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]):
            final_upper[i] = final_upper[i - 1]
        if not (basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]):
            final_lower[i] = final_lower[i - 1]
        prev_direction = direction[i - 1]
        if prev_direction == 1.0:
            direction[i] = -1.0 if close[i] < final_lower[i] else 1.0
        else:
            direction[i] = 1.0 if close[i] > final_upper[i] else -1.0
        line[i] = final_lower[i] if direction[i] == 1.0 else final_upper[i]
    return pd.Series(line, index=frame.index), pd.Series(direction, index=frame.index)


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def build_indicator_series(frame: pd.DataFrame, kind: str, period: int = 14, column: str = "close", lookback: int | None = None) -> pd.Series:
    """Thin caching wrapper -- see app.strategy.indicator_cache for why.
    The actual per-kind math is unchanged, in _build_indicator_series_uncached
    below; every existing caller and behavior is identical, just memoized
    per (frame, kind, period, column, lookback) within this process."""
    from app.strategy import indicator_cache

    return indicator_cache.get_or_compute(
        frame, kind, period, column, lookback,
        compute_fn=lambda: _build_indicator_series_uncached(frame, kind, period, column, lookback),
    )


def _build_indicator_series_uncached(frame: pd.DataFrame, kind: str, period: int = 14, column: str = "close", lookback: int | None = None) -> pd.Series:
    kind = kind.lower()
    p = _period(period)
    source = frame[column] if column in frame.columns else frame["close"]

    if kind == "sma":
        return sma(source, p)
    if kind == "ema":
        return ema(source, p)
    if kind == "wma":
        return wma(source, p)
    if kind == "rsi":
        return rsi(source, p)
    if kind == "vwap":
        return vwap(frame)
    if kind == "atr":
        return atr(frame, p)
    if kind == "stdev":
        return stdev(source, p)
    if kind == "macd":
        return macd(source)[0]
    if kind == "macd_signal":
        return macd(source)[1]
    if kind == "macd_histogram":
        return macd(source)[2]
    if kind == "bollinger_mid":
        return bollinger(source, p)[0]
    if kind == "bollinger_upper":
        return bollinger(source, p)[1]
    if kind == "bollinger_lower":
        return bollinger(source, p)[2]
    if kind == "highest_high":
        return highest_high(frame["high"], p)
    if kind == "lowest_low":
        return lowest_low(frame["low"], p)
    if kind == "average_volume":
        volume = frame["volume"] if "volume" in frame.columns else pd.Series(1.0, index=frame.index)
        return average_volume(volume, p)
    if kind == "candle_range":
        return candle_range(frame)
    if kind == "percentage_change":
        return percentage_change(source, p)
    if kind == "relative_volume":
        return relative_volume(frame, p)
    if kind == "volume_delta":
        return volume_delta(frame, p)
    if kind == "pair_ratio":
        return pair_ratio(frame)
    if kind == "pair_zscore":
        return pair_zscore(frame, p)
    if kind == "adx":
        return adx(frame, p)
    if kind == "stoch_k":
        return stochastic(frame, p)[0]
    if kind == "stoch_d":
        return stochastic(frame, p)[1]
    if kind == "cci":
        return cci(frame, p)
    if kind == "obv":
        return obv(frame)
    if kind == "obv_ema":
        return obv_ema(frame, p)
    if kind == "keltner_mid":
        return keltner(frame, p)[0]
    if kind == "keltner_upper":
        return keltner(frame, p)[1]
    if kind == "keltner_lower":
        return keltner(frame, p)[2]
    if kind == "donchian_mid":
        return donchian(frame, p)[0]
    if kind == "donchian_upper":
        return donchian(frame, p)[1]
    if kind == "donchian_lower":
        return donchian(frame, p)[2]
    if kind == "supertrend_line":
        return supertrend(frame, p)[0]
    if kind == "supertrend_direction":
        return supertrend(frame, p)[1]
    raise KeyError(kind)


# Legacy PineScript/MQL5 adapters expect this mapping to contain callables
# accepting (series, period). Keep those functions intact and add the new
# generic visual-builder series separately.
INDICATOR_FUNCS = {
    "sma": sma,
    "ema": ema,
    "wma": wma,
    "rsi": rsi,
}
