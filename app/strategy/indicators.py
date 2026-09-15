"""Technical indicators and deterministic market-derived series for T58."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Indicators whose value is only ever meaningful inside a fixed, known
# numeric range. A condition that compares one of these to a threshold
# outside its range (e.g. "RSI > 102.56") is not just an unusual setting
# -- it is IMPOSSIBLE, since the indicator can mathematically never reach
# that value. An impossible condition silently turns that branch of an
# AND-chain into permanent dead code: the strategy still "runs" and still
# produces trades and metrics from whatever conditions ARE live, with
# nothing in the output flagging that part of the intended entry/exit
# logic never fires at all. This is the single shared source of truth for
# that range, used by both app.optimize.parameter_space (so a GA can
# never mutate a condition's threshold outside it) and
# app.strategy.manual's validate_bounded_conditions (so a hand-built or
# already-saved config gets the same check surfaced as a plain warning).
BOUNDED_OSCILLATOR_RANGES: dict[str, tuple[float, float]] = {
    "rsi": (0.0, 100.0),
    "stoch_k": (0.0, 100.0),
    "stoch_d": (0.0, 100.0),
    "mfi": (0.0, 100.0),
    "connors_rsi": (0.0, 100.0),
    "ultimate_oscillator": (0.0, 100.0),
    "choppiness_index": (0.0, 100.0),
    "williams_r": (-100.0, 0.0),
}


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


def news_feature_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Reads a column app.data.economic_calendar.merge_news_features()
    already merged into `frame` (minutes_since_high_impact_news /
    minutes_until_high_impact_news). Same contract as pair_ratio() above:
    raises KeyError if the merge step hasn't happened yet, so a
    calendar-dependent family fails loudly rather than silently trading
    on garbage/NaN."""
    if column not in frame.columns:
        raise KeyError(
            f"'{column}' not found in market data -- an economic-calendar strategy "
            "requires news features to be merged in first via "
            "app.data.economic_calendar.merge_news_features()."
        )
    return frame[column]


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


def williams_r(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R -- the same trailing high/low range Stochastic uses,
    scaled -100 (at the range low) to 0 (at the range high) instead of
    Stochastic's 0-100 -- a pure rescale of the identical raw position,
    kept as its own indicator since -20/-80 are this scale's conventional
    overbought/oversold thresholds, not Stochastic's 80/20."""
    p = _period(period)
    hh = highest_high(frame["high"], p)
    ll = lowest_low(frame["low"], p)
    return (-100 * (hh - frame["close"]) / (hh - ll).replace(0, np.nan)).fillna(-50.0)


def roc(frame: pd.DataFrame, period: int = 10, column: str = "close") -> pd.Series:
    """Rate of Change -- percentage change vs. the close `period` bars ago.
    A pure momentum measure, distinct from RSI/Stochastic (which measure
    position within a recent range, not raw percentage change) and from
    MACD (a difference of two EMAs, not a simple lookback delta)."""
    p = _period(period)
    source = frame[column] if column in frame.columns else frame["close"]
    shifted = source.shift(p)
    return (100 * (source - shifted) / shifted.replace(0, np.nan)).fillna(0.0)


def awesome_oscillator(frame: pd.DataFrame) -> pd.Series:
    """Awesome Oscillator -- SMA(5) of the midpoint price minus SMA(34) of
    it, a momentum indicator with FIXED periods (unlike every other
    indicator in this module) since 5/34 is how it's conventionally
    defined; zero-line crosses are its standard trigger, distinct from
    MACD's EMA-based (not SMA-based) fast/slow difference on CLOSE (not
    midpoint) prices."""
    midpoint = (frame["high"] + frame["low"]) / 2.0
    return sma(midpoint, 5) - sma(midpoint, 34)


def chaikin_money_flow(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow -- volume-weighted accumulation/distribution
    over a rolling window, bounded roughly -1 to +1. Distinct from OBV
    (a running CUMULATIVE total with no window/bound) and from
    relative_volume (which only measures volume SIZE, not whether that
    volume traded closer to the bar's high or low)."""
    p = _period(period)
    high, low, close = frame["high"], frame["low"], frame["close"]
    volume = frame["volume"] if "volume" in frame.columns else pd.Series(0.0, index=frame.index)
    money_flow_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    money_flow_vol = money_flow_mult.fillna(0.0) * volume
    return (money_flow_vol.rolling(p, min_periods=p).sum() / volume.rolling(p, min_periods=p).sum().replace(0, np.nan)).fillna(0.0)


def parabolic_sar(frame: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> tuple[pd.Series, pd.Series]:
    """Parabolic SAR -- a flip-based, ACCELERATING trailing stop (the step
    multiplier grows every bar the trend continues, unlike SuperTrend's
    fixed ATR multiple), the original Wilder trend-following/stop-and-
    reverse system. Returns (sar, direction) where direction is +1.0
    while in an uptrend (SAR trails below price) and -1.0 while in a
    downtrend (SAR trails above price). Implemented as a sequential
    ratchet for the same reason supertrend() is -- each bar depends on
    the prior bar's SAR, extreme point, and acceleration factor, not a
    stateless rolling window."""
    high, low, close = frame["high"].to_numpy(), frame["low"].to_numpy(), frame["close"].to_numpy()
    n = len(frame)
    sar = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    if n == 0:
        return pd.Series(sar, index=frame.index), pd.Series(direction, index=frame.index)

    direction[0] = 1.0
    sar[0] = low[0]
    ep = high[0]   # extreme point -- highest high (uptrend) or lowest low (downtrend) since the last flip
    af = af_start
    for i in range(1, n):
        prev_sar = sar[i - 1]
        if direction[i - 1] == 1.0:
            candidate = prev_sar + af * (ep - prev_sar)
            candidate = min(candidate, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < candidate:
                direction[i] = -1.0
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                direction[i] = 1.0
                sar[i] = candidate
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            candidate = prev_sar + af * (ep - prev_sar)
            candidate = max(candidate, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > candidate:
                direction[i] = 1.0
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                direction[i] = -1.0
                sar[i] = candidate
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
    return pd.Series(sar, index=frame.index), pd.Series(direction, index=frame.index)


# ---------------------------------------------------------------------------
# Expansion round 7: Ichimoku, Fibonacci, Pivot Points, Heikin-Ashi, MFI,
# TRIX, Ultimate Oscillator, Aroon, Choppiness Index, DPO, Anchored VWAP,
# Linear Regression Channel, rolling correlation, Chandelier Exit.
# Same convention as prior rounds -- pure functions taking a DataFrame/Series
# and returning a Series (or tuple of Series), dispatched by name from
# _build_indicator_series_uncached below.
# ---------------------------------------------------------------------------

def ichimoku(frame: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns (tenkan, kijun, senkou_a, senkou_b, chikou).
    senkou_a/b are plotted `kijun_period` bars AHEAD on a real Ichimoku chart;
    here they are returned already shifted forward so `series[i]` is the
    cloud value overhanging bar i -- i.e. directly comparable to close[i]
    without the caller needing to know the forward-shift convention.
    chikou is close shifted `kijun_period` bars BACK (lagging span)."""
    high, low, close = frame["high"], frame["low"], frame["close"]
    tenkan = (high.rolling(tenkan_period, min_periods=tenkan_period).max() + low.rolling(tenkan_period, min_periods=tenkan_period).min()) / 2.0
    kijun = (high.rolling(kijun_period, min_periods=kijun_period).max() + low.rolling(kijun_period, min_periods=kijun_period).min()) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(kijun_period)
    senkou_b = ((high.rolling(senkou_b_period, min_periods=senkou_b_period).max() + low.rolling(senkou_b_period, min_periods=senkou_b_period).min()) / 2.0).shift(kijun_period)
    chikou = close.shift(-kijun_period)
    return tenkan, kijun, senkou_a, senkou_b, chikou


def fibonacci_levels(frame: pd.DataFrame, period: int = 50) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rolling-window Fibonacci retracement levels (38.2/50/61.8%) between
    the period's high and low, recomputed every bar off the CLOSED prior
    window (shifted 1) so no bar's own high/low leaks into its own levels."""
    high = frame["high"].shift(1).rolling(period, min_periods=period).max()
    low = frame["low"].shift(1).rolling(period, min_periods=period).min()
    span = high - low
    fib_382 = high - span * 0.382
    fib_500 = high - span * 0.5
    fib_618 = high - span * 0.618
    return fib_382, fib_500, fib_618


def pivot_points(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Classic daily floor-trader pivots computed from the PRIOR calendar
    day's high/low/close, forward-filled across the current day -- so a
    bar's pivot levels are always knowable before that bar prints."""
    if "timestamp" not in frame.columns:
        raise KeyError("pivot_points requires a 'timestamp' column")
    ts = pd.to_datetime(frame["timestamp"])
    day = ts.dt.normalize()
    daily = frame.groupby(day)
    prev_high = daily["high"].max().shift(1)
    prev_low = daily["low"].min().shift(1)
    prev_close = daily["close"].last().shift(1)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    return (day.map(pivot), day.map(r1), day.map(s1), day.map(r2), day.map(s2))


def heikin_ashi(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Smoothed Heikin-Ashi OHLC transform. ha_open is seeded from the
    first bar's real open/close average and then recurses, so it is
    computed with a simple forward loop rather than a vectorized rolling
    op (each bar's ha_open depends on the PRIOR bar's ha_open/ha_close)."""
    o, h, l, c = frame["open"].to_numpy(), frame["high"].to_numpy(), frame["low"].to_numpy(), frame["close"].to_numpy()
    n = len(c)
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty(n)
    if n:
        ha_open[0] = (o[0] + c[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum.reduce([h, ha_open, ha_close]) if n else h
    ha_low = np.minimum.reduce([l, ha_open, ha_close]) if n else l
    idx = frame.index
    return (pd.Series(ha_open, index=idx), pd.Series(ha_high, index=idx),
            pd.Series(ha_low, index=idx), pd.Series(ha_close, index=idx))


def money_flow_index(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI-style volume-weighted oscillator (MFI) -- distinct from Chaikin
    Money Flow, which weights by where the close sits within the bar's
    range rather than by raw typical-price direction."""
    volume = frame["volume"] if "volume" in frame.columns else pd.Series(1.0, index=frame.index)
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    raw_flow = typical * volume
    direction = typical.diff()
    pos_flow = raw_flow.where(direction > 0, 0.0)
    neg_flow = raw_flow.where(direction < 0, 0.0)
    p = _period(period)
    pos_sum = pos_flow.rolling(p, min_periods=p).sum()
    neg_sum = neg_flow.rolling(p, min_periods=p).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    result = 100 - (100 / (1 + ratio))
    return result.where(neg_sum.ne(0), 100).fillna(50)


def trix(series: pd.Series, period: int = 15) -> pd.Series:
    """Rate of change of a triple-smoothed EMA -- filters out minor cycles
    that a single or double EMA still passes through."""
    p = _period(period)
    e1 = series.ewm(span=p, adjust=False, min_periods=p).mean()
    e2 = e1.ewm(span=p, adjust=False, min_periods=p).mean()
    e3 = e2.ewm(span=p, adjust=False, min_periods=p).mean()
    return e3.pct_change() * 100


def ultimate_oscillator(frame: pd.DataFrame, period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
    """Weighted blend of three lookback periods' buying pressure, damping
    the single-period whipsaws that plain RSI/Stochastic are prone to."""
    close, high, low = frame["close"], frame["high"], frame["low"]
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)
    avg1 = bp.rolling(period1, min_periods=period1).sum() / tr.rolling(period1, min_periods=period1).sum().replace(0, np.nan)
    avg2 = bp.rolling(period2, min_periods=period2).sum() / tr.rolling(period2, min_periods=period2).sum().replace(0, np.nan)
    avg3 = bp.rolling(period3, min_periods=period3).sum() / tr.rolling(period3, min_periods=period3).sum().replace(0, np.nan)
    return (100 * (4 * avg1 + 2 * avg2 + avg3) / 7).fillna(50)


def aroon(frame: pd.DataFrame, period: int = 25) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (aroon_up, aroon_down, aroon_oscillator). Measures bars
    since the most recent period-high/low, not the magnitude of any move --
    a genuinely different mechanism from every momentum/range oscillator
    already in this module."""
    p = _period(period)
    high, low = frame["high"], frame["low"]

    def _bars_since_max(x: np.ndarray) -> float:
        return float(p - np.argmax(x[::-1]))

    def _bars_since_min(x: np.ndarray) -> float:
        return float(p - np.argmin(x[::-1]))

    bars_since_high = high.rolling(p + 1, min_periods=p + 1).apply(_bars_since_max, raw=True)
    bars_since_low = low.rolling(p + 1, min_periods=p + 1).apply(_bars_since_min, raw=True)
    up = 100 * (p - bars_since_high) / p
    down = 100 * (p - bars_since_low) / p
    return up, down, up - down


def choppiness_index(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """0-100 regime gauge: near 100 means a choppy/ranging market (true
    range is large relative to the net high-low span), near 0 means a
    strongly trending one. Distinct from ADX -- ADX measures directional
    strength, this measures range-vs-noise regardless of direction."""
    p = _period(period)
    tr = true_range(frame)
    tr_sum = tr.rolling(p, min_periods=p).sum()
    hh = frame["high"].rolling(p, min_periods=p).max()
    ll = frame["low"].rolling(p, min_periods=p).min()
    span = (hh - ll).replace(0, np.nan)
    result = 100 * np.log10(tr_sum / span) / np.log10(p)
    return result.fillna(50)


def dpo(series: pd.Series, period: int = 20) -> pd.Series:
    """Detrended Price Oscillator: price minus an SMA shifted back to
    remove the long-term trend component, isolating shorter cycles."""
    p = _period(period)
    shift = p // 2 + 1
    return series - sma(series, p).shift(shift)


def anchored_vwap(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    """VWAP re-anchored every `period` bars (a rolling anchor) rather than
    the existing session-anchored `vwap()` -- lets a strategy react to a
    volume-weighted average from an arbitrary recent point rather than
    always the start of the session."""
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = frame["volume"] if "volume" in frame.columns else pd.Series(1.0, index=frame.index)
    p = _period(period)
    pv = (typical * volume).rolling(p, min_periods=p).sum()
    vsum = volume.rolling(p, min_periods=p).sum().replace(0, np.nan)
    return (pv / vsum).fillna(typical)


def linreg_channel(series: pd.Series, period: int = 50, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rolling linear-regression midline +/- std_mult * residual stdev --
    distinct from Bollinger (which bands a simple moving average, not a
    fitted trendline)."""
    p = _period(period)
    x = np.arange(p, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _endpoint(y: np.ndarray) -> float:
        slope = ((x - x_mean) * (y - y.mean())).sum() / denom
        intercept = y.mean() - slope * x_mean
        return float(slope * x[-1] + intercept)

    def _resid_std(y: np.ndarray) -> float:
        slope = ((x - x_mean) * (y - y.mean())).sum() / denom
        intercept = y.mean() - slope * x_mean
        fitted = slope * x + intercept
        return float(np.std(y - fitted))

    mid = series.rolling(p, min_periods=p).apply(_endpoint, raw=True)
    resid = series.rolling(p, min_periods=p).apply(_resid_std, raw=True)
    return mid, mid + resid * std_mult, mid - resid * std_mult


def rolling_correlation(frame: pd.DataFrame, period: int = 50, pair_column: str = "pair_close") -> pd.Series:
    """Rolling Pearson correlation of close-to-close returns against a
    second merged instrument column -- a general regime filter usable by
    any family, distinct from `pair_zscore` (which measures spread
    dislocation, not co-movement strength)."""
    if pair_column not in frame.columns:
        raise KeyError(f"rolling_correlation requires a '{pair_column}' column")
    a = frame["close"].pct_change()
    b = frame[pair_column].pct_change()
    return a.rolling(_period(period), min_periods=_period(period)).corr(b).fillna(0.0)


def chandelier_exit(frame: pd.DataFrame, period: int = 22, atr_mult: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Returns (chandelier_long_stop, chandelier_short_stop): highest-high
    minus an ATR multiple (for longs) / lowest-low plus an ATR multiple
    (for shorts) -- a different sensitivity profile from the existing
    fixed-percent ATR trailing stop used elsewhere in the risk engine."""
    p = _period(period)
    atr_series = atr(frame, p)
    long_stop = frame["high"].rolling(p, min_periods=p).max() - atr_series * atr_mult
    short_stop = frame["low"].rolling(p, min_periods=p).min() + atr_series * atr_mult
    return long_stop, short_stop


def volume_profile(frame: pd.DataFrame, period: int = 100, n_bins: int = 24, value_area_pct: float = 0.70) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rolling-window Volume Profile: for each bar, bins the trailing
    `period` bars' volume by price into `n_bins` buckets and returns
    (poc, vah, val) -- the point of control (highest-volume bin's price)
    and the value-area high/low (the tightest band of bins around POC
    whose combined volume reaches `value_area_pct` of the window total).
    Genuinely absent from every other indicator in this module -- every
    existing level (Donchian/session/pivot/etc.) is a price extreme, never
    a volume-weighted price DISTRIBUTION. Uses each bar's OWN high/low/
    close/volume approximated as a single point at the bar's typical
    price rather than splitting volume across the bar's full range --
    the standard simplification for OHLCV-bar (not tick) volume profiles."""
    p = _period(period)
    high, low, close = frame["high"].to_numpy(), frame["low"].to_numpy(), frame["close"].to_numpy()
    volume = (frame["volume"] if "volume" in frame.columns else pd.Series(1.0, index=frame.index)).to_numpy()
    typical = (high + low + close) / 3.0
    n = len(close)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    for i in range(p - 1, n):
        lo, hi = i - p + 1, i + 1
        window_high = high[lo:hi].max()
        window_low = low[lo:hi].min()
        span = window_high - window_low
        if span <= 0:
            poc[i] = vah[i] = val[i] = typical[i]
            continue
        edges = np.linspace(window_low, window_high, n_bins + 1)
        bin_idx = np.clip(np.digitize(typical[lo:hi], edges) - 1, 0, n_bins - 1)
        bin_volume = np.zeros(n_bins)
        np.add.at(bin_volume, bin_idx, volume[lo:hi])
        total = bin_volume.sum()
        if total <= 0:
            poc[i] = vah[i] = val[i] = typical[i]
            continue
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        poc_bin = int(np.argmax(bin_volume))
        poc[i] = bin_centers[poc_bin]
        # Expand outward from the POC bin, each step adding whichever
        # neighbor (below or above) has more volume, until the included
        # bins' volume reaches the target value-area percentage.
        lo_bin = hi_bin = poc_bin
        included = bin_volume[poc_bin]
        target = total * value_area_pct
        while included < target and (lo_bin > 0 or hi_bin < n_bins - 1):
            below = bin_volume[lo_bin - 1] if lo_bin > 0 else -1.0
            above = bin_volume[hi_bin + 1] if hi_bin < n_bins - 1 else -1.0
            if above >= below:
                hi_bin += 1
                included += bin_volume[hi_bin]
            else:
                lo_bin -= 1
                included += bin_volume[lo_bin]
        val[i] = bin_centers[lo_bin]
        vah[i] = bin_centers[hi_bin]
    idx = frame.index
    return pd.Series(poc, index=idx), pd.Series(vah, index=idx), pd.Series(val, index=idx)


# ---------------------------------------------------------------------------
# Expansion round 8 (Sep 2026) -- lower-lag trend MAs, trend-strength
# confirmation, a first-class squeeze indicator, volume/price-distribution
# complements, and two short-term mean-reversion oscillators. See
# app.search.strategy_space for the new families built on top of these.
# ---------------------------------------------------------------------------

def hma(series: pd.Series, period: int = 20) -> pd.Series:
    """Hull Moving Average -- WMA(2*WMA(n/2) - WMA(n), sqrt(n)). Removes far
    more lag than a plain WMA/EMA of the same period at the cost of a small
    overshoot on sharp reversals; a common fast-entry MA in prop-eval-style
    scalping where a standard EMA reacts too slowly to matter within a
    challenge's time limit."""
    p = _period(period)
    half = max(p // 2, 1)
    sqrt_p = max(int(round(np.sqrt(p))), 1)
    raw = 2 * wma(series, half) - wma(series, p)
    return wma(raw, sqrt_p)


def dema(series: pd.Series, period: int = 20) -> pd.Series:
    """Double EMA -- 2*EMA(n) - EMA(EMA(n)). Cuts EMA lag roughly in half by
    subtracting out the lag of a second EMA pass."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2


def tema(series: pd.Series, period: int = 20) -> pd.Series:
    """Triple EMA -- 3*EMA(n) - 3*EMA(EMA(n)) + EMA(EMA(EMA(n))). Lower lag
    still than DEMA, at the cost of more overshoot/whipsaw in a genuine
    chop regime -- the standard tradeoff every lower-lag MA in this section
    makes."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def kama(series: pd.Series, period: int = 10, fast_period: int = 2, slow_period: int = 30) -> pd.Series:
    """Kaufman's Adaptive Moving Average -- an EMA whose smoothing constant
    is itself driven by an efficiency ratio (net directional move over the
    window / sum of bar-to-bar absolute moves), so it hugs price tightly in
    a clean trend and flattens out (barely moving) in chop, without needing
    a separate regime filter the way a fixed-period MA would."""
    p = _period(period)
    change = (series - series.shift(p)).abs()
    volatility = series.diff().abs().rolling(p, min_periods=p).sum()
    efficiency_ratio = (change / volatility.replace(0, np.nan)).fillna(0.0)
    fast_sc = 2.0 / (max(fast_period, 1) + 1)
    slow_sc = 2.0 / (max(slow_period, 1) + 1)
    smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2

    values = series.to_numpy(dtype=float)
    sc = smoothing_constant.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    start = p
    if start >= len(values):
        return pd.Series(out, index=series.index)
    out[start] = values[start]
    for i in range(start + 1, len(values)):
        prev = out[i - 1]
        if np.isnan(prev):
            out[i] = values[i]
            continue
        step = sc[i] if not np.isnan(sc[i]) else slow_sc
        out[i] = prev + step * (values[i] - prev)
    return pd.Series(out, index=series.index)


def vortex(frame: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series]:
    """Vortex Indicator (VI+/VI-) -- trend-strength/direction confirmation,
    cheap to add given True Range already exists (adx() above already
    builds its own True-Range-based smoothing). VM+ / VM- are the absolute
    distance between the current high/prior low and current low/prior high;
    each is Wilder-summed over `period` and normalized by summed True
    Range. A VI+/VI- crossover is the standard entry trigger; both lines
    trending apart signals a strengthening trend either direction."""
    p = _period(period)
    vm_plus = (frame["high"] - frame["low"].shift(1)).abs()
    vm_minus = (frame["low"] - frame["high"].shift(1)).abs()
    tr_sum = true_range(frame).rolling(p, min_periods=p).sum()
    vi_plus = vm_plus.rolling(p, min_periods=p).sum() / tr_sum.replace(0, np.nan)
    vi_minus = vm_minus.rolling(p, min_periods=p).sum() / tr_sum.replace(0, np.nan)
    return vi_plus.fillna(1.0), vi_minus.fillna(1.0)


def elder_ray(frame: pd.DataFrame, period: int = 13) -> tuple[pd.Series, pd.Series]:
    """Elder Ray (Bull Power / Bear Power) -- Dr. Alexander Elder's trend-
    strength confirmation pair: an EMA defines the consensus trend, and
    Bull Power (high - EMA) / Bear Power (low - EMA) measure how far bulls/
    bears pushed price away from it on the current bar. Cheap to add given
    ema() already exists; a common confirmation layer alongside ADX/Aroon
    for "is this trend actually still being pushed, or just drifting"."""
    baseline = ema(frame["close"], period)
    bull_power = frame["high"] - baseline
    bear_power = frame["low"] - baseline
    return bull_power, bear_power


def ttm_squeeze(frame: pd.DataFrame, bb_period: int = 20, bb_mult: float = 2.0,
                 kc_period: int = 20, kc_mult: float = 1.5, momentum_period: int = 12) -> tuple[pd.Series, pd.Series]:
    """TTM Squeeze as a first-class, standalone indicator -- returns
    (squeeze_on, momentum). Prior to this, a Bollinger-inside-Keltner
    squeeze only existed implicitly as the `keltner_squeeze_breakout`
    strategy family's own inline condition (bollinger_upper < keltner_upper
    AND bollinger_lower > keltner_lower); this gives that same test a
    reusable name plus John Carter's momentum "dot" -- a linear-regression
    value of (close - average of [midpoint of highest-high/lowest-low over
    kc_period, and an SMA of close over kc_period]) over momentum_period --
    so a squeeze release can be read as bullish/bearish, not just on/off.
    squeeze_on is a bool Series (True while compressed); momentum is a
    signed float Series (its zero-crossings/slope are the usual triggers
    once a squeeze fires)."""
    _, bb_upper, bb_lower = bollinger(frame["close"], bb_period, bb_mult)
    _, kc_upper, kc_lower = keltner(frame, kc_period, kc_mult)
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    p = _period(kc_period)
    hh = highest_high(frame["high"], p)
    ll = lowest_low(frame["low"], p)
    donchian_mid = (hh + ll) / 2.0
    sma_close = sma(frame["close"], p)
    reference = (donchian_mid + sma_close) / 2.0
    deviation = frame["close"] - reference

    mp = _period(momentum_period)
    x = np.arange(mp, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _linreg_last(values: np.ndarray) -> float:
        if denom == 0 or np.isnan(values).any():
            return np.nan
        slope = ((x - x_mean) * (values - values.mean())).sum() / denom
        intercept = values.mean() - slope * x_mean
        return slope * x[-1] + intercept

    momentum = deviation.rolling(mp, min_periods=mp).apply(_linreg_last, raw=True)
    return squeeze_on.fillna(False), momentum


def vwma(frame: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
    """Volume-Weighted Moving Average -- sum(price*volume)/sum(volume) over
    a rolling window. A common complement to plain SMA/EMA trend lines that
    weights the bars where real size actually traded more heavily; falls
    back to a plain SMA (volume=1 for every bar) when no volume column is
    present, same convention as every other volume-aware indicator here."""
    p = _period(period)
    volume = frame["volume"] if "volume" in frame.columns else pd.Series(1.0, index=frame.index)
    price = frame[column] if column in frame.columns else frame["close"]
    pv = (price * volume).rolling(p, min_periods=p).sum()
    vsum = volume.rolling(p, min_periods=p).sum()
    return pv / vsum.replace(0, np.nan)


def accumulation_distribution(frame: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line -- a running (non-windowed) cumulative
    volume-flow line, unlike CMF/OBV which already exist here: CMF is a
    BOUNDED rolling-window ratio and OBV only looks at close-to-close
    direction, ignoring where in the bar's own range the close landed. ADL
    uses the close's location within each bar's own high-low range (the
    Money Flow Multiplier) times that bar's volume, cumulatively summed --
    the standard building block chaikin_oscillator() below differences."""
    high, low, close = frame["high"], frame["low"], frame["close"]
    volume = frame["volume"] if "volume" in frame.columns else pd.Series(1.0, index=frame.index)
    range_ = (high - low).replace(0, np.nan)
    money_flow_multiplier = ((close - low) - (high - close)) / range_
    money_flow_multiplier = money_flow_multiplier.fillna(0.0)
    money_flow_volume = money_flow_multiplier * volume
    return money_flow_volume.cumsum()


def chaikin_oscillator(frame: pd.DataFrame, fast_period: int = 3, slow_period: int = 10) -> pd.Series:
    """Chaikin Oscillator -- EMA(fast) - EMA(slow) of the Accumulation/
    Distribution Line above, the standard MACD-style momentum-of-volume-flow
    read on top of it (a rising ADL that's also accelerating vs. one that's
    merely drifting upward)."""
    adl = accumulation_distribution(frame)
    return ema(adl, fast_period) - ema(adl, slow_period)


def fisher_transform(frame: pd.DataFrame, period: int = 10) -> tuple[pd.Series, pd.Series]:
    """Ehlers' Fisher Transform -- returns (fisher, fisher_signal). Rescales
    the rolling position of price within its own high/low range to [-1, 1]
    (clamped just short of the +-1 asymptotes) and applies the inverse
    hyperbolic tangent, which sharpens turning points into distinct,
    Gaussian-ish peaks -- a popular short-term mean-reversion trigger that
    reads more cleanly than a bounded oscillator like RSI/Stochastic right
    at extremes, where those saturate and flatten instead of peaking."""
    p = _period(period)
    high, low = frame["high"], frame["low"]
    typical = (high + low) / 2.0
    hh = typical.rolling(p, min_periods=p).max()
    ll = typical.rolling(p, min_periods=p).min()
    raw_range = (hh - ll).replace(0, np.nan)
    normalized = 2.0 * ((typical - ll) / raw_range - 0.5)
    normalized = normalized.fillna(0.0).clip(-0.999, 0.999)

    values = normalized.to_numpy(dtype=float)
    fisher = np.zeros(len(values))
    smoothed = 0.0
    for i in range(1, len(values)):
        smoothed = 0.33 * values[i] + 0.67 * smoothed
        smoothed = min(max(smoothed, -0.999), 0.999)
        fisher[i] = 0.5 * np.log((1 + smoothed) / (1 - smoothed)) + 0.5 * fisher[i - 1]
    fisher_series = pd.Series(fisher, index=frame.index)
    return fisher_series, fisher_series.shift(1)


def connors_rsi(frame: pd.DataFrame, rsi_period: int = 3, streak_period: int = 2, rank_period: int = 100) -> pd.Series:
    """Connors RSI -- the average of three components, each bounded 0-100:
    (1) a short RSI(3) of price itself, (2) an RSI of the current up/down
    STREAK length (how many consecutive bars price has closed the same
    direction, signed), and (3) the percentile rank of today's 1-bar return
    among the trailing `rank_period` bars' 1-bar returns. Popular short-
    term mean-reversion signal that reads faster and cleaner than a plain
    RSI at the same short period, since a lone RSI(3) alone is dominated by
    noise."""
    close = frame["close"]
    price_rsi = rsi(close, rsi_period)

    change = close.diff()
    direction = np.sign(change).fillna(0.0)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if direction.iloc[i] == 0:
            streak[i] = 0.0
        elif direction.iloc[i] == direction.iloc[i - 1]:
            streak[i] = streak[i - 1] + direction.iloc[i]
        else:
            streak[i] = direction.iloc[i]
    streak_series = pd.Series(streak, index=close.index)
    streak_rsi = rsi(streak_series, streak_period)

    one_bar_return = close.pct_change(1)
    p = _period(rank_period)

    def _percent_rank(values: np.ndarray) -> float:
        if np.isnan(values[-1]):
            return np.nan
        return 100.0 * (values[:-1] < values[-1]).sum() / (len(values) - 1) if len(values) > 1 else 50.0

    percent_rank = one_bar_return.rolling(p + 1, min_periods=p + 1).apply(_percent_rank, raw=True)

    return ((price_rsi + streak_rsi + percent_rank.fillna(50.0)) / 3.0).fillna(50.0)


def average_daily_range(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Daily Range (ADR) -- an explicit named indicator, distinct
    from candle_range() (per-BAR range) and atr() (Wilder-smoothed, still
    per-bar). Groups bars by calendar day, takes each day's own (high-low)
    range, and returns the rolling `period`-day average of that -- the
    "how much room is typically left in a day" figure used constantly for
    prop-firm intraday position sizing and same-day "how much of today's
    ADR has already printed" checks. Forward-filled across every intraday
    bar of a day so it lines up with an intraday index like every other
    indicator here, rather than being its own daily-only series."""
    ts = pd.to_datetime(frame["timestamp"])
    day = ts.dt.normalize()
    daily_high = frame["high"].groupby(day).transform("max")
    daily_low = frame["low"].groupby(day).transform("min")
    daily_range = (daily_high - daily_low)
    # One range value per calendar day, then a rolling mean of the last
    # `period` DAYS (not bars) of those values, forward-filled back onto
    # every intraday bar.
    per_day = daily_range.groupby(day).first()
    p = _period(period)
    adr_by_day = per_day.rolling(p, min_periods=p).mean()
    return day.map(adr_by_day)


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
    if kind == "williams_r":
        return williams_r(frame, p)
    if kind == "roc":
        return roc(frame, p, column)
    if kind == "awesome_oscillator":
        return awesome_oscillator(frame)
    if kind == "cmf":
        return chaikin_money_flow(frame, p)
    if kind == "psar_line":
        return parabolic_sar(frame)[0]
    if kind == "psar_direction":
        return parabolic_sar(frame)[1]
    # Expansion round 7.
    if kind == "ichimoku_tenkan":
        return ichimoku(frame, tenkan_period=p)[0]
    if kind == "ichimoku_kijun":
        return ichimoku(frame, kijun_period=p)[1]
    if kind == "ichimoku_senkou_a":
        return ichimoku(frame)[2]
    if kind == "ichimoku_senkou_b":
        return ichimoku(frame)[3]
    if kind == "ichimoku_chikou":
        return ichimoku(frame)[4]
    if kind == "fib_382":
        return fibonacci_levels(frame, p)[0]
    if kind == "fib_500":
        return fibonacci_levels(frame, p)[1]
    if kind == "fib_618":
        return fibonacci_levels(frame, p)[2]
    if kind == "pivot_point":
        return pivot_points(frame)[0]
    if kind == "pivot_r1":
        return pivot_points(frame)[1]
    if kind == "pivot_s1":
        return pivot_points(frame)[2]
    if kind == "pivot_r2":
        return pivot_points(frame)[3]
    if kind == "pivot_s2":
        return pivot_points(frame)[4]
    if kind == "heikin_ashi_open":
        return heikin_ashi(frame)[0]
    if kind == "heikin_ashi_high":
        return heikin_ashi(frame)[1]
    if kind == "heikin_ashi_low":
        return heikin_ashi(frame)[2]
    if kind == "heikin_ashi_close":
        return heikin_ashi(frame)[3]
    if kind == "mfi":
        return money_flow_index(frame, p)
    if kind == "trix":
        return trix(source, p)
    if kind == "ultimate_oscillator":
        return ultimate_oscillator(frame)
    if kind == "aroon_up":
        return aroon(frame, p)[0]
    if kind == "aroon_down":
        return aroon(frame, p)[1]
    if kind == "aroon_oscillator":
        return aroon(frame, p)[2]
    if kind == "choppiness_index":
        return choppiness_index(frame, p)
    if kind == "dpo":
        return dpo(source, p)
    if kind == "anchored_vwap":
        return anchored_vwap(frame, p)
    if kind == "linreg_mid":
        return linreg_channel(source, p)[0]
    if kind == "linreg_upper":
        return linreg_channel(source, p)[1]
    if kind == "linreg_lower":
        return linreg_channel(source, p)[2]
    if kind == "correlation":
        return rolling_correlation(frame, p)
    if kind == "chandelier_long":
        return chandelier_exit(frame, p)[0]
    if kind == "chandelier_short":
        return chandelier_exit(frame, p)[1]
    if kind == "volume_profile_poc":
        return volume_profile(frame, p)[0]
    if kind == "volume_profile_vah":
        return volume_profile(frame, p)[1]
    if kind == "volume_profile_val":
        return volume_profile(frame, p)[2]
    # Expansion round 8.
    if kind == "hma":
        return hma(source, p)
    if kind == "dema":
        return dema(source, p)
    if kind == "tema":
        return tema(source, p)
    if kind == "kama":
        return kama(source, p)
    if kind == "vortex_plus":
        return vortex(frame, p)[0]
    if kind == "vortex_minus":
        return vortex(frame, p)[1]
    if kind == "elder_bull_power":
        return elder_ray(frame, p)[0]
    if kind == "elder_bear_power":
        return elder_ray(frame, p)[1]
    if kind == "ttm_squeeze_on":
        return ttm_squeeze(frame, kc_period=p)[0].astype(int)
    if kind == "ttm_squeeze_momentum":
        return ttm_squeeze(frame, kc_period=p)[1]
    if kind == "vwma":
        return vwma(frame, p, column)
    if kind == "adl":
        return accumulation_distribution(frame)
    if kind == "chaikin_oscillator":
        return chaikin_oscillator(frame)
    if kind == "fisher_transform":
        return fisher_transform(frame, p)[0]
    if kind == "fisher_transform_signal":
        return fisher_transform(frame, p)[1]
    if kind == "connors_rsi":
        return connors_rsi(frame)
    if kind == "adr":
        return average_daily_range(frame, p)
    if kind == "news_minutes_since_high_impact":
        return news_feature_column(frame, "minutes_since_high_impact_news")
    if kind == "news_minutes_until_high_impact":
        return news_feature_column(frame, "minutes_until_high_impact_news")
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
