"""Indicator pool for the Discovery Lab: every causal indicator the app can compute, as one searchable family (``ind_rule``).

``POOL_KINDS`` is the list of indicator kinds (base set + v7..v13) that (a) need no outside data feed or companion market, (b) produce real,
non-constant values on ordinary OHLCV and (c) pass the prefix-invariance test (tests/backtest/test_prefix_invariance_all_kinds.py +
test_indicators_v13.py): a bar's value never changes when later bars are removed. Calendar/event/outside-feed kinds are left out because
they are NaN without their data and would only waste search budget.

Turning an indicator into a signal without peeking: the raw value is standardised against ITS OWN trailing window (``win`` bars, causal),
so "high" and "low" mean "high/low relative to the recent past", never relative to the whole sample.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

POOL_KINDS: tuple = (
    'adl', 'adr', 'adx', 'alligator_spread', 'alma_dist', 'amihud', 'amihud_z', 'analog_match', 'anchored_vwap', 'aroon_down',
    'aroon_oscillator', 'aroon_up', 'atr', 'atr_percentile', 'autocorr_lag1', 'average_volume', 'avwap_dist_month',
    'avwap_dist_swing', 'avwap_dist_week', 'awesome_oscillator', 'balance_of_power', 'base_signal_expectancy',
    'bipower_jump_ratio', 'bocpd_prob', 'body_ratio', 'bollinger_bandwidth', 'bollinger_lower', 'bollinger_mid',
    'bollinger_upper', 'box_duration', 'box_position', 'box_tightness', 'box_touches', 'breakout_retest', 'bull_flag_score',
    'bvc_imbalance', 'candle_range', 'cci', 'chaikin_oscillator', 'chandelier_long', 'chandelier_short', 'choppiness_index',
    'close_location_value', 'cluster_regime', 'cmf', 'cmo', 'connors_rsi', 'coppock', 'corwin_schultz', 'cusum_stat',
    'cyber_cycle', 'cycle_breakout_state', 'daily_ema_distance', 'daily_nr4', 'daily_nr7', 'day_open_dist', 'day_range_pos',
    'day_range_used', 'dd_depth_pct', 'dd_recovery', 'dd_time_under', 'dema', 'dfa_exponent', 'donchian_lower', 'donchian_mid',
    'donchian_mid_distance', 'donchian_upper', 'downside_dev', 'dpo', 'ease_of_movement', 'efficiency_ratio', 'ehlers_itrend',
    'ehlers_sine', 'ehlers_sine_lead', 'ehlers_supersmoother', 'elder_bear_power', 'elder_bull_power', 'elder_impulse', 'ema',
    'ema_stack_score', 'equal_highs_density', 'equal_lows_density', 'event_day_any', 'event_day_fomc', 'event_day_nfp',
    'ewma_vol', 'fib_382', 'fib_500', 'fib_618', 'first_trading_day', 'fisher_transform', 'fisher_transform_signal',
    'flow_absorption', 'flow_cvd_slope', 'flow_delta_z', 'force_index_norm', 'fractal_strength', 'frama_dist', 'gamma_convexity',
    'garch_vol', 'garman_klass_vol', 'gaussian_slope', 'gex_call_wall_distance', 'gex_flip_distance', 'gex_put_wall_distance',
    'gex_regime', 'heikin_ashi_close', 'heikin_ashi_high', 'heikin_ashi_low', 'heikin_ashi_open', 'highest_high',
    'hilbert_period', 'hma', 'hmm_bull_prob', 'htf_consensus', 'hurst_exponent', 'ichimoku_chikou', 'ichimoku_kijun',
    'ichimoku_senkou_a', 'ichimoku_senkou_b', 'ichimoku_tenkan', 'inside_break', 'inv_fisher_rsi', 'kalman_level', 'kalman_slope',
    'kalman_slope_z', 'kama', 'katz_fractal_dim', 'keltner_lower', 'keltner_mid', 'keltner_upper', 'klinger_osc', 'kst',
    'kyle_lambda', 'linreg_lower', 'linreg_mid', 'linreg_r2', 'linreg_slope', 'linreg_upper', 'ljung_box_q', 'lower_wick_ratio',
    'lowest_low', 'lowfreq_power_share', 'macd', 'macd_histogram', 'macd_signal', 'mama_dist', 'mama_fama_spread',
    'mann_kendall_z', 'mass_index', 'mcginley_dist', 'meta_label_prob', 'mfi', 'minus_di', 'minutes_since_fomc',
    'minutes_since_nfp', 'minutes_since_open', 'minutes_to_close', 'minutes_to_fomc', 'minutes_to_nfp', 'mo_open_dist',
    'mo_range_break', 'month_end_window', 'morning_move', 'ms_var_prob', 'naked_poc_count', 'naked_poc_dist', 'nested_squeeze',
    'nr4', 'nr7', 'nr_breakout', 'obv', 'obv_ema', 'opex_week', 'or_failure', 'or_state', 'or_width_atr', 'ou_halflife',
    'parkinson_vol', 'pdc_dist', 'percentage_change', 'perm_entropy', 'pivot_point', 'pivot_r1', 'pivot_r2', 'pivot_s1',
    'pivot_s2', 'plus_di', 'pmo_close_dist', 'pmo_high_dist', 'pmo_low_dist', 'po3_state', 'ppo', 'pre_holiday',
    'premium_discount', 'premium_discount_multi', 'prior_day_range_atr', 'psar_direction', 'psar_line', 'pvo', 'pwk_close_dist',
    'pwk_high_dist', 'pwk_low_dist', 'range_tod_ratio', 'range_tod_zscore', 'relative_volume', 'ridge_forecast', 'roc',
    'rogers_satchell_vol', 'roll_spread', 'round_dist_auto', 'round_dist_es', 'round_dist_gc', 'round_dist_nq', 'round_tag',
    'rsi', 'runs_z', 'rvgi', 'rvol_tod', 'sample_entropy', 'schaff_trend_cycle', 'seas_tod', 'seasonal_cv_forecast',
    'semivar_ratio', 'semivol_down', 'semivol_up', 'slope_tstat', 'sma', 'smi', 'spread_z', 'stdev', 'stoch_d', 'stoch_k',
    'stochrsi_d', 'stochrsi_k', 'supertrend_direction', 'supertrend_line', 'swing_bos', 'swing_choch', 't3_dist', 'td_countdown',
    'td_setup', 'tema', 'theil_sen_dist', 'theta_clock', 'tpo_poc_dist', 'tpo_poor_high', 'tpo_poor_low',
    'tpo_single_print_share', 'tpo_va_position', 'triple_barrier_prob', 'trix', 'tsmom_blend', 'ttm_squeeze_momentum',
    'ttm_squeeze_on', 'turn_of_month', 'ulcer_index', 'ultimate_oscillator', 'updown_vol_ratio', 'upper_wick_ratio',
    'variance_ratio', 'vhf', 'vol_cone_pct', 'vol_of_vol', 'vol_term_ratio', 'volume_clock_return', 'volume_delta',
    'volume_expansion', 'volume_profile_poc', 'volume_profile_vah', 'volume_profile_val', 'volume_zscore', 'vortex_minus',
    'vortex_plus', 'vote_score', 'vp_dist_poc', 'vp_in_lvn', 'vp_location', 'vpin', 'vr_profile_z', 'vr_z', 'vw_close_location',
    'vwap', 'vwma', 'wavelet_dominant_scale', 'wavelet_low_share', 'wavetrend', 'wedge_break', 'weekday_number', 'weis_wave',
    'williams_r', 'wk_open_dist', 'wma', 'yang_zhang_vol', 'zigzag_leg_atr', 'zigzag_leg_bars', 'zigzag_prev_leg_atr',
    'zigzag_retrace', 'zlema_dist'
)

_SLOW: set = set()          # kinds that took too long on this machine / data: dropped from the rest of the run (their signals are all-False)
_SLOW_SECONDS = 6.0


def series_for(df: pd.DataFrame, kind: str, period: int) -> np.ndarray:
    """The indicator as a float array aligned to ``df`` (NaN when it cannot be computed). Slow kinds are blacklisted after the first try."""
    n = len(df)
    if kind in _SLOW:
        return np.full(n, np.nan)
    from app.strategy.indicators import build_indicator_series
    t0 = time.time()
    try:
        out = build_indicator_series(df, kind, int(period)).to_numpy(float)
    except Exception:  # noqa: BLE001 -- a kind that cannot run on this frame simply never fires
        return np.full(n, np.nan)
    if time.time() - t0 > _SLOW_SECONDS:
        _SLOW.add(kind)
    return out


def zscore_trailing(x: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(x)
    m = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std(ddof=0)
    return ((s - m) / sd.where(sd > 0)).to_numpy()


def rule_signals(ind, p: dict):
    """(long, short) for one ``ind_rule`` setting. Modes:
    trend   - the standardised indicator crosses UP through +k (long) or DOWN through -k (short): ride the shift.
    revert  - it crosses back inside from beyond -k (long) / +k (short): fade the extreme.
    slope   - it has risen on every one of the last 2k bars while still below its trailing mean (long) / the mirror (short)."""
    x = ind.memo(("poolx", p["kind"], p["period"]), lambda: series_for(ind.df, p["kind"], p["period"]))
    z = ind.memo(("poolz", p["kind"], p["period"], p["win"]), lambda: zscore_trailing(x, int(p["win"])))
    k = float(p["k"])
    prev = np.concatenate([[np.nan], z[:-1]])
    ok = ~np.isnan(z) & ~np.isnan(prev)
    mode = p["mode"]
    if mode == "trend":
        lg, sh = (z > k) & (prev <= k), (z < -k) & (prev >= -k)
    elif mode == "revert":
        lg, sh = (z > -k) & (prev <= -k), (z < k) & (prev >= k)
    else:
        w = max(2, int(round(k * 2)))
        d = np.diff(z, prepend=np.nan)
        n_up = pd.Series((d > 0).astype(float)).rolling(w, min_periods=w).sum().to_numpy()
        n_dn = pd.Series((d < 0).astype(float)).rolling(w, min_periods=w).sum().to_numpy()
        lg, sh = (n_up >= w) & (z < 0), (n_dn >= w) & (z > 0)
    return lg & ok, sh & ok
