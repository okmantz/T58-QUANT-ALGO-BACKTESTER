"""Session-anchored VWAP +/- sigma value-area bands + VPVR POC.

Port of astra-quant-agent's ``compute_vwap_volume_profile``
(``scripts/factors/okx_quant_factors.py:532-606``), re-anchored from a 24h
rolling crypto window to T58's trading-day session: VWAP (and its bands and
POC) reset at the session roll -- 17:00 America/Chicago by default, the
Globex day roll -- instead of at a UTC calendar midnight.

Per-bar and fully causal: bar ``i``'s VWAP/sigma/bands use session data up
to and including bar ``i`` only (cumsum-based, never a full-frame groupby
aggregate), and bar ``i``'s POC is the volume histogram over the
session-to-date range only. The behavioral lookahead gate
(``app.strategy.lookahead_check``) must stay clean.

Why it matters (Oct-4 analysis, Part A port #3): T58 has swing BOS/CHOCH
but zero volume-profile structure. VWAP +/- 1 sigma is the value area
(~70% of session volume), +/- 2 sigma the extremes (~95%); POC is the
session's cost-basis magnet. Additive, not a duplicate of anything in
``app.strategy.indicators`` (whose ``vwap()`` is calendar-day anchored and
has no bands or POC).
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Session anchoring
# ---------------------------------------------------------------------------

DEFAULT_ROLL_HOUR = 17            # Globex day roll, 17:00 exchange time
DEFAULT_EXCHANGE_TZ = "America/Chicago"
DEFAULT_SOURCE_TZ = "UTC"         # naive timestamps are assumed to be UTC


def session_keys(
    timestamps: pd.Series | pd.DatetimeIndex,
    *,
    roll_hour: int = DEFAULT_ROLL_HOUR,
    exchange_tz: str = DEFAULT_EXCHANGE_TZ,
    source_tz: str = DEFAULT_SOURCE_TZ,
) -> pd.Series:
    """Map each bar timestamp to its trading-session label.

    A new session starts at ``roll_hour``:00 exchange-local time, so e.g.
    2024-01-02 16:59 CT and 2024-01-02 17:00 CT belong to different
    sessions. Naive timestamps are interpreted as ``source_tz`` (UTC --
    matches the naive-UTC market data the importer produces).
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if ts.tz is None:
        ts = ts.tz_localize(source_tz)
    local = ts.tz_convert(ZoneInfo(exchange_tz))
    shifted = local - pd.Timedelta(hours=int(roll_hour))
    labels = shifted.date.astype(str)
    return pd.Series(labels, index=pd.RangeIndex(len(labels)))


# ---------------------------------------------------------------------------
# Session VWAP + sigma bands (vectorized, causal via cumsum)
# ---------------------------------------------------------------------------

def _typical_price(df: pd.DataFrame) -> np.ndarray:
    return ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy(dtype=float)


def _session_volume(df: pd.DataFrame) -> np.ndarray:
    if "volume" in df.columns:
        v = df["volume"].to_numpy(dtype=float)
        v = np.where(np.isfinite(v) & (v > 0), v, 0.0)
        if v.sum() > 0:
            return v
    # No usable volume column: fall back to equal weights (same convention
    # as app.strategy.indicators.vwap).
    return np.ones(len(df), dtype=float)


def session_vwap_profile(
    df: pd.DataFrame,
    *,
    roll_hour: int = DEFAULT_ROLL_HOUR,
    exchange_tz: str = DEFAULT_EXCHANGE_TZ,
    source_tz: str = DEFAULT_SOURCE_TZ,
    value_area_sigma: float = 1.0,
    extreme_sigma: float = 2.0,
    poc_buckets: int = 30,
) -> pd.DataFrame:
    """Per-bar session-anchored VWAP profile.

    Columns: ``vwap``, ``sigma``, ``vah`` (VWAP + value_area_sigma * sigma),
    ``val`` (VWAP - value_area_sigma * sigma), ``upper_2`` / ``lower_2``
    (VWAP +/- extreme_sigma * sigma), ``poc`` (session-to-date VPVR point of
    control), ``zscore`` ((close - vwap) / sigma).

    VWAP and sigma come from volume-weighted cumsums reset at each session
    boundary -- causal by construction. POC is recomputed per bar over the
    session-to-date typical-price range split into ``poc_buckets`` equal
    buckets (the astra semantics), taking the highest-volume bucket's
    midpoint; also causal. Bars before any volume prints read NaN rather
    than 0.0 (a 0.0 VWAP would read as "price is at zero").
    """
    n = len(df)
    keys = session_keys(df["timestamp"], roll_hour=roll_hour, exchange_tz=exchange_tz, source_tz=source_tz)
    typ = _typical_price(df)
    vol = _session_volume(df)
    close = df["close"].to_numpy(dtype=float)

    key_arr = keys.to_numpy()
    # Session boundaries: positions where the key changes.
    boundaries = np.concatenate([[0], np.flatnonzero(key_arr[1:] != key_arr[:-1]) + 1, [n]])

    vwap = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    upper_2 = np.full(n, np.nan)
    lower_2 = np.full(n, np.nan)
    poc = np.full(n, np.nan)

    pv = typ * vol
    pv2 = typ * typ * vol

    for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
        sv = np.cumsum(vol[b0:b1])
        valid = sv > 0
        if not np.any(valid):
            continue
        svt = np.cumsum(pv[b0:b1])
        svt2 = np.cumsum(pv2[b0:b1])
        vw = svt / sv
        var = svt2 / sv - vw * vw
        var = np.where(var > 0, var, 0.0)
        sg = np.sqrt(var)
        idx = slice(b0, b1)
        vwap[idx] = np.where(valid, vw, np.nan)
        sigma[idx] = np.where(valid, sg, np.nan)
        vah[idx] = np.where(valid, vw + value_area_sigma * sg, np.nan)
        val[idx] = np.where(valid, vw - value_area_sigma * sg, np.nan)
        upper_2[idx] = np.where(valid, vw + extreme_sigma * sg, np.nan)
        lower_2[idx] = np.where(valid, vw - extreme_sigma * sg, np.nan)
        poc[b0:b1] = _session_poc_series(
            typ[b0:b1], df["high"].to_numpy(dtype=float)[b0:b1],
            df["low"].to_numpy(dtype=float)[b0:b1], vol[b0:b1],
            buckets=max(int(poc_buckets), 1),
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = (close - vwap) / sigma
    zscore[~np.isfinite(zscore)] = np.nan

    return pd.DataFrame(
        {
            "vwap": vwap,
            "sigma": sigma,
            "vah": vah,
            "val": val,
            "upper_2": upper_2,
            "lower_2": lower_2,
            "poc": poc,
            "zscore": zscore,
        },
        index=df.index,
    )


def _session_poc_series(
    typ: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    vol: np.ndarray,
    *,
    buckets: int,
) -> np.ndarray:
    """Per-bar point of control over the session-to-date.

    For each bar ``i``: split the session-to-date [min(low), max(high)]
    range into ``buckets`` equal buckets, accumulate volume by each bar's
    typical price, and take the highest-volume bucket's midpoint -- the
    astra ``compute_vwap_volume_profile`` semantics, made per-bar and
    causal (bar ``i`` only ever sees bars ``<= i``). ``argmax`` takes the
    first maximum, matching the scalar source's ``max(range(...), key=...)``
    tie-break.
    """
    m = len(typ)
    out = np.full(m, np.nan)
    if m == 0 or buckets < 1:
        return out
    run_lo = np.minimum.accumulate(lows)
    run_hi = np.maximum.accumulate(highs)
    # Incremental fast path: when the session-to-date [lo, hi] range hasn't
    # changed since the previous bar, the bucket boundaries are identical and
    # the histogram just gains the new bar (O(1)). A range expansion forces
    # one full recompute over the prefix. Same numbers as the naive per-bar
    # loop, ~30x faster on typical sessions.
    hist: np.ndarray | None = None
    prev_lo = prev_hi = np.nan
    width = np.nan
    for i in range(m):
        lo, hi = run_lo[i], run_hi[i]
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            hist = None
            prev_lo = prev_hi = np.nan
            continue
        if hist is not None and lo == prev_lo and hi == prev_hi:
            idx = int((typ[i] - lo) / width)
            idx = 0 if idx < 0 else (buckets - 1 if idx >= buckets else idx)
            hist[idx] += vol[i]
        else:
            width = (hi - lo) / buckets
            seg = typ[: i + 1]
            idx = np.floor((seg - lo) / width).astype(int)
            idx = np.clip(idx, 0, buckets - 1)
            hist = np.bincount(idx, weights=vol[: i + 1], minlength=buckets)
            prev_lo, prev_hi = lo, hi
        if hist.sum() <= 0:
            continue
        best = int(np.argmax(hist))
        out[i] = lo + (best + 0.5) * width
    return out


# ---------------------------------------------------------------------------
# Manual-builder operand surface
# ---------------------------------------------------------------------------

# Manual kind -> session_vwap_profile column.
KIND_TO_COLUMN = {
    "session_vwap": "vwap",
    "vwap_sigma": "sigma",
    "vwap_vah": "vah",
    "vwap_val": "val",
    "vwap_upper_2": "upper_2",
    "vwap_lower_2": "lower_2",
    "vwap_poc": "poc",
    "vwap_zscore": "zscore",
    # Position flags: 1.0 when the condition holds, else 0.0.
    "vwap_above": None,   # close > vwap
    "vwap_below": None,   # close < vwap
    "vwap_outside_value_area": None,  # |zscore| >= value_area_sigma
}

VWAP_PROFILE_KINDS = frozenset(KIND_TO_COLUMN)


def vwap_operand_series(
    df: pd.DataFrame,
    kind: str,
    *,
    roll_hour: int = DEFAULT_ROLL_HOUR,
    value_area_sigma: float = 1.0,
    extreme_sigma: float = 2.0,
    poc_buckets: int = 30,
    profile: pd.DataFrame | None = None,
) -> pd.Series:
    """One Manual-builder operand series for a VWAP-profile kind.

    ``profile`` may be a precomputed ``session_vwap_profile`` frame (the
    caller can share one across several kinds); otherwise it is computed
    here. Position-flag kinds return 1.0/0.0 floats so conditions can use
    "is true".
    """
    kind = str(kind).lower()
    if kind not in KIND_TO_COLUMN:
        raise ValueError(f"Unknown VWAP profile kind '{kind}'.")
    if profile is None:
        profile = session_vwap_profile(
            df, roll_hour=roll_hour, value_area_sigma=value_area_sigma,
            extreme_sigma=extreme_sigma, poc_buckets=poc_buckets,
        )
    col = KIND_TO_COLUMN[kind]
    if col is not None:
        return profile[col].astype(float)
    close = df["close"].to_numpy(dtype=float)
    vwap = profile["vwap"].to_numpy(dtype=float)
    if kind == "vwap_above":
        return pd.Series(np.where(close > vwap, 1.0, 0.0), index=df.index)
    if kind == "vwap_below":
        return pd.Series(np.where(close < vwap, 1.0, 0.0), index=df.index)
    # vwap_outside_value_area
    z = profile["zscore"].to_numpy(dtype=float)
    flag = np.where(np.isfinite(z) & (np.abs(z) >= value_area_sigma), 1.0, 0.0)
    return pd.Series(flag, index=df.index)
