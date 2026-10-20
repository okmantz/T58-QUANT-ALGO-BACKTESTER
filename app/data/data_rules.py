"""Data rules for the next data round (report 2026-10-20, section 5).

* Proxy history: MNQ / MES / MYM / M2K / MGC contracts only exist since 2019, but they track NQ / ES / YM / RTY / GC point for point
  (the micro is 1/10th the multiplier). ``proxy_history`` lets the micro INHERIT the full-size history for backtests, and stamps
  ``df.attrs['proxy_history']`` so every report can say "history from NQ".
* Gaps: continuous-contract rolls create price gaps that are not tradable. ``flag_roll_gaps`` marks bars whose open-to-prior-close
  gap is a roll (supplied roll dates) and ``unadjusted_gap`` returns the gap against the PRIOR SETTLE (unadjusted), which is what an
  overnight-gap strategy can actually trade.
* Session marks: RTH / ETH, holidays, half days, DST. ``add_session_marks`` adds boolean columns computed from the exchange
  calendar, so strategies and audits agree on which bars are regular, shortened or closed.
* Gold: ``resample_gold`` builds 15m-60m bars from finer data and attaches the 4h / daily filter columns used by the gold families.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROXY_MAP = {"MNQ": "NQ", "MES": "ES", "MYM": "YM", "M2K": "RTY", "MGC": "GC", "SIL": "SI"}


def proxy_source_for(symbol: str) -> str | None:
    return PROXY_MAP.get(str(symbol or "").strip().upper())


def proxy_history(micro_symbol: str, micro_df: pd.DataFrame | None, full_df: pd.DataFrame) -> pd.DataFrame:
    """Micro bars where they exist, the full-size contract's bars before that. Same price scale (point-for-point), so no rescale.

    The result carries attrs['proxy_history'] = {'symbol', 'source', 'proxy_rows', 'proxy_until'}; reports must print it."""
    src = proxy_source_for(micro_symbol) or "?"
    f = full_df.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    if micro_df is None or micro_df.empty:
        out = f
        n_proxy, until = len(f), f["timestamp"].max()
    else:
        m = micro_df.copy()
        m["timestamp"] = pd.to_datetime(m["timestamp"])
        first = m["timestamp"].min()
        head = f[f["timestamp"] < first]
        out = pd.concat([head, m], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        n_proxy, until = len(head), (head["timestamp"].max() if len(head) else None)
    out.attrs["proxy_history"] = {"symbol": micro_symbol.upper(), "source": src, "proxy_rows": int(n_proxy),
                                  "proxy_until": None if until is None or pd.isna(until) else str(until)}
    return out


def proxy_label(df: pd.DataFrame) -> str | None:
    ph = getattr(df, "attrs", {}).get("proxy_history")
    if not ph or not ph.get("proxy_rows"):
        return None
    return f"{ph['symbol']} history before {ph.get('proxy_until')} is PROXY data from {ph['source']} ({ph['proxy_rows']:,} bars)."


def flag_roll_gaps(df: pd.DataFrame, roll_dates: list, window_bars: int = 1) -> pd.Series:
    """True for the first bar after each roll date (the gap there is a contract change, not a market move)."""
    t = pd.to_datetime(df["timestamp"])
    flag = pd.Series(False, index=df.index)
    for rd in roll_dates or []:
        pos = np.searchsorted(t.to_numpy(), np.datetime64(pd.Timestamp(rd).tz_localize(None) if pd.Timestamp(rd).tzinfo else pd.Timestamp(rd)))
        flag.iloc[pos:pos + window_bars] = True if pos < len(df) else False
    return flag


def unadjusted_gap(df: pd.DataFrame, settle_hour_et: int = 16, settle_minute: int = 0) -> pd.Series:
    """Per-bar gap vs the prior session's UNADJUSTED settle (last close at/before the RTH close), in price points, NaN until known.
    Using the unadjusted settle keeps back-adjusted continuous-contract history from fabricating or hiding overnight gaps."""
    t = pd.to_datetime(df["timestamp"])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    m = t.dt.hour * 60 + t.dt.minute
    is_settle = m <= settle_hour_et * 60 + settle_minute
    day = (t + pd.Timedelta(hours=6)).dt.normalize()
    settle = df["close"].where(is_settle).groupby(day).transform("last")
    prior_settle = settle.groupby(day).first().shift(1)
    return df["close"] - day.map(prior_settle).astype(float)


def _nyse_holidays(start, end) -> pd.DatetimeIndex:
    from app.strategy.indicators_v11 import _HOLS
    return _HOLS.holidays(start=pd.Timestamp(start) - pd.Timedelta(days=5), end=pd.Timestamp(end) + pd.Timedelta(days=5))


def half_days(year: int) -> list[pd.Timestamp]:
    """Early-close days (13:00 ET): the day after Thanksgiving, Dec 24 (weekday), and July 3 when a weekday before July 4."""
    from pandas.tseries.holiday import USThanksgivingDay
    tg = USThanksgivingDay.dates(f"{year}-11-01", f"{year}-11-30")[0]
    out = [tg + pd.Timedelta(days=1)]
    d24 = pd.Timestamp(year=year, month=12, day=24)
    if d24.dayofweek < 5:
        out.append(d24)
    j3 = pd.Timestamp(year=year, month=7, day=3)
    if j3.dayofweek < 5 and pd.Timestamp(year=year, month=7, day=4).dayofweek < 5:
        out.append(j3)
    return out


def add_session_marks(df: pd.DataFrame, tz_offset_hours: float = 0.0) -> pd.DataFrame:
    """Adds is_rth, is_holiday, is_half_day, is_early_close_window and dst_week columns (Eastern wall-clock timestamps in)."""
    out = df.copy()
    t = pd.to_datetime(out["timestamp"])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    if tz_offset_hours:
        t = t + pd.Timedelta(hours=tz_offset_hours)
    m = t.dt.hour * 60 + t.dt.minute
    day = t.dt.normalize()
    hol = set(_nyse_holidays(t.min(), t.max()).normalize())
    halves = {h for y in range(t.min().year, t.max().year + 1) for h in half_days(y)}
    out["is_holiday"] = day.isin(hol)
    out["is_half_day"] = day.isin(halves)
    out["is_rth"] = (m >= 570) & (m < np.where(out["is_half_day"], 13 * 60, 16 * 60)) & ~out["is_holiday"]
    out["is_early_close_window"] = out["is_half_day"] & (m >= 13 * 60)
    # DST: the US switches on the 2nd Sunday of March and 1st Sunday of November; flag the Mon-Fri week after each switch, when
    # non-US sessions (London open) shift by an hour relative to New York.
    dst_days = set()
    for y in range(t.min().year, t.max().year + 1):
        mar = pd.Timestamp(year=y, month=3, day=1)
        nov = pd.Timestamp(year=y, month=11, day=1)
        sun_mar = mar + pd.Timedelta(days=(6 - mar.dayofweek) % 7 + 7)
        sun_nov = nov + pd.Timedelta(days=(6 - nov.dayofweek) % 7)
        for s in (sun_mar, sun_nov):
            dst_days.update(s + pd.Timedelta(days=k) for k in range(0, 6))
    out["dst_week"] = day.isin(dst_days)
    return out


def resample_gold(df: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    """15m-60m gold bars from finer data plus prior-completed 4h and daily trend columns (no lookahead: shifted one higher-tf bar)."""
    if minutes not in (15, 30, 45, 60):
        raise ValueError("gold bars are 15, 30, 45 or 60 minutes")
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.set_index("timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    base = d.resample(f"{minutes}min").agg(agg).dropna(subset=["open"])
    for name, rule in (("h4", "4h"), ("d1", "1D")):
        hi = d.resample(rule).agg(agg).dropna(subset=["open"])
        trend = np.sign(hi["close"] - hi["close"].ewm(span=20, adjust=False).mean()).shift(1)
        base[f"{name}_trend"] = trend.reindex(base.index, method="ffill")
    return base.reset_index()
