"""v11 indicators: session, calendar, event and regime primitives (report 2026-10-20, section 5/6).

The report's diagnosis: the search space was oscillator thresholds on 5m bars, where there is little structural edge. The
primitives below are the building blocks for structural / event / cross-asset families instead:

  session clock      minutes_since_open, minutes_to_close, session_phase, rth_flag
  daily context      daily_trend_state, daily_ema_distance, prior_day_range_atr        (prior COMPLETED days only)
  overnight          overnight_range_atr, overnight_position
  calendar flags     day_of_week, month_end_window, turn_of_month, opex_week, triple_witching, pre_holiday,
                     quarter_end_week, first_trading_day
  event proximity    minutes_to_{fomc,nfp,cpi,pce}, minutes_since_{...}, event_day_{...}, event_day_any
  regime inputs      vix_level, vix_term_slope, vix_percentile, curve_10y2y, dollar_change_20d   (prior-day values)

Conventions
* Bar timestamps are exchange wall-clock (US/Eastern) and tz-naive; set ``frame.attrs['tz_offset_hours']`` if your file is in
  another zone (value is ADDED to the timestamp to get Eastern time).
* No lookahead: schedule-derived features use only the clock; daily features use prior completed days (shift(1) on the daily
  table); overnight features are frozen at the open; macro series enter through ``prior_day_value`` (strictly earlier dates).
* Missing auxiliary data: event/session/calendar kinds are self-contained. Regime kinds return NaN when the cached macro series
  is absent, so a condition built on them is simply False (no trades) instead of firing on a fake neutral value.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
                                    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday)
from pandas.tseries.offsets import CustomBusinessDay

V11IndicatorBuilder = Callable[[pd.DataFrame, int, str, int], pd.Series]
NO_EVENT_MINUTES = 100_000.0
RTH_OPEN_MIN, RTH_CLOSE_MIN = 9 * 60 + 30, 16 * 60


class _NYSEHolidays(AbstractHolidayCalendar):
    """Approximate NYSE full-day closures (no Columbus/Veterans day; Good Friday included; Juneteenth from 2022)."""
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday), USMartinLutherKingJr, USPresidentsDay, GoodFriday,
        USMemorialDay, Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("USIndependenceDay", month=7, day=4, observance=nearest_workday), USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_HOLS = _NYSEHolidays()


def _bday(start, end) -> CustomBusinessDay:
    return CustomBusinessDay(holidays=_HOLS.holidays(start=pd.Timestamp(start) - pd.Timedelta(days=370), end=pd.Timestamp(end) + pd.Timedelta(days=370)))


def _ts(frame: pd.DataFrame) -> pd.Series:
    """Exchange wall-clock timestamps (tz-naive, DST-aware) aligned to frame.index (NaT-safe).

    P0 (report 2026-10-20): vendor bars are UTC. This reads the importer-stamped ``exchange_time`` column, else converts
    from ``attrs['source_timezone']`` to the instrument's exchange zone (America/New_York for US futures / metals),
    else (hand-built frames with no provenance) keeps the legacy "naive == exchange wall-clock" reading.
    ``attrs['tz_offset_hours']`` still works for legacy files (hours ADDED to the stamp)."""
    from app.data.session_clock import frame_exchange_time
    if "timestamp" not in frame.columns and not isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(pd.NaT, index=frame.index)
    return frame_exchange_time(frame)


def _nan(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=frame.index)


def _minute_of_day(t: pd.Series) -> pd.Series:
    return t.dt.hour * 60 + t.dt.minute


# ---------------------------------------------------------------------------------------------------- session clock
def minutes_since_open(f, p, c, l):
    t = _ts(f)
    return (_minute_of_day(t) - RTH_OPEN_MIN).astype(float)


def minutes_to_close(f, p, c, l):
    t = _ts(f)
    return (RTH_CLOSE_MIN - _minute_of_day(t)).astype(float)


def session_phase(f, p, c, l):
    """0 = Asia/overnight (18:00-03:00), 1 = London/pre-market (03:00-09:30), 2 = RTH (09:30-16:00), 3 = post-close (16:00-18:00)."""
    m = _minute_of_day(_ts(f))
    out = np.where((m >= RTH_OPEN_MIN) & (m < RTH_CLOSE_MIN), 2, np.where((m >= 180) & (m < RTH_OPEN_MIN), 1,
                   np.where((m >= RTH_CLOSE_MIN) & (m < 18 * 60), 3, 0)))
    return pd.Series(out.astype(float), index=f.index)


def rth_flag(f, p, c, l):
    m = _minute_of_day(_ts(f))
    return ((m >= RTH_OPEN_MIN) & (m < RTH_CLOSE_MIN)).astype(float)


# ------------------------------------------------------------------------------------------------------ daily context
def _trading_day(t: pd.Series) -> pd.Series:
    """Trading-day label: the 18:00 ET session start rolls into the NEXT date (so the overnight belongs to the day it leads into)."""
    return (t + pd.Timedelta(hours=6)).dt.normalize()


def _frame_day(f: pd.DataFrame) -> pd.Series:
    """Trading-day label per bar: the importer-stamped ``trading_day`` column (17:00 CT roll), else derived from the clock."""
    if "trading_day" in f.columns:
        return pd.to_datetime(f["trading_day"])
    return _trading_day(_ts(f))


def _daily_table(f: pd.DataFrame) -> pd.DataFrame:
    t = _ts(f)
    d = pd.DataFrame({"day": _frame_day(f), "h": f["high"].astype(float), "l": f["low"].astype(float), "c": f["close"].astype(float)})
    g = d.groupby("day", sort=True)
    daily = pd.DataFrame({"high": g["h"].max(), "low": g["l"].min(), "close": g["c"].last()})
    prev_close = daily["close"].shift(1)
    tr = pd.concat([daily["high"] - daily["low"], (daily["high"] - prev_close).abs(), (daily["low"] - prev_close).abs()], axis=1).max(axis=1)
    daily["atr14"] = tr.rolling(14, min_periods=5).mean()
    return daily


def _prior_day_map(f: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Map a per-trading-day series to bars using the PREVIOUS trading day's value (shift(1) on the daily table)."""
    day = _frame_day(f)
    return pd.Series(day.map(values.shift(1)).to_numpy(dtype=float), index=f.index)


def daily_trend_state(f, p, c, l):
    """+1 when the prior daily close is above EMA50 and EMA50 above EMA200, -1 for the mirror image, else 0 (0 until 200 days exist)."""
    try:
        d = _daily_table(f)
        e50, e200 = d["close"].ewm(span=50, adjust=False).mean(), d["close"].ewm(span=200, adjust=False).mean()
        ok = d["close"].expanding().count() >= 200
        s = pd.Series(np.where(ok & (d["close"] > e50) & (e50 > e200), 1.0, np.where(ok & (d["close"] < e50) & (e50 < e200), -1.0, 0.0)), index=d.index)
        return _prior_day_map(f, s).fillna(0.0)
    except Exception:  # noqa: BLE001
        return pd.Series(0.0, index=f.index)


def daily_ema_distance(f, p, c, l):
    """(prior daily close - daily EMA(p, default 50)) / daily ATR14, both as of the prior completed day."""
    try:
        n = int(p) if p and int(p) > 1 else 50
        d = _daily_table(f)
        dist = (d["close"] - d["close"].ewm(span=n, adjust=False).mean()) / d["atr14"].replace(0, np.nan)
        return _prior_day_map(f, dist)
    except Exception:  # noqa: BLE001
        return _nan(f)


def prior_day_range_atr(f, p, c, l):
    try:
        d = _daily_table(f)
        return _prior_day_map(f, (d["high"] - d["low"]) / d["atr14"].replace(0, np.nan))
    except Exception:  # noqa: BLE001
        return _nan(f)


# --------------------------------------------------------------------------------------------------------- overnight
def _overnight_frame(f: pd.DataFrame) -> pd.DataFrame:
    t = _ts(f)
    m = _minute_of_day(t)
    day = _frame_day(f)
    ovn = (m < RTH_OPEN_MIN) | (m >= 18 * 60)
    h = f["high"].astype(float).where(ovn)
    lo = f["low"].astype(float).where(ovn)
    df = pd.DataFrame({"day": day, "h": h, "l": lo})
    hi_run = df.groupby("day")["h"].cummax()
    lo_run = df.groupby("day")["l"].cummin()
    # forward-fill within the day so RTH bars keep the frozen overnight range
    df["hi"] = hi_run.groupby(df["day"]).ffill()
    df["lo"] = lo_run.groupby(df["day"]).ffill()
    return df


def overnight_range_atr(f, p, c, l):
    try:
        o = _overnight_frame(f)
        d = _daily_table(f)
        atr_prior = _prior_day_map(f, d["atr14"])
        return ((o["hi"] - o["lo"]) / atr_prior.replace(0, np.nan)).astype(float)
    except Exception:  # noqa: BLE001
        return _nan(f)


def overnight_position(f, p, c, l):
    """Where the close sits inside the overnight (18:00-09:30) range so far, 0 = at the low, 1 = at the high (clipped)."""
    try:
        o = _overnight_frame(f)
        rng = (o["hi"] - o["lo"]).replace(0, np.nan)
        return ((f["close"].astype(float) - o["lo"]) / rng).clip(0.0, 1.0)
    except Exception:  # noqa: BLE001
        return _nan(f)


# ----------------------------------------------------------------------------------------------------- calendar flags
def _dates(f):
    t = _ts(f)
    return t, t.dt.normalize()


def _calendar_lookup(f: pd.DataFrame, fn) -> pd.Series:
    """Evaluate ``fn(date, bday)`` once per distinct date and map back to bars."""
    t, d = _dates(f)
    uniq = d.dropna().unique()
    if len(uniq) == 0:
        return pd.Series(0.0, index=f.index)
    bd = _bday(uniq.min(), uniq.max())
    table = {u: float(fn(pd.Timestamp(u), bd)) for u in uniq}
    return pd.Series(d.map(table).to_numpy(dtype=float), index=f.index).fillna(0.0)


def weekday_number(f, p, c, l):
    return _ts(f).dt.dayofweek.astype(float)


def _third_friday(y: int, m: int) -> pd.Timestamp:
    first = pd.Timestamp(year=y, month=m, day=1)
    return first + pd.Timedelta(days=(4 - first.dayofweek) % 7 + 14)


def month_end_window(f, p, c, l):
    """1 on the last ``p`` (default 3) trading days of the month."""
    n = int(p) if p and int(p) > 0 else 3
    def fn(d, bd):
        end = (d + pd.offsets.MonthEnd(0))
        last = end if bd.is_on_offset(end) else end - bd
        return 1.0 if d >= last - bd * (n - 1) and d <= last else 0.0
    return _calendar_lookup(f, fn)


def turn_of_month(f, p, c, l):
    """1 from the last trading day of a month through the 3rd trading day of the next (the classic turn-of-month window)."""
    def fn(d, bd):
        end = d + pd.offsets.MonthEnd(0)
        last = end if bd.is_on_offset(end) else end - bd
        start = d.replace(day=1)
        first = start if bd.is_on_offset(start) else start + bd
        return 1.0 if (d == last or (bd.is_on_offset(d) and d <= first + bd * 2 and d >= first)) else 0.0
    return _calendar_lookup(f, fn)


def first_trading_day(f, p, c, l):
    def fn(d, bd):
        start = d.replace(day=1)
        first = start if bd.is_on_offset(start) else start + bd
        return 1.0 if d == first else 0.0
    return _calendar_lookup(f, fn)


def opex_week(f, p, c, l):
    """1 on every day of the week (Mon-Fri) that contains the monthly options expiration (3rd Friday)."""
    def fn(d, bd):
        tf = _third_friday(d.year, d.month)
        monday = tf - pd.Timedelta(days=4)
        return 1.0 if monday <= d <= tf else 0.0
    return _calendar_lookup(f, fn)


def triple_witching(f, p, c, l):
    """1 on the quarterly expiration day (3rd Friday of Mar/Jun/Sep/Dec; the prior Thursday if Friday is a holiday)."""
    def fn(d, bd):
        if d.month not in (3, 6, 9, 12):
            return 0.0
        tf = _third_friday(d.year, d.month)
        tf = tf if bd.is_on_offset(tf) else tf - bd
        return 1.0 if d == tf else 0.0
    return _calendar_lookup(f, fn)


def quarter_end_week(f, p, c, l):
    """1 on the last 5 trading days of Mar/Jun/Sep/Dec (rebalancing flows)."""
    def fn(d, bd):
        if d.month not in (3, 6, 9, 12):
            return 0.0
        end = d + pd.offsets.MonthEnd(0)
        last = end if bd.is_on_offset(end) else end - bd
        return 1.0 if last - bd * 4 <= d <= last else 0.0
    return _calendar_lookup(f, fn)


def pre_holiday(f, p, c, l):
    """1 on the trading day immediately before an exchange holiday (next calendar weekday is a closure)."""
    def fn(d, bd):
        if not bd.is_on_offset(d):
            return 0.0
        nxt = d + pd.Timedelta(days=1)
        while nxt.dayofweek >= 5:
            nxt += pd.Timedelta(days=1)
        return 0.0 if bd.is_on_offset(nxt) else 1.0
    return _calendar_lookup(f, fn)


# ---------------------------------------------------------------------------------------------------- event proximity
def _events(f: pd.DataFrame, name: str | None) -> pd.DataFrame:
    ev = f.attrs.get("events") if hasattr(f, "attrs") else None
    if not isinstance(ev, pd.DataFrame) or ev.empty:
        from app.data.event_calendar import load_events
        ev = load_events()
    ev = ev.copy()
    ev.columns = [str(x).lower() for x in ev.columns]
    if "event" not in ev.columns:
        ev["event"] = "EVENT"
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])
    if name:
        ev = ev[ev["event"].astype(str).str.upper() == name.upper()]
    return ev.sort_values("timestamp")


def _minutes(f: pd.DataFrame, name: str | None, direction: str) -> pd.Series:
    ev = _events(f, name)
    t = _ts(f)
    if ev.empty or t.isna().all():
        return pd.Series(NO_EVENT_MINUTES, index=f.index)
    left = pd.DataFrame({"t": t.to_numpy(), "i": np.arange(len(f))}).sort_values("t")
    right = ev[["timestamp"]].rename(columns={"timestamp": "e"})
    m = pd.merge_asof(left, right.assign(t=right["e"]), on="t", direction=direction)
    delta = (m["t"] - m["e"]).dt.total_seconds() / 60.0 if direction == "backward" else (m["e"] - m["t"]).dt.total_seconds() / 60.0
    out = np.full(len(f), NO_EVENT_MINUTES)
    out[m["i"].to_numpy()] = delta.fillna(NO_EVENT_MINUTES).to_numpy()
    return pd.Series(out, index=f.index)


def _event_day(f: pd.DataFrame, name: str | None) -> pd.Series:
    ev = _events(f, name)
    t, d = _dates(f)
    days = set(ev["timestamp"].dt.normalize().tolist())
    return pd.Series(d.isin(days).astype(float).to_numpy(), index=f.index)


def _mk_event_kinds(name: str):
    key = name.lower()
    return (
        (f"minutes_to_{key}", lambda f, p, c, l: _minutes(f, name, "forward")),
        (f"minutes_since_{key}", lambda f, p, c, l: _minutes(f, name, "backward")),
        (f"event_day_{key}", lambda f, p, c, l: _event_day(f, name)),
    )


def event_day_any(f, p, c, l):
    return _event_day(f, None)


# --------------------------------------------------------------------------------------------------------- regime
def _macro(f: pd.DataFrame, name: str) -> pd.Series:
    attrs = getattr(f, "attrs", {}) or {}
    m = attrs.get("macro")
    if isinstance(m, dict) and name in m:
        return m[name]
    from app.data.macro_data import load_series
    return load_series(name)


def _prior(f: pd.DataFrame, s: pd.Series) -> np.ndarray:
    from app.data.macro_data import prior_day_value
    return prior_day_value(_ts(f), s, lag_days=1)


def vix_level(f, p, c, l):
    s = _macro(f, "VIX")
    return pd.Series(_prior(f, s), index=f.index) if len(s) else _nan(f)


def vix_term_slope(f, p, c, l):
    """VIX / VIX3M - 1 as of the prior close: > 0 is backwardation (stress), < 0 contango."""
    a, b = _macro(f, "VIX"), _macro(f, "VIX3M")
    if not len(a) or not len(b):
        return _nan(f)
    r = (a / b.reindex(a.index)).dropna() - 1.0
    return pd.Series(_prior(f, r), index=f.index)


def vix_percentile(f, p, c, l):
    """Percentile rank (0-100) of the prior-day VIX within its trailing ``p`` (default 252) days, prior-day known only."""
    s = _macro(f, "VIX")
    if not len(s):
        return _nan(f)
    n = int(p) if p and int(p) > 20 else 252
    rank = s.rolling(n, min_periods=60).apply(lambda w: 100.0 * (w[:-1] < w[-1]).mean() if len(w) > 1 else np.nan, raw=True)
    return pd.Series(_prior(f, rank.dropna()), index=f.index)


def curve_10y2y(f, p, c, l):
    a, b = _macro(f, "DGS10"), _macro(f, "DGS2")
    if not len(a) or not len(b):
        return _nan(f)
    return pd.Series(_prior(f, (a - b.reindex(a.index)).dropna()), index=f.index)


def dollar_change_20d(f, p, c, l):
    s = _macro(f, "DTWEXBGS")
    if not len(s):
        return _nan(f)
    n = int(p) if p and int(p) > 1 else 20
    return pd.Series(_prior(f, s.pct_change(n).dropna() * 100.0), index=f.index)


V11_INDICATOR_BUILDERS: dict[str, V11IndicatorBuilder] = {
    "minutes_since_open": minutes_since_open, "minutes_to_close": minutes_to_close, "session_phase": session_phase, "rth_flag": rth_flag,
    "daily_trend_state": daily_trend_state, "daily_ema_distance": daily_ema_distance, "prior_day_range_atr": prior_day_range_atr,
    "overnight_range_atr": overnight_range_atr, "overnight_position": overnight_position,
    "weekday_number": weekday_number, "month_end_window": month_end_window, "turn_of_month": turn_of_month, "first_trading_day": first_trading_day,
    "opex_week": opex_week, "triple_witching": triple_witching, "quarter_end_week": quarter_end_week, "pre_holiday": pre_holiday,
    "event_day_any": event_day_any,
    "vix_level": vix_level, "vix_term_slope": vix_term_slope, "vix_percentile": vix_percentile,
    "curve_10y2y": curve_10y2y, "dollar_change_20d": dollar_change_20d,
}
for _n in ("FOMC", "NFP", "CPI", "PCE"):
    for _k, _fn in _mk_event_kinds(_n):
        V11_INDICATOR_BUILDERS[_k] = _fn
V11_INDICATOR_KINDS: tuple[str, ...] = tuple(V11_INDICATOR_BUILDERS)
# Flags that need only the frame's own clock/prices -> safe to hand to the random-search grammar.
V11_SELF_CONTAINED: tuple[str, ...] = (
    "minutes_since_open", "minutes_to_close", "session_phase", "rth_flag", "daily_trend_state", "daily_ema_distance",
    "prior_day_range_atr", "overnight_range_atr", "overnight_position", "weekday_number", "month_end_window", "turn_of_month",
    "first_trading_day", "opex_week", "triple_witching", "quarter_end_week", "pre_holiday",
    "minutes_to_fomc", "minutes_since_fomc", "event_day_fomc", "minutes_to_nfp", "minutes_since_nfp", "event_day_nfp", "event_day_any",
)
_FLAGS = ("rth_flag", "month_end_window", "turn_of_month", "first_trading_day", "opex_week", "triple_witching", "quarter_end_week",
          "pre_holiday", "event_day_any", "event_day_fomc", "event_day_nfp", "event_day_cpi", "event_day_pce")
V11_BOUNDED_RANGES: dict[str, tuple[float, float]] = {
    **{k: (0.0, 1.0) for k in _FLAGS}, "session_phase": (0.0, 3.0), "weekday_number": (0.0, 6.0),
    "daily_trend_state": (-1.0, 1.0), "overnight_position": (0.0, 1.0), "vix_percentile": (0.0, 100.0),
    "minutes_since_open": (-1440.0, 1440.0), "minutes_to_close": (-1440.0, 1440.0),
}
V11_DESCRIPTIONS: dict[str, str] = {
    "minutes_since_open": "Minutes since the 09:30 ET cash open (negative before it).",
    "minutes_to_close": "Minutes until the 16:00 ET cash close.",
    "session_phase": "0 overnight/Asia, 1 London/pre-market, 2 cash session, 3 post-close.",
    "rth_flag": "1 during the 09:30-16:00 ET cash session.",
    "daily_trend_state": "+1 / -1 when the prior daily close and EMA50/EMA200 stack agree up / down, else 0.",
    "daily_ema_distance": "Prior daily close minus its daily EMA, in daily ATR.",
    "prior_day_range_atr": "Prior day's high-low range in daily ATR.",
    "overnight_range_atr": "Overnight (18:00-09:30) range so far in prior daily ATR.",
    "overnight_position": "Where price sits in the overnight range, 0 low .. 1 high.",
    "weekday_number": "0 Monday .. 4 Friday.", "month_end_window": "1 on the last N trading days of the month.",
    "turn_of_month": "1 from the last trading day of a month through the 3rd of the next.",
    "first_trading_day": "1 on the first trading day of the month.", "opex_week": "1 during the week of monthly options expiration.",
    "triple_witching": "1 on quarterly expiration day.", "quarter_end_week": "1 in the last 5 trading days of a quarter.",
    "pre_holiday": "1 on the trading day before an exchange holiday.",
    "event_day_any": "1 on any FOMC/NFP/CPI/PCE release day.",
    "vix_level": "Prior-day VIX close.", "vix_term_slope": "Prior-day VIX/VIX3M - 1 (positive = backwardation).",
    "vix_percentile": "Prior-day VIX percentile within the trailing year.", "curve_10y2y": "Prior-day 10y minus 2y Treasury yield.",
    "dollar_change_20d": "Prior-day broad dollar index 20-day % change.",
}
for _n in ("FOMC", "NFP", "CPI", "PCE"):
    V11_DESCRIPTIONS[f"minutes_to_{_n.lower()}"] = f"Minutes until the next scheduled {_n} release ({int(NO_EVENT_MINUTES)} when none is scheduled)."
    V11_DESCRIPTIONS[f"minutes_since_{_n.lower()}"] = f"Minutes since the last {_n} release."
    V11_DESCRIPTIONS[f"event_day_{_n.lower()}"] = f"1 on days with a scheduled {_n} release."
