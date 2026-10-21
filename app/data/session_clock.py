"""Canonical exchange clock (report 2026-10-20, P0).

Vendor bars are stamped UTC, but every session / gap / opening-range / event /
VWAP operand in the app means *exchange* time. Reading UTC as exchange time made
those families trade the wrong hours (on NQ the "09:30-16:00 RTH" window held
45.2% of volume under the old clock vs 76.4% with a true Eastern conversion).

This module is the single place the conversion lives:

* ``exchange_zone_for(symbol)``        -- the IANA zone a market's sessions are quoted in.
* ``session_profile(symbol)``          -- zone + trading-day roll rule + RTH window.
* ``stamp_session_columns(df, ...)``   -- adds ``exchange_time`` (tz-naive exchange wall-clock,
                                          DST-aware) and ``trading_day`` (session-day label) at import.
* ``frame_exchange_time(frame)``       -- what every operand reads: the stamped column if present,
                                          else a conversion driven by ``attrs``, else the legacy
                                          "naive == exchange wall-clock" assumption.
* ``frame_trading_day(frame)``         -- same idea for the day boundary.
* ``clock_quality_issues(df, symbol)`` -- RTH-volume-share / peak-minute checks used by the importer
                                          warning, the data-quality gate and the regression tests.

Convention: a naive timestamp is UTC unless ``df.attrs['source_timezone']`` says otherwise
(``app/data/trading_day.py`` and the importer already follow this). Frames with NO provenance at all
(hand-built test frames, pasted data) keep the legacy reading -- naive == exchange wall-clock -- and
``has_clock_provenance`` reports False so audits can flag them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from app.data.trading_day import trading_days

EXCHANGE_TIME_COL = "exchange_time"
TRADING_DAY_COL = "trading_day"

NY = "America/New_York"
CHI = "America/Chicago"
UTC = "UTC"

# Equity-index futures: the markets the RTH-share regression tests apply to.
EQUITY_INDEX_ROOTS = {"ES", "MES", "NQ", "MNQ", "YM", "MYM", "RTY", "M2K"}
# Other CME/COMEX/NYMEX/CBOT roots quoted on the New York clock for session purposes.
US_FUTURES_ROOTS = EQUITY_INDEX_ROOTS | {"GC", "MGC", "SI", "SIL", "HG", "PL", "CL", "MCL", "QM", "NG",
                                         "ZN", "ZB", "ZF", "ZT", "UB", "6E", "6B", "6J", "6A", "6C", "6S", "E7"}
FX_QUOTES = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}
CRYPTO_ROOTS = {"BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "LTC", "BITCOIN", "ETHEREUM", "SOLANA", "DOGECOIN"}
METAL_SPOT = {"XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD", "XCUUSD"}
INDEX_CFDS = {"NAS100", "SPX500", "US30", "US2000", "US500", "USTEC", "GER40", "UK100", "JP225"}
ENERGY_CFDS = {"BRENT", "UKOIL", "USOIL", "WTI", "NATGAS"}
# folder / file-name words -> canonical symbol (the bundled data folders are named like "S&P 500" or "NASDAQ 100")
NAME_ALIASES = (("S&P 500", "SPX500"), ("S&P500", "SPX500"), ("NASDAQ", "NAS100"), ("DOW JONES", "US30"),
                ("RUSSELL", "US2000"), ("BRENT", "BRENT"), ("SILVER", "XAGUSD"), ("GOLD", "XAUUSD"),
                ("COPPER", "XCUUSD"), ("PLATINUM", "XPTUSD"), ("BITCOIN", "BTC"), ("ETHEREUM", "ETH"),
                ("SOLANA", "SOL"), ("DOGECOIN", "DOGE"))

# Stated session zone per FX pair. All pairs default to the New York clock (the FX trading day rolls
# at 17:00 New York); edit this table or drop a ``config/exchange_zones.json`` ({"USDJPY": "Asia/Tokyo"})
# to quote a pair's windows on another clock. Every override is DST-aware via zoneinfo.
FX_SESSION_ZONES: dict[str, str] = {}

RTH_OPEN_MIN = 9 * 60 + 30
RTH_CLOSE_MIN = 16 * 60
MIN_RTH_VOLUME_SHARE = 0.65     # equity-index futures must clear this on a correctly stamped clock


@dataclass(frozen=True)
class SessionProfile:
    symbol: str
    kind: str                 # equity_index | us_future | fx | metal_spot | crypto | unknown
    zone: str                 # IANA zone the session windows are quoted in
    roll_tz: str              # zone of the trading-day roll
    roll_hour: int            # local hour at which the trading day rolls into the next date
    has_rth: bool             # True when a 09:30-16:00 ET cash session is meaningful


def _config_overrides() -> dict[str, str]:
    try:
        p = Path(__file__).resolve().parents[2] / "config" / "exchange_zones.json"
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            return {str(k).upper(): str(v) for k, v in raw.items()}
    except Exception:  # noqa: BLE001 - a bad override file must never break an import
        pass
    return {}


def normalize_symbol(symbol: str | None) -> str:
    """'NQ1!' -> 'NQ', 'futures_NQ.F_1m' -> 'NQ', 'XAUUSD (GOLD)' -> 'XAUUSD', 'MNQ1!' -> 'MNQ'."""
    s = str(symbol or "").strip().upper()
    if not s:
        return ""
    m = re.search(r"FUTURES[_\- ]([A-Z0-9]{1,4})[._\-]", s)
    if m:
        return m.group(1)
    s = re.split(r"[\s(]", s)[0]
    s = re.sub(r"\d*!$", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    m = re.match(r"^([A-Z]{1,4}?)[FGHJKMNQUVXZ]\d{1,2}$", s)          # ESZ6 -> ES
    if m and m.group(1) in US_FUTURES_ROOTS:
        return m.group(1)
    return s


def infer_symbol(source_name: str | None) -> str:
    """Best-effort instrument root from a file / folder name; '' when nothing is recognisable."""
    name = str(source_name or "")
    if not name:
        return ""
    up = name.upper()
    for key, sym in NAME_ALIASES:
        if key in up:
            return sym
    stem = Path(name).name if ("/" in name or "\\" in name) else name
    parent = Path(name).parent.name if ("/" in name or "\\" in name) else ""
    for cand in (stem, parent):
        sym = normalize_symbol(re.split(r"[_\- .]", cand)[0] if cand.lower().startswith(("nq", "es", "ym", "gc", "si", "mnq", "mes", "mgc", "ys", "rty", "m2k", "cl", "zn", "zb", "6e")) else cand)
        if sym in US_FUTURES_ROOTS or sym in METAL_SPOT or _is_fx(sym) or sym in CRYPTO_ROOTS:
            return sym
    m = re.search(r"(?:^|[_\- /])((?:M?ES|M?NQ|M?YM|M2K|RTY|M?GC|SIL?|MCL|CL|ZN|ZB|ZF|6E))(?:[_\-.!1]|$)", stem.upper())
    if m:
        return m.group(1)
    m = re.search(r"([A-Z]{3}(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|NZD))", stem.upper())
    if m:
        return m.group(1)
    try:
        from app.data.instrument_specs import guess_any_instrument_symbol
        g = guess_any_instrument_symbol(name)
        return normalize_symbol(g) if g else ""
    except Exception:  # noqa: BLE001
        return ""


def _is_fx(sym: str) -> bool:
    return len(sym) == 6 and sym[:3] in FX_QUOTES and sym[3:] in FX_QUOTES and sym not in METAL_SPOT


def session_profile(symbol: str | None) -> SessionProfile:
    sym = normalize_symbol(symbol)
    over = {**FX_SESSION_ZONES, **_config_overrides()}
    if sym in EQUITY_INDEX_ROOTS:
        return SessionProfile(sym, "equity_index", over.get(sym, NY), CHI, 17, True)
    if sym in US_FUTURES_ROOTS:
        return SessionProfile(sym, "us_future", over.get(sym, NY), CHI, 17, sym in EQUITY_INDEX_ROOTS)
    if sym in INDEX_CFDS:
        return SessionProfile(sym, "index_cfd", over.get(sym, NY), NY, 17, False)
    if sym in ENERGY_CFDS:
        return SessionProfile(sym, "energy_cfd", over.get(sym, NY), NY, 17, False)
    if sym in METAL_SPOT:
        return SessionProfile(sym, "metal_spot", over.get(sym, NY), NY, 17, False)
    if _is_fx(sym):
        return SessionProfile(sym, "fx", over.get(sym, NY), NY, 17, False)
    if sym in CRYPTO_ROOTS or sym.endswith(("USDT", "BTC")):
        return SessionProfile(sym, "crypto", UTC, UTC, 0, False)
    # Unknown market: US convention (the families' windows are written for New York); never UTC-by-accident.
    return SessionProfile(sym, "unknown", over.get(sym, NY), CHI, 17, False)


def exchange_zone_for(symbol: str | None) -> str:
    return session_profile(symbol).zone


# ------------------------------------------------------------------------------------------------ conversion
def to_exchange_wallclock(ts: pd.Series | pd.DatetimeIndex, *, source_tz: str = UTC, zone: str = NY) -> pd.Series | pd.DatetimeIndex:
    """Timestamps as tz-NAIVE wall-clock on ``zone`` (DST-aware). Naive input is read on ``source_tz``.

    Ambiguous / non-existent local times (only possible for non-UTC sources) become NaT / shift forward rather than
    guessing silently."""
    is_series = isinstance(ts, pd.Series)
    s = pd.to_datetime(ts)
    idx = pd.DatetimeIndex(s)
    if idx.tz is None:
        if str(source_tz).upper() in ("UTC", "ETC/UTC"):
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
    out = idx.tz_convert(zone).tz_localize(None)
    if is_series:
        return pd.Series(out, index=s.index)
    return out


def _timestamp_series(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" in frame.columns:
        t = pd.to_datetime(frame["timestamp"])
    elif isinstance(frame.index, pd.DatetimeIndex):
        t = pd.Series(frame.index, index=frame.index)
    else:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    return t


def has_clock_provenance(frame: pd.DataFrame) -> bool:
    """True when the frame says what its timestamps mean (a stamped exchange_time column or a source_timezone)."""
    attrs = getattr(frame, "attrs", {}) or {}
    return (EXCHANGE_TIME_COL in frame.columns) or ("source_timezone" in attrs) or ("tz_offset_hours" in attrs)


def frame_exchange_zone(frame: pd.DataFrame, default: str = NY) -> str:
    attrs = getattr(frame, "attrs", {}) or {}
    return str(attrs.get("exchange_timezone") or default)


def frame_exchange_time(frame: pd.DataFrame, zone: str | None = None) -> pd.Series:
    """Tz-naive exchange wall-clock for every bar (aligned to ``frame.index``). This is what operands read.

    Priority: stamped ``exchange_time`` column > ``attrs['source_timezone']`` conversion > legacy
    ``attrs['tz_offset_hours']`` (hours ADDED to the stamp) > naive-as-exchange-wall-clock (legacy / hand-built frames)."""
    if EXCHANGE_TIME_COL in frame.columns and zone is None:
        t = pd.to_datetime(frame[EXCHANGE_TIME_COL])
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_localize(None)
        return t
    t = _timestamp_series(frame)
    attrs = getattr(frame, "attrs", {}) or {}
    tz_zone = zone or frame_exchange_zone(frame)
    if "source_timezone" in attrs:
        out = to_exchange_wallclock(t, source_tz=str(attrs["source_timezone"]), zone=tz_zone)
    elif attrs.get("tz_offset_hours"):
        base = t.dt.tz_localize(None) if getattr(t.dt, "tz", None) is not None else t
        out = base + pd.Timedelta(hours=float(attrs["tz_offset_hours"]))
    else:
        if getattr(t.dt, "tz", None) is not None:
            out = to_exchange_wallclock(t, zone=tz_zone)
        else:
            out = t
    out.index = frame.index
    return out


def frame_trading_day(frame: pd.DataFrame, symbol: str | None = None) -> pd.Series:
    """Session-day label (tz-naive midnight Timestamp) per bar, using the instrument's roll rule (CME: 17:00 CT)."""
    if TRADING_DAY_COL in frame.columns:
        return pd.to_datetime(frame[TRADING_DAY_COL])
    attrs = getattr(frame, "attrs", {}) or {}
    prof = session_profile(symbol or attrs.get("instrument") or attrs.get("symbol"))
    t = _timestamp_series(frame)
    if has_clock_provenance(frame) and "source_timezone" in attrs:
        utc = to_exchange_wallclock(t, source_tz=str(attrs["source_timezone"]), zone=UTC)
        days = trading_days(pd.DatetimeIndex(utc.to_numpy()), tz=prof.roll_tz, roll_hour=prof.roll_hour)
    else:
        # exchange wall-clock in, no UTC to anchor on: roll at 18:00 ET (== 17:00 CT) on the New York clock.
        ex = frame_exchange_time(frame)
        shift = (24 - (18 if prof.roll_tz in (CHI, NY) and prof.kind != "fx" else prof.roll_hour)) % 24
        days = (ex + pd.Timedelta(hours=shift)).dt.normalize().to_numpy()
    return pd.Series(pd.to_datetime(days), index=frame.index)


def session_day_key(frame: pd.DataFrame) -> pd.Series:
    """Day label used by VWAP / previous-day / pivot / ADR style operands.

    Frames with clock provenance group by the instrument's trading day (CME: 17:00 CT roll, so the evening Globex session
    belongs to the NEXT day); hand-built frames with none keep the legacy calendar-day grouping of their stamps."""
    if TRADING_DAY_COL in frame.columns:
        return pd.to_datetime(frame[TRADING_DAY_COL])
    if has_clock_provenance(frame):
        return frame_trading_day(frame)
    t = _timestamp_series(frame)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)          # wall-clock date of the stamp (legacy behaviour for tz-aware hand-built frames)
    return t.dt.normalize()


def stamp_session_columns(df: pd.DataFrame, *, symbol: str | None = None, source_tz: str | None = None,
                          overwrite: bool = True) -> pd.DataFrame:
    """Add ``exchange_time`` + ``trading_day`` columns and the matching attrs. Mutates and returns ``df``."""
    if df is None or df.empty or "timestamp" not in df.columns:
        return df
    prof = session_profile(symbol)
    src = source_tz or (getattr(df, "attrs", {}) or {}).get("source_timezone") or UTC
    t = pd.to_datetime(df["timestamp"])
    if getattr(t.dt, "tz", None) is not None:
        src = "UTC" if str(t.dt.tz) in ("UTC", "tzutc()") else src
    if overwrite or EXCHANGE_TIME_COL not in df.columns:
        df[EXCHANGE_TIME_COL] = to_exchange_wallclock(t, source_tz=src, zone=prof.zone)
    if overwrite or TRADING_DAY_COL not in df.columns:
        utc = to_exchange_wallclock(t, source_tz=src, zone=UTC)
        days = trading_days(pd.DatetimeIndex(utc.to_numpy()), tz=prof.roll_tz, roll_hour=prof.roll_hour)
        df[TRADING_DAY_COL] = pd.to_datetime(days)
    df.attrs["source_timezone"] = src
    df.attrs["exchange_timezone"] = prof.zone
    df.attrs["session_kind"] = prof.kind
    if prof.symbol:
        df.attrs.setdefault("instrument", prof.symbol)
    return df


# -------------------------------------------------------------------------------------------- quality checks
def rth_volume_share(frame: pd.DataFrame, *, zone: str | None = None) -> float | None:
    """Share of weekday volume inside 09:30-16:00 on the exchange clock (None when it cannot be measured)."""
    if "volume" not in frame.columns or len(frame) == 0:
        return None
    ex = frame_exchange_time(frame, zone=zone) if zone else frame_exchange_time(frame)
    vol = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    m = (ex.dt.hour * 60 + ex.dt.minute).to_numpy()
    wk = (ex.dt.dayofweek < 5).to_numpy()
    tot = float(vol[wk].sum())
    if tot <= 0:
        return None
    return float(vol[wk & (m >= RTH_OPEN_MIN) & (m < RTH_CLOSE_MIN)].sum() / tot)


def peak_volume_minute(frame: pd.DataFrame, *, bucket_minutes: int = 1, start_minute: int = 0, end_minute: int = 24 * 60) -> int | None:
    """Minute-of-day (exchange clock, bucketed) with the highest total weekday volume inside ``[start_minute, end_minute)``;
    ``None`` if no volume. Use a window (e.g. 04:00-12:00 ET) to find the OPEN spike: on NQ the single busiest minute overall is
    the 15:59 close, so the unwindowed peak is not an open-clock test."""
    if "volume" not in frame.columns or len(frame) == 0:
        return None
    ex = frame_exchange_time(frame)
    vol = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    mod = ex.dt.hour * 60 + ex.dt.minute
    keep = (ex.dt.dayofweek < 5) & (mod >= start_minute) & (mod < end_minute)
    m = (mod // bucket_minutes * bucket_minutes)[keep]
    g = vol[keep].groupby(m).sum()
    return int(g.idxmax()) if len(g) and g.max() > 0 else None


def minute_volume(frame: pd.DataFrame, minute_of_day: int) -> float:
    """Total weekday volume at one exchange-clock minute (for open-spike ratios)."""
    ex = frame_exchange_time(frame)
    vol = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    mod = ex.dt.hour * 60 + ex.dt.minute
    return float(vol[(ex.dt.dayofweek < 5) & (mod == minute_of_day)].sum())


def detect_fx_server_offset(frame: pd.DataFrame) -> dict | None:
    """Estimate how far a FX/CFD file's stamps sit from UTC using the weekly open.

    Spot FX re-opens Sunday 17:00 New York = 21:00 UTC (summer) / 22:00 UTC (winter). A broker-server file
    (GMT+2 / GMT+3, NY-close) re-opens Monday 00:00 server time, which is Sunday 22:00 / 21:00 UTC +2/+3 hours.
    Returns {'first_bar_hour': h, 'looks_like': 'utc' | 'broker_server_time' | 'unknown'} or None.
    Warn-only: the importer never shifts data on this evidence."""
    try:
        t = _timestamp_series(frame).dropna()
        if len(t) < 5000:
            return None
        gap = t.diff()
        reopen = t[gap > pd.Timedelta(hours=40)]
        if len(reopen) < 20:
            return None
        hours = reopen.dt.hour.value_counts()
        h = int(hours.idxmax())
        dow = int(reopen.dt.dayofweek.mode().iloc[0])
        if dow == 6 and h in (21, 22):
            kind = "utc"
        elif dow == 0 and h in (0, 1):
            kind = "broker_server_time"
        else:
            kind = "unknown"
        return {"first_bar_hour": h, "first_bar_weekday": dow, "looks_like": kind}
    except Exception:  # noqa: BLE001
        return None


def clock_quality_issues(df: pd.DataFrame, symbol: str | None = None) -> list[tuple[str, str]]:
    """(severity, message) pairs describing session-clock problems. Empty list == clock looks right."""
    out: list[tuple[str, str]] = []
    try:
        if df is None or len(df) < 2000:
            return out
        sym = normalize_symbol(symbol or (getattr(df, "attrs", {}) or {}).get("instrument"))
        prof = session_profile(sym)
        ts = _timestamp_series(df).dropna()
        if len(ts) < 2000 or ts.diff().median() > pd.Timedelta(hours=1):
            return out                      # coarse bars carry no intraday session information
        if not has_clock_provenance(df):
            out.append(("warning", "Session clock: this frame carries no timezone provenance (no exchange_time column or "
                                   "source_timezone), so session operands read the stamps as exchange wall-clock. Re-import the file."))
        if prof.kind == "equity_index":
            share = rth_volume_share(df)
            if share is not None and share < MIN_RTH_VOLUME_SHARE:
                out.append(("warning", f"Session clock: only {share:.1%} of {sym or 'this market'}'s volume falls inside "
                                       f"09:30-16:00 ET (a correctly stamped equity-index futures file holds > {MIN_RTH_VOLUME_SHARE:.0%}). "
                                       "The file is probably not UTC (or is in a different zone) - session, gap, opening-range, "
                                       "VWAP and event families would trade the wrong hours. Set source_timezone or re-export in UTC."))
        elif prof.kind in ("fx", "metal_spot"):
            info = detect_fx_server_offset(df)
            if info and info["looks_like"] == "broker_server_time":
                out.append(("warning", "Session clock: the weekly re-open lands on Monday 00:xx, which is broker server time "
                                       "(GMT+2/+3), not UTC. Pass source_timezone='Europe/Athens' (or your broker's zone) so "
                                       "exchange_time is DST-correct."))
    except Exception:  # noqa: BLE001 - diagnostics must never break an import
        return out
    return out


def assert_clock_ok(df: pd.DataFrame, symbol: str | None = None) -> None:
    """Raise ValueError when the clock check fails (used by the data-quality gate and tests)."""
    issues = clock_quality_issues(df, symbol)
    bad = [m for sev, m in issues if sev in ("warning", "error")]
    if bad:
        raise ValueError("; ".join(bad))


def event_bar_timestamps(events: Iterable[pd.Timestamp], frame: pd.DataFrame) -> pd.Series:
    """Map exchange-wall-clock event stamps (FOMC 14:00 ET, NFP/CPI 08:30 ET) to the matching bar index label.

    Returns the frame index label of the bar whose exchange_time is the last bar at or before each event."""
    ex = frame_exchange_time(frame)
    order = np.argsort(ex.to_numpy())
    vals = ex.to_numpy()[order]
    ev = pd.to_datetime(pd.Series(list(events))).to_numpy()
    pos = np.searchsorted(vals, ev, side="right") - 1
    pos = np.clip(pos, 0, len(vals) - 1)
    return pd.Series(frame.index[order][pos], index=range(len(ev)))
