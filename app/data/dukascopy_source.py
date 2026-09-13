"""
Free forex/CFD/futures historical tick data from Dukascopy -- fills the
gap flagged in app.data.alpaca_source's own docstring: Alpaca (the only
built-in fetch source before this file existed) covers US equities and
crypto only, and this is explicitly a prop-firm-first tool where most
eval accounts are forex/CFD or futures.

Dukascopy publishes historical tick data for free, with no account,
API key, or rate-limit tier to pay for, at a stable public URL pattern:
    https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM-1:02d}/{DD:02d}/{HH:02d}h_ticks.bi5
(Dukascopy's month component is zero-based -- January is "00" -- which is
the single most common mistake reproducing this URL by hand.)

Each `.bi5` file is one hour of ticks for one symbol, LZMA-compressed
(the "LZMA_ALONE"/legacy format, not the newer .xz container -- decoded
here with `lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)`). Decompressed,
it's a flat array of 20-byte big-endian records: (ms-since-hour-start:
int32, ask*point: int32, bid*point: int32, ask-volume: float32,
bid-volume: float32). `point` is symbol-specific (100000 for most 5-digit
FX pairs, 1000 for 3-digit JPY pairs, 100 for metals like XAUUSD) -- see
_POINT_VALUES below.

This fetches ONE HOUR PER HTTP REQUEST, which is the honest cost of "no
paid tier": a full year of hourly bars is ~8,760 requests. Fine for
filling in a few days/weeks of recent history to complement your
existing static datasets under data/raw; impractical as your only source
for a multi-year backtest window -- for that, Dukascopy also has zipped
historical data downloads on its own site you'd import as CSV via the
existing app.data.importer path instead of hitting this API thousands of
times.

Symbol coverage: the _SYMBOL_MAP below only lists the FX majors/minors,
metals, and a handful of index CFDs Dukascopy is well known for. Futures
and many CFD symbols use Dukascopy-specific codes not included here --
look up the exact code from Dukascopy's own historical data page
(https://www.dukascopy.com/swiss/english/marketwatch/historical/) before
assuming a symbol isn't available just because it's not in this map.

HONESTY NOTE: written against Dukascopy's long-stable, widely
reverse-engineered public tick-data format (the same one used by several
open-source tools), not exercised end-to-end against the live endpoint by
anyone on this project. If a fetch call raises DukascopyFetchError with an
HTTP or decompression error, the most likely cause is a symbol code typo
or a date this specific instrument didn't trade -- not a broken
implementation, but confirm by trying a well-known symbol/date (e.g.
EURUSD, a recent weekday) before assuming otherwise.
"""
from __future__ import annotations

import lzma
import struct
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

_BASE_URL = "https://datafeed.dukascopy.com/datafeed"
_TICK_RECORD = struct.Struct(">iiiff")  # ms_offset, ask*point, bid*point, ask_vol, bid_vol

# symbol -> (dukascopy path code, point divisor)
_SYMBOL_MAP: dict[str, tuple[str, float]] = {
    "EURUSD": ("EURUSD", 100000.0), "GBPUSD": ("GBPUSD", 100000.0), "USDJPY": ("USDJPY", 1000.0),
    "USDCHF": ("USDCHF", 100000.0), "USDCAD": ("USDCAD", 100000.0), "AUDUSD": ("AUDUSD", 100000.0),
    "NZDUSD": ("NZDUSD", 100000.0), "EURGBP": ("EURGBP", 100000.0), "EURJPY": ("EURJPY", 1000.0),
    "GBPJPY": ("GBPJPY", 1000.0), "EURCHF": ("EURCHF", 100000.0), "AUDJPY": ("AUDJPY", 1000.0),
    "XAUUSD": ("XAUUSD", 1000.0), "XAGUSD": ("XAGUSD", 1000.0),
    "USA30IDXUSD": ("USA30IDXUSD", 100.0),   # Dow Jones CFD
    "USA500IDXUSD": ("USA500IDXUSD", 100.0),  # S&P 500 CFD
    "USATECHIDXUSD": ("USATECHIDXUSD", 100.0),  # Nasdaq 100 CFD
    "DEUIDXEUR": ("DEUIDXEUR", 100.0),  # DAX CFD
}

TIMEFRAME_LABELS = ["1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day"]
_RESAMPLE_RULE = {"1Min": "1min", "5Min": "5min", "15Min": "15min", "30Min": "30min",
                   "1Hour": "1h", "4Hour": "4h", "1Day": "1D"}


class DukascopyFetchError(RuntimeError):
    """Raised for bad symbol/date input, or when Dukascopy's endpoint
    returns something this fetcher doesn't recognize."""


def known_symbols() -> list[str]:
    return sorted(_SYMBOL_MAP)


def _hour_url(dukascopy_code: str, dt: datetime) -> str:
    return f"{_BASE_URL}/{dukascopy_code}/{dt.year:04d}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"


def _fetch_hour_ticks(dukascopy_code: str, point: float, dt: datetime, session: requests.Session) -> pd.DataFrame:
    url = _hour_url(dukascopy_code, dt)
    try:
        resp = session.get(url, timeout=20)
    except Exception as exc:
        raise DukascopyFetchError(f"Network error fetching {url}: {exc}") from exc
    if resp.status_code == 404 or not resp.content:
        return pd.DataFrame(columns=["timestamp", "bid", "ask"])  # no trading that hour (weekend/holiday) -- not an error
    if resp.status_code != 200:
        raise DukascopyFetchError(f"Dukascopy returned HTTP {resp.status_code} for {url}.")
    try:
        raw = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(resp.content)
    except lzma.LZMAError as exc:
        raise DukascopyFetchError(
            f"Could not decompress tick data for {dukascopy_code} at {dt.isoformat()} -- "
            f"the .bi5 format may have changed, or this wasn't valid tick data: {exc}"
        ) from exc

    n_records = len(raw) // _TICK_RECORD.size
    rows = []
    hour_start = dt.replace(minute=0, second=0, microsecond=0)
    for i in range(n_records):
        ms_offset, ask_raw, bid_raw, _ask_vol, _bid_vol = _TICK_RECORD.unpack_from(raw, i * _TICK_RECORD.size)
        rows.append((hour_start + timedelta(milliseconds=ms_offset), bid_raw / point, ask_raw / point))
    return pd.DataFrame(rows, columns=["timestamp", "bid", "ask"])


def fetch_ohlcv(
    symbol: str, timeframe_label: str, start: str, end: str,
    max_hours: int = 24 * 14, polite_delay_seconds: float = 0.05,
) -> pd.DataFrame:
    """Fetches tick data hour-by-hour across [start, end) and resamples
    it into OHLCV bars in this app's standard schema (timestamp, open,
    high, low, close, volume), using the tick mid-price. `volume` is
    always 0 -- Dukascopy's per-tick ask/bid volume fields exist but
    aggregating them into a meaningful per-bar volume figure is not
    attempted here; treat volume as informational-only for this source,
    unlike Alpaca's real trade volume.

    Raises DukascopyFetchError immediately (before fetching anything) if
    the requested range exceeds `max_hours` -- this is a deliberate,
    adjustable guardrail against accidentally queuing thousands of
    sequential HTTP requests against a free public endpoint; widen it
    explicitly if you really want a long backfill; but consider
    Dukascopy's own bulk historical-data download for anything beyond a
    few weeks (see module docstring).
    """
    if symbol.upper() not in _SYMBOL_MAP:
        raise DukascopyFetchError(
            f"'{symbol}' isn't in this fetcher's known symbol map ({', '.join(known_symbols())}). "
            "Look up the exact Dukascopy instrument code from their historical-data page and add "
            "it to _SYMBOL_MAP in app/data/dukascopy_source.py, or import it as a CSV instead."
        )
    if timeframe_label not in _RESAMPLE_RULE:
        raise DukascopyFetchError(f"Unknown timeframe '{timeframe_label}'. Choose one of: {TIMEFRAME_LABELS}.")

    dukascopy_code, point = _SYMBOL_MAP[symbol.upper()]
    try:
        start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime().replace(minute=0, second=0, microsecond=0)
        end_dt = pd.Timestamp(end, tz="UTC").to_pydatetime()
    except Exception as exc:
        raise DukascopyFetchError(f"Could not parse start/end date: {exc}") from exc

    total_hours = int((end_dt - start_dt).total_seconds() // 3600) + 1
    if total_hours <= 0:
        raise DukascopyFetchError("`end` must be after `start`.")
    if total_hours > max_hours:
        raise DukascopyFetchError(
            f"Requested range is {total_hours} hours, over the {max_hours}-hour guardrail "
            "(~one HTTP request per hour against a free public endpoint -- see this function's "
            "docstring). Narrow the date range, or pass a larger max_hours if you really mean it."
        )

    session = requests.Session()
    frames: list[pd.DataFrame] = []
    cursor = start_dt
    while cursor <= end_dt:
        frames.append(_fetch_hour_ticks(dukascopy_code, point, cursor, session))
        cursor += timedelta(hours=1)
        if polite_delay_seconds:
            time.sleep(polite_delay_seconds)

    ticks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "bid", "ask"])
    if ticks.empty:
        raise DukascopyFetchError(
            f"No ticks returned for {symbol} between {start} and {end} -- check the date range "
            "covers a period the market was actually open (Dukascopy has no weekend/holiday data)."
        )

    ticks["mid"] = (ticks["bid"] + ticks["ask"]) / 2.0
    ticks = ticks.set_index(pd.DatetimeIndex(ticks["timestamp"]))
    ohlc = ticks["mid"].resample(_RESAMPLE_RULE[timeframe_label]).ohlc().dropna()
    out = ohlc.reset_index().rename(columns={"index": "timestamp"})
    out["volume"] = 0.0
    out = out[["timestamp", "open", "high", "low", "close", "volume"]]
    if out.empty:
        raise DukascopyFetchError(f"Ticks were returned for {symbol} but resampling produced no complete bars.")
    return out


def save_bars_as_csv(df: pd.DataFrame, symbol: str, timeframe_label: str):
    """Writes fetched bars into data/raw/<SYMBOL>/, mirroring
    app.data.alpaca_source.save_bars_as_csv exactly (same folder
    convention, so it shows up for free in the existing dataset
    list/grouping) with a "_dukascopy" filename suffix instead of
    "_alpaca" so the two sources' files never collide for the same
    symbol/timeframe."""
    from app.data.storage import get_raw_data_dir

    safe_symbol = "".join(c for c in symbol.upper() if c.isalnum() or c in ("-", "_")) or "SYMBOL"
    folder = get_raw_data_dir() / safe_symbol
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{safe_symbol}_{timeframe_label}_dukascopy.csv"
    df.to_csv(dest, index=False)
    return dest
