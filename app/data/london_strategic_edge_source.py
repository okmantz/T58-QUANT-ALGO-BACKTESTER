"""Fetch OHLCV market data from London Strategic Edge (LSE) for use as
backtest input.

Mirrors app.data.alpaca_source's pattern exactly (same normalize-to-
standard-schema, save-as-csv, fetch_bars-style dispatch shape), so the two
data sources are interchangeable from the UI's point of view. Unlike
Alpaca, London Strategic Edge is a SINGLE-API-key service covering stocks,
forex, crypto, commodities, indices, ETFs and futures all through one
account (see https://londonstrategicedge.com/free-market-data-api/ and
https://github.com/londonstrategicedge/lse-data) -- there is no separate
secret key, no asset-class-specific client class, and no feed/adjustment
concept the way Alpaca has for equities.

Requires the optional `lse-data` package (see config/requirements.txt --
`pip install lse-data`). The import is deferred into each function so the
rest of the app still works if it isn't installed; callers should catch
LSEImportError and show it as a plain message rather than a stack trace.
This is the actual integration behind the API key app.accounts.api_keys
already had secure storage for -- before this module existed, that key
was stored but nothing in the app ever used it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.storage import get_raw_data_dir

# (label, LSE timeframe string) -- label is what the UI shows/stores; LSE's
# own timeframe strings (see the lse-data README) are already short and
# readable, so the label and the wire value are identical here, unlike
# Alpaca's TimeFrame(amount, unit) object construction.
TIMEFRAME_CHOICES = [
    "1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1mo",
]

# LSE covers every one of these asset classes through the SAME client and
# SAME candles() call -- unlike Alpaca, there is no separate method per
# asset class, so this list is informational (populates a UI dropdown of
# example instrument formats) rather than something fetch_candles branches
# on. Matches client.catalog()'s own category names from the lse-data
# README (options/futures/economics/bonds/etc. are catalog-only categories
# without OHLCV candles and are intentionally not offered here).
ASSET_CLASSES = ["Stock", "Forex", "Crypto", "Commodity", "Index", "ETF", "Futures"]


class LSEImportError(RuntimeError):
    """Raised when the optional lse-data dependency isn't installed."""


class LSEFetchError(RuntimeError):
    """Raised when a request to London Strategic Edge fails (bad key, bad
    symbol/timeframe, no data returned for the range, network error,
    etc.)."""


def _require_lse() -> None:
    try:
        import lse  # noqa: F401
    except ImportError as exc:
        raise LSEImportError(
            "The 'lse-data' package isn't installed. Run `pip install lse-data` "
            "(or rebuild the .exe with it bundled) to enable fetching data from "
            "London Strategic Edge."
        ) from exc


def _client(api_key: str):
    _require_lse()
    from lse import LSE

    if not api_key:
        raise LSEFetchError("Enter a London Strategic Edge API key first.")
    return LSE(api_key=api_key)


def test_connection(api_key: str) -> str:
    """Verifies the key works with a small, cheap, real call -- the
    catalog listing (client.catalog()) is a single authenticated read with
    no vault-export allowance cost, unlike candles()/history(), so this
    never eats into the hourly export budget just to check the key is
    valid. Raises LSEImportError/LSEFetchError on failure; returns a
    short human-readable success message otherwise."""
    client = _client(api_key)
    from lse import LSEError

    try:
        rows = client.catalog("forex")
    except LSEError as exc:
        raise LSEFetchError(f"London Strategic Edge rejected the key: {exc.message} (status {exc.status}).") from exc
    except Exception as exc:  # noqa: BLE001 -- network errors, etc.
        raise LSEFetchError(f"Could not connect to London Strategic Edge: {exc}") from exc
    count = len(rows) if rows is not None else 0
    return f"Connected. Key is valid ({count} forex instruments visible in the catalog)."


def _normalize_candles(raw, symbol: str) -> pd.DataFrame:
    """Normalizes whatever candles()/history() handed back -- a list of
    dicts (candles()'s documented return shape) or a DataFrame
    (history()'s shape, or candles() too if the optional `[frames]` extra
    is installed) -- into this app's standard OHLCV schema (timestamp,
    open, high, low, close, volume), same convention
    app.data.alpaca_source._normalize_bars uses. Field names are matched
    case-insensitively and against the couple of aliases a candle/tick
    payload could plausibly use (t/time/timestamp, o/open, ...), since the
    exact on-the-wire key casing isn't pinned down by the client's public
    docs beyond \"OHLCV candles\"."""
    if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
        raise LSEFetchError(
            f"London Strategic Edge returned no candles for {symbol} in that date range/timeframe -- "
            "try a wider date range, a coarser timeframe, or double-check the symbol against catalog()."
        )
    df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(list(raw))
    if df.empty:
        raise LSEFetchError(f"London Strategic Edge returned no candles for {symbol} in that date range.")

    def _find(*aliases: str) -> str | None:
        lower_cols = {c.lower(): c for c in df.columns}
        for alias in aliases:
            if alias in lower_cols:
                return lower_cols[alias]
        return None

    col_map = {
        "timestamp": _find("timestamp", "time", "t", "date", "datetime"),
        "open": _find("open", "o"),
        "high": _find("high", "h"),
        "low": _find("low", "l"),
        "close": _find("close", "c"),
        "volume": _find("volume", "v", "vol"),
    }
    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        raise LSEFetchError(
            f"Unexpected response shape from London Strategic Edge for {symbol} "
            f"(missing columns: {missing}; got: {list(df.columns)})."
        )
    out = pd.DataFrame({std: df[actual] for std, actual in col_map.items()})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    if out.empty:
        raise LSEFetchError(f"London Strategic Edge returned no usable candles for {symbol} in that date range.")
    return out


def fetch_candles(
    api_key: str,
    symbol: str,
    timeframe_label: str,
    start: str,
    end: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Fetches historical OHLCV candles for one instrument (any asset
    class LSE covers -- 'AAPL', 'EUR/USD', 'BTC/USD', 'GC' futures code,
    etc.). start/end are 'YYYY-MM-DD' (or full ISO 8601) strings. A single
    candles() call returns one page (LSE's own pagination, per the
    lse-data README); for a deep multi-year pull beyond one page,
    fetch_history() below runs LSE's vault export job instead."""
    client = _client(api_key)
    from lse import LSEError

    if timeframe_label not in TIMEFRAME_CHOICES:
        raise LSEFetchError(f"Unknown timeframe '{timeframe_label}'. Choose one of: {', '.join(TIMEFRAME_CHOICES)}.")
    kwargs = {"start": start, "end": end}
    if limit:
        kwargs["limit"] = limit
    try:
        raw = client.candles(symbol, timeframe_label, **kwargs)
    except LSEError as exc:
        raise LSEFetchError(f"London Strategic Edge request failed for {symbol}: {exc.message} (status {exc.status}).") from exc
    except Exception as exc:  # noqa: BLE001
        raise LSEFetchError(f"London Strategic Edge request failed for {symbol}: {exc}") from exc
    return _normalize_candles(raw, symbol)


def fetch_history(api_key: str, symbol: str, timeframe_label: str, start: str) -> pd.DataFrame:
    """Deep multi-year pull via LSE's vault export job (history()), for
    ranges too large for one paginated candles() call. Counts against the
    account's hourly export budget (GET /vault/usage) -- prefer
    fetch_candles() above for anything that fits in one page."""
    client = _client(api_key)
    from lse import LSEError

    if timeframe_label not in TIMEFRAME_CHOICES:
        raise LSEFetchError(f"Unknown timeframe '{timeframe_label}'. Choose one of: {', '.join(TIMEFRAME_CHOICES)}.")
    try:
        raw = client.history(symbol, timeframe=timeframe_label, start=start)
    except LSEError as exc:
        raise LSEFetchError(f"London Strategic Edge history export failed for {symbol}: {exc.message} (status {exc.status}).") from exc
    except Exception as exc:  # noqa: BLE001
        raise LSEFetchError(f"London Strategic Edge history export failed for {symbol}: {exc}") from exc
    return _normalize_candles(raw, symbol)


def save_bars_as_csv(df: pd.DataFrame, symbol: str, timeframe_label: str) -> Path:
    """Writes fetched candles into data/raw/<SYMBOL>/, alongside any
    manually imported CSVs or Alpaca-fetched bars for the same instrument
    -- same convention as app.data.alpaca_source.save_bars_as_csv, so it
    shows up for free in the existing Market Data Library grouping with
    no changes needed to that selection UI."""
    safe_symbol = "".join(c for c in symbol.upper() if c.isalnum() or c in ("-", "_")) or "SYMBOL"
    folder = get_raw_data_dir() / safe_symbol
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{safe_symbol}_{timeframe_label}_lse.csv"
    df.to_csv(dest, index=False)
    return dest
