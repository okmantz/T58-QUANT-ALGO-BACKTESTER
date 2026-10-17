"""One front door for every market-data provider that can be reached with a saved API key.

The Market-data dropdown on each test page lists the files in ``data/raw`` and,
under "Download from a data provider", every provider that has a key saved on
Settings > API Keys (plus Dukascopy, which needs none). Picking one asks only
for instrument / timeframe / dates; ``fetch_to_library`` downloads the bars,
stores them in ``data/raw/<SYMBOL>/`` like any imported file, and the page then
selects that new file -- so every downstream feature keeps working on plain
files and nothing else needs to know an API was involved.

Fetchers take an injectable ``http_get`` so they are unit-tested without a
network. Alpaca, London Strategic Edge, Twelve Data, Polygon and Dukascopy reuse
the existing modules; Tiingo, Alpha Vantage and Databento are implemented to
each provider's documented REST schema and mock-tested only.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from app.data import free_sources
from app.data.free_sources import DataSourceError, HttpGet

STANDARD_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    hint: str                     # placeholder for the instrument box
    needs_key: bool = True
    key_field: str = ""           # attribute on ApiKeysSettings (or "alpaca")
    timeframes: tuple[str, ...] = tuple(STANDARD_TIMEFRAMES)


PROVIDERS: dict[str, Provider] = {p.id: p for p in [
    Provider("alpaca", "Alpaca", "AAPL or BTC/USD", key_field="alpaca",
             timeframes=("1m", "5m", "15m", "30m", "1h", "4h", "1d")),
    Provider("lse", "London Strategic Edge", "EUR/USD, GC, AAPL", key_field="london_strategic_edge_key"),
    Provider("twelve_data", "Twelve Data", "EUR/USD, AAPL", key_field="twelve_data_api_key"),
    Provider("polygon", "Polygon", "AAPL, C:EURUSD, X:BTCUSD", key_field="polygon_api_key"),
    Provider("tiingo", "Tiingo", "AAPL, eurusd, btcusd", key_field="tiingo_api_key"),
    Provider("alpha_vantage", "Alpha Vantage", "IBM, EURUSD", key_field="alpha_vantage_api_key",
             timeframes=("1m", "5m", "15m", "30m", "1h", "1d")),
    Provider("databento", "Databento", "ES.c.0, NQ.c.0", key_field="databento_api_key"),
    Provider("dukascopy", "Dukascopy (no key)", "EURUSD, XAUUSD", needs_key=False,
             timeframes=("1m", "5m", "15m", "30m", "1h", "4h", "1d")),
]}


# ----------------------------------------------------------------- keys

def _key_for(provider: Provider) -> str | tuple[str, str]:
    if provider.key_field == "alpaca":
        from app.data import alpaca_credentials
        creds = alpaca_credentials.load_credentials()
        return (creds.api_key, creds.secret_key) if creds and creds.is_usable() else ""
    if not provider.key_field:
        return "free"
    from app.accounts.api_keys import load_settings
    return getattr(load_settings(), provider.key_field, "") or ""


def provider_list() -> list[dict]:
    """For the dropdown: only providers that are usable right now, plus those
    missing a key (flagged so the UI can point at Settings > API Keys)."""
    out = []
    for p in PROVIDERS.values():
        ready = (not p.needs_key) or bool(_key_for(p))
        out.append({"id": p.id, "label": p.label, "hint": p.hint, "ready": ready,
                    "timeframes": list(p.timeframes)})
    return out


# ----------------------------------------------------------------- helpers

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df[_COLS].copy()
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    for c in _COLS[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["volume"] = out["volume"].fillna(0.0)
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    return out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    r = df.set_index("timestamp").resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
    return r.reset_index()


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end) + pd.Timedelta(days=1)]
    return df.reset_index(drop=True)


def _default_dates(start: str, end: str, tf: str) -> tuple[str, str]:
    end = end or date.today().isoformat()
    if not start:
        days = 30 if _MINUTES[tf] < 60 else (365 if tf != "1d" else 365 * 5)
        start = (date.fromisoformat(end) - timedelta(days=days)).isoformat()
    return start, end


# ----------------------------------------------------------------- per-provider fetchers

def _fetch_tiingo(symbol: str, tf: str, start: str, end: str, key: str, http_get: HttpGet) -> pd.DataFrame:
    if tf == "1d":
        data = http_get(f"https://api.tiingo.com/tiingo/daily/{symbol}/prices",
                        {"startDate": start, "endDate": end, "token": key})
        rows = [{"timestamp": r["date"], "open": r["open"], "high": r["high"], "low": r["low"],
                 "close": r["close"], "volume": r.get("volume", 0)} for r in (data if isinstance(data, list) else [])]
    else:
        freq = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1hour", "4h": "4hour"}[tf]
        data = http_get(f"https://api.tiingo.com/iex/{symbol}/prices",
                        {"startDate": start, "endDate": end, "resampleFreq": freq, "token": key, "columns": "open,high,low,close,volume"})
        rows = [{"timestamp": r["date"], "open": r["open"], "high": r["high"], "low": r["low"],
                 "close": r["close"], "volume": r.get("volume", 0)} for r in (data if isinstance(data, list) else [])]
    if not rows:
        raise DataSourceError(f"Tiingo returned no bars for {symbol}: {str(data)[:120]}")
    return pd.DataFrame(rows)


def _fetch_alpha_vantage(symbol: str, tf: str, start: str, end: str, key: str, http_get: HttpGet) -> pd.DataFrame:
    if tf == "1d":
        data = http_get("https://www.alphavantage.co/query",
                        {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "full", "apikey": key})
        series = data.get("Time Series (Daily)")
    else:
        interval = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "60min"}[tf]
        data = http_get("https://www.alphavantage.co/query",
                        {"function": "TIME_SERIES_INTRADAY", "symbol": symbol, "interval": interval,
                         "outputsize": "full", "apikey": key})
        series = data.get(f"Time Series ({interval})")
    if not series:
        raise DataSourceError(f"Alpha Vantage returned no bars: {data.get('Note') or data.get('Information') or data.get('Error Message') or str(data)[:120]}")
    rows = [{"timestamp": ts, "open": v["1. open"], "high": v["2. high"], "low": v["3. low"],
             "close": v["4. close"], "volume": v.get("5. volume", 0)} for ts, v in series.items()]
    return pd.DataFrame(rows)


def _fetch_databento(symbol: str, tf: str, start: str, end: str, key: str, http_get: Callable | None) -> pd.DataFrame:
    """Databento Historical HTTP API (CSV). Futures default dataset GLBX.MDP3; a
    ``DATASET:SYMBOL`` instrument overrides it (e.g. ``XNAS.ITCH:AAPL``)."""
    dataset, sym = ("GLBX.MDP3", symbol)
    if ":" in symbol:
        dataset, sym = symbol.split(":", 1)
    schema = {"1m": "ohlcv-1m", "5m": "ohlcv-1m", "15m": "ohlcv-1m", "30m": "ohlcv-1m",
              "1h": "ohlcv-1h", "4h": "ohlcv-1h", "1d": "ohlcv-1d"}[tf]
    params = {"dataset": dataset, "symbols": sym, "schema": schema, "start": start, "end": end,
              "encoding": "csv", "pretty_px": "true", "pretty_ts": "true",
              "stype_in": "continuous" if ".c." in sym or ".v." in sym or ".n." in sym else "raw_symbol"}
    if http_get is not None:
        text = http_get("https://hist.databento.com/v0/timeseries.get_range", {**params, "_key": key})
        text = text if isinstance(text, str) else text.get("csv", "")
    else:
        import requests
        resp = requests.post("https://hist.databento.com/v0/timeseries.get_range", data=params, auth=(key, ""), timeout=120)
        if resp.status_code != 200:
            raise DataSourceError(f"Databento HTTP {resp.status_code}: {resp.text[:160]}")
        text = resp.text
    df = pd.read_csv(io.StringIO(text))
    if df.empty or "ts_event" not in df:
        raise DataSourceError("Databento returned no bars for that instrument/range.")
    df = df.rename(columns={"ts_event": "timestamp"})
    return df


def fetch(provider_id: str, symbol: str, timeframe: str, start: str = "", end: str = "",
          http_get: HttpGet | None = None) -> pd.DataFrame:
    """Download ``symbol`` at ``timeframe`` (one of STANDARD_TIMEFRAMES) from one
    provider. Returns standard OHLCV (naive UTC timestamps)."""
    p = PROVIDERS.get(provider_id)
    if p is None:
        raise DataSourceError(f"Unknown data provider '{provider_id}'.")
    symbol = (symbol or "").strip()
    if not symbol:
        raise DataSourceError("Enter an instrument.")
    if timeframe not in STANDARD_TIMEFRAMES:
        raise DataSourceError(f"Unknown timeframe '{timeframe}'.")
    key = _key_for(p)
    if p.needs_key and not key:
        raise DataSourceError(f"No {p.label} key saved. Add it under Settings > API Keys.")
    start, end = _default_dates(start, end, timeframe)
    minutes = _MINUTES[timeframe]
    get = http_get or free_sources._default_get

    if provider_id == "alpaca":
        from app.data.alpaca_source import fetch_bars
        label = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min", "1h": "1Hour", "4h": "1Hour", "1d": "1Day"}[timeframe]
        asset = "Crypto" if "/" in symbol and symbol.upper().split("/")[1] in ("USD", "USDT", "USDC", "BTC", "ETH") else "Stock"
        df = fetch_bars(key[0], key[1], symbol, asset, label, start, end)
        df = df.rename(columns={c: c.lower() for c in df.columns})
        if "timestamp" not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
        df = _clean(df)
        return _resample(df, 240) if timeframe == "4h" else df
    if provider_id == "lse":
        from app.data.london_strategic_edge_source import fetch_candles
        return _clean(fetch_candles(key, symbol, timeframe, start, end))
    if provider_id == "twelve_data":
        interval = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1day"}[timeframe]
        return _slice(_clean(free_sources.fetch_twelve_data_ohlcv(symbol, interval, key, outputsize=5000, http_get=http_get)), start, end)
    if provider_id == "polygon":
        mult, span = {"1m": (1, "minute"), "5m": (5, "minute"), "15m": (15, "minute"), "30m": (30, "minute"),
                      "1h": (1, "hour"), "4h": (4, "hour"), "1d": (1, "day")}[timeframe]
        return _clean(free_sources.fetch_polygon_ohlcv(symbol, mult, span, start, end, key, http_get=http_get))
    if provider_id == "tiingo":
        return _slice(_clean(_fetch_tiingo(symbol, timeframe, start, end, key, get)), start, end)
    if provider_id == "alpha_vantage":
        return _slice(_clean(_fetch_alpha_vantage(symbol, timeframe, start, end, key, get)), start, end)
    if provider_id == "databento":
        df = _clean(_fetch_databento(symbol, "1h" if timeframe in ("1h", "4h") else ("1d" if timeframe == "1d" else "1m"),
                                     start, end, key, http_get))
        if timeframe in ("5m", "15m", "30m", "4h"):
            df = _resample(df, minutes)
        return df
    if provider_id == "dukascopy":
        from app.data.dukascopy_source import fetch_ohlcv
        label = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min", "1h": "1Hour", "4h": "4Hour", "1d": "1Day"}[timeframe]
        return _clean(fetch_ohlcv(symbol.replace("/", ""), label, start, end))
    raise DataSourceError(f"Provider '{provider_id}' is not implemented.")


def fetch_to_library(provider_id: str, symbol: str, timeframe: str, start: str = "", end: str = "",
                     http_get: HttpGet | None = None) -> dict:
    """Fetch and save into data/raw/<SYMBOL>/. Returns {"full_name", "rows", "first", "last"}
    where full_name is the value the Market-data dropdown uses for that file."""
    from app.data.storage import get_raw_data_dir

    df = fetch(provider_id, symbol, timeframe, start, end, http_get)
    if df.empty:
        raise DataSourceError("The provider returned no bars for that instrument and date range.")
    safe = "".join(c for c in symbol.upper() if c.isalnum() or c in ("-", "_")) or "SYMBOL"
    folder = get_raw_data_dir() / safe
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{safe}_{timeframe}_{provider_id}.csv"
    df.to_csv(dest, index=False)
    return {"full_name": dest.relative_to(get_raw_data_dir()).as_posix(), "rows": int(len(df)),
            "first": str(df["timestamp"].iloc[0]), "last": str(df["timestamp"].iloc[-1]),
            "name": dest.name, "instrument": safe}
