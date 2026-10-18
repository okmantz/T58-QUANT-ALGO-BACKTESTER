"""Alternative market-data providers (AUDIT 2026-10-12).

Keys are stored on the Settings > API Keys page (app.accounts.api_keys) and
read here. Every fetcher takes an injectable ``http_get(url, params) -> dict``
so it is unit-tested without the network. NOT verified against the live
services -- the audit sandbox only allows package registries. Treat each as
"implemented to the provider's documented schema, mock-tested".

Returned OHLCV frames use the repo's standard columns:
timestamp (naive UTC), open, high, low, close, volume.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

HttpGet = Callable[[str, dict], dict]


class DataSourceError(RuntimeError):
    pass


def _default_get(url: str, params: dict) -> dict:
    import requests

    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise DataSourceError(f"HTTP {resp.status_code} from {url.split('?')[0]}")
    return resp.json()


def fetch_fred_series(series_id: str, api_key: str, http_get: HttpGet | None = None) -> pd.Series:
    """Daily FRED series as a float Series indexed by date (missing '.' dropped).
    E.g. VIXCLS (CBOE VIX close) for a volatility-regime gate."""
    if not api_key:
        raise DataSourceError("No FRED API key saved (Settings > API Keys).")
    data = (http_get or _default_get)(
        "https://api.stlouisfed.org/fred/series/observations",
        {"series_id": series_id, "api_key": api_key, "file_type": "json"},
    )
    obs = data.get("observations")
    if obs is None:
        raise DataSourceError(f"FRED returned no observations: {str(data)[:120]}")
    s = pd.Series(
        {pd.Timestamp(o["date"]): pd.to_numeric(o["value"], errors="coerce") for o in obs},
        dtype=float,
    ).dropna()
    s.index.name = "date"
    return s.sort_index()


def fetch_twelve_data_ohlcv(symbol: str, interval: str, api_key: str,
                            outputsize: int = 5000, http_get: HttpGet | None = None,
                            start: str = "", end: str = "") -> pd.DataFrame:
    """Twelve Data /time_series (interval like '1min','5min','1h','1day'). Requested in UTC. ``start``/``end``
    (YYYY-MM-DD) ask the service for that window directly instead of "the newest N bars", so an older range works."""
    if not api_key:
        raise DataSourceError("No Twelve Data API key saved (Settings > API Keys).")
    params = {"symbol": symbol, "interval": interval, "apikey": api_key, "outputsize": outputsize,
              "timezone": "UTC", "order": "ASC", "format": "JSON"}
    if start:
        params["start_date"] = start
    if end:
        params["end_date"] = end
    data = (http_get or _default_get)("https://api.twelvedata.com/time_series", params)
    if data.get("status") == "error" or "values" not in data:
        raise DataSourceError(f"Twelve Data error: {data.get('message', str(data)[:120])}")
    df = pd.DataFrame(data["values"])
    out = pd.DataFrame({"timestamp": pd.to_datetime(df["datetime"])})
    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0) if "volume" in df else 0.0
    out = out.dropna(subset=["open", "high", "low", "close"]).sort_values("timestamp")
    return out.drop_duplicates("timestamp").reset_index(drop=True)


def fetch_polygon_ohlcv(ticker: str, multiplier: int, timespan: str, start: str, end: str, api_key: str,
                        http_get: HttpGet | None = None) -> pd.DataFrame:
    """Polygon aggregates (/v2/aggs). Timestamps are bar-OPEN, ms since epoch UTC."""
    if not api_key:
        raise DataSourceError("No Polygon API key saved (Settings > API Keys).")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    data = (http_get or _default_get)(url, {"apiKey": api_key, "adjusted": "false", "sort": "asc", "limit": 50000})
    res = data.get("results")
    if not res:
        raise DataSourceError(f"Polygon returned no bars: {data.get('error') or data.get('status')}")
    df = pd.DataFrame(res)
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["t"], unit="ms"),
        "open": df["o"].astype(float), "high": df["h"].astype(float),
        "low": df["l"].astype(float), "close": df["c"].astype(float),
        "volume": df["v"].astype(float) if "v" in df else 0.0,
    })
    return out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
