"""Daily price history + quotes for stocks, indices (^GSPC) and crypto (BTC-USD). Yahoo chart endpoint, Stooq fallback; injectable for tests."""
from __future__ import annotations

import time
from typing import Callable

import pandas as pd

_cache: dict = {}
_TTL = 600


def _yahoo(symbol: str, range_: str) -> pd.DataFrame:
    import requests
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", params={"range": range_, "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose") or q["close"]
    idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": adj, "volume": q["volume"]}, index=idx)
    return df[~df.index.duplicated(keep="last")].dropna(subset=["close"])


def _stooq(symbol: str, range_: str) -> pd.DataFrame:
    import io
    import requests
    s = symbol.lower()
    s = s if ("." in s or s.startswith("^")) else s + ".us"
    r = requests.get(f"https://stooq.com/q/d/l/?s={s}&i=d", timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
    df.columns = [c.lower() for c in df.columns]
    return df.dropna(subset=["close"])


def history(symbol: str, range_: str = "5y", provider: Callable[[str, str], pd.DataFrame] | None = None) -> pd.DataFrame:
    key = (symbol.upper(), range_)
    hit = _cache.get(key)
    if provider is None and hit and time.time() - hit[0] < _TTL:
        return hit[1]
    errs = []
    for fn in ([provider] if provider else [_yahoo, _stooq]):
        try:
            df = fn(symbol, range_)
            if len(df):
                if provider is None:
                    _cache[key] = (time.time(), df)
                return df
        except Exception as e:  # noqa: BLE001
            errs.append(str(e))
    raise RuntimeError(f"No price data for {symbol}: {'; '.join(errs) or 'empty'}")


def quote(symbol: str, provider=None) -> dict:
    df = history(symbol, "1y", provider)
    c = df["close"]
    last = float(c.iloc[-1])
    prev = float(c.iloc[-2]) if len(c) > 1 else last
    hi, lo = float(c.tail(252).max()), float(c.tail(252).min())
    return {"symbol": symbol.upper(), "price": last, "change": last - prev, "change_pct": (last / prev - 1) * 100 if prev else 0.0,
            "high_52w": hi, "low_52w": lo, "as_of": str(df.index[-1].date()), "spark": [round(float(x), 4) for x in c.tail(60)]}
