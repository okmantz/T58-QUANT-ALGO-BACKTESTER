"""Free macro/regime daily series: VIX, VIX3M (CBOE), rates and dollar (FRED), CFTC COT, plus a snapshot recorder.

Report 2026-10-20 (section 5, "different data"). Everything here is DAILY and enters a backtest only through
``prior_day_value`` so a bar never sees the same day's close (no lookahead). Fetchers need network access and are meant to be
run on the user's machine (``python -m scripts.update_macro_data``); the loaders read the cached CSVs under ``data/macro``
and return an empty series when a file is absent, so a backtest never crashes on missing macro data.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

MACRO_DIR = Path(__file__).resolve().parents[2] / "data" / "macro"
CBOE_URLS = {
    "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
}
FRED_SERIES = {"DGS10": "10y Treasury yield", "DGS2": "2y Treasury yield", "DTWEXBGS": "Broad trade-weighted dollar index"}


def _path(name: str) -> Path:
    return MACRO_DIR / f"{name.lower()}.csv"


def save_series(name: str, s: pd.Series) -> Path:
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    out = s.dropna().rename("value").rename_axis("date").reset_index()
    out.to_csv(_path(name), index=False)
    return _path(name)


def load_series(name: str, macro_dir: str | Path | None = None) -> pd.Series:
    """Daily series indexed by normalized date; empty (float) series when the cache file is missing."""
    p = (Path(macro_dir) if macro_dir else MACRO_DIR) / f"{name.lower()}.csv"
    if not p.exists():
        return pd.Series(dtype=float, name=name)
    try:
        df = pd.read_csv(p)
        s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(), index=pd.to_datetime(df["date"]).dt.normalize(), name=name)
        return s[~s.index.duplicated(keep="last")].sort_index().dropna()
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float, name=name)


def fetch_cboe(name: str = "VIX", timeout: int = 30) -> pd.Series:
    import requests
    r = requests.get(CBOE_URLS[name], timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    dc = next(c for c in df.columns if c.upper() == "DATE")
    cc = next(c for c in df.columns if c.upper() in ("CLOSE", name.upper()))
    return pd.Series(pd.to_numeric(df[cc], errors="coerce").to_numpy(), index=pd.to_datetime(df[dc]).dt.normalize(), name=name)


def fetch_fred(series_id: str, timeout: int = 30) -> pd.Series:
    """Public FRED CSV (no key): https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES"""
    import requests
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id}, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    return pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(), index=pd.to_datetime(df["date"]).dt.normalize(), name=series_id)


def fetch_cftc_cot_nasdaq(year: int, timeout: int = 60) -> pd.DataFrame:
    """CFTC Traders-in-Financial-Futures (legacy futures-only) annual file; returns Nasdaq-100 mini net leveraged-fund position.
    Released Fridays for the prior Tuesday, so ``available_date`` = report_date + 3 days (when it became public)."""
    import zipfile
    import requests
    r = requests.get(f"https://www.cftc.gov/files/dea/history/fut_fin_xls_{year}.zip", timeout=timeout)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        df = pd.read_excel(z.open(z.namelist()[0]))
    name_col = next(c for c in df.columns if "Market_and_Exchange_Names" in str(c))
    d = df[df[name_col].astype(str).str.contains("NASDAQ-100 Consolidated|NASDAQ MINI", case=False, regex=True)].copy()
    date_col = next(c for c in df.columns if "Report_Date_as_YYYY-MM-DD" in str(c) or "As_of_Date_In_Form_YYMMDD" in str(c))
    d["report_date"] = pd.to_datetime(d[date_col].astype(str))
    lev_long = next(c for c in df.columns if str(c).startswith("Lev_Money_Positions_Long_All"))
    lev_short = next(c for c in df.columns if str(c).startswith("Lev_Money_Positions_Short_All"))
    d["net_lev"] = pd.to_numeric(d[lev_long], errors="coerce") - pd.to_numeric(d[lev_short], errors="coerce")
    d["available_date"] = d["report_date"] + pd.Timedelta(days=3)
    return d[["report_date", "available_date", "net_lev"]].sort_values("report_date").reset_index(drop=True)


def update_all(log=print) -> dict:
    """Fetch and cache everything reachable; one failure never stops the rest."""
    done: dict = {}
    for n in CBOE_URLS:
        try:
            done[n] = str(save_series(n, fetch_cboe(n)))
        except Exception as exc:  # noqa: BLE001
            log(f"{n}: {exc}")
    for sid in FRED_SERIES:
        try:
            done[sid] = str(save_series(sid, fetch_fred(sid)))
        except Exception as exc:  # noqa: BLE001
            log(f"{sid}: {exc}")
    return done


def prior_day_value(index_ts: pd.Series | pd.DatetimeIndex, daily: pd.Series, lag_days: int = 1) -> np.ndarray:
    """For each bar timestamp, the last daily value dated at least ``lag_days`` BEFORE the bar's calendar date. NaN when none."""
    if daily is None or len(daily) == 0:
        return np.full(len(index_ts), np.nan)
    ts = pd.to_datetime(pd.Series(index_ts).reset_index(drop=True))
    day = ts.dt.normalize()
    # lag_days == 1: look up (bar date - 1 day) with a backward asof, i.e. the latest value dated before the bar's own day
    left = pd.DataFrame({"d": day - pd.Timedelta(days=int(lag_days))})
    right = daily.rename("v").rename_axis("d").reset_index().sort_values("d")
    m = pd.merge_asof(left.reset_index().sort_values("d"), right, on="d", direction="backward").sort_values("index")
    return m["v"].to_numpy(dtype=float)


def record_snapshot(name: str, value: float, when: pd.Timestamp | None = None, macro_dir: str | Path | None = None) -> Path:
    """Append one dated reading to ``data/macro/<name>_snapshots.csv`` (for sources with no free history, e.g. a daily GEX print).
    Run it daily; a year later you own a year of history. Never overwrites an existing date."""
    d = Path(macro_dir) if macro_dir else MACRO_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name.lower()}_snapshots.csv"
    day = pd.Timestamp(when or pd.Timestamp.now()).normalize()
    if p.exists():
        cur = pd.read_csv(p)
        if (pd.to_datetime(cur["date"]).dt.normalize() == day).any():
            return p
        cur.loc[len(cur)] = [day.date().isoformat(), float(value)]
        cur.to_csv(p, index=False)
    else:
        pd.DataFrame({"date": [day.date().isoformat()], "value": [float(value)]}).to_csv(p, index=False)
    return p


# ======================================================================================================================
# 2026-10-25  External (Ext) feeds for the v13 indicators -- every feed carries its RELEASE-TIME LAG
# ======================================================================================================================
# A value is dated when it describes the market, but it only becomes KNOWN later (a COT report is dated Tuesday and published
# Friday afternoon; an ICE BofA spread is published the next business day). The v13 indicators read these series only through
# ``ext_aligned``, which hands a bar the latest value dated at least ``release_lag_days`` before the bar's date, so a backtest
# never trades on a number the market did not yet have. Daily feeds enter through ``prior_day_value``; intraday feeds (crypto
# funding, premium) are as-of merged on the bar timestamp and are dated at their SETTLEMENT time.
#
# Fetchers need network access and run on the user's machine (``python -m scripts.update_macro_data``); series with no free
# public history (MOVE, put/call history, economic surprise, rate expectations, futures term structure, open interest) take a
# two-column CSV (``date,value``) dropped into ``data/macro/<feed>[_<SYMBOL>].csv`` and are used exactly like the fetched ones.
from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class ExtFeed:
    name: str                 # cache file stem (lower case); per-instrument feeds append _<SYMBOL>
    kind: str                 # 'daily' | 'intraday'
    release_lag_days: int     # whole days between the value's date and the first bar allowed to see it (daily feeds)
    source: str               # where the data comes from / how to supply it
    per_symbol: bool = False  # True when the series belongs to one instrument (GEX/event-like: such families cannot pool)
    fetch_id: str = ""        # FRED series id / CBOE name when a free fetcher exists


EXT_FEEDS: dict[str, ExtFeed] = {f.name: f for f in (
    ExtFeed("hy_oas", "daily", 2, "FRED BAMLH0A0HYM2 (ICE BofA US high-yield OAS); published next business day", fetch_id="BAMLH0A0HYM2"),
    ExtFeed("real_yield_10y", "daily", 2, "FRED DFII10 (10y TIPS yield); H.15 published next business day", fetch_id="DFII10"),
    ExtFeed("breakeven_10y", "daily", 2, "FRED T10YIE (10y breakeven inflation); H.15 published next business day", fetch_id="T10YIE"),
    ExtFeed("move", "daily", 1, "ICE BofA MOVE index close; no free history -- drop a date,value CSV into data/macro/move.csv"),
    ExtFeed("vvix", "daily", 1, "CBOE VVIX_History.csv (close known at the close)", fetch_id="VVIX"),
    ExtFeed("skew", "daily", 1, "CBOE SKEW_History.csv (close known at the close)", fetch_id="SKEW"),
    ExtFeed("putcall", "daily", 1, "CBOE total put/call ratio (totalpc.csv or your own CSV)"),
    ExtFeed("econ_surprise", "daily", 1, "Economic surprise index (e.g. Citi ESI) -- drop a CSV into data/macro/econ_surprise.csv"),
    ExtFeed("fed_funds_path", "daily", 1, "Implied policy-rate path change around meetings (fed funds futures) -- CSV drop-in"),
    ExtFeed("fedfunds", "daily", 2, "FRED DFF (effective fed funds rate); used for the carry differential", fetch_id="DFF"),
    ExtFeed("cot_net_spec", "daily", 4, "CFTC COT net speculator position, one file per instrument (cot_net_spec_<SYMBOL>.csv); "
            "dated at the report date (Tuesday), published Friday -> lag 4 days", per_symbol=True),
    ExtFeed("open_interest", "daily", 1, "Futures open interest, one file per instrument (open_interest_<SYMBOL>.csv); published next morning", per_symbol=True),
    ExtFeed("term_slope", "daily", 1, "Futures term-structure slope (next/front - 1) per instrument (term_slope_<SYMBOL>.csv)", per_symbol=True),
    ExtFeed("rate_diff", "daily", 2, "Interest-rate differential (base minus quote) per FX pair (rate_diff_<SYMBOL>.csv)", per_symbol=True),
    ExtFeed("funding_rate", "intraday", 0, "Perpetual funding rate per symbol (Binance fapi), dated at settlement", per_symbol=True),
    ExtFeed("perp_premium", "intraday", 0, "Perp-minus-spot premium per symbol (Binance premiumIndexKlines), dated at bar close", per_symbol=True),
    ExtFeed("crypto_oi", "intraday", 0, "Perpetual open interest per symbol (Binance openInterestHist), dated at the sample time", per_symbol=True),
)}


def feed_path(feed: str, symbol: str | None = None, macro_dir: str | Path | None = None) -> Path:
    spec = EXT_FEEDS[feed]
    stem = f"{spec.name}_{str(symbol).upper()}" if spec.per_symbol and symbol else spec.name
    return (Path(macro_dir) if macro_dir else MACRO_DIR) / f"{stem.lower()}.csv"


def load_ext(feed: str, symbol: str | None = None, macro_dir: str | Path | None = None) -> pd.Series:
    """The cached/dropped-in series for ``feed`` (empty float series when the file is missing). Intraday feeds keep their timestamps."""
    p = feed_path(feed, symbol, macro_dir)
    if not p.exists():
        return pd.Series(dtype=float, name=feed)
    try:
        df = pd.read_csv(p)
        idx = pd.to_datetime(df["date"])
        if EXT_FEEDS[feed].kind == "daily":
            idx = idx.dt.normalize()
        s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(), index=idx, name=feed)
        return s[~s.index.duplicated(keep="last")].sort_index().dropna()
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float, name=feed)


def _frame_symbol(frame: pd.DataFrame) -> str | None:
    attrs = getattr(frame, "attrs", {}) or {}
    for k in ("symbol", "instrument", "market", "name"):
        if attrs.get(k):
            return str(attrs[k]).upper().split("_")[0].split("-")[0]
    return None


def ext_series(frame: pd.DataFrame, feed: str) -> pd.Series:
    """Resolve a feed for ``frame``: ``frame.attrs['ext'][feed]`` (tests / in-memory data) else the cached CSV."""
    from app.strategy.frame_attrs import frame_attr
    ext = frame_attr(frame, "ext")
    if isinstance(ext, dict) and feed in ext and len(ext[feed]):
        return ext[feed]
    return load_ext(feed, _frame_symbol(frame))


def ext_aligned(frame: pd.DataFrame, feed: str, series: pd.Series | None = None) -> np.ndarray:
    """Per-bar value of ``feed`` that was already PUBLIC at the bar: daily feeds via the release lag, intraday feeds via as-of on timestamp."""
    spec = EXT_FEEDS[feed]
    s = series if series is not None else ext_series(frame, feed)
    if s is None or len(s) == 0:
        return np.full(len(frame), np.nan)
    from app.strategy.indicators_v11 import _ts
    ts = _ts(frame)
    if spec.kind == "daily":
        return prior_day_value(ts, s, lag_days=max(1, int(spec.release_lag_days)))
    left = pd.DataFrame({"t": pd.to_datetime(ts).reset_index(drop=True)})
    right = s.rename("v").rename_axis("t").reset_index().sort_values("t")
    right["t"] = pd.to_datetime(right["t"])
    m = pd.merge_asof(left.reset_index().sort_values("t"), right, on="t", direction="backward").sort_values("index")
    return m["v"].to_numpy(dtype=float)


def fetch_cboe_putcall(timeout: int = 30) -> pd.Series:
    import requests
    r = requests.get("https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv", timeout=timeout)
    r.raise_for_status()
    lines = r.text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.upper().startswith("DATE"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    col = next(c for c in df.columns if "P/C" in c.upper() or "RATIO" in c.upper())
    return pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(), index=pd.to_datetime(df["DATE"]).dt.normalize(), name="putcall")


def fetch_binance_funding(symbol: str = "BTCUSDT", limit: int = 1000, timeout: int = 30) -> pd.Series:
    import requests
    r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", params={"symbol": symbol, "limit": limit}, timeout=timeout)
    r.raise_for_status()
    rows = r.json()
    return pd.Series([float(x["fundingRate"]) for x in rows], index=pd.to_datetime([x["fundingTime"] for x in rows], unit="ms"), name="funding_rate")


def fetch_binance_premium(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 1000, timeout: int = 30) -> pd.Series:
    import requests
    r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndexKlines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=timeout)
    r.raise_for_status()
    rows = r.json()
    # [openTime, o, h, l, c, v, closeTime, ...]: the premium is known at the bar CLOSE
    return pd.Series([float(x[4]) for x in rows], index=pd.to_datetime([x[6] for x in rows], unit="ms").floor("s"), name="perp_premium")


def fetch_binance_oi(symbol: str = "BTCUSDT", period: str = "1h", limit: int = 500, timeout: int = 30) -> pd.Series:
    import requests
    r = requests.get("https://fapi.binance.com/futures/data/openInterestHist", params={"symbol": symbol, "period": period, "limit": limit}, timeout=timeout)
    r.raise_for_status()
    rows = r.json()
    return pd.Series([float(x["sumOpenInterest"]) for x in rows], index=pd.to_datetime([x["timestamp"] for x in rows], unit="ms"), name="crypto_oi")


def update_ext(symbols: dict | None = None, log=print) -> dict:
    """Fetch every free Ext feed that has a fetcher; ``symbols`` = {'funding_rate': ['BTCUSDT'], ...} for per-symbol crypto feeds.
    Feeds with no free source are skipped with a note telling the user where to drop the CSV. One failure never stops the rest."""
    done: dict = {}
    for name, spec in EXT_FEEDS.items():
        try:
            if spec.fetch_id and name in ("vvix", "skew"):
                s = fetch_cboe(spec.fetch_id)
            elif spec.fetch_id:
                s = fetch_fred(spec.fetch_id)
            elif name == "putcall":
                s = fetch_cboe_putcall()
            elif name in ("funding_rate", "perp_premium", "crypto_oi"):
                fetcher = {"funding_rate": fetch_binance_funding, "perp_premium": fetch_binance_premium, "crypto_oi": fetch_binance_oi}[name]
                for sym in (symbols or {}).get(name, []):
                    s2 = fetcher(sym)
                    p = feed_path(name, sym)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    s2.dropna().rename("value").rename_axis("date").reset_index().to_csv(p, index=False)
                    done[f"{name}:{sym}"] = str(p)
                continue
            else:
                log(f"{name}: no free public history -- drop a date,value CSV at {feed_path(name, 'SYMBOL' if spec.per_symbol else None)}")
                continue
            p = feed_path(name)
            p.parent.mkdir(parents=True, exist_ok=True)
            s.dropna().rename("value").rename_axis("date").reset_index().to_csv(p, index=False)
            done[name] = str(p)
        except Exception as exc:  # noqa: BLE001
            log(f"{name}: {exc}")
    return done
