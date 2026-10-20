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
