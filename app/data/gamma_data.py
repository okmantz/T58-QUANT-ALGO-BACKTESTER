"""
Free gamma-exposure (GEX) data for the instruments this app trades (ES, NQ, GC).

Why this exists (2026-10-17): the gamma-exposure map in the video this feature was modelled on (ITMatrixHQ.com) is a
paid/behind-a-login product. Everything the video teaches -- dealer gamma, open-interest "largest levels", the gamma
flip, call/put walls, positive- vs negative-gamma regimes -- can be computed for free from two public sources:

  1. CBOE delayed option chains (https://cdn.cboe.com/api/global/delayed_quotes/options/<SYMBOL>.json): every listed
     option with its open interest, volume, IV and greeks (including gamma), no key and no account. This is the LIVE
     source: ``fetch_chain`` -> ``compute_gex`` -> per-strike table + levels, ``record_snapshot`` to build history.
  2. SqueezeMetrics' public daily series (https://squeezemetrics.com/monitor/static/DIX.csv): date, SPX price, DIX and
     GEX back to 2011. This is the HISTORICAL source a backtest can use: ``fetch_squeezemetrics_history``.

Instrument mapping (what the option chain is an ETF/index proxy for):
    ES -> SPX   ("_SPX" on CBOE; SPY as the ETF fallback)
    NQ -> NDX   ("_NDX"; QQQ fallback)
    GC -> GLD   (gold ETF; options on gold futures are not on CBOE's free feed)

Honest limits (kept in the UI too): CBOE's feed is delayed ~15 minutes and only describes TODAY. There is no free
historical per-strike archive, so historical wall/flip levels only exist for days this app recorded a snapshot (it keeps
every snapshot you take in data/gamma/<SYMBOL>.csv) or that you import (``import_levels_csv``). Daily net GEX history
comes from SqueezeMetrics. Where a backtest has neither, the gex_* indicators in app.strategy.indicators_v9 fall back to
clearly-labelled price-derived proxies instead of failing.

Dealer convention (the standard "GEX" one, used by SqueezeMetrics/SpotGamma-style tools): customers are net long calls
and net long puts, so dealers are LONG gamma on calls and SHORT gamma on puts. Per-strike dollar gamma for a 1% move:
    call_gex = +gamma * OI * 100 * spot^2 * 0.01
    put_gex  = -gamma * OI * 100 * spot^2 * 0.01
    net_gex  = call_gex + put_gex       (positive = dealers dampen moves, negative = dealers amplify them)
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from app.data.storage import get_app_base_dir

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
SQUEEZEMETRICS_URL = "https://squeezemetrics.com/monitor/static/DIX.csv"

# instrument -> (primary CBOE symbol, fallback ETF symbol, how to map option-underlying levels onto the futures price)
INSTRUMENT_MAP: dict[str, dict[str, str]] = {
    "ES": {"label": "E-mini S&P 500", "cboe": "_SPX", "fallback": "SPY", "underlying": "SPX", "scale": "offset"},
    "NQ": {"label": "E-mini Nasdaq 100", "cboe": "_NDX", "fallback": "QQQ", "underlying": "NDX", "scale": "offset"},
    "GC": {"label": "Gold futures", "cboe": "GLD", "fallback": "GLD", "underlying": "GLD", "scale": "ratio"},
}

LEVEL_COLUMNS = ["date", "spot", "net_gex", "gamma_flip", "call_wall", "put_wall", "put_call_oi_ratio", "source"]
FRAME_COLUMNS = ["gex_net", "gex_flip", "gex_call_wall", "gex_put_wall", "gex_put_call_oi"]

_OCC_RE = re.compile(r"^(?P<root>[A-Z0-9_.]{1,8}?)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<strike>\d{8})$")


class GammaDataError(Exception):
    """Raised for a network/format problem the caller should show to the person verbatim."""


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def gamma_dir() -> Path:
    d = get_app_base_dir() / "data" / "gamma"
    d.mkdir(parents=True, exist_ok=True)
    return d


def levels_path(instrument: str) -> Path:
    return gamma_dir() / f"{_inst(instrument)}_levels.csv"


def history_path() -> Path:
    return gamma_dir() / "squeezemetrics_spx_history.csv"


def _inst(instrument: str) -> str:
    key = (instrument or "").strip().upper()
    for alias, canon in (("ES1!", "ES"), ("MES", "ES"), ("NQ1!", "NQ"), ("MNQ", "NQ"), ("GC1!", "GC"), ("MGC", "GC"),
                         ("XAUUSD", "GC"), ("SPX", "ES"), ("NDX", "NQ")):
        if key == alias:
            key = canon
    if key not in INSTRUMENT_MAP:
        raise GammaDataError(f"Gamma data is available for ES, NQ and GC -- '{instrument}' is not one of them.")
    return key


# ---------------------------------------------------------------------------
# network (isolated so tests never touch it)
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float = 20.0) -> bytes:
    import requests

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "T58-Backtester/1.0 (+gamma-data)"})
    if resp.status_code != 200:
        raise GammaDataError(f"{url} answered HTTP {resp.status_code}.")
    return resp.content


fetcher: Callable[[str], bytes] = _http_get  # tests replace this


def fetch_chain(symbol: str) -> dict:
    """Downloads one CBOE delayed option chain and returns its ``data`` object."""
    try:
        payload = json.loads(fetcher(CBOE_URL.format(symbol=symbol)).decode("utf-8", "replace"))
    except GammaDataError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GammaDataError(f"Could not download the {symbol} option chain from CBOE: {exc}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not data or not data.get("options"):
        raise GammaDataError(f"CBOE returned no options for {symbol}.")
    return data


# ---------------------------------------------------------------------------
# computation
# ---------------------------------------------------------------------------

def parse_occ(option: str) -> Optional[dict]:
    """'SPX251017C06800000' -> {'expiry': date, 'type': 'C', 'strike': 6800.0}. None if it is not an OCC symbol."""
    m = _OCC_RE.match(str(option or "").strip().upper().lstrip("_"))
    if not m:
        return None
    try:
        expiry = date(2000 + int(m["yy"]), int(m["mm"]), int(m["dd"]))
    except ValueError:
        return None
    return {"expiry": expiry, "type": m["cp"], "strike": int(m["strike"]) / 1000.0}


def _num(v: Any) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


@dataclass
class GexResult:
    instrument: str
    symbol: str
    spot: float
    as_of: str
    max_dte: Optional[int]
    net_gex: float                      # dollars per 1% move
    gamma_flip: Optional[float]
    call_wall: Optional[float]
    put_wall: Optional[float]
    put_call_oi_ratio: Optional[float]
    regime: str                         # "positive" | "negative"
    strikes: list[dict] = field(default_factory=list)
    top_levels: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument, "symbol": self.symbol, "spot": self.spot, "as_of": self.as_of,
            "max_dte": self.max_dte, "net_gex": self.net_gex, "gamma_flip": self.gamma_flip,
            "call_wall": self.call_wall, "put_wall": self.put_wall, "put_call_oi_ratio": self.put_call_oi_ratio,
            "regime": self.regime, "strikes": self.strikes, "top_levels": self.top_levels, "notes": self.notes,
        }


def compute_gex(data: dict, *, instrument: str = "ES", symbol: str = "", max_dte: Optional[int] = 7,
                strike_window_pct: float = 8.0, today: Optional[date] = None) -> GexResult:
    """Turns a CBOE ``data`` object into per-strike dollar gamma and the levels traders read off a GEX map.

    ``max_dte`` keeps expirations up to that many days out (None = all). Near-dated options carry nearly all the gamma
    that moves intraday price, which is what a futures strategy cares about. ``strike_window_pct`` limits the table to
    strikes within +/- that percent of spot (the far wings are noise)."""
    today = today or datetime.now(timezone.utc).date()
    spot = _num(data.get("current_price") or data.get("close") or data.get("prev_day_close"))
    if spot <= 0:
        raise GammaDataError("The option chain has no usable spot price.")
    notes: list[str] = []
    per: dict[float, dict] = {}
    n_used = 0
    for opt in data.get("options") or []:
        meta = parse_occ(opt.get("option", ""))
        if meta is None:
            continue
        dte = (meta["expiry"] - today).days
        if dte < 0 or (max_dte is not None and dte > max_dte):
            continue
        oi, gamma = _num(opt.get("open_interest")), _num(opt.get("gamma"))
        if oi <= 0 or gamma <= 0:
            continue
        k = meta["strike"]
        if abs(k - spot) / spot * 100.0 > strike_window_pct:
            continue
        dollar = gamma * oi * 100.0 * spot * spot * 0.01
        row = per.setdefault(k, {"strike": k, "call_gex": 0.0, "put_gex": 0.0, "call_oi": 0.0, "put_oi": 0.0,
                                 "call_vol": 0.0, "put_vol": 0.0})
        if meta["type"] == "C":
            row["call_gex"] += dollar
            row["call_oi"] += oi
            row["call_vol"] += _num(opt.get("volume"))
        else:
            row["put_gex"] -= dollar
            row["put_oi"] += oi
            row["put_vol"] += _num(opt.get("volume"))
        n_used += 1
    if not per:
        raise GammaDataError("No options with open interest and gamma fell inside the chosen expiry/strike window.")
    strikes = sorted(per.values(), key=lambda r: r["strike"])
    for r in strikes:
        r["net_gex"] = r["call_gex"] + r["put_gex"]
    net_total = sum(r["net_gex"] for r in strikes)

    # Gamma flip: the strike where cumulative net gamma (from the lowest strike up) changes sign nearest spot.
    flip: Optional[float] = None
    cum, prev_k, prev_c = 0.0, None, None
    best = None
    for r in strikes:
        cum += r["net_gex"]
        if prev_c is not None and (prev_c < 0 <= cum or prev_c > 0 >= cum):
            span = cum - prev_c
            k = r["strike"] if span == 0 else prev_k + (r["strike"] - prev_k) * (-prev_c / span)
            if best is None or abs(k - spot) < abs(best - spot):
                best = k
        prev_k, prev_c = r["strike"], cum
    flip = round(best, 2) if best is not None else None

    above = [r for r in strikes if r["strike"] >= spot and r["call_gex"] > 0]
    below = [r for r in strikes if r["strike"] <= spot and r["put_gex"] < 0]
    call_wall = max(above, key=lambda r: r["call_gex"])["strike"] if above else None
    put_wall = min(below, key=lambda r: r["put_gex"])["strike"] if below else None
    coi, poi = sum(r["call_oi"] for r in strikes), sum(r["put_oi"] for r in strikes)
    pcr = round(poi / coi, 3) if coi > 0 else None

    ranked = sorted(strikes, key=lambda r: abs(r["net_gex"]), reverse=True)[:6]
    top = [{"strike": r["strike"], "net_gex": r["net_gex"], "kind": "call-heavy (resistance/magnet)" if r["net_gex"] > 0
            else "put-heavy (support/accelerant)"} for r in ranked]
    if flip is None:
        notes.append("Net gamma never changes sign inside this window, so there is no gamma flip level.")
    sym = symbol or INSTRUMENT_MAP.get(instrument, {}).get("cboe", "")
    return GexResult(
        instrument=instrument, symbol=sym, spot=round(spot, 2),
        as_of=str(data.get("last_trade_time") or datetime.now(timezone.utc).isoformat(timespec="minutes")),
        max_dte=max_dte, net_gex=net_total, gamma_flip=flip, call_wall=call_wall, put_wall=put_wall,
        put_call_oi_ratio=pcr, regime="positive" if net_total >= 0 else "negative", strikes=strikes,
        top_levels=top, notes=notes,
    )


def live_gex(instrument: str, *, max_dte: Optional[int] = 7, today: Optional[date] = None) -> GexResult:
    """Downloads the chain for an instrument (primary symbol, then the ETF fallback) and computes it."""
    inst = _inst(instrument)
    cfg = INSTRUMENT_MAP[inst]
    last_err: Optional[Exception] = None
    for sym in dict.fromkeys([cfg["cboe"], cfg["fallback"]]):
        try:
            return compute_gex(fetch_chain(sym), instrument=inst, symbol=sym, max_dte=max_dte, today=today)
        except GammaDataError as exc:
            last_err = exc
    raise GammaDataError(str(last_err) if last_err else "Could not load the option chain.")


# ---------------------------------------------------------------------------
# stored levels (one row per day, appended every time a snapshot is recorded or imported)
# ---------------------------------------------------------------------------

def record_snapshot(result: GexResult, *, day: Optional[date] = None) -> Path:
    """Upserts today's levels for the instrument into data/gamma/<INST>_levels.csv. Returns the file path."""
    day = day or datetime.now(timezone.utc).date()
    path = levels_path(result.instrument)
    df = load_levels(result.instrument)
    row = {"date": day.isoformat(), "spot": result.spot, "net_gex": result.net_gex, "gamma_flip": result.gamma_flip,
           "call_wall": result.call_wall, "put_wall": result.put_wall, "put_call_oi_ratio": result.put_call_oi_ratio,
           "source": "cboe_snapshot"}
    df = df[df["date"] != row["date"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).sort_values("date")
    df.to_csv(path, index=False)
    return path


def load_levels(instrument: str) -> pd.DataFrame:
    path = levels_path(instrument)
    if not path.exists():
        return pd.DataFrame(columns=LEVEL_COLUMNS)
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=LEVEL_COLUMNS)
    for c in LEVEL_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df["date"] = df["date"].astype(str).str[:10]
    return df[LEVEL_COLUMNS]


def import_levels_csv(text: str, instrument: str, *, source: str = "imported") -> int:
    """Imports daily levels from ANY source (ITMatrixHQ, Barchart, SpotGamma, a spreadsheet you keep). Needs a ``date``
    column plus any of: gamma_flip / call_wall / put_wall / net_gex / spot / put_call_oi_ratio (a few common aliases are
    accepted). Existing days are overwritten. Returns the number of rows imported."""
    inst = _inst(instrument)
    rows = list(csv.DictReader(io.StringIO((text or "").strip())))
    if not rows:
        raise GammaDataError("The CSV is empty.")
    low = {(k or "").strip().lower().replace(" ", "_"): k for k in rows[0].keys()}
    aliases = {
        "date": ("date", "day", "timestamp"),
        "spot": ("spot", "price", "close", "last"),
        "net_gex": ("net_gex", "gex", "net_gamma", "total_gex"),
        "gamma_flip": ("gamma_flip", "flip", "zero_gamma", "gex_flip", "zero_gamma_level"),
        "call_wall": ("call_wall", "call_resistance", "gex_call_wall"),
        "put_wall": ("put_wall", "put_support", "gex_put_wall"),
        "put_call_oi_ratio": ("put_call_oi_ratio", "put_call_ratio", "pcr", "oi_put_call"),
    }
    cols: dict[str, Optional[str]] = {}
    for canon, names in aliases.items():
        cols[canon] = next((low[n] for n in names if n in low), None)
    if cols["date"] is None:
        raise GammaDataError("The CSV needs a 'date' column.")
    if not any(cols[c] for c in ("gamma_flip", "call_wall", "put_wall", "net_gex")):
        raise GammaDataError("The CSV needs at least one of: gamma_flip, call_wall, put_wall, net_gex.")
    new = []
    for r in rows:
        try:
            d = pd.to_datetime(r[cols["date"]]).date().isoformat()
        except Exception:  # noqa: BLE001
            continue
        item = {"date": d, "source": source}
        for c in ("spot", "net_gex", "gamma_flip", "call_wall", "put_wall", "put_call_oi_ratio"):
            raw = r.get(cols[c]) if cols[c] else None
            try:
                item[c] = float(str(raw).replace(",", "")) if raw not in (None, "") else None
            except ValueError:
                item[c] = None
        new.append(item)
    if not new:
        raise GammaDataError("No rows with a readable date were found.")
    cur = load_levels(inst)
    nd = pd.DataFrame(new)
    cur = cur[~cur["date"].isin(nd["date"])]
    out = pd.concat([cur, nd], ignore_index=True).sort_values("date")[LEVEL_COLUMNS]
    out.to_csv(levels_path(inst), index=False)
    return len(new)


def fetch_squeezemetrics_history() -> pd.DataFrame:
    """Free daily SPX series from SqueezeMetrics: date, price, dix, gex. Cached to data/gamma/. Raises GammaDataError."""
    try:
        raw = fetcher(SQUEEZEMETRICS_URL).decode("utf-8", "replace")
        df = pd.read_csv(io.StringIO(raw))
    except GammaDataError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GammaDataError(f"Could not download the SqueezeMetrics history: {exc}") from exc
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"date", "gex"}.issubset(df.columns):
        raise GammaDataError("The SqueezeMetrics file did not have the expected date/gex columns.")
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df = df.sort_values("date").drop_duplicates("date")
    df.to_csv(history_path(), index=False)
    return df


def load_squeezemetrics_history() -> pd.DataFrame:
    p = history_path()
    if not p.exists():
        return pd.DataFrame(columns=["date", "price", "dix", "gex"])
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["date", "price", "dix", "gex"])


# ---------------------------------------------------------------------------
# merging into a price frame (lookahead-safe)
# ---------------------------------------------------------------------------

def merge_gamma_levels(df: pd.DataFrame, instrument: str, *, levels: Optional[pd.DataFrame] = None,
                       history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Returns a COPY of ``df`` with gex_net / gex_flip / gex_call_wall / gex_put_wall / gex_put_call_oi columns.

    Lookahead safety: a day's levels are only applied to bars of LATER days (each bar sees the most recent snapshot dated
    strictly before its own calendar day). Index-option levels (SPX/NDX) are shifted onto the futures price with the
    previous day's basis (previous futures close minus the snapshot spot); GLD levels are scaled by the previous-day
    futures/ETF ratio. Days with no snapshot stay NaN, and app.strategy.indicators_v9 then falls back to its proxy."""
    inst = _inst(instrument)
    out = df.copy()
    for c in FRAME_COLUMNS:
        out[c] = float("nan")
    if "timestamp" not in out.columns or out.empty:
        return out
    lv = load_levels(inst) if levels is None else levels.copy()
    hist = load_squeezemetrics_history() if history is None else history.copy()
    ts = pd.to_datetime(out["timestamp"])
    day = ts.dt.tz_localize(None).dt.normalize() if ts.dt.tz is not None else ts.dt.normalize()

    # daily GEX history (SPX/ES and NQ only -- gold has no equivalent free series). Each bar sees the PREVIOUS day's value.
    if inst in ("ES", "NQ") and len(hist):
        h = hist.copy()
        h["d"] = pd.to_datetime(h["date"]).dt.normalize() + pd.Timedelta(days=1)
        h = h.sort_values("d")[["d", "gex"]].rename(columns={"gex": "gex_net"})
        left = pd.DataFrame({"d": day.values}, index=out.index).sort_values("d")
        merged = pd.merge_asof(left, h, on="d", direction="backward")
        merged.index = left.index
        out["gex_net"] = merged["gex_net"].reindex(out.index).values

    if len(lv):
        l = lv.copy()
        l["d"] = pd.to_datetime(l["date"]).dt.normalize()
        l = l.sort_values("d")
        shifted = l.assign(d=l["d"] + pd.Timedelta(days=1))
        left = pd.DataFrame({"d": day.values}, index=out.index).sort_values("d")
        m = pd.merge_asof(left, shifted, on="d", direction="backward")
        m.index = left.index
        m = m.reindex(out.index)
        # previous futures close, per day, for the basis
        close_by_day = out.groupby(day.values)["close"].last()
        prev_close = pd.Series(day.values, index=out.index).map(
            lambda d: close_by_day[close_by_day.index < d].iloc[-1] if (close_by_day.index < d).any() else float("nan"))
        spot = m["spot"].astype(float)
        if INSTRUMENT_MAP[inst]["scale"] == "ratio":
            factor = (prev_close / spot).where(spot > 0)
            for src, dst in (("gamma_flip", "gex_flip"), ("call_wall", "gex_call_wall"), ("put_wall", "gex_put_wall")):
                out[dst] = m[src].astype(float) * factor
        else:
            basis = prev_close - spot
            for src, dst in (("gamma_flip", "gex_flip"), ("call_wall", "gex_call_wall"), ("put_wall", "gex_put_wall")):
                out[dst] = m[src].astype(float) + basis
        out["gex_put_call_oi"] = m["put_call_oi_ratio"].astype(float)
        if out["gex_net"].isna().all():
            out["gex_net"] = m["net_gex"].astype(float)
    return out


def gamma_status() -> dict:
    """What gamma data this machine holds -- shown on the Gamma page and used to label backtest data sources."""
    out: dict[str, Any] = {"instruments": {}, "history_rows": int(len(load_squeezemetrics_history()))}
    for inst in INSTRUMENT_MAP:
        lv = load_levels(inst)
        out["instruments"][inst] = {
            "rows": int(len(lv)),
            "first": (str(lv["date"].iloc[0]) if len(lv) else None),
            "last": (str(lv["date"].iloc[-1]) if len(lv) else None),
        }
    return out
