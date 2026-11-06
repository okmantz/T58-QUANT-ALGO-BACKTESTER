"""Independent-market pool for the pooled cross-market edge test (2026-11-05 audit).

The pooled test asks "does the SAME fixed strategy work on other markets too?" and fails closed below 8 markets. Two problems made it
unreachable and then, once reachable, misleading:

  1. The web pool builder only accepted files named ``<digits>.csv`` at the same bar size, capped at 6 markets, so shipped data (ES, NQ,
     GC, SI ... parquet) gave a pool of 0 and nothing launched from the web UI could reach the sealed-slice step.
  2. ES and MES (or NQ and MNQ) are the SAME price bars. Counting both toward the floor of 8 would overstate the evidence: two entries
     are one observation.

``build_market_pool`` accepts any stored file (csv/parquet/...) whose bar size matches, or can be built exactly from a finer file by
resampling; ``dedupe_independent_frames`` collapses identical or near-identical series (a micro and its full-size contract, two files of
one market, any pair whose aligned returns correlate above ``CORR_THRESHOLD``) so only genuinely distinct markets count. The floor of 8
is NOT lowered anywhere: more distinct data is the fix, not a looser floor.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

POOL_FLOOR = 8                 # PromotionConfig.pooled_min_markets -- never lowered here
POOL_MAX_MARKETS = 12          # the pooled test is specified for 8-12 markets
# Calibrated on the shipped data with 15-minute log returns: the SAME market from two vendors correlates 0.98-0.99 (ES vs SPX500 CFD 0.983,
# NQ vs NAS100 CFD 0.981, GC vs XAUUSD 0.990) while distinct-but-related markets stay below it (ES vs NQ 0.953, SPX vs NAS100 0.945,
# GC vs SI 0.716, everything else under 0.5). 0.975 separates the two groups.
CORR_THRESHOLD = 0.975
MIN_OVERLAP_BARS = 300
COARSE_MINUTES = 15            # series used for the cross-file comparison (fine bars are too noisy across vendors' clocks)

_NAME_PATTERNS = (
    (re.compile(r"(?<![A-Za-z0-9])(\d{1,4})\s*(?:minutes|minute|mins|min|m)(?![A-Za-z])", re.I), 1),     # "ES 5min", "NQ_1m", "..._1m.parquet"
    (re.compile(r"[_\-\s](\d{1,2})\s*(?:hour|hr|h)(?![A-Za-z])", re.I), 60),                              # "..._4h"
    (re.compile(r"[A-Za-z]{3,}(\d{1,4})\.(?:csv|tsv|txt)$", re.I), 1),                         # "EURUSD5.csv" (MT-style minutes suffix)
)


def bar_minutes_from_name(name: str) -> int | None:
    """Bar size in minutes read from a file name, or None when the name does not say."""
    base = str(name or "").replace("\\", "/").split("/")[-1]
    for pat, mult in _NAME_PATTERNS:
        m = pat.search(base)
        if m:
            v = int(m.group(1)) * mult
            if v > 0:
                return v
    return None


def bar_minutes_of(df: pd.DataFrame) -> int | None:
    """Bar size in minutes from the data itself (median positive timestamp step over the last 5,000 bars)."""
    try:
        ts = pd.DatetimeIndex(df["timestamp"].tail(5000)).asi8
        d = np.diff(ts)
        d = d[d > 0]
        if not len(d):
            return None
        return max(1, int(round(float(np.median(d)) / 60e9)))
    except Exception:  # noqa: BLE001
        return None


def frame_fingerprint(df: pd.DataFrame, n: int = 2000) -> str:
    """Hash of the bar CONTENT (timestamps and rounded closes of the last ``n`` bars): identical files give identical fingerprints."""
    tail = df.tail(n)
    h = hashlib.sha1()
    h.update(pd.DatetimeIndex(tail["timestamp"]).asi8.tobytes())
    h.update(np.round(tail["close"].to_numpy(dtype=float), 6).tobytes())
    return h.hexdigest()


def returns_correlation(a: pd.DataFrame, b: pd.DataFrame, min_overlap: int = MIN_OVERLAP_BARS) -> float | None:
    """Correlation of bar-to-bar log returns on the timestamps both frames share; None when they overlap on fewer than ``min_overlap`` bars."""
    try:
        sa = pd.Series(a["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(a["timestamp"]))
        sb = pd.Series(b["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(b["timestamp"]))
        sa = sa[~sa.index.duplicated(keep="last")]
        sb = sb[~sb.index.duplicated(keep="last")]
        common = sa.index.intersection(sb.index)
        if len(common) < min_overlap:
            return None
        common = common[-20000:]
        ra = np.diff(np.log(np.clip(sa.loc[common].to_numpy(), 1e-12, None)))
        rb = np.diff(np.log(np.clip(sb.loc[common].to_numpy(), 1e-12, None)))
        if ra.std() == 0 or rb.std() == 0:
            return 1.0 if np.allclose(ra, rb) else None
        return float(np.corrcoef(ra, rb)[0, 1])
    except Exception:  # noqa: BLE001
        return None


def coarse_closes(df: pd.DataFrame, minutes: int = COARSE_MINUTES) -> pd.Series:
    """Last close of each ``minutes``-minute window over the WHOLE history: a small stand-in for a big file, so two vendors' versions of one
    market are comparable even when the bars kept for the pool (a recent tail) do not overlap in time."""
    try:
        idx = pd.DatetimeIndex(df["timestamp"])
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        ser = pd.Series(df["close"].to_numpy(dtype=float), index=idx)
        return ser.resample(f"{int(minutes)}min").last().dropna()
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)


def coarse_correlation(a: pd.Series, b: pd.Series, min_overlap: int = MIN_OVERLAP_BARS) -> float | None:
    """Correlation of log returns over the windows both coarse series cover (last 50,000); None when they share fewer than ``min_overlap``."""
    try:
        both = a.index.intersection(b.index)
        if len(both) < min_overlap:
            return None
        both = both[-50000:]
        ra = np.diff(np.log(np.clip(a.loc[both].to_numpy(), 1e-12, None)))
        rb = np.diff(np.log(np.clip(b.loc[both].to_numpy(), 1e-12, None)))
        if ra.std() == 0 or rb.std() == 0:
            return None
        return float(np.corrcoef(ra, rb)[0, 1])
    except Exception:  # noqa: BLE001
        return None


def underlying_symbol(sym: str | None) -> str:
    """Micro -> full-size symbol (MES -> ES, MNQ -> NQ, MGC -> GC, SIL -> SI); anything else maps to itself."""
    s = str(sym or "").strip().upper()
    try:
        from app.data.instrument_specs import MICRO_EQUIVALENT, MICRO_EQUIVALENT_CROSS
        for full, micro in list(MICRO_EQUIVALENT.items()) + list(MICRO_EQUIVALENT_CROSS.items()):
            if str(micro).upper() == s:
                return str(full).upper()
    except Exception:  # noqa: BLE001
        pass
    return s


def dedupe_independent_frames(frames: dict, corr_threshold: float = CORR_THRESHOLD, *, full_frames: dict | None = None,
                              coarse: dict | None = None) -> tuple[dict, dict]:
    """Keep one entry per genuinely distinct market.

    Two entries are the same market when they share an underlying symbol (a micro and its parent), an identical bar-content fingerprint,
    or aligned log-return correlation above ``corr_threshold``. The FIRST entry in ``frames`` order wins (callers put the primary market
    first). ``full_frames`` optionally supplies longer histories for the correlation test (tail windows of different vendors may not
    overlap in time). Returns ({kept symbol: frame}, {dropped symbol: "same market as <kept>"})."""
    kept: dict = {}
    dropped: dict = {}
    fps: dict = {}
    for sym, df in frames.items():
        dup_of = None
        u = underlying_symbol(sym)
        for k in kept:
            if underlying_symbol(k) == u:
                dup_of = k
                break
        if dup_of is None and df is not None and len(df):
            fp = frame_fingerprint(df)
            dup_of = fps.get(fp)
        if dup_of is None and df is not None:
            for k, kdf in kept.items():
                c = None
                if coarse and sym in coarse and k in coarse:
                    c = coarse_correlation(coarse[sym], coarse[k])
                if c is None:
                    c = returns_correlation((full_frames or {}).get(sym, df), (full_frames or {}).get(k, kdf))
                if c is not None and c >= corr_threshold:
                    dup_of = k
                    break
        if dup_of is not None:
            dropped[sym] = f"same market as {dup_of} (identical or near-identical bars)"
            continue
        kept[sym] = df
        if df is not None and len(df):
            fps[frame_fingerprint(df)] = sym
    return kept, dropped


def dedupe_r_series(per_market: dict, corr_threshold: float = CORR_THRESHOLD) -> tuple[dict, dict]:
    """Safety net at the statistic level: collapse markets whose per-trade R series are the same observation.

    ``per_market`` is {symbol: (r_list, day_list)}. Identical (day, R) sequences, or day-aligned daily R sums correlating above
    ``corr_threshold`` over at least 20 shared days, are one market (first wins). Returns (kept, {dropped: reason})."""
    kept: dict = {}
    dropped: dict = {}
    daily: dict = {}
    sigs: dict = {}
    for sym, (rs, days) in per_market.items():
        rs = list(rs or [])
        days = list(days) if days is not None else [None] * len(rs)
        if not rs:
            kept[sym] = (rs, days)
            continue
        sig = hashlib.sha1(repr([(str(d)[:10], round(float(r), 6)) for d, r in zip(days, rs)]).encode()).hexdigest()
        if sig in sigs:
            dropped[sym] = f"same trades as {sigs[sig]}"
            continue
        ser = pd.Series([float(r) for r in rs]).groupby([str(d)[:10] for d in days]).sum()
        dup = None
        for k, kser in daily.items():
            both = ser.index.intersection(kser.index)
            if len(both) >= 20 and ser.loc[both].std() > 0 and kser.loc[both].std() > 0:
                if float(np.corrcoef(ser.loc[both], kser.loc[both])[0, 1]) >= corr_threshold:
                    dup = k
                    break
        if dup is not None:
            dropped[sym] = f"daily R moves with {dup} (correlation >= {corr_threshold:g}): one observation, not two"
            continue
        kept[sym] = (rs, days)
        daily[sym] = ser
        sigs[sig] = sym
    return kept, dropped


# ------------------------------------------------------------------------------------------------------------- the pool builder
@dataclass
class PoolBuild:
    frames: dict = field(default_factory=dict)             # {SYMBOL: DataFrame} other independent markets (primary NOT included)
    risks: dict = field(default_factory=dict)
    primary: str = ""
    duplicates: dict = field(default_factory=dict)         # {symbol: why it was not counted}
    skipped: dict = field(default_factory=dict)            # {symbol or file: why it could not be used}
    bar_minutes: int | None = None
    distinct_total: int = 0                                # distinct markets found INCLUDING the primary, before the 12-market cap

    @property
    def n_independent(self) -> int:
        """Distinct markets available INCLUDING the primary market (uncapped count; the pool itself holds at most 12)."""
        return max(self.distinct_total, len(self.frames) + 1)

    def summary(self, floor: int = POOL_FLOOR) -> str:
        n = self.n_independent
        txt = f"pooled test: {n} of {floor} markets available"
        if n >= floor:
            return txt + " - ready"
        return txt + f" - {floor - n} more distinct market(s) needed; the floor of {floor} is not lowered"

    def to_dict(self, floor: int = POOL_FLOOR) -> dict:
        return {"available": self.n_independent, "floor": floor, "ready": self.n_independent >= floor, "markets": [self.primary] + sorted(self.frames),
                "duplicates": dict(self.duplicates), "skipped": dict(self.skipped), "bar_minutes": self.bar_minutes, "text": self.summary(floor)}


_POOL_CACHE: dict = {}


def _resample_exact(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate finer bars into ``minutes``-minute OHLCV bars (open first, high max, low min, close last, volume sum)."""
    d = df.copy()
    d["timestamp"] = pd.DatetimeIndex(d["timestamp"])
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in d.columns:
        agg["volume"] = "sum"
    out = d.set_index("timestamp").resample(f"{int(minutes)}min", label="left", closed="left").agg(agg).dropna(subset=["open", "close"])
    return out.reset_index()


def _symbol_for_name(name: str) -> str:
    from app.data.instrument_specs import guess_any_instrument_symbol
    return (guess_any_instrument_symbol(name) or re.sub(r"\W+", "", str(name).replace("\\", "/").split("/")[0]).upper() or "MARKET")


def build_market_pool(label: str, df: pd.DataFrame, *, bars: int | None = None, limit: int | None = None, risk_factory=None,
                      corr_threshold: float = CORR_THRESHOLD) -> PoolBuild:
    """Other stored markets, at the primary's bar size, for the pooled cross-market test. Never raises; never blocks a run.

    Accepts any stored file type; a market whose only files are finer than the primary's bar size is resampled EXACTLY (open first, high
    max, low min, close last). Bar size comes from the file name when it says, else from the data. One file per symbol (the one with the
    longest history). Genuinely distinct markets only (see ``dedupe_independent_frames``), at most ``limit`` (default 12)."""
    limit = POOL_MAX_MARKETS - 1 if limit is None else int(limit)
    mine = _symbol_for_name(label)
    out = PoolBuild(primary=mine)
    target = bar_minutes_from_name(label) or bar_minutes_of(df)
    out.bar_minutes = target
    if not target:
        out.skipped["(primary)"] = "could not tell this dataset's bar size"
        return out
    if bars is None:
        bars = 12000 * max(1, int(round(5.0 / target)))   # ~equal wall-clock history whether the data is 1-minute or 5-minute+
    try:
        from app.data.importer import import_csv
        from app.data.storage import list_stored_datasets
        datasets = list_stored_datasets()
    except Exception as exc:  # noqa: BLE001
        out.skipped["(storage)"] = f"could not list stored datasets: {exc}"
        return out

    by_symbol: dict[str, list] = {}
    for d in datasets:
        sym = _symbol_for_name(d.name)
        if sym == mine or underlying_symbol(sym) == underlying_symbol(mine):
            if sym != mine:
                out.duplicates[sym] = f"same underlying market as {mine}"
            continue
        by_symbol.setdefault(sym, []).append(d)

    loaded: dict = {}
    full: dict = {}
    coarse: dict = {mine: coarse_closes(df, max(COARSE_MINUTES, int(target)))}
    for sym in sorted(by_symbol):
        files = by_symbol[sym]

        def _rank(d, target=target):
            nm = bar_minutes_from_name(d.name)
            exact = nm == target
            finer = nm is not None and nm < target and target % nm == 0
            return (0 if exact else (1 if finer else (2 if nm is None else 3)), -int(d.size_bytes))
        files = sorted(files, key=_rank)
        got = False
        for d in files:
            nm = bar_minutes_from_name(d.name)
            if nm is not None and nm > target:
                continue
            if nm is not None and nm != target and target % nm != 0:
                continue
            if int(getattr(d, "size_bytes", 1) or 0) < 2048:
                out.skipped[d.name] = "placeholder-sized file"
                continue
            key = (str(d.path), int(getattr(d, "mtime_ns", 0)), int(bars), int(target))
            if key in _POOL_CACHE:
                tail_df, full_df, coarse_s = _POOL_CACHE[key]
            else:
                try:
                    res = import_csv(d.path)
                except Exception as exc:  # noqa: BLE001
                    out.skipped[d.name] = f"could not read: {exc}"
                    continue
                if not getattr(res, "is_valid", False) or res.dataframe is None or not len(res.dataframe):
                    out.skipped[d.name] = "file is not valid market data"
                    continue
                fdf = res.dataframe
                actual = bar_minutes_of(fdf)
                if actual is None:
                    out.skipped[d.name] = "could not tell this file's bar size"
                    continue
                if actual > target or (actual != target and target % actual != 0):
                    out.skipped[d.name] = f"bar size {actual}m does not fit the {target}m primary"
                    continue
                tail_src = fdf.tail(int(bars) * max(1, target // actual) + 10).reset_index(drop=True)
                tail_df = (_resample_exact(tail_src, target) if actual != target else tail_src).tail(int(bars)).reset_index(drop=True)
                full_df = tail_df
                coarse_s = coarse_closes(fdf, max(COARSE_MINUTES, int(target)))
                _POOL_CACHE[key] = (tail_df, full_df, coarse_s)
                while len(_POOL_CACHE) > 24:
                    _POOL_CACHE.pop(next(iter(_POOL_CACHE)))
            if len(tail_df) < 500:
                out.skipped[d.name] = f"only {len(tail_df)} bars"
                continue
            loaded[sym] = tail_df
            full[sym] = full_df
            coarse[sym] = coarse_s
            got = True
            break
        if not got and sym not in out.skipped and not any(k.startswith(sym) for k in out.skipped):
            out.skipped[sym] = f"no stored file at the {target}-minute bar size"

    ordered = {mine: df, **loaded}
    kept, dropped = dedupe_independent_frames(ordered, corr_threshold, full_frames={mine: df, **full}, coarse=coarse)
    out.duplicates.update(dropped)
    out.distinct_total = len(kept)
    for sym, f in kept.items():
        if sym == mine:
            continue
        if len(out.frames) >= limit:
            out.skipped[sym] = f"beyond the {POOL_MAX_MARKETS}-market cap"
            continue
        out.frames[sym] = f
        if risk_factory is not None:
            try:
                out.risks[sym] = risk_factory(sym)
            except Exception:  # noqa: BLE001
                out.frames.pop(sym, None)
                out.skipped[sym] = "could not build its risk settings"
    return out
