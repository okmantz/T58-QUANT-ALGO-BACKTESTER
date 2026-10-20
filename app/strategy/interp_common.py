"""
Shared pieces for the Pine and MQL5 interpreters (app.strategy.pine_interp, app.strategy.mql5_interp).

Both interpreters walk the script once per bar, keep a tiny simulated broker so scripts that look at their own
position behave, and finish by handing the engine the same thing every other adapter hands it: a +1/0/-1 signal
series plus per-bar stop/target distances. The backtest engine still does the real fills, stops and costs.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

NAN = float("nan")


def isnan(x) -> bool:
    return isinstance(x, float) and x != x


def time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """The bar timestamps: the engine's `timestamp` column when present, else a DatetimeIndex."""
    if "timestamp" in df.columns:
        return pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
    return pd.DatetimeIndex(df.index)


def guess_mintick(df: pd.DataFrame) -> float:
    """Smallest price step in the data (0.0001 for FX, 0.25 for ES, 0.01 for stocks ...), found from the decimals of a
    sample of closes. Used for Pine's syminfo.mintick and MQL5's _Point. Never raises."""
    try:
        sample = df["close"].dropna().iloc[:: max(1, len(df) // 400)].head(400).astype(float).to_numpy()
        if len(sample) == 0:
            return 0.01
        q = np.round(sample * 4, 6)
        if np.all(np.abs(q - np.round(q)) < 1e-4) and np.any(np.abs(np.round(sample * 100, 6) - np.round(sample * 100)) > 1e-4):
            return 0.25
        for dec in range(0, 9):
            scaled = np.round(sample * (10 ** dec), 6)
            if np.all(np.abs(scaled - np.round(scaled)) < 1e-4):
                return 10.0 ** (-dec)
        return 0.0001
    except Exception:  # noqa: BLE001
        return 0.01


def bar_minutes(df: pd.DataFrame) -> float:
    try:
        d = np.diff(time_index(df).values.astype("datetime64[s]").astype("int64"))
        d = d[d > 0]
        return float(np.median(d)) / 60.0 if len(d) else 1.0
    except Exception:  # noqa: BLE001
        return 1.0


def parse_tf_minutes(label) -> float | None:
    """'15', '15m', '1h', '60', '4H', 'D', '1D', 'W', 'M', 'PERIOD_H1', 'H4', 'M15' -> minutes (None if unknown/empty)."""
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    u = s.upper().replace("PERIOD_", "")
    m = re.fullmatch(r"([MHDW])(\d+)", u)            # MQL5 style: M15 H4 D1 W1 MN1
    if m:
        unit, n = m.group(1), int(m.group(2))
        return n * {"M": 1, "H": 60, "D": 1440, "W": 10080}[unit]
    if u in ("MN1", "MN"):
        return 43200.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([A-Z]*)", u)
    if m:
        n = float(m.group(1))
        unit = m.group(2)
        if unit in ("", "M", "MIN", "MINUTE", "MINUTES"):
            return n
        if unit in ("S", "SEC"):
            return n / 60.0
        if unit in ("H", "HR", "HOUR", "HOURS"):
            return n * 60
        if unit in ("D", "DAY", "DAYS"):
            return n * 1440
        if unit in ("W", "WK", "WEEK"):
            return n * 10080
        if unit in ("MO", "MON", "MONTH"):
            return n * 43200
    if u in ("D", "1D"):
        return 1440.0
    if u == "W":
        return 10080.0
    if u == "M":      # Pine 'M' alone = 1 month (a bare number is minutes)
        return 43200.0
    return None


def tf_to_resample_label(minutes: float) -> str:
    if minutes >= 43200 and abs(minutes % 43200) < 1e-6:
        return f"{int(minutes // 43200)}M"
    if minutes >= 10080 and abs(minutes % 10080) < 1e-6:
        return f"{int(minutes // 10080)}w"
    if minutes >= 1440 and abs(minutes % 1440) < 1e-6:
        return f"{int(minutes // 1440)}d"
    if minutes >= 60 and abs(minutes % 60) < 1e-6:
        return f"{int(minutes // 60)}h"
    return f"{int(round(minutes))}m"


def resample_to(df: pd.DataFrame, minutes: float) -> pd.DataFrame:
    """Resample to `minutes` using the app's own resampler (timestamp column in, timestamp column out)."""
    from app.data.timeframe_resample import resample_ohlcv
    work = df if "timestamp" in df.columns else df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
    return resample_ohlcv(work, tf_to_resample_label(minutes))


def completed_map(base_index: pd.DatetimeIndex, base_minutes: float, htf_index: pd.DatetimeIndex, htf_minutes: float) -> np.ndarray:
    """For every base bar i, the position of the last HTF bar that is COMPLETE once base bar i has closed (-1 if none).
    An HTF bar stamped T covers [T, T + htf_minutes). It is complete when base_bar_close >= T + htf_minutes. This is the
    look-ahead-safe alignment: a still-forming HTF bar is never visible. lead_bars=1 asks 'complete at the OPEN of
    base bar i+1' (== close of bar i), which is the same thing for contiguous data."""
    base_close = (base_index.values.astype("datetime64[s]").astype("int64") + int(base_minutes * 60))
    htf_end = pd.DatetimeIndex(htf_index).values.astype("datetime64[s]").astype("int64") + int(htf_minutes * 60)
    # searchsorted: number of HTF bars whose end <= base close
    pos = np.searchsorted(htf_end, base_close, side="right") - 1
    return pos.astype(int)


class SimBroker:
    """A deliberately small broker: one net position, market fills at the next bar's open, and stop/target hit tests from
    the bar's high/low. Its job is to give the script an honest `position_size` / PositionsTotal(); the real simulation is
    app.backtest.engine.run_backtest."""

    def __init__(self, df: pd.DataFrame, default_qty: float = 1.0, pyramiding: int = 0):
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        n = len(df)
        self.n = n
        self.signal = np.zeros(n, dtype=int)
        self.sl_dist = np.full(n, np.nan)
        self.tp_dist = np.full(n, np.nan)
        self.default_qty = float(default_qty)
        self.pyramiding = int(pyramiding)
        self.pos = 0            # net units, signed
        self.avg = NAN
        self.n_entries = 0
        self.pending: list[tuple] = []     # ("entry", dir, qty) | ("close",) applied at next open
        self.stop_price = NAN
        self.limit_price = NAN
        self.stop_ticks = NAN
        self.limit_ticks = NAN
        self.closed = []        # (dir, entry, exit, qty)
        self.netprofit = 0.0
        self.last_entry_bar = -1
        self.exit_ids = {}

    # -- called by the interpreters -----------------------------------------------------------------------
    def entry(self, direction: int, qty: float | None = None):
        self.pending.append(("entry", int(direction), float(qty) if qty else self.default_qty))

    def close(self):
        self.pending.append(("close",))

    def set_exit(self, stop=None, limit=None, stop_ticks=None, limit_ticks=None):
        if stop is not None and not isnan(stop):
            self.stop_price = float(stop)
        if limit is not None and not isnan(limit):
            self.limit_price = float(limit)
        if stop_ticks is not None and not isnan(stop_ticks):
            self.stop_ticks = float(stop_ticks)
        if limit_ticks is not None and not isnan(limit_ticks):
            self.limit_ticks = float(limit_ticks)

    def _flat(self, price: float):
        if self.pos != 0 and not isnan(self.avg):
            d = 1 if self.pos > 0 else -1
            self.netprofit += d * (price - self.avg) * abs(self.pos)
            self.closed.append((d, self.avg, price, abs(self.pos)))
        self.pos = 0
        self.avg = NAN
        self.n_entries = 0
        self.stop_price = self.limit_price = self.stop_ticks = self.limit_ticks = NAN

    def begin_bar(self, i: int, mintick: float):
        """Apply orders queued on the previous bar at this bar's open, then test stops/targets inside this bar."""
        if i > 0 and self.pending:
            orders, self.pending = self.pending, []
            px = self.o[i]
            for od in orders:
                if od[0] == "close":
                    self._flat(px)
                else:
                    d, q = od[1], od[2]
                    cur = 1 if self.pos > 0 else (-1 if self.pos < 0 else 0)
                    if cur == d:
                        if self.pyramiding > 0 and self.n_entries <= self.pyramiding:
                            tot = abs(self.pos) + q
                            self.avg = (self.avg * abs(self.pos) + px * q) / tot
                            self.pos = d * tot
                            self.n_entries += 1
                        continue
                    if cur != 0:
                        keep = (self.stop_price, self.limit_price, self.stop_ticks, self.limit_ticks)
                        self._flat(px)
                        self.stop_price, self.limit_price, self.stop_ticks, self.limit_ticks = keep
                    self.pos = d * q
                    self.avg = px
                    self.n_entries = 1
                    self.last_entry_bar = i
        self._check_exits(i, mintick)

    def _levels(self, mintick: float):
        d = 1 if self.pos > 0 else -1
        stop = self.stop_price
        lim = self.limit_price
        if isnan(stop) and not isnan(self.stop_ticks):
            stop = self.avg - d * self.stop_ticks * mintick
        if isnan(lim) and not isnan(self.limit_ticks):
            lim = self.avg + d * self.limit_ticks * mintick
        return stop, lim

    def _check_exits(self, i: int, mintick: float):
        if self.pos == 0:
            return
        stop, lim = self._levels(mintick)
        d = 1 if self.pos > 0 else -1
        hit_stop = (not isnan(stop)) and ((d > 0 and self.l[i] <= stop) or (d < 0 and self.h[i] >= stop))
        hit_lim = (not isnan(lim)) and ((d > 0 and self.h[i] >= lim) or (d < 0 and self.l[i] <= lim))
        if hit_stop:                    # stop wins an ambiguous bar (worst case)
            self._flat(stop)
        elif hit_lim:
            self._flat(lim)

    def end_bar(self, i: int, mintick: float):
        target = 1 if self.pos > 0 else (-1 if self.pos < 0 else 0)
        for od in self.pending:
            target = 0 if od[0] == "close" else od[1]
        self.signal[i] = target
        if target != 0:
            # the script's absolute stop/target prices are measured from the price the trade will actually fill at
            ref = self.o[i + 1] if i + 1 < self.n else self.c[i]
            # while a position is open and the entry has not changed, keep the last known distances
            stop, lim = (self._levels(mintick) if self.pos != 0 else (self.stop_price, self.limit_price))
            if isnan(stop) and not isnan(self.stop_ticks):
                stop = ref - target * self.stop_ticks * mintick
            if isnan(lim) and not isnan(self.limit_ticks):
                lim = ref + target * self.limit_ticks * mintick
            if not isnan(stop):
                dist = (ref - stop) * target
                if dist > 0:
                    self.sl_dist[i] = dist
            if not isnan(lim):
                dist = (lim - ref) * target
                if dist > 0:
                    self.tp_dist[i] = dist

    # -- results ---------------------------------------------------------------------------------------
    def distances(self, index):
        sl = pd.Series(self.sl_dist, index=index)
        tp = pd.Series(self.tp_dist, index=index)
        sl = sl.ffill() if sl.notna().any() else None
        tp = tp.ffill() if tp.notna().any() else None
        return sl, tp
