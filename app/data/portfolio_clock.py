"""
Master clock for multi-instrument (portfolio) backtests.

``app.data.pairs`` aligns TWO instruments by merging the second one's close into the first one's frame as-of each
bar. A portfolio account needs the generalisation: ONE clock that every instrument is stepped against, so equity
can be marked to market across all open positions at every point in time.

``build_portfolio_clock`` takes each leg's (already timeframe-aligned) frame and returns:

  * ``master``      the sorted union of every leg's bar timestamps (UTC instants, as datetime64[ns]);
  * ``bar_at[k][m]``   leg ``k``'s own bar index at master step ``m``, or -1 when that leg has NO bar there;
  * ``last_bar[k][m]`` the most recent bar of leg ``k`` at or before step ``m`` (-1 before its first bar) --
                    the as-of mark. It is strictly backward-looking: a step can never see a bar that has not
                    printed yet, the same no-lookahead guarantee ``pairs.merge_pair_series`` gives.
  * ``leg_step[k][b]`` the master step of leg ``k``'s bar ``b`` (inverse of ``bar_at``).

Different sessions and holidays need no special casing: when a market is closed its leg simply has no bar on
that step. Nothing can trigger for it (a closed market cannot fill a stop), its open position is carried at the
last known close, and the other legs keep trading. The clock also reports how much of the master timeline each
leg actually covers and the longest stretch it is stale, so a portfolio of, say, a CME future and a 24/7 crypto
pair is not silently scored as if both traded every minute.

Timestamps follow the engine's convention: tz-aware columns are converted to UTC instants, naive columns are
taken to already be UTC.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class PortfolioClockError(ValueError):
    """Raised when instruments cannot be placed on one clock."""


def _utc_ns(ts: pd.Series) -> np.ndarray:
    """Leg timestamps as datetime64[ns] UTC instants (tz-aware -> UTC, naive -> taken as UTC)."""
    s = pd.to_datetime(ts)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.to_numpy(dtype="datetime64[ns]")


@dataclass
class LegCoverage:
    name: str
    bars: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    coverage: float                 # fraction of master steps on which this leg has a bar
    longest_gap_hours: float        # longest stretch (between two of ITS OWN bars) with no bar


@dataclass
class PortfolioClock:
    master: np.ndarray                              # datetime64[ns], strictly increasing
    bar_at: list = field(default_factory=list)      # per leg: int64[len(master)], -1 where the leg has no bar
    last_bar: list = field(default_factory=list)    # per leg: int64[len(master)], as-of bar index (-1 before first)
    leg_step: list = field(default_factory=list)    # per leg: int64[n_leg_bars], master step of each leg bar
    names: list = field(default_factory=list)
    coverage: list = field(default_factory=list)    # list[LegCoverage]
    notes: list = field(default_factory=list)
    window: str = "union"

    def __len__(self) -> int:
        return int(len(self.master))

    @property
    def n_legs(self) -> int:
        return len(self.names)

    def report(self) -> dict:
        return {
            "window": self.window,
            "master_steps": len(self),
            "start": str(pd.Timestamp(self.master[0])) if len(self) else None,
            "end": str(pd.Timestamp(self.master[-1])) if len(self) else None,
            "legs": [
                {"name": c.name, "bars": c.bars, "coverage": round(c.coverage, 4),
                 "first": str(c.first) if c.first is not None else None,
                 "last": str(c.last) if c.last is not None else None,
                 "longest_gap_hours": round(c.longest_gap_hours, 2)}
                for c in self.coverage
            ],
            "notes": list(self.notes),
        }


def build_portfolio_clock(
    frames: list,
    names: list | None = None,
    window: str = "union",
) -> PortfolioClock:
    """Put every leg's frame on one master clock.

    window="union" (default): the master spans the earliest to the latest bar of ANY leg; a leg with no data
        for part of it just does not trade there. One leg => the master IS that leg's own timestamps (a
        single-leg portfolio steps exactly the bars ``run_backtest`` would).
    window="overlap": the master is trimmed to the span every leg covers, so all legs are live for the whole
        run -- cleaner for comparing portfolios, at the cost of discarding non-overlapping history.
    """
    if window not in ("union", "overlap"):
        raise PortfolioClockError("window must be 'union' or 'overlap'")
    if not frames:
        raise PortfolioClockError("A portfolio clock needs at least one instrument frame.")
    names = list(names) if names is not None else [f"leg{i + 1}" for i in range(len(frames))]
    if len(names) != len(frames):
        raise PortfolioClockError("names and frames must have the same length.")

    leg_ts = []
    for name, df in zip(names, frames):
        if df is None or "timestamp" not in df.columns:
            raise PortfolioClockError(f"Leg '{name}' has no 'timestamp' column to align on.")
        ts = _utc_ns(df["timestamp"])
        if len(ts) > 1:
            d = np.diff(ts.astype("int64"))
            if (d == 0).any():
                raise PortfolioClockError(
                    f"Leg '{name}' has duplicate timestamps; its bars cannot be placed on a shared clock. "
                    "Drop or aggregate the duplicates first.")
            if (d < 0).any():
                raise PortfolioClockError(f"Leg '{name}' timestamps are not sorted ascending.")
        leg_ts.append(ts)

    notes: list[str] = []
    master = leg_ts[0]
    for ts in leg_ts[1:]:
        master = np.union1d(master, ts)
    if window == "overlap" and len(leg_ts) > 1:
        lo = max(ts[0] for ts in leg_ts if len(ts))
        hi = min(ts[-1] for ts in leg_ts if len(ts))
        if lo > hi:
            raise PortfolioClockError("The legs' data ranges do not overlap at all; use window='union'.")
        master = master[(master >= lo) & (master <= hi)]
        notes.append(f"Trimmed to the span all legs cover ({pd.Timestamp(lo)} -> {pd.Timestamp(hi)}).")
    if len(master) == 0:
        raise PortfolioClockError("The master clock is empty -- no bars to step.")

    M = len(master)
    clock = PortfolioClock(master=master, names=names, window=window)
    for name, ts in zip(names, leg_ts):
        step = np.searchsorted(master, ts).astype(np.int64)
        in_window = (step < M) & (master[np.minimum(step, M - 1)] == ts) if len(ts) else np.zeros(0, dtype=bool)
        bar_at = np.full(M, -1, dtype=np.int64)
        valid_bars = np.arange(len(ts), dtype=np.int64)[in_window]
        bar_at[step[in_window]] = valid_bars
        last = np.full(M, -1, dtype=np.int64)
        last[step[in_window]] = valid_bars
        last = np.maximum.accumulate(last)
        # leg_step maps each leg bar to its master step; bars trimmed away by window='overlap' map to -1
        leg_step = np.where(in_window, step, -1).astype(np.int64)
        clock.bar_at.append(bar_at)
        clock.last_bar.append(last)
        clock.leg_step.append(leg_step)

        n_here = int((bar_at >= 0).sum())
        gap_h = 0.0
        if len(ts) > 1:
            gap_h = float(np.diff(ts.astype("int64")).max() / 3.6e12)   # ns -> hours
        clock.coverage.append(LegCoverage(
            name=name, bars=len(ts),
            first=pd.Timestamp(ts[0]) if len(ts) else None, last=pd.Timestamp(ts[-1]) if len(ts) else None,
            coverage=n_here / M, longest_gap_hours=gap_h,
        ))

    # Honest disclosure when the instruments do not share a rhythm (different sessions / holidays / timeframes).
    if len(frames) > 1:
        covs = [c.coverage for c in clock.coverage]
        if max(covs) - min(covs) > 0.25:
            sparse = [c.name for c in clock.coverage if c.coverage < max(covs) - 0.25]
            notes.append(
                "Instruments trade on different calendars/timeframes: " + ", ".join(sparse) +
                " have bars on well under the busiest leg's share of master steps. While a leg's market is closed "
                "(or its bar has not printed) nothing can trigger for it and its open position is marked at its "
                "last known close.")
        firsts = [c.first for c in clock.coverage if c.first is not None]
        lasts = [c.last for c in clock.coverage if c.last is not None]
        if firsts and lasts and (max(firsts) - min(firsts) > pd.Timedelta(days=1) or max(lasts) - min(lasts) > pd.Timedelta(days=1)):
            notes.append("The legs' data ranges differ; a leg only trades while its own data exists.")
    clock.notes = notes
    return clock
