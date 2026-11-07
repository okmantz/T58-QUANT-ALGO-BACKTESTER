"""Day-aware path sampler shared by every resampling engine (2026-11-06 audit, "day-level bootstrap everywhere").

Daily-loss limits, consistency rules, minimum-trading-days and end-of-day trailing floors all depend on WITHIN-DAY structure: which
trades happened on the same session day. ``run_monte_carlo`` has resampled whole session days (``method="day_block_bootstrap"``,
the default) since the 2026-10-07 accuracy overhaul. The other engines that draw their own paths did not:

    * ``app.prop.survival_engine`` (Payout Probability / lifecycle funnel / reset chain) defaulted to the i.i.d. ``"bootstrap"``;
    * ``app.monte_carlo.bankroll`` defaulted to 5-TRADE blocks, which cut straight across trading days;
    * ``app.prop.scaling`` passed the Monte Carlo config's ``"day_block_bootstrap"`` to ``_resample_pnls``, which has no session
      labels and silently fell through to the i.i.d. branch.

An i.i.d. resample scatters a bad day's trades across many days, so a cluster of losers that really did land on ONE day (and really
did breach the daily loss limit) almost never co-occurs in the simulation. That flatters every pass / payout probability whenever
the daily-loss rule matters. ``PathSampler`` is the one place those engines now get their paths from.

    sampler = PathSampler(base_dates, rules, method, block_size)
    idx, sim_ds = sampler.draw(rng)          # sim_ds is a DayStructure for day methods, None otherwise
    sim_pnls = base_pnls[idx]
    simulate_account(sim_pnls, base_dates, rules, _day_structure=sim_ds if sim_ds is not None else base_structure)

For every non-day method the draw is exactly ``_resample_pnls``'s old draw (same RNG stream, same indices), so explicitly choosing
``method="bootstrap"`` / ``"block_bootstrap"`` / ``"shuffle"`` reproduces the pre-fix numbers byte for byte.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY_BLOCK_METHOD = "day_block_bootstrap"


class PathSampler:
    """Draws resampled trade-index paths, whole-session-day aware when ``method == "day_block_bootstrap"``.

    ``base_dates`` is the per-trade date list the engine hands ``simulate_account`` (any order; groups are built chronologically and
    stably, like ``app.monte_carlo.engine._day_groups``). Days are the same session days ``simulate_account`` derives from
    ``rules.session_timezone`` / ``rules.session_roll_hour``, so the resampled structure and the simulator agree on what a "day" is.
    """

    def __init__(self, base_dates, rules, method: str, block_size: int | None = None, day_block_days: int = 1):
        from app.monte_carlo.engine import _effective_block_size

        self.method = str(method)
        self.n = len(base_dates)
        self.block_days = max(1, int(day_block_days or 1))
        self.day_mode = self.method == DAY_BLOCK_METHOD and self.n > 0
        self._groups = self._gdates = self._tables = None
        self.n_days = 0
        if self.day_mode:
            self._build_groups(base_dates, rules)
        else:
            # Same block-size resolution the old _resample_pnls(...) call used (day count unknown -> trades-per-day floor of 2).
            class _Cfg:  # minimal stand-in; _effective_block_size only reads .block_size
                pass
            _cfg = _Cfg()
            _cfg.block_size = block_size
            self._eff_block = _effective_block_size(_cfg, self.n, None)

    def _build_groups(self, base_dates, rules) -> None:
        from app.data.trading_day import trading_days
        from app.monte_carlo.engine import _day_block_tables

        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in base_dates])
        if idx.isna().any():
            raise ValueError("PathSampler: base_dates contains NaT; cannot group trades into session days.")
        order = np.argsort(idx.asi8, kind="stable")
        tz = getattr(rules, "session_timezone", "America/Chicago") or "America/Chicago"
        roll = int(getattr(rules, "session_roll_hour", 17) if getattr(rules, "session_roll_hour", None) is not None else 17)
        day_dates = pd.DatetimeIndex(pd.to_datetime(list(trading_days(idx[order], tz=tz, roll_hour=roll))))
        change = np.ones(len(order), dtype=bool)
        change[1:] = day_dates.asi8[1:] != day_dates.asi8[:-1]
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], len(order))
        self._groups = [order[a:b].astype(np.int64) for a, b in zip(starts, ends)]
        self._gdates = [pd.Timestamp(day_dates[a]) for a in starts]
        self._tables = _day_block_tables(self._groups)
        self.n_days = len(self._groups)

    def draw(self, rng: np.random.Generator):
        """One resampled path: ``(trade_index_array, DayStructure | None)``. The array indexes the ORIGINAL per-trade arrays."""
        from app.monte_carlo.engine import _resample_day_blocks, _resample_indices

        if self.day_mode:
            return _resample_day_blocks(rng, self._groups, self._gdates, self.block_days, self._tables)
        return _resample_indices(rng, self.n, self.method, self._eff_block), None

    def path_dates(self, sim_ds) -> list:
        """Per-trade dates for a drawn day-block path (the SLOT day each trade now sits on), for callers whose simulator takes dates
        rather than a DayStructure. Non-day paths (``sim_ds is None``) are the caller's own base dates."""
        return [sim_ds.day_dates[i] for i in sim_ds.day_index_per_trade]
