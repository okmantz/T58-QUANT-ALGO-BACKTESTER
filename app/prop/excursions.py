"""Per-trade excursion (MAE / MFE) arrays for the prop-account simulators.

Why this exists (2026-10 audit, weak point 2): floating-drawdown and intraday-trailing
rules need each trade's worst and best open P&L, not just its closed P&L. The Monte
Carlo engine used to call ``simulate_account`` without them, so a trade that dipped $900
and closed +$200 looked like a +$200 trade to a firm that trails drawdown on equity.

``trade_excursions(trades)`` returns two aligned lists in DOLLARS:
    mae <= 0   adverse excursion (most negative open P&L while the trade was open)
    mfe >= 0   favorable excursion
computed exactly like ``Trade.pnl`` is (price difference x size x direction), so the units
agree with the account. Trades that carry no ``worst_price`` / ``mfe_price`` (resting-order
fills, the vectorized stage-1 path, imported trade lists) fall back to the CLOSED-P&L proxy
``mae = min(pnl, 0)``, ``mfe = max(pnl, 0)``; the count of such trades is returned so
callers can disclose it instead of pretending the numbers are measured.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class ExcursionSet:
    mae: list
    mfe: list
    n_measured: int
    n_proxied: int

    @property
    def measured_fraction(self) -> float:
        n = self.n_measured + self.n_proxied
        return self.n_measured / n if n else 1.0


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def trade_excursions(trades: Sequence) -> ExcursionSet:
    mae: list = []
    mfe: list = []
    measured = proxied = 0
    for t in trades:
        pnl = float(getattr(t, "pnl", 0.0) or 0.0)
        if not math.isfinite(pnl):
            pnl = 0.0
        entry = getattr(t, "entry_price", None)
        size = getattr(t, "size", None)
        direction = getattr(t, "direction", None)
        worst = getattr(t, "worst_price", None)
        best = getattr(t, "mfe_price", None)
        if _finite(entry) and _finite(size) and direction in (1, -1) and _finite(worst) and _finite(best):
            a = (float(worst) - float(entry)) * float(direction) * float(size)
            f = (float(best) - float(entry)) * float(direction) * float(size)
            # excursions bracket the realized outcome (commission can push pnl slightly outside)
            a = min(a, 0.0, pnl)
            f = max(f, 0.0, pnl)
            measured += 1
        else:
            a, f = min(pnl, 0.0), max(pnl, 0.0)
            proxied += 1
        mae.append(a)
        mfe.append(f)
    return ExcursionSet(mae, mfe, measured, proxied)
