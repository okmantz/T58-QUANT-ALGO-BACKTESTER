"""Per-run equity ledger check (2026-10-31 audit, section 3, fix 7).

The synthesis found final equity about $3,000 below ``50,000 + net profit - payouts`` for three no-reset strategies. A run must be able
to prove where every dollar of its final equity came from:

    final_equity == starting_equity + sum(trade P&L) + sum(logged equity adjustments)

where an *adjustment* is any change to the engine's equity that is NOT a settled trade: a payout withdrawal, a reset to a fresh account
after a breach or an attempt end, or the prop account sync. ``run_execution`` records each adjustment through one helper, so an equity
change that has no trade behind it (the shape of the leak the audit saw) makes the identity fail instead of passing unnoticed.

``reconcile_equity`` is a pure function (no engine state), ``EquityReconciliationError`` is what a failing run raises. The policy is the
``T58_EQUITY_RECON`` environment variable: ``strict`` (default, raise), ``warn`` (warning only) or ``off`` (record only).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

# One cent: equity is a float sum of up to ~1e5 trades, so allow float rounding but nothing a human would call a dollar.
TOLERANCE_DOLLARS = 0.01
_REL_TOLERANCE = 1e-9


class EquityReconciliationError(AssertionError):
    """The run's final equity is not explained by its trades plus its logged payouts/resets."""


@dataclass
class EquityReconciliation:
    starting_equity: float
    trade_pnl: float
    payouts: float                       # total withdrawn through payout events (positive number)
    reset_adjustments: float             # net effect of resets / account syncs that are not payouts (signed)
    expected_final_equity: float
    final_equity: float
    difference: float                    # final - expected (0 when the ledger balances)
    ok: bool
    n_trades: int = 0
    adjustments_by_kind: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def message(self) -> str:
        return (
            f"Equity ledger does not balance: final equity ${self.final_equity:,.2f} != starting ${self.starting_equity:,.2f} "
            f"+ trade P&L ${self.trade_pnl:,.2f} - payouts ${self.payouts:,.2f} + reset adjustments ${self.reset_adjustments:,.2f} "
            f"= ${self.expected_final_equity:,.2f} (difference ${self.difference:,.2f} over {self.n_trades} trades). "
            "Some equity change has no trade behind it."
        )


def reconcile_equity(
    starting_equity: float,
    trade_pnls: Iterable[float],
    adjustments: Sequence[tuple[str, float]],
    final_equity: float,
    payout_total: float | None = None,
) -> EquityReconciliation:
    """Balance ``starting + sum(pnl) + sum(adjustments) == final``.

    ``adjustments`` is ``[(kind, signed_amount), ...]``. ``payout_total`` (the sum of the run's payout events) is only used to split the
    reported adjustments into "payouts" and "other resets"; the identity itself does not depend on it."""
    pnls = [float(p) for p in trade_pnls]
    trade_pnl = float(sum(pnls))
    by_kind: dict[str, float] = {}
    for kind, amt in adjustments:
        by_kind[str(kind)] = by_kind.get(str(kind), 0.0) + float(amt)
    adj_total = float(sum(by_kind.values()))
    expected = float(starting_equity) + trade_pnl + adj_total
    diff = float(final_equity) - expected
    scale = max(1.0, abs(float(starting_equity)), abs(trade_pnl))
    ok = abs(diff) <= max(TOLERANCE_DOLLARS, _REL_TOLERANCE * scale)
    payouts = float(payout_total) if payout_total is not None else -sum(v for k, v in by_kind.items() if k == "payout")
    return EquityReconciliation(
        starting_equity=float(starting_equity), trade_pnl=trade_pnl, payouts=payouts,
        reset_adjustments=adj_total + payouts, expected_final_equity=expected, final_equity=float(final_equity),
        difference=diff, ok=ok, n_trades=len(pnls), adjustments_by_kind=by_kind,
    )


def recon_policy() -> str:
    v = (os.environ.get("T58_EQUITY_RECON", "strict") or "strict").strip().lower()
    return v if v in ("strict", "warn", "off") else "strict"


def enforce(recon: EquityReconciliation) -> None:
    """Apply the policy: raise (strict), warn, or stay silent (off). A balanced ledger never does anything."""
    if recon.ok:
        return
    policy = recon_policy()
    if policy == "strict":
        raise EquityReconciliationError(recon.message())
    if policy == "warn":
        import warnings
        warnings.warn(recon.message(), RuntimeWarning, stacklevel=3)
