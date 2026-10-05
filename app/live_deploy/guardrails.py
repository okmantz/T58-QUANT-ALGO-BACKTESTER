"""
Futures-prop-safe guardrail defaults for live deployment (v7, P0-2).

Before v7, both Deploy Live entry points (the desktop tab and the web
``/deploy-live/start`` route) constructed a bare ``PropRules(...)`` with
every protective default OFF -- no news blackout, weekend holding
allowed, hedging allowed, no per-order lot cap -- and an inline comment
admitted the widgets "aren't exposed yet". A live session could therefore
blow a prop evaluation on a pure rule technicality (news trade,
weekend gap, hedge) even with perfect strategy performance.

This module is the single source of truth for the safe defaults both
UIs now enforce:

* news blackout ON, with a default window set (the two highest-impact
  daily US release slots),
* weekend_hold_allowed=False,
* hedging_allowed=False,
* max_lot_size set (broker-native contracts/lots per single order),
* max-drawdown halt enforced by LiveExecutionSession on every poll.

Nothing here touches backtest/validation gates.
"""
from __future__ import annotations

import re

from app.prop.simulator import PropRules

# Default news-blackout windows, in SERVER/BROKER time (what
# execution_engine's BlackoutWindow matching runs against). Covers the two
# highest-impact recurring US release slots: 8:30am ET economic data
# (NFP, CPI, PPI, jobless claims...) and the 1-2pm ET FOMC window,
# expressed here in CT (Tradovate and most US-futures broker servers run
# on CT). One window per line; the desktop entry widget accepts commas
# too (see normalize_blackout_text). Adjust to your broker's timezone.
DEFAULT_NEWS_BLACKOUT_WINDOWS = "07:25-07:40\n12:55-13:10"

# Hard cap on a single order, in BROKER-NATIVE units (contracts on
# Tradovate, lots on MT5/cTrader/TradeLocker/DXtrade) -- applied AFTER the
# sizing-unit -> broker-quantity conversion in LiveExecutionSession, so
# "5" means 5 contracts/lots, not 5 sizing units. Futures-prop-safe:
# evals are won with small size; nothing in an eval needs >5 per ticket.
DEFAULT_MAX_LOT_SIZE = 5.0

# FX standard: 1 MT5/cTrader lot = 100,000 base-currency units, and the
# sizing "units" domain is exactly base-currency units for FX
# (PnL = units x price_move), so units_per_lot = 100,000 converts
# correctly. For futures, 1 broker lot/contract = contract_size units.
FX_STANDARD_UNITS_PER_LOT = 100_000.0

_MONTH_CODE_RE = re.compile(r"^(.*?)([FGHJKMNQUVXZ])(\d{1,2})?$", re.IGNORECASE)


def default_news_blackout_entry_text() -> str:
    """Single-line (comma-separated) form of the default windows, for the
    desktop tab's one-line entry widget."""
    return DEFAULT_NEWS_BLACKOUT_WINDOWS.replace("\n", ", ")


def normalize_blackout_text(raw: str) -> str:
    """Accept commas/semicolons as well as newlines between windows and
    return the canonical newline-joined form that
    execution_engine.parse_blackout_windows expects. Unparseable lines
    are still skipped by the parser, never fatal."""
    parts = re.split(r"[,\n;]+", raw or "")
    return "\n".join(p.strip() for p in parts if p.strip())


def contract_size_for_symbol(symbol: str) -> float | None:
    """Best-effort contract_size lookup for a broker symbol.

    Tries the bare root symbol against app.data.instrument_specs (case
    insensitive), then strips a trailing futures month-code + year
    (e.g. "ESZ25" -> "ES", "MNQ" stays "MNQ") and retries. Returns None
    (never raises) when the symbol isn't a known futures root -- the
    caller must then either ask the user or refuse to start; it must
    NOT invent a number (see to_broker_qty's no-guessing rule).
    """
    if not symbol:
        return None
    try:
        from app.data.instrument_specs import get_instrument_spec
    except Exception:
        return None
    root = str(symbol).strip().upper()
    spec = get_instrument_spec(root)
    if spec is not None:
        return float(spec.contract_size)
    m = _MONTH_CODE_RE.match(root)
    if m:
        spec = get_instrument_spec(m.group(1))
        if spec is not None:
            return float(spec.contract_size)
    return None


def resolve_units_per_lot(contract_size: float | None, explicit: float | None = None) -> float:
    """Resolve the units_per_lot conversion factor.

    An explicitly provided positive value always wins. Otherwise futures
    (known contract_size) use contract_size -- 1 broker lot/contract is
    contract_size sizing units -- and anything else falls back to the FX
    standard 100,000. Never returns <= 0.
    """
    if explicit and explicit > 0:
        return float(explicit)
    if contract_size and contract_size > 0:
        return float(contract_size)
    return FX_STANDARD_UNITS_PER_LOT


def futures_prop_safe_rules(
    account_size: float,
    *,
    news_blackout_windows: str | None = None,
    weekend_hold_allowed: bool = False,
    hedging_allowed: bool = False,
    max_lot_size: float | None = DEFAULT_MAX_LOT_SIZE,
    daily_loss_limit_pct: float | None = None,
    max_drawdown_pct: float | None = None,
) -> PropRules:
    """Build a PropRules with the futures-prop-safe guardrails enforced.

    news_blackout_windows=None selects the default window set above;
    pass an explicit string (already normalized) to override. The two
    booleans default to the SAFE value (False); callers pass True only
    from an explicit user opt-in widget. max_lot_size is broker-native
    (contracts/lots per order), not sizing units.
    """
    kwargs: dict = dict(
        account_size=account_size,
        news_blackout_windows=normalize_blackout_text(
            DEFAULT_NEWS_BLACKOUT_WINDOWS if news_blackout_windows is None else news_blackout_windows
        ),
        weekend_hold_allowed=weekend_hold_allowed,
        hedging_allowed=hedging_allowed,
        max_lot_size=max_lot_size,
    )
    if daily_loss_limit_pct is not None:
        kwargs["daily_loss_limit_pct"] = daily_loss_limit_pct
    if max_drawdown_pct is not None:
        kwargs["max_drawdown_pct"] = max_drawdown_pct
    return PropRules(**kwargs)


def enforced_guardrails_summary(rules: PropRules) -> str:
    """One human-readable summary of the enforced guardrails, for the
    on-screen notice both UIs must show before a session starts."""
    windows = (getattr(rules, "news_blackout_windows", "") or "").replace("\n", ", ")
    return (
        "ENFORCED PROP GUARDRAILS -- news blackout ON "
        f"({windows or 'no windows?!'}; server time); "
        f"weekend hold {'ALLOWED' if getattr(rules, 'weekend_hold_allowed', True) else 'BLOCKED (flatten Friday)'}; "
        f"hedging {'ALLOWED' if getattr(rules, 'hedging_allowed', True) else 'BLOCKED'}; "
        f"max {getattr(rules, 'max_lot_size', None)} contracts/lots per order; "
        f"max-drawdown halt at {getattr(rules, 'max_drawdown_pct', '?')}% "
        f"({getattr(rules, 'drawdown_type', '?')}) -- breach flattens everything and stops the session."
    )
