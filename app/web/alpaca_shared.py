"""
Shared helper for pages with a "fetch data from Alpaca" card.

Lives outside server.py so blueprint modules (extra_routes.py,
hedge_fund_routes.py, risk_sweep_routes.py) can use it without importing
server.py -- these blueprints deliberately have no import-time dependency
on server.py (server.py is the one that imports blueprints, not the
reverse; see each blueprint module's own docstring).
"""
from __future__ import annotations

from app.data import alpaca_credentials
from app.data.alpaca_source import ASSET_CLASSES, ADJUSTMENT_CHOICES, FEED_CHOICES, TIMEFRAME_LABELS


def alpaca_template_context() -> dict:
    """Shared context injected wherever a Market Data card is rendered --
    the dropdown/option lists plus whether keys are already saved, so the
    form can pre-check "save keys" and (for privacy) never echo a saved
    secret back into the page source."""
    return {
        "alpaca_asset_classes": ASSET_CLASSES,
        "alpaca_timeframes": TIMEFRAME_LABELS,
        "alpaca_feeds": FEED_CHOICES,
        "alpaca_adjustments": ADJUSTMENT_CHOICES,
        "alpaca_has_saved_keys": alpaca_credentials.has_saved_credentials(),
    }
