"""
Shared helper for pages with a "fetch data from London Strategic Edge" card.

Mirrors app.web.alpaca_shared exactly, one key instead of two (London
Strategic Edge has no separate secret key -- see
app.data.london_strategic_edge_source's module docstring).
"""
from __future__ import annotations

from app.accounts import api_keys
from app.data.london_strategic_edge_source import ASSET_CLASSES, TIMEFRAME_CHOICES


def lse_template_context() -> dict:
    """Shared context injected wherever a Market Data card is rendered --
    the dropdown lists plus whether a key is already saved, so the form
    can pre-check "save key" and never echo a saved secret back into the
    page source."""
    return {
        "lse_asset_classes": ASSET_CLASSES,
        "lse_timeframes": TIMEFRAME_CHOICES,
        "lse_has_saved_key": bool(api_keys.load_settings().london_strategic_edge_key),
    }
