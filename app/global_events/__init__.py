"""Global Events -- a cache-first market / geopolitical monitor (page: /events, sidebar: under AI Assistant).

Architecture (clean-room, own code; see docs/GLOBAL_EVENTS.md):

    poller thread --fetch--> sources.* --normalize--> classify.* --write--> SQLite (store)
    browser <--JSON-- routes (/events/api/*) <--read only-- store

Request handlers only ever READ the local store; they never call an upstream service, so the page is instant,
offline-tolerant and rate-limit-safe. The poller is a single low-priority daemon thread that is started lazily on the
first visit to the page (never at app start-up or import) and goes back to sleep when nobody has the page open.
"""
from __future__ import annotations

__all__ = ["get_context_snapshot"]


def get_context_snapshot(max_headlines: int = 8) -> dict:
    """Compact, store-only snapshot for the AI Assistant ("anything I should worry about before NY open?").

    Never touches the network; carries freshness stamps so the assistant can say how old each piece is."""
    from . import store

    return store.context_snapshot(max_headlines=max_headlines)
