# Global Events

Sidebar page (`/events/`) directly under **AI Assistant**: a cache-first world-events monitor.

- **Architecture:** a background poller (started lazily on first page visit; sleeps after ~20 min unseen; never started under pytest or when `T58_GLOBAL_EVENTS_POLLER=0`) fetches RSS/GDELT-style feeds, USGS, NASA EONET, GDACS, Stooq quotes and the economic calendar, classifies headlines, and writes to SQLite (WAL). Every HTTP endpoint is a read-only query of that cache, so page loads never wait on the network.
- **Endpoints:** `/events/api/bootstrap` (ETag), `headlines`, `events`, `geo`, `health`, `POST refresh` (rate-limited), `POST brief`, `snapshot`.
- **Map:** offline canvas using Natural Earth land (public domain, `static/global-events/land.json`); cached land layer, rAF-coalesced redraws, DPR-aware.
- **X (Twitter) section:** add usernames in the page; the latest posts for each are fetched with the Bearer token saved under API Keys → X (`app/global_events/x_feed.py`, endpoints `/events/api/x`, `/events/api/x/accounts`, `/events/api/x/accounts/remove`). Reading posts needs a paid X API plan; with no/invalid token the section shows a message and the rest of the page is unaffected. Posts are cached and rate-limited; rendered via `textContent`.
- **Safety:** all DOM written via `textContent` (no `innerHTML`); feed failures never raise.
- **Config:** `T58_GLOBAL_EVENTS_DB` overrides the DB path (default `data/global_events/global_events.db`).
- **Theater Heat** shows "calibrating" until ~4 days of history exist.
- **Caveats:** feed URLs are catalogued in `app/global_events/sources.py` and should be checked against provider terms before commercial distribution (notably Stooq quotes).
