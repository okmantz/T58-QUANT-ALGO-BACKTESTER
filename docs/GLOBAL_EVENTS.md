# Global Events

Sidebar page (`/events/`) directly under **AI Assistant**: a cache-first world-events monitor.

- **Architecture:** a background poller (started lazily on first page visit; sleeps after ~20 min unseen; never started under pytest or when `T58_GLOBAL_EVENTS_POLLER=0`) fetches RSS/GDELT-style feeds, USGS, NASA EONET, GDACS, Stooq quotes and the economic calendar, classifies headlines, and writes to SQLite (WAL). Every HTTP endpoint is a read-only query of that cache, so page loads never wait on the network.
- **Endpoints:** `/events/api/bootstrap` (ETag), `headlines`, `events`, `geo`, `health`, `intel` (ETag; market intelligence below), `POST refresh` (rate-limited), `POST brief`, `snapshot`.
- **Map:** offline canvas using Natural Earth land (public domain, `static/global-events/land.json`); cached land layer, rAF-coalesced redraws, DPR-aware.
- **X (Twitter) section:** add usernames in the page; the latest posts for each are fetched with the Bearer token saved under API Keys → X (`app/global_events/x_feed.py`, endpoints `/events/api/x`, `/events/api/x/accounts`, `/events/api/x/accounts/remove`). Reading posts needs a paid X API plan; with no/invalid token the section shows a message and the rest of the page is unaffected. Posts are cached and rate-limited; rendered via `textContent`.
- **Safety:** all DOM written via `textContent` (no `innerHTML`); feed failures never raise.
- **Config:** `T58_GLOBAL_EVENTS_DB` overrides the DB path (default `data/global_events/global_events.db`).
- **Theater Heat** shows "calibrating" until ~4 days of history exist.
- **Caveats:** feed URLs are catalogued in `app/global_events/sources.py` and should be checked against provider terms before commercial distribution (notably Stooq quotes).

## Market intelligence (`/events/api/intel`, `app/global_events/intel.py`)

Computed locally from stored **daily bars** (Stooq CSV, fetched by the poller every 6 hours into `ge_daily`, up to 400 sessions per symbol), the economic calendar and the
theater scores. Nothing is a prediction or a signal, and every label is printed with the rule and numbers that produced it.

| Panel | What it is |
|---|---|
| Market Regime Intelligence | Per instrument (ES, NQ, YM, RTY, GC, CL, DXY). Priority order: **Event-driven** (a high-impact release for that instrument within the next 24 h or the last 3 h) > **High-vol** (20-day realised vol at or above the 80th percentile of its own past year) > **Trending up/down** (Kaufman efficiency ratio over 20 days >= 0.35) > **Range-bound** (efficiency ratio <= 0.20 and vol percentile <= 60) > **Mixed**. Fewer than 25 bars: Unknown. Thresholds live in `intel.RULES`. |
| Calendar and releases | CPI, PPI, jobs, GDP, PCE, retail sales, ISM and central-bank events from the live calendar feed (current week), grouped Inflation / Jobs / Growth / Central banks. |
| Central banks | Next decision date for the Fed, ECB, BoE and BoJ from each bank's published 2026 schedule (`intel.CENTRAL_BANKS`) with a link to the bank's page; a live-calendar match adds forecast and previous. The list ends in December 2026 and must be extended from the banks' pages. |
| Sessions | Tokyo, London, New York cash and CME Globex, in each market's own time zone (DST-correct; needs the `tzdata` package on Windows). |
| Futures stats | Last, 1/5/20-day change, position in the 20-day range, 20-day realised vol, ATR14 as % of price. |
| Volatility regime | VIX bucket (calm < 15, normal 15-20, elevated 20-30, stress 30+), VIX year percentile, ES realised vol and the implied-minus-realised gap. |
| Cross-asset monitor | 1/5/20-day moves, a 1-day z-score against the last 60 days (>= 2 flagged), and a Risk-on / Risk-off tone: one point each over 5 days for S&P up, Nasdaq up, Russell up, VIX down, Dollar down; +3 or more Risk-on, -3 or less Risk-off. |
| Correlations | Pearson correlation of daily log returns over the last 60 sessions (blank below 30 shared days). |
| Geopolitics | The five hottest theaters, the instruments they touch and their lead headline. |

Until the first daily fetch completes the panels say so instead of showing blanks. Stooq terms apply (see Caveats).
