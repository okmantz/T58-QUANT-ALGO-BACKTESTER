"""Background poller for Global Events: one daemon thread, started lazily, asleep when nobody is looking.

Why this cannot slow the app down:
  * it is never started at import or app start-up -- only when someone opens /events (``ensure_started``);
  * it only works while the page was seen in the last ``ACTIVE_WINDOW_S`` seconds (``touch`` is called by the API); after
    that it sleeps on a 30 s tick and does no network or database work at all;
  * all network I/O happens here, never in a request handler; per-request timeouts are short and bounded;
  * fetches run on a small bounded pool (``MAX_PARALLEL``) so a burst of RSS feeds cannot starve the rest of the app.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable

from . import sources, store

_log = logging.getLogger(__name__)

ACTIVE_WINDOW_S = 20 * 60
TICK_S = 5.0
IDLE_TICK_S = 30.0
MAX_PARALLEL = 6
MIN_MANUAL_REFRESH_S = 20.0


@dataclass
class Job:
    id: str
    interval_s: float
    run: Callable[[], None]
    next_due: float = 0.0
    running: bool = False
    jitter: float = field(default_factory=lambda: random.uniform(0.0, 4.0))


class Poller:
    def __init__(self) -> None:
        self.jobs: list[Job] = []
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_seen: float | None = None   # None = the page has never been opened
        self._last_manual = 0.0
        self._stop = threading.Event()
        self._pool: ThreadPoolExecutor | None = None
        self._build_jobs()

    # ------------------------------------------------------------------ jobs
    def _build_jobs(self) -> None:
        self.jobs = [
            Job("calendar", 900, self._job_calendar),
            Job("feeds", 300, self._job_feeds),
            Job("usgs", 900, lambda: self._job_geo("usgs", "USGS earthquakes")),
            Job("eonet", 1800, lambda: self._job_geo("eonet", "NASA EONET")),
            Job("gdacs", 1800, lambda: self._job_geo("gdacs", "GDACS alerts")),
            Job("quotes", 120, self._job_quotes),
            Job("scores", 300, self._job_scores),
            Job("prune", 86400, store.prune),
        ]

    @staticmethod
    def _job_calendar() -> None:
        items, err = sources.calendar_items()
        if err:
            store.record_health("ff_calendar", "calendar", "Economic calendar", 2700, ok=False, error=err)
            return
        store.upsert_events(items)
        store.record_health("ff_calendar", "calendar", "Economic calendar", 2700, ok=True, count=len(items))

    def _job_feeds(self) -> None:
        pool = self._pool or ThreadPoolExecutor(max_workers=MAX_PARALLEL, thread_name_prefix="ge-feed")

        def one(feed: dict) -> tuple[dict, list[dict], str | None]:
            items, err = sources.fetch_feed(feed)
            return feed, items, err

        for feed, items, err in pool.map(one, sources.FEEDS):
            label = feed["publisher"] + (" (Google News)" if feed.get("gnews") else "")
            sid = f"rss:{feed['id']}"
            if err:
                store.record_health(sid, "rss", label, sources.FEED_MAX_STALE_SEC, ok=False, error=err)
                continue
            if items:
                store.ingest_headlines(items)
            store.record_health(sid, "rss", label, sources.FEED_MAX_STALE_SEC, ok=True, count=len(items))

    @staticmethod
    def _job_geo(which: str, label: str) -> None:
        items, err = sources.fetch_geo(which)
        sid = f"geo:{which}"
        if err:
            store.record_health(sid, "geo", label, 2700 if which == "usgs" else 5400, ok=False, error=err)
            return
        store.upsert_geo(items)
        store.record_health(sid, "geo", label, 2700 if which == "usgs" else 5400, ok=True, count=len(items))

    @staticmethod
    def _job_quotes() -> None:
        rows, err = sources.fetch_quotes()
        if err:
            store.record_health("quotes:stooq", "quotes", "Quotes (Stooq)", 600, ok=False, error=err)
            return
        store.insert_quotes(rows)
        store.record_health("quotes:stooq", "quotes", "Quotes (Stooq)", 600, ok=True, count=len(rows))

    @staticmethod
    def _job_scores() -> None:
        store.compute_theater_scores()

    # ------------------------------------------------------------------ lifecycle
    def touch(self) -> None:
        self._last_seen = time.monotonic()

    @property
    def active(self) -> bool:
        return self._last_seen is not None and (time.monotonic() - self._last_seen) < ACTIVE_WINDOW_S

    def ensure_started(self) -> bool:
        """Start the poller thread once. Returns False (and does nothing) when disabled or under pytest."""
        self.touch()
        if os.environ.get("T58_GLOBAL_EVENTS_POLLER") == "0":
            return False
        if "PYTEST_CURRENT_TEST" in os.environ and os.environ.get("T58_GLOBAL_EVENTS_FORCE_POLLER") != "1":
            return False
        with self._lock:
            if self._thread and self._thread.is_alive():
                return True
            self._stop.clear()
            self._pool = ThreadPoolExecutor(max_workers=MAX_PARALLEL, thread_name_prefix="ge-feed")
            self._thread = threading.Thread(target=self._loop, name="global-events-poller", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop.set()
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)

    def request_refresh(self) -> bool:
        """Manual "refresh now": every job becomes due, at most once per MIN_MANUAL_REFRESH_S."""
        now = time.monotonic()
        if now - self._last_manual < MIN_MANUAL_REFRESH_S:
            return False
        self._last_manual = now
        for j in self.jobs:
            if j.id != "prune":
                j.next_due = 0.0
        return True

    def _loop(self) -> None:
        stagger = 0.0
        for j in self.jobs:   # first run: spread the jobs out so start-up is a trickle, not a burst
            j.next_due = time.monotonic() + stagger
            stagger += 1.5
        while not self._stop.is_set():
            if not self.active:
                self._stop.wait(IDLE_TICK_S)
                continue
            now = time.monotonic()
            for j in self.jobs:
                if j.running or now < j.next_due:
                    continue
                j.running = True
                try:
                    j.run()
                except Exception as exc:  # noqa: BLE001 -- one bad job must never end the loop
                    _log.warning("global-events job %s failed: %s", j.id, exc)
                finally:
                    j.running = False
                    j.next_due = time.monotonic() + self._interval_for(j) + j.jitter
                if self._stop.is_set() or not self.active:
                    break
            self._stop.wait(TICK_S)

    def _interval_for(self, job: Job) -> float:
        """Calendar polls faster inside the hour before a red-folder release so the actual lands quickly."""
        if job.id != "calendar":
            return job.interval_s
        try:
            now = store.utcnow()
            soon = store.events(now - timedelta(minutes=10), now + timedelta(minutes=60), min_impact=3, limit=1)
            return 300 if soon else job.interval_s
        except Exception:  # noqa: BLE001
            return job.interval_s


POLLER = Poller()
