"""SEC EDGAR client (company facts, filings, financial statements). Free public API; no key.

SEC fair-access rules: every request needs a descriptive User-Agent with contact info, and at most 10 requests/second.
The User-Agent is USER-CONFIGURED (env ``T58_SEC_USER_AGENT`` or ``set_user_agent``); nothing is hardcoded and no personal address is sent
unless the user supplied one. Responses are cached on disk, so repeated page loads cost no requests.
All network access goes through one injectable ``fetch`` callable, so tests run offline against fixtures.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable

_DATA = "https://data.sec.gov"
_WWW = "https://www.sec.gov"
_MIN_INTERVAL = 0.12          # ~8 req/s, under the 10/s ceiling
_UA: dict = {"value": None}
_lock = threading.Lock()
_last = {"t": 0.0}


class SecError(RuntimeError):
    pass


def set_user_agent(value: str | None) -> None:
    _UA["value"] = (value or "").strip() or None


def user_agent() -> str | None:
    return _UA["value"] or (os.environ.get("T58_SEC_USER_AGENT") or "").strip() or None


def cache_dir() -> Path:
    d = Path(os.environ.get("T58_SEC_CACHE_DIR") or (Path.home() / ".t58" / "sec_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _http_fetch(url: str) -> bytes:
    ua = user_agent()
    if not ua:
        raise SecError("SEC EDGAR needs a contact User-Agent (e.g. 'T58 Trading you@example.com'). Set it on the Investing page or in T58_SEC_USER_AGENT.")
    import requests
    with _lock:
        wait = _MIN_INTERVAL - (time.time() - _last["t"])
        if wait > 0:
            time.sleep(wait)
        _last["t"] = time.time()
    r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=25)
    if r.status_code != 200:
        raise SecError(f"SEC returned HTTP {r.status_code} for {url}")
    return r.content


class SecClient:
    def __init__(self, fetch: Callable[[str], bytes] | None = None, cache: Path | None = None, ttl_hours: float = 12.0):
        self._fetch = fetch or _http_fetch
        self._cache = Path(cache) if cache else None
        self._ttl = ttl_hours * 3600

    def _get_json(self, url: str, key: str):
        path = None
        if self._cache is not None or self._fetch is _http_fetch:
            path = (self._cache or cache_dir()) / f"{key}.json"
            if path.exists() and time.time() - path.stat().st_mtime < self._ttl:
                try:
                    return json.loads(path.read_text())
                except Exception:  # noqa: BLE001
                    pass
        data = json.loads(self._fetch(url))
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            except OSError:
                pass
        return data

    # ------------------------------------------------------------------ lookups
    def tickers(self) -> list[dict]:
        raw = self._get_json(f"{_WWW}/files/company_tickers.json", "company_tickers")
        rows = raw.values() if isinstance(raw, dict) else raw
        return [{"cik": int(r["cik_str"]), "ticker": str(r["ticker"]).upper(), "name": r["title"]} for r in rows]

    def search(self, query: str, limit: int = 12) -> list[dict]:
        q = (query or "").strip().lower()
        if not q:
            return []
        rows = self.tickers()
        exact = [r for r in rows if r["ticker"].lower() == q]
        starts = [r for r in rows if r["ticker"].lower().startswith(q) and r not in exact]
        names = [r for r in rows if q in r["name"].lower() and r not in exact and r not in starts]
        return (exact + starts + names)[:limit]

    def cik_for(self, ticker: str) -> int | None:
        t = (ticker or "").upper()
        for r in self.tickers():
            if r["ticker"] == t:
                return r["cik"]
        return None

    def submissions(self, cik: int) -> dict:
        return self._get_json(f"{_DATA}/submissions/CIK{int(cik):010d}.json", f"sub_{int(cik):010d}")

    def company_facts(self, cik: int) -> dict:
        return self._get_json(f"{_DATA}/api/xbrl/companyfacts/CIK{int(cik):010d}.json", f"facts_{int(cik):010d}")

    def recent_filings(self, cik: int, forms: tuple = ("10-K", "10-Q", "8-K", "20-F", "40-F", "6-K"), limit: int = 25) -> list[dict]:
        rec = (self.submissions(cik).get("filings") or {}).get("recent") or {}
        out = []
        for i, form in enumerate(rec.get("form", [])):
            if forms and form not in forms:
                continue
            acc = rec["accessionNumber"][i]
            doc = rec.get("primaryDocument", [""] * (i + 1))[i]
            out.append({
                "form": form, "filed": rec["filingDate"][i], "period": (rec.get("reportDate") or [""] * (i + 1))[i],
                "accession": acc, "description": (rec.get("primaryDocDescription") or [""] * (i + 1))[i],
                "url": f"{_WWW}/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}" if doc else None,
            })
            if len(out) >= limit:
                break
        return out
