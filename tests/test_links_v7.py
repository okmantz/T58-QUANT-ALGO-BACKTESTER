"""v7 link check (worker D, 2026-10-05) -- CI-runnable.

Crawls README.md and docs/** for http(s) URLs and verifies each one is
alive. A URL counts as alive on: 2xx/3xx, or 401/403/405/429 (the server
answered -- it just wants auth, blocks bots, or rate-limits; the link
itself is not dead). Dead: 404/410, 5xx, DNS/connection failures.

Skipped (not failures):
  * URL templates with {placeholders} and localhost/127.0.0.1 addresses
    (runtime code interpolation, not real links),
  * the bot-block allowlist below (LinkedIn etc. legitimately refuse
    automated checks),
  * the whole module, when the machine has no network (a sandbox without
    internet must not go red -- the one-off crawl in the v7 bundle notes
    verified these links with network access).

One-off full crawl (README + docs + app/ link strings, 2026-10-05):
62 unique URLs; the only dead user-facing link was tradovate.com/api
(404, fixed to https://api.tradovate.com in broker_tradovate.py).
"""
from __future__ import annotations

import pathlib
import re
import socket
import urllib.request

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
URL_RE = re.compile(r"https?://[^\s\"'\)\]\<>]+")

# Hosts/substrings that legitimately block automated checks -- a 403/429
# from these is "alive", and they are exempt from even being fetched.
BOT_BLOCK_ALLOWLIST = (
    "linkedin.com",
    "babypips.com",
    "cmegroup.com",
    "investopedia.com",
)

# Never real links.
SKIP_SUBSTRINGS = (
    "127.0.0.1",
    "localhost",
    "example.com",
    "somefirm.com",
    "w3.org/2000/svg",  # XML namespace constant, not a browsable link
)


def _has_network() -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False


def collect_doc_urls() -> dict[str, list[str]]:
    """URL -> list of files it appears in, over README.md + docs/**."""
    found: dict[str, list[str]] = {}
    targets = [REPO_ROOT / "README.md"] + sorted((REPO_ROOT / "docs").rglob("*.md"))
    for path in targets:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in URL_RE.finditer(text):
            url = m.group(0).rstrip(".,;:!?").rstrip("`")
            if url == "https://":
                continue
            found.setdefault(url, []).append(str(path.relative_to(REPO_ROOT)))
    return found


def _is_skippable(url: str) -> str | None:
    if "{" in url or "}" in url:
        return "URL template (runtime interpolation)"
    for sub in SKIP_SUBSTRINGS:
        if sub in url:
            return f"skip-substring {sub}"
    for sub in BOT_BLOCK_ALLOWLIST:
        if sub in url:
            return f"bot-block allowlist ({sub})"
    return None


def _check_url(url: str, timeout: int = 20) -> tuple[bool, str]:
    """Returns (alive, detail)."""
    req = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "Mozilla/5.0 (compatible; T58-linkcheck/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.status
    except urllib.error.HTTPError as exc:
        code = exc.code
    except Exception as exc:  # noqa: BLE001 -- DNS refused, timeout, TLS: dead
        return False, f"connection failed: {type(exc).__name__}: {exc}"
    if 200 <= code < 400 or code in (401, 403, 405, 429):
        return True, f"HTTP {code}"
    return False, f"HTTP {code}"


pytestmark = pytest.mark.skipif(
    not _has_network(), reason="no network access -- link check needs the internet"
)


def test_every_documented_link_is_alive():
    urls = collect_doc_urls()
    assert urls, "no URLs found -- the crawler is broken, not the docs"
    checked, skipped = 0, 0
    dead: list[str] = []
    for url in sorted(urls):
        reason = _is_skippable(url)
        if reason:
            skipped += 1
            continue
        checked += 1
        alive, detail = _check_url(url)
        if not alive:
            dead.append(f"{url} ({detail}) -- in {', '.join(urls[url])}")
    assert not dead, (
        f"{len(dead)} dead link(s) out of {checked} checked ({skipped} skipped):\n"
        + "\n".join(dead)
    )
