#!/usr/bin/env python3
"""Cross-platform smoke test for the web app. Works the same on Windows, macOS and Linux:
starts the real Flask server on a free port, requests every main page over real HTTP, and
checks every link on the Validate page resolves. Exit code 0 = all good.

    python scripts/web_smoke_test.py
"""
from __future__ import annotations

import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGES = [
    "/activate", "/static/theme.css", "/manifest.json", "/dashboard", "/validate", "/library", "/replay",
    "/optimize", "/full-pipeline", "/walk-forward-opt", "/walk-forward-ga", "/cpcv", "/pbo", "/sensitivity",
    "/parameter-robustness", "/regime-matrix", "/payout-probability", "/forward-test", "/live-market",
    "/start-here/create", "/start-here/test", "/start-here/champion", "/start-here/deployment",
    "/start-here/graveyard", "/start-here/quantlab", "/start-here/account", "/support", "/user-manual",
]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""


def main() -> int:
    from app.web import server

    server._license_ok_cached = True  # this test is about pages, not licensing
    port = _free_port()
    threading.Thread(
        target=lambda: server.app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False), daemon=True,
    ).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            _get(base + "/activate")
            break
        except Exception:
            time.sleep(0.5)
    else:
        print("FAIL: server did not start")
        return 1

    failures = []
    for path in PAGES:
        status, _ = _get(base + path)
        print(f"{'ok  ' if status == 200 else 'FAIL'} {status} {path}")
        if status != 200:
            failures.append(path)

    _, validate_html = _get(base + "/validate")
    for href in sorted(set(re.findall(r'class="t58-method-card" href="(/[^"]*)"', validate_html))):
        status, _ = _get(base + href)
        print(f"{'ok  ' if status == 200 else 'FAIL'} {status} {href}  (Validate card)")
        if status != 200:
            failures.append(href)

    print(f"\n{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}  "
          f"[{sys.platform}, Python {sys.version.split()[0]}]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
