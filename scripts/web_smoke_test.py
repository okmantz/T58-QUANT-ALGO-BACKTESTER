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


def _activate_smoke_license():
    """v9.14: this smoke test used to poke server._license_ok_cached
    directly. Since the license gate now engages on first launch on
    every build (v9.14 restored it), activate for real instead --
    through app.licensing.client.activate() with a smoke-only master
    key whose hash is supplied via the client's documented env
    override -- so the run exercises the same code a first-launch user
    hits. Any pre-existing license state on this machine is snapshotted
    first and restored afterwards (the keyring is bypassed in favor of
    the file fallback so the restore is exact). Returns (restore_fn)."""
    import hashlib
    import os

    from app.licensing import client as lic

    test_key = "T58-SMOKE-TEST"
    state_path, key_path = lic._state_path(), lic._key_fallback_path()
    snapshot = {p: (p.read_bytes() if p.exists() else None) for p in (state_path, key_path)}
    old_env = os.environ.get("T58_MASTER_LICENSE_KEY_HASH")
    old_keyring = lic._try_keyring
    lic._try_keyring = lambda: None  # force the file fallback -- snapshot/restore covers it
    os.environ["T58_MASTER_LICENSE_KEY_HASH"] = hashlib.sha256(
        test_key.strip().upper().encode("utf-8")
    ).hexdigest()
    ok, msg = lic.activate("smoke-test@t58.local", test_key)
    if not ok:
        raise SystemExit(f"FAIL: smoke-test license activation failed: {msg}")

    def _restore():
        for path, data in snapshot.items():
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(data)
        if old_env is None:
            os.environ.pop("T58_MASTER_LICENSE_KEY_HASH", None)
        else:
            os.environ["T58_MASTER_LICENSE_KEY_HASH"] = old_env
        lic._try_keyring = old_keyring

    return _restore


def main() -> int:
    from app.web import server

    restore_license = _activate_smoke_license()
    try:
        return _run_pages(server)
    finally:
        restore_license()
        server._license_ok_cached = None


def _run_pages(server) -> int:
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
