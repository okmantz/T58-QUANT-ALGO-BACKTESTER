#!/usr/bin/env python3
"""Cross-platform smoke test for the web app. Works the same on Windows, macOS and Linux:
starts the real Flask server on a free port, requests every main page over real HTTP, and
checks every link on the Validate page resolves. Exit code 0 = all good.

    python scripts/web_smoke_test.py
"""
from __future__ import annotations

import faulthandler
import os
import re
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

# CI runners (macOS especially) have no unlocked OS keychain; a keyring call there can block waiting for a UI prompt
# that never comes. The smoke test never needs a real keyring, so default to the no-op backend unless the caller chose one.
os.environ.setdefault("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGES = [
    "/activate", "/static/theme.css", "/manifest.json", "/dashboard", "/validate", "/library", "/replay",
    "/optimize", "/full-pipeline", "/walk-forward-opt", "/walk-forward-ga", "/cpcv", "/pbo", "/sensitivity",
    "/parameter-robustness", "/regime-matrix", "/payout-probability", "/forward-test", "/live-market",
    "/start-here/create", "/start-here/test", "/start-here/champion", "/start-here/deployment",
    "/start-here/graveyard", "/support", "/user-manual", "/webhook-relay", "/settings/api-keys", "/data/api/providers",
    # 2026-10-17 release
    "/journal", "/gamma", "/runs", "/education", "/mobile-access", "/sw.js", "/static/t58-metrics.js",
    "/static/t58-propsim.js", "/static/t58-edu-labs.js", "/api/journal", "/api/gamma/status", "/api/library/control",
    "/api/prop-sim/presets", "/api/metrics/profit_factor", "/api/running-jobs.json",
    # 2026-10-18 release
    "/research-labs", "/research-labs/volume-profile", "/research-labs/order-flow", "/research-labs/order-book", "/research-labs/lead-lag",
    "/research-labs/stat-arb", "/research-labs/event-study", "/research-labs/hypothesis", "/pine-indicator/", "/education/knowledge-check",
    "/quant-lab/", "/api/memory",
    # 2026-10-21 release: Global Events (store-only endpoints; must answer instantly even with no internet)
    "/events/", "/events/api/bootstrap", "/events/api/health", "/static/global-events/global-events.js", "/static/global-events/land.json",
]

# v9.15 (Owen): the Quant Lab and Account Start Here pages were
# deliberately removed from the web app (Quant Lab's explainer moved
# onto the Quant Lab page itself; the shared guide data stays for the
# desktop app). They must 404 here, not 200.
EXPECTED_404 = ["/start-here/quantlab", "/start-here/account"]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


REQUEST_TIMEOUT = float(os.environ.get("T58_SMOKE_TIMEOUT", "60"))
RETRIES = int(os.environ.get("T58_SMOKE_RETRIES", "2"))


def _get(url: str) -> tuple[int, str]:
    """GET with a generous timeout and retries on transient network trouble. HTTP error statuses are returned (not
    raised); a request that still cannot complete returns status 0 plus the reason, so ONE slow page is reported by name
    instead of aborting the whole run with a bare traceback."""
    last = ""
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, ""
        except Exception as exc:  # noqa: BLE001 -- URLError, timeout, connection reset ...
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0 + attempt)
    print(f"      request failed after {RETRIES + 1} attempts: {last}")
    return 0, ""


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
    # If any request wedges the server, dump every thread's stack to stderr so the CI log shows WHERE (not just "exit code 1").
    faulthandler.enable()
    faulthandler.dump_traceback_later(int(os.environ.get("T58_SMOKE_DUMP_AFTER", "420")), repeat=True, exit=False)
    port = _free_port()
    threading.Thread(
        target=lambda: server.app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False), daemon=True,
    ).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            with urllib.request.urlopen(base + "/activate", timeout=5):
                break
        except urllib.error.HTTPError:
            break
        except Exception:  # noqa: BLE001 -- not listening yet
            time.sleep(0.5)
    else:
        print("FAIL: server did not start")
        return 1

    failures = []
    for path in PAGES:
        t0 = time.time()
        status, _ = _get(base + path)
        print(f"{'ok  ' if status == 200 else 'FAIL'} {status} {path}  ({time.time() - t0:.1f}s)", flush=True)
        if status != 200:
            failures.append(path)
    for path in EXPECTED_404:
        status, _ = _get(base + path)
        print(f"{'ok  ' if status == 404 else 'FAIL'} {status} {path}  (removed -- expect 404)")
        if status != 404:
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
    try:
        code = main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    faulthandler.cancel_dump_traceback_later()
    sys.stdout.flush()
    os._exit(code)   # the Flask thread is a daemon; do not let any lingering worker thread keep the runner alive
