#!/usr/bin/env python3
"""MT5 bridge helper for Linux / macOS (or a remote Windows host). Normally you do NOT need this: the app starts and
prepares the bridge itself the first time you connect. Use it to check or pre-warm the setup.

    python scripts/mt5_bridge_server.py            # show what is detected on this machine and what is missing
    python scripts/mt5_bridge_server.py --check    # start the bridge if needed and test that the terminal answers
    python scripts/mt5_bridge_server.py --setup    # one-time preparation: Windows Python + MetaTrader5 inside Wine, then start
    python scripts/mt5_bridge_server.py --serve    # (WINDOWS Python only) run the bridge server in the foreground
    python scripts/mt5_bridge_server.py --log      # print the MT5 helper's own log (what it said when it last started/stopped)

See docs/MT5_CROSS_PLATFORM.md.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.forward_test import mt5_bridge_manager as mgr  # noqa: E402
from app.forward_test import mt5_connector as mc  # noqa: E402


def _status() -> str:
    kind = mc.platform_kind()
    h, p = mc.bridge_endpoint()
    lines = [f"Detected OS: {kind}.  Bridge endpoint: {h}:{p}"]
    if kind == "windows":
        lines.append("Windows: the native MetaTrader5 package is used; no bridge needed.")
    else:
        lines.append(f"wine: {mgr.find_wine() or 'NOT FOUND (install wine64 / wine-stable)'}")
        cands = mc.find_terminal_candidates()
        lines.append("terminal64.exe: " + (cands[0] if cands else "not found (install your broker's MT5 in Wine, or set 'Terminal path')"))
        prefix = mgr.prefix_of(cands[0]) if cands else None
        py = mgr.find_wine_python(prefix) if prefix else None
        lines.append("Windows Python in Wine: " + (py or "not yet (the app installs it on first connect)"))
        lines.append(f"rpyc installed here: {mc._bridge_package_available()}")
        lines.append(f"bridge listening: {mgr.port_open(h, p)}")
        lines.append(f"helper log file: {mgr.bridge_log_path()}   (python scripts/mt5_bridge_server.py --log)")
    return "\n".join(lines)


def _check(install: bool = True) -> int:
    if not mc.is_available():
        print(mc.unavailable_reason())
        return 2
    if not mc._activate_backend():
        print(f"Bridge not ready: {mc._BRIDGE_ERROR}")
        return 1
    be = mc.get_backend()
    ok = be.initialize()
    print(f"backend={mc.backend_name()} initialize={ok}")
    if ok:
        print("terminal:", be.terminal_info())
        be.shutdown()
        return 0
    print("last_error:", be.last_error())
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=mc.DEFAULT_BRIDGE_PORT)
    ap.add_argument("--token", default=os.environ.get("T58_MT5_BRIDGE_TOKEN", ""))
    a = ap.parse_args()
    if a.log:
        tail = mgr.read_log_tail(6000)
        print(f"[{mgr.bridge_log_path()}]\n" + (tail or "(empty: the helper has not been started on this machine yet)"))
        return 0
    if a.serve:
        sys.argv = ["mt5_rpc_server.py", "--host", a.host, "--port", str(a.port)] + (["--token", a.token] if a.token else [])
        runpy.run_path(str(mgr.SERVER_SCRIPT), run_name="__main__")
        return 0
    if a.setup:
        cands = mc.find_terminal_candidates()
        ok, msg = mgr.ensure_bridge(a.host, a.port, cands[0] if cands else None, log=print)
        print(("OK: " if ok else "FAILED: ") + msg)
        return 0 if ok else 1
    if a.check:
        return _check()
    print(_status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
