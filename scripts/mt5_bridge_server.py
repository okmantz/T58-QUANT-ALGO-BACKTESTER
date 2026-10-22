#!/usr/bin/env python3
"""MT5 bridge helper for Linux / macOS (or a remote Windows host).

MetaQuotes only ships the `MetaTrader5` Python package for Windows. On other systems the terminal runs under
Wine and a small RPyC server (`mt5linux`) runs next to it; the app connects to that server over TCP and drives
the same API. This script does two things:

    python scripts/mt5_bridge_server.py            # print the exact setup commands for THIS machine
    python scripts/mt5_bridge_server.py --check    # test that a bridge is reachable and the terminal answers
    python scripts/mt5_bridge_server.py --serve    # (run on the machine that has Wine/Windows Python) start the server

The server side is the stock `mt5linux` server, run by the WINDOWS Python inside Wine (or on a Windows PC/VPS):

    wine python -m pip install MetaTrader5 mt5linux
    wine python -m mt5linux --host 0.0.0.0 --port 18812     # or 127.0.0.1 if the app runs on the same machine

See docs/MT5_CROSS_PLATFORM.md.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.forward_test import mt5_connector as mc  # noqa: E402


def _instructions() -> str:
    kind = mc.platform_kind()
    h, p = mc.bridge_endpoint()
    lines = [f"Detected OS: {kind}.  Bridge endpoint this app will use: {h}:{p}", ""]
    if kind == "windows":
        lines += ["Windows: nothing to set up -- the native MetaTrader5 package is used.",
                  "(To drive a terminal on ANOTHER machine, run the server below there and set T58_MT5_BRIDGE_HOST.)", ""]
    else:
        lines += ["1. Install Wine and the MT5 terminal:"]
        if kind == "macos":
            lines += ["     MetaQuotes' macOS installer (https://www.metatrader5.com/en/terminal/help/start_advanced/install_mac)",
                      "     or CrossOver / `brew install --cask wine-stable`, then install the broker's MT5 setup.exe in it."]
        else:
            lines += ["     sudo apt install wine64 winetricks        # Debian/Ubuntu (dnf/pacman equivalents exist)",
                      "     wine setup.exe                               # your broker's MT5 installer; log in once"]
        lines += ["2. Install Windows Python inside the same Wine prefix, then the server packages:",
                  "     wine python -m pip install MetaTrader5 mt5linux",
                  "3. Start the bridge server (leave it running):",
                  f"     wine python -m mt5linux --host 127.0.0.1 --port {p}",
                  "4. In THIS environment:  pip install mt5linux",
                  "5. Open the Forward Test tab and Save & Test Connection (Terminal path can stay blank; Auto-detect looks in your Wine prefixes).", ""]
    wp = shutil.which("wine64") or shutil.which("wine")
    lines.append(f"wine on PATH: {wp or 'not found'}")
    lines.append(f"mt5linux client installed here: {mc._bridge_package_available()}")
    cands = mc.find_terminal_candidates()
    lines.append("terminal64.exe found: " + (cands[0] if cands else "none (set 'Terminal path' manually)"))
    return "\n".join(lines)


def _check() -> int:
    if not mc.is_available():
        print(mc.unavailable_reason())
        return 2
    if not mc._activate_backend():
        print(f"Bridge not reachable: {mc._BRIDGE_ERROR}")
        return 1
    ok = mc.get_backend().initialize()
    print(f"backend={mc.backend_name()} initialize={ok}")
    if ok:
        print("terminal:", mc.get_backend().terminal_info())
        mc.get_backend().shutdown()
        return 0
    print("last_error:", mc.get_backend().last_error())
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--serve", action="store_true", help="run the mt5linux server with this Python (must be Windows Python)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=mc.DEFAULT_BRIDGE_PORT)
    a = ap.parse_args()
    if a.check:
        return _check()
    if a.serve:
        return subprocess.call([sys.executable, "-m", "mt5linux", "--host", a.host, "--port", str(a.port)])
    print(_instructions())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
