"""Starts and prepares the MT5 bridge server for the user, so on Linux/macOS saving the account is enough.

Flow of ``ensure_bridge()`` (only for a bridge on THIS machine; a remote host manages its own server):
  1. already listening on host:port?  -> done.
  2. find Wine and the Wine prefix that holds the MT5 terminal (or the default prefix).
  3. find the Windows Python inside that prefix; if the MetaTrader5/rpyc/numpy packages are missing, pip-install them
     inside Wine (one time).
  4. start app/forward_test/mt5_rpc_server.py with that Python, wait until it answers.
Every step that cannot be done returns a plain-language reason plus the single command that fixes it.
"""
from __future__ import annotations

import glob
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

SERVER_SCRIPT = Path(__file__).with_name("mt5_rpc_server.py")
_PROC: subprocess.Popen | None = None
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0", "")
PY_INSTALLER_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"


def is_local_host(host: str) -> bool:
    return (host or "").strip().lower() in LOCAL_HOSTS


def port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1" if is_local_host(host) else host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def find_wine() -> str | None:
    for name in ("wine64", "wine"):
        p = shutil.which(name)
        if p:
            return p
    for p in ("/opt/wine-stable/bin/wine64", "/opt/homebrew/bin/wine64", "/usr/local/bin/wine64",
              "/Applications/Wine Stable.app/Contents/Resources/wine/bin/wine64",
              "/Applications/Wine Staging.app/Contents/Resources/wine/bin/wine64"):
        if os.path.exists(p):
            return p
    return None


def prefix_of(path: str | None) -> str | None:
    if not path:
        return None
    low = path.replace("\\", "/")
    i = low.lower().find("/drive_c/")
    return low[:i] if i > 0 else None


def find_wine_python(prefix: str) -> str | None:
    pats = [os.path.join(prefix, "drive_c", "users", "*", "AppData", "Local", "Programs", "Python", "Python3*", "python.exe"),
            os.path.join(prefix, "drive_c", "Python3*", "python.exe"),
            os.path.join(prefix, "drive_c", "Program Files", "Python3*", "python.exe"),
            os.path.join(prefix, "drive_c", "windows", "py.exe")]
    found = [p for pat in pats for p in glob.glob(pat)]
    found = [p for p in found if not p.endswith("py.exe")] or found
    found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return found[0] if found else None


def _env(prefix: str | None) -> dict:
    env = dict(os.environ)
    if prefix:
        env["WINEPREFIX"] = prefix
    env.setdefault("WINEDEBUG", "-all")
    return env


def wine_path(host_path: str) -> str:
    """Z: drive path Wine can open for a file on the host filesystem."""
    return "Z:" + os.path.abspath(host_path).replace("/", "\\")


def _wine_run(wine, prefix, args, timeout=120):
    return subprocess.run([wine, *args], env=_env(prefix), capture_output=True, text=True, timeout=timeout)


def packages_ready(wine: str, prefix: str, py: str) -> bool:
    try:
        r = _wine_run(wine, prefix, [py, "-c", "import MetaTrader5, rpyc, numpy"], timeout=90)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def install_packages(wine: str, prefix: str, py: str, log=lambda m: None) -> tuple[bool, str]:
    log("Installing MetaTrader5 + rpyc inside Wine (one time, needs internet)...")
    try:
        r = _wine_run(wine, prefix, [py, "-m", "pip", "install", "--upgrade", "MetaTrader5", "rpyc>=5.3,<7", "numpy"], timeout=900)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return r.returncode == 0, (r.stdout + r.stderr)[-600:]


def install_wine_python(wine: str, prefix: str, log=lambda m: None) -> tuple[bool, str]:
    """Downloads the Python installer and runs it silently inside the Wine prefix."""
    import urllib.request
    dst = Path(os.path.expanduser("~")) / ".cache" / "t58" / "python-3.11.9-amd64.exe"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            log("Downloading Python for Wine...")
            urllib.request.urlretrieve(PY_INSTALLER_URL, dst)
        log("Installing Python inside Wine...")
        r = _wine_run(wine, prefix, [str(dst), "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0"], timeout=900)
        return r.returncode == 0 and find_wine_python(prefix) is not None, (r.stdout + r.stderr)[-400:]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def ensure_bridge(host: str, port: int, terminal_path: str | None = None, *, install: bool = True,
                  log=lambda m: None, wait: float = 60.0) -> tuple[bool, str]:
    """(ok, message). Safe to call repeatedly; returns fast when the bridge is already up."""
    global _PROC
    if port_open(host, port):
        return True, "Bridge is running."
    if not is_local_host(host):
        return False, (f"Nothing is listening at {host}:{port}. On that machine start the bridge with the Windows Python: "
                       f"`python mt5_rpc_server.py --host 0.0.0.0 --port {port} --token <secret>` (copy app/forward_test/mt5_rpc_server.py there; "
                       "set the same T58_MT5_BRIDGE_TOKEN here), and open that port in its firewall.")
    if sys.platform.startswith("win"):
        return False, (f"Nothing is listening at {host}:{port}. On Windows the app talks to MT5 directly (`pip install MetaTrader5`), "
                       "so no bridge is needed.")
    wine = find_wine()
    if not wine:
        hint = ("`brew install --cask wine-stable` (or install MetaQuotes' MetaTrader 5 for macOS)" if sys.platform == "darwin"
                else "`sudo apt install wine64` (or your distro's wine package)")
        return False, f"Wine is not installed, and the MT5 terminal needs it on this OS. Install it with {hint}, install your broker's MT5 in it, then try again."
    prefix = prefix_of(terminal_path)
    if not prefix:
        from app.forward_test.mt5_connector import find_terminal_candidates, wine_prefix_candidates
        prefix = prefix_of((find_terminal_candidates() or [None])[0]) or (wine_prefix_candidates() or [None])[0] \
            or os.path.join(os.path.expanduser("~"), ".wine")
    py = find_wine_python(prefix)
    if not py and install:
        ok, tail = install_wine_python(wine, prefix, log)
        py = find_wine_python(prefix) if ok else None
        if not py:
            return False, ("Could not set up Windows Python inside Wine automatically "
                           f"({tail.strip()[-200:] or 'installer failed'}). Run it by hand once: download {PY_INSTALLER_URL} and "
                           f"`WINEPREFIX=\"{prefix}\" wine python-3.11.9-amd64.exe /quiet InstallAllUsers=0 PrependPath=1`, then try again.")
    if not py:
        return False, f"No Windows Python found inside the Wine prefix {prefix}."
    if not packages_ready(wine, prefix, py):
        if not install:
            return False, "MetaTrader5/rpyc are not installed in the Wine Python."
        ok, tail = install_packages(wine, prefix, py, log)
        if not ok or not packages_ready(wine, prefix, py):
            return False, ("Installing MetaTrader5 inside Wine failed. Run once: "
                           f"`WINEPREFIX=\"{prefix}\" wine \"{py}\" -m pip install MetaTrader5 rpyc numpy`. Last output: {tail.strip()[-200:]}")
    log("Starting the MT5 bridge...")
    try:
        _PROC = subprocess.Popen([wine, py, wine_path(str(SERVER_SCRIPT)), "--host", "127.0.0.1", "--port", str(port)],
                                 env=_env(prefix), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not start the bridge under Wine: {type(exc).__name__}: {exc}"
    deadline = time.time() + wait
    while time.time() < deadline:
        if port_open("127.0.0.1", port):
            return True, "Bridge started."
        if _PROC.poll() is not None:
            return False, f"The bridge process exited immediately (code {_PROC.returncode}). Run `python scripts/mt5_bridge_server.py --check` for details."
        time.sleep(0.7)
    return False, f"The bridge did not start listening on port {port} within {int(wait)} seconds."


def stop_bridge() -> None:
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        try:
            _PROC.terminate()
        except Exception:  # noqa: BLE001
            pass
    _PROC = None
