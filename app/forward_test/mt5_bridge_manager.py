"""Starts and prepares the MT5 bridge server for the user, so on Linux/macOS saving the account is enough.

Flow of ``ensure_bridge_detailed()`` (only for a bridge on THIS machine; a remote host manages its own server):
  1. already listening on host:port?  -> done.
  2. find Wine and the Wine prefix that holds the MT5 terminal (or the default prefix).
  3. find the Windows Python inside that prefix; install it silently if missing; pip-install MetaTrader5/rpyc/numpy
     inside Wine if missing (one time).
  4. copy mt5_rpc_server.py into the prefix's own C: drive, start it with that Python, wait until it answers.
     The helper's output goes to data/logs/mt5_bridge.log, and if it dies, the last lines of that log are what the
     user (and support) get to see.
Every step that cannot be done returns a ``Problem`` (app.forward_test.mt5_diagnostics): a plain-language title,
numbered fix steps and the technical text kept separate. ``ensure_bridge()`` keeps the old (ok, message) shape.
"""
from __future__ import annotations

import glob
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from app.forward_test import mt5_diagnostics as diag

SERVER_SCRIPT = Path(__file__).with_name("mt5_rpc_server.py")
_PROC: subprocess.Popen | None = None
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0", "")
PY_INSTALLER_URL = diag.PY_INSTALLER_URL
WINE_SERVER_DIR = "t58"          # <prefix>/drive_c/t58/mt5_rpc_server.py  ==  C:\t58\mt5_rpc_server.py


@dataclass
class BridgeResult:
    ok: bool
    message: str
    problem: "diag.Problem | None" = None


def is_local_host(host: str) -> bool:
    return (host or "").strip().lower() in LOCAL_HOSTS


def port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1" if is_local_host(host) else host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


# ---- log file ------------------------------------------------------------------------------------------------

def log_dir() -> Path:
    try:
        from app.data.storage import get_app_base_dir
        d = get_app_base_dir() / "data" / "logs"
    except Exception:  # noqa: BLE001
        d = Path.home() / ".t58" / "logs"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        d = Path.home() / ".t58" / "logs"
        d.mkdir(parents=True, exist_ok=True)
    return d


def bridge_log_path() -> Path:
    return log_dir() / "mt5_bridge.log"


def read_log_tail(max_chars: int = 1500) -> str:
    try:
        text = bridge_log_path().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


# ---- finding the pieces --------------------------------------------------------------------------------------

def find_wine() -> str | None:
    for name in ("wine64", "wine"):
        p = shutil.which(name)
        if p:
            return p
    fixed = ["/opt/wine-stable/bin/wine64", "/opt/homebrew/bin/wine64", "/usr/local/bin/wine64",
             "/Applications/Wine Stable.app/Contents/Resources/wine/bin/wine64",
             "/Applications/Wine Staging.app/Contents/Resources/wine/bin/wine64"]
    for p in fixed:
        if os.path.exists(p):
            return p
    # MetaQuotes' own macOS app (MetaTrader 5.app) ships its own Wine; use it when it is the only one around.
    for pat in ("/Applications/MetaTrader*.app/Contents/SharedSupport/wine/bin/wine64",
                "/Applications/MetaTrader*.app/Contents/SharedSupport/wine/bin/wine",
                os.path.expanduser("~/Applications/MetaTrader*.app/Contents/SharedSupport/wine/bin/wine64")):
        for p in sorted(glob.glob(pat)):
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
    """Z: drive path Wine can open for a file on the host filesystem (works only when the prefix still has a Z: drive)."""
    return "Z:" + os.path.abspath(host_path).replace("/", "\\")


def stage_server_script(prefix: str) -> tuple[str, str] | None:
    """Copies mt5_rpc_server.py into the prefix's own C: drive and returns (host_dir, windows_path), e.g.
    (<prefix>/drive_c/t58, 'C:\\t58\\mt5_rpc_server.py'). Wine can always read its own C: drive, whereas a Z: path
    to the app folder fails on prefixes without a Z: drive or with unusual characters in the path. Always overwritten,
    so an app update's new protocol is never shadowed by a stale copy. None when the prefix has no drive_c."""
    drive_c = Path(prefix) / "drive_c"
    if not drive_c.is_dir():
        return None
    dest = drive_c / WINE_SERVER_DIR
    try:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SERVER_SCRIPT, dest / SERVER_SCRIPT.name)
    except OSError:
        return None
    return str(dest), f"C:\\{WINE_SERVER_DIR}\\{SERVER_SCRIPT.name}"


def _wine_run(wine, prefix, args, timeout=120):
    return subprocess.run([wine, *args], env=_env(prefix), capture_output=True, text=True, timeout=timeout)


def check_packages(wine: str, prefix: str, py: str) -> tuple[bool, str]:
    try:
        r = _wine_run(wine, prefix, [py, "-c", "import MetaTrader5, rpyc, numpy"], timeout=90)
        return r.returncode == 0, (r.stdout + r.stderr)[-600:]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def packages_ready(wine: str, prefix: str, py: str) -> bool:
    return check_packages(wine, prefix, py)[0]


def install_packages(wine: str, prefix: str, py: str, log=lambda m: None) -> tuple[bool, str]:
    log("Installing the MT5 connection packages inside Wine (one time, needs internet)...")
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
            log("Downloading Python for Wine (one time)...")
            urllib.request.urlretrieve(PY_INSTALLER_URL, dst)
        log("Installing Python inside Wine (one time, about a minute)...")
        r = _wine_run(wine, prefix, [str(dst), "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0"], timeout=900)
        return r.returncode == 0 and find_wine_python(prefix) is not None, (r.stdout + r.stderr)[-400:]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


# ---- starting the server -------------------------------------------------------------------------------------

def _start_once(wine: str, prefix: str, py: str, script_arg: str, cwd: str | None, port: int, wait: float, label: str):
    """Launches the helper once. Returns (listening, returncode_or_None, log_text_written_by_this_attempt)."""
    global _PROC
    path = bridge_log_path()
    try:
        size_before = path.stat().st_size
    except OSError:
        size_before = 0
    cmd = [wine, py, script_arg, "--host", "127.0.0.1", "--port", str(port)]
    with open(path, "ab", buffering=0) as logf:
        logf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  starting MT5 helper ({label}): {' '.join(cmd)}\n".encode())
        try:
            _PROC = subprocess.Popen(cmd, env=_env(prefix), cwd=cwd if cwd and os.path.isdir(cwd) else None,
                                     stdout=logf, stderr=subprocess.STDOUT, close_fds=True)
        except Exception as exc:  # noqa: BLE001
            logf.write(f"could not launch: {type(exc).__name__}: {exc}\n".encode())
            return False, -1, f"{type(exc).__name__}: {exc}"
    deadline = time.time() + wait
    while time.time() < deadline:
        if port_open("127.0.0.1", port):
            return True, None, ""
        if _PROC.poll() is not None:
            time.sleep(0.3)           # let the OS flush the last lines into the log
            break
        time.sleep(0.7)
    rc = _PROC.poll()
    try:
        with open(path, "rb") as f:
            f.seek(size_before)
            written = f.read().decode("utf-8", errors="replace")
    except OSError:
        written = ""
    return False, rc, written


def _fail(problem: "diag.Problem") -> BridgeResult:
    return BridgeResult(False, problem.plain_text(), problem)


def ensure_bridge_detailed(host: str, port: int, terminal_path: str | None = None, *, install: bool = True,
                           log=lambda m: None, wait: float = 60.0) -> BridgeResult:
    """Safe to call repeatedly; returns fast when the bridge is already up."""
    if port_open(host, port):
        return BridgeResult(True, "Bridge is running.")
    if not is_local_host(host):
        return _fail(diag.remote_unreachable(host, port))
    if sys.platform.startswith("win"):
        return _fail(diag.windows_package_missing(f"Nothing is listening at {host}:{port}; on Windows the app talks to MT5 "
                                                  "directly through the MetaTrader5 package, so no bridge is needed."))
    log("Looking for Wine and MetaTrader 5...")
    wine = find_wine()
    if not wine:
        return _fail(diag.wine_missing())
    prefix = prefix_of(terminal_path)
    if not prefix:
        from app.forward_test.mt5_connector import find_terminal_candidates, wine_prefix_candidates
        cands = find_terminal_candidates()
        prefix = prefix_of(cands[0] if cands else None)
        if not prefix and (terminal_path or "").strip()[1:2] == ":":
            prefix = (wine_prefix_candidates() or [os.path.join(os.path.expanduser("~"), ".wine")])[0]   # a C:\ path from settings
        if not prefix:
            return _fail(diag.mt5_not_found())
    py = find_wine_python(prefix)
    if not py and install:
        log("Setting up Python for MT5 (first time only, about 2 to 5 minutes)...")
        ok, tail = install_wine_python(wine, prefix, log)
        py = find_wine_python(prefix) if ok else None
        if not py:
            return _fail(diag.python_setup_failed(prefix, tail))
    if not py:
        return _fail(diag.python_setup_failed(prefix, "No Windows Python found inside the Wine prefix."))
    ok, out = check_packages(wine, prefix, py)
    if not ok:
        if not install:
            return _fail(diag.packages_failed(prefix, py, out or "MetaTrader5/rpyc are not installed in the Wine Python."))
        ok, tail = install_packages(wine, prefix, py, log)
        ok2, out2 = check_packages(wine, prefix, py) if ok else (False, tail)
        if not ok or not ok2:
            return _fail(diag.packages_failed(prefix, py, out2 or tail))

    log("Starting the MT5 connection helper...")
    staged = stage_server_script(prefix)
    attempts = []
    if staged:
        attempts.append(("C: copy", staged[1], staged[0]))
    attempts.append(("Z: path", wine_path(str(SERVER_SCRIPT)), None))
    last_rc, last_log = None, ""
    for n, (label, script_arg, cwd) in enumerate(attempts):
        listening, rc, written = _start_once(wine, prefix, py, script_arg, cwd, port, wait, label)
        if listening:
            return BridgeResult(True, "Bridge started.")
        last_rc, last_log = rc, written
        if rc is None:                                   # still running but never opened its port
            return _fail(diag.bridge_timeout(int(wait), written))
        low = written.lower()
        if "no module named" in low and install and n == 0:
            log("A package is missing inside Wine, reinstalling it...")
            install_packages(wine, prefix, py, log)
            listening, rc, written = _start_once(wine, prefix, py, script_arg, cwd, port, wait, label + ", after reinstall")
            if listening:
                return BridgeResult(True, "Bridge started.")
            last_rc, last_log = rc, written
        if "bad cpu type" in low:
            break                                         # retrying with another path cannot fix this
    return _fail(diag.bridge_failed(last_rc, last_log or read_log_tail(), prefix, str(bridge_log_path())))


def ensure_bridge(host: str, port: int, terminal_path: str | None = None, *, install: bool = True,
                  log=lambda m: None, wait: float = 60.0) -> tuple[bool, str]:
    """(ok, message): the original shape, kept for callers that only want text."""
    r = ensure_bridge_detailed(host, port, terminal_path, install=install, log=log, wait=wait)
    return r.ok, r.message


def stop_bridge() -> None:
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        try:
            _PROC.terminate()
        except Exception:  # noqa: BLE001
            pass
    _PROC = None
