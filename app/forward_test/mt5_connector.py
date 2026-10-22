"""
Thin wrapper around MetaTrader 5's Python API, on Windows, Linux and macOS.

This is the ONLY file in the app that imports `MetaTrader5` directly, and
every import is guarded: everywhere else in the app (including this module's
own class definitions) must load cleanly even when no backend is present --
the Forward Test tab checks `is_available()` and shows a clear explanation
instead of a traceback if it's not.

Backends (chosen automatically, see `_activate_backend`):

  native   Windows. The official `MetaTrader5` package talks to the terminal
           directly. (MetaQuotes only ships it for Windows.)
  bridge   Linux, macOS -- and also a Windows terminal on another machine or
           VPS. The MT5 terminal runs under Wine (or on that remote Windows
           host) next to a small RPyC server (`mt5linux`); this process
           connects to it over TCP and drives the SAME API (`initialize`,
           `login`, `copy_rates_from_pos`, `order_send`, ...), so every
           method below works unchanged. Host/port come from the saved
           settings or T58_MT5_BRIDGE_HOST / T58_MT5_BRIDGE_PORT
           (default localhost:18812). See docs/MT5_CROSS_PLATFORM.md and
           scripts/mt5_bridge_server.py.

Getting a free MT5 demo account: any MT5-supporting broker's own website
("open a demo account"), or the prop firm's own demo/trial account if they
offer one. No paid subscription anywhere in this path -- that's the entire
point of building this on MT5 instead of TradingView (see the
app.forward_test package docstring for why).
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

DEFAULT_BRIDGE_HOST = "localhost"
DEFAULT_BRIDGE_PORT = 18812

try:
    import MetaTrader5 as mt5  # type: ignore
    _MT5_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001 -- absence is expected on non-Windows/dev machines
    mt5 = None  # type: ignore
    _MT5_IMPORT_ERROR = str(exc)

_NATIVE_MT5 = mt5            # remembered so a bridge can be swapped out again
_BRIDGE_ERROR: Optional[str] = None
_BRIDGE_KEY: Optional[tuple] = None


def platform_kind() -> str:
    """'windows' | 'macos' | 'linux' | 'other' -- the OS this process runs on (not the terminal's)."""
    p = sys.platform
    if p.startswith("win") or os.name == "nt":
        return "windows"
    if p == "darwin":
        return "macos"
    if p.startswith("linux"):
        return "linux"
    return "other"


def _bridge_package_available() -> bool:
    try:
        return importlib.util.find_spec("mt5linux") is not None
    except Exception:  # noqa: BLE001
        return False


def bridge_endpoint(host: str | None = None, port: int | str | None = None) -> tuple[str, int]:
    """Bridge address: explicit args, else T58_MT5_BRIDGE_HOST/PORT, else the saved MT5 settings, else
    localhost:18812."""
    saved_host, saved_port = "", 0
    if not (host and port):
        try:
            from app.forward_test.mt5_settings import load_settings
            st = load_settings()
            saved_host, saved_port = st.bridge_host, st.bridge_port
        except Exception:  # noqa: BLE001
            pass
    h = (host or os.environ.get("T58_MT5_BRIDGE_HOST") or saved_host or DEFAULT_BRIDGE_HOST).strip()
    raw = port or os.environ.get("T58_MT5_BRIDGE_PORT") or saved_port or DEFAULT_BRIDGE_PORT
    try:
        pt = int(raw)
    except (TypeError, ValueError):
        pt = DEFAULT_BRIDGE_PORT
    return h, pt


def _activate_backend(host: str | None = None, port: int | str | None = None, force_bridge: bool = False) -> bool:
    """Makes the module-level ``mt5`` usable. Native package if loaded (unless force_bridge), else an
    RPyC bridge client. Returns True when ``mt5`` is ready; on failure stores the reason in _BRIDGE_ERROR."""
    global mt5, _BRIDGE_ERROR, _BRIDGE_KEY
    if mt5 is not None and not force_bridge and mt5 is _NATIVE_MT5:
        return True
    if mt5 is not None and mt5 is not _NATIVE_MT5 and not force_bridge:
        # already a bridge client (or a test double): keep it unless the endpoint changed
        if _BRIDGE_KEY is None or _BRIDGE_KEY == bridge_endpoint(host, port):
            return True
    if _NATIVE_MT5 is not None and not force_bridge:
        mt5 = _NATIVE_MT5
        return True
    if not _bridge_package_available():
        _BRIDGE_ERROR = "the 'mt5linux' package is not installed"
        return False
    h, pt = bridge_endpoint(host, port)
    try:
        from mt5linux import MetaTrader5 as _Bridge  # type: ignore
        client = _Bridge(host=h, port=pt)
    except Exception as exc:  # noqa: BLE001 -- connection refused, version mismatch, ...
        _BRIDGE_ERROR = f"could not reach the MT5 bridge at {h}:{pt} ({type(exc).__name__}: {exc})"
        return False
    mt5 = client
    _BRIDGE_KEY = (h, pt)
    _BRIDGE_ERROR = None
    return True


def get_backend():
    """The active MT5 API object (native module or bridge client), or None. Use this instead of importing
    MetaTrader5 so the call works on every OS."""
    return mt5


def to_local(value):
    """Bridge results (numpy rate arrays, tuples) arrive as RPyC proxies; copy them into this process so pandas
    and numpy see real data. No-op for the native backend and for plain values."""
    if backend_name() != "bridge" or value is None:
        return value
    try:
        from rpyc.utils.classic import obtain  # type: ignore
        return obtain(value)
    except Exception:  # noqa: BLE001
        return value


def backend_name() -> str:
    """'native' | 'bridge' | 'none' for the currently active backend."""
    if mt5 is None:
        return "none"
    return "native" if mt5 is _NATIVE_MT5 else "bridge"


def is_available() -> bool:
    """True if a MetaTrader5 API is usable here: the native Windows package, or the mt5linux bridge client
    package (Linux/macOS/remote). Does NOT mean a terminal is actually running/logged in -- call connect()
    to find that out."""
    return mt5 is not None or _bridge_package_available()


def unavailable_reason() -> str:
    if is_available():
        return ""
    kind = platform_kind()
    if kind == "windows":
        return (
            "The MetaTrader5 package isn't installed. Forward Test requires a running MT5 terminal and "
            "`pip install MetaTrader5` (already in config/requirements.txt on Windows)."
            + (f" Import error: {_MT5_IMPORT_ERROR}" if _MT5_IMPORT_ERROR and "No module named" not in _MT5_IMPORT_ERROR else ""))
    where = "macOS" if kind == "macos" else "Linux" if kind == "linux" else "this operating system"
    return (
        f"MetaQuotes only ships the MetaTrader5 Python package for Windows, so on {where} the app talks to the terminal "
        "through the MT5 bridge: run the MT5 terminal under Wine (or on a Windows PC/VPS), start the bridge server next "
        "to it, and `pip install mt5linux` here. Step-by-step setup is in docs/MT5_CROSS_PLATFORM.md "
        "(scripts/mt5_bridge_server.py prints the exact commands for this machine)."
    )


# Human-readable explanations for the MT5 error codes this app's users
# actually hit in practice, keyed off mt5.last_error()'s numeric code.
# -10003 in particular ("IPC initialize failed") is NOT a login/credential
# problem -- it means the MetaTrader5 Python package could not find or
# start an actual MT5 terminal executable on this machine at all, so no
# login was ever attempted. The generic message the package itself
# returns for this doesn't say any of that, which is why it reads as
# mysterious.
_MT5_ERROR_EXPLANATIONS = {
    -10003: (
        "MT5 terminal not found/started (IPC initialize failed). This means the MetaTrader5 "
        "Python package couldn't locate or launch an actual MT5 terminal on this machine -- it "
        "never got as far as trying your login. Most common causes, in order: (1) the MT5 "
        "terminal isn't installed at all -- download it from your broker's site and log into "
        "your demo account in it at least once; (2) it IS installed, but not where this package "
        "auto-detects -- enter the exact path to terminal64.exe in 'Terminal path' below (use "
        "Browse/Auto-detect); (3) a 32-bit-only terminal build, which the MetaTrader5 Python "
        "package cannot drive -- reinstall the 64-bit terminal from your broker."
    ),
    -10004: "MT5 terminal found but not responding in time -- it may still be starting up. Try again in a few seconds.",
    -10005: ("MT5 IPC timeout -- the terminal opened but didn't answer in time. The app already waited up to 2 minutes and retried. "
             "Check, in order: (1) MT5 is not showing a pop-up (login/'new account'/update/EULA) -- close it; (2) in MT5 open Tools > Options > "
             "Expert Advisors and tick 'Allow algorithmic trading'; (3) the terminal is not running as Administrator while this app is not "
             "(or the reverse) -- run both normally; (4) log into the account inside MT5 once so the server list is loaded; (5) only ONE MT5 "
             "install should be open -- leave 'Terminal path' blank to auto-detect, or Browse to the exact terminal64.exe; then click Test again."),
    -2: "Invalid MT5 login/password/server combination -- double-check them against your broker's demo account details.",
    -6: "MT5 authorization failed -- the account number, password, or server name is wrong, or this account has been disabled.",
    -1: "MT5 returned a generic internal failure -- try restarting the MT5 terminal, then Save & Test Connection again.",
}


def describe_mt5_error(code: int, desc: str) -> str:
    """Turns an mt5.last_error() (code, desc) pair into an actionable
    explanation for whichever specific failure this app's users have
    actually hit, falling back to the package's own description
    unembellished for anything not in the table above."""
    explanation = _MT5_ERROR_EXPLANATIONS.get(code)
    if explanation:
        return f"MT5 connection failed ({code}): {desc}\n\n{explanation}"
    return f"MT5 connection failed ({code}): {desc}"


def wine_prefix_candidates() -> list[str]:
    """Wine prefixes that may hold an MT5 install on Linux/macOS (existing directories only)."""
    home = os.path.expanduser("~")
    raw = [
        os.environ.get("WINEPREFIX"),
        os.path.join(home, ".wine"),
        os.path.join(home, ".mt5"),
        os.path.join(home, ".wine-mt5"),
        os.path.join(home, ".local", "share", "wineprefixes", "mt5"),
        os.path.join(home, ".local", "share", "wineprefixes", "metatrader5"),
        # MetaQuotes' own macOS app bundle keeps its prefix here
        os.path.join(home, "Library", "Application Support", "net.metaquotes.wine.metatrader5"),
        os.path.join(home, "Library", "Application Support", "MetaTrader 5"),
        os.path.join(home, "Library", "Containers", "net.metaquotes.wine.metatrader5", "Data", "Library",
                     "Application Support", "net.metaquotes.wine.metatrader5"),
        # CrossOver bottles
        *(_glob(os.path.join(home, "Library", "Application Support", "CrossOver", "Bottles", "*"))),
        *(_glob(os.path.join(home, ".cxoffice", "*"))),
        # Bottles (Flatpak) and Lutris
        *(_glob(os.path.join(home, ".var", "app", "com.usebottles.bottles", "data", "bottles", "bottles", "*"))),
        *(_glob(os.path.join(home, "Games", "*"))),
    ]
    seen, out = set(), []
    for d in raw:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def _glob(pattern: str) -> list[str]:
    import glob
    try:
        return glob.glob(pattern)
    except Exception:  # noqa: BLE001
        return []


def to_wine_path(path: str) -> str:
    """Translates a host path inside a Wine prefix (``<prefix>/drive_c/Program Files/X/terminal64.exe``) to the
    Windows path the terminal-side Python needs (``C:\\Program Files\\X\\terminal64.exe``). Windows-style paths
    and anything outside a prefix are returned unchanged."""
    if not path:
        return path
    if len(path) > 1 and path[1] == ":":
        return path
    norm = path.replace("\\", "/")
    marker = "/drive_c/"
    idx = norm.lower().find(marker)
    if idx < 0:
        return path
    rest = norm[idx + len(marker):]
    return "C:\\" + rest.replace("/", "\\")


def find_terminal_candidates() -> list[str]:
    """Best-effort search of the places an MT5 install actually lands, for when auto-detection inside the
    MetaTrader5 package itself fails (the -10003 case).

      * Windows: Program Files and per-user AppData (the paths as Windows sees them).
      * Linux/macOS: ``terminal64.exe`` inside Wine/CrossOver/MetaQuotes-app prefixes (host paths; they are
        translated to ``C:\\...`` for the bridge by ``to_wine_path`` when initialising).

    Existing paths only, most-recently-modified first so a freshly (re)installed terminal sorts ahead of stale
    leftovers. Silently returns an empty list on any error -- a convenience for the UI's 'Auto-detect' button,
    never something a caller should treat as authoritative."""
    import glob

    patterns: list[str] = []
    if platform_kind() == "windows":
        program_files = [os.environ.get("ProgramFiles", r"C:\Program Files"),
                          os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
        for pf in program_files:
            if pf:
                patterns.append(os.path.join(pf, "*", "terminal64.exe"))
                patterns.append(os.path.join(pf, "*", "*", "terminal64.exe"))
        # The far more common location in practice: MetaQuotes/brokers install
        # per-user under AppData\Roaming\MetaQuotes\Terminal\<hash>\, not
        # Program Files at all.
        appdata = os.environ.get("APPDATA")
        if appdata:
            patterns.append(os.path.join(appdata, "MetaQuotes", "Terminal", "*", "terminal64.exe"))
    else:
        for prefix in wine_prefix_candidates():
            for pf in ("Program Files", "Program Files (x86)"):
                patterns.append(os.path.join(prefix, "drive_c", pf, "*", "terminal64.exe"))
                patterns.append(os.path.join(prefix, "drive_c", pf, "*", "*", "terminal64.exe"))
            patterns.append(os.path.join(prefix, "drive_c", "users", "*", "AppData", "Roaming", "MetaQuotes",
                                         "Terminal", "*", "terminal64.exe"))
            patterns.append(os.path.join(prefix, "drive_c", "MetaTrader*", "terminal64.exe"))

    found: list[str] = []
    for pattern in patterns:
        try:
            found.extend(glob.glob(pattern))
        except Exception:
            continue

    seen = set()
    unique = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    try:
        unique.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    except Exception:
        pass
    return unique


# Maps this app's timeframe-in-minutes convention (matching the data/raw/
# CSV naming, e.g. XAUUSD15.csv = 15 minutes) to MT5's TIMEFRAME_* constants.
_TIMEFRAME_MAP = {
    1: "TIMEFRAME_M1", 5: "TIMEFRAME_M5", 15: "TIMEFRAME_M15", 30: "TIMEFRAME_M30",
    60: "TIMEFRAME_H1", 240: "TIMEFRAME_H4", 1440: "TIMEFRAME_D1",
}


def _mt5_timeframe(minutes: int):
    name = _TIMEFRAME_MAP.get(minutes)
    if name is None:
        raise ValueError(
            f"Unsupported timeframe: {minutes} minutes. Supported: {sorted(_TIMEFRAME_MAP)}."
        )
    return getattr(mt5, name)


@dataclass
class ConnectionResult:
    ok: bool
    message: str
    account_login: int | None = None
    account_server: str | None = None
    balance: float | None = None
    equity: float | None = None
    currency: str | None = None
    resolved_terminal_path: str | None = None  # set only when auto-detection found a working path the caller didn't supply


@dataclass
class OrderResult:
    ok: bool
    message: str
    ticket: int | None = None
    price: float | None = None
    volume: float | None = None


@dataclass
class OpenPosition:
    ticket: int
    symbol: str
    direction: int  # 1 long, -1 short
    volume: float
    open_price: float
    sl: float
    tp: float
    open_time: pd.Timestamp
    profit: float


class MT5Connector:
    """One connector per forward-test session. Not thread-safe against
    other simultaneous connectors in the same process (the underlying
    MetaTrader5 package is a single global connection to one terminal) --
    the Forward Test tab only ever runs one session at a time for this
    reason."""

    def __init__(self, login: str, password: str, server: str, terminal_path: str = "",
                 bridge_host: str | None = None, bridge_port: int | str | None = None):
        self.login = login
        self.password = password
        self.server = server
        self.terminal_path = terminal_path or None
        self.bridge_host = bridge_host or None
        self.bridge_port = bridge_port or None
        self._connected = False

    def _bridge_is_local(self) -> bool:
        return bridge_endpoint(self.bridge_host, self.bridge_port)[0].lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0")

    def connect(self) -> ConnectionResult:
        if not is_available():
            return ConnectionResult(ok=False, message=unavailable_reason())
        if not _activate_backend(self.bridge_host, self.bridge_port):
            h, pt = bridge_endpoint(self.bridge_host, self.bridge_port)
            return ConnectionResult(ok=False, message=(
                f"MT5 bridge unavailable: {_BRIDGE_ERROR}.\n\n"
                f"Start the bridge server next to your MT5 terminal (it must listen on {h}:{pt}) -- "
                "run `python scripts/mt5_bridge_server.py --help` for the exact commands, or see "
                "docs/MT5_CROSS_PLATFORM.md. If the terminal is on another machine, set the bridge host/port "
                "(T58_MT5_BRIDGE_HOST / T58_MT5_BRIDGE_PORT) and make sure that port is reachable."))
        try:
            login_int = int(self.login)
        except ValueError:
            return ConnectionResult(ok=False, message=f"MT5 login must be numeric, got '{self.login}'.")

        # Why "terminal didn't respond in time" (-10005/-10004) happens: initialize(login=..., password=...) asks a COLD terminal to
        # start, load its history and authorize in one call, and the package gives up after its timeout while the terminal is still
        # opening. Fix: (1) start the terminal ourselves and wait for it, (2) attach WITHOUT credentials using a long timeout and
        # retries, (3) only then log in on the already-running terminal (login() has its own timeout).
        ok, code, desc = self._attach(login_int)
        if not ok:
            return ConnectionResult(ok=False, message=describe_mt5_error(code, desc))

        info = mt5.account_info()
        if info is None:
            code, desc = mt5.last_error()
            mt5.shutdown()
            return ConnectionResult(ok=False, message=f"Connected but could not read account info ({code}): {desc}")

        self._connected = True
        return ConnectionResult(
            ok=True, message="Connected." if not self.terminal_path else f"Connected (using terminal at {self.terminal_path}).",
            account_login=info.login, account_server=info.server,
            balance=info.balance, equity=info.equity, currency=info.currency,
            resolved_terminal_path=self.terminal_path or None,
        )

    # ---- robust start-up -------------------------------------------------------------------------------------
    INIT_TIMEOUT_MS = 120_000

    def _terminal_running(self) -> bool:
        import subprocess
        try:
            if platform_kind() == "windows":
                out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq terminal64.exe"], capture_output=True, text=True, timeout=10).stdout
                return "terminal64.exe" in out.lower()
            if shutil.which("pgrep"):
                # Under Wine the process shows up as terminal64.exe (Linux) or inside the Wine host process (macOS)
                r = subprocess.run(["pgrep", "-fi", "terminal64"], capture_output=True, text=True, timeout=10)
                return r.returncode == 0 and bool(r.stdout.strip())
            out = subprocess.run(["ps", "ax"], capture_output=True, text=True, timeout=10).stdout
            return "terminal64" in out.lower()
        except Exception:
            return False

    def launch_terminal(self, wait_seconds: float = 45.0) -> str | None:
        """Starts MT5 if it is not already running (path given or auto-detected) and waits for it to come up.
        Returns the path used, or None when nothing could be started.

        Windows: runs terminal64.exe. Linux/macOS: runs it under Wine in the prefix it was found in -- only when
        the bridge is on THIS machine (a remote bridge host manages its own terminal; initialize() on the remote
        Python starts it if needed)."""
        import subprocess
        kind = platform_kind()
        if kind != "windows" and not self._bridge_is_local():
            return None
        path = self.terminal_path or (find_terminal_candidates() or [None])[0]
        if not path:
            return None
        if len(path) > 1 and path[1] == ":" and kind != "windows":
            return None          # a C:\ path from settings: only the Wine-side Python can see it; initialize() handles it
        if not os.path.exists(path):
            return None
        if not self._terminal_running():
            try:
                if kind == "windows":
                    subprocess.Popen([path], cwd=os.path.dirname(path), close_fds=True)
                else:
                    wine = shutil.which("wine64") or shutil.which("wine")
                    if wine is None:
                        return path       # nothing to launch with; the bridge-side initialize() may still start it
                    env = dict(os.environ)
                    low = path.replace("\\", "/")
                    i = low.lower().find("/drive_c/")
                    if i > 0:
                        env.setdefault("WINEPREFIX", low[:i])
                    env.setdefault("WINEDEBUG", "-all")
                    subprocess.Popen([wine, path], cwd=os.path.dirname(path), env=env, close_fds=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                return None
            deadline = time.time() + wait_seconds
            while time.time() < deadline and not self._terminal_running():
                time.sleep(1.0)
            time.sleep(min(8.0, wait_seconds / 4))     # let it finish loading before the first IPC call
        self.terminal_path = self.terminal_path or path
        return path

    def _init_once(self, path: str | None):
        kw = {"timeout": self.INIT_TIMEOUT_MS}
        if path:
            # the Python that owns the terminal is the one reading this path: under Wine/remote it must be C:\...
            kw["path"] = to_wine_path(path) if backend_name() == "bridge" else path
        ok = mt5.initialize(**kw)
        return (True, 0, "") if ok else (False, *mt5.last_error())

    def _attach(self, login_int: int, attempts: int = 4):
        """Returns (ok, code, desc). Order: attach to the running/auto-launched terminal, then authorize."""
        path = self.terminal_path
        self.launch_terminal()
        path = self.terminal_path
        ok, code, desc = False, -10003, "not attempted"
        for n in range(attempts):
            ok, code, desc = self._init_once(path)
            if not ok and code == -10003 and not path:
                for cand in find_terminal_candidates():
                    ok, code, desc = self._init_once(cand)
                    if ok:
                        self.terminal_path = path = cand
                        break
            if ok:
                break
            if code in (-10004, -10005, -10001, -1):     # terminal still starting / busy: wait and retry
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                time.sleep(4.0 + 3.0 * n)
                continue
            break
        if not ok:
            return False, code, desc
        info = mt5.account_info()
        if info is None or int(getattr(info, "login", 0) or 0) != login_int:
            authorised = False
            for n in range(3):
                try:
                    authorised = bool(mt5.login(login_int, password=self.password, server=self.server, timeout=60_000))
                except TypeError:
                    authorised = bool(mt5.login(login_int, password=self.password, server=self.server))
                if authorised:
                    break
                code, desc = mt5.last_error()
                if code not in (-10004, -10005, -1):
                    break
                time.sleep(3.0 + 2.0 * n)
            if not authorised:
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                return False, code, desc
        return True, 0, ""

    def disconnect(self) -> None:
        if is_available() and self._connected:
            try:
                mt5.shutdown()
            except Exception:
                pass
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def is_alive(self) -> bool:
        """Actually asks the terminal, instead of trusting the local
        `_connected` flag left over from the last successful connect().
        The package can report a stale `initialize()` success for a
        terminal that has since been closed, lost its network link, or
        logged out of the account -- both `terminal_info()` and
        `account_info()` return None the moment that happens, which is
        the only reliable signal available."""
        if not self._connected or not is_available():
            return False
        try:
            return mt5.terminal_info() is not None and mt5.account_info() is not None
        except Exception:
            return False

    def ensure_connected(self, max_attempts: int = 3, retry_delay_seconds: float = 2.0) -> ConnectionResult:
        """Verifies the connection is actually live and, if not,
        transparently reconnects using the same stored credentials/path
        instead of leaving the session silently dead until the user
        notices and manually restarts it. Called at the top of every
        forward-test poll. Returns the last ConnectionResult -- ok=True
        means the caller can proceed with this poll as normal."""
        if self.is_alive():
            return ConnectionResult(ok=True, message="Connected.")
        # Drop any stale handle before retrying -- initialize() on top of
        # a half-dead connection is one of the more common causes of the
        # generic -1 "internal failure" error.
        if is_available():
            try:
                mt5.shutdown()
            except Exception:
                pass
        self._connected = False
        last_result = ConnectionResult(ok=False, message="Not connected.")
        for attempt in range(1, max_attempts + 1):
            last_result = self.connect()
            if last_result.ok:
                return last_result
            if attempt < max_attempts:
                time.sleep(retry_delay_seconds)
        return last_result

    def account_summary(self) -> Optional[dict]:
        if not self._connected:
            return None
        info = mt5.account_info()
        if info is None:
            # The terminal/connection died since the last successful call --
            # flip the flag so ensure_connected() knows to actually
            # reconnect rather than silently returning None forever.
            self._connected = False
            return None
        return {
            "login": info.login, "server": info.server, "balance": info.balance,
            "equity": info.equity, "margin_free": info.margin_free, "currency": info.currency,
        }

    def symbol_point(self, symbol: str) -> float:
        """The instrument's minimum price increment -- this app's `pip_size`
        for a given symbol, read straight from the broker instead of
        guessed from price magnitude."""
        si = mt5.symbol_info(symbol)
        if si is None:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Symbol '{symbol}' not found/enabled on this MT5 account.")
            si = mt5.symbol_info(symbol)
        return float(si.point)

    def fetch_completed_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        """Returns the last `count` fully-CLOSED bars as a DataFrame with
        this app's standard timestamp/open/high/low/close/volume schema
        (see app.data.importer). The most recent still-forming bar is
        always excluded so signals never see a partial bar."""
        tf = _mt5_timeframe(timeframe_minutes)
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select symbol '{symbol}' on this MT5 account.")
        # Pull one extra bar and drop the most recent -- MT5's rates array
        # includes the currently-forming bar at index -1.
        rates = to_local(mt5.copy_rates_from_pos(symbol, tf, 0, count + 1))
        if rates is None or len(rates) < 2:
            code, desc = mt5.last_error()
            raise RuntimeError(f"Could not fetch rates for {symbol} ({code}): {desc}")
        df = pd.DataFrame(rates)
        df = df.iloc[:-1]  # drop the still-forming bar
        out = pd.DataFrame({
            "timestamp": pd.to_datetime(df["time"], unit="s", utc=True),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["tick_volume"].astype(float),
        }).reset_index(drop=True)
        return out

    def latest_completed_bar_time(self, symbol: str, timeframe_minutes: int) -> Optional[pd.Timestamp]:
        df = self.fetch_completed_bars(symbol, timeframe_minutes, 1)
        if df.empty:
            return None
        return df["timestamp"].iloc[-1]

    def get_open_positions(self, symbol: Optional[str] = None) -> list[OpenPosition]:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        out: list[OpenPosition] = []
        if not positions:
            return out
        for p in positions:
            out.append(OpenPosition(
                ticket=p.ticket, symbol=p.symbol,
                direction=1 if p.type == mt5.ORDER_TYPE_BUY else -1,
                volume=p.volume, open_price=p.price_open, sl=p.sl, tp=p.tp,
                open_time=pd.to_datetime(p.time, unit="s", utc=True), profit=p.profit,
            ))
        return out

    def place_market_order(
        self, symbol: str, direction: int, volume: float,
        sl_price: float | None = None, tp_price: float | None = None,
        comment: str = "T58 Forward Test", deviation: int = 20,
    ) -> OrderResult:
        if direction not in (1, -1):
            return OrderResult(ok=False, message=f"Invalid direction {direction}, must be 1 or -1.")
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(ok=False, message=f"No tick data for {symbol}.")
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == 1 else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if sl_price:
            request["sl"] = float(sl_price)
        if tp_price:
            request["tp"] = float(tp_price)
        result = mt5.order_send(request)
        if result is None:
            code, desc = mt5.last_error()
            return OrderResult(ok=False, message=f"order_send returned None ({code}): {desc}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(ok=False, message=f"Order rejected (retcode {result.retcode}): {result.comment}")
        return OrderResult(ok=True, message="Filled.", ticket=self.position_id_for_order(result) or result.order,
                           price=result.price, volume=result.volume)

    def close_position(self, ticket: int, comment: str = "T58 Forward Test close") -> OrderResult:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(ok=False, message=f"No open position with ticket {ticket}.")
        p = positions[0]
        tick = mt5.symbol_info_tick(p.symbol)
        if tick is None:
            return OrderResult(ok=False, message=f"No tick data for {p.symbol}.")
        is_long = p.type == mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": tick.bid if is_long else tick.ask,
            "deviation": 20,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            code, desc = mt5.last_error()
            return OrderResult(ok=False, message=f"order_send returned None ({code}): {desc}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(ok=False, message=f"Close rejected (retcode {result.retcode}): {result.comment}")
        return OrderResult(ok=True, message="Closed.", ticket=result.order, price=result.price, volume=result.volume)

    def closed_position_result(self, ticket: int) -> Optional[dict]:
        """AUDIT 2026-10-12: realized outcome of a position from the broker's
        deal history -- {"exit_price", "pnl"} where pnl = profit + commission +
        swap + fee summed over every deal of the position (entry deals carry
        commission too). Returns None when history is unavailable or the
        position has no closing deal yet. Used for broker-side SL/TP exits,
        which the forward-test journal previously recorded as exit=entry,
        pnl=0."""
        try:
            deals = mt5.history_deals_get(position=int(ticket))
        except Exception:  # noqa: BLE001
            return None
        if not deals:
            return None
        out_deals = [d for d in deals if getattr(d, "entry", None) in (mt5.DEAL_ENTRY_OUT, getattr(mt5, "DEAL_ENTRY_OUT_BY", -1))]
        if not out_deals:
            return None
        pnl = sum(float(getattr(d, "profit", 0.0)) + float(getattr(d, "commission", 0.0))
                  + float(getattr(d, "swap", 0.0)) + float(getattr(d, "fee", 0.0)) for d in deals)
        last = max(out_deals, key=lambda d: getattr(d, "time_msc", getattr(d, "time", 0)))
        return {"exit_price": float(last.price), "pnl": float(pnl)}

    def position_id_for_order(self, order_result) -> Optional[int]:
        """The POSITION ticket for a just-filled market order. On hedging
        accounts it equals the order ticket; on netting accounts it does not,
        and tracking the order ticket made the session think its position had
        vanished on the next poll."""
        try:
            deal = getattr(order_result, "deal", 0)
            if deal:
                got = mt5.history_deals_get(ticket=int(deal))
                if got:
                    return int(got[0].position_id)
        except Exception:  # noqa: BLE001
            pass
        return None

    def close_all(self, symbol: Optional[str] = None) -> list[OrderResult]:
        return [self.close_position(p.ticket) for p in self.get_open_positions(symbol)]
