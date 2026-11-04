"""Plain-language explanations for everything that can stop an MT5 forward test from starting.

Every failure on the way to "your strategy is trading" is turned into a ``Problem``: a short title, one sentence
saying what happened in normal words, numbered steps that fix it, optional copy-paste commands, and the raw
technical text kept separately (collapsed in the UI, appended in plain-text form for the desktop app and logs).

``preflight()`` runs the same checks BEFORE the user presses Start, so the page can say "ready" or "this one thing
needs you" instead of discovering it at click time. Nothing here changes any setting or launches anything; it only
looks (the one exception is ``deep=True``, which asks the Wine Python whether its packages import).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

DOWNLOAD_MT5_URL = "https://www.metatrader5.com/en/download"
PY_INSTALLER_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"


@dataclass
class Problem:
    kind: str
    title: str
    summary: str
    steps: list = field(default_factory=list)
    commands: list = field(default_factory=list)
    technical: str = ""
    retry_ok: bool = True          # True = pressing Start again can help once the steps are done

    def to_dict(self) -> dict:
        return {"kind": self.kind, "title": self.title, "summary": self.summary, "steps": list(self.steps),
                "commands": list(self.commands), "technical": self.technical, "retry_ok": self.retry_ok}

    def plain_text(self) -> str:
        out = [self.title, "", self.summary]
        if self.steps:
            out += ["", "How to fix it:"] + [f"  {i}. {s}" for i, s in enumerate(self.steps, 1)]
        if self.commands:
            out += ["", "Command to copy:"] + [f"  {c}" for c in self.commands]
        if self.technical:
            out += ["", "Technical details: " + self.technical.strip()]
        return "\n".join(out)


def _os_kind() -> str:
    from app.forward_test import mt5_connector as mc
    return mc.platform_kind()


def _tail(text: str, n: int = 700) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else "..." + text[-n:]


# --------------------------------------------------------------------------------------------------------------
# Problems raised while preparing the connection helper (the "bridge") on macOS / Linux
# --------------------------------------------------------------------------------------------------------------

def wine_missing() -> Problem:
    if _os_kind() == "macos":
        return Problem(
            "wine_missing", "MetaTrader 5 needs a free helper called Wine, and it isn't installed",
            "MetaTrader 5 is a Windows program. On a Mac it runs through Wine, and T58 couldn't find Wine on this computer.",
            ["Easiest: download the official MetaTrader 5 for Mac from " + DOWNLOAD_MT5_URL + " (or from your broker), "
             "install it, open it once and log in to your demo account. It brings Wine with it.",
             "Already use Wine another way? In Terminal run the command below, then install your broker's MT5 into it.",
             "Come back here and press Start again. T58 does the rest."],
            ["brew install --cask --no-quarantine wine-stable"])
    return Problem(
        "wine_missing", "MetaTrader 5 needs a free helper called Wine, and it isn't installed",
        "MetaTrader 5 is a Windows program. On Linux it runs through Wine, and T58 couldn't find Wine on this computer.",
        ["Install Wine with the command below (Debian/Ubuntu shown; Fedora: sudo dnf install wine).",
         "Install your broker's MetaTrader 5 into Wine, open it once and log in to your demo account.",
         "Come back here and press Start again."],
        ["sudo apt install wine64"])


def mt5_not_found() -> Problem:
    return Problem(
        "mt5_not_found", "T58 couldn't find MetaTrader 5 on this computer",
        "T58 looks in the usual places for the MT5 program and found nothing, so it can't open it for you.",
        ["Install MetaTrader 5 from your broker or prop firm (or " + DOWNLOAD_MT5_URL + ").",
         "Open it once and log in to your demo account (File > Login to Trade Account).",
         "Press Start again. If MT5 is installed somewhere unusual, open Settings > API Keys, edit this account and "
         "paste the full path to terminal64.exe into 'Terminal path'."])


def python_setup_failed(prefix: str, detail: str) -> Problem:
    return Problem(
        "python_setup_failed", "T58 couldn't finish its one-time setup inside Wine",
        "To talk to MT5, T58 installs a small Windows Python inside Wine the first time. That install didn't complete "
        "(often a blocked or missing internet connection).",
        ["Check that this computer is online, then press Start again. The setup resumes.",
         f"Still failing? Download {PY_INSTALLER_URL} and run the command below once, then press Start again."],
        [f'WINEPREFIX="{prefix}" wine python-3.11.9-amd64.exe /quiet InstallAllUsers=0 PrependPath=1'],
        technical=_tail(detail))


def packages_failed(prefix: str, py: str, detail: str) -> Problem:
    return Problem(
        "packages_failed", "T58 couldn't install the MT5 connection packages",
        "The one-time install of the MetaTrader5 Python package inside Wine did not complete (usually no internet or a "
        "proxy/firewall blocking pip).",
        ["Check that this computer is online, then press Start again. The install resumes.",
         "Still failing? Run the command below once in Terminal, then press Start again."],
        [f'WINEPREFIX="{prefix}" wine "{py}" -m pip install --upgrade MetaTrader5 "rpyc>=5.3,<7" numpy'],
        technical=_tail(detail))


def bridge_failed(returncode, log_tail: str, prefix: str = "", log_path: str = "") -> Problem:
    """The helper process exited, or never opened its port. The wording depends on what its own log says."""
    low = (log_tail or "").lower()
    technical = f"exit code {returncode}. " + _tail(log_tail, 900)
    if "bad cpu type" in low:
        return Problem(
            "needs_rosetta", "Your Mac needs Rosetta before Wine can run",
            "Wine is built for Intel chips and this Mac needs Apple's Rosetta translator to run it.",
            ["Run the command below in Terminal (one time, takes a minute), then press Start again."],
            ["softwareupdate --install-rosetta --agree-to-license"], technical=technical)
    if "can't open file" in low or "no such file or directory" in low:
        return Problem(
            "bridge_file_unreachable", "Wine couldn't open T58's helper file",
            "Wine started, but it couldn't read the small helper script T58 gave it. T58 now copies that script into "
            "Wine's own C: drive to avoid this; if you still see this, Wine's drive mapping is unusual.",
            ["Press Start again (T58 retries with a different file location).",
             "Still failing? Open Terminal, run the command below, choose Drive Mappings, and make sure a Z: drive pointing "
             "to / exists, then press Start again."],
            [f'WINEPREFIX="{prefix}" winecfg'] if prefix else [], technical=technical)
    m = re.search(r"No module named '?([A-Za-z0-9_\.]+)", log_tail or "")
    if m:
        return Problem(
            "bridge_packages_missing", f"The MT5 helper is missing a package ({m.group(1)})",
            "The Python inside Wine is missing something T58 needs, usually because the one-time install was interrupted.",
            ["Press Start again. T58 reinstalls the packages automatically.",
             "Still failing? Run the command below once, then press Start again."],
            [f'WINEPREFIX="{prefix}" wine python -m pip install --upgrade MetaTrader5 "rpyc>=5.3,<7" numpy'] if prefix else [],
            technical=technical)
    if "address already in use" in low or "10048" in low or "only one usage" in low:
        return Problem(
            "bridge_port_busy", "Port 18812 is already used by another program",
            "T58's MT5 helper needs one local network port and something else is holding it.",
            ["Quit any other MT5 helper or T58 window that is open, or restart your computer, then press Start again."],
            technical=technical)
    return Problem(
        "bridge_exit", "T58's MT5 helper stopped right after it started",
        "T58 started its small MT5 helper program inside Wine, but Wine closed it again straight away. The helper's own "
        "error is shown under Technical details (also saved to " + (log_path or "data/logs/mt5_bridge.log") + ").",
        ["Press Start again. A first start sometimes fails while Wine finishes setting itself up.",
         "Still failing? Quit MetaTrader 5 and Wine completely (restarting the computer does this), then press Start again.",
         "If it keeps happening, send the Technical details text to support: it names the exact cause."],
        technical=technical)


def bridge_timeout(wait: int, log_tail: str = "") -> Problem:
    return Problem(
        "bridge_timeout", "T58's MT5 helper didn't come up in time",
        f"The helper was started but still wasn't answering after {wait} seconds.",
        ["Wine can be slow the very first time. Press Start again and wait a little longer.",
         "If it repeats, restart the computer (this clears stuck Wine processes), then press Start again."],
        technical=_tail(log_tail, 900))


def remote_unreachable(host: str, port: int) -> Problem:
    return Problem(
        "remote_unreachable", f"Can't reach the MT5 helper on {host}",
        f"You've pointed T58 at another computer ({host}:{port}) for MT5, and nothing is answering there.",
        [f"On that computer start T58's helper: python mt5_rpc_server.py --host 0.0.0.0 --port {port} --token <secret>.",
         "Make sure the same secret is set here as T58_MT5_BRIDGE_TOKEN and the port is open in its firewall.",
         "Or clear the bridge host in the account settings to use MT5 on this computer."],
        retry_ok=True)


def windows_package_missing(import_error: str = "") -> Problem:
    return Problem(
        "windows_package_missing", "The MetaTrader5 Python package isn't available",
        "On Windows, T58 talks to MT5 through MetaQuotes' MetaTrader5 package, and it can't be loaded.",
        ["Run the command below in the same Python that runs T58, then restart T58.",
         "If pip says no matching version exists, install Python 3.11 or 3.12 (the package supports those) and run T58 with it."],
        ["pip install MetaTrader5"], technical=_tail(import_error))


def rpyc_missing() -> Problem:
    return Problem(
        "rpyc_missing", "One Python package is missing: rpyc",
        "T58 uses a tiny package called rpyc to talk to MT5 on Mac and Linux, and it isn't installed here.",
        ["Run the command below in the same Python that runs T58, then restart T58."],
        ['pip install "rpyc>=5.3,<7"'])


def protocol_mismatch(detail: str) -> Problem:
    return Problem(
        "protocol_mismatch", "An old MT5 helper is still running",
        "A helper from an earlier T58 version is still open and doesn't match this version.",
        ["Quit MetaTrader 5 and Wine completely (or restart the computer), then press Start again."],
        technical=_tail(detail))


def bridge_unreachable(host: str, port: int, exc_text: str) -> Problem:
    low = (exc_text or "").lower()
    if "protocol mismatch" in low:
        return protocol_mismatch(exc_text)
    if "no module named" in low or "importerror" in low or "modulenotfounderror" in low:
        return Problem(
            "bridge_packages_missing", "The MT5 helper is missing a package",
            "The helper started but couldn't load the MetaTrader5 package inside Wine.",
            ["Press Start again. If it repeats, quit MT5/Wine completely and try once more."],
            technical=_tail(exc_text) + "  (Check from a terminal: python scripts/mt5_bridge_server.py --check)")
    return Problem(
        "bridge_unreachable", "T58 couldn't talk to its MT5 helper",
        f"The helper didn't answer at {host}:{port}.",
        ["Press Start again. T58 restarts the helper if it stopped.",
         "If it repeats, quit MetaTrader 5 and Wine completely (or restart the computer), then press Start again."],
        technical=_tail(f"could not reach the MT5 bridge at {host}:{port} ({exc_text}). "
                        "To test from a terminal: python scripts/mt5_bridge_server.py --check"))


# --------------------------------------------------------------------------------------------------------------
# Problems raised while connecting to the terminal / logging in
# --------------------------------------------------------------------------------------------------------------

def problem_from_mt5_error(code: int, desc: str) -> Problem:
    tech = f"MT5 error {code}: {desc}"
    if code == -10003:
        return Problem(
            "terminal_not_found", "T58 couldn't find or open MetaTrader 5",
            "No MT5 program could be started, so no login was even tried.",
            ["Install MetaTrader 5 (64-bit) from your broker and open it once, logging in to your demo account.",
             "Make sure only one MT5 is installed. If it's in an unusual place, open Settings > API Keys, edit this account and "
             "paste the full path to terminal64.exe into 'Terminal path'.",
             "Press Start again."], technical=tech)
    if code in (-10004, -10005):
        return Problem(
            "terminal_slow", "MetaTrader 5 opened but didn't answer in time",
            "T58 waited up to 2 minutes and retried four times. MT5 is usually stuck behind a pop-up window.",
            ["Look at the MT5 window and close any pop-up (login, 'open an account', update, licence).",
             "Run MT5 and T58 as the same normal user. Don't run one of them as Administrator.",
             "Keep only one MT5 open, and log in to your account inside MT5 once so the server list loads.",
             "Press Start again."], technical=tech)
    if code in (-2, -6) or "authoriz" in (desc or "").lower() or "invalid account" in (desc or "").lower():
        return Problem(
            "login_failed", "MT5 rejected the account details",
            "MT5 started, but the login number, password or server name wasn't accepted.",
            ["In MT5 open File > Login to Trade Account and compare the login number and server name letter for letter.",
             "Use the TRADING password. The read-only 'investor' password logs in but can't place trades.",
             "Fix them under Settings > API Keys > Brokers/Prop Firms, save, and press Start again."], technical=tech)
    if code == -1:
        return Problem(
            "terminal_error", "MetaTrader 5 returned a general error",
            "MT5 reported an internal failure without saying why.",
            ["Quit MetaTrader 5 completely, then press Start again (T58 reopens it).",
             "If it repeats, restart the computer and try once more."], technical=tech)
    return Problem(
        "connect_failed", "T58 couldn't connect to MetaTrader 5",
        f"MT5 answered with an error: {desc}.",
        ["Quit MetaTrader 5 completely, then press Start again.",
         "Make sure you're logged in to your account inside MT5 and you can see live prices."], technical=tech)


def algo_trading_off() -> Problem:
    return Problem(
        "algo_trading_off", "Algo Trading is switched off in MetaTrader 5",
        "T58 is connected, but MT5 is set to refuse automated orders, so your strategy couldn't place a trade.",
        ["In the MT5 window, click the 'Algo Trading' button in the top toolbar so it turns green.",
         "If the button stays red: Tools > Options > Expert Advisors > tick 'Allow algorithmic trading' > OK.",
         "Press Start again."])


def symbol_not_found(symbol: str) -> Problem:
    return Problem(
        "symbol_not_found", f"Your broker doesn't have a symbol called '{symbol}'",
        "Brokers name instruments differently (for example XAUUSD, XAUUSD.r or GOLD).",
        ["In MT5 press Ctrl+M to open Market Watch, right-click it > Symbols, and search for the instrument.",
         "Type the symbol here exactly as MT5 spells it (capital letters and any .suffix included), then press Start again."])


def login_not_numeric(login: str) -> Problem:
    return Problem(
        "login_not_numeric", "The MT5 login number isn't a number",
        f"The saved account has login '{login}', but an MT5 login is digits only.",
        ["Open Settings > API Keys > Brokers/Prop Firms, edit the account and enter the login number from MT5 "
         "(File > Login to Trade Account), then press Start again."])


def account_missing() -> Problem:
    return Problem(
        "account_missing", "No MT5 account is selected",
        "T58 needs the login details of the MT5 account it should trade on.",
        ["Open Settings > API Keys > Brokers/Prop Firms and add your MT5 account (login number, password, server).",
         "Come back here, pick it in the 'MT5 account' box and press Start."], retry_ok=False)


def strategy_missing() -> Problem:
    return Problem(
        "strategy_missing", "Pick a strategy to forward test",
        "T58 needs to know which saved strategy should trade.",
        ["Choose one in the 'Saved strategy' box (Python, PineScript or MQL5), then press Start."], retry_ok=False)


def strategy_error(message: str) -> Problem:
    return Problem(
        "strategy_error", "T58 couldn't load that strategy",
        "The saved strategy raised an error when T58 tried to load it, before MT5 was involved. The error is below.",
        ["Open the strategy in Create or the Library, fix it (or run a backtest on it to see the same error), then press Start again.",
         "Or pick a different saved strategy."],
        technical=_tail(message, 1200), retry_ok=False)


def unknown_problem(message: str) -> Problem:
    return Problem(
        "unknown", "Something unexpected stopped the forward test from starting",
        "T58 hit an error it doesn't have a specific fix for. The details are below.",
        ["Press Start again.", "If it repeats, quit MetaTrader 5 completely and try once more, or send the Technical details to support."],
        technical=_tail(message, 1200))


def order_help(retcode: int) -> str:
    """One plain sentence for the MT5 order retcodes people actually hit (empty string = no extra help)."""
    return {
        10004: "The price moved (requote). T58 will try again on the next signal.",
        10006: "The broker refused the order.",
        10013: "The order request was invalid.",
        10014: "The broker rejected the position size: it is below the minimum lot, above the maximum, or not a valid step.",
        10015: "The order price was invalid.",
        10016: "The stop-loss or take-profit is too close to the current price for this broker.",
        10017: "Trading is disabled for this symbol or account.",
        10018: "The market is closed for this symbol right now.",
        10019: "Not enough free margin on the account for this size.",
        10020: "Prices changed before the order could fill.",
        10021: "No price quotes are available for this symbol yet.",
        10026: "Automated trading is disabled on the broker's server for this account.",
        10027: "Algo Trading is switched off in MT5: click the 'Algo Trading' button in the toolbar so it turns green.",
        10030: "The broker doesn't accept this order-filling type for the symbol.",
        10031: "MT5 has no connection to the broker's server.",
        10032: "This broker only allows trading on real accounts.",
    }.get(int(retcode), "")


# --------------------------------------------------------------------------------------------------------------
# Pre-flight: what is ready, what T58 will do by itself, what needs the user
# --------------------------------------------------------------------------------------------------------------

def _check(cid, label, status, detail, fix=None, commands=None) -> dict:
    return {"id": cid, "label": label, "status": status, "detail": detail,
            "fix": list(fix or []), "commands": list(commands or [])}


def preflight(terminal_path: str = "", deep: bool = False) -> dict:
    """Looks (never changes anything) at everything the forward test needs and returns
    {os, ready, headline, first_time_setup, checks:[{id,label,status,detail,fix,commands}]}.
    status: ok = done; auto = T58 does it by itself when you press Start; action = you need to do something."""
    from app.forward_test import mt5_bridge_manager as mgr
    from app.forward_test import mt5_connector as mc
    kind = mc.platform_kind()
    checks: list = []
    first_time = False
    cands = []
    try:
        cands = mc.find_terminal_candidates()
    except Exception:  # noqa: BLE001
        pass
    tpath = (terminal_path or "").strip()
    term_known = bool(cands) or (bool(tpath) and (len(tpath) > 1 and tpath[1] == ":" or os.path.exists(tpath)))

    if kind == "windows":
        native = mc._NATIVE_MT5 is not None
        if native:
            checks.append(_check("package", "MetaTrader5 Python package", "ok", "Installed."))
        else:
            p = windows_package_missing(mc._MT5_IMPORT_ERROR or "")
            checks.append(_check("package", "MetaTrader5 Python package", "action", p.summary, p.steps, p.commands))
    else:
        if mc._bridge_package_available():
            checks.append(_check("rpyc", "Connection package (rpyc)", "ok", "Installed."))
        else:
            p = rpyc_missing()
            checks.append(_check("rpyc", "Connection package (rpyc)", "action", p.summary, p.steps, p.commands))
        wine = mgr.find_wine()
        if wine:
            checks.append(_check("wine", "Wine (runs MetaTrader 5)", "ok", wine))
        else:
            p = wine_missing()
            checks.append(_check("wine", "Wine (runs MetaTrader 5)", "action", p.summary, p.steps, p.commands))

    if term_known:
        shown = tpath or (cands[0] if cands else "")
        checks.append(_check("terminal", "MetaTrader 5 installed", "ok", shown))
    else:
        p = mt5_not_found()
        checks.append(_check("terminal", "MetaTrader 5 installed", "action", p.summary, p.steps))

    if kind != "windows":
        h, port = mc.bridge_endpoint()
        prefix = mgr.prefix_of(tpath or (cands[0] if cands else None)) if (tpath or cands) else None
        py = mgr.find_wine_python(prefix) if prefix else None
        if mgr.port_open(h, port):
            checks.append(_check("bridge", "T58 MT5 helper", "ok", f"Running on {h}:{port}."))
        elif not mgr.is_local_host(h):
            p = remote_unreachable(h, port)
            checks.append(_check("bridge", "T58 MT5 helper", "action", p.summary, p.steps))
        else:
            ready = False
            if py and deep and mgr.find_wine():
                ready = mgr.packages_ready(mgr.find_wine(), prefix, py)
            if py and (ready or not deep):
                checks.append(_check("bridge", "T58 MT5 helper", "auto",
                                     "Not running yet. T58 starts it for you when you press Start (a few seconds)."))
            else:
                first_time = True
                checks.append(_check("bridge", "T58 MT5 helper", "auto",
                                     "First-time setup: when you press Start, T58 installs a small Python inside Wine and the MT5 "
                                     "connection packages. This needs internet and takes about 2 to 5 minutes, once."))

    actions = [c for c in checks if c["status"] == "action"]
    if actions:
        headline = (f"1 thing needs you before you can start: {actions[0]['label']}." if len(actions) == 1
                    else f"{len(actions)} things need you before you can start.")
    elif first_time:
        headline = "Ready. The first Start also does a one-time setup (about 2 to 5 minutes), then it's instant."
    else:
        headline = "Ready. Press Start and T58 opens MT5, logs in and starts your strategy."
    return {"os": kind, "ready": not actions, "headline": headline, "first_time_setup": first_time, "checks": checks}
