"""One-click public address for the TradingView webhook relay.

TradingView's servers can only call a public https address on port 443 (or http on port 80), so the app's
own ``http://127.0.0.1:<port>`` address can never work. A Cloudflare "quick tunnel" gives this computer a
public https address with no account and no port forwarding::

    cloudflared tunnel --url http://127.0.0.1:<port>

This module starts and stops that process for the person, reads the ``https://<words>.trycloudflare.com``
address it prints, and hands it to ``config.set_public_url`` so the webhook URLs on the page fill in by
themselves. The only prerequisite is the free ``cloudflared`` program being installed (``install_hint``
says how on this operating system). Nothing here is required: ngrok / Tailscale Funnel still work by pasting
their address into the page.

No cloudflared? There is a no-install fallback: ``ssh`` (already part of Windows 10+, macOS and Linux) can open a
free tunnel through localhost.run with nothing to download::

    ssh -R 80:127.0.0.1:<port> nokey@localhost.run

``start(..., provider="auto")`` uses cloudflared when it is installed and ssh otherwise.

The address changes every time the tunnel is started (that is how quick tunnels work), so after a restart the
person needs to start it again and then paste the NEW webhook URL into the TradingView alert.
"""
from __future__ import annotations

import atexit
import re
import shutil
import subprocess
import sys
import threading
from collections import deque
from typing import Callable, Optional

_URL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.(?:trycloudflare\.com|lhr\.life|localhost\.run)", re.IGNORECASE)
_SSH_HOST = "nokey@localhost.run"
_LABELS = {"cloudflared": "cloudflared", "ssh": "ssh (localhost.run)"}

_LOCK = threading.RLock()
_PROC: Optional[subprocess.Popen] = None
_STATE = "stopped"          # stopped | starting | running | error
_URL = ""
_ERROR = ""
_PROVIDER = ""
_LOG: deque = deque(maxlen=40)


class TunnelError(RuntimeError):
    """The tunnel could not be started (message is meant for the person)."""


def find_binary() -> Optional[str]:
    return shutil.which("cloudflared") or shutil.which("cloudflared.exe")


def find_ssh() -> Optional[str]:
    return shutil.which("ssh") or shutil.which("ssh.exe")


def pick_provider(preferred: str = "auto") -> Optional[str]:
    """"cloudflared" or "ssh" (whichever is usable), or None when neither is."""
    preferred = (preferred or "auto").lower()
    if preferred == "cloudflared":
        return "cloudflared" if find_binary() else None
    if preferred == "ssh":
        return "ssh" if find_ssh() else None
    if find_binary():
        return "cloudflared"
    return "ssh" if find_ssh() else None


def install_hint() -> str:
    if sys.platform.startswith("win"):
        return "Open a terminal and run:  winget install --id Cloudflare.cloudflared   (then restart T58)"
    if sys.platform == "darwin":
        return "Open Terminal and run:  brew install cloudflared   (then restart T58)"
    return ("Install cloudflared (free): https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
            "   (then restart T58)")


def status() -> dict:
    with _LOCK:
        running = _PROC is not None and _PROC.poll() is None
        state = _STATE
        if state in ("starting", "running") and not running:
            state = "stopped" if state == "running" else "error"
        return {
            "available": pick_provider() is not None, "cloudflared_available": find_binary() is not None,
            "ssh_available": find_ssh() is not None, "provider": _PROVIDER, "install_hint": install_hint(),
            "state": state, "url": _URL if state == "running" else "", "error": _ERROR, "log": list(_LOG)[-8:],
        }


def _reader(proc: subprocess.Popen, on_url: Callable[[str], None]) -> None:
    global _STATE, _URL, _ERROR
    try:
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if not line:
                continue
            with _LOCK:
                _LOG.append(line[-300:])
            m = _URL_RE.search(line)
            if m and proc is _PROC:
                url = m.group(0).lower()
                try:
                    on_url(url)
                except Exception as exc:  # noqa: BLE001 -- the tunnel is still up even if saving failed
                    with _LOCK:
                        _ERROR = f"The tunnel is up at {url} but the address could not be saved: {exc}"
                with _LOCK:
                    _URL, _STATE = url, "running"
    except Exception:  # noqa: BLE001
        pass
    finally:
        with _LOCK:
            if proc is _PROC and _STATE != "stopped":
                if _STATE == "starting":
                    _STATE, _ERROR = "error", _ERROR or f"{_LABELS.get(_PROVIDER, 'The tunnel')} exited before it printed a public address."
                else:
                    _STATE = "stopped"


def start(port: int, on_url: Callable[[str], None], provider: str = "auto") -> dict:
    """Start a tunnel to ``http://127.0.0.1:<port>``. Returns ``status()`` immediately; the address shows
    up in a few seconds (poll ``status()``). ``provider``: "auto" (cloudflared if installed, else ssh),
    "cloudflared" or "ssh"."""
    global _PROC, _STATE, _URL, _ERROR, _PROVIDER
    chosen = pick_provider(provider)
    if chosen is None:
        if (provider or "auto").lower() == "ssh":
            raise TunnelError("ssh was not found on this computer. Windows 10+, macOS and Linux normally include it; "
                              "on Windows turn on 'OpenSSH Client' under Settings > Optional features.")
        raise TunnelError("Neither cloudflared nor ssh was found. " + install_hint())
    port = int(port)
    if chosen == "ssh":
        cmd = [find_ssh(), "-T", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
               "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes",
               "-R", f"80:127.0.0.1:{port}", _SSH_HOST]
    else:
        cmd = [find_binary(), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]
    with _LOCK:
        if _PROC is not None and _PROC.poll() is None:
            return status()
        _STATE, _URL, _ERROR, _PROVIDER = "starting", "", "", chosen
        _LOG.clear()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, bufsize=1)
    except OSError as exc:
        with _LOCK:
            _STATE, _ERROR = "error", str(exc)
        raise TunnelError(f"Could not start {_LABELS[chosen]}: {exc}") from exc
    with _LOCK:
        _PROC = proc
    threading.Thread(target=_reader, args=(proc, on_url), daemon=True, name="t58-tunnel-reader").start()
    return status()


def stop() -> dict:
    global _PROC, _STATE, _URL
    with _LOCK:
        proc, _PROC = _PROC, None
        _STATE, _URL = "stopped", ""
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
    return status()


atexit.register(stop)     # never leave a tunnel process running after T58 closes
