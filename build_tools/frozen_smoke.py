#!/usr/bin/env python3
"""Smoke-test a FROZEN (PyInstaller) build by actually running it.

    python build_tools/frozen_smoke.py web     dist/T58-Web-App/T58-Web-App.exe
    python build_tools/frozen_smoke.py desktop dist/T58-Quant-Algo-Backtester/T58-Quant-Algo-Backtester.exe

web      starts the packaged web app, waits for its HTTP server, and requests
         the pages that need bundled templates / static files / config. Proves
         the .exe boots, every ``app`` module imports, and the data files made
         it into the bundle. (Pages behind the license gate are not needed:
         /activate, /static/* and /manifest.json are always served.)
desktop  runs ``<exe> --help``: argparse exits 0 only after app.main and its
         whole top-level import chain loaded from inside the bundle. (The
         Tk window itself needs a display, which a CI runner does not have.)

Exit code 0 = healthy. Works on Windows, macOS and Linux.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PAGES_200 = ["/activate", "/static/theme.css", "/manifest.json"]
PORTS = range(5000, 5025)  # the launcher takes the first free port from 5000


def _fetch(url: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0, ""


def _sandbox_copy(exe: Path) -> tuple[Path, Path]:
    """Copy the build folder to a temp dir and return (temp root, exe inside the copy).

    The packaged app writes runtime files next to its exe on first launch (its
    Flask secret key, logs, license state ...). Running it from the real build
    folder would leave those behind, and the release zip is made from that
    folder -- shipping one shared secret key to every customer. So the smoke
    test only ever runs a throwaway copy."""
    root = Path(tempfile.mkdtemp(prefix="t58_frozen_"))
    dest = root / exe.parent.name
    shutil.copytree(exe.parent, dest, symlinks=True)
    return root, dest / exe.name


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform.startswith("win"):
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _tail(path: Path, lines: int = 60) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no output captured)"


def smoke_web(exe: Path, timeout: float) -> int:
    sandbox, exe = _sandbox_copy(exe)
    log = sandbox / "web.log"
    with open(log, "wb") as fh:
        proc = subprocess.Popen([str(exe)], cwd=str(exe.parent), stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
    try:
        base = None
        deadline = time.time() + timeout
        while time.time() < deadline and base is None:
            if proc.poll() is not None:
                print(f"FAIL: the app exited early with code {proc.returncode}\n--- output ---\n{_tail(log)}")
                return 1
            for port in PORTS:
                status, _ = _fetch(f"http://127.0.0.1:{port}/activate", timeout=2)
                if status == 200:
                    base = f"http://127.0.0.1:{port}"
                    break
            if base is None:
                time.sleep(2)
        if base is None:
            print(f"FAIL: no HTTP server answered within {timeout:.0f}s\n--- output ---\n{_tail(log)}")
            return 1
        print(f"server is up at {base}")
        failures = []
        for page in PAGES_200:
            status, body = _fetch(base + page)
            print(f"  GET {page:<22} -> {status} ({len(body)} bytes)")
            if status != 200 or not body.strip():
                failures.append(page)
        # The dashboard is behind the license gate: 200 (licensed) or a redirect to
        # /activate (urllib follows it, so the final page is the 200 activation page).
        status, _ = _fetch(base + "/dashboard")
        print(f"  GET {'/dashboard':<22} -> {status}")
        if status != 200:
            failures.append("/dashboard")
        if failures:
            print(f"FAIL: bad response for {failures}\n--- output ---\n{_tail(log)}")
            return 1
        print("OK: frozen web app boots and serves its bundled templates and static files")
        return 0
    finally:
        _kill_tree(proc)
        shutil.rmtree(sandbox, ignore_errors=True)


def smoke_desktop(exe: Path, timeout: float) -> int:
    sandbox, exe = _sandbox_copy(exe)
    try:
        res = subprocess.run([str(exe), "--help"], cwd=str(exe.parent), capture_output=True, timeout=timeout,
                             stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        print(f"FAIL: `--help` did not finish within {timeout:.0f}s")
        return 1
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
    if res.returncode != 0:
        out = (res.stdout + res.stderr).decode("utf-8", "replace")
        print(f"FAIL: `--help` exited with code {res.returncode}\n{out[-3000:]}")
        return 1
    print("OK: frozen desktop app starts and loads app.main from the bundle")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in {"web", "desktop"}:
        print(__doc__)
        return 2
    exe = Path(argv[2]).resolve()
    if not exe.is_file():
        print(f"FAIL: executable not found: {exe}")
        return 1
    timeout = float(argv[3]) if len(argv) > 3 else 180.0
    return smoke_web(exe, timeout) if argv[1] == "web" else smoke_desktop(exe, timeout)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
