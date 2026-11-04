"""The MT5 helper (bridge) start-up path on Mac/Linux, exercised with a fake `wine` that behaves like the real thing:
it maps C:\\ into the prefix's drive_c and Z:\\ into /, can be told to lose a drive, to crash with Wine's/Python's real
error text, or to refuse to run (Apple Silicon without Rosetta). No terminal and no network are needed."""
import socket
import stat
import sys

import pytest

pytest.importorskip("rpyc")

from app.forward_test import mt5_bridge, mt5_bridge_manager as mgr

FAKE_WINE = '''#!{py}
import os, subprocess, sys
mode = os.environ.get("FAKE_WINE_MODE", "ok")
args = sys.argv[2:]                       # sys.argv[1] is the (fake) python.exe path
is_server = any(a.endswith("mt5_rpc_server.py") for a in args)
if is_server and mode == "bad_cpu":
    sys.stderr.write("bash: wine64: Bad CPU type in executable\\n"); sys.exit(86)
if is_server and mode == "always_missing":
    sys.stderr.write("python.exe: can't open file 'Z:\\\\x\\\\mt5_rpc_server.py': [Errno 2] No such file or directory\\n"); sys.exit(2)
if is_server and mode == "no_module":
    sys.stderr.write("Traceback (most recent call last):\\nModuleNotFoundError: No module named 'rpyc'\\n"); sys.exit(1)
def conv(a):
    if a.startswith("Z:"):
        return a[2:].replace("\\\\", "/")
    if a.startswith("C:\\\\") and mode != "no_c":
        return os.path.join(os.environ["WINEPREFIX"], "drive_c", a[3:].replace("\\\\", "/"))
    return a
env = dict(os.environ); env["PYTHONPATH"] = {stub!r}
os.execve({py!r}, [{py!r}] + [conv(a) for a in args], env)
'''


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    prefix = tmp_path / "pfx"
    exe = prefix / "drive_c" / "Program Files" / "MT5" / "terminal64.exe"
    exe.parent.mkdir(parents=True); exe.write_bytes(b"MZ")
    (prefix / "drive_c" / "Python311").mkdir(); (prefix / "drive_c" / "Python311" / "python.exe").write_bytes(b"MZ")
    stub = tmp_path / "stub"; stub.mkdir()
    (stub / "MetaTrader5.py").write_text("def last_error():\n    return (1,'ok')\n")
    wine = tmp_path / "wine"
    wine.write_text(FAKE_WINE.format(py=sys.executable, stub=str(stub)))
    wine.chmod(wine.stat().st_mode | stat.S_IEXEC)
    logs = tmp_path / "logs"; logs.mkdir()
    monkeypatch.setattr(mgr, "find_wine", lambda: str(wine))
    monkeypatch.setattr(mgr, "log_dir", lambda: logs)
    monkeypatch.setattr(mgr.sys, "platform", "linux")
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    yield {"prefix": prefix, "exe": str(exe), "port": port, "monkeypatch": monkeypatch}
    mgr.stop_bridge()


def _run(rig, mode, **kw):
    rig["monkeypatch"].setenv("FAKE_WINE_MODE", mode)
    return mgr.ensure_bridge_detailed("localhost", rig["port"], rig["exe"], install=False, wait=15, **kw)


def test_script_is_staged_into_wine_c_drive_and_started_from_there(rig):
    r = _run(rig, "ok")
    assert r.ok, r.message
    staged = rig["prefix"] / "drive_c" / "t58" / "mt5_rpc_server.py"
    assert staged.exists() and staged.read_text() == mgr.SERVER_SCRIPT.read_text()
    log = mgr.bridge_log_path().read_text()
    assert "(C: copy)" in log and "C:\\t58\\mt5_rpc_server.py" in log
    assert mt5_bridge.MT5BridgeClient("127.0.0.1", rig["port"]).last_error() == (1, "ok")


def test_falls_back_to_z_path_when_wine_cannot_read_c_copy(rig):
    r = _run(rig, "no_c")
    assert r.ok, r.message
    log = mgr.bridge_log_path().read_text()
    assert "(C: copy)" in log and "(Z: path)" in log


def test_exit_code_2_is_explained_not_just_reported(rig):
    """The exact user-facing failure: 'bridge process exited immediately (code 2)'. Now it names the cause and carries the log."""
    r = _run(rig, "always_missing")
    assert not r.ok and r.problem.kind == "bridge_file_unreachable"
    assert "exit code 2" in r.problem.technical and "can't open file" in r.problem.technical
    assert r.problem.steps and "Wine couldn't open" in r.problem.title
    assert "can't open file" in mgr.read_log_tail()          # the helper's own words are kept in the log file
    assert "code 2" not in r.problem.title


def test_rosetta_problem_is_recognised_and_not_retried(rig):
    r = _run(rig, "bad_cpu")
    assert not r.ok and r.problem.kind == "needs_rosetta"
    assert any("softwareupdate --install-rosetta" in c for c in r.problem.commands)
    assert mgr.bridge_log_path().read_text().count("starting MT5 helper") == 1


def test_missing_package_in_helper_is_named(rig):
    r = _run(rig, "no_module")
    assert not r.ok and r.problem.kind == "bridge_packages_missing" and "rpyc" in r.problem.title


def test_no_terminal_anywhere_gives_install_instructions(monkeypatch, tmp_path):
    from app.forward_test import mt5_connector as mc
    monkeypatch.setattr(mgr, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(mgr, "find_wine", lambda: "/usr/bin/wine")
    monkeypatch.setattr(mgr.sys, "platform", "linux")
    monkeypatch.setattr(mc, "find_terminal_candidates", lambda: [])
    r = mgr.ensure_bridge_detailed("localhost", 18812, None, install=False)
    assert not r.ok and r.problem.kind == "mt5_not_found" and any("terminal64.exe" in s for s in r.problem.steps)


def test_wine_missing_is_os_specific(monkeypatch):
    from app.forward_test import mt5_connector as mc
    monkeypatch.setattr(mgr, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(mgr, "find_wine", lambda: None)
    monkeypatch.setattr(mgr.sys, "platform", "darwin")
    monkeypatch.setattr(mc, "platform_kind", lambda: "macos")
    r = mgr.ensure_bridge_detailed("localhost", 18812)
    assert r.problem.kind == "wine_missing" and any("brew install" in c for c in r.problem.commands)
    monkeypatch.setattr(mc, "platform_kind", lambda: "linux")
    r = mgr.ensure_bridge_detailed("localhost", 18812)
    assert any("apt install" in c for c in r.problem.commands)


def test_legacy_tuple_api_still_works(monkeypatch):
    monkeypatch.setattr(mgr, "port_open", lambda *a, **k: True)
    assert mgr.ensure_bridge("localhost", 1) == (True, "Bridge is running.")
