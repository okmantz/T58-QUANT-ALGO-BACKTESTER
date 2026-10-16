"""OS-portability of the web app's launch + data-location logic."""
from __future__ import annotations

import socket
import sys
from pathlib import Path

from app.data import storage
from app.web import network_info


def test_find_free_port_skips_a_busy_port():
    blocker = socket.socket()
    blocker.bind(("0.0.0.0", 0))
    busy = blocker.getsockname()[1]
    blocker.listen(1)
    try:
        found = network_info.find_free_port(busy)
        assert found != busy and found > busy
    finally:
        blocker.close()


def test_active_port_drives_every_url():
    original = network_info.get_active_port()
    try:
        network_info.set_active_port(5099)
        assert network_info.lan_url().endswith(":5099")
        assert network_info.lan_url(6000).endswith(":6000")  # an explicit port still wins
    finally:
        network_info.set_active_port(original)


def test_user_data_dir_uses_each_operating_systems_convention(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))

    monkeypatch.setattr(sys, "platform", "win32")
    assert storage.user_data_dir() == tmp_path / "Local" / "T58 Prop Algo Backtester"
    monkeypatch.setattr(sys, "platform", "darwin")
    assert storage.user_data_dir() == tmp_path / "Library" / "Application Support" / "T58 Prop Algo Backtester"
    monkeypatch.setattr(sys, "platform", "linux")
    assert storage.user_data_dir() == tmp_path / ".local" / "share" / "T58 Prop Algo Backtester"
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert storage.user_data_dir() == tmp_path / "xdg" / "T58 Prop Algo Backtester"


def test_read_only_install_folder_falls_back_to_user_data_dir(monkeypatch, tmp_path):
    exe = tmp_path / "ro" / "app"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    monkeypatch.setattr(storage, "user_data_dir", lambda: tmp_path / "fallback")
    assert storage.get_app_base_dir() == tmp_path / "fallback"
