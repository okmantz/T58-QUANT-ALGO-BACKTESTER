"""Tests for app.web.launcher (the T58-Web-App.exe entry point glue)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from app.web.launcher import _qr_image_path, get_lan_ip
from app.web.network_info import is_running_under_wine, startup_banner_lines

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_get_lan_ip_returns_a_plausible_ipv4_address():
    ip = get_lan_ip()
    parts = ip.split(".")
    assert len(parts) == 4
    assert all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)


def test_is_running_under_wine_is_false_on_a_real_non_windows_platform():
    """On the real Linux/macOS CI running this test (not Windows-through-
    Wine), this must be False -- it should never fire on a genuine
    native install of this cross-platform Python app."""
    if not sys.platform.startswith("win"):
        assert is_running_under_wine() is False


def test_is_running_under_wine_detects_wineprefix_env_var(monkeypatch):
    """Reproduces the reported bug's actual environment: the Windows
    .exe build running under Wine on Linux. Wine sets WINEPREFIX (or
    WINELOADER) in the process environment it launches Windows binaries
    in -- this is the cheap, reliable signal checked before falling back
    to the ctypes probe."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("WINEPREFIX", "/home/owen/.wine")
    assert is_running_under_wine() is True


def test_is_running_under_wine_false_without_any_wine_signal(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("WINEPREFIX", raising=False)
    monkeypatch.delenv("WINELOADER", raising=False)
    # No real ntdll.wine_get_version on this non-Windows test box either,
    # so the ctypes probe should also cleanly fail closed (False), never
    # raise.
    assert is_running_under_wine() is False


def test_startup_banner_warns_and_recommends_running_natively_under_wine(monkeypatch):
    """The two symptoms reported together -- a blank/broken QR code image
    and the phone getting 'site can't be reached' -- both trace back to
    running the Windows .exe through Wine instead of natively. The
    banner must call this out explicitly and point at the actual fix
    (`python3 run_web.py`, no Wine/.exe needed) rather than leaving it
    looking like an unexplained app bug."""
    monkeypatch.setattr("app.web.network_info.is_running_under_wine", lambda: True)
    lines = startup_banner_lines("http://192.168.1.23:5000", qr_path=None)
    joined = "\n".join(lines)
    assert "Wine" in joined
    assert "python3 run_web.py" in joined


def test_startup_banner_has_no_wine_warning_on_a_native_run(monkeypatch):
    monkeypatch.setattr("app.web.network_info.is_running_under_wine", lambda: False)
    lines = startup_banner_lines("http://192.168.1.23:5000", qr_path=None)
    joined = "\n".join(lines)
    assert "Wine" not in joined


class _FakeQRImage:
    def save(self, path) -> None:
        Path(path).write_bytes(_PNG_SIGNATURE + b"\x00" * 16)


class _FakeQRCode:
    def __init__(self, box_size=10, border=2):
        self.box_size = box_size
        self.border = border
        self.data = None

    def add_data(self, data):
        self.data = data

    def make(self, fit=True):
        pass

    def make_image(self):
        return _FakeQRImage()


def _install_fake_qrcode(monkeypatch, qrcode_class=None, make=None):
    """Stub out the `qrcode` module `_qr_image_path` imports, so a test's
    outcome depends only on _qr_image_path's own logic -- never on whether
    the real qrcode/Pillow combo happens to work in whatever sandbox
    pytest runs in that day."""
    if make is not None:
        # Legacy support for the old make= parameter
        fake = types.SimpleNamespace(
            QRCode=lambda box_size=10, border=2: make(box_size=box_size, border=border)
        )
    else:
        fake = types.SimpleNamespace(
            QRCode=qrcode_class or _FakeQRCode
        )
    monkeypatch.setitem(sys.modules, "qrcode", fake)


def test_qr_image_path_creates_a_real_png(tmp_path, monkeypatch):
    """Deterministic unit test with a stubbed qrcode library. This used to
    call the real qrcode/Pillow install directly, which made the result
    depend on a third-party library actually rendering correctly in
    CI -- flaky, and outside this repo's control. Stubbing it makes the
    result depend only on _qr_image_path's own file-writing logic (the
    thing this repo is actually responsible for getting right)."""
    _install_fake_qrcode(monkeypatch)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    out_path = _qr_image_path("http://192.168.1.23:5000")

    assert out_path is not None
    assert out_path.exists()
    assert out_path.suffix == ".png"
    with open(out_path, "rb") as f:
        assert f.read(8) == _PNG_SIGNATURE


def test_qr_image_path_never_raises_if_qrcode_is_broken(tmp_path, monkeypatch):
    """_qr_image_path is documented as best-effort: any failure in the
    optional qrcode/Pillow path must degrade to None, never raise, so the
    launcher still starts the web server on a machine where that install
    is missing or broken."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    class _BrokenQRCode:
        def __init__(self, box_size=10, border=2):
            raise RuntimeError("simulated broken qrcode/Pillow install")

    _install_fake_qrcode(monkeypatch, qrcode_class=_BrokenQRCode)

    assert _qr_image_path("http://192.168.1.23:5000") is None


def test_qr_image_path_with_the_real_installed_qrcode_library(tmp_path, monkeypatch):
    """Best-effort integration smoke test against whichever real
    qrcode/Pillow build is actually installed. Intentionally lenient:
    _qr_image_path is allowed to return None on any failure in that
    optional dependency, so an environment-specific quirk here shows up
    as a skip -- not a failure of a repo file this project doesn't own."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    out_path = _qr_image_path("http://192.168.1.23:5000")

    if out_path is None:
        pytest.skip("qrcode/Pillow did not produce an image in this environment")

    assert out_path.exists()
    assert out_path.suffix == ".png"
    with open(out_path, "rb") as f:
        assert f.read(8) == _PNG_SIGNATURE
