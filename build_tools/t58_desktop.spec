# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec -- T58 Quant Algo Backtester desktop app (entry point: run_app.py).

Build (any OS):   pyinstaller --noconfirm --clean build_tools/t58_desktop.spec
Output:           dist/T58-Quant-Algo-Backtester/T58-Quant-Algo-Backtester(.exe) + _internal/

ONE-FOLDER build on purpose -- see the note at the top of t58_web.spec.
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

sys.path.insert(0, str(Path(SPECPATH)))
import spec_common as common  # noqa: E402

ROOT = common.repo_root(SPECPATH)
NAME = "T58-Quant-Algo-Backtester"

datas = common.data_entries(ROOT, web=False)
binaries = []
hiddenimports = list(collect_submodules("app")) + list(common.LAZY_THIRD_PARTY)
hiddenimports += collect_submodules("keyring.backends")

for pkg in common.COLLECT_ALL:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# collect_all() also drags in each library's own test-suite (hundreds of
# pyarrow.tests.* modules/files) -- dead weight and the deepest paths in the bundle.
hiddenimports, datas = common.without_test_suites(hiddenimports, datas)

a = Analysis(
    [str(ROOT / "run_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=common.EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=False,
    strip=False,
    upx=False,          # UPX-packed exes are a classic antivirus false-positive trigger
    console=False,      # windowed Tk app: no console window
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=NAME,
)
