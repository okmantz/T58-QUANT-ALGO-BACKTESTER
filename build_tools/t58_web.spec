# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec -- T58 Web App (entry point: run_web.py).

Build (any OS):   pyinstaller --noconfirm --clean build_tools/t58_web.spec
Output:           dist/T58-Web-App/T58-Web-App(.exe) + dist/T58-Web-App/_internal/

ONE-FOLDER build on purpose. The old one-file build had to unpack several
hundred MB (pandas, numpy, scipy, scikit-learn, lightgbm, pyarrow ...) into a
temp folder on every launch; antivirus or a low-disk %TEMP% turned that into
"Failed to extract ... decompression resulted in return code -1" errors, and
every launch was slow. A folder build is unpacked once, by the zip.
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

sys.path.insert(0, str(Path(SPECPATH)))
import spec_common as common  # noqa: E402

ROOT = common.repo_root(SPECPATH)
NAME = "T58-Web-App"

datas = common.data_entries(ROOT, web=True)
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
    [str(ROOT / "run_web.py")],
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
    console=True,       # the web edition prints its address/QR banner in a console window
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=NAME,
)
