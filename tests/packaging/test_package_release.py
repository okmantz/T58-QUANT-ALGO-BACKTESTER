"""Release packaging (build_tools/) -- the zips customers download must extract cleanly.

Covers the file-name checks that stop a Windows-unextractable zip from shipping,
the zip writer (forward slashes, CRC test, .sha256 file), the PyInstaller spec
helpers, the updater's choice of release asset, and the shape of the build
workflows. Pure-Python: no PyInstaller, no Windows runner needed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pr = _load("t58_package_release", "build_tools/package_release.py")
sc = _load("t58_spec_common", "build_tools/spec_common.py")


def _entries(tmp_path: Path, names: list[str]) -> list[tuple[Path, str]]:
    out = []
    for n in names:
        p = tmp_path / ("f_" + re.sub(r"[^A-Za-z0-9]", "_", n))
        p.write_bytes(b"x")
        out.append((p, n))
    return out


# --------------------------------------------------------------------------- #
# validate_names
# --------------------------------------------------------------------------- #
def test_clean_names_pass(tmp_path):
    names = ["T58.exe", "_internal/python313.dll", "data/raw/ES1!/futures_ES.F_1m.parquet",
             "data/raw/S&P 500/US500.csv", "strategies/python/NQ1! Gap Drive 1m.py"]
    assert pr.validate_names(_entries(tmp_path, names)) == []


@pytest.mark.parametrize("bad", [
    "data/raw/what?.csv", 'data/raw/a"b.csv', "data/raw/a:b.csv", "data/raw/a<b>.csv",
    "data/raw/trailing.", "data/raw/trailing ", "data/CON/x.csv", "data/raw/nul.txt", "data/aux",
    "../escape.txt", "/abs.txt", "data//double.csv",
])
def test_windows_unextractable_names_are_rejected(tmp_path, bad):
    assert pr.validate_names(_entries(tmp_path, [bad])), f"{bad!r} should have been flagged"


def test_case_only_collision_is_rejected(tmp_path):
    problems = pr.validate_names(_entries(tmp_path, ["data/raw/EURUSD5.csv", "data/raw/eurusd5.csv"]))
    assert any("case-only collision" in p for p in problems)


def test_over_long_path_is_rejected(tmp_path):
    long_name = "_internal/" + "/".join(["d" * 40] * 5) + "/file.py"
    problems = pr.validate_names(_entries(tmp_path, [long_name]), zip_stem="T58-Quant-Algo-Backtester-Windows")
    assert any("too long" in p for p in problems)


# --------------------------------------------------------------------------- #
# write_zip
# --------------------------------------------------------------------------- #
def test_write_zip_roundtrip(tmp_path):
    entries = []
    for arc, data in [("a.exe", b"MZ" * 100), ("_internal/lib.dll", b"\x00" * 5000),
                      ("data/raw/X/x.parquet", bytes(range(256)) * 40), ("README_FIRST.txt", b"hi")]:
        p = tmp_path / arc.replace("/", "__")
        p.write_bytes(data)
        entries.append((p, arc))
    out = tmp_path / "out" / "T58-Test.zip"
    info = pr.write_zip(entries, out)

    assert out.is_file() and info["files"] == 4
    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None
        assert sorted(zf.namelist()) == sorted(a for _, a in entries)
        assert not any("\\" in n for n in zf.namelist())
        assert zf.getinfo("data/raw/X/x.parquet").compress_type == zipfile.ZIP_STORED  # incompressible: stored
        assert zf.getinfo("_internal/lib.dll").compress_type == zipfile.ZIP_DEFLATED

    sha_file = Path(f"{out}.sha256")
    digest, name = sha_file.read_text(encoding="ascii").split()
    assert name == out.name
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest() == info["sha256"]
    assert sha_file.read_bytes().endswith(b"\n") and b"\r" not in sha_file.read_bytes()


def test_write_zip_refuses_oversize(tmp_path, monkeypatch):
    p = tmp_path / "f.bin"
    p.write_bytes(b"y" * 4096)
    monkeypatch.setattr(pr, "MAX_ZIP_BYTES", 10)
    with pytest.raises(pr.PackagingError, match="2 GiB"):
        pr.write_zip([(p, "f.bin")], tmp_path / "big.zip")


def test_walk_rejects_symlinks_and_skips_junk(tmp_path):
    (tmp_path / "ok.txt").write_text("ok")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"c")
    (tmp_path / "mod.pyc").write_bytes(b"c")
    (tmp_path / ".DS_Store").write_bytes(b"c")
    assert [a for _, a in pr._walk(tmp_path, "pkg")] == ["pkg/ok.txt"]
    try:
        (tmp_path / "link").symlink_to(tmp_path / "ok.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")
    with pytest.raises(pr.PackagingError, match="symlink"):
        pr._walk(tmp_path, "pkg")


# --------------------------------------------------------------------------- #
# app / data zip contents
# --------------------------------------------------------------------------- #
def test_app_zip_contents(tmp_path):
    dist = tmp_path / "dist" / "T58-Web-App"
    (dist / "_internal").mkdir(parents=True)
    (dist / "T58-Web-App.exe").write_bytes(b"MZ")
    (dist / "_internal" / "base_library.zip").write_bytes(b"PK")
    entries = pr.collect_app_entries(dist, tmp_path / "txt")
    names = {a for _, a in entries}

    assert "T58-Web-App.exe" in names and "_internal/base_library.zip" in names
    assert "README_FIRST.txt" in names and "docs/INSTALL.md" in names
    assert not any(n.startswith("data/examples/") for n in names)  # sample dataset retired
    assert any(n.startswith("strategies/") for n in names)
    # The market data is a separate download -- never inside the app zip.
    assert not any(n.startswith("data/raw/") for n in names)
    # Owner-only templates must not ship to customers.
    assert not any("TEMPLATE" in n or "SENTRY" in n for n in names)
    assert pr.validate_names(entries, zip_stem="T58-Web-App-Windows") == []


@pytest.mark.parametrize("leak", [".flask_secret_key", "data/logs/t58-web.log", "reports/search/x.json"])
def test_app_zip_refuses_a_build_folder_the_app_has_been_run_from(tmp_path, leak):
    """Running the app next to its exe creates a Flask secret key, logs, license state ...
    -- none of it may ever be zipped up and shipped to customers."""
    dist = tmp_path / "dist" / "T58-Web-App"
    (dist / "_internal").mkdir(parents=True)
    (dist / "T58-Web-App.exe").write_bytes(b"MZ")
    (dist / "_internal" / "base_library.zip").write_bytes(b"PK")
    leaked = dist / leak
    leaked.parent.mkdir(parents=True, exist_ok=True)
    leaked.write_text("secret")
    with pytest.raises(pr.PackagingError, match="while RUNNING"):
        pr.collect_app_entries(dist, tmp_path / "txt")


def test_frozen_smoke_runs_a_throwaway_copy_not_the_build_folder(tmp_path):
    smoke = _load("t58_frozen_smoke", "build_tools/frozen_smoke.py")
    build = tmp_path / "T58-Web-App"
    (build / "_internal").mkdir(parents=True)
    exe = build / "T58-Web-App.exe"
    exe.write_bytes(b"MZ")
    (build / "_internal" / "x.dll").write_bytes(b"x")

    sandbox, exe_copy = smoke._sandbox_copy(exe)
    try:
        assert exe_copy.is_file() and exe_copy.name == exe.name
        assert build not in exe_copy.parents and sandbox in exe_copy.parents
        assert (exe_copy.parent / "_internal" / "x.dll").is_file()
        (exe_copy.parent / ".flask_secret_key").write_text("written by the running app")
    finally:
        import shutil
        shutil.rmtree(sandbox, ignore_errors=True)
    assert sorted(p.name for p in build.iterdir()) == ["T58-Web-App.exe", "_internal"]  # original untouched


def test_app_zip_requires_a_real_onefolder_build(tmp_path):
    empty = tmp_path / "dist"
    empty.mkdir()
    with pytest.raises(pr.PackagingError):
        pr.collect_app_entries(empty, tmp_path / "txt")
    onefile_only = tmp_path / "dist2"
    onefile_only.mkdir()
    (onefile_only / "T58.exe").write_bytes(b"MZ")
    with pytest.raises(pr.PackagingError, match="_internal"):
        pr.collect_app_entries(onefile_only, tmp_path / "txt")


def test_real_data_folder_names_are_windows_safe(tmp_path):
    """Guards data/raw itself: every file that ships in the data pack must extract on Windows."""
    try:
        entries = pr.collect_data_entries(tmp_path / "txt")
    except pr.PackagingError as exc:
        if "LFS" in str(exc):  # CI checks out with lfs:false; the release job fetches the real data
            pytest.skip("market data is Git LFS pointers in this checkout")
        raise
    assert len(entries) > 1
    assert pr.validate_names(entries, zip_stem="T58-Market-Data") == []


# --------------------------------------------------------------------------- #
# PyInstaller spec helpers
# --------------------------------------------------------------------------- #
def test_spec_data_entries_cover_what_the_app_reads():
    web = dict((d, s) for s, d in sc.data_entries(REPO_ROOT, web=True))
    desktop = dict((d, s) for s, d in sc.data_entries(REPO_ROOT, web=False))
    for dest in ("config", "strategies", "app/ui/assets"):
        assert dest in web and dest in desktop, f"{dest} must be bundled in both builds"
    assert "app/web/templates" in web and "app/web/static" in web
    assert "app/web/templates" not in desktop
    assert (REPO_ROOT / "config" / "pyproject.toml").is_file()  # the Account tab's version source


def test_spec_filter_drops_library_test_suites_but_keeps_real_modules():
    hidden, datas = sc.without_test_suites(
        ["pyarrow", "pyarrow.parquet", "pyarrow.tests.parquet.test_basic", "pyarrow.tests", "numpy.testing",
         "app.web.server", "app.quant_lab.portfolio_optimizer", "pkg.test_helpers"],
        [("/x/pyarrow/lib.so", "pyarrow"), ("/x/pyarrow/tests/data/a.parquet", "pyarrow/tests/data"),
         ("/x/py7zr/tests/b.7z", "py7zr\\tests"), ("/x/app/web/static/a.js", "app/web/static")],
    )
    assert hidden == ["pyarrow", "pyarrow.parquet", "numpy.testing", "app.web.server",
                      "app.quant_lab.portfolio_optimizer"]
    assert datas == [("/x/pyarrow/lib.so", "pyarrow"), ("/x/app/web/static/a.js", "app/web/static")]


def test_no_real_app_module_would_be_caught_by_the_test_suite_filter():
    for path in (REPO_ROOT / "app").rglob("*.py"):
        dotted = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
        assert not sc.is_test_module(dotted), f"{dotted} would be dropped from the packaged build"


def test_spec_data_entries_fail_loudly_when_a_folder_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="refusing to build"):
        sc.data_entries(tmp_path, web=False)


def test_spec_files_exist_and_point_at_the_right_entry_points():
    web = (REPO_ROOT / "build_tools" / "t58_web.spec").read_text(encoding="utf-8")
    desktop = (REPO_ROOT / "build_tools" / "t58_desktop.spec").read_text(encoding="utf-8")
    assert "run_web.py" in web and "console=True" in web and "COLLECT(" in web
    assert "run_app.py" in desktop and "console=False" in desktop and "COLLECT(" in desktop
    for text in (web, desktop):
        assert 'collect_submodules("app")' in text
        assert "upx=False" in text


# --------------------------------------------------------------------------- #
# Updater: choosing the asset + its own checksum
# --------------------------------------------------------------------------- #
def test_updater_never_picks_a_data_pack_and_pairs_the_right_checksum():
    from app.accounts import app_info as ai

    names = ["SHA256SUMS.txt", "T58-Market-Data.zip", "T58-Market-Data.zip.sha256", "T58-Data-Examples-Slim.zip",
             "T58-Web-App-Windows.zip", "T58-Web-App-Windows.zip.sha256",
             "T58-Quant-Algo-Backtester-Windows.zip", "T58-Quant-Algo-Backtester-Windows.zip.sha256"]
    assets = [{"name": n, "browser_download_url": f"https://example.invalid/{n}"} for n in names]

    picked = ai._pick_release_asset(assets)
    assert picked["name"] == "T58-Quant-Algo-Backtester-Windows.zip"
    assert ai._find_checksum_asset(assets, picked["name"])["name"] == "T58-Quant-Algo-Backtester-Windows.zip.sha256"
    assert ai._find_checksum_asset(assets, "T58-Web-App-Windows.zip")["name"] == "T58-Web-App-Windows.zip.sha256"
    # Backward compatible: no asset name -> first checksum-looking asset, as before.
    assert ai._find_checksum_asset(assets)["name"] == "SHA256SUMS.txt"
    # Only data zips in the release -> nothing to install.
    only_data = [a for a in assets if "Data" in a["name"] and a["name"].endswith(".zip")]
    assert ai._pick_release_asset(only_data) is None


def test_updater_prefers_exe_when_present():
    from app.accounts import app_info as ai

    assets = [{"name": "T58-Market-Data.zip"}, {"name": "T58-Setup-Windows.exe"}, {"name": "Other.exe"}]
    assert ai._pick_release_asset(assets)["name"] == "T58-Setup-Windows.exe"


# --------------------------------------------------------------------------- #
# Workflows (static)
# --------------------------------------------------------------------------- #
APP_WORKFLOWS = {
    ".github/workflows/desktop-cross-platform.yml": ("t58_desktop.spec", "T58-Quant-Algo-Backtester-${{ matrix.label }}.zip", "desktop"),
    ".github/workflows/web-cross-platform.yml": ("t58_web.spec", "T58-Web-App-${{ matrix.label }}.zip", "web"),
}


@pytest.mark.parametrize("rel", sorted(APP_WORKFLOWS))
def test_app_workflows_use_the_one_folder_build_and_stay_slim(rel):
    spec, zip_name, smoke_mode = APP_WORKFLOWS[rel]
    # Judge the commands, not the explanatory comments (which name the old approach on purpose).
    text = "\n".join(
        line for line in (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert (REPO_ROOT / "build_tools" / spec).is_file()
    assert f"build_tools/{spec}" in text, f"{rel} must build from build_tools/{spec}"
    assert "--onefile" not in text, "one-file builds unpack hundreds of MB into %TEMP% on every launch"
    assert "Compress-Archive" not in text, "use build_tools/package_release.py (validated, CRC-tested zips)"
    assert "package_release.py app" in text and zip_name in text
    assert f"frozen_smoke.py {smoke_mode}" in text, f"{rel} must start the built exe before shipping it"
    # The 740 MB market-data tree must not be copied into the app zip any more.
    assert not re.search(r"Copy-Item[^\n]*data\\raw", text)
    assert "compression-level: 0" in text  # the zip is already compressed; don't re-deflate it


def test_data_pack_workflow_builds_the_separate_data_zip():
    text = (REPO_ROOT / ".github/workflows/build-data-pack.yml").read_text(encoding="utf-8")
    assert "package_release.py data" in text
    assert "T58-Market-Data.zip.sha256" in text
    assert "action-gh-release" in text


def test_workflow_yaml_is_valid():
    yaml = pytest.importorskip("yaml")
    for wf in (REPO_ROOT / ".github" / "workflows").glob("*.yml"):
        data = yaml.safe_load(wf.read_text(encoding="utf-8"))
        assert isinstance(data, dict) and "jobs" in data, wf.name


def test_exactly_one_web_and_one_desktop_workflow_each_covering_three_oses():
    names = sorted(p.name for p in (REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    assert names == ["build-data-pack.yml", "desktop-cross-platform.yml", "full-tests.yml", "secrets-scan.yml", "web-cross-platform.yml"]
    for rel in APP_WORKFLOWS:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        # Linux runners are pinned to ubuntu-24.04 in the workflows (ubuntu-latest is also accepted).
        for os_names in (("windows-latest",), ("macos-latest",), ("ubuntu-latest", "ubuntu-24.04")):
            assert any(n in text for n in os_names), f"{rel} must build on {os_names[0]}"
