#!/usr/bin/env python3
"""Build the release zips for the T58 builds -- identically on Windows, macOS and Linux.

    # the app (a PyInstaller one-folder build + examples + strategies + docs)
    python build_tools/package_release.py app --dist dist/T58-Web-App --out T58-Web-App-Windows.zip

    # the market data (data/raw/**), shipped separately from the app
    python build_tools/package_release.py data --out T58-Market-Data.zip

Each run writes ``<out>`` and ``<out>.sha256`` (``<hex>  <name>`` -- the format
app/accounts/app_info.py's updater parses).

Why this exists instead of PowerShell's ``Compress-Archive``:

* the old workflow zipped the app together with the whole ``data/raw`` tree
  (~740 MB, mostly incompressible .parquet) into one ~1 GB archive -- one
  download, one extraction, any failure = start over. The app and the data
  are now separate zips.
* it validates every file name BEFORE writing, so a name that Windows cannot
  extract (illegal character, reserved device name, trailing dot/space, a
  case-only collision, an over-long path) fails the build with a clear
  message instead of shipping a zip that errors on the customer's PC.
* it re-opens and fully CRC-tests the finished zip, and refuses an archive
  over GitHub's 2 GiB release-asset limit.
* entries always use forward slashes and Zip64-capable headers.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# GitHub rejects release assets larger than 2 GiB.
MAX_ZIP_BYTES = 2 * 1024 ** 3 - 1024 ** 2
# Explorer's "Extract All" prefixes the archive's paths with the folder it
# extracts into (e.g. C:\Users\<name>\Downloads\T58-Quant-Algo-Backtester-Windows\).
# The classic Windows limit is 260 characters for the whole path.
EXTRACT_PREFIX_BUDGET = 90
WINDOWS_MAX_PATH = 260

_ILLEGAL_CHARS = re.compile(r'[<>:"|?*\x00-\x1f\\]')
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
# Already-compressed formats: storing them is faster and makes the zip no bigger.
_STORE_EXT = {".parquet", ".zip", ".7z", ".png", ".jpg", ".jpeg", ".gz", ".xz", ".whl"}

# Windows cannot extract symlinks, and a Windows PyInstaller build never contains any,
# so a symlink in a release folder is an error. Test builds made on Linux/macOS DO
# contain them; --follow-symlinks (below) stores the target's content instead.
FOLLOW_SYMLINKS = False

# Files/folders the app creates next to its exe the first time it RUNS (see
# app.data.storage.get_app_base_dir). They never belong in a release zip.
RUNTIME_ONLY_TOP_LEVEL = {".flask_secret_key", "data", "reports", "strategies", "research", ".t58_write_test"}

SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
SKIP_NAMES = {".DS_Store", "Thumbs.db", ".gitkeep", ".t58_write_test"}


class PackagingError(RuntimeError):
    """Raised for anything that would produce a zip a customer cannot extract."""


# --------------------------------------------------------------------------- #
# Entry collection
# --------------------------------------------------------------------------- #
Entry = tuple[Path, str]  # (file on disk, path inside the zip, forward slashes)


def _walk(src: Path, arc_root: str) -> list[Entry]:
    """Every file under ``src`` mapped to ``arc_root/<relative path>``."""
    if not src.is_dir():
        raise PackagingError(f"folder not found: {src}")
    out: list[Entry] = []
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES:
                continue
            if path.is_symlink() and not FOLLOW_SYMLINKS:
                raise PackagingError(f"symlink not allowed in a release zip (Windows cannot extract it): {path}")
            rel = path.relative_to(src).as_posix()
            out.append((path, f"{arc_root}/{rel}" if arc_root else rel))
    return out


def _text_entry(tmp_dir: Path, arcname: str, text: str) -> Entry:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / arcname.replace("/", "__")
    # CRLF so the file opens correctly in Windows Notepad of any age.
    path.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
    return path, arcname


APP_README = """\
T58 Quant Algo Backtester  --  READ ME FIRST
============================================

1. EXTRACT THE WHOLE ZIP FIRST.
   Right-click the zip -> Extract All... (or use 7-Zip) into a SHORT folder
   such as  C:\\T58\\  . Do NOT run the .exe from inside the zip window --
   Windows will say files are missing.

2. Open the extracted folder and double-click the .exe.
   Keep the "_internal" folder next to the .exe -- the app will not start
   without it. If you move the app, move the whole folder.

3. Windows SmartScreen "Unknown publisher" warning:
   click  More info -> Run anyway.  (Goes away once the builds are code-signed.)

4. MARKET DATA is a separate download (so this zip stays small):
   download  T58-Market-Data.zip  from the same release/run and extract it
   INTO THIS SAME FOLDER, so you end up with  data\\raw\\EURUSD\\...  next to
   the .exe. Restart the app. Without the market data the app opens but
   has no datasets to test on.

Troubleshooting
---------------
* "Error: path too long" while extracting  -> extract to a shorter folder (C:\\T58\\).
* Antivirus quarantined a file               -> restore it and add the folder to
                                                 the exclusion list, then re-run.
* Extraction is slow                         -> the app is thousands of small
                                                 files; 7-Zip (free) is much faster
                                                 than Windows' built-in extractor.
* The app window never appears               -> look in  data\\logs\\  for t58.log
                                                 / crash_log.txt and send it to support.
"""

DATA_README = """\
T58 Market Data pack
====================

This pack holds EVERY dataset in the repo's data/raw folder -- one folder per
instrument, every timeframe of that instrument inside it:

    data\\raw\\EURUSD\\EURUSD1.csv, EURUSD5.csv, ..., fx_EUR_USD_1m.parquet
    data\\raw\\ES1!\\futures_ES.F_1m.parquet, ES_5min_....csv
    data\\raw\\NVIDIA\\nvda-1999-2014.csv      ... and so on.

OPTION A -- nothing to import (recommended)
  Extract this zip INTO the same folder as the T58 app (the folder that holds
  the .exe), so the files land in  data\\raw\\<instrument>\\...  next to it.
  Windows will offer to merge the "data" folder -- say yes.
  Then restart the app: every dataset is already there.

OPTION B -- import it from inside the app
  Extract the zip anywhere, then open  Data Center -> Import folder  and drop
  the extracted folder (or its  data\\raw  folder) onto the box. The app keeps
  the instrument folders, so each instrument shows up under ONE header in the
  dataset dropdowns with all of its timeframes listed under it.

Safe to re-extract / re-import over an older copy to update: files that are
already there are left alone.

{contents}"""


def _data_contents(entries: "list[Entry]") -> str:
    """Per-instrument file counts for DATA_README.txt, built from what is actually in the zip."""
    folders: dict[str, int] = {}
    total = 0
    for _, arc in entries:
        parts = arc.split("/")
        if len(parts) >= 4 and parts[0] == "data" and parts[1] == "raw":
            folders[parts[2]] = folders.get(parts[2], 0) + 1
            total += 1
    lines = [f"CONTENTS: {total} data files across {len(folders)} instruments", ""]
    lines += [f"  {name:<24} {count:>3} file{'s' if count != 1 else ''}" for name, count in sorted(folders.items())]
    return "\n".join(lines) + "\n"


UNIX_README = """\
T58 Quant Algo Backtester  --  READ ME FIRST  ({platform})
============================================

1. Unzip this archive anywhere (your home folder is fine).

2. Open a terminal in the unzipped folder and start the app:

       chmod +x ./{exe}      # only needed once
       ./{exe}

   {launch_note}
   Keep the "_internal" folder next to the program -- it will not start
   without it. If you move the app, move the whole folder.
{mac_note}
3. MARKET DATA is a separate download (so this zip stays small):
   download  T58-Market-Data.zip  from the same release/run and unzip it
   INTO THIS SAME FOLDER, so you end up with  data/raw/EURUSD/...  next to
   the program. Restart the app. Without the market data the app opens but
   has no datasets to test on.

Troubleshooting
---------------
* The app does not start    -> run it from the terminal and read the message;
                               logs are in  data/logs/  next to the program.
* Linux: "cannot open shared object file"  -> install the missing system
                               library (for the desktop app: python3-tk / tk).
"""

MAC_NOTE = """\
   macOS may say the app "cannot be opened because the developer cannot be
   verified" (the builds are not notarized yet). Run once:
       xattr -dr com.apple.quarantine .
   or right-click the program -> Open -> Open.
"""


def app_readme(platform: str, exe_name: str, desktop: bool) -> str:
    """README_FIRST.txt for the app zip. Windows keeps its long-standing text."""
    if platform == "windows":
        return APP_README
    launch = ("It opens the desktop window." if desktop else
              "It starts the web app and opens it in your browser (or open the address it prints).")
    return UNIX_README.format(platform="macOS" if platform == "macos" else "Linux", exe=exe_name,
                              launch_note=launch, mac_note=("\n" + MAC_NOTE) if platform == "macos" else "")


def collect_app_entries(dist: Path, tmp_dir: Path, platform: str = "windows") -> list[Entry]:
    """The app zip: PyInstaller folder at the zip root, plus the small extras."""
    entries = _walk(dist, "")
    # A PyInstaller build folder holds ONLY the exe and _internal/. Anything else
    # (.flask_secret_key, data/, reports/ ...) was written by RUNNING the app from
    # this folder, and would be shipped to every customer -- including a secret key.
    leaked = sorted({a.split("/")[0] for _, a in entries if a.split("/")[0] in RUNTIME_ONLY_TOP_LEVEL})
    if leaked:
        raise PackagingError(
            f"{dist} contains files the app creates while RUNNING ({', '.join(leaked)}) -- they must not be "
            f"shipped (e.g. .flask_secret_key). Test the app from a copy of the build folder, never from it."
        )
    top_level = [a for _, a in entries if "/" not in a]
    # Windows: NAME.exe; Linux/macOS builds (used for testing): an extension-less binary.
    if not any(a.lower().endswith(".exe") or "." not in a for a in top_level):
        raise PackagingError(f"{dist} has no top-level executable -- did the PyInstaller build finish?")
    if not any(a.startswith("_internal/") for _, a in entries):
        raise PackagingError(f"{dist} has no _internal/ folder -- expected a PyInstaller one-folder build.")
    entries += _walk(ROOT / "strategies", "strategies")
    # The Library's research papers ship beside the app (the Library tab reads <app folder>/research); without this
    # the tab opens empty on a customer's machine.
    entries += _walk(ROOT / "research", "research")
    # Customer-facing docs only -- the EULA / privacy / refund / Sentry files in
    # docs/buyer are templates with placeholders for the owner, not for customers.
    for doc in ("INSTALL.md", "QUICKSTART.md"):
        path = ROOT / "docs" / "buyer" / doc
        if path.is_file():
            entries.append((path, f"docs/{doc}"))
    phone = ROOT / "HOW_TO_OPEN_ON_YOUR_PHONE.md"
    if phone.is_file():
        entries.append((phone, "HOW_TO_OPEN_ON_YOUR_PHONE.md"))
    entries.append(_text_entry(tmp_dir, "README_FIRST.txt",
                               app_readme(platform, dist.name, desktop="Web" not in dist.name)))
    return entries


def collect_data_entries(tmp_dir: Path) -> list[Entry]:
    """The data zip: data/raw/** exactly as the app expects it."""
    raw = ROOT / "data" / "raw"
    entries = _walk(raw, "data/raw")
    if not entries:
        raise PackagingError(f"{raw} contains no files -- nothing to package.")
    # Completeness guard: the pack must carry EVERY data file. A Git LFS pointer stub (a ~130-byte
    # text file starting "version https://git-lfs...") means the checkout did not fetch the real
    # data, and zipping it would ship a "dataset" that cannot be imported -- fail the build instead.
    stubs = []
    for path, arc in entries:
        try:
            if path.stat().st_size < 1024:
                with open(path, "rb") as fh:
                    if fh.read(40).startswith(b"version https://git-lfs"):
                        stubs.append(arc)
        except OSError:
            stubs.append(arc)
    if stubs:
        raise PackagingError(
            f"{len(stubs)} file(s) under data/raw are Git LFS pointers / unreadable, not real data "
            f"(first: {stubs[0]}). Check out the repo with LFS files fetched, then rebuild."
        )
    entries.append(_text_entry(tmp_dir, "DATA_README.txt", DATA_README.format(contents=_data_contents(entries))))
    return entries


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_names(entries: list[Entry], *, zip_stem: str = "T58-Release") -> list[str]:
    """Return a list of human-readable problems (empty = every name is safe)."""
    problems: list[str] = []
    seen: dict[str, str] = {}
    prefix_len = EXTRACT_PREFIX_BUDGET + len(zip_stem)
    for _, arc in entries:
        if arc.startswith("/") or ".." in arc.split("/"):
            problems.append(f"unsafe path: {arc!r}")
        if _ILLEGAL_CHARS.search(arc):
            problems.append(f"character Windows cannot use in a file name: {arc!r}")
        for part in arc.split("/"):
            if not part:
                problems.append(f"empty path component: {arc!r}")
                continue
            if part != part.rstrip(". "):
                problems.append(f"name ends with a dot or space (Windows drops it): {arc!r}")
            if part.split(".")[0].lower() in _RESERVED:
                problems.append(f"reserved Windows device name {part!r}: {arc!r}")
        key = arc.lower()
        if key in seen and seen[key] != arc:
            problems.append(f"case-only collision (Windows treats these as one file): {seen[key]!r} vs {arc!r}")
        seen[key] = arc
        if len(arc) + prefix_len > WINDOWS_MAX_PATH:
            problems.append(
                f"path too long once extracted ({len(arc)} + ~{prefix_len} > {WINDOWS_MAX_PATH}): {arc!r}"
            )
    return problems


# --------------------------------------------------------------------------- #
# Writing + verification
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_zip(entries: list[Entry], out: Path) -> dict:
    """Write, re-open and CRC-test the zip, then write ``<out>.sha256``."""
    out = Path(out)
    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(entries, key=lambda e: e[1])
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for path, arc in entries:
            ctype = zipfile.ZIP_STORED if Path(arc).suffix.lower() in _STORE_EXT else zipfile.ZIP_DEFLATED
            zf.write(path, arc, compress_type=ctype)

    size = out.stat().st_size
    if size > MAX_ZIP_BYTES:
        raise PackagingError(
            f"{out.name} is {size / 1024 ** 2:.0f} MiB -- over GitHub's 2 GiB release-asset limit. "
            f"Split the contents into more than one zip."
        )
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        if len(names) != len(entries):
            raise PackagingError(f"{out.name}: wrote {len(entries)} files but the zip lists {len(names)}")
        bad = zf.testzip()
        if bad is not None:
            raise PackagingError(f"{out.name}: CRC check failed for {bad!r}")
        if any("\\" in n for n in names):
            raise PackagingError(f"{out.name}: found a backslash in an entry name")

    digest = _sha256(out)
    Path(f"{out}.sha256").write_bytes(f"{digest}  {out.name}\n".encode("ascii"))
    return {"zip": out, "files": len(entries), "bytes": size, "sha256": digest,
            "longest": max((a for _, a in entries), key=len)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    app_p = sub.add_parser("app", help="zip a PyInstaller one-folder build plus examples/strategies/docs")
    app_p.add_argument("--dist", required=True, type=Path, help="PyInstaller output folder, e.g. dist/T58-Web-App")
    app_p.add_argument("--out", required=True, type=Path)
    app_p.add_argument("--follow-symlinks", action="store_true",
                       help="testing only: store symlink targets (builds made on Linux/macOS contain symlinks)")
    app_p.add_argument("--platform", choices=["windows", "macos", "linux"], default="windows",
                       help="selects the README_FIRST.txt wording (default: windows)")
    data_p = sub.add_parser("data", help="zip data/raw/** as the separate market-data pack")
    data_p.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    global FOLLOW_SYMLINKS
    FOLLOW_SYMLINKS = bool(getattr(args, "follow_symlinks", False))

    tmp_dir = Path(os.environ.get("RUNNER_TEMP") or ROOT / ".package_tmp") / "t58_pkg_text"
    try:
        entries = collect_app_entries(args.dist, tmp_dir, args.platform) if args.kind == "app" else collect_data_entries(tmp_dir)
        problems = validate_names(entries, zip_stem=args.out.stem)
        if problems:
            print(f"REFUSING to build {args.out.name}: {len(problems)} file-name problem(s) Windows could not extract:",
                  file=sys.stderr)
            for p in problems[:40]:
                print(f"  - {p}", file=sys.stderr)
            return 1
        info = write_zip(entries, args.out)
    except PackagingError as exc:
        print(f"PACKAGING FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"{info['zip'].name}: {info['files']} files, {info['bytes'] / 1024 ** 2:.1f} MiB, sha256 {info['sha256']}")
    print(f"longest entry ({len(info['longest'])} chars): {info['longest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
