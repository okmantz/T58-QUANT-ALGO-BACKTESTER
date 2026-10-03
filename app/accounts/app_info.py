"""
App Version / Check for Updates -- the Account tab's App section.

APP_VERSION is read straight from config/pyproject.toml's own
[project].version (the same version string `pip show`/packaging already
use for this project) so there is exactly one place that number lives --
bumping a release only ever means editing pyproject.toml.

Check for Updates compares APP_VERSION against the latest GitHub Release
tag for this repo. GITHUB_REPO below is a placeholder ("OWNER/REPO") --
fill in this project's real GitHub "owner/repo" slug once it has one, or
leave it blank to disable the check (the Account tab shows a plain "not
configured" message rather than erroring). api.github.com is reachable
from this app's sandboxed network egress list, same as api.anthropic.com.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Fill in once this project has a public GitHub repo to check releases
# against, e.g. "T58Trading/T58-QUANT-ALGO-BACKTESTER". Left blank means
# "Check for Updates" reports itself as not configured instead of guessing.
GITHUB_REPO = "okmantz/T58-QUANT-ALGO-BACKTESTER"

_VERSION_RE = re.compile(r'(?m)^\s*version\s*=\s*"([^"]+)"')
_REQUEST_TIMEOUT_SECONDS = 6

# BUGFIX (2026-09): the Account tab was showing "0.0.0" in the built Windows
# .exe (both the desktop and web-app PyInstaller builds), even though a real
# version has always lived in config/pyproject.toml. Root cause: neither
# .github/workflows/build-exe.yml nor build-web-exe.yml ever passed
# `--add-data "config;config"` to PyInstaller, so config/pyproject.toml was
# never bundled into the frozen .exe at all -- every run of the built app
# hit the "file not found" except-branch below and silently fell back.
# That workflow gap is fixed separately (config/ is now bundled), but this
# reader is also made frozen-aware directly so a future packaging change
# can't quietly reintroduce the same silent fallback: when running from a
# PyInstaller bundle (sys.frozen / sys._MEIPASS set), it checks the bundle's
# extraction dir and the folder the .exe lives in, in addition to the normal
# source-tree path used in dev. If every candidate is missing, the fallback
# is now "1.0.0" -- a real release number -- rather than the alarming-
# looking "0.0.0", which only ever indicated this exact bundling bug.
_FALLBACK_VERSION = "1.0.0"


def _candidate_pyproject_paths() -> list[Path]:
    candidates: list[Path] = []
    # Normal case: running from source, this file is app/accounts/app_info.py
    # so parents[2] is the repo root.
    candidates.append(Path(__file__).resolve().parents[2] / "config" / "pyproject.toml")
    # PyInstaller --onefile bundle: files added via --add-data land under
    # sys._MEIPASS (a temp extraction dir) at runtime.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "config" / "pyproject.toml")
    # Frozen build in general (onefile or onedir): also check right next to
    # the actual .exe, in case a future packaging step ships config/ as a
    # loose folder alongside it instead of as embedded PyInstaller data.
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "config" / "pyproject.toml")
        candidates.append(exe_dir / "pyproject.toml")
    return candidates


def _read_pyproject_version() -> str:
    for path in _candidate_pyproject_paths():
        try:
            text = path.read_text(encoding="utf-8")
            match = _VERSION_RE.search(text)
            if match:
                return match.group(1)
        except Exception:  # noqa: BLE001 -- a missing/unreadable pyproject.toml must never crash the Account tab
            continue
    return _FALLBACK_VERSION


APP_VERSION = _read_pyproject_version()


def _parse_version_tuple(label: str) -> tuple[int, ...]:
    """Best-effort numeric-tuple parse of a version-ish string ("v1.2.3",
    "1.2", "1.2.3-beta" -> (1, 2, 3)) for a simple newer-than comparison.
    Falls back to (0,) for anything unparseable rather than raising."""
    cleaned = label.strip().lstrip("vV")
    parts = re.split(r"[.\-+]", cleaned)
    nums: list[int] = []
    for part in parts:
        if part.isdigit():
            nums.append(int(part))
        else:
            break
    return tuple(nums) if nums else (0,)


@dataclass
class UpdateCheckResult:
    configured: bool
    checked: bool
    current_version: str
    latest_version: str = ""
    release_url: str = ""
    update_available: bool = False
    error: str = ""


def check_for_updates() -> UpdateCheckResult:
    """Best-effort, never-raises check against
    https://api.github.com/repos/<GITHUB_REPO>/releases/latest. Any
    network failure, missing repo config, or unexpected response shape
    comes back as a populated `error` string rather than an exception --
    this is a convenience check, not something that should ever be able
    to break the Account tab."""
    if not GITHUB_REPO:
        return UpdateCheckResult(
            configured=False, checked=False, current_version=APP_VERSION,
            error="Update checking isn't configured yet -- no GitHub repo is set (see app.accounts.app_info.GITHUB_REPO).",
        )
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "T58-QUANT-ALGO-BACKTESTER"})
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latest_tag = str(data.get("tag_name", "")).strip()
        release_url = str(data.get("html_url", ""))
        if not latest_tag:
            return UpdateCheckResult(
                configured=True, checked=False, current_version=APP_VERSION,
                error="GitHub returned no release tag to compare against.",
            )
        update_available = _parse_version_tuple(latest_tag) > _parse_version_tuple(APP_VERSION)
        return UpdateCheckResult(
            configured=True, checked=True, current_version=APP_VERSION,
            latest_version=latest_tag, release_url=release_url, update_available=update_available,
        )
    except urllib.error.HTTPError as exc:
        return UpdateCheckResult(
            configured=True, checked=False, current_version=APP_VERSION,
            error=f"GitHub returned an error ({exc.code}) -- the repo may be private or have no releases yet.",
        )
    except Exception as exc:  # noqa: BLE001
        return UpdateCheckResult(configured=True, checked=False, current_version=APP_VERSION, error=f"Could not check for updates: {exc}")


# ---------------------------------------------------------------------------
# In-app update DOWNLOAD (P1-7, Oct 2026)
#
# check_for_updates() above stays check-ONLY (the default path -- the
# Account tab never downloads anything on its own). download_update()
# below is the explicit, user-confirmed second step: it fetches the
# latest release's Windows asset from GitHub Releases and verifies its
# SHA-256 against the release's published checksum file before handing
# the file back.
#
# Rules, deliberately strict:
#   1. confirmed must be True (the UI's "Yes, download this update"
#      button passes it) -- a False here returns an error, never a file.
#      There is NO silent/auto install anywhere in this module.
#   2. If the release publishes no recognizable checksum asset, the
#      download is REFUSED: an unverifiable binary is not installed,
#      period.
#   3. On checksum mismatch the downloaded bytes are DELETED and the
#      error says so -- a corrupted or tampered binary never sits on
#      disk looking ready to run.
#   4. The returned file is NOT executed and does NOT replace anything:
#      the user runs/installs it themselves. This module never
#      self-modifies a running app.
# ---------------------------------------------------------------------------

_DOWNLOAD_TIMEOUT_SECONDS = 30
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MB chunks -- a 350 MB exe never sits whole in RAM twice


@dataclass
class UpdateDownloadResult:
    ok: bool
    path: str = ""          # where the verified asset was saved ("" unless ok)
    version: str = ""       # the release tag it came from
    asset_name: str = ""
    verified: bool = False  # True only when sha256 matched the published checksum
    error: str = ""


def _github_api_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "T58-QUANT-ALGO-BACKTESTER"},
    )
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pick_release_asset(assets: list[dict]) -> dict | None:
    """Pick the Windows install asset from a release's asset list:
    prefer a *.exe (the PyInstaller onefile build), fall back to *.zip."""
    exes = [a for a in assets if str(a.get("name", "")).lower().endswith(".exe")]
    zips = [a for a in assets if str(a.get("name", "")).lower().endswith(".zip")]
    if exes:
        # Prefer the asset whose name mentions windows if there are several.
        for a in exes:
            if "windows" in str(a.get("name", "")).lower():
                return a
        return exes[0]
    return zips[0] if zips else None


def _find_checksum_asset(assets: list[dict]) -> dict | None:
    """Find the release's published checksum file (SHA256SUMS.txt,
    checksums.txt, or an asset with 'sha256'/'checksum' in the name)."""
    for a in assets:
        name = str(a.get("name", "")).lower()
        if "sha256" in name or "checksum" in name:
            return a
    return None


def _published_sha256_for(checksum_text: str, asset_name: str) -> str | None:
    """Parse a checksum file's '<hex> <filename>' lines for asset_name."""
    for line in checksum_text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[1].lstrip("*") == asset_name:
            digest = parts[0].strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                return digest
    return None


def _download_bytes(url: str, hasher: "hashlib._Hash | None" = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "T58-QUANT-ALGO-BACKTESTER"})
    chunks: list[bytes] = []
    with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp:
        while True:
            chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            if hasher is not None:
                hasher.update(chunk)
            chunks.append(chunk)
    return b"".join(chunks)


def download_update(confirmed: bool, download_dir: "str | Path | None" = None) -> UpdateDownloadResult:
    """Download the latest release's asset with SHA-256 verification.

    confirmed: must be True -- the caller passes this only after an
        explicit user confirmation ("Yes, download the update"). False
        always returns an error; nothing is downloaded, installed, or
        modified.
    download_dir: where to save the verified asset (defaults to the
        user's Downloads folder, falling back to a temp dir).
    """
    if not confirmed:
        return UpdateDownloadResult(
            ok=False,
            error="Update download refused: it requires an explicit user confirmation "
                  "(this app never downloads or installs updates on its own).",
        )
    if not GITHUB_REPO:
        return UpdateDownloadResult(
            ok=False, error="Update downloads aren't configured -- no GitHub repo is set "
                            "(see app.accounts.app_info.GITHUB_REPO)."
        )
    check = check_for_updates()
    if check.error or not check.checked:
        return UpdateDownloadResult(ok=False, error=f"Could not check for updates: {check.error or 'unknown'}")
    if not check.update_available:
        return UpdateDownloadResult(
            ok=False, version=check.latest_version,
            error=f"Already on the latest version ({check.current_version}) -- nothing to download.",
        )
    try:
        release = _github_api_json(f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest")
        assets = release.get("assets") or []
        asset = _pick_release_asset(assets)
        if asset is None or not asset.get("browser_download_url"):
            return UpdateDownloadResult(
                ok=False, version=check.latest_version,
                error="The latest release has no downloadable .exe/.zip asset.",
            )
        asset_name = str(asset["name"])
        checksum_asset = _find_checksum_asset(assets)
        if checksum_asset is None or not checksum_asset.get("browser_download_url"):
            return UpdateDownloadResult(
                ok=False, version=check.latest_version, asset_name=asset_name,
                error="The latest release publishes no checksum file -- refusing to "
                      "download an unverifiable binary. (The release needs a "
                      "SHA256SUMS.txt-style asset for in-app updates to work.)",
            )
        checksum_text = _download_bytes(str(checksum_asset["browser_download_url"])).decode("utf-8", errors="replace")
        expected = _published_sha256_for(checksum_text, asset_name)
        if expected is None:
            return UpdateDownloadResult(
                ok=False, version=check.latest_version, asset_name=asset_name,
                error=f"The release's checksum file has no entry for {asset_name!r} -- "
                      "refusing to download an unverifiable binary.",
            )

        target_dir = Path(download_dir) if download_dir else (Path.home() / "Downloads")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            import tempfile
            target_dir = Path(tempfile.mkdtemp(prefix="t58-update-"))
        dest = target_dir / f"T58-Update-{check.latest_version}-{asset_name}"

        hasher = hashlib.sha256()
        body = _download_bytes(str(asset["browser_download_url"]), hasher=hasher)
        actual = hasher.hexdigest()
        if actual != expected:
            return UpdateDownloadResult(
                ok=False, version=check.latest_version, asset_name=asset_name,
                error=f"CHECKSUM MISMATCH for {asset_name}: expected {expected}, got {actual}. "
                      "The download was discarded and nothing was written to disk -- do not "
                      "install this release from any other source either.",
            )
        dest.write_bytes(body)
        return UpdateDownloadResult(
            ok=True, path=str(dest), version=check.latest_version,
            asset_name=asset_name, verified=True,
        )
    except urllib.error.HTTPError as exc:
        return UpdateDownloadResult(
            ok=False, error=f"GitHub returned an error ({exc.code}) while downloading the update."
        )
    except Exception as exc:  # noqa: BLE001
        return UpdateDownloadResult(ok=False, error=f"Update download failed: {exc}")
