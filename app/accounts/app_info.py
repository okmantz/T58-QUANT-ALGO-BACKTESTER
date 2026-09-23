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

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Fill in once this project has a public GitHub repo to check releases
# against, e.g. "T58Trading/T58-QUANT-ALGO-BACKTESTER". Left blank means
# "Check for Updates" reports itself as not configured instead of guessing.
GITHUB_REPO = "okmantz/T58-QUANT-ALGO-BACKTESTER"

_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "config" / "pyproject.toml"
_VERSION_RE = re.compile(r'(?m)^\s*version\s*=\s*"([^"]+)"')
_REQUEST_TIMEOUT_SECONDS = 6


def _read_pyproject_version() -> str:
    try:
        text = _PYPROJECT_PATH.read_text(encoding="utf-8")
        match = _VERSION_RE.search(text)
        if match:
            return match.group(1)
    except Exception:  # noqa: BLE001 -- a missing/unreadable pyproject.toml must never crash the Account tab
        pass
    return "0.0.0"


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
