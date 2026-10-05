"""
In-app data-pack downloader (v7, worker D, 2026-10-05).

Why this exists: release zips used to bundle the entire data/raw tree
(~1 GB), which makes for a poor buyer experience and forces a full
re-download for every data update. The slim release carries only
data/examples/; everything else arrives through this module, which fetches
data packs (zip/tar/csv/parquet archives published at a URL) with:

  * checksum verification (SHA-256 manifest entries -- a pack whose bytes
    don't match is rejected, never installed),
  * resume-friendly downloads (HTTP Range; a partial .part file continues
    where it left off instead of restarting),
  * a manifest format (packs.json) listing available packs, so the UI can
    show "ES 1-minute 2020-2024 (412 MB)" without hardcoding URLs.

Stdlib only (urllib) -- no new dependency. UI wiring: call
download_data_pack() from a progress-callback thread; see the docstring
of that function for the exact one-spot insertion suggestion.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

log = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 256  # 256 KiB per read
DEFAULT_TIMEOUT_S = 60


class DataPackError(RuntimeError):
    """Raised for any data-pack failure (network, checksum, manifest)."""


@dataclass
class DataPack:
    """One downloadable data pack, as described by a manifest entry."""
    name: str            # e.g. "es_1m_2020_2024"
    url: str             # direct download URL (https)
    sha256: str          # expected hex digest of the downloaded file
    size_bytes: int = 0  # advisory only (progress display); 0 = unknown
    description: str = ""
    filename: str = ""   # local filename; defaults to the URL's basename

    def local_filename(self) -> str:
        if self.filename:
            return self.filename
        return Path(self.url.split("?")[0]).name or f"{self.name}.zip"


@dataclass
class DownloadProgress:
    bytes_downloaded: int
    bytes_total: int  # 0 when unknown
    resumed: bool


ProgressCallback = Callable[[DownloadProgress], None]


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_manifest(manifest_url: str, timeout: int = DEFAULT_TIMEOUT_S) -> list[DataPack]:
    """Download and parse a packs.json manifest. Expected shape:
    {"packs": [{"name": ..., "url": ..., "sha256": ..., "size_bytes": ...,
    "description": ..., "filename": ...}, ...]}. Raises DataPackError on
    any failure."""
    req = urllib.request.Request(manifest_url, headers={"User-Agent": "T58-Backtester-DataDownloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception as exc:  # noqa: BLE001
        raise DataPackError(f"Could not fetch manifest {manifest_url!r}: {exc}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise DataPackError(f"Manifest at {manifest_url!r} is not valid JSON: {exc}") from exc
    packs: list[DataPack] = []
    for entry in data.get("packs", []) or []:
        try:
            packs.append(DataPack(
                name=str(entry["name"]),
                url=str(entry["url"]),
                sha256=str(entry["sha256"]).lower(),
                size_bytes=int(entry.get("size_bytes", 0) or 0),
                description=str(entry.get("description", "")),
                filename=str(entry.get("filename", "")),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise DataPackError(f"Malformed manifest entry {entry!r}: {exc}") from exc
    return packs


def download_data_pack(
    pack: DataPack,
    dest_dir: str | os.PathLike,
    progress: ProgressCallback | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
    verify: bool = True,
) -> Path:
    """Download `pack` into `dest_dir`, verifying its SHA-256.

    Resume behavior: bytes are streamed into `<filename>.part`; if that
    file already exists, the request carries `Range: bytes=<size>-` and a
    206 response continues the download. If the server ignores Range (200
    instead of 206), the partial file is discarded and the download
    restarts from zero. On completion the checksum is verified BEFORE the
    .part file is renamed into place -- a failed checksum raises
    DataPackError and leaves the .part file for inspection/retry; a
    passed checksum atomically replaces any previous file of the same
    name.

    Returns the final Path. Raises DataPackError on network failure or
    checksum mismatch. `progress` is called after every chunk (safe to
    update a UI progress bar from it -- it runs on the caller's thread,
    so call this from a worker thread, not the UI thread).
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    final_path = dest / pack.local_filename()
    part_path = dest / (pack.local_filename() + ".part")

    # Fast path: already downloaded and checksum-verified.
    if verify and final_path.exists() and pack.sha256:
        if _sha256_of_file(final_path) == pack.sha256:
            log.info("Data pack %s already present and verified.", pack.name)
            return final_path
        log.warning("Existing %s failed checksum -- re-downloading.", final_path.name)

    start_at = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": "T58-Backtester-DataDownloader/1.0"}
    if start_at > 0:
        headers["Range"] = f"bytes={start_at}-"
    req = urllib.request.Request(pack.url, headers=headers)

    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        raise DataPackError(f"Download failed for {pack.name}: {exc}") from exc

    with resp:
        status = getattr(resp, "status", 200)
        resumed = status == 206 and start_at > 0
        if start_at > 0 and not resumed:
            # Server ignored the Range request -- restart cleanly rather
            # than appending a full body onto a partial prefix.
            log.info("Server ignored Range for %s -- restarting download.", pack.name)
            start_at = 0
        total = pack.size_bytes or 0
        content_range = resp.headers.get("Content-Range", "")
        if not total and content_range.startswith("bytes"):
            try:
                total = int(content_range.split("/")[-1])
            except ValueError:
                total = 0
        if not total:
            try:
                total = int(resp.headers.get("Content-Length", "0") or 0) + start_at
            except ValueError:
                total = 0

        mode = "ab" if resumed else "wb"
        downloaded = start_at
        if progress is not None:
            progress(DownloadProgress(downloaded, total, resumed))
        try:
            with open(part_path, mode) as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(DownloadProgress(downloaded, total, resumed))
        except Exception as exc:  # noqa: BLE001
            raise DataPackError(f"Download interrupted for {pack.name} "
                                f"({downloaded} bytes so far): {exc}") from exc

    if verify and pack.sha256:
        actual = _sha256_of_file(part_path)
        if actual != pack.sha256:
            raise DataPackError(
                f"Checksum mismatch for {pack.name}: expected {pack.sha256}, "
                f"got {actual}. The file was NOT installed.")

    # Atomic-ish install: rename into place (same directory, so atomic on
    # both Windows and POSIX), replacing any previous copy.
    tmp_final = part_path.with_suffix(part_path.suffix + ".verified")
    shutil.move(str(part_path), str(tmp_final))
    os.replace(str(tmp_final), str(final_path))
    log.info("Data pack %s installed at %s (%d bytes).", pack.name, final_path, downloaded)
    return final_path


def list_local_packs(dest_dir: str | os.PathLike) -> list[Path]:
    """Archive files already present in `dest_dir` (downloaded packs)."""
    dest = Path(dest_dir)
    if not dest.exists():
        return []
    exts = {".zip", ".tar", ".gz", ".7z", ".csv", ".parquet"}
    return sorted(p for p in dest.iterdir()
                  if p.is_file() and (p.suffix.lower() in exts or ".tar." in p.name.lower()))


def download_packs(
    packs: Iterable[DataPack],
    dest_dir: str | os.PathLike,
    progress: ProgressCallback | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> list[Path]:
    """Download several packs in sequence. Returns the installed paths.
    A single pack's failure aborts the batch (the caller can catch
    DataPackError per pack instead by calling download_data_pack in a
    loop)."""
    return [download_data_pack(p, dest_dir, progress=progress, timeout=timeout)
            for p in packs]
