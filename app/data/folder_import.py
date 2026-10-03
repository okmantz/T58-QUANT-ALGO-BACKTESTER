"""Folder import for market data (v2, Oct 2026).

Shared by the desktop Market Data / Data Center tabs and the web Data
Center: take a local folder (desktop, where the folder is on disk) or a
list of uploaded (name, bytes) pairs (web, where the folder lives on the
client), and feed every recognized file through the EXISTING importer
pipeline -- app.data.importer.import_csv_bytes() for validation and
app.data.storage.store_csv_bytes() for persistence. There is deliberately
no parallel importer: one pipeline, one set of validation rules.

Recognized types mirror the Data Center's single-file import: .csv, .tsv,
.txt, .parquet. Anything else is reported as skipped, never failed, so a
folder containing READMEs or license files doesn't look like an error.

Filenames are flattened into data/raw/ by store_csv_bytes() (which also
applies the v1 path-traversal sanitization); the outcome's `name` keeps
the folder-relative path so the per-file report stays readable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

RECOGNIZED_SUFFIXES = (".csv", ".tsv", ".txt", ".parquet")

# Safety caps for one import action. A single folder drop should not
# accidentally queue tens of thousands of files.
MAX_FILES_PER_IMPORT = 2000


@dataclass
class FileImportOutcome:
    """One file's result. status is "imported" | "skipped" | "failed"."""

    name: str  # folder-relative path (or the upload's relative path)
    status: str
    detail: str = ""


@dataclass
class FolderImportReport:
    outcomes: list[FileImportOutcome] = field(default_factory=list)

    @property
    def imported(self) -> list[FileImportOutcome]:
        return [o for o in self.outcomes if o.status == "imported"]

    @property
    def skipped(self) -> list[FileImportOutcome]:
        return [o for o in self.outcomes if o.status == "skipped"]

    @property
    def failed(self) -> list[FileImportOutcome]:
        return [o for o in self.outcomes if o.status == "failed"]

    def summary_line(self) -> str:
        return (
            f"{len(self.imported)} imported, "
            f"{len(self.skipped)} skipped, "
            f"{len(self.failed)} failed "
            f"({len(self.outcomes)} files total)"
        )

    def detail_lines(self, max_lines: int = 40) -> list[str]:
        lines = [
            f"[{o.status.upper()}] {o.name}" + (f" -- {o.detail}" if o.detail else "")
            for o in self.outcomes
        ]
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... and {len(lines) - max_lines} more"]
        return lines


def import_file_bytes(name: str, content: bytes) -> FileImportOutcome:
    """Validate + persist one file through the existing pipeline."""
    # Local imports: app.data.importer is heavy (pandas etc.); keep this
    # module import-light so UI code can import it at startup cheaply.
    from app.data.importer import import_csv_bytes
    from app.data.storage import store_csv_bytes

    if not content:
        return FileImportOutcome(name, "failed", "empty file (0 bytes)")
    try:
        result = import_csv_bytes(content, filename=name)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the batch
        return FileImportOutcome(name, "failed", f"importer error: {exc}")
    if not result.is_valid:
        errors = "; ".join(getattr(result, "errors", None) or []) or "validation failed"
        return FileImportOutcome(name, "failed", errors)
    try:
        dest = store_csv_bytes(content, name)
    except Exception as exc:  # noqa: BLE001
        return FileImportOutcome(name, "failed", f"store error: {exc}")
    stored = dest.name if isinstance(dest, Path) else str(dest)
    return FileImportOutcome(name, "imported", f"saved as {stored}")


def _iter_candidate_files(root: Path) -> list[tuple[str, Path]]:
    """Recursively collect (relative_name, path) pairs. Hidden files and
    directories are skipped silently (dotfiles are never datasets)."""
    found: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            p = Path(dirpath) / fn
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                continue
            found.append((rel, p))
            if len(found) >= MAX_FILES_PER_IMPORT:
                return found
    return found


def import_folder_tree(folder: str | os.PathLike) -> FolderImportReport:
    """Import every recognized file under `folder` (recursive).

    Unrecognized suffixes are reported as skipped (not failed). Files
    beyond MAX_FILES_PER_IMPORT are reported as skipped with the reason.
    A missing/not-a-directory `folder` yields a report with a single
    failed outcome instead of raising.
    """
    report = FolderImportReport()
    root = Path(folder)
    if not root.is_dir():
        report.outcomes.append(
            FileImportOutcome(str(folder), "failed", "not a directory or no longer exists")
        )
        return report

    candidates = _iter_candidate_files(root)
    for rel, path in candidates:
        if Path(rel).suffix.lower() not in RECOGNIZED_SUFFIXES:
            report.outcomes.append(
                FileImportOutcome(rel, "skipped", f"unrecognized type '{Path(rel).suffix or '(none)'}'")
            )
            continue
        try:
            content = path.read_bytes()
        except Exception as exc:  # noqa: BLE001
            report.outcomes.append(FileImportOutcome(rel, "failed", f"read error: {exc}"))
            continue
        outcome = import_file_bytes(rel, content)
        report.outcomes.append(outcome)
    return report


def import_uploaded_files(files: list[tuple[str, bytes]]) -> FolderImportReport:
    """Web-edition entry point: (client-relative path, bytes) pairs from
    the folder drop zone / folder picker. Same pipeline and same caps as
    import_folder_tree; unrecognized suffixes are skipped, not failed."""
    report = FolderImportReport()
    for rel_name, content in files[:MAX_FILES_PER_IMPORT]:
        name = (rel_name or "upload").replace("\\", "/")
        if Path(name).suffix.lower() not in RECOGNIZED_SUFFIXES:
            report.outcomes.append(
                FileImportOutcome(name, "skipped", f"unrecognized type '{Path(name).suffix or '(none)'}'")
            )
            continue
        report.outcomes.append(import_file_bytes(name, content))
    if len(files) > MAX_FILES_PER_IMPORT:
        report.outcomes.append(
            FileImportOutcome(
                "(additional files)",
                "skipped",
                f"over the per-import cap of {MAX_FILES_PER_IMPORT} files",
            )
        )
    return report
