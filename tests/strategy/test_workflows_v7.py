"""v7 static regression -- release workflows publish SHA-256 checksums.

The app's download_update() verifies the release zip against a published
`<zip>.sha256` file and REFUSES the download when it's missing, so the
workflows must publish one alongside every release zip. This test reads
the workflow YAMLs as text (no YAML parser needed, no Windows runner
needed) and fails loudly if the checksum step ever goes missing again.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# workflow file -> release zip name pattern it produces (one zip per OS, named via $PACKAGE)
WORKFLOWS = {
    ".github/workflows/desktop-cross-platform.yml": "T58-Quant-Algo-Backtester-${{ matrix.label }}.zip",
    ".github/workflows/web-cross-platform.yml": "T58-Web-App-${{ matrix.label }}.zip",
}
PKG = "${{ env.PACKAGE }}"


def _read(rel: str) -> str:
    path = REPO_ROOT / rel
    assert path.exists(), f"workflow file missing: {rel}"
    return path.read_text(encoding="utf-8")


def test_checksum_step_present_in_both_workflows():
    # build_tools/package_release.py writes <zip>.sha256 for every app zip
    # (and CRC-tests the archive), on every OS.
    for rel, zip_name in WORKFLOWS.items():
        text = _read(rel)
        assert "package_release.py app" in text and f"PACKAGE: {zip_name}" in text, rel
    assert "sha256" in (REPO_ROOT / "build_tools/package_release.py").read_text(encoding="utf-8").lower()


def test_checksum_file_written_beside_the_zip():
    for rel in WORKFLOWS:
        text = _read(rel)
        assert f"{PKG}.sha256" in text, f"{rel}: the checksum file must be uploaded next to the zip"


def test_checksum_attached_to_github_release():
    for rel in WORKFLOWS:
        text = _read(rel)
        assert "action-gh-release" in text, f"{rel}: no release-attach step"
        files_input = text.split("action-gh-release")[-1].split("files:")[1]
        assert f"{PKG}.sha256" in files_input and PKG in files_input, rel


def test_checksum_uploaded_as_artifact():
    for rel in WORKFLOWS:
        text = _read(rel)
        assert "upload-artifact" in text, f"{rel}: no artifact-upload step"
        artifact_block = text.split("upload-artifact")[1].split("- name:")[0]
        assert f"{PKG}.sha256" in artifact_block, rel
