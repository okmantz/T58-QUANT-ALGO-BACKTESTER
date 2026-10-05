"""v7 static regression -- release workflows publish SHA-256 checksums.

The app's download_update() verifies the release zip against a published
`<zip>.sha256` file and REFUSES the download when it's missing, so the
workflows must publish one alongside every release zip. This test reads
the workflow YAMLs as text (no YAML parser needed, no Windows runner
needed) and fails loudly if the checksum step ever goes missing again.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# workflow file -> release zip it produces
WORKFLOWS = {
    ".github/workflows/build-exe.yml": "T58-Quant-Algo-Backtester-Windows.zip",
    ".github/workflows/build-web-exe.yml": "T58-Web-App-Windows.zip",
}


def _read(rel: str) -> str:
    path = REPO_ROOT / rel
    assert path.exists(), f"workflow file missing: {rel}"
    return path.read_text(encoding="utf-8")


def test_checksum_step_present_in_both_workflows():
    for rel, zip_name in WORKFLOWS.items():
        text = _read(rel).lower()
        assert "get-filehash" in text and "sha256" in text, (
            f"{rel}: no SHA-256 checksum step found "
            f"(expected a Get-FileHash -Algorithm SHA256 step)"
        )


def test_checksum_file_written_beside_the_zip():
    for rel, zip_name in WORKFLOWS.items():
        text = _read(rel)
        assert f"{zip_name}.sha256" in text, (
            f"{rel}: the checksum step must write {zip_name}.sha256"
        )


def test_checksum_attached_to_github_release():
    # The checksum is useless if it never leaves the runner: it must be
    # attached to the GitHub Release next to the zip (softprops
    # action-gh-release `files:`), where download_update() fetches it.
    for rel, zip_name in WORKFLOWS.items():
        text = _read(rel)
        assert "action-gh-release" in text, f"{rel}: no release-attach step"
        release_block = text.split("action-gh-release")[-1]
        # only look at the release step's own `files:` input, not the whole file
        files_input = release_block.split("files:")[1].split("\n\n")[0]
        assert f"{zip_name}.sha256" in files_input, (
            f"{rel}: {zip_name}.sha256 is not attached to the GitHub Release"
        )
        assert zip_name in files_input, f"{rel}: {zip_name} missing from release files"


def test_checksum_uploaded_as_artifact():
    # Artifact parity: the checksum should travel with the zip in the
    # uploaded artifact too, for debugging a release without re-running.
    for rel, zip_name in WORKFLOWS.items():
        text = _read(rel)
        assert "upload-artifact" in text, f"{rel}: no artifact-upload step"
        artifact_block = text.split("upload-artifact")[-1].split("\n\n")[0]
        assert f"{zip_name}.sha256" in artifact_block, (
            f"{rel}: {zip_name}.sha256 is not in the uploaded artifact"
        )
