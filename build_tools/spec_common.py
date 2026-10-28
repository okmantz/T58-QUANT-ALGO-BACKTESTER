"""Shared helpers for the PyInstaller spec files in this folder.

Both ``t58_desktop.spec`` and ``t58_web.spec`` import this module so the two
builds always bundle the same things. It is plain Python (no PyInstaller
import at module level) so ``tests/packaging/test_package_release.py`` can import and
check it on any machine.

Why specs instead of the long ``pyinstaller --hidden-import ...`` command
lines the workflows used before:

* ``--add-data "src;dest"`` uses ``;`` on Windows and ``:`` everywhere else.
  A spec file uses plain ``(src, dest)`` tuples, so the SAME build can be
  run and tested on Linux/macOS as well as on the Windows runner.
* The old hand-written ``--hidden-import`` list was missing most of the
  ``app`` package (licensing, optimize, search, quant_lab, live_deploy ...)
  and relied on PyInstaller's import tracing to find the rest.
  ``collect_submodules("app")`` bundles every module, so a lazily imported
  module can never be the thing that is missing in a packaged build.
* Both old workflows forgot ``config/`` (the Account tab then showed a
  fallback version) and the web build forgot ``app/ui/assets``.
"""
from __future__ import annotations

from pathlib import Path

# (source folder relative to the repo root, destination inside the bundle).
# Every entry is REQUIRED: a missing folder fails the build loudly instead of
# shipping a build that silently lacks it.
COMMON_DATA = [
    ("config", "config"),                    # config/pyproject.toml -> version shown in the Account tab
    ("strategies", "strategies"),            # built-in Strategy Library (seeded on first run)
    ("app/ui/assets", "app/ui/assets"),      # logo/icons used by the desktop UI and activation window
    ("app/search/seed_data", "app/search/seed_data"),  # starter Strategy Graveyard seed (merged on first use)
    ("research", "research"),                # bundled research papers (PDFs + reading lists) seeded into the Library on first use
]
WEB_DATA = [
    ("app/web/templates", "app/web/templates"),
    ("app/web/static", "app/web/static"),
]

# Third-party packages that are imported lazily (inside functions, by strategy
# files that are exec'd at runtime, or by entry points), so PyInstaller's static
# scan can miss them. A package that isn't installed is skipped with a warning
# by PyInstaller, never a failure.
LAZY_THIRD_PARTY = [
    "pandas", "numpy", "flask", "requests", "qrcode", "PIL",
    "sklearn", "sklearn.ensemble", "sklearn.linear_model", "sklearn.preprocessing",
    "sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes",
    "scipy", "scipy.optimize", "scipy.stats", "scipy.special", "scipy.sparse",
    "lightgbm", "optuna", "cma", "reportlab", "pypdf", "alpaca", "keyring", "psutil", "pynescript", "antlr4",
]

# Packages with data files / native pieces / plugin loading that need the
# heavier collect_all treatment (this is what the old workflows already did).
COLLECT_ALL = ["py7zr", "pyarrow"]

# Never needed at runtime; keeps the bundle smaller.
EXCLUDES = ["pytest", "_pytest", "IPython", "matplotlib", "notebook", "jupyter"]


def is_test_module(name: str) -> bool:
    """True for a dotted module name that belongs to a library's own test-suite
    (``pyarrow.tests.parquet.test_basic``, ``foo.testing`` is kept -- only real
    ``tests`` / ``test_*`` parts count)."""
    return any(part == "tests" or part.startswith("test_") or part == "conftest" for part in name.split("."))


def is_test_path(dest: str) -> bool:
    """True for a bundle-relative data path that sits inside a ``tests`` folder."""
    return "tests" in dest.replace("\\", "/").split("/")


def without_test_suites(hiddenimports: list[str], datas: list[tuple[str, str]]) -> tuple[list[str], list[tuple[str, str]]]:
    """Drop third-party test-suites that ``collect_all`` drags in.

    ``collect_all("pyarrow")`` alone bundles hundreds of ``pyarrow.tests.*``
    modules and data files: tens of MB nobody runs, and some of the deepest
    paths in the whole app (the ones that hit Windows' 260-character limit
    when a customer extracts into a long folder)."""
    hidden = [m for m in hiddenimports if not is_test_module(m)]
    kept = [(src, dest) for src, dest in datas if not is_test_path(dest)]
    return hidden, kept


def repo_root(spec_dir: str | Path) -> Path:
    """The repository root, given the folder the .spec file lives in."""
    return Path(spec_dir).resolve().parent


def data_entries(root: Path, *, web: bool) -> list[tuple[str, str]]:
    """PyInstaller ``datas`` tuples. Raises FileNotFoundError if a required
    folder is missing so a broken checkout cannot produce a broken build."""
    wanted = list(COMMON_DATA) + (list(WEB_DATA) if web else [])
    entries: list[tuple[str, str]] = []
    for src, dest in wanted:
        path = root / src
        if not path.is_dir():
            raise FileNotFoundError(
                f"build_tools/spec_common.py: required folder '{src}' not found under {root} "
                f"-- refusing to build an app that is missing it."
            )
        entries.append((str(path), dest))
    return entries
