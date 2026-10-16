"""Test helper: the desktop UI's source as one text blob.

app/ui/main_window.py used to hold the whole UI; it is now a thin facade over
app/ui/mixins/*.py (+ theme/common/widgets). Tests that assert on the UI's *source text*
read it through here so they keep working wherever the code lives.
"""
from __future__ import annotations

from pathlib import Path

_UI = Path(__file__).resolve().parents[1] / "app" / "ui"


def desktop_ui_source() -> str:
    files = [_UI / "main_window.py", _UI / "theme.py", _UI / "common.py"]
    files += sorted((_UI / "mixins").glob("*.py"))
    files += sorted((_UI / "widgets").glob("*.py"))
    return "\n".join(p.read_text(encoding="utf-8") for p in files)
