"""
T58 Licensing -- an isolated boundary around the application, not
something embedded into individual features. See client.py and gate.py's
own module docstrings for the full architecture.

Public API:
    ensure_licensed()  -- call once at startup, before launching the GUI.
    client.deactivate() -- "log out" / free this device's license slot.
"""
from __future__ import annotations

from app.licensing.gate import ensure_licensed

__all__ = ["ensure_licensed"]
