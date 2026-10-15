"""One naming rule for every saved report: ``<Strategy>__<Test>__<YYYYmmdd-HHMMSS>``.

Reports used to be saved under a job id or a bare timestamp (``cpcv_3f9a1c.html``,
``full_pipeline_ab12cd34.html``), so a folder of reports said nothing about WHAT was tested or
HOW. Every report-writing tool now builds its file stem here, so the file title always carries the
strategy name and the kind of test (Full Pipeline, Validation, Optimize, CPCV, ...), e.g.::

    NQ1_15m_EMA_Cross__Full-Pipeline__20261014-204312.html
    NQ1_15m_EMA_Cross__Validation__20261014-204930.json

The stem is filesystem-safe on Windows/macOS/Linux (no spaces, slashes, colons or other characters
that break a path or a URL), is capped in length, and never collides with an existing report in the
same folder (a ``_2``, ``_3`` ... suffix is added instead of overwriting an earlier run).
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

_MAX_STRATEGY_CHARS = 60
_MAX_TEST_CHARS = 30

# Sibling files a tool may write next to the main report; a stem is "taken" if any of these exist.
_KNOWN_SUFFIXES = (".html", ".json", ".txt", ".csv", "_summary.csv", "_trades.csv", ".pdf", ".md")


_STRATEGY_EXTENSIONS = (".json", ".py", ".pine", ".pinescript", ".mq5", ".mql5")


def strip_strategy_extension(name: str) -> str:
    """"NQ1 15m EMA.json" -> "NQ1 15m EMA" (library filenames carry the file extension)."""
    text = str(name or "").strip()
    low = text.lower()
    for ext in _STRATEGY_EXTENSIONS:
        if low.endswith(ext):
            return text[: -len(ext)]
    return text


def safe_part(text: str, *, max_chars: int, fallback: str) -> str:
    """Letters, digits, dot, underscore and dash only; runs of anything else become one underscore."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip()).strip("._-")
    cleaned = re.sub(r"_{2,}", "_", cleaned)
    return cleaned[:max_chars].strip("._-") or fallback


def strategy_label(strategy=None, fallback: str = "Strategy", *, library_ref=None, explicit: str = "") -> str:
    """Best human name for "the strategy that was tested", in priority order: an explicit name the
    caller already resolved, the Strategy Library filename, the strategy object's own name."""
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    if library_ref:
        try:
            return Path(str(library_ref[1])).stem
        except Exception:  # noqa: BLE001
            pass
    name = getattr(strategy, "name", None) if strategy is not None else None
    if name and str(name).strip():
        return str(name).strip()
    return fallback


def report_stem(strategy_name: str, test_label: str, *, when: Optional[float] = None,
                output_dir: Optional[Path] = None) -> str:
    """``<Strategy>__<Test>__<timestamp>`` (see module docstring). When ``output_dir`` is given the
    stem is made unique inside it."""
    strat = safe_part(strip_strategy_extension(strategy_name), max_chars=_MAX_STRATEGY_CHARS, fallback="Strategy")
    test = safe_part(str(test_label or "").replace(" ", "-"), max_chars=_MAX_TEST_CHARS, fallback="Report")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(when if when is not None else time.time()))
    stem = f"{strat}__{test}__{stamp}"
    if output_dir is None:
        return stem
    folder = Path(output_dir)
    candidate, n = stem, 1
    while any((folder / f"{candidate}{suf}").exists() for suf in _KNOWN_SUFFIXES):
        n += 1
        candidate = f"{stem}_{n}"
    return candidate


def report_title(strategy_name: str, test_label: str) -> str:
    """Human-readable form for page titles / headings: ``NQ1 15m EMA Cross — Full Pipeline``."""
    return f"{str(strategy_name or 'Strategy').strip()} — {str(test_label or '').strip()}".rstrip(" —")
