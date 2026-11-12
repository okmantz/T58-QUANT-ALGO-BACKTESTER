"""Shared helper for verdict-gate tests.

Since the 2026-11-12 audit (P1-10) READY needs a *measured* unseen-data expectancy bound (>= 30 trades after 2x costs), so a test
that wants to reach READY and then exercise a different gate must supply one. ``GOOD_BOUND`` is a comfortably positive bound.
"""
from __future__ import annotations

GOOD_BOUND = {"n": 200, "mean": 0.2, "lb": 0.05, "ub": 0.35}


def make_verdict_with_bound(*args, **kwargs):
    """``_make_verdict`` with a healthy expectancy bound unless the caller passes its own."""
    import app.orchestration.full_pipeline as fp
    kwargs.setdefault("expectancy_bound", dict(GOOD_BOUND))
    return fp._make_verdict(*args, **kwargs)
