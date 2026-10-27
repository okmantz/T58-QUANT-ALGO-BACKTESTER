"""Discovery Lab -- the fast, cost-aware strategy finder behind the Discovery Engine page.

    from app.discovery_lab import run_lab, LabConfig
    result = run_lab(df, cost_price=..., cfg=LabConfig(effort="standard"), progress=print)

See engine.py for the method (train -> validate -> sealed test + neighbour-robustness + 2x cost stress) and families.py for the
strategy families / indicators it searches.
"""
from .engine import LabConfig, LabResult, run_lab  # noqa: F401
