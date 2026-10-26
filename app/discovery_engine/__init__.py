"""Strategy Discovery Engine -- the single discovery lab.

Search Lab, Evolution Lab, Forge, Speed Run and the app/discovery layer are one engine with one objective (``edge_lb_r``), one
entry floor, one lifetime trial ledger and one promotion sequence. The old lab names survive as *presets* (see ``presets``).

Importing this package is cheap: heavy submodules are imported lazily by the code that needs them.
"""
from app.discovery_engine.objective import (  # noqa: F401
    DEFAULT_FLOOR,
    EDGE_FOLD_METRIC,
    EDGE_OBJECTIVE,
    EntryFloor,
    entry_floor_reasons,
    passes_entry_floor,
)

ENGINE_NAME = "Strategy Discovery Engine"
