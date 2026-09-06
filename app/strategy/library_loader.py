"""
Turns a Strategy Library entry (app.strategy.library.StoredStrategy) into
a runnable app.strategy.base.Strategy object. Single source of truth for
this so app.strategy.auto_regime_selector and app.portfolio.composer (and
any future caller) don't each reinvent "how do I load a library item
given its strategy_type" slightly differently -- app.web.server has its
own build_strategy_from_code() for the same purpose given raw code text
already resolved from a request, but that helper lives in the web layer
and doesn't cover "manual" (JSON) strategies or reading straight from the
library's own saved file, which every backend-only caller needs.
"""
from __future__ import annotations

import json

from app.strategy.base import Strategy, StrategyError
from app.strategy.library import StoredStrategy, list_saved_strategies, load_strategy_text
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy


def load_strategy_object(stored: StoredStrategy) -> Strategy:
    """Instantiates the right Strategy subclass for a library entry.
    Python strategies are loaded directly from their saved path on disk
    (no tempfile copy needed -- the library file already IS a real .py
    file); the other three types are loaded from their saved text."""
    if stored.strategy_type == "python":
        return PythonStrategy(stored.path)
    text = load_strategy_text(stored.strategy_type, stored.name)
    if stored.strategy_type == "pinescript":
        return PineScriptStrategy(text)
    if stored.strategy_type == "mql5":
        return MQL5Strategy(text)
    if stored.strategy_type == "manual":
        try:
            config = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StrategyError(f"'{stored.name}' is not valid JSON: {exc}") from exc
        return ManualStrategy(config)
    raise StrategyError(f"Unknown strategy_type '{stored.strategy_type}' for library entry '{stored.name}'.")


def load_validated_candidates(
    strategy_type: str | None = None,
    status: str = "validated",
    tag: str | None = None,
    market: str | None = None,
) -> dict[str, Strategy]:
    """Convenience wrapper: pulls every Strategy Library entry matching
    the given filters (default: status == "validated", every language)
    and returns {display_name: Strategy}. A single bad file (parse
    error, corrupt JSON) is skipped with its filename still reported via
    the returned dict's absence -- callers that want to know WHY an
    expected strategy is missing should call list_saved_strategies() and
    load_strategy_object() themselves instead for per-item error handling."""
    out: dict[str, Strategy] = {}
    for stored in list_saved_strategies(strategy_type=strategy_type, status=status, tag=tag, market=market):
        try:
            out[stored.name] = load_strategy_object(stored)
        except Exception:  # noqa: BLE001 -- one bad library file must not break the whole pool
            continue
    return out
