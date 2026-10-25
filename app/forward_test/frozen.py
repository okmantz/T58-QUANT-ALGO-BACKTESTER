"""Frozen-code guard for the forward test (report 2026-10-20, section 6).

A forward test only counts if the thing being tested never changes. ``freeze`` hashes the strategy config + risk config + the
cost profile + the code of the modules that generate signals and fills; ``verify`` refuses to continue a forward test when any of
them differ, so "I just tweaked a threshold" cannot silently restart the evidence clock.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

_CODE_MODULES = (
    "app/backtest/engine.py", "app/backtest/execution.py", "app/strategy/manual.py", "app/strategy/indicators.py",
    "app/strategy/indicators_v11.py", "app/strategy/indicators_v12.py", "app/data/cost_model.py",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def code_hash(modules=_CODE_MODULES) -> str:
    h = hashlib.sha256()
    for m in modules:
        p = _root() / m
        h.update(m.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()


def _jsonable(x):
    if is_dataclass(x) and not isinstance(x, type):
        return asdict(x)
    return x


def freeze(strategy_config: dict, risk=None, cost_line=None, include_code: bool = True) -> dict:
    payload = {"strategy": strategy_config, "risk": _jsonable(risk), "cost": _jsonable(cost_line)}
    cfg_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return {"config_hash": cfg_hash, "code_hash": code_hash() if include_code else None, "payload": payload}


def verify(frozen: dict, strategy_config: dict, risk=None, cost_line=None) -> tuple[bool, list]:
    now = freeze(strategy_config, risk, cost_line, include_code=frozen.get("code_hash") is not None)
    problems = []
    if now["config_hash"] != frozen["config_hash"]:
        problems.append("strategy/risk/cost configuration changed since the forward test was frozen")
    if frozen.get("code_hash") and now["code_hash"] != frozen["code_hash"]:
        problems.append("signal/fill code changed since the forward test was frozen")
    return (not problems), problems
