"""Saved portfolios.

A portfolio is a recipe (which saved strategy on which instrument, at what weight, under what account and
engine settings) plus the verified result it produced. It lives next to the strategy library, in
``<app base>/portfolios/*.portfolio.json``, so it follows the same persistent-folder rules (packaged exe,
plain checkout, container) as ``app.strategy.library`` without touching that module's per-language folders.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from app.data.storage import get_app_base_dir

SUFFIX = ".portfolio.json"


class SavedPortfolioNotFound(FileNotFoundError):
    pass


def portfolio_library_dir() -> Path:
    d = Path(get_app_base_dir()) / "portfolios"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stem(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._ -]+", "", (name or "").strip()).strip(" .") or "portfolio"
    return s[:80]


def save_portfolio(name: str, spec: dict, result_summary: dict | None = None, overwrite: bool = False) -> Path:
    """spec: {"engine", "initial_balance", "legs": [{"name", "strategy_mode", "strategy_name", "weight", "dataset"}], ...}."""
    d = portfolio_library_dir()
    path = d / (_stem(name) + SUFFIX)
    if path.exists() and not overwrite:
        i = 2
        while (d / f"{_stem(name)} ({i}){SUFFIX}").exists():
            i += 1
        path = d / f"{_stem(name)} ({i}){SUFFIX}"
    payload = {"name": name, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "spec": spec,
               "result": _jsonable(result_summary)}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def list_portfolios() -> list[dict]:
    out = []
    for p in sorted(portfolio_library_dir().glob("*" + SUFFIX), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        res = data.get("result") or {}
        out.append({"filename": p.name, "name": data.get("name", p.stem), "saved_at": data.get("saved_at"),
                    "engine": (data.get("spec") or {}).get("engine"),
                    "legs": [l.get("name") for l in (data.get("spec") or {}).get("legs", [])],
                    "net_profit": res.get("net_profit"), "max_dd": res.get("max_dd")})
    return out


def load_portfolio(filename: str) -> dict:
    p = portfolio_library_dir() / Path(filename).name
    if not p.exists():
        raise SavedPortfolioNotFound(filename)
    return json.loads(p.read_text(encoding="utf-8"))


def delete_portfolio(filename: str) -> bool:
    p = portfolio_library_dir() / Path(filename).name
    if p.exists():
        p.unlink()
        return True
    return False


def _jsonable(obj):
    if obj is None:
        return None
    return json.loads(json.dumps(obj, default=str))
