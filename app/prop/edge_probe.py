"""
Edge probe for the Interactive Replay's prop-firm evaluation simulator.

The simulator (app/web/static/t58-propsim.js) only needs four numbers to model an
attempt: win rate, average win in R (loss = 1R), trades per day and expectancy.
Until now it could only read those from a strategy's *recorded* last run, which meant
every strategy that had never been backtested (all the preloaded ones) was filtered out
of the "load a saved strategy" dropdown.

``measure_edge`` closes that gap: it loads ANY saved strategy (manual JSON, Python,
PineScript or MQL5) through the universal loader, backtests it on a chosen (or
auto-matched) stored dataset using the same ``run_backtest`` every other tool uses,
and returns the simulator's inputs measured from the real trade list. Results are
cached on disk so re-picking a strategy is instant.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Raw bars read for the probe. Enough for a few hundred trades on intraday data while
# keeping the call fast; the dataset is trimmed from the END (most recent bars).
DEFAULT_MAX_RAW_BARS = 250_000
_CACHE_TTL_SECONDS = 7 * 24 * 3600

# Instrument code -> substrings that identify its data folder / file names.
_INSTRUMENT_ALIASES = {
    "NQ": ("NASDAQ", "NAS100", "USATECH", "NQ", "USTEC", "MNQ"),
    "ES": ("S&P", "SPX", "US500", "ES1", "ES_", "MES"),
    "GC": ("XAU", "GOLD", "GC", "MGC"),
}


def _cache_path() -> Path:
    from app.data.storage import get_app_base_dir
    d = Path(get_app_base_dir()) / "data" / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d / "propsim_edge_cache.json"


def _load_cache() -> dict:
    try:
        return json.loads(_cache_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_cache(cache: dict) -> None:
    try:
        _cache_path().write_text(json.dumps(cache), encoding="utf-8")
    except Exception:  # noqa: BLE001 -- the cache is a convenience only
        pass


def suggest_dataset(strategy_type: str, name: str, meta: dict | None = None) -> str | None:
    """Best stored dataset (value usable as ``existing_dataset``) for a strategy: matches the
    instrument named in its market tag / filename, preferring a bar size close to the one in
    its name (e.g. '... 5m'), else the largest-history file for that instrument."""
    from app.data.storage import list_datasets_by_instrument
    from app.strategy.library_control import normalize_instrument, normalize_timeframe

    groups = list_datasets_by_instrument(count_rows=False)
    if not groups:
        return None
    meta = meta or {}
    code = normalize_instrument(meta.get("market"), name)
    aliases = _INSTRUMENT_ALIASES.get(code, (code,)) if code != "OTHER" else ()
    want_tf = normalize_timeframe(meta.get("timeframe"), name)
    want_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}.get(want_tf)

    def file_minutes(fname: str) -> int | None:
        import re
        stem = Path(fname).stem.lower()
        if "1m" in stem and "parquet" not in fname:
            return 1
        m = re.search(r"(\d+)$", stem)
        if m and int(m.group(1)) in (1, 5, 15, 30, 60, 240, 1440):
            return int(m.group(1))
        m = re.search(r"_(\d+)m\b", stem)
        if m:
            return int(m.group(1))
        return None

    best, best_score = None, None
    for g in groups:
        gname = str(g.get("instrument", ""))
        for f in g.get("files", []):
            hay = f"{gname} {f.get('name', '')}".upper()
            if aliases and not any(a.upper() in hay for a in aliases):
                continue
            if f.get("empty"):
                continue
            mins = file_minutes(str(f.get("name", "")))
            tf_pen = 0 if (want_minutes is None or mins is None) else abs(np.log2(max(mins, 1)) - np.log2(want_minutes))
            size_bonus = -min(float(f.get("size_bytes", 0)) / 5e7, 1.0)
            score = tf_pen + size_bonus
            if best_score is None or score < best_score:
                best, best_score = f.get("full_name"), score
    if best:
        return best
    # No instrument match: first non-empty file overall so the probe still works.
    for g in groups:
        for f in g.get("files", []):
            if not f.get("empty"):
                return f.get("full_name")
    return None


def _trade_r_multiples(trades: list) -> list[float]:
    out: list[float] = []
    for t in trades:
        denom = None
        ird = getattr(t, "intended_risk_dollars", None)
        if ird and ird > 0:
            denom = float(ird)
        else:
            ir = getattr(t, "initial_risk", None)
            size = getattr(t, "size", None)
            if ir and size:
                denom = abs(float(ir) * float(size))
        if denom and denom > 0:
            out.append(float(t.pnl) / denom)
    return out


def measure_edge(strategy_type: str, name: str, dataset: str | None = None,
                 max_raw_bars: int = DEFAULT_MAX_RAW_BARS, use_cache: bool = True) -> dict[str, Any]:
    """Backtest one saved library strategy and return the simulator's edge inputs.

    Raises ValueError with a readable message for anything the person can act on
    (unknown strategy, no data available, strategy takes no trades)."""
    from app.backtest.engine import run_backtest
    from app.backtest.risk import RiskConfig, suggest_pip_size
    from app.data.importer import import_csv
    from app.data.instrument_specs import apply_instrument_spec, get_instrument_spec, guess_instrument_symbol
    from app.data.storage import resolve_stored_dataset
    from app.data.timeframe_resample import prepare_timeframe_aligned_data
    from app.strategy.library import list_saved_strategies
    from app.strategy.library_loader import load_strategy_object

    stored = next((s for s in list_saved_strategies(strategy_type) if s.name == name), None)
    if stored is None:
        raise ValueError(f"Saved strategy '{name}' ({strategy_type}) was not found in the library.")

    if not dataset:
        dataset = suggest_dataset(strategy_type, name, stored.metadata)
    if not dataset:
        raise ValueError("No market data is available to measure this strategy on. Add data under Account > Data Center "
                         "or fetch it from a data provider on any run page first.")

    key = hashlib.sha1(f"{strategy_type}|{name}|{dataset}|{max_raw_bars}|{stored.modified}".encode()).hexdigest()
    cache = _load_cache()
    hit = cache.get(key)
    if use_cache and hit and (time.time() - hit.get("at", 0)) < _CACHE_TTL_SECONDS:
        return {**hit["result"], "cached": True}

    path = resolve_stored_dataset(dataset)
    if path is None:
        raise ValueError(f"Dataset '{dataset}' could not be found.")
    imported = import_csv(path)
    if not imported.is_valid:
        raise ValueError("That dataset could not be read: " + "; ".join(imported.errors))
    df = imported.dataframe
    if len(df) > max_raw_bars:
        df = df.tail(max_raw_bars).reset_index(drop=True)

    strategy = load_strategy_object(stored)
    symbol = guess_instrument_symbol(dataset)
    risk = RiskConfig(initial_balance=100000.0, risk_mode="percent", risk_value=0.5,
                      pip_size=suggest_pip_size(df), contract_size=None)
    if symbol and get_instrument_spec(symbol) is not None:
        risk = apply_instrument_spec(risk, symbol)

    aligned, _ = prepare_timeframe_aligned_data(df, strategy)
    result = run_backtest(df, strategy, risk, lookahead_check="skip")
    trades = result.trades
    if not trades:
        raise ValueError(f"'{name}' took no trades on {dataset} (last {len(aligned):,} bars). Try a different dataset.")

    rs = _trade_r_multiples(trades)
    pnls = np.array([float(t.pnl) for t in trades])
    if len(rs) >= max(5, len(trades) // 2):
        r = np.array(rs)
        wins, losses = r[r > 0], r[r <= 0]
        avg_win_r = float(wins.mean()) if len(wins) else 0.0
        avg_loss_r = float(abs(losses.mean())) if len(losses) else 1.0
    else:  # no per-trade risk recorded -> fall back to dollar P&L ratios
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        avg_win_r = float(wins.mean()) if len(wins) else 0.0
        avg_loss_r = float(abs(losses.mean())) if len(losses) else 1.0
        r = pnls / max(avg_loss_r, 1e-9)
    win_rate = float((np.array(pnls) > 0).mean())
    payoff = avg_win_r / avg_loss_r if avg_loss_r > 1e-9 else float(avg_win_r)
    days = pd.to_datetime([t.entry_time for t in trades]).normalize().nunique()
    first, last = pd.to_datetime(trades[0].entry_time), pd.to_datetime(trades[-1].exit_time)
    gross_w, gross_l = float(pnls[pnls > 0].sum()), float(abs(pnls[pnls <= 0].sum()))
    out = {
        "ok": True, "strategy": name, "type": strategy_type, "dataset": dataset,
        "bars_used": int(len(aligned)), "trades": int(len(trades)), "trading_days": int(days),
        "win_rate": round(win_rate * 100, 1), "payoff_r": round(max(payoff, 0.05), 2),
        "trades_per_day": round(len(trades) / max(days, 1), 2),
        "profit_factor": round(gross_w / gross_l, 2) if gross_l > 0 else None,
        "expectancy_r": round(float(np.mean(r)), 3),
        "period": f"{first.date()} -> {last.date()}",
        "cached": False,
    }
    cache[key] = {"at": time.time(), "result": out}
    if len(cache) > 400:
        for k in sorted(cache, key=lambda k: cache[k].get("at", 0))[:100]:
            cache.pop(k, None)
    _save_cache(cache)
    return out
