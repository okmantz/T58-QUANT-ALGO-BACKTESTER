"""Fill ledger: intended vs actual fills, and the realised cost per instrument (report section 5, "real fill costs").

Every paper or live fill is appended as (time, symbol, side, intended_price, fill_price, size, commission). ``realised_costs``
turns the ledger into slippage in TICKS per side, which ``to_broker_profile`` converts into an app.data.cost_model.BrokerProfile -
so backtests and the 2x stress can run on MEASURED costs instead of the registry's typical ones.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "reports" / "fill_ledger.jsonl"


def record_fill(symbol: str, side: int, intended_price: float, fill_price: float, size: float, commission: float = 0.0,
                when: float | None = None, kind: str = "entry", path: Path | None = None) -> dict:
    row = {"t": time.time() if when is None else float(when), "symbol": symbol.upper(), "side": int(side), "intended": float(intended_price),
           "fill": float(fill_price), "size": float(size), "commission": float(commission), "kind": kind}
    p = Path(path) if path else DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load_fills(path: Path | None = None) -> list[dict]:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return out


def realised_costs(fills: list[dict] | None = None, path: Path | None = None, min_fills: int = 20) -> dict:
    """Mean adverse slippage in ticks per side, per symbol. Adverse = paid more than intended on buys, received less on sells."""
    from app.data.instrument_specs import get_any_instrument_spec
    rows = fills if fills is not None else load_fills(path)
    by: dict[str, list[float]] = {}
    for r in rows:
        spec = get_any_instrument_spec(r["symbol"])
        if spec is None or spec.tick_size <= 0:
            continue
        adverse = (r["fill"] - r["intended"]) * (1 if r["side"] > 0 else -1)
        by.setdefault(r["symbol"], []).append(adverse / spec.tick_size)
    out = {}
    for sym, xs in by.items():
        a = np.asarray(xs, dtype=float)
        out[sym] = {"n": int(a.size), "mean_slippage_ticks": float(a.mean()), "p90_ticks": float(np.percentile(a, 90)),
                    "enough": bool(a.size >= min_fills)}
    return out


def to_broker_profile(name: str = "measured", fills: list[dict] | None = None, path: Path | None = None, min_fills: int = 20):
    """BrokerProfile carrying measured slippage (spread folded in: measured slippage is vs the intended/quoted price, so the
    spread tick count is set to 0 for instruments with enough fills to avoid double counting)."""
    from app.data.cost_model import BrokerProfile
    prof = BrokerProfile(name)
    for sym, d in realised_costs(fills, path, min_fills).items():
        if d["enough"]:
            prof.slippage_ticks[sym] = max(0.0, d["mean_slippage_ticks"])
            prof.spread_ticks[sym] = 0.0
    return prof
