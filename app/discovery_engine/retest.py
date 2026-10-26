"""Automatic out-of-sample retests: the lab is ranked by what its champions do on data they have never seen.

Every champion and every near-miss is registered with the end of the data window it was selected on. When newer bars arrive, the
frozen candidate is re-run on ONLY the bars after that window (it cannot have been tuned on them) and the outcome is appended to its
record. The lab's own track record -- by family, by preset, by generator -- is then the share of its past champions whose edge
survived, shrunk toward the prior so three lucky retests do not crown a family.

This is the honest answer to "which parts of the lab actually find things": not the in-sample score they were crowned with.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.discovery_engine import objective as obj

MIN_NEW_BARS = 1500            # about 19 sessions of 5-minute bars: less than that is noise, not a retest
MIN_RETEST_TRADES = 30
PRIOR_HITS, PRIOR_N = 1.0, 4.0   # shrinkage: a family with no retests ranks at 25%, not at its first lucky run


def default_path() -> Path:
    from app.validation import trial_ledger
    return Path(trial_ledger._PATH).parent / "retest_log.json"  # noqa: SLF001


def spec_key(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _end_of(df: pd.DataFrame):
    if df is None or not len(df):
        return None
    col = df["timestamp"] if "timestamp" in df.columns else pd.Series(df.index)
    return pd.to_datetime(col).iloc[-1]


class RetestStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else default_path()

    def _load(self) -> dict:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def _save(self, d: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:  # noqa: BLE001 -- best effort
            pass

    def register(self, spec: dict, *, kind: str, window_end, family: str | None = None, preset: str = "", instrument: str = "",
                 recorded: dict | None = None) -> str:
        """Record a candidate (kind "champion" | "near_miss") with the end of the data window it was selected on. Idempotent per spec."""
        d = self._load()
        key = spec_key(spec)
        rec = d.get(key)
        if rec is None:
            rec = {"spec": spec, "family": family, "preset": preset, "instrument": instrument, "kind": kind, "registered_at": time.time(),
                   "window_end": str(pd.Timestamp(window_end)) if window_end is not None else None, "recorded": recorded or {}, "history": []}
        elif kind == "champion":
            rec["kind"] = "champion"          # a near-miss that later becomes a champion is a champion
        d[key] = rec
        self._save(d)
        return key

    def all(self) -> dict:
        return self._load()

    def due(self, df: pd.DataFrame, instrument: str = "", min_new_bars: int = MIN_NEW_BARS) -> list:
        """Keys of candidates (for this instrument, when given) with at least ``min_new_bars`` bars of unseen data in ``df``."""
        out = []
        ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.Series(df.index)
        for k, rec in self._load().items():
            if instrument and rec.get("instrument") and rec["instrument"] != instrument:
                continue
            last = (rec["history"][-1]["window_end"] if rec["history"] else rec.get("window_end"))
            if last is None:
                continue
            if int((ts > pd.Timestamp(last)).sum()) >= min_new_bars:
                out.append(k)
        return out

    def record(self, key: str, outcome: dict) -> None:
        d = self._load()
        if key in d:
            d[key]["history"].append(outcome)
            self._save(d)


def retest_candidate(rec: dict, df: pd.DataFrame, risk, *, min_trades: int = MIN_RETEST_TRADES) -> dict | None:
    """Re-run the frozen spec on ONLY the bars after its last window. None when there is not enough new data."""
    from app.discovery_engine.engine_steps import edge_record, evaluate_spec
    last = rec["history"][-1]["window_end"] if rec.get("history") else rec.get("window_end")
    if last is None:
        return None
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.Series(df.index)
    new = df[(ts > pd.Timestamp(last)).to_numpy()].reset_index(drop=True)
    if len(new) < MIN_NEW_BARS:
        return None
    trades = evaluate_spec(rec["spec"], new, risk, 1.0)
    e = edge_record(trades, rec["spec"])
    enough = e["n_trades"] >= min_trades
    # a hit: positive mean R after costs with a one-sided t above 1.28 (~90%), on enough trades. Too few trades = inconclusive, not a miss.
    hit = bool(enough and e["mean_r"] > 0 and np.isfinite(e["t_stat"]) and e["t_stat"] >= 1.28)
    return {"at": time.time(), "window_start": str(pd.to_datetime(new["timestamp"]).iloc[0]),
            "window_end": str(_end_of(new)), "n_bars": int(len(new)), **e, "conclusive": bool(enough), "hit": hit}


def run_retests(df: pd.DataFrame, risk, store: RetestStore | None = None, *, instrument: str = "", min_new_bars: int = MIN_NEW_BARS,
                progress=None) -> list:
    """Retest every due candidate on its unseen bars; appends to the store and returns [(key, outcome)]."""
    store = store or RetestStore()
    log = progress or (lambda m: None)
    data = store.all()
    done = []
    for key in store.due(df, instrument, min_new_bars):
        out = retest_candidate(data[key], df, risk)
        if out is None:
            continue
        store.record(key, out)
        done.append((key, out))
        log(f"retest {key} ({data[key].get('family')}): n={out['n_trades']} meanR={out['mean_r']:+.3f} t={out['t_stat']:.2f} -> {'HIT' if out['hit'] else ('miss' if out['conclusive'] else 'inconclusive')}")
    return done


def _score(hits: float, n: float) -> float:
    return (hits + PRIOR_HITS) / (n + PRIOR_N)


def lab_ranking(store: RetestStore | None = None, by: str = "family") -> list:
    """Rank families / presets by their out-of-sample record. Each row: {name, retests, hits, shrunk_hit_rate, mean_oos_r}; best first.
    Only CONCLUSIVE retests count (an inconclusive one says nothing); champions and near-misses both count, champions weigh double."""
    rows: dict = {}
    for rec in (store or RetestStore()).all().values():
        name = rec.get(by) or "unknown"
        w = 2.0 if rec.get("kind") == "champion" else 1.0
        for h in rec.get("history", []):
            if not h.get("conclusive"):
                continue
            r = rows.setdefault(name, {"name": name, "retests": 0.0, "hits": 0.0, "r_sum": 0.0})
            r["retests"] += w
            r["hits"] += w * (1.0 if h.get("hit") else 0.0)
            r["r_sum"] += w * float(h.get("mean_r") or 0.0)
    out = []
    for r in rows.values():
        out.append({"name": r["name"], "retests": r["retests"], "hits": r["hits"], "shrunk_hit_rate": _score(r["hits"], r["retests"]),
                    "mean_oos_r": r["r_sum"] / r["retests"] if r["retests"] else 0.0})
    return sorted(out, key=lambda x: (x["shrunk_hit_rate"], x["mean_oos_r"]), reverse=True)


def register_run(result, df: pd.DataFrame, store: RetestStore | None = None, instrument: str = "") -> int:
    """Register an ``EngineResult``'s champions (passed every step) and near-misses (positive edge, failed a step) for later retests."""
    store = store or RetestStore()
    n = 0
    end = _end_of(df)
    for c, p in zip(result.champions, result.promotions):
        if not c.get("spec"):
            continue
        kind = "champion" if p.get("passed") else ("near_miss" if (c.get("edge_lb_r") or -1) > 0 else None)
        if kind:
            store.register(c["spec"], kind=kind, window_end=end, family=c.get("family"), preset=result.preset, instrument=instrument,
                           recorded=c.get("edge") or {})
            n += 1
    return n
