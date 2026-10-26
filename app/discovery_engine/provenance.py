"""Provenance: every champion carries the seed, data fingerprint, git hash and full config it was found with, and is RE-RUN
before promotion. A result that does not reproduce within tolerance is dropped.

``data_fingerprint`` is also what the trial ledger uses to decide whether two searches looked at the SAME data window (same
fingerprint -> the selection bias of the earlier search is still fully present -> its trials must not decay).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_fingerprint(df: pd.DataFrame | None, *, sample_rows: int = 4096) -> str:
    """Stable 16-hex fingerprint of a bar frame: row count, first/last timestamp and a checksum of an evenly spaced sample of
    OHLCV values. Identical for the same data window however it was loaded; changes when bars are appended, removed or edited.
    (Hashing every bar of a multi-million-row frame would cost more than the search it guards; the evenly spaced sample plus
    both endpoints catches every realistic change.)"""
    if df is None or len(df) == 0:
        return "empty"
    h = hashlib.sha256()
    n = len(df)
    ts_col = "timestamp" if "timestamp" in df.columns else None
    first = str(df[ts_col].iloc[0]) if ts_col else str(df.index[0])
    last = str(df[ts_col].iloc[-1]) if ts_col else str(df.index[-1])
    h.update(f"{n}|{first}|{last}".encode())
    idx = np.unique(np.linspace(0, n - 1, num=min(n, int(sample_rows))).astype(int))
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            vals = pd.to_numeric(df[col].iloc[idx], errors="coerce").to_numpy(dtype=float)
            h.update(np.round(np.nan_to_num(vals, nan=-1.0), 6).tobytes())
    return h.hexdigest()[:16]


def data_window(df: pd.DataFrame | None) -> dict:
    """{"start", "end", "bars"} of a frame (ISO strings), for the provenance record and the retest registry."""
    if df is None or len(df) == 0:
        return {"start": None, "end": None, "bars": 0}
    if "timestamp" in df.columns:
        a, b = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    else:
        a, b = df.index[0], df.index[-1]
    return {"start": pd.Timestamp(a).isoformat(), "end": pd.Timestamp(b).isoformat(), "bars": int(len(df))}


def git_hash(root: Path | str | None = None) -> str:
    """Current commit hash of the repo (``T58_GIT_HASH`` env wins, for frozen builds); ``unknown`` when it cannot be read."""
    env = os.environ.get("T58_GIT_HASH")
    if env:
        return env.strip()
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root or REPO_ROOT), capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        try:
            return _jsonable(vars(obj))
        except Exception:  # noqa: BLE001
            return str(obj)
    return obj


@dataclass
class Provenance:
    """Everything needed to reproduce one champion."""

    seed: int | None
    data_fp: str
    git: str
    config: dict
    spec: dict | None = None
    instrument: str = ""
    timeframe: str = ""
    window: dict = field(default_factory=dict)
    objective: str = "edge_lb_r"
    recorded_edge_lb_r: float | None = None
    recorded_mean_r: float | None = None
    recorded_n_trades: int | None = None
    created_at: float = field(default_factory=time.time)
    env: dict = field(default_factory=lambda: {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__})

    def to_dict(self) -> dict:
        return _jsonable(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "Provenance":
        d = dict(d)
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        return cls(**{k: v for k, v in d.items() if k in known})

    def digest(self) -> str:
        """Hash of the reproducible inputs (seed, data, git, config, spec) -- two champions with the same digest are the same run."""
        payload = json.dumps(_jsonable({"seed": self.seed, "data": self.data_fp, "git": self.git, "config": self.config, "spec": self.spec}),
                             sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def capture(*, seed: int | None, df: pd.DataFrame | None, config, spec: dict | None = None, instrument: str = "",
            timeframe: str = "", edge: dict | None = None, git: str | None = None) -> Provenance:
    """Build the provenance record for a champion at the moment it is crowned."""
    cfg = _jsonable(config if isinstance(config, dict) else getattr(config, "__dict__", {"config": str(config)}))
    return Provenance(
        seed=None if seed is None else int(seed), data_fp=data_fingerprint(df), git=git or git_hash(), config=cfg, spec=_jsonable(spec),
        instrument=instrument, timeframe=timeframe, window=data_window(df),
        recorded_edge_lb_r=(edge or {}).get("edge_lb_r"), recorded_mean_r=(edge or {}).get("mean_r"),
        recorded_n_trades=(edge or {}).get("n_trades"),
    )


@dataclass
class ReproductionResult:
    reproduced: bool
    reason: str
    recorded: dict
    rerun: dict
    tolerance: dict

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _close(a, b, abs_tol: float, rel_tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if not (math.isfinite(a) and math.isfinite(b)):
        return (not math.isfinite(a)) and (not math.isfinite(b))
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def verify_reproduction(prov: Provenance, rerun: dict, *, df: pd.DataFrame | None = None, abs_tol: float = 0.02,
                        rel_tol: float = 0.10, trade_tol: float = 0.05, require_same_data: bool = True) -> ReproductionResult:
    """Compare a fresh re-run of the champion (``rerun`` = {edge_lb_r, mean_r, n_trades}) with what was recorded.

    Reproduces when the data fingerprint matches (unless ``require_same_data=False``), the edge lower bound and mean R agree
    within ``max(abs_tol, rel_tol * |value|)`` and the trade count within ``trade_tol`` (fraction). The edge bound is a bootstrap
    statistic, so a small tolerance is expected; a result that drifts by more than that is not reproducible and is dropped."""
    tol = {"abs_tol": abs_tol, "rel_tol": rel_tol, "trade_tol": trade_tol}
    recorded = {"edge_lb_r": prov.recorded_edge_lb_r, "mean_r": prov.recorded_mean_r, "n_trades": prov.recorded_n_trades}
    if require_same_data and df is not None and data_fingerprint(df) != prov.data_fp:
        return ReproductionResult(False, "data fingerprint differs from the one the champion was found on", recorded, rerun, tol)
    if prov.recorded_edge_lb_r is None and prov.recorded_mean_r is None:
        return ReproductionResult(False, "no recorded edge to compare against", recorded, rerun, tol)
    for key in ("edge_lb_r", "mean_r"):
        if recorded.get(key) is not None and not _close(float(recorded[key]), rerun.get(key), abs_tol, rel_tol):
            return ReproductionResult(False, f"{key} did not reproduce: recorded {recorded[key]:+.4f}, re-run {rerun.get(key)}", recorded, rerun, tol)
    rn, n0 = rerun.get("n_trades"), recorded.get("n_trades")
    if n0 is not None and rn is not None and abs(int(rn) - int(n0)) > max(2, trade_tol * max(int(n0), 1)):
        return ReproductionResult(False, f"trade count did not reproduce: recorded {n0}, re-run {rn}", recorded, rerun, tol)
    return ReproductionResult(True, "re-run reproduces the recorded edge within tolerance", recorded, rerun, tol)
