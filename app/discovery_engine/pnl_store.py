"""Per-run daily-P&L store.

The family-wise test (Reality Check / SPA) and the whole-pool PBO need the daily P&L of EVERY candidate that cleared the
prefilter -- not the 500 trade P&Ls Evolution Lab's checkpoint keeps, and not just the survivors. This module persists them to a
compressed per-run file:

    store = PnlStore(path)
    store.add("cand-1", "mean_reversion", daily_series)     # any number of times, from any generation
    store.flush()                                           # appends a compressed part file next to ``path``
    mat, ids, families, calendar = PnlStore.load(path)      # everything, merged

Parts are append-only (``<path>.partNNNN.npz``) so a long Evolution run never rewrites what it already saved and a crash loses at
most the unflushed generation. Values are stored as float32 (daily R / P&L needs no more) and as integer day numbers.
"""
from __future__ import annotations

import glob
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd


def _day_number(ts) -> int:
    t = pd.Timestamp(ts)
    t = t.tz_localize(None) if t.tzinfo else t
    return int(t.normalize().value // 86_400_000_000_000)


class PnlStore:
    def __init__(self, path: str | Path, max_candidates: int | None = 50_000):
        self.path = Path(path)
        self.max_candidates = max_candidates
        self._pending: dict[str, tuple[str, pd.Series]] = {}
        self._lock = threading.Lock()
        self._part = len(glob.glob(str(self.path) + ".part*.npz"))
        self.n_saved = 0
        self._seen: set = set(self._load_ids()) if self._part else set()

    # ------------------------------------------------------------------ write
    def add(self, candidate_id: str, family: str, daily: pd.Series) -> bool:
        """Queue one candidate. Returns False when it is a duplicate or the store is at ``max_candidates``."""
        cid = str(candidate_id)
        with self._lock:
            if cid in self._seen or cid in self._pending:
                return False
            if self.max_candidates is not None and (len(self._seen) + len(self._pending)) >= int(self.max_candidates):
                return False
            self._pending[cid] = (str(family or "?"), daily if daily is not None else pd.Series(dtype=float))
        return True

    def add_trades(self, candidate_id: str, family: str, trades, unit: str = "r") -> bool:
        from app.discovery_engine.pool_stats import daily_series_from_trades
        return self.add(candidate_id, family, daily_series_from_trades(trades, unit=unit))

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen) + len(self._pending)

    def flush(self) -> int:
        """Write queued candidates as one compressed part; returns how many were written."""
        with self._lock:
            pending, self._pending = self._pending, {}
        if not pending:
            return 0
        ids, fams, offs, days, vals = [], [], [0], [], []
        for cid, (fam, s) in pending.items():
            s = s.dropna()
            ids.append(cid)
            fams.append(fam)
            days.append(np.fromiter((_day_number(i) for i in s.index), dtype=np.int64, count=len(s)))
            vals.append(s.to_numpy(dtype=np.float32))
            offs.append(offs[-1] + len(s))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        part = f"{self.path}.part{self._part:04d}.npz"
        tmp = part + ".tmp.npz"
        np.savez_compressed(
            tmp, ids=np.asarray(ids, dtype=object).astype(str), families=np.asarray(fams, dtype=object).astype(str),
            offsets=np.asarray(offs, dtype=np.int64),
            days=np.concatenate(days) if days else np.zeros(0, dtype=np.int64),
            values=np.concatenate(vals) if vals else np.zeros(0, dtype=np.float32),
        )
        os.replace(tmp, part)
        self._part += 1
        self.n_saved += len(ids)
        with self._lock:
            self._seen.update(ids)
        return len(ids)

    # ------------------------------------------------------------------- read
    def _load_ids(self) -> list[str]:
        out: list[str] = []
        for f in sorted(glob.glob(str(self.path) + ".part*.npz")):
            try:
                with np.load(f, allow_pickle=False) as z:
                    out.extend(z["ids"].astype(str).tolist())
            except Exception:  # noqa: BLE001
                continue
        return out

    @staticmethod
    def load_series(path: str | Path) -> tuple[dict, dict]:
        """({candidate_id: daily Series}, {candidate_id: family}) merged across every part file."""
        series: dict = {}
        fams: dict = {}
        for f in sorted(glob.glob(str(path) + ".part*.npz")):
            try:
                with np.load(f, allow_pickle=False) as z:
                    ids, fl, offs = z["ids"].astype(str), z["families"].astype(str), z["offsets"]
                    days, vals = z["days"], z["values"]
            except Exception:  # noqa: BLE001 -- a corrupt part is skipped, never fatal
                continue
            for i, cid in enumerate(ids):
                a, b = int(offs[i]), int(offs[i + 1])
                idx = pd.to_datetime(days[a:b].astype("datetime64[D]"))
                series[str(cid)] = pd.Series(vals[a:b].astype(float), index=idx)
                fams[str(cid)] = str(fl[i])
        return series, fams

    @classmethod
    def load(cls, path: str | Path):
        """(matrix T x N, ids, families {id: family}, calendar) -- see ``pool_stats.build_matrix``."""
        from app.discovery_engine.pool_stats import build_matrix
        series, fams = cls.load_series(path)
        mat, ids, cal = build_matrix(series)
        return mat, ids, fams, cal

    @staticmethod
    def delete(path: str | Path) -> int:
        n = 0
        for f in glob.glob(str(path) + ".part*.npz"):
            try:
                os.unlink(f)
                n += 1
            except OSError:
                pass
        return n


# --------------------------------------------------------------------------------------------- worker -> parent transport
def pack_series(daily: pd.Series) -> dict:
    """Compact, picklable/JSON-able form of a daily series (day numbers + float32-precision values) for worker -> parent transport."""
    s = daily.dropna()
    return {"d": [_day_number(i) for i in s.index], "v": [round(float(x), 6) for x in s.to_numpy(dtype=float)]}


def unpack_series(packed: dict | None) -> pd.Series:
    if not packed or not packed.get("d"):
        return pd.Series(dtype=float)
    idx = pd.to_datetime(np.asarray(packed["d"], dtype=np.int64).astype("datetime64[D]"))
    return pd.Series(np.asarray(packed["v"], dtype=float), index=idx)


def pack_trades(trades, unit: str = "r") -> dict | None:
    """Pack a trade list's per-day result (None when it has no measurable trades)."""
    from app.discovery_engine.pool_stats import daily_series_from_trades
    s = daily_series_from_trades(trades, unit=unit)
    return pack_series(s) if len(s) else None
