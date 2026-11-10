"""T58 market data pack: one canonical, versioned ES / NQ / GC (plus MES / MNQ / MGC) dataset with a FROZEN sealed window.

Why: when every user (and every run) carves its own "last 20%" holdout from whatever file it loaded, the holdout drifts as data is
appended and nobody can say which bars were ever used to choose a strategy. The pack fixes the boundary on a calendar date
(``SEALED_FROM``) and records a content hash of the sealed window in ``data/data_pack.json``. Anything that finds a different hash
for the sealed window knows the data changed under it and must treat prior sealed-slice evidence as void.

* Dev window    : everything before ``SEALED_FROM`` (>= 5 years, so the READY data gate can pass).
* Sealed window : ``SEALED_FROM`` onward. Read it through ``app.discovery_engine.holdout.SealedLedger`` (each read is a booked look).
* MES / MNQ / MGC : the same price series as ES / NQ / GC with a tenth of the dollars per point -- one canonical history, two contract specs.
* Frozen        : ``python scripts/freeze_data_pack.py`` writes the manifest. ``verify()`` compares the files on disk against it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

SEALED_FROM = "2025-09-01"          # calendar boundary; never "last N%" -- it must not move when new bars are appended
PACK_VERSION = 1
MANIFEST_NAME = "data_pack.json"

# symbol -> (source file under data/raw, base symbol, $ per index point)
PACK_SYMBOLS = {
    "ES": ("ES1!/futures_ES.F_1m.parquet", "ES", 50.0),
    "NQ": ("NQ1!/futures_NQ.F_1m.parquet", "NQ", 20.0),
    "MES": ("ES1!/futures_ES.F_1m.parquet", "ES", 5.0),
    "MNQ": ("NQ1!/futures_NQ.F_1m.parquet", "NQ", 2.0),
    "GC": ("GC1!/futures_GC.F_1m.parquet", "GC", 100.0),
    "MGC": ("GC1!/futures_GC.F_1m.parquet", "GC", 10.0),
}


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def manifest_path(root: Path | None = None) -> Path:
    return (root or _root()) / "data" / MANIFEST_NAME


def _default_loader(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    ts = "ts" if "ts" in df.columns else ("timestamp" if "timestamp" in df.columns else "datetime")
    out = df.rename(columns={ts: "timestamp"})[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out.sort_values("timestamp").reset_index(drop=True)


def split(df: pd.DataFrame, sealed_from: str = SEALED_FROM) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(dev, sealed) split on the frozen calendar boundary."""
    cut = pd.Timestamp(sealed_from, tz="UTC")
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return df[ts < cut].reset_index(drop=True), df[ts >= cut].reset_index(drop=True)


def content_hash(df: pd.DataFrame) -> str:
    """sha256 over every bar of the slice (timestamp + OHLCV). Full, not sampled: the sealed window is small and must be exact."""
    if df is None or not len(df):
        return "empty"
    h = hashlib.sha256()
    ts = pd.to_datetime(df["timestamp"], utc=True).astype("int64").to_numpy()
    h.update(ts.tobytes())
    for c in ("open", "high", "low", "close", "volume"):
        h.update(df[c].astype("float64").to_numpy().tobytes())
    return h.hexdigest()


def describe(df: pd.DataFrame) -> dict:
    if df is None or not len(df):
        return {"bars": 0, "start": None, "end": None, "sha256": "empty"}
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return {"bars": int(len(df)), "start": ts.iloc[0].isoformat(), "end": ts.iloc[-1].isoformat(), "sha256": content_hash(df)}


def build_manifest(root: Path | None = None, loader: Callable[[Path], pd.DataFrame] = _default_loader, sealed_from: str = SEALED_FROM) -> dict:
    root = root or _root()
    files: dict[str, dict] = {}
    cache: dict[str, pd.DataFrame] = {}
    for sym, (rel, base, point_value) in PACK_SYMBOLS.items():
        p = root / "data" / "raw" / rel
        if not p.exists():
            files[sym] = {"available": False, "file": rel, "reason": "source file not found"}
            continue
        if rel not in cache:
            cache[rel] = loader(p)
        dev, sealed = split(cache[rel], sealed_from)
        files[sym] = {"available": True, "file": rel, "base": base, "dollars_per_point": point_value,
                      "dev": describe(dev), "sealed": describe(sealed)}
    return {"schema": 1, "pack_version": PACK_VERSION, "sealed_from": sealed_from, "symbols": files}


def freeze(root: Path | None = None, loader=_default_loader) -> Path:
    m = build_manifest(root, loader)
    out = manifest_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1), encoding="utf-8")
    return out


def load_manifest(root: Path | None = None) -> Optional[dict]:
    p = manifest_path(root)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


def verify(root: Path | None = None, loader=_default_loader) -> dict:
    """Compare the files on disk with the frozen manifest. Returns {ok, symbols:{SYM:{status, detail}}}.
    status: ok | sealed_changed | dev_extended | missing | no_manifest. A changed SEALED hash voids prior sealed-slice evidence;
    new bars appended AFTER the boundary are the one thing that changes it, which is why the pack is versioned and re-frozen on purpose."""
    frozen = load_manifest(root)
    if frozen is None:
        return {"ok": False, "status": "no_manifest", "symbols": {}, "detail": "Run python scripts/freeze_data_pack.py to freeze the pack."}
    now = build_manifest(root, loader, frozen.get("sealed_from", SEALED_FROM))
    res, ok = {}, True
    for sym, f in frozen["symbols"].items():
        cur = now["symbols"].get(sym, {})
        if not f.get("available"):
            res[sym] = {"status": "missing", "detail": "was not available when frozen"}
            continue
        if not cur.get("available"):
            res[sym] = {"status": "missing", "detail": "source file no longer found"}
            ok = False
        elif cur["sealed"]["sha256"] != f["sealed"]["sha256"]:
            res[sym] = {"status": "sealed_changed", "detail": "The sealed window's bars changed since freeze; earlier sealed-slice results no longer apply."}
            ok = False
        elif cur["dev"]["sha256"] != f["dev"]["sha256"]:
            res[sym] = {"status": "dev_extended", "detail": "Development bars changed; sealed window is intact."}
        else:
            res[sym] = {"status": "ok", "detail": "matches the frozen manifest"}
    return {"ok": ok, "symbols": res, "sealed_from": frozen.get("sealed_from"), "pack_version": frozen.get("pack_version")}


def load_split(symbol: str, root: Path | None = None, loader=_default_loader) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(dev, sealed) bars for ES / NQ / GC / MES / MNQ / MGC. The micros share their parent contract's price history."""
    sym = symbol.upper()
    if sym not in PACK_SYMBOLS:
        raise KeyError(f"{symbol} is not in the T58 data pack ({', '.join(PACK_SYMBOLS)})")
    p = (root or _root()) / "data" / "raw" / PACK_SYMBOLS[sym][0]
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    frozen = load_manifest(root)
    return split(loader(p), (frozen or {}).get("sealed_from", SEALED_FROM))
