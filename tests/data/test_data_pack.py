"""T58 data pack: calendar-frozen sealed window, content hashes, drift detection."""
import numpy as np
import pandas as pd

from app.data import data_pack as dp


def _frame(start="2024-01-01", end="2026-02-01"):
    ts = pd.date_range(start, end, freq="1D", tz="UTC")
    c = 100 + np.arange(len(ts), dtype=float)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 10.0})


def _setup(tmp_path, frame):
    for rel in {v[0] for v in dp.PACK_SYMBOLS.values()}:
        (tmp_path / "data" / "raw" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "raw" / rel).write_text("x")
    box = {"df": frame}
    return lambda p: box["df"], box


def test_split_is_on_calendar_boundary_not_a_percentage():
    dev, sealed = dp.split(_frame(), "2025-09-01")
    assert dev["timestamp"].max() < pd.Timestamp("2025-09-01", tz="UTC") <= sealed["timestamp"].min()
    # appending later bars must not move the boundary: dev stays identical
    more = pd.concat([_frame(), _frame("2026-02-02", "2026-04-01")], ignore_index=True)
    dev2, sealed2 = dp.split(more, "2025-09-01")
    assert dev.equals(dev2) and len(sealed2) > len(sealed)


def test_freeze_then_verify_ok_and_detect_sealed_change(tmp_path):
    loader, box = _setup(tmp_path, _frame())
    dp.freeze(tmp_path, loader)
    assert dp.verify(tmp_path, loader)["ok"] is True
    changed = _frame().copy()
    changed.loc[changed.index[-1], "close"] += 5          # edit a sealed bar
    box["df"] = changed
    r = dp.verify(tmp_path, loader)
    assert r["ok"] is False and r["symbols"]["ES"]["status"] == "sealed_changed"


def test_dev_change_alone_does_not_void_sealed(tmp_path):
    loader, box = _setup(tmp_path, _frame())
    dp.freeze(tmp_path, loader)
    changed = _frame().copy()
    changed.loc[changed.index[0], "close"] += 5
    box["df"] = changed
    r = dp.verify(tmp_path, loader)
    assert r["ok"] is True and r["symbols"]["NQ"]["status"] == "dev_extended"


def test_micro_contracts_share_history_with_different_point_value():
    assert dp.PACK_SYMBOLS["MES"][0] == dp.PACK_SYMBOLS["ES"][0] and dp.PACK_SYMBOLS["MNQ"][0] == dp.PACK_SYMBOLS["NQ"][0]
    assert dp.PACK_SYMBOLS["MES"][2] * 10 == dp.PACK_SYMBOLS["ES"][2]
    assert dp.PACK_SYMBOLS["MNQ"][2] * 10 == dp.PACK_SYMBOLS["NQ"][2]


def test_no_manifest_reports_clearly(tmp_path):
    r = dp.verify(tmp_path)
    assert r["ok"] is False and r["status"] == "no_manifest"
