"""Frame stamp + residual-roll check (2026-10-31 audit, section 6).

The raw NQ vendor file carries 14 quarterly contract splices (all upward, +76..+297 points) that land in the overnight session and
contaminate every gap / overnight feature. The importer back-adjusts them; these tests lock down that (a) a frame says whether it was
adjusted, (b) a frame that still carries splices is caught, and (c) the exact NQ file the pipeline loads has none left."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.continuous_contract import back_adjust_continuous
from app.data.frame_stamp import (
    ResidualRollError,
    assert_no_residual_rolls,
    frame_stamp,
    frame_stamp_line,
    residual_rolls,
)
from tests.backtest.test_roll_splice_detection import _minute_frame, _shift_from

REPO = Path(__file__).resolve().parents[2]
NQ_FILE = REPO / "data" / "raw" / "NQ1!" / "futures_NQ.F_1m.parquet"
ROLL_KW = {"roll_months": (3, 6, 9, 12), "roll_days": range(4, 22)}


def _raw_with_splice() -> tuple[pd.DataFrame, int]:
    df = _minute_frame("2024-12-16 12:00", 3 * 24 * 60)
    k = int(np.argmax(df["timestamp"] >= pd.Timestamp("2024-12-17 00:00", tz="UTC")))
    return _shift_from(df, k, 60.0), k


def test_raw_frame_with_a_splice_is_stamped_unadjusted_and_flagged():
    raw, _ = _raw_with_splice()
    st = frame_stamp(raw, "NQ test")
    assert st["adjusted"] is False and st["roll_count"] == 0
    assert (st["residual_splices"] or 0) >= 1
    line = frame_stamp_line(st)
    assert "NOT back-adjusted" in line and "WARNING" in line and st["fingerprint"] in line
    with pytest.raises(ResidualRollError):
        assert_no_residual_rolls(raw, "NQ test")


def test_back_adjusted_frame_is_stamped_adjusted_with_no_residual_rolls():
    raw, k = _raw_with_splice()
    adj, rep = back_adjust_continuous(raw, **ROLL_KW)
    adj = adj.copy()
    adj["roll_bar"] = False
    adj.loc[k, "roll_bar"] = True
    adj.attrs["roll_adjusted"] = True
    st = assert_no_residual_rolls(adj, "NQ test")
    assert st["adjusted"] is True and st["roll_count"] == 1
    assert st["residual_splices"] == 0 and st["residual_interleaves"] == 0
    assert "back-adjusted (1 roll(s) removed); 0 residual contract splices" in frame_stamp_line(st)


def test_stamp_fingerprint_is_stable_and_changes_with_the_data():
    raw, _ = _raw_with_splice()
    assert frame_stamp(raw)["fingerprint"] == frame_stamp(raw.copy())["fingerprint"]
    assert frame_stamp(raw)["fingerprint"] != frame_stamp(raw.iloc[:-500])["fingerprint"]


def test_short_or_empty_frames_are_handled():
    assert frame_stamp(None)["bars"] == 0
    assert frame_stamp(pd.DataFrame())["fingerprint"] == "empty"
    short = _minute_frame("2024-12-16 12:00", 200)
    assert residual_rolls(short) is None                          # too short to measure -> "not measurable", never a fake zero
    assert "not measurable" in frame_stamp_line(frame_stamp(short))
    assert frame_stamp_line(None) == ""


def _file_is_real_parquet(p: Path) -> bool:
    try:
        return p.is_file() and p.read_bytes()[:4] == b"PAR1"
    except OSError:
        return False


@pytest.mark.skipif(not _file_is_real_parquet(NQ_FILE), reason="NQ 1-minute parquet not present (git-lfs pointer or not downloaded)")
def test_the_exact_nq_file_the_pipeline_loads_has_no_residual_splices_after_import():
    pytest.importorskip("pyarrow")
    from app.data.importer import import_csv
    df = import_csv(NQ_FILE, historical_complete=True).dataframe
    st = assert_no_residual_rolls(df, "NQ1!")
    assert st["adjusted"] and st["roll_count"] >= 10               # the audit counted 14 quarterly splices (+ weekend-window ones)
    ts = pd.to_datetime(df["timestamp"])
    contiguous = (ts.diff().dt.total_seconds().to_numpy()[1:] <= 60)
    jump = np.abs(df["open"].to_numpy()[1:] - df["close"].to_numpy()[:-1])[contiguous]
    assert np.nanmax(jump) < 76.0                                  # the smallest of the 14 fake splices was +76 points


@pytest.mark.skipif(not _file_is_real_parquet(NQ_FILE), reason="NQ 1-minute parquet not present (git-lfs pointer or not downloaded)")
def test_the_raw_nq_file_without_import_would_be_caught():
    pytest.importorskip("pyarrow")
    raw = pd.read_parquet(NQ_FILE).rename(columns={"ts": "timestamp"})
    assert (residual_rolls(raw) or (0, 0))[0] >= 10                # the raw file carries the splices; the stamp would say so
    assert frame_stamp(raw)["adjusted"] is False
