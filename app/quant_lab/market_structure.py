"""
Market Structure -- Wyckoff events (spring/upthrust/SOS/SOW + phase) and
classic HH/HL/LH/LL swing structure with BOS (break of structure) / ChoCH
(change of character) labeling.

Ported from the HyperTA project's Structures module
(HyperTA/Structures/structures.py + utils.py -- calculateWyckoff,
calculateHhLl and their internal helpers), which was real working code,
not a stub -- see that project's own analysis for the full picture of
what else it does and doesn't have implemented. Adapted here to this
app's lowercase OHLCV column convention (timestamp/open/high/low/close/
volume, same as every other module under app/) instead of HyperTA's
Date/Open/High/Low/Close/Volume, and to return plain dataclasses instead
of loose DataFrames-of-everything, so this plugs cleanly into both:

  - app.ai.market_intelligence -- Owen AI's market-bias proxy today is
    purely EMA50-vs-EMA200 (see that module's own docstring/caveat).
    summarize_market_structure() below gives it real, deterministic
    BOS/ChoCH and Wyckoff phase facts to reason from instead, without
    asking the LLM to eyeball structure from price alone.
  - the Quant Lab "Market Structure" tool (web + desktop) -- a
    standalone way to inspect any loaded dataset's structure directly.

No new dependencies: pandas + numpy only, same as every other app/
module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class MarketStructureError(Exception):
    """Raised when the input data can't be normalized to OHLC, or is too
    short for the requested swing/range window."""


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes to timestamp/open/high/low/close(/volume), sorted --
    this app's standard OHLCV shape (see e.g. app.quant_lab.sentiment_price
    or app.data.importer), not HyperTA's Date/Open/High/Low/Close."""
    out = df.copy()
    if "timestamp" not in out.columns:
        out = out.reset_index()
        for candidate in ("timestamp", "Date", "date", "Datetime", "datetime", "index"):
            if candidate in out.columns:
                if candidate != "timestamp":
                    out = out.rename(columns={candidate: "timestamp"})
                break
    if "timestamp" not in out.columns:
        raise MarketStructureError("DataFrame must contain a 'timestamp' column (or a datetime index).")
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            raise MarketStructureError(f"DataFrame must contain a '{col}' column.")
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fractal swings + HH/HL/LH/LL + BOS/ChoCH
# (port of HyperTA's _fractalSwings / _labelSwingStructure / _annotateHhLl)
# ---------------------------------------------------------------------------

def calculate_swing_points(df: pd.DataFrame, *, left: int = 5, right: int = 5) -> pd.DataFrame:
    """Classic fractal swing highs/lows: a swing high at i is the max High
    over [i-left, i+right]; a swing low is the min Low over the same
    window (and each must be the UNIQUE max/min in that window, so a flat
    top/bottom doesn't produce a run of duplicate swing points)."""
    if left < 1 or right < 1:
        raise MarketStructureError("left and right must both be >= 1.")
    ohlc = _ensure_ohlc(df)
    highs = ohlc["high"].to_numpy(dtype=float)
    lows = ohlc["low"].to_numpy(dtype=float)
    n = len(ohlc)
    rows = []
    for i in range(left, n - right):
        window_h = highs[i - left: i + right + 1]
        window_l = lows[i - left: i + right + 1]
        if highs[i] == np.nanmax(window_h) and np.sum(window_h == highs[i]) == 1:
            rows.append({"timestamp": ohlc.at[i, "timestamp"], "price": float(highs[i]), "kind": "high", "index": i})
        if lows[i] == np.nanmin(window_l) and np.sum(window_l == lows[i]) == 1:
            rows.append({"timestamp": ohlc.at[i, "timestamp"], "price": float(lows[i]), "kind": "low", "index": i})
    if not rows:
        return pd.DataFrame(columns=["timestamp", "price", "kind", "index"])
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _label_swing_structure(swings: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'structure' column: HH/HL/LH/LL relative to the previous
    same-kind swing (or a bare 'H'/'L' for the first of each kind)."""
    out = swings.copy()
    if out.empty:
        out["structure"] = pd.Series(dtype=object)
        return out
    last_high = last_low = None
    labels = []
    for _, row in out.iterrows():
        if row["kind"] == "high":
            labels.append("H" if last_high is None else ("HH" if row["price"] > last_high else "LH"))
            last_high = row["price"]
        else:
            labels.append("L" if last_low is None else ("HL" if row["price"] > last_low else "LL"))
            last_low = row["price"]
    out["structure"] = labels
    return out


def calculate_hh_ll_structure(df: pd.DataFrame, *, left: int = 5, right: int = 5) -> pd.DataFrame:
    """Higher-high / higher-low / lower-high / lower-low market structure.

    Builds on calculate_swing_points(), labels each swing HH/HL/LH/LL,
    tracks the running trend (up/down/range), and flags:
      - bos   -- break of structure IN the direction of the trend
                 (HH while already trending up, LL while already down)
      - choch -- change of character AGAINST the prior trend
                 (HH while trending down, LL while trending up)

    Returns a DataFrame: timestamp, price, kind, structure, trend, event, index.
    """
    swings = calculate_swing_points(df, left=left, right=right)
    empty_cols = ["timestamp", "price", "kind", "structure", "trend", "event", "index"]
    if swings.empty:
        return pd.DataFrame(columns=empty_cols)

    out = _label_swing_structure(swings)
    trend = "range"
    trends: list[str] = []
    events: list[str | None] = []
    for _, row in out.iterrows():
        label = str(row["structure"])
        event = None
        if label == "HH":
            if trend == "down":
                event = "choch"
            elif trend == "up":
                event = "bos"
            trend = "up"
        elif label == "LL":
            if trend == "up":
                event = "choch"
            elif trend == "down":
                event = "bos"
            trend = "down"
        elif label == "HL":
            if trend != "down":
                trend = "up"
        elif label == "LH":
            if trend != "up":
                trend = "down"
        trends.append(trend)
        events.append(event)
    out["trend"] = trends
    out["event"] = events
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Consolidation ranges + Wyckoff events
# (port of HyperTA's _detectRanges / _mergeOverlappingRanges / _detectWyckoff)
# ---------------------------------------------------------------------------

def _merge_overlapping_ranges(ranges: list[dict]) -> list[dict]:
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda r: (r["start_index"], r["end_index"]))
    merged = [dict(ranges[0])]
    for r in ranges[1:]:
        prev = merged[-1]
        if r["start_index"] <= prev["end_index"] + 2:
            prev["end_index"] = max(prev["end_index"], r["end_index"])
            prev["end_timestamp"] = max(prev["end_timestamp"], r["end_timestamp"])
            prev["top"] = max(prev["top"], r["top"])
            prev["bottom"] = min(prev["bottom"], r["bottom"])
            prev["touches_top"] = max(prev["touches_top"], r["touches_top"])
            prev["touches_bottom"] = max(prev["touches_bottom"], r["touches_bottom"])
            mid = 0.5 * (prev["top"] + prev["bottom"])
            prev["mid"] = mid
            prev["width_pct"] = (prev["top"] - prev["bottom"]) / max(abs(mid), 1e-12)
            prev["n_bars"] = int(prev["end_index"] - prev["start_index"] + 1)
        else:
            merged.append(dict(r))
    return merged


def _detect_ranges(
    ohlc: pd.DataFrame, *, window: int = 40, max_width_pct: float = 0.12,
    min_touches: int = 2, touch_tol: float = 0.008, step: int = 5,
) -> pd.DataFrame:
    """Sliding-window consolidation boxes: a window counts as a range when
    (max High - min Low)/mid <= max_width_pct AND both edges are touched
    at least min_touches times. Tries `window` and a few nearby sizes."""
    cols = ["start_timestamp", "end_timestamp", "start_index", "end_index", "top", "bottom",
            "mid", "width_pct", "touches_top", "touches_bottom", "n_bars"]
    n = len(ohlc)
    windows = sorted({int(window), max(15, int(window * 0.75)), int(window * 1.25), 30, 50})
    windows = [w for w in windows if w < n]
    if not windows:
        return pd.DataFrame(columns=cols)

    highs = ohlc["high"].to_numpy(dtype=float)
    lows = ohlc["low"].to_numpy(dtype=float)
    timestamps = ohlc["timestamp"]
    found: list[dict] = []
    for win in windows:
        for start in range(0, n - win + 1, max(1, int(step))):
            end = start + win - 1
            top = float(np.nanmax(highs[start:end + 1]))
            bottom = float(np.nanmin(lows[start:end + 1]))
            mid = 0.5 * (top + bottom)
            width_pct = (top - bottom) / max(abs(mid), 1e-12)
            if width_pct > max_width_pct or width_pct <= 0:
                continue
            tol = touch_tol * max(abs(mid), 1e-12)
            touches_top = int(np.sum(highs[start:end + 1] >= top - tol))
            touches_bottom = int(np.sum(lows[start:end + 1] <= bottom + tol))
            if touches_top < min_touches or touches_bottom < min_touches:
                continue
            found.append({
                "start_timestamp": timestamps.iloc[start], "end_timestamp": timestamps.iloc[end],
                "start_index": int(start), "end_index": int(end), "top": top, "bottom": bottom,
                "mid": mid, "width_pct": float(width_pct), "touches_top": touches_top,
                "touches_bottom": touches_bottom, "n_bars": int(win),
            })
    merged = _merge_overlapping_ranges(found)
    if not merged:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(merged).sort_values(["n_bars", "width_pct"], ascending=[False, True]).reset_index(drop=True)


def calculate_wyckoff_events(
    df: pd.DataFrame, *, window: int = 40, max_width_pct: float = 0.15,
    lookforward: int = 30, volume_mult: float = 1.2,
) -> pd.DataFrame:
    """Wyckoff-style events around consolidation ranges: spring / upthrust
    (false breaks) and sos / sow (sign of strength / weakness -- decisive
    closes outside the range, optionally volume-confirmed), plus a rough
    phase label (accumulation / distribution / markup / markdown / ranging).

    Returns a DataFrame: timestamp, event, phase, price, range_top,
    range_bottom, range_start, range_end, index.
    """
    ohlc = _ensure_ohlc(df)
    cols = ["timestamp", "event", "phase", "price", "range_top", "range_bottom",
            "range_start", "range_end", "index"]
    ranges = _detect_ranges(ohlc, window=window, max_width_pct=max_width_pct, min_touches=2, step=5)
    if ranges.empty:
        return pd.DataFrame(columns=cols)

    highs = ohlc["high"].to_numpy(dtype=float)
    lows = ohlc["low"].to_numpy(dtype=float)
    closes = ohlc["close"].to_numpy(dtype=float)
    timestamps = ohlc["timestamp"]
    n = len(ohlc)
    has_vol = "volume" in ohlc.columns
    vol = ohlc["volume"].astype(float).to_numpy() if has_vol else None
    vol_ma = pd.Series(vol).rolling(20, min_periods=5).mean().to_numpy() if has_vol else None

    rows = []
    for _, rg in ranges.iterrows():
        top, bottom = float(rg["top"]), float(rg["bottom"])
        rs, re = int(rg["start_index"]), int(rg["end_index"])
        i0, i1 = rs, min(n - 1, re + int(lookforward))

        spring_i = upthrust_i = sos_i = sow_i = None
        for i in range(i0, i1 + 1):
            if spring_i is None and lows[i] < bottom and closes[i] > bottom and i >= rs:
                spring_i = i
            if upthrust_i is None and highs[i] > top and closes[i] < top and i >= rs:
                upthrust_i = i
            vol_ok = True
            if has_vol and vol_ma is not None and not np.isnan(vol_ma[i]):
                vol_ok = vol[i] >= volume_mult * vol_ma[i]
            if sos_i is None and closes[i] > top and vol_ok and i >= re - 2:
                sos_i = i
            if sow_i is None and closes[i] < bottom and vol_ok and i >= re - 2:
                sow_i = i

        after = closes[min(n - 1, re + 1): min(n, re + lookforward + 1)]
        if len(after) == 0:
            phase = "ranging"
        else:
            last = float(after[-1])
            if last > top:
                phase = "markup"
            elif last < bottom:
                phase = "markdown"
            else:
                phase = "accumulation" if spring_i is not None else ("distribution" if upthrust_i is not None else "ranging")

        def _add(event, idx):
            if idx is None:
                return
            rows.append({
                "timestamp": timestamps.iloc[idx], "event": event, "phase": phase, "price": float(closes[idx]),
                "range_top": top, "range_bottom": bottom, "range_start": rg["start_timestamp"],
                "range_end": rg["end_timestamp"], "index": int(idx),
            })

        _add("spring", spring_i)
        _add("upthrust", upthrust_i)
        _add("sos", sos_i)
        _add("sow", sow_i)
        if spring_i is None and upthrust_i is None and sos_i is None and sow_i is None:
            rows.append({
                "timestamp": timestamps.iloc[re], "event": "range", "phase": phase, "price": float(closes[re]),
                "range_top": top, "range_bottom": bottom, "range_start": rg["start_timestamp"],
                "range_end": rg["end_timestamp"], "index": int(re),
            })

    if not rows:
        return pd.DataFrame(columns=cols)
    return (
        pd.DataFrame(rows).drop_duplicates(subset=["timestamp", "event"], keep="first")
        .sort_values("timestamp").reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# One-shot summary -- what app.ai.market_intelligence / the Quant Lab tool
# actually consume, rather than the raw per-event DataFrames above.
# ---------------------------------------------------------------------------

@dataclass
class MarketStructureSummary:
    latest_trend: str                  # "up" | "down" | "range" | "unknown" (no swings found yet)
    latest_structure_event: str | None  # "bos" | "choch" | None (most recent swing had neither)
    n_bos: int
    n_choch: int
    latest_wyckoff_phase: str | None    # None if no consolidation range was ever detected
    latest_wyckoff_event: str | None    # "spring" | "upthrust" | "sos" | "sow" | "range" | None
    bias: str                          # "bullish" | "bearish" | "neutral" -- see render_summary for how
    warnings: list = field(default_factory=list)

    def render_summary(self) -> str:
        lines = [
            f"Trend (fractal swing structure): {self.latest_trend.upper()}",
            f"Most recent structure event: {self.latest_structure_event or 'none yet'}"
            + (f" (BOS so far: {self.n_bos}, ChoCH so far: {self.n_choch})" if (self.n_bos or self.n_choch) else ""),
        ]
        if self.latest_wyckoff_phase:
            lines.append(f"Wyckoff phase: {self.latest_wyckoff_phase.upper()}"
                         + (f" (latest event: {self.latest_wyckoff_event})" if self.latest_wyckoff_event else ""))
        else:
            lines.append("Wyckoff phase: no consolidation range detected in this window.")
        lines.append(f"Structure bias: {self.bias.upper()}")
        for w in self.warnings:
            lines.append(f"  note: {w}")
        return "\n".join(lines)


def summarize_market_structure(
    df: pd.DataFrame, *, swing_left: int = 5, swing_right: int = 5,
    wyckoff_window: int = 40, wyckoff_max_width_pct: float = 0.15,
    wyckoff_lookforward: int = 30, wyckoff_volume_mult: float = 1.2,
) -> MarketStructureSummary:
    """Rolls calculate_hh_ll_structure() + calculate_wyckoff_events() up
    into one answer: current trend, most recent BOS/ChoCH, most recent
    Wyckoff phase/event, and a single bullish/bearish/neutral bias --
    the deterministic facts app.ai.market_intelligence hands to Owen AI
    instead of asking the model to eyeball structure from price alone."""
    warnings: list[str] = []
    hh_ll = calculate_hh_ll_structure(df, left=swing_left, right=swing_right)
    if hh_ll.empty:
        warnings.append(
            f"Not enough bars to find a single fractal swing at left={swing_left}/right={swing_right} "
            "-- try a smaller left/right or more data."
        )
        latest_trend, latest_event, n_bos, n_choch = "unknown", None, 0, 0
    else:
        latest_trend = str(hh_ll.iloc[-1]["trend"])
        events = hh_ll["event"].dropna()
        latest_event = str(events.iloc[-1]) if not events.empty else None
        n_bos = int((hh_ll["event"] == "bos").sum())
        n_choch = int((hh_ll["event"] == "choch").sum())

    wyckoff = calculate_wyckoff_events(
        df, window=wyckoff_window, max_width_pct=wyckoff_max_width_pct,
        lookforward=wyckoff_lookforward, volume_mult=wyckoff_volume_mult,
    )
    if wyckoff.empty:
        latest_phase, latest_wyckoff_event = None, None
    else:
        latest_phase = str(wyckoff.iloc[-1]["phase"])
        latest_wyckoff_event = str(wyckoff.iloc[-1]["event"])

    # Bias: trend direction is the primary vote; a Wyckoff phase that
    # agrees reinforces it, one that actively disagrees (e.g. distribution
    # while swing-trend still reads "up") pulls it back to neutral rather
    # than silently picking one signal over the other.
    if latest_trend == "up":
        bias = "bearish" if latest_phase == "distribution" else "bullish"
    elif latest_trend == "down":
        bias = "bullish" if latest_phase == "accumulation" else "bearish"
    else:
        bias = "neutral"

    return MarketStructureSummary(
        latest_trend=latest_trend, latest_structure_event=latest_event, n_bos=n_bos, n_choch=n_choch,
        latest_wyckoff_phase=latest_phase, latest_wyckoff_event=latest_wyckoff_event, bias=bias,
        warnings=warnings,
    )
