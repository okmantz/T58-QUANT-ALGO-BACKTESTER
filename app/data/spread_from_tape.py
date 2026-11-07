"""Measure the real bid/ask spread from the user's OWN data (2026-11-06 audit, "slippage from your own tape").

The backtester charges ``RiskConfig.spread_pips`` (stock default 1.0 pip) on every fill side. That number is an assumption; the tape the
strategy was tested on usually says what the market actually charged. This module measures it and reports two numbers:

    floor   -- the typical (median) spread: the least a fill costs on an ordinary bar. Charging less than this is optimistic.
    stress  -- the 90th-percentile spread: what a fill costs on a bad bar (news, thin session). A strategy that only works at the
               floor is paying for a spread it will not always get.

Two sources, best first:
    "quoted"     the data has bid/ask (or an explicit spread) columns -> the spread is read directly, no estimation.
    "estimated"  OHLC only -> Corwin & Schultz (2012) high-low estimator: the part of a two-bar high-low range that does not grow with time
                 is the spread. It is a statistical estimate (noisy bar to bar, floored at 0), so it is reported with its sample size and is
                 only used as a cross-check on the configured spread, never silently substituted for it.

INFORMATIONAL: nothing here changes a backtest, a trade count or a verdict. It reports ``configured`` next to ``measured`` so an optimistic
spread assumption is visible before the numbers are trusted.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

_MIN_SAMPLES = 200
_K = 3.0 - 2.0 * math.sqrt(2.0)


def _col(df: pd.DataFrame, *names):
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def quoted_spread(df: pd.DataFrame) -> pd.Series | None:
    """Per-bar spread in PRICE units from bid/ask (or an explicit spread column); None when the data has neither."""
    bid, ask = _col(df, "bid", "bid_close", "bidclose"), _col(df, "ask", "ask_close", "askclose")
    if bid is not None and ask is not None:
        s = pd.to_numeric(df[ask], errors="coerce") - pd.to_numeric(df[bid], errors="coerce")
    else:
        sc = _col(df, "spread")
        if sc is None:
            return None
        s = pd.to_numeric(df[sc], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s >= 0]
    return s if len(s) else None


def corwin_schultz(df: pd.DataFrame) -> pd.Series | None:
    """Per-bar-pair spread estimate in PRICE units (rolling pairs of consecutive bars), floored at 0. None without usable OHLC."""
    h, l = _col(df, "high"), _col(df, "low")
    if h is None or l is None:
        return None
    hi = pd.to_numeric(df[h], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df[l], errors="coerce").to_numpy(float)
    if len(hi) < 3:
        return None
    ok = (hi > 0) & (lo > 0) & (hi >= lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(hi / lo)
        beta = hl[:-1] ** 2 + hl[1:] ** 2
        h2 = np.maximum(hi[:-1], hi[1:])
        l2 = np.minimum(lo[:-1], lo[1:])
        gamma = np.log(h2 / l2) ** 2
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
        spread_frac = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    valid = ok[:-1] & ok[1:] & np.isfinite(spread_frac)
    mid = (hi[:-1] + lo[:-1] + hi[1:] + lo[1:]) / 4.0
    est = np.where(valid, np.maximum(spread_frac, 0.0) * mid, np.nan)
    s = pd.Series(est, index=df.index[1:]).dropna()
    return s if len(s) else None


def measure_spread(df: pd.DataFrame, pip_size: float, configured_spread_pips: float | None = None,
                   floor_q: float = 0.5, stress_q: float = 0.9) -> dict:
    """Floor / stress spread (price units and pips) from the tape, against the configured assumption."""
    out: dict = {"source": None, "n": 0, "floor_price": None, "stress_price": None, "floor_pips": None, "stress_pips": None,
                 "configured_pips": None if configured_spread_pips is None else float(configured_spread_pips),
                 "verdict": "unmeasured", "text": ""}
    pip = float(pip_size) if pip_size and pip_size > 0 else None
    s = quoted_spread(df)
    src = "quoted"
    if s is None:
        s = corwin_schultz(df)
        src = "estimated"
    if s is None or len(s) < (20 if src == "quoted" else _MIN_SAMPLES) or pip is None:
        out["text"] = "Not enough bid/ask or high/low data to measure the real spread, so the configured spread is unchecked."
        out["source"] = src if s is not None else None
        out["n"] = 0 if s is None else int(len(s))
        return out
    floor = float(np.quantile(s, floor_q))
    stress = max(float(np.quantile(s, stress_q)), floor)
    out.update(source=src, n=int(len(s)), floor_price=floor, stress_price=stress, floor_pips=floor / pip, stress_pips=stress / pip)
    cfg = out["configured_pips"]
    how = "read from the quoted bid/ask" if src == "quoted" else "estimated from bar highs and lows (Corwin-Schultz; a noisy statistical estimate, not a quote)"
    if cfg is None:
        out["verdict"] = "measured"
        out["text"] = f"Measured spread {how}: floor {out['floor_pips']:.2f} pips (typical), stress {out['stress_pips']:.2f} pips (bad bars)."
    elif cfg + 1e-12 < out["floor_pips"]:
        out["verdict"] = "optimistic"
        out["text"] = (f"Configured spread {cfg:.2f} pips is BELOW the typical spread on your own data ({out['floor_pips']:.2f} pips, {how}). "
                       f"Results are flattered; on bad bars the cost is {out['stress_pips']:.2f} pips.")
    elif cfg + 1e-12 < out["stress_pips"]:
        out["verdict"] = "floor_only"
        out["text"] = (f"Configured spread {cfg:.2f} pips covers the typical spread ({out['floor_pips']:.2f} pips) but not bad bars "
                       f"({out['stress_pips']:.2f} pips, {how}). Re-run at the stress value to see if the edge survives it.")
    else:
        out["verdict"] = "covers_stress"
        out["text"] = f"Configured spread {cfg:.2f} pips covers even the stress spread on your data ({out['stress_pips']:.2f} pips, {how})."
    return out
