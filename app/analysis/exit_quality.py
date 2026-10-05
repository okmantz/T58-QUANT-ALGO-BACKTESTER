"""
Exit-quality analysis (Part A port #1 of the 2026-10-04 deep analysis --
ported from astra-quant-agent's scripts/evolution/exit_quality.py).

The ledger tells you "this trade made/lost $X". It cannot tell you "this
trade REACHED +$Y and gave most of it back". This module answers the second
question with per-trade MFE (max favorable excursion) evidence:

    give_back_r         = mfe_r - realized_r           -- how many R were given back
    exit_efficiency_pct = realized_pct / mfe_pct * 100 -- what fraction of the peak was captured

Three honesty disciplines, carried over from the port:

1. R multiples only ever use REAL evidence: give_back_r needs mfe_r AND
   realized_r (i.e. the trade must carry both an mfe_price AND a nonzero
   initial_risk distance from the same backtest). Never fabricate a 1R
   distance. The pct-denominated metrics (give_back_pct /
   exit_efficiency_pct) need no stop at all and are always reported
   alongside.
2. Thin samples don't get verdicts: every table carries its n; anything
   below MIN_SAMPLE is still reported but is additionally filed under
   `insufficient` so no structural conclusion rests on it.
3. Never present inference as fact: the per-exit-mechanism stats only use
   rows whose exit_cause is machine-verified (exit_reason_source ==
   "mechanism"). Rows whose cause was derived from the engine's generic
   exit_reason are labeled "inferred" and enter the overall table only.

Input rows are the engine's Trade objects (app.backtest.execution) or
plain dicts with the same keys. A sibling worker adds `mfe_price` and
`exit_cause` to the Trade object in execution.py; every read here goes
through getattr/.get with fallbacks so this module keeps working on
trades that predate those fields -- such rows simply contribute no MFE
evidence (classify_row returns None for them) or an "inferred" cause.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

#: A conclusion below this sample size is reported but filed under
#: `insufficient` -- no structural judgement is allowed to rest on it.
MIN_SAMPLE = 8

#: Machine-verified exit causes -> human labels (report and any prompt
#: consumer read this one mapping, so wording can't drift between places).
CAUSE_LABELS = {
    "hard_stop": "Hard stop",
    "stop_loss": "Hard stop",
    "trailing_stop": "Trailing stop",
    "breakeven": "Breakeven",
    "time_stop": "Time stop",
    "take_profit": "Take profit",
    "partial_exit": "Partial exit",
    "scale_out": "Scale out",
    "signal": "Signal exit",
    "end_of_data": "End of data",
    "daily_loss_limit": "Daily loss limit",
    "max_drawdown": "Max drawdown",
    "adaptive_risk": "Adaptive risk exit",
    "rescue_minimum": "Rescue minimum (1 contract)",
    "unknown": "Unknown",
}

#: The engine's generic exit_reason values (execution.py) -> best-guess
#: cause when no machine-verified exit_cause is present on the trade.
#: These rows are honestly labeled exit_reason_source="inferred".
_EXIT_REASON_FALLBACK = {
    "stop_loss": "hard_stop",
    "take_profit": "take_profit",
    "signal": "signal",
    "end_of_data": "end_of_data",
}

_SIDE_SIGNS = {"long": 1, "buy": 1, 1: 1, 1.0: 1,
               "short": -1, "sell": -1, -1: -1, -1.0: -1}


def _get(trade: Any, name: str, default: Any = None) -> Any:
    """Read a field off a Trade dataclass OR a plain dict."""
    if isinstance(trade, dict):
        return trade.get(name, default)
    return getattr(trade, name, default)


def _num(value: Any) -> Optional[float]:
    """Coerce to a finite float; bool / junk / non-finite -> None
    (float(True) == 1.0 would otherwise impersonate a price)."""
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _side_sign(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        return _SIDE_SIGNS.get(value, _SIDE_SIGNS.get(str(value or "").strip().lower()))
    except (TypeError, ValueError):
        return None


def _mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def _median(values: List[Optional[float]]) -> Optional[float]:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return round(clean[mid], 4)
    return round((clean[mid - 1] + clean[mid]) / 2.0, 4)


def classify_row(trade: Any) -> Optional[Dict[str, Any]]:
    """Fold one closed trade into an exit-quality sample (pure computation).

    Entry bar: a row only counts if it carries excursion evidence --
    `mfe_price` (favorable) or `worst_price`/`mae` (adverse). A row with
    only open/close prices can compute what it realized, not what it
    REACHED -- exactly the question this module exists to answer -- so it
    is excluded from evidence rather than silently counted.

    Returns None when the row has no usable excursion evidence.
    """
    if trade is None:
        return None
    mfe_price = _num(_get(trade, "mfe_price"))
    mae_price = _num(_get(trade, "worst_price", _get(trade, "mae_price")))
    if mfe_price is None and mae_price is None:
        return None

    entry = _num(_get(trade, "entry_price", _get(trade, "open_px")))
    exit_px = _num(_get(trade, "exit_price", _get(trade, "close_px")))
    sign = _side_sign(_get(trade, "direction", _get(trade, "side")))
    one_r = _num(_get(trade, "initial_risk"))  # |entry - stop| in price units
    if one_r is not None and one_r <= 0:
        one_r = None
    if entry is None or entry <= 0 or sign is None:
        return None

    # --- MFE side -------------------------------------------------------
    mfe_pct = mfe_r = None
    if mfe_price is not None:
        mfe_pct = round(sign * (mfe_price - entry) / entry * 100.0, 4)
        if one_r:
            mfe_r = round(sign * (mfe_price - entry) / one_r, 4)

    # --- realized side --------------------------------------------------
    realized_pct = realized_r = None
    if exit_px is not None:
        realized_pct = round(sign * (exit_px - entry) / entry * 100.0, 4)
        if one_r:
            realized_r = round(sign * (exit_px - entry) / one_r, 4)

    # --- MAE side --------------------------------------------------------
    mae_pct = mae_r = None
    if mae_price is not None:
        # worst_price: most ADVERSE extreme (low for long, high for short);
        # positive = how far it went against the position.
        mae_pct = round(-sign * (mae_price - entry) / entry * 100.0, 4)
        if one_r:
            mae_r = round(-sign * (mae_price - entry) / one_r, 4)

    # --- the two headline metrics ----------------------------------------
    give_back_pct = give_back_r = None
    if mfe_pct is not None and realized_pct is not None:
        give_back_pct = round(mfe_pct - realized_pct, 4)
    if mfe_r is not None and realized_r is not None:
        give_back_r = round(mfe_r - realized_r, 4)

    # Capture ratio: only meaningful when the trade was ever in profit
    # (MFE <= 0 makes "efficiency" a pseudo-question).
    efficiency = None
    if mfe_pct is not None and mfe_pct > 0 and realized_pct is not None:
        efficiency = round(realized_pct / mfe_pct * 100.0, 2)

    # --- exit mechanism ----------------------------------------------------
    raw_cause = _get(trade, "exit_cause")
    if raw_cause:
        exit_cause = str(raw_cause).strip().lower()
        source = "mechanism"  # machine-verified at exit time
    else:
        exit_cause = _EXIT_REASON_FALLBACK.get(
            str(_get(trade, "exit_reason") or "").strip().lower(), "unknown")
        source = "inferred"

    return {
        "inst": str(_get(trade, "instrument", _get(trade, "symbol")) or ""),
        "side": "long" if sign == 1 else "short",
        "exit_cause": exit_cause,
        "exit_reason_source": source,
        "net_pnl": _num(_get(trade, "pnl", _get(trade, "net_pnl"))),
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "realized_pct": realized_pct,
        "realized_r": realized_r,
        "give_back_pct": give_back_pct,
        "give_back_r": give_back_r,
        "exit_efficiency_pct": efficiency,
    }


def _group_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate one sample group (win rate / pnl / MFE / MAE / give-back /
    capture ratio)."""
    pnls = [s["net_pnl"] for s in samples if s["net_pnl"] is not None]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "n": len(samples),
        "wins": wins,
        "win_rate_pct": round(wins / len(pnls) * 100.0, 1) if pnls else None,
        "net_pnl": round(sum(pnls), 2) if pnls else None,
        "avg_mfe_r": _mean([s["mfe_r"] for s in samples]),
        "avg_mae_r": _mean([s["mae_r"] for s in samples]),
        "avg_give_back_r": _mean([s["give_back_r"] for s in samples]),
        "median_give_back_r": _median([s["give_back_r"] for s in samples]),
        "avg_give_back_pct": _mean([s["give_back_pct"] for s in samples]),
        "avg_efficiency_pct": _mean([s["exit_efficiency_pct"] for s in samples]),
        # r_sample: how many rows genuinely carry R-denominated evidence
        # (the honest denominator for the R metrics).
        "r_sample": sum(1 for s in samples if s["mfe_r"] is not None),
    }


def _exit_diagnosis(overall: Optional[Dict[str, Any]], min_sample: int) -> Optional[str]:
    """One-line verdict the optimizer and the report can quote, or None."""
    if not overall or (overall.get("r_sample") or 0) < min_sample:
        return None
    avg_mfe = overall.get("avg_mfe_r")
    avg_give = overall.get("avg_give_back_r")
    if avg_mfe is None or avg_give is None:
        return None
    if avg_mfe >= 1.0 and avg_give >= 0.5 * avg_mfe:
        return (f"Entries look fine (avg peak +{avg_mfe:.2f}R) but exits give back "
                f"+{avg_give:.2f}R per trade on average -- the problem is exits, "
                "not entries. Tune exit logic (trailing stops, take-profit "
                "levels, time stops) before re-tuning entries.")
    if avg_mfe < 0.5:
        return ("Average peak excursion is under +0.5R -- entries rarely get "
                "into profit at all. This is an ENTRY problem; exit tuning "
                "cannot rescue it.")
    return None


def analyze_exit_quality(
    closed_trades: Any, min_sample: int = MIN_SAMPLE
) -> Dict[str, Any]:
    """Full exit-quality analysis (pure function, zero I/O, plain dicts out).

    Structure:

    - `overall`: all evidence rows aggregated (incl. `r_sample`, the
      honest count of R-denominated rows);
    - `by_mechanism`: per-exit-cause stats, counting ONLY machine-verified
      causes (exit_reason_source == "mechanism");
    - `per_symbol`: per-instrument exit matrix (finds "entries right,
      exits broken" instruments);
    - `time_stop`: time-stop effectiveness (did timed exits cut trades
      that had been in profit?);
    - `insufficient`: thin-sample flags -- conclusions that cannot be
      asserted on the available evidence;
    - `diagnosis`: one-line entries-vs-exits verdict, or None.
    """
    rows = [t for t in (closed_trades or [])] if isinstance(closed_trades, (list, tuple)) else []
    samples = [s for s in (classify_row(t) for t in rows) if s]
    mechanism = [s for s in samples if s["exit_reason_source"] == "mechanism" and s["exit_cause"]]

    insufficient: List[str] = []
    result: Dict[str, Any] = {
        "total_rows": len(rows),
        "evidence_rows": len(samples),
        "r_rows": sum(1 for s in samples if s["mfe_r"] is not None),
        "min_sample": min_sample,
        "overall": _group_stats(samples) if samples else None,
        "by_mechanism": [],
        "per_symbol": [],
        "time_stop": None,
        "insufficient": insufficient,
        "diagnosis": None,
    }
    if not samples:
        insufficient.append(
            "No closed trade carries MFE/MAE excursion evidence: exit quality is not assessable")
        return result

    overall = result["overall"]
    result["diagnosis"] = _exit_diagnosis(overall, min_sample)
    if overall["r_sample"] < min_sample:
        insufficient.append(
            f"Only {overall['r_sample']} rows carry R-denominated evidence (< {min_sample}): "
            "R give-back/capture figures cannot be asserted -- use the pct-denominated metrics")

    # -- per-exit-mechanism table (machine-verified causes only) -----------
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for s in mechanism:
        buckets.setdefault(s["exit_cause"], []).append(s)
    by_mechanism = []
    for cause, group in buckets.items():
        stats = _group_stats(group)
        stats["exit_cause"] = cause
        stats["label"] = CAUSE_LABELS.get(cause, cause)
        by_mechanism.append(stats)
    by_mechanism.sort(key=lambda x: (x["avg_give_back_r"] if x["avg_give_back_r"] is not None else 0.0),
                      reverse=True)
    result["by_mechanism"] = by_mechanism
    if mechanism and len(mechanism) < min_sample:
        insufficient.append(
            f"Only {len(mechanism)} machine-verified exit samples (< {min_sample}): "
            "mechanism-by-mechanism comparisons cannot be asserted")
    if samples and not mechanism:
        insufficient.append(
            "No machine-verified exit causes (exit_cause is unset): the per-mechanism "
            "table is empty -- this is not 'all mechanisms are fine'")
    thin_causes = [m["label"] for m in by_mechanism if m["n"] < min_sample]
    if thin_causes:
        insufficient.append(
            f"These exit mechanisms have < {min_sample} samples (for reference only): "
            f"{', '.join(thin_causes[:8])}")

    # -- per-instrument exit matrix -----------------------------------------
    per_inst: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        if s["inst"]:
            per_inst.setdefault(s["inst"], []).append(s)
    matrix = []
    for inst, group in per_inst.items():
        stats = _group_stats(group)
        stats["inst"] = inst
        matrix.append(stats)
    matrix.sort(key=lambda x: (x["avg_give_back_r"] if x["avg_give_back_r"] is not None else 0.0),
                reverse=True)
    result["per_symbol"] = matrix
    thin = [m["inst"] for m in matrix if m["n"] < min_sample]
    if thin:
        insufficient.append(
            f"These instruments have < {min_sample} samples (for reference only): "
            f"{', '.join(thin[:8])}")

    # -- time-stop effectiveness ---------------------------------------------
    time_stop_rows = [s for s in mechanism if s["exit_cause"] == "time_stop"]
    if time_stop_rows:
        stats = _group_stats(time_stop_rows)
        stats["avg_mfe_pct"] = _mean([s["mfe_pct"] for s in time_stop_rows])
        stats["avg_mae_pct"] = _mean([s["mae_pct"] for s in time_stop_rows])
        stats["avg_realized_pct"] = _mean([s["realized_pct"] for s in time_stop_rows])
        # "Did the time stop cut winners?": positive peak but closed at a
        # loss = it was given a chance, then taken away.
        stats["cut_while_positive"] = sum(
            1 for s in time_stop_rows
            if (s["mfe_pct"] or 0) > 0 and (s["net_pnl"] or 0) < 0)
        result["time_stop"] = stats
        if stats["n"] < min_sample:
            insufficient.append(
                f"Only {stats['n']} time-stop samples (< {min_sample}): whether "
                "the time-stop horizon should change cannot be asserted")
    else:
        insufficient.append("No time stops occurred in the sample: time-stop effectiveness cannot be assessed")

    return result
