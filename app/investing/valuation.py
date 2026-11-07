"""Undervaluation research engine: SEC fundamentals + price => a transparent 0-100 score with the reasons behind it.

Honest framing: this ranks *candidates* by cheapness-with-quality (value + quality + growth + price-vs-trend). It is a research screen,
not a prediction; the Investing backtest lets the user test the same ranking historically before committing money.
"""
from __future__ import annotations


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _band(x, bad, good):
    """Linear 0..1 where ``good`` -> 1 and ``bad`` -> 0 (works for either direction)."""
    if x is None:
        return None
    return _clip((x - bad) / (good - bad)) if good != bad else None


def score(snap: dict, price: float | None, trend: dict | None = None) -> dict:
    if not snap.get("available") or not price:
        return {"score": None, "grade": "n/a", "reasons": ["No SEC annual data (or no price) for this symbol."], "components": {}}
    sh = snap.get("shares")
    eps = snap.get("eps") if snap.get("eps") is not None else ((snap["net_income"] / sh) if (snap.get("net_income") is not None and sh) else None)
    pe = price / eps if eps and eps > 0 else None
    pb = price / snap["book_value_per_share"] if snap.get("book_value_per_share") and snap["book_value_per_share"] > 0 else None
    pfcf = price / snap["fcf_per_share"] if snap.get("fcf_per_share") and snap["fcf_per_share"] > 0 else None
    earn_yield = (1 / pe) if pe else None
    comps = {
        "earnings_yield": _band(earn_yield, 0.02, 0.10),
        "price_to_book": _band(pb, 6.0, 1.0),
        "price_to_fcf": _band(pfcf, 35.0, 8.0),
        "profitability": _band(snap.get("roe"), 0.0, 0.20),
        "margins": _band(snap.get("net_margin"), 0.0, 0.18),
        "growth": _band(snap.get("revenue_cagr") if snap.get("revenue_cagr") is not None else snap.get("revenue_growth"), -0.02, 0.12),
        "balance_sheet": _band(snap.get("debt_to_equity"), 4.0, 0.8),
        "consistency": _band(snap.get("profitable_years"), 0, snap.get("years") or 1),
    }
    weights = {"earnings_yield": 2.0, "price_to_book": 1.0, "price_to_fcf": 1.5, "profitability": 1.5, "margins": 1.0, "growth": 1.5,
               "balance_sheet": 1.0, "consistency": 1.0}
    if trend:
        # buying a cheap stock still falling hard is a value trap; reward stabilising / recovering price
        comps["price_vs_trend"] = _band(trend.get("pct_from_52w_high"), 0.0, -35.0) if trend.get("pct_from_52w_high") is not None else None
        weights["price_vs_trend"] = 1.0
    have = {k: v for k, v in comps.items() if v is not None}
    if len(have) < 3:
        return {"score": None, "grade": "n/a", "reasons": ["Not enough reported data to score."], "components": comps}
    tw = sum(weights[k] for k in have)
    total = 100 * sum(weights[k] * v for k, v in have.items()) / tw
    reasons = []
    if pe: reasons.append(f"P/E {pe:.1f} (earnings yield {earn_yield*100:.1f}%)")
    if pb: reasons.append(f"Price/book {pb:.2f}")
    if pfcf: reasons.append(f"Price/free-cash-flow {pfcf:.1f}")
    if snap.get("roe") is not None: reasons.append(f"Return on equity {snap['roe']*100:.1f}%")
    if snap.get("revenue_cagr") is not None: reasons.append(f"Revenue CAGR {snap['revenue_cagr']*100:.1f}% over {snap['years']-1}y")
    if snap.get("debt_to_equity") is not None: reasons.append(f"Liabilities/equity {snap['debt_to_equity']:.2f}")
    flags = []
    if (snap.get("net_income") or 0) < 0: flags.append("Currently loss-making")
    if (snap.get("debt_to_equity") or 0) > 4: flags.append("Highly leveraged")
    grade = "Deeply undervalued" if total >= 75 else "Undervalued" if total >= 60 else "Fair" if total >= 40 else "Expensive / weak"
    return {"score": round(total, 1), "grade": grade, "pe": pe, "pb": pb, "p_fcf": pfcf, "reasons": reasons, "flags": flags,
            "components": {k: (None if v is None else round(v, 3)) for k, v in comps.items()}, "coverage": f"{len(have)}/{len(comps)} inputs"}


def trend_stats(close) -> dict:
    c = close.dropna()
    if len(c) < 30:
        return {}
    hi = float(c.tail(252).max())
    out = {"pct_from_52w_high": (float(c.iloc[-1]) / hi - 1) * 100}
    for n, k in ((21, "ret_1m"), (63, "ret_3m"), (126, "ret_6m"), (252, "ret_12m")):
        if len(c) > n:
            out[k] = (float(c.iloc[-1]) / float(c.iloc[-1 - n]) - 1) * 100
    return out
