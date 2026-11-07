"""Turn SEC companyfacts into a clean fundamentals snapshot and a history of annual statements."""
from __future__ import annotations

# First tag that exists wins. us-gaap names, with common alternates.
TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "shares": ["WeightedAverageNumberOfDilutedSharesOutstanding", "EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
}
_FLOW = {"revenue", "net_income", "operating_income", "gross_profit", "eps_diluted", "operating_cash_flow", "capex"}


def _series(facts: dict, names: list[str], annual: bool) -> list[dict]:
    """Deduplicated (by period end) observations for the first available tag. Annual = 10-K full-year (fp == FY)."""
    for ns in ("us-gaap", "ifrs-full", "dei"):
        block = (facts.get("facts") or {}).get(ns) or {}
        for name in names:
            node = block.get(name)
            if not node:
                continue
            units = node.get("units") or {}
            rows = units.get("USD") or units.get("USD/shares") or units.get("shares") or next(iter(units.values()), [])
            by_end: dict = {}
            for r in rows:
                if annual and (r.get("fp") != "FY" or not str(r.get("form", "")).startswith(("10-K", "20-F", "40-F"))):
                    continue
                end = r.get("end")
                if not end:
                    continue
                if annual and r.get("start"):
                    # full-year duration only
                    from datetime import date
                    try:
                        d = (date.fromisoformat(end) - date.fromisoformat(r["start"])).days
                    except ValueError:
                        continue
                    if not 330 <= d <= 380:
                        continue
                prev = by_end.get(end)
                if prev is None or str(r.get("filed", "")) >= str(prev.get("filed", "")):
                    by_end[end] = r
            if by_end:
                return [by_end[k] for k in sorted(by_end)]
    return []


def annual_history(facts: dict, years: int = 8) -> list[dict]:
    """One row per fiscal year-end with the headline statement lines (revenue, earnings, assets, liabilities, equity...)."""
    table: dict = {}
    for key, names in TAGS.items():
        for r in _series(facts, names, annual=True):
            row = table.setdefault(r["end"], {"period_end": r["end"]})
            row[key] = r["val"]
            row.setdefault("filed", r.get("filed"))
    rows = [table[k] for k in sorted(table)]
    return rows[-years:]


def snapshot(facts: dict) -> dict:
    """Latest annual figures + growth + quality ratios. Missing inputs yield None (never fabricated)."""
    hist = annual_history(facts)
    if not hist:
        return {"available": False, "history": []}
    last, prev = hist[-1], (hist[-2] if len(hist) > 1 else {})

    def g(k, row=None):
        return (row or last).get(k)

    def ratio(a, b):
        return (a / b) if (a is not None and b not in (None, 0)) else None

    def growth(k):
        a, b = g(k), g(k, prev)
        return ((a - b) / abs(b)) if (a is not None and b not in (None, 0)) else None

    rev_cagr = None
    if len(hist) >= 4 and hist[0].get("revenue") and hist[-1].get("revenue") and hist[0]["revenue"] > 0 and hist[-1]["revenue"] > 0:
        n = len(hist) - 1
        rev_cagr = (hist[-1]["revenue"] / hist[0]["revenue"]) ** (1 / n) - 1
    profitable_years = sum(1 for r in hist if (r.get("net_income") or 0) > 0)
    fcf = (g("operating_cash_flow") - abs(g("capex") or 0)) if g("operating_cash_flow") is not None else None
    return {
        "available": True, "period_end": last["period_end"], "filed": last.get("filed"),
        "revenue": g("revenue"), "net_income": g("net_income"), "eps": g("eps_diluted"), "assets": g("assets"),
        "liabilities": g("liabilities"), "equity": g("equity"), "cash": g("cash"), "long_term_debt": g("long_term_debt"),
        "free_cash_flow": fcf, "shares": g("shares"),
        "revenue_growth": growth("revenue"), "earnings_growth": growth("net_income"), "revenue_cagr": rev_cagr,
        "net_margin": ratio(g("net_income"), g("revenue")), "gross_margin": ratio(g("gross_profit"), g("revenue")),
        "roe": ratio(g("net_income"), g("equity")), "roa": ratio(g("net_income"), g("assets")),
        "debt_to_equity": ratio(g("liabilities"), g("equity")), "current_ratio": ratio(g("current_assets"), g("current_liabilities")),
        "book_value_per_share": ratio(g("equity"), g("shares")), "fcf_per_share": ratio(fcf, g("shares")),
        "profitable_years": profitable_years, "years": len(hist), "history": hist,
    }
