"""Research service: ties SEC EDGAR + prices + valuation together for search, asset pages and the undervaluation screen.
Also exposes ``sec_value_factor`` so the discovery/undervaluation engine and Global Events can consume the same SEC data."""
from __future__ import annotations

from app.investing import fundamentals as fnd, market_data as md, valuation as val
from app.investing.sec_edgar import SecClient, SecError

CRYPTO = {"BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "LTC-USD", "AVAX-USD", "LINK-USD", "DOT-USD"}
INDICES = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite", "^DJI": "Dow Jones", "^RUT": "Russell 2000", "^VIX": "VIX"}
ETFS = {"SPY": "S&P 500 ETF", "QQQ": "Nasdaq-100 ETF", "VTI": "Total US Market ETF", "IWM": "Russell 2000 ETF", "TLT": "20+yr Treasuries",
        "GLD": "Gold ETF", "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care", "VXUS": "International"}
DEFAULT_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM", "V", "UNH", "XOM", "JNJ", "PG", "HD", "COST", "KO", "PFE", "INTC", "CVX", "WMT"]


def universe(kinds) -> list:
    u = []
    if "stocks" in kinds: u += DEFAULT_STOCKS
    if "etf" in kinds: u += list(ETFS)
    if "crypto" in kinds: u += ["BTC-USD", "ETH-USD", "SOL-USD"]
    return u


def search(q: str, sec: SecClient | None = None, limit: int = 12) -> list:
    q = (q or "").strip()
    out, up = [], q.upper()
    for sym, name in {**INDICES, **ETFS}.items():
        if up in sym or q.lower() in name.lower():
            out.append({"symbol": sym, "name": name, "kind": "index" if sym.startswith("^") else "etf"})
    for sym in CRYPTO:
        if up.replace(" ", "") in sym or (q.lower() in ("crypto", "bitcoin") and sym == "BTC-USD") or (q.lower() == "ethereum" and sym == "ETH-USD"):
            out.append({"symbol": sym, "name": sym.split("-")[0] + " (crypto)", "kind": "crypto"})
    try:
        for r in (sec or SecClient()).search(q, limit):
            out.append({"symbol": r["ticker"], "name": r["name"], "kind": "stock", "cik": r["cik"]})
    except SecError:
        if q.isalpha() and len(q) <= 5:
            out.append({"symbol": up, "name": up, "kind": "stock"})
    seen, res = set(), []
    for r in out:
        if r["symbol"] not in seen:
            seen.add(r["symbol"]); res.append(r)
    return res[:limit]


def asset(symbol: str, sec: SecClient | None = None, price_provider=None) -> dict:
    symbol = symbol.upper()
    q = md.quote(symbol, price_provider)
    out = {"symbol": symbol, "quote": q, "kind": "index" if symbol.startswith("^") else "crypto" if symbol in CRYPTO else "stock",
           "fundamentals": None, "valuation": None, "filings": [], "sec_error": None}
    if out["kind"] == "stock":
        try:
            s = sec or SecClient()
            cik = s.cik_for(symbol)
            if cik:
                snap = fnd.snapshot(s.company_facts(cik))
                trend = val.trend_stats(md.history(symbol, "2y", price_provider)["close"])
                out["fundamentals"] = {k: v for k, v in snap.items() if k != "history"}
                out["statements"] = snap.get("history", [])
                out["valuation"] = val.score(snap, q["price"], trend)
                out["filings"] = s.recent_filings(cik, limit=10)
            else:
                out["kind"] = "etf"
        except SecError as e:
            out["sec_error"] = str(e)
    return out


def screen(symbols, sec: SecClient | None = None, price_provider=None, top: int = 15) -> list:
    s, rows = sec or SecClient(), []
    for sym in symbols:
        try:
            a = asset(sym, s, price_provider)
        except Exception as e:  # noqa: BLE001
            rows.append({"symbol": sym, "error": str(e)}); continue
        v = a.get("valuation") or {}
        if v.get("score") is not None:
            rows.append({"symbol": sym, "price": a["quote"]["price"], "score": v["score"], "grade": v["grade"], "pe": v.get("pe"),
                         "pb": v.get("pb"), "reasons": v["reasons"], "flags": v.get("flags", []),
                         "pct_from_high": (a["quote"]["price"] / a["quote"]["high_52w"] - 1) * 100})
    return sorted([r for r in rows if "score" in r], key=lambda r: -r["score"])[:top]


def sec_value_factor(symbol: str, sec: SecClient | None = None, price_provider=None) -> float | None:
    """0-100 SEC-derived undervaluation score for one ticker, or None. Used by the discovery engine as an event/regime factor."""
    try:
        return (asset(symbol, sec, price_provider).get("valuation") or {}).get("score")
    except Exception:  # noqa: BLE001
        return None
