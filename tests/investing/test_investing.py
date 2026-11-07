import json

import numpy as np
import pandas as pd

from app.investing import algos, backtest, fundamentals as fnd, profile, valuation
from app.investing.paper import PaperAccount
from app.investing.sec_edgar import SecClient


def _facts():
    def fy(end, val, start=None, form="10-K", filed=None):
        d = {"end": end, "val": val, "fp": "FY", "form": form, "filed": filed or end[:4] + "-12-31"}
        if start: d["start"] = start
        return d
    yrs = range(2018, 2024)
    flow = lambda scale: [fy(f"{y}-12-31", scale * (1.1 ** (y - 2018)), f"{y}-01-01") for y in yrs]
    inst = lambda scale: [fy(f"{y}-12-31", scale * (1.08 ** (y - 2018))) for y in yrs]
    return {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": flow(1000)}}, "NetIncomeLoss": {"units": {"USD": flow(150)}},
        "Assets": {"units": {"USD": inst(2000)}}, "Liabilities": {"units": {"USD": inst(800)}},
        "StockholdersEquity": {"units": {"USD": inst(1200)}},
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": flow(100)}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": flow(1.5)}}}}}


def test_snapshot_and_score():
    s = fnd.snapshot(_facts())
    assert s["available"] and s["years"] == 6 and s["revenue_cagr"] > 0.09 and s["profitable_years"] == 6
    v = valuation.score(s, price=12.0)
    assert v["score"] is not None and v["reasons"]
    cheap, dear = valuation.score(s, 8.0)["score"], valuation.score(s, 400.0)["score"]
    assert cheap > dear


def test_sec_client_offline_search_and_filings():
    pages = {"company_tickers": {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}},
             "submissions": {"filings": {"recent": {"form": ["10-K", "4"], "filingDate": ["2024-11-01", "2024-11-02"], "reportDate": ["2024-09-28", ""],
                                                    "accessionNumber": ["0000320193-24-000123", "x"], "primaryDocument": ["a.htm", "b.xml"], "primaryDocDescription": ["10-K", "4"]}}}}
    def fetch(url):
        return json.dumps(pages["company_tickers"] if "company_tickers" in url else pages["submissions"]).encode()
    c = SecClient(fetch=fetch, cache=None)
    assert c.search("appl")[0]["ticker"] == "AAPL" and c.cik_for("aapl") == 320193
    f = c.recent_filings(320193)
    assert len(f) == 1 and f[0]["form"] == "10-K" and "000032019324000123" in f[0]["url"]


def test_profile_maps_to_config():
    agg = profile.build_profile({"risk": "buy", "horizon": "gt10", "goal": "max", "style": "auto", "max_loss": "35", "universe": ["stocks", "crypto"]})
    con = profile.build_profile({"risk": "sell", "horizon": "lt1", "goal": "preserve", "style": "manual", "max_loss": "5", "universe": ["etf"]})
    assert agg.risk_score > con.risk_score and agg.label == "Aggressive" and con.label == "Conservative"
    a, c = profile.algo_config(agg), profile.algo_config(con)
    assert a["max_weight"] > c["max_weight"] and a["cash_floor"] < c["cash_floor"] and c["crypto_cap"] == 0 < a["crypto_cap"]
    assert abs(a["value_weight"] + a["momentum_weight"] + a["quality_weight"] - 1) < 1e-9


def _prices(n=700, k=8, seed=1):
    rng = np.random.default_rng(seed)
    drift = np.linspace(-0.0002, 0.0008, k)
    r = rng.normal(drift, 0.01, size=(n, k))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=pd.bdate_range("2021-01-01", periods=n), columns=[f"S{i}" for i in range(k)])


def test_algos_and_backtest():
    p = _prices()
    cfg = profile.algo_config(profile.build_profile({"risk": "hold", "horizon": "y3_10", "goal": "growth", "max_loss": "20"}))
    for a in ("momentum", "mean_reversion", "low_vol_trend", "t58_composite"):
        w = algos.target_weights(a, p, cfg)
        assert sum(w.values()) <= 1 - cfg["cash_floor"] + 1e-6 and all(v <= cfg["max_weight"] + 1e-6 for v in w.values())
    res = backtest.backtest(p, "momentum", cfg)
    assert res["trades"] > 0 and res["equity"] and "benchmark_total_return_pct" in res


def test_paper_account(tmp_path):
    a = PaperAccount(tmp_path / "p.json")
    a.reset(1000)
    a.order("AAPL", "buy", 5, 100)
    st = a.state({"AAPL": 110})
    assert round(st["equity"], 2) == 1050 and st["positions"][0]["pnl"] == 50
    try:
        a.order("AAPL", "buy", 100, 100); assert False
    except ValueError:
        pass
    placed = a.rebalance({"MSFT": 0.5}, {"AAPL": 110, "MSFT": 200})
    s2 = a.state({"AAPL": 110, "MSFT": 200})
    assert placed and {p["symbol"] for p in s2["positions"]} == {"MSFT"}
