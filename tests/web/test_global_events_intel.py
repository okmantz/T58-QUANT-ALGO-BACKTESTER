"""Market intelligence layer for Global Events: transparent regime rules, correlations, sessions, central banks."""
import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from app.global_events import intel


def _bars(closes, start="2026-01-01"):
    d0 = datetime.fromisoformat(start)
    return [{"d": (d0 + timedelta(days=i)).strftime("%Y-%m-%d"), "o": c, "h": c * 1.004, "l": c * 0.996, "c": c} for i, c in enumerate(closes)]


NOW = datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc)


def test_trending_range_and_highvol_labels():
    trend = _bars([100 * (1.006 ** i) for i in range(120)])
    r = intel.regime_for("ES", trend, [], NOW)
    assert r["regime"].startswith("Trending up") and r["metrics"]["efficiency_ratio_20d"] > 0.9
    rng = _bars([100 + (1 if i % 2 else -1) * (2.5 if i < 100 else 0.4) for i in range(160)])
    assert intel.regime_for("ES", rng, [], NOW)["regime"] == "Range-bound"
    random.seed(1)
    calm = [100.0]
    for i in range(240):
        calm.append(calm[-1] * (1 + random.gauss(0, 0.002)))
    wild = calm + [calm[-1] * (1 + random.gauss(0, 0.03)) for _ in range(1)]
    for _ in range(25):
        wild.append(wild[-1] * (1 + random.gauss(0, 0.03)))
    assert intel.regime_for("NQ", _bars(wild), [], NOW)["regime"] == "High-vol"


def test_event_driven_outranks_and_needs_instrument_match():
    bars = _bars([100 * (1.006 ** i) for i in range(120)])
    ev = [{"name": "CPI", "impact": 3, "instruments": ["ES"], "when": (NOW + timedelta(hours=5)).isoformat()}]
    assert intel.regime_for("ES", bars, ev, NOW)["regime"] == "Event-driven"
    assert intel.regime_for("GC", bars, ev, NOW)["regime"] != "Event-driven"
    far = [{"name": "CPI", "impact": 3, "instruments": ["ES"], "when": (NOW + timedelta(hours=40)).isoformat()}]
    assert intel.regime_for("ES", bars, far, NOW)["regime"] != "Event-driven"


def test_unknown_with_too_few_bars():
    assert intel.regime_for("ES", _bars([1, 2, 3]), [], NOW)["regime"] == "Unknown"


def test_correlation_matrix_perfect_and_inverse():
    base = [100 + 3 * math.sin(i / 3) + i * 0.05 for i in range(90)]
    inv = [200 - (b - 100) for b in base]
    daily = {"ES": _bars(base), "NQ": _bars([b * 2 for b in base]), "GC": _bars(inv)}
    c = intel.correlations(daily)
    i = {s: k for k, s in enumerate(c["symbols"])}
    assert c["matrix"][i["ES"]][i["NQ"]] > 0.99
    assert c["matrix"][i["ES"]][i["GC"]] < -0.9


def test_central_banks_next_date_and_sessions():
    cbs = {b["id"]: b for b in intel.central_banks(NOW)}
    assert cbs["fed"]["next"] == "2026-12-09" and cbs["boe"]["next"] == "2026-11-05"
    s = {x["name"]: x for x in intel.sessions(datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc))}
    if "New York (cash)" in s:
        assert s["New York (cash)"]["open"] is True      # 10:00 ET
    sat = intel.sessions(datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc))
    assert all(not x["open"] for x in sat)


def test_macro_groups_and_cross_asset_tone():
    ev = [{"name": "Core CPI m/m", "impact": 3, "country": "USD", "when": NOW.isoformat(), "instruments": ["ES"]},
          {"name": "Non-Farm Employment Change", "impact": 3, "country": "USD", "when": NOW.isoformat(), "instruments": []}]
    g = {r["name"]: r["group"] for r in intel.macro_releases(ev, NOW)}
    assert g["Core CPI m/m"] == "Inflation" and g["Non-Farm Employment Change"] == "Jobs"
    up = _bars([100 * 1.004 ** i for i in range(40)])
    dn = _bars([100 * 0.996 ** i for i in range(40)])
    ca = intel.cross_asset({"ES": up, "NQ": up, "RTY": up, "VIX": dn, "DXY": dn})
    assert ca["tone"] == "Risk-on"


def test_build_has_every_section_and_rules():
    out = intel.build({}, [], [], [], NOW)
    for k in ("sessions", "central_banks", "macro", "futures", "vol", "correlations", "cross_asset", "regimes", "geopolitics", "rules"):
        assert k in out
    assert len(out["rules"]["text"]) == 5


def test_stooq_daily_parser_skips_nd():
    from app.global_events import sources
    txt = "Date,Open,High,Low,Close,Volume\n2026-11-03,1,2,0.5,1.5,10\n2026-11-04,N/D,N/D,N/D,N/D,N/D\n"
    assert sources.parse_stooq_daily(txt) == [{"d": "2026-11-03", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5}]
