from __future__ import annotations

import json

import pytest

from app.quant_lab.vol_surface import (
    OptionQuote,
    VolSurfaceError,
    build_iv_surface,
    export_surface_html,
    render_surface_html,
    synthetic_demo_chain,
)


def test_synthetic_demo_chain_shape():
    chain = synthetic_demo_chain(spot=100)
    assert len(chain) > 0
    assert all(isinstance(q, OptionQuote) for q in chain)
    assert all(q.market_price > 0 for q in chain)


def test_build_iv_surface_recovers_the_intended_skew():
    chain = synthetic_demo_chain(spot=100, base_vol=0.22, skew=0.15, term_slope=0.0,
                                  expiries_years=[90 / 365])
    surface = build_iv_surface(chain, r=0.04)
    row = surface.sort_values("strike")
    # Lower strikes (further OTM puts) should show HIGHER implied vol than
    # higher strikes -- the intended negative skew.
    assert row.iloc[0]["implied_vol"] > row.iloc[-1]["implied_vol"]


def test_build_iv_surface_term_structure():
    chain = synthetic_demo_chain(spot=100, base_vol=0.20, skew=0.0, term_slope=0.10,
                                  strikes=[100.0], expiries_years=[30 / 365, 365 / 365])
    surface = build_iv_surface(chain, r=0.04)
    short = surface[surface.expiry_years < 0.5].iloc[0]["implied_vol"]
    long = surface[surface.expiry_years >= 0.5].iloc[0]["implied_vol"]
    assert long > short


def test_build_iv_surface_raises_when_all_quotes_unusable():
    # market_price (1.0) is below intrinsic value (150-100=50) for a call --
    # no volatility under Black-Scholes can explain this price.
    bad_quote = OptionQuote(strike=100, expiry_years=1.0, option_type="call",
                             market_price=1.0, underlying_price=150.0)
    with pytest.raises(VolSurfaceError):
        build_iv_surface([bad_quote])


def test_render_surface_html_embeds_plotly_and_payload():
    chain = synthetic_demo_chain(spot=100)
    surface = build_iv_surface(chain)
    html = render_surface_html(surface, title="Test Surface")
    assert "plot.ly" in html or "cdn.plot.ly" in html
    assert "Test Surface" in html
    # The embedded JSON payload should be valid and contain the right shape.
    start = html.index("const data = ") + len("const data = ")
    end = html.index(";", start)
    payload = json.loads(html[start:end])
    assert "strikes" in payload and "expiries" in payload and "z" in payload
    assert len(payload["z"]) == len(payload["expiries"])
    assert len(payload["z"][0]) == len(payload["strikes"])


def test_export_surface_html_writes_a_file(tmp_path):
    chain = synthetic_demo_chain(spot=100)
    surface = build_iv_surface(chain)
    out_path = export_surface_html(surface, tmp_path / "surface.html", title="Exported")
    assert out_path.exists()
    assert "Exported" in out_path.read_text(encoding="utf-8")
