"""
Implied Volatility Surface -- pulls real (or synthetic demo) options
quotes across a range of strikes and expirations, inverts each one's
market price into an implied volatility using this app's own from-scratch
Black-Scholes solver (app.quant_lab.options_pricing.implied_volatility),
and renders the result as an interactive 3D surface (strike x expiry x
implied vol) -- the standard way options desks visualize "the market's
own view of future volatility," including the skew/smile and term
structure a flat single-number volatility assumption completely misses.

Real options-chain data requires Alpaca's options market data (a paid
add-on beyond the free stock-data tier most of this app's other Alpaca
usage relies on) -- `fetch_option_chain()` tries it and raises a clear
error naming that requirement if it fails, matching
app.data.alpaca_source's own "clear error, not a stack trace" pattern for
optional/entitlement-gated integrations. `synthetic_demo_chain()` is a
NO-CREDENTIALS-NEEDED fallback that generates a chain with a realistic
skew (higher implied vol for OTM puts -- the well-known "volatility
smirk") and term structure, so the surface itself can be built and
inspected without a live options subscription.

The 3D surface itself is rendered as a standalone HTML file using
Plotly.js loaded from its public CDN -- no new Python plotting dependency
(matplotlib, plotly's Python package, etc.) is added; this module only
ever writes a JSON data payload into a small HTML/JS template.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.quant_lab.options_pricing import (
    OptionsPricingError,
    black_scholes_price,
    implied_volatility,
)


class VolSurfaceError(Exception):
    """Raised when a chain can't be fetched, or too few quotes survive
    implied-vol inversion to build a surface."""


@dataclass
class OptionQuote:
    strike: float
    expiry_years: float      # time to expiry in YEARS (fractional), not a calendar date
    option_type: str          # "call" | "put"
    market_price: float
    underlying_price: float


def synthetic_demo_chain(
    spot: float = 100.0,
    strikes: list[float] | None = None,
    expiries_years: list[float] | None = None,
    base_vol: float = 0.22,
    skew: float = 0.15,
    term_slope: float = 0.05,
    r: float = 0.04,
    option_type: str = "put",
) -> list[OptionQuote]:
    """Generates a demo options chain with a realistic volatility SKEW
    (OTM puts / low strikes priced at higher implied vol than ATM -- the
    market's well-documented 'volatility smirk', modeled here as a simple
    linear function of moneyness for illustration) and TERM STRUCTURE
    (vol rising modestly with time to expiry via `term_slope`), then
    prices each option at that assumed volatility via Black-Scholes so
    the resulting chain's market_price, once inverted, reproduces the
    intended surface -- letting the whole pipeline (fetch/generate ->
    invert -> visualize) be exercised with no live options subscription.
    """
    strikes = strikes or [spot * m for m in (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)]
    expiries_years = expiries_years or [30 / 365, 60 / 365, 90 / 365, 180 / 365, 365 / 365]

    quotes = []
    for T in expiries_years:
        for K in strikes:
            moneyness = (K - spot) / spot
            # Skew: lower strikes (moneyness < 0, OTM puts / ITM calls) get
            # higher vol -- the classic negative skew seen in equity index
            # options, scaled by `skew`.
            sigma = base_vol - skew * moneyness + term_slope * T
            sigma = max(sigma, 0.02)
            price = black_scholes_price(spot, K, T, r, sigma, option_type)
            quotes.append(OptionQuote(strike=K, expiry_years=T, option_type=option_type,
                                       market_price=price, underlying_price=spot))
    return quotes


def fetch_option_chain(
    api_key: str, secret_key: str, underlying_symbol: str, option_type: str = "put",
) -> list[OptionQuote]:
    """Fetches a real options chain snapshot from Alpaca (requires an
    options-data-entitled account). Deferred import so the rest of this
    app works without alpaca-py installed or without options entitlement,
    matching app.data.alpaca_source's own pattern."""
    try:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import OptionChainRequest, StockLatestTradeRequest
    except ImportError as exc:
        raise VolSurfaceError(
            "The 'alpaca-py' package isn't installed, or lacks options data support. "
            "Run `pip install -U alpaca-py`, and confirm your Alpaca account has options market "
            "data enabled, to fetch a real chain -- or use synthetic_demo_chain() for a demo surface."
        ) from exc

    stock_client = StockHistoricalDataClient(api_key, secret_key)
    try:
        spot_trade = stock_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=underlying_symbol))
        spot = float(spot_trade[underlying_symbol].price)
    except Exception as exc:  # noqa: BLE001
        raise VolSurfaceError(f"Could not fetch the underlying's spot price for '{underlying_symbol}': {exc}") from exc

    option_client = OptionHistoricalDataClient(api_key, secret_key)
    try:
        chain = option_client.get_option_chain(OptionChainRequest(underlying_symbol=underlying_symbol))
    except Exception as exc:  # noqa: BLE001
        raise VolSurfaceError(f"Alpaca options chain request failed for '{underlying_symbol}': {exc}") from exc

    quotes: list[OptionQuote] = []
    occ_re = re.compile(r"(\d{6})([CP])(\d{8})$")
    for symbol, snapshot in chain.items():
        try:
            match = occ_re.search(symbol)
            if not match:
                continue
            contract_type = "call" if match.group(2) == "C" else "put"
            if contract_type != option_type:
                continue
            strike = float(snapshot.strike_price) if hasattr(snapshot, "strike_price") else None
            expiry = getattr(snapshot, "expiration_date", None)
            quote = getattr(snapshot, "latest_quote", None)
            if strike is None or expiry is None or quote is None:
                continue
            mid = (float(quote.bid_price) + float(quote.ask_price)) / 2.0
            if mid <= 0:
                continue
            expiry_years = max((pd.Timestamp(expiry) - pd.Timestamp.now()).days / 365.0, 1 / 365.0)
            quotes.append(OptionQuote(strike=strike, expiry_years=expiry_years, option_type=option_type,
                                       market_price=mid, underlying_price=spot))
        except (AttributeError, ValueError, TypeError):
            continue
    if not quotes:
        raise VolSurfaceError(
            f"No usable option quotes were parsed from Alpaca's chain response for '{underlying_symbol}'."
        )
    return quotes


def build_iv_surface(quotes: list[OptionQuote], r: float = 0.04, q: float = 0.0) -> pd.DataFrame:
    """Inverts every quote's market price into an implied volatility via
    Black-Scholes. A quote that fails to invert (price inconsistent with
    any volatility, e.g. below intrinsic value -- see
    app.quant_lab.options_pricing.implied_volatility's own docstring) is
    skipped rather than aborting the whole surface."""
    rows = []
    skipped = 0
    for quote in quotes:
        try:
            iv = implied_volatility(quote.market_price, quote.underlying_price, quote.strike,
                                     quote.expiry_years, r, quote.option_type, q)
        except OptionsPricingError:
            skipped += 1
            continue
        rows.append({
            "strike": quote.strike, "expiry_years": quote.expiry_years, "option_type": quote.option_type,
            "moneyness": quote.strike / quote.underlying_price, "implied_vol": iv,
        })
    if not rows:
        raise VolSurfaceError(f"Every quote failed implied-vol inversion ({skipped} skipped) -- no surface to build.")
    return pd.DataFrame(rows)


def render_surface_html(surface_df: pd.DataFrame, title: str = "Implied Volatility Surface") -> str:
    """Renders a standalone, self-contained HTML file with an interactive
    3D Plotly surface (strike x expiry x implied vol), using Plotly.js
    from its public CDN -- no Python plotting library required."""
    strikes = sorted(surface_df["strike"].unique())
    expiries = sorted(surface_df["expiry_years"].unique())
    pivot = surface_df.pivot_table(index="expiry_years", columns="strike", values="implied_vol", aggfunc="mean")
    pivot = pivot.reindex(index=expiries, columns=strikes)
    z = pivot.to_numpy()
    # Fill any missing (strike, expiry) combos by linear interpolation
    # along each row so a sparse real chain doesn't leave holes in the
    # rendered surface. Falls back to leaving NaN (Plotly renders a gap)
    # if an entire row/column is empty.
    pivot_interp = pivot.interpolate(axis=1, limit_direction="both")
    z = pivot_interp.to_numpy()

    payload = {
        "strikes": [float(x) for x in strikes],
        "expiries": [float(x) for x in expiries],
        "z": [[None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in row] for row in z],
        "title": title,
    }
    return _HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload)).replace("__TITLE_PLACEHOLDER__", title)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE_PLACEHOLDER__</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; margin: 0; background: #0b0d10; color: #e7e9ec; }
  .header { padding: 18px 28px; border-bottom: 1px solid #22262c; }
  .header h1 { margin: 0; font-size: 16px; font-weight: 700; }
  .header p { margin: 4px 0 0; font-size: 12px; color: #9aa1ac; }
  #surface { width: 100%; height: 82vh; }
</style>
</head>
<body>
<div class="header">
  <h1 id="title"></h1>
  <p>Strike (x) &middot; Time to expiry in years (y) &middot; Implied volatility (z) -- generated by T58's
  Universal Strategy Translator's sibling tool, app.quant_lab.vol_surface</p>
</div>
<div id="surface"></div>
<script>
  const data = __PAYLOAD__;
  document.getElementById("title").innerText = data.title;
  document.title = data.title;
  const trace = {
    type: "surface",
    x: data.strikes,
    y: data.expiries,
    z: data.z,
    colorscale: "Viridis",
    colorbar: { title: "IV" },
  };
  const layout = {
    paper_bgcolor: "#0b0d10",
    plot_bgcolor: "#0b0d10",
    font: { color: "#e7e9ec" },
    scene: {
      xaxis: { title: "Strike" },
      yaxis: { title: "Expiry (years)" },
      zaxis: { title: "Implied Vol", tickformat: ".0%" },
    },
    margin: { l: 0, r: 0, t: 10, b: 0 },
  };
  Plotly.newPlot("surface", [trace], layout, { responsive: true });
</script>
</body>
</html>
"""


def export_surface_html(surface_df: pd.DataFrame, path: str | Path, title: str = "Implied Volatility Surface") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = render_surface_html(surface_df, title)
    path.write_text(html, encoding="utf-8")
    return path
