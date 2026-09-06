"""
Black-Scholes Options Pricing Calculator -- built from scratch: the model
price, all five standard Greeks, and an implied-volatility solver, using
nothing but the standard-normal CDF/PDF derived from `math.erf` (no
scipy.stats dependency for the core math -- the normal distribution's CDF
has a closed-form relationship to the error function, so this needs no
external statistics library at all).

Model (European options, continuous dividend yield q):
    d1 = (ln(S/K) + (r - q + 0.5*sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    Call = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
    Put  = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)

Comparing to the real market: `implied_volatility()` inverts the model
(Newton-Raphson using the option's own vega, falling back to bisection
when vega is too small to trust -- deep ITM/OTM options have near-zero
vega, where Newton-Raphson can overshoot or diverge) to answer "what
volatility would Black-Scholes need to reproduce this market price
exactly", and `compare_to_market()` reports both the raw price gap (model
price vs. observed market price, using a volatility YOU supply -- e.g. a
historical volatility estimate) and, separately, the implied volatility
the market is actually pricing in. These are two different, complementary
comparisons: the first says "is my volatility assumption wrong or is the
option mispriced", the second says "what does the market itself believe
volatility is, independent of my own estimate."

Known, explicit limitation: this is the vanilla European Black-Scholes
model. It does not price American-style early exercise (a real
difference for equity puts and dividend-paying calls), and assumes
constant volatility and interest rates over the option's life -- both
standard simplifications for a "from scratch" reference implementation,
not a production options desk's model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class OptionsPricingError(Exception):
    """Raised for invalid inputs (non-positive S/K/T/sigma) or a solver
    that fails to converge."""


def _validate(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0 or K <= 0:
        raise OptionsPricingError("Spot price and strike must both be positive.")
    if T <= 0:
        raise OptionsPricingError("Time to expiry (in years) must be positive.")
    if sigma <= 0:
        raise OptionsPricingError("Volatility must be positive.")


def norm_cdf(x: float) -> float:
    """Standard normal CDF, via the exact closed-form relationship to the
    error function -- no scipy.stats.norm needed."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0,
) -> float:
    """S: spot, K: strike, T: years to expiry, r: risk-free rate (annualized,
    decimal, e.g. 0.05), sigma: annualized volatility (decimal, e.g. 0.20),
    option_type: "call" | "put", q: continuous dividend yield (decimal)."""
    _validate(S, K, T, sigma)
    option_type = option_type.lower()
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    if option_type == "put":
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)
    raise OptionsPricingError(f"option_type must be 'call' or 'put', got '{option_type}'.")


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float     # per 1.00 (100 percentage points) change in volatility; divide by 100 for "per 1 vol point"
    theta: float    # per YEAR of time decay; divide by 365 for "per calendar day"
    rho: float      # per 1.00 (100 percentage points) change in the risk-free rate; divide by 100 for "per 1%"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def black_scholes_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0,
) -> Greeks:
    _validate(S, K, T, sigma)
    option_type = option_type.lower()
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_t = math.sqrt(T)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    pdf_d1 = norm_pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t

    if option_type == "call":
        delta = disc_q * norm_cdf(d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
                 - r * K * disc_r * norm_cdf(d2)
                 + q * S * disc_q * norm_cdf(d1))
        rho = K * T * disc_r * norm_cdf(d2)
    elif option_type == "put":
        delta = disc_q * (norm_cdf(d1) - 1.0)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
                 + r * K * disc_r * norm_cdf(-d2)
                 - q * S * disc_q * norm_cdf(-d1))
        rho = -K * T * disc_r * norm_cdf(-d2)
    else:
        raise OptionsPricingError(f"option_type must be 'call' or 'put', got '{option_type}'.")

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float, option_type: str = "call", q: float = 0.0,
    initial_guess: float = 0.3, max_iterations: int = 100, tolerance: float = 1e-6,
) -> float:
    """Solves for the volatility that makes black_scholes_price(...) equal
    `market_price`, via Newton-Raphson (using the option's own vega as the
    derivative) with a bisection fallback for the well-known cases where
    Newton-Raphson misbehaves on options priced from real markets:
    near-zero vega (deep ITM/OTM), or a starting guess that overshoots
    into a negative/absurd volatility."""
    if S <= 0 or K <= 0 or T <= 0:
        raise OptionsPricingError("Spot price, strike, and time to expiry must all be positive.")
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if market_price < intrinsic - 1e-8:
        raise OptionsPricingError(
            f"Market price ({market_price:.4f}) is below intrinsic value ({intrinsic:.4f}) -- "
            "no volatility can explain this price under Black-Scholes; check the inputs."
        )

    sigma = initial_guess
    for _ in range(max_iterations):
        try:
            price = black_scholes_price(S, K, T, r, sigma, option_type, q)
            vega = black_scholes_greeks(S, K, T, r, sigma, option_type, q).vega
        except OptionsPricingError:
            break
        diff = price - market_price
        if abs(diff) < tolerance:
            return sigma
        if vega < 1e-8:
            break  # vega too small to trust a Newton step -- fall through to bisection
        sigma -= diff / vega
        if sigma <= 0 or sigma > 10:
            break  # Newton stepped somewhere absurd -- fall through to bisection

    # Bisection fallback: robust (if slower) whenever Newton-Raphson
    # doesn't converge cleanly, which happens often enough on real market
    # data (illiquid strikes, near-zero vega) to always have a real answer.
    lo, hi = 1e-4, 5.0
    price_lo = black_scholes_price(S, K, T, r, lo, option_type, q) - market_price
    price_hi = black_scholes_price(S, K, T, r, hi, option_type, q) - market_price
    if price_lo * price_hi > 0:
        raise OptionsPricingError(
            "Could not bracket a volatility that reproduces this market price in [0.01%, 500%] -- "
            "the market price may be inconsistent with the other inputs."
        )
    for _ in range(200):
        mid = (lo + hi) / 2.0
        price_mid = black_scholes_price(S, K, T, r, mid, option_type, q) - market_price
        if abs(price_mid) < tolerance:
            return mid
        if price_lo * price_mid < 0:
            hi = mid
        else:
            lo, price_lo = mid, price_mid
    return (lo + hi) / 2.0


@dataclass
class MarketComparison:
    model_price: float             # black_scholes_price() at YOUR supplied sigma
    market_price: float
    price_diff: float               # model_price - market_price
    price_diff_pct: float           # price_diff / market_price * 100
    your_sigma: float                # the volatility you supplied for model_price
    implied_vol: float               # the volatility the market price itself implies
    vol_diff: float                  # implied_vol - your_sigma

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def render_summary(self) -> str:
        return (
            f"Model price (sigma={self.your_sigma:.2%}): {self.model_price:.4f}   "
            f"Market price: {self.market_price:.4f}   "
            f"Diff: {self.price_diff:+.4f} ({self.price_diff_pct:+.2f}%)\n"
            f"Market-implied volatility: {self.implied_vol:.2%}   "
            f"vs. your volatility: {self.your_sigma:.2%}   "
            f"(implied - yours: {self.vol_diff:+.2%})"
        )


def compare_to_market(
    market_price: float, S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0,
) -> MarketComparison:
    """Compares your Black-Scholes price (at your own volatility estimate
    `sigma`) against a real observed market price, AND separately backs
    out what volatility the market itself is implying -- see module
    docstring for why these are two different, complementary numbers."""
    model_price = black_scholes_price(S, K, T, r, sigma, option_type, q)
    implied = implied_volatility(market_price, S, K, T, r, option_type, q, initial_guess=sigma)
    diff = model_price - market_price
    return MarketComparison(
        model_price=model_price, market_price=market_price, price_diff=diff,
        price_diff_pct=(diff / market_price * 100.0) if market_price else float("nan"),
        your_sigma=sigma, implied_vol=implied, vol_diff=implied - sigma,
    )


def fetch_market_option_quote(api_key: str, secret_key: str, option_symbol: str) -> float:
    """Fetches a real option's latest mid quote from Alpaca's options data
    API (requires an options-data-entitled Alpaca account), for use as
    `market_price` in compare_to_market(). Deferred import so the rest of
    this module works without alpaca-py installed, matching
    app.data.alpaca_source's own pattern. `option_symbol` is Alpaca's
    OCC-style option symbol, e.g. 'AAPL240119C00195000'."""
    try:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionLatestQuoteRequest
    except ImportError as exc:
        raise OptionsPricingError(
            "The 'alpaca-py' package isn't installed, or your installed version lacks options data "
            "support. Run `pip install -U alpaca-py`, and confirm your Alpaca account has options "
            "market data enabled, to fetch real market quotes."
        ) from exc

    client = OptionHistoricalDataClient(api_key, secret_key)
    try:
        quote = client.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=option_symbol))
        q = quote[option_symbol]
    except Exception as exc:  # noqa: BLE001
        raise OptionsPricingError(f"Alpaca options quote request failed for '{option_symbol}': {exc}") from exc
    bid, ask = float(q.bid_price), float(q.ask_price)
    if bid <= 0 or ask <= 0:
        raise OptionsPricingError(f"Alpaca returned an unusable quote for '{option_symbol}' (bid={bid}, ask={ask}).")
    return (bid + ask) / 2.0
