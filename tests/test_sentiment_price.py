from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.quant_lab import sentiment_price
from app.quant_lab.sentiment_price import (
    FinancialSentimentScorer,
    Headline,
    SentimentPriceError,
    correlate_sentiment_with_price,
    fetch_headlines,
    render_comparison_svg,
    score_headlines,
)

_SAMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0">
<channel>
<title>Google News</title>
<item>
  <title>Stock surges after strong earnings beat - Reuters</title>
  <link>https://example.com/1</link>
  <pubDate>Mon, 01 Jan 2024 09:00:00 GMT</pubDate>
  <source url="https://reuters.com">Reuters</source>
</item>
<item>
  <title>Shares plunge on fraud investigation - Bloomberg</title>
  <link>https://example.com/2</link>
  <pubDate>Tue, 02 Jan 2024 09:00:00 GMT</pubDate>
  <source url="https://bloomberg.com">Bloomberg</source>
</item>
</channel>
</rss>"""


class _FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise sentiment_price.requests.HTTPError(f"status {self.status_code}")


def test_scorer_positive_negative_neutral():
    scorer = FinancialSentimentScorer()
    assert scorer.score("Stock surges after strong earnings beat").label == "positive"
    assert scorer.score("Company shares plunge on fraud investigation").label == "negative"
    assert scorer.score("Markets flat ahead of Fed meeting").label == "neutral"


def test_negation_flips_polarity():
    scorer = FinancialSentimentScorer()
    negated = scorer.score("Shares are not profitable, warns CEO")
    plain = scorer.score("Shares are profitable")
    assert negated.score < 0
    assert plain.score > 0


def test_intensifier_strengthens_score():
    scorer = FinancialSentimentScorer()
    plain = scorer.score("Stock is higher today")
    intensified = scorer.score("Stock is sharply higher today")
    assert intensified.score > plain.score


def test_fetch_headlines_parses_rss(monkeypatch):
    def fake_get(url, timeout=10.0, headers=None):
        return _FakeResponse(_SAMPLE_RSS.encode("utf-8"))

    monkeypatch.setattr(sentiment_price.requests, "get", fake_get)
    headlines = fetch_headlines("AAPL Apple stock")
    assert len(headlines) == 2
    assert headlines[0].source == "Reuters"
    assert "surges" in headlines[0].title.lower()


def test_fetch_headlines_raises_on_empty_result(monkeypatch):
    def fake_get(url, timeout=10.0, headers=None):
        return _FakeResponse(b"<rss version='2.0'><channel></channel></rss>")

    monkeypatch.setattr(sentiment_price.requests, "get", fake_get)
    with pytest.raises(SentimentPriceError):
        fetch_headlines("NoSuchTicker")


def _synthetic_price_df(n=30, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="1D")
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"timestamp": ts, "open": price, "high": price + 0.5, "low": price - 0.5,
                          "close": price, "volume": 1000.0})


def test_score_headlines_and_correlate_end_to_end():
    ts = pd.date_range("2024-01-01", periods=30, freq="1D", tz="UTC")
    headlines = [
        Headline(title="Stock surges on strong earnings", published=ts[i], link="", source="Test")
        if i % 2 == 0 else
        Headline(title="Shares plunge on weak outlook", published=ts[i], link="", source="Test")
        for i in range(30)
    ]
    sentiment_df = score_headlines(headlines)
    price_df = _synthetic_price_df(n=30)
    result = correlate_sentiment_with_price(sentiment_df, price_df)
    assert result.n_days > 0
    assert -1.0 <= result.correlation <= 1.0
    assert "correlation" in result.render_summary().lower()


def test_correlate_raises_with_too_little_overlap():
    ts = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    headlines = [Headline(title="Stock surges", published=ts[0], link="", source="Test"),
                 Headline(title="Stock plunges", published=ts[1], link="", source="Test")]
    sentiment_df = score_headlines(headlines)
    price_df = _synthetic_price_df(n=2)
    with pytest.raises(SentimentPriceError):
        correlate_sentiment_with_price(sentiment_df, price_df)


def test_render_comparison_svg_produces_svg_markup():
    ts = pd.date_range("2024-01-01", periods=20, freq="1D", tz="UTC")
    headlines = [Headline(title="Stock surges" if i % 2 == 0 else "Stock plunges",
                           published=ts[i], link="", source="Test") for i in range(20)]
    sentiment_df = score_headlines(headlines)
    price_df = _synthetic_price_df(n=20)
    result = correlate_sentiment_with_price(sentiment_df, price_df)
    svg = render_comparison_svg(result)
    assert svg.strip().startswith("<svg")
