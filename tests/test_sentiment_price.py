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


# ---------------------------------------------------------------------------
# Regression tests (2026-10-06): the failures a user actually hits with
# this tool -- price data that doesn't reach into the ~30-day headline
# window, a defined-but-blank Max headlines field, no-variance sentiment,
# and a blocked headline source. Headline fetches are stubbed throughout;
# nothing here needs the network.
# ---------------------------------------------------------------------------

def test_overlap_error_names_both_date_windows():
    ts = pd.date_range("2026-09-01", periods=10, freq="1D", tz="UTC")
    headlines = [Headline(title="Stock surges", published=ts[i], link="", source="T") for i in range(10)]
    sentiment_df = score_headlines(headlines)
    price_df = _synthetic_price_df(n=10)  # Jan 2024 -- can never overlap Sep 2026 headlines
    with pytest.raises(SentimentPriceError) as excinfo:
        correlate_sentiment_with_price(sentiment_df, price_df)
    msg = str(excinfo.value)
    assert "0 overlapping days" in msg
    assert "2026-09-01" in msg and "2024-01-01" in msg  # both windows named
    assert "30 days" in msg  # and the remedy: price data must reach the headline window


def test_zero_variance_sentiment_correlation_renders_na_not_nan():
    ts = pd.date_range("2026-09-01", periods=10, freq="1D", tz="UTC")
    headlines = [Headline(title="Markets await Fed decision", published=ts[i], link="", source="T")
                 for i in range(10)]  # no lexicon words -> sentiment is a constant 0.0
    sentiment_df = score_headlines(headlines)
    rng = np.random.default_rng(3)
    price = 100 + np.cumsum(rng.normal(0, 1, 10))
    price_df = pd.DataFrame({"timestamp": ts.tz_localize(None), "close": price})
    result = correlate_sentiment_with_price(sentiment_df, price_df)
    assert "nan" not in result.render_summary().lower()
    assert result.warnings


def test_correlate_missing_close_column_is_clean_error():
    ts = pd.date_range("2026-09-01", periods=10, freq="1D", tz="UTC")
    headlines = [Headline(title="Stock surges", published=ts[i], link="", source="T") for i in range(10)]
    sentiment_df = score_headlines(headlines)
    bad_price_df = pd.DataFrame({"timestamp": ts.tz_localize(None), "price": range(10)})
    with pytest.raises(SentimentPriceError) as excinfo:
        correlate_sentiment_with_price(sentiment_df, bad_price_df)
    assert "close" in str(excinfo.value)


def test_fetch_network_failure_message_is_actionable(monkeypatch):
    def _boom(url, timeout=10.0, headers=None):
        raise sentiment_price.requests.ConnectionError("HTTPSConnectionPool: name resolution failed")

    monkeypatch.setattr(sentiment_price.requests, "get", _boom)
    with pytest.raises(SentimentPriceError) as excinfo:
        fetch_headlines("AAPL Apple stock")
    assert "Could not reach Google News" in str(excinfo.value)


def test_fetch_http_error_message_names_the_failure(monkeypatch):
    monkeypatch.setattr(sentiment_price.requests, "get",
                        lambda url, timeout=10.0, headers=None: _FakeResponse(b"slow down", status=429))
    with pytest.raises(SentimentPriceError) as excinfo:
        fetch_headlines("AAPL Apple stock")
    assert "Google News returned an error" in str(excinfo.value)


def _recent_headlines(n=12):
    ts = pd.date_range("2026-09-20", periods=n, freq="1D", tz="UTC")
    return [
        Headline(title="Stock surges on strong earnings" if i % 2 == 0 else "Shares plunge on weak outlook",
                 published=ts[i], link="", source="Test")
        for i in range(n)
    ]


def _price_upload_bytes(start: str, n: int, seed: int = 5) -> bytes:
    ts = pd.date_range(start, periods=n, freq="1D")
    rng = np.random.default_rng(seed)
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({"timestamp": ts, "open": price, "high": price + 0.5, "low": price - 0.5,
                       "close": price, "volume": 1000.0})
    return df.to_csv(index=False).encode()


def test_route_happy_path_with_price_upload(monkeypatch):
    import io

    from app.web.server import app
    monkeypatch.setattr(sentiment_price, "fetch_headlines", lambda query, max_results=50, **kw: _recent_headlines())
    client = app.test_client()
    r = client.post(
        "/quant-lab/sentiment-price",
        data={"query": "TEST stock", "max_results": "12",
              "price_csv": (io.BytesIO(_price_upload_bytes("2026-09-20", 12)), "prices.csv")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert 'class="card result-ok"' in body
    assert "OVERALL SCORE" in body
    assert "Same-day correlation" in body


def test_route_keeps_sentiment_result_when_price_does_not_overlap(monkeypatch):
    import io

    from app.web.server import app
    monkeypatch.setattr(sentiment_price, "fetch_headlines", lambda query, max_results=50, **kw: _recent_headlines())
    client = app.test_client()
    r = client.post(
        "/quant-lab/sentiment-price",
        data={"query": "TEST stock", "max_results": "12",
              "price_csv": (io.BytesIO(_price_upload_bytes("2020-01-01", 30)), "old_prices.csv")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert 'class="card result-ok"' in body
    assert "OVERALL SCORE" in body
    assert "Price correlation unavailable" in body
    assert "overlapping days" in body


def test_route_blank_max_results_defaults_instead_of_erroring(monkeypatch):
    from app.web.server import app
    monkeypatch.setattr(sentiment_price, "fetch_headlines", lambda query, max_results=50, **kw: _recent_headlines())
    client = app.test_client()
    r = client.post("/quant-lab/sentiment-price", data={"query": "TEST stock", "max_results": ""})
    assert r.status_code == 200
    assert 'class="card result-ok"' in r.get_data(as_text=True)


def test_route_surfaces_fetch_failure_as_clean_error(monkeypatch):
    from app.web.server import app

    def _boom(query, max_results=50, **kw):
        raise SentimentPriceError(
            "Could not reach Google News to fetch headlines for 'TEST stock'. "
            "Check your internet connection and try again."
        )

    monkeypatch.setattr(sentiment_price, "fetch_headlines", _boom)
    client = app.test_client()
    r = client.post("/quant-lab/sentiment-price", data={"query": "TEST stock", "max_results": "10"})
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert 'class="card result-error"' in body
    assert "Could not reach Google News" in body
    assert "Traceback" not in body
