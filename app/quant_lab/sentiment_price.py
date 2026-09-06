"""
Sentiment-Price Correlation Tool -- scrapes recent financial headlines for
a ticker/query (Google News' public RSS search endpoint, no API key
needed), scores each headline's sentiment with a from-scratch, finance-
tuned lexicon scorer, aggregates to a daily sentiment series, and
correlates it against that instrument's actual daily price returns.

Sentiment scoring is a compact, rule-based (VADER-style) approach rather
than a pulled-in ML model: a hand-curated lexicon of finance-relevant
positive/negative words and phrases, with negation handling ("not
profitable" flips the polarity of "profitable") and intensifier handling
("sharply higher" is stronger than "higher"). This is deliberately not a
state-of-the-art sentiment model -- it needs no downloaded model weights,
no GPU, and no internet access at inference time, and it is transparent
enough that every score it produces can be traced back to which words
triggered it, which matters for anyone using this to sanity-check whether
a "sentiment signal" is really about the news or an artifact of a model
neither the tool's author nor its user can inspect.

Correlation is computed on DAILY buckets (average sentiment of all
headlines published that day vs. that day's close-to-close return) --
matching sentiment articles to individual intraday price ticks would
overstate precision this tool doesn't actually have (RSS results don't
carry a reliable publish time down to the minute for every source).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

# A compact, hand-curated finance-oriented sentiment lexicon. Weights are
# roughly -1.0 (strongly negative) to +1.0 (strongly positive); words not
# in this lexicon score 0 (neutral) and don't affect the headline's score.
_POSITIVE_WORDS = {
    "surge": 0.8, "surges": 0.8, "surged": 0.8, "soar": 0.9, "soars": 0.9, "soared": 0.9,
    "rally": 0.7, "rallies": 0.7, "rallied": 0.7, "jump": 0.6, "jumps": 0.6, "jumped": 0.6,
    "gain": 0.5, "gains": 0.5, "gained": 0.5, "rise": 0.4, "rises": 0.4, "rose": 0.4,
    "climb": 0.4, "climbs": 0.4, "climbed": 0.4, "higher": 0.3, "up": 0.2,
    "beat": 0.6, "beats": 0.6, "beating": 0.6, "outperform": 0.6, "outperforms": 0.6,
    "upgrade": 0.6, "upgraded": 0.6, "upgrades": 0.6, "bullish": 0.7, "optimistic": 0.5,
    "profit": 0.4, "profits": 0.4, "profitable": 0.5, "record": 0.5, "strong": 0.4,
    "growth": 0.4, "grows": 0.4, "grew": 0.4, "expand": 0.3, "expands": 0.3, "expanded": 0.3,
    "boom": 0.6, "booming": 0.6, "recovery": 0.4, "recovers": 0.4, "rebound": 0.5, "rebounds": 0.5,
    "breakthrough": 0.7, "win": 0.4, "wins": 0.4, "winning": 0.4, "success": 0.5, "successful": 0.5,
    "buy": 0.3, "buys": 0.3, "buying": 0.3, "positive": 0.4, "exceed": 0.5, "exceeds": 0.5,
}
_NEGATIVE_WORDS = {
    "plunge": -0.9, "plunges": -0.9, "plunged": -0.9, "crash": -0.9, "crashes": -0.9, "crashed": -0.9,
    "tumble": -0.7, "tumbles": -0.7, "tumbled": -0.7, "slump": -0.6, "slumps": -0.6, "slumped": -0.6,
    "fall": -0.4, "falls": -0.4, "fell": -0.4, "drop": -0.5, "drops": -0.5, "dropped": -0.5,
    "decline": -0.4, "declines": -0.4, "declined": -0.4, "lower": -0.3, "down": -0.2,
    "miss": -0.6, "misses": -0.6, "missed": -0.6, "underperform": -0.6, "underperforms": -0.6,
    "downgrade": -0.6, "downgraded": -0.6, "downgrades": -0.6, "bearish": -0.7, "pessimistic": -0.5,
    "loss": -0.5, "losses": -0.5, "unprofitable": -0.5, "weak": -0.4, "weakness": -0.4,
    "shrink": -0.4, "shrinks": -0.4, "shrank": -0.4, "contraction": -0.4, "recession": -0.6,
    "layoff": -0.6, "layoffs": -0.6, "fired": -0.4, "bankruptcy": -0.9, "bankrupt": -0.9,
    "fraud": -0.9, "scandal": -0.8, "investigation": -0.5, "lawsuit": -0.5, "sued": -0.5,
    "sell": -0.3, "sells": -0.3, "selling": -0.3, "negative": -0.4, "warn": -0.4, "warns": -0.4,
    "warning": -0.4, "cut": -0.4, "cuts": -0.4, "slashed": -0.6, "risk": -0.2, "risks": -0.2,
    "concern": -0.3, "concerns": -0.3, "worried": -0.4, "worries": -0.4, "volatile": -0.2,
}
_NEGATIONS = {"not", "no", "never", "without", "n't", "hardly", "barely"}
_INTENSIFIERS = {"very": 1.4, "sharply": 1.5, "significantly": 1.3, "extremely": 1.6, "massively": 1.6, "slightly": 0.6}


@dataclass
class SentimentScore:
    text: str
    score: float           # roughly -1.0 to +1.0
    matched_words: list = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.score > 0.15:
            return "positive"
        if self.score < -0.15:
            return "negative"
        return "neutral"


class FinancialSentimentScorer:
    """A rule-based, from-scratch financial headline sentiment scorer.
    See module docstring for why this is a transparent lexicon approach
    rather than a black-box ML model."""

    _TOKEN_RE = re.compile(r"[a-z']+")

    def score(self, text: str) -> SentimentScore:
        tokens = self._TOKEN_RE.findall(text.lower())
        total = 0.0
        matched = []
        pending_negation = False
        pending_intensifier = 1.0
        for tok in tokens:
            if tok in _NEGATIONS:
                pending_negation = True
                continue
            if tok in _INTENSIFIERS:
                pending_intensifier *= _INTENSIFIERS[tok]
                continue
            base = _POSITIVE_WORDS.get(tok, _NEGATIVE_WORDS.get(tok))
            if base is None:
                continue
            # A pending negation/intensifier applies to the very next
            # sentiment word found and then resets -- this deliberately
            # does NOT use a fixed lookback window, so one negation near
            # the start of a headline can't also leak onto an unrelated
            # sentiment word several tokens later in the same sentence.
            weight = base * pending_intensifier
            if pending_negation:
                weight = -weight * 0.9   # negation flips and slightly softens
            total += weight
            matched.append(tok)
            pending_negation = False
            pending_intensifier = 1.0
        # Squash to roughly [-1, 1] regardless of headline length/word count
        # via a soft normalization, so a headline with many matched words
        # doesn't run away to an unbounded score.
        normalized = np.tanh(total)
        return SentimentScore(text=text, score=float(normalized), matched_words=matched)

    def score_many(self, texts: list[str]) -> list[SentimentScore]:
        return [self.score(t) for t in texts]


@dataclass
class Headline:
    title: str
    published: pd.Timestamp
    link: str
    source: str


class SentimentPriceError(Exception):
    """Raised when headlines can't be fetched/parsed, or there's too
    little overlapping data to compute a correlation."""


def fetch_headlines(query: str, max_results: int = 50, timeout: float = 10.0) -> list[Headline]:
    """Fetches recent headlines matching `query` from Google News' public
    RSS search endpoint -- no API key required. `query` is typically a
    ticker plus a company name for better precision, e.g. 'AAPL Apple
    stock'. Uses the stdlib's xml.etree (RSS is plain XML), so no new
    scraping dependency (BeautifulSoup, etc.) is needed."""
    url = _GOOGLE_NEWS_RSS.format(query=quote(query))
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise SentimentPriceError(f"Failed to fetch headlines for '{query}': {exc}") from exc

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        raise SentimentPriceError(f"Could not parse RSS response for '{query}': {exc}") from exc

    headlines: list[Headline] = []
    for item in root.findall(".//item")[:max_results]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = item.findtext("pubDate")
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else ""
        if not title:
            continue
        try:
            published = pd.to_datetime(pub_date, utc=True) if pub_date else pd.NaT
        except (ValueError, TypeError):
            published = pd.NaT
        headlines.append(Headline(title=title, published=published, link=link, source=source))
    if not headlines:
        raise SentimentPriceError(f"No headlines found for '{query}'.")
    return headlines


def score_headlines(headlines: list[Headline]) -> pd.DataFrame:
    scorer = FinancialSentimentScorer()
    rows = []
    for h in headlines:
        result = scorer.score(h.title)
        rows.append({
            "timestamp": h.published, "title": h.title, "source": h.source,
            "sentiment": result.score, "label": result.label,
        })
    return pd.DataFrame(rows)


_BIAS_THRESHOLD = 0.15  # same +/-0.15 cutoff SentimentScore.label already uses per-headline


@dataclass
class OverallSentiment:
    """Rolls every scored headline up into one overall score and a
    bullish/bearish/neutral market-bias verdict, so the tool's output
    ends with a single answer to "so what does the news say overall"
    instead of only a per-headline list."""
    mean_score: float
    median_score: float
    n_headlines: int
    n_positive: int
    n_negative: int
    n_neutral: int
    bias: str  # "bullish" | "bearish" | "neutral"

    def render_summary(self) -> str:
        return (
            f"OVERALL SCORE: {self.mean_score:+.3f}  (median {self.median_score:+.3f}) "
            f"across {self.n_headlines} headlines\n"
            f"Bullish: {self.n_positive}   Bearish: {self.n_negative}   Neutral: {self.n_neutral}\n"
            f"MARKET BIAS: {self.bias.upper()}"
        )


def aggregate_sentiment(sentiment_df: pd.DataFrame) -> OverallSentiment:
    """Aggregates score_headlines()'s per-headline output into one
    overall score (mean and median, since a handful of extreme headlines
    can otherwise dominate a plain mean) and a bullish/bearish/neutral
    market-bias verdict, using the SAME +/-0.15 cutoff
    SentimentScore.label already applies per-headline -- applied here to
    the aggregate instead of a single score."""
    if sentiment_df.empty:
        raise SentimentPriceError("No scored headlines to aggregate an overall score from.")
    scores = sentiment_df["sentiment"].astype(float)
    mean_score = float(scores.mean())
    n_positive = int((scores > _BIAS_THRESHOLD).sum())
    n_negative = int((scores < -_BIAS_THRESHOLD).sum())
    n_neutral = len(scores) - n_positive - n_negative
    if mean_score > _BIAS_THRESHOLD:
        bias = "bullish"
    elif mean_score < -_BIAS_THRESHOLD:
        bias = "bearish"
    else:
        bias = "neutral"
    return OverallSentiment(
        mean_score=mean_score, median_score=float(scores.median()), n_headlines=len(scores),
        n_positive=n_positive, n_negative=n_negative, n_neutral=n_neutral, bias=bias,
    )


@dataclass
class SentimentPriceCorrelation:
    merged: pd.DataFrame            # one row per day: date, avg_sentiment, headline_count, close, daily_return
    correlation: float               # Pearson correlation of daily avg sentiment vs. same-day return
    lagged_correlation: float        # sentiment(t) vs return(t+1) -- does sentiment LEAD price by a day?
    n_days: int
    warnings: list = field(default_factory=list)

    def render_summary(self) -> str:
        return (
            f"{self.n_days} overlapping trading days.\n"
            f"Same-day correlation (sentiment vs. that day's return): {self.correlation:+.3f}\n"
            f"Next-day correlation (sentiment vs. the FOLLOWING day's return): {self.lagged_correlation:+.3f}"
        )


def correlate_sentiment_with_price(sentiment_df: pd.DataFrame, price_df: pd.DataFrame) -> SentimentPriceCorrelation:
    """sentiment_df: output of score_headlines() (or any DataFrame with
    'timestamp' and 'sentiment' columns). price_df: standard OHLCV
    DataFrame. Aggregates sentiment to a daily average and correlates it
    against that day's close-to-close return, plus a next-day-lagged
    version (does today's sentiment predict TOMORROW's move, rather than
    just describing today's -- the more interesting question for a
    tradeable signal, since correlating same-day sentiment with same-day
    price partly just reflects the news reporting on a move that already
    happened)."""
    warnings: list[str] = []
    s = sentiment_df.dropna(subset=["timestamp"]).copy()
    if s.empty:
        raise SentimentPriceError("No headlines had a usable publish timestamp -- cannot bucket by day.")
    s["date"] = pd.to_datetime(s["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
    daily_sentiment = s.groupby("date").agg(avg_sentiment=("sentiment", "mean"), headline_count=("sentiment", "size"))

    p = price_df[["timestamp", "close"]].copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"])
    if p["timestamp"].dt.tz is not None:
        p["timestamp"] = p["timestamp"].dt.tz_localize(None)
    p["date"] = p["timestamp"].dt.normalize()
    daily_price = p.groupby("date")["close"].last().to_frame()
    daily_price["daily_return"] = daily_price["close"].pct_change()

    merged = daily_sentiment.join(daily_price, how="inner").dropna(subset=["daily_return"])
    if len(merged) < 5:
        raise SentimentPriceError(
            f"Only {len(merged)} overlapping days between headlines and price data -- need at least 5 "
            "to compute a meaningful correlation. Try a wider headline search window or date range."
        )

    merged = merged.reset_index().rename(columns={"index": "date"})
    correlation = float(merged["avg_sentiment"].corr(merged["daily_return"]))

    merged["next_day_return"] = merged["daily_return"].shift(-1)
    lagged = merged.dropna(subset=["next_day_return"])
    lagged_correlation = float(lagged["avg_sentiment"].corr(lagged["next_day_return"])) if len(lagged) >= 5 else float("nan")
    if len(lagged) < 5:
        warnings.append("Not enough days to compute a reliable next-day-lagged correlation.")

    return SentimentPriceCorrelation(
        merged=merged, correlation=correlation, lagged_correlation=lagged_correlation,
        n_days=len(merged), warnings=warnings,
    )


def render_comparison_svg(result: SentimentPriceCorrelation, width: int = 760, height: int = 280) -> str:
    """A normalized (min-max scaled to [0,1]) dual-series line chart --
    sentiment and price live on very different scales, so both are
    rescaled onto a shared 0-1 axis purely to compare SHAPE (does
    sentiment move with, ahead of, or against price), not absolute level.
    Reuses this app's existing hand-rolled SVG chart helper
    (app.reports.charts.svg_multi_line_chart) rather than adding a
    plotting dependency."""
    from app.reports.charts import svg_multi_line_chart

    def _minmax(values: pd.Series) -> list[float]:
        arr = values.to_numpy(dtype=float)
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        if hi - lo < 1e-12:
            return [0.5] * len(arr)
        return list((arr - lo) / (hi - lo))

    sentiment_norm = _minmax(result.merged["avg_sentiment"])
    price_norm = _minmax(result.merged["close"])
    return svg_multi_line_chart(
        series=[("Sentiment (normalized)", sentiment_norm, "#B8862F"), ("Price (normalized)", price_norm, "#111827")],
        width=width, height=height, title="Sentiment vs. Price (normalized)",
        x_label="Trading day", y_label="Normalized level",
    )
