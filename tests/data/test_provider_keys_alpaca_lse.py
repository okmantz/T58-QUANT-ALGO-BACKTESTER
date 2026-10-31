"""Saved Alpaca keys must light up the provider list; London Strategic Edge must find forex/futures and page through long ranges."""
import sys
import types

import pandas as pd
import pytest

from app.data import alpaca_credentials, api_sources
from app.data.free_sources import DataSourceError


@pytest.fixture
def alpaca_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(alpaca_credentials, "get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(alpaca_credentials, "_try_keyring", lambda: None)
    alpaca_credentials.save_credentials("AKTESTKEY1234", "secretsecret99")
    return tmp_path


def test_saved_alpaca_keys_are_ready(alpaca_saved):
    p = api_sources.PROVIDERS["alpaca"]
    assert api_sources._key_for(p) == ("AKTESTKEY1234", "secretsecret99")
    row = {r["id"]: r for r in api_sources.provider_list()}["alpaca"]
    assert row["ready"] is True and row["key_hint"].endswith("1234")


def test_alpaca_half_saved_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(alpaca_credentials, "get_app_base_dir", lambda: tmp_path)
    monkeypatch.setattr(alpaca_credentials, "_try_keyring", lambda: None)
    alpaca_credentials.save_credentials("onlykey", "")
    assert api_sources._key_for(api_sources.PROVIDERS["alpaca"]) == ""


def test_alpaca_symbols():
    assert api_sources._alpaca_symbol("btcusd") == ("BTC/USD", "Crypto")
    assert api_sources._alpaca_symbol("ETH/USD") == ("ETH/USD", "Crypto")
    assert api_sources._alpaca_symbol("aapl") == ("AAPL", "Stock")
    with pytest.raises(DataSourceError):
        api_sources._alpaca_symbol("EURUSD")


def test_alpaca_fetch_uses_saved_keys(alpaca_saved, monkeypatch):
    import app.data.alpaca_source as src
    seen = {}

    def fake(key, secret, symbol, asset, tf, start, end, *a, **k):
        seen.update(key=key, secret=secret, symbol=symbol, asset=asset)
        ts = pd.date_range("2024-01-02", periods=5, freq="5min", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10})
    monkeypatch.setattr(src, "fetch_bars", fake)
    df = api_sources.fetch("alpaca", "BTCUSD", "5m", "2024-01-01", "2024-01-03")
    assert seen == {"key": "AKTESTKEY1234", "secret": "secretsecret99", "symbol": "BTC/USD", "asset": "Crypto"} and len(df) == 5


# ------------------------------------------------------------------ London Strategic Edge
class _Err(Exception):
    def __init__(self, status, message, code="request-failed"):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


def _rows(start, n):
    ts = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return [{"timestamp": t.isoformat(), "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 3, "trades": 1} for t in ts]


def _install_fake_lse(monkeypatch, holds, total=12000):
    """A fake lse module whose catalogue holds each symbol in one market only (ambiguous without market=, like the real one)."""
    calls = []

    class FakeLSE:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def candles(self, symbol, timeframe="1m", start=None, end=None, limit=5000, market=None, **kw):
            calls.append((symbol, market, start))
            if market is None:
                raise _Err(400, f"'{symbol}' exists in two markets; pass market=", "market-ambiguous")
            if holds.get(symbol) != market:
                raise _Err(404, "not listed", "symbol-unknown")
            first = pd.Timestamp("2024-01-02", tz="UTC")
            want = max(first, pd.Timestamp(start, tz="UTC")) if start else first
            done = -(-int((want - first).total_seconds()) // 300)          # the first bar at or after ``start``
            begin = first + pd.Timedelta(minutes=5 * done)
            n = max(0, min(limit, total - done))
            return _rows(begin, n)
    mod = types.ModuleType("lse")
    mod.LSE, mod.LSEError = FakeLSE, _Err
    monkeypatch.setitem(sys.modules, "lse", mod)
    return calls


def test_lse_forex_symbol_finds_its_market_and_pages(monkeypatch):
    from app.data import london_strategic_edge_source as lse
    calls = _install_fake_lse(monkeypatch, {"EUR/USD": "forex"})
    df = lse.fetch_candles("k", "EUR/USD", "5m", "2024-01-02", "2024-03-01")
    assert len(df) == 12000 and df["timestamp"].is_monotonic_increasing and df["timestamp"].is_unique
    assert {c[1] for c in calls} <= {None, "forex"} and sum(1 for c in calls if c[1] == "forex") == 3


def test_lse_futures_symbol_is_found_by_trying_markets(monkeypatch):
    from app.data import london_strategic_edge_source as lse
    _install_fake_lse(monkeypatch, {"GC": "futures"}, total=300)
    df = lse.fetch_candles("k", "GC", "5m", "2024-01-02", "2024-01-05")
    assert len(df) == 300


def test_lse_unknown_symbol_explains_itself(monkeypatch):
    from app.data import london_strategic_edge_source as lse
    _install_fake_lse(monkeypatch, {})
    with pytest.raises(lse.LSEFetchError) as e:
        lse.fetch_candles("k", "NOPE", "5m", "2024-01-02", "2024-01-05")
    assert "NOPE" in str(e.value) and "tried" in str(e.value)


def test_lse_limit_caps_rows(monkeypatch):
    from app.data import london_strategic_edge_source as lse
    _install_fake_lse(monkeypatch, {"ES": "futures"})
    assert len(lse.fetch_candles("k", "ES", "5m", "2024-01-02", "2024-03-01", limit=6500)) == 6500


def test_lse_provider_path_uses_saved_key(monkeypatch):
    from app.accounts import api_keys
    _install_fake_lse(monkeypatch, {"EUR/USD": "forex"}, total=400)
    monkeypatch.setattr(api_sources, "_key_for", lambda p: "lse-key-abcdef")
    df = api_sources.fetch("lse", "EUR/USD", "5m", "2024-01-02", "2024-01-04")
    assert len(df) > 0 and list(df.columns)[:6] == ["timestamp", "open", "high", "low", "close", "volume"]
