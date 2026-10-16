"""Regression tests for app/data/importer.py's tick/quote-stream ->
OHLCV aggregation (UPGRADE: previously a raw tick/Level-1-quote export
-- timestamp + a single price column, no open/high/low -- failed with
"Could not identify required column(s): ['open', 'high', 'low']" even
though the data was perfectly good, just not pre-aggregated into bars.
Reproduces the exact real-world case that surfaced this: an
NxCore/CQG/eSignal-style "Timestamp, Price, Volume, DataType,
Correction, MarketState" time & sales export, where DataType is
A=ask/B=bid/T=trade.
"""
import pandas as pd
import pytest

from app.data.importer import import_csv, import_csv_bytes


def _tick_csv_text(n=200, start="2024-01-01 12:00:00.000", trade_every=5):
    """Builds a synthetic tick/quote file in the same shape as Owen's
    real MNQ/MGC exports: dense sub-second timestamps, mostly bid/ask
    quotes with a trade every `trade_every`th row."""
    lines = ["Timestamp,Price,Volume,DataType,Correction,MarketState"]
    ts0 = pd.Timestamp(start)
    price = 30000.0
    for i in range(n):
        ts = ts0 + pd.Timedelta(milliseconds=i * 50)
        price += (0.25 if i % 3 == 0 else -0.25)
        data_type = "T" if i % trade_every == 0 else ("A" if i % 2 == 0 else "B")
        ts_str = ts.strftime("%Y%m%d %H:%M:%S.%f")[:-3]
        lines.append(f"{ts_str},{price:.2f},{1 + i % 4},{data_type},R,N")
    return "\n".join(lines) + "\n"


def test_tick_stream_with_trade_marker_aggregates_to_ohlcv(tmp_path):
    text = _tick_csv_text(n=200, trade_every=5)
    p = tmp_path / "MNQ2026M.txt"
    p.write_text(text)

    result = import_csv(str(p))

    assert result.is_valid, result.errors
    assert result.dataframe is not None
    df = result.dataframe
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) > 0
    # High >= low, high >= open/close, low <= open/close for every bar.
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    # A clear warning explains what happened -- this must never be a
    # silent transformation.
    assert any("tick/quote-level" in w for w in result.warnings)
    assert any("executed trade" in w for w in result.warnings)


def test_tick_stream_prefers_trades_over_quotes(tmp_path):
    """When there are enough real trade ('T') rows, bars must be built
    from those, not from the far more numerous bid/ask quotes -- a
    quote was never an actual traded price."""
    lines = ["Timestamp,Price,Volume,DataType,Correction,MarketState"]
    ts0 = pd.Timestamp("2024-01-01 12:00:00.000")
    # 100 quotes at a wildly different price than the 10 trades, packed
    # into the same tiny window (so the whole thing still reads as one
    # tick stream / one bar).
    for i in range(100):
        ts = ts0 + pd.Timedelta(milliseconds=i * 5)
        ts_str = ts.strftime("%Y%m%d %H:%M:%S.%f")[:-3]
        lines.append(f"{ts_str},9999.00,1,{'A' if i % 2 == 0 else 'B'},R,N")
    for i in range(10):
        ts = ts0 + pd.Timedelta(milliseconds=i * 5)
        ts_str = ts.strftime("%Y%m%d %H:%M:%S.%f")[:-3]
        lines.append(f"{ts_str},100.00,1,T,R,N")
    text = "\n".join(lines) + "\n"

    p = tmp_path / "ticks.txt"
    p.write_text(text)
    result = import_csv(str(p))

    assert result.is_valid, result.errors
    # Every bar's close must reflect the trade price (100), never the
    # quote price (9999) -- proves the trade-only filter actually took
    # effect rather than blending everything together.
    assert (result.dataframe["close"] < 200).all()


def test_tick_stream_falls_back_to_all_rows_without_enough_trades(tmp_path):
    """No usable trade marker (or too few trade rows) -> aggregate from
    every row rather than refusing to import at all."""
    lines = ["Timestamp,Price,Volume"]
    ts0 = pd.Timestamp("2024-01-01 12:00:00.000")
    for i in range(150):
        ts = ts0 + pd.Timedelta(milliseconds=i * 20)
        ts_str = ts.strftime("%Y%m%d %H:%M:%S.%f")[:-3]
        lines.append(f"{ts_str},{1900 + i * 0.01:.2f},{1 + i % 3}")
    text = "\n".join(lines) + "\n"

    p = tmp_path / "quotes_only.csv"
    p.write_text(text)
    result = import_csv(str(p))

    assert result.is_valid, result.errors
    assert len(result.dataframe) > 0
    assert any("tick/quote-level" in w for w in result.warnings)


def test_short_span_tick_data_produces_many_small_bars(tmp_path):
    """A very short tick sample (Owen's real MNQ file spans only ~51
    seconds) must still be aggregated into multiple fine-grained bars,
    not crushed into one giant bar by defaulting to something like a
    1-minute or 1-hour interval."""
    text = _tick_csv_text(n=300, trade_every=4)
    p = tmp_path / "short_sample.txt"
    p.write_text(text)
    result = import_csv(str(p))

    assert result.is_valid, result.errors
    assert len(result.dataframe) >= 5


def test_normal_ohlcv_file_is_never_touched_by_tick_aggregation(tmp_path):
    """A real OHLCV file (already has open/high/low/close) must import
    exactly as before -- the tick-aggregation path only ever engages
    when open/high/low are ALL absent."""
    ts = pd.date_range("2024-01-01", periods=50, freq="15min")
    price = [1900.0 + i * 0.1 for i in range(50)]
    df = pd.DataFrame({
        "timestamp": ts, "open": price, "high": price, "low": price, "close": price, "volume": 10.0,
    })
    p = tmp_path / "normal.csv"
    df.to_csv(p, index=False)

    result = import_csv(str(p))
    assert result.is_valid
    assert len(result.dataframe) == len(df)
    assert not any("tick/quote-level" in w for w in result.warnings)


def test_daily_close_only_series_is_not_misdetected_as_tick_data(tmp_path):
    """A legitimate close-only series with normal daily spacing (no
    open/high/low, but NOT a tick stream) must still get the original
    "missing required column" error rather than being silently
    aggregated into nonsense bars -- _is_probable_tick_stream's average-
    gap check exists specifically to keep this file out of the tick path."""
    ts = pd.date_range("2024-01-01", periods=100, freq="1D")
    price = [1900.0 + i * 0.5 for i in range(100)]
    df = pd.DataFrame({"timestamp": ts, "close": price})
    p = tmp_path / "daily_close.csv"
    df.to_csv(p, index=False)

    result = import_csv(str(p))
    assert not result.is_valid
    assert "open" in result.errors[0] or "['open'" in result.errors[0]


def test_quote_board_snapshot_is_not_misdetected_as_tick_data(tmp_path):
    """A Barchart-style multi-contract quote-board snapshot (one row per
    contract expiry, not a time series) has too few rows to look like a
    tick stream and must still fail cleanly rather than being
    misinterpreted as one bar per contract."""
    text = (
        'Contract,Latest,Change,Open,High,Low,Previous,Volume,"Open Int",Time\n'
        '"ETZ26 (Dec \'26)",7733.75,-38.75,7775.5,7779.5,7710,7772.5,230259,124243,"07:15 CT"\n'
        '"ETH27 (Mar \'27)",7816.5,-38.5,7858.25,7861,7791,7855,769,1790,"07:14 CT"\n'
        '"ETM27 (Jun \'27)",7942,-56.5,7939,7943,7931,7998.5,12,63,2026-09-23\n'
    )
    p = tmp_path / "snapshot.csv"
    p.write_text(text)
    result = import_csv(str(p))
    # "Latest" doesn't match any close alias -- this must still fail
    # with a clear "missing column" error, not be silently accepted.
    assert not result.is_valid
    assert "close" in result.errors[0]


def test_import_csv_bytes_still_works_with_tick_aggregation(tmp_path):
    """The Flask/Tkinter upload path (import_csv_bytes) gets the same
    aggregation as the on-disk path."""
    text = _tick_csv_text(n=150, trade_every=6)
    result = import_csv_bytes(text.encode("utf-8"), filename="MGC2026M.txt")
    assert result.is_valid, result.errors
    assert len(result.dataframe) > 0
