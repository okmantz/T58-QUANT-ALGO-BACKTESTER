from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.forward_test.journal import ForwardTestJournal
from app.strategy.library import delete_saved_strategy, save_strategy_metadata, save_strategy_text
from app.web.server import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def _has_result(body: str) -> bool:
    return 'class="card result-ok"' in body


def _get_error(body: str) -> str | None:
    m = re.search(r'card result-error">(.*?)</div>\s*</div>', body, re.S)
    return m.group(1) if m else None


def _synthetic_ohlcv_bytes(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    price = 1900 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({"timestamp": ts, "open": price, "high": price + 0.3, "low": price - 0.3,
                        "close": price, "volume": 100.0})
    return df.to_csv(index=False).encode()


TOOL_PAGES = [
    "/quant-lab/translator", "/quant-lab/regime-selector", "/quant-lab/strategy-health",
    "/quant-lab/portfolio-composer", "/quant-lab/pairs-screen", "/quant-lab/pairs-backtest",
    "/quant-lab/options-pricing", "/quant-lab/order-book", "/quant-lab/sentiment-price",
    "/quant-lab/portfolio-optimizer", "/quant-lab/vol-surface", "/quant-lab/factor-model",
]


def test_landing_page_lists_every_tool(client):
    r = client.get("/quant-lab", follow_redirects=True)
    assert r.status_code == 200
    for url in TOOL_PAGES:
        assert url.encode() in r.data


@pytest.mark.parametrize("path", TOOL_PAGES)
def test_every_tool_page_loads_via_get(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert b"<form" in r.data


def test_options_pricing_post(client):
    r = client.post("/quant-lab/options-pricing", data={
        "spot": "100", "strike": "100", "expiry_years": "1", "rate": "0.05", "vol": "0.2",
        "option_type": "call", "market_price": "",
    })
    body = r.data.decode()
    assert r.status_code == 200
    assert _has_result(body)
    assert "10.45" in body


def test_options_pricing_compare_mode(client):
    r = client.post("/quant-lab/options-pricing", data={
        "spot": "100", "strike": "100", "expiry_years": "1", "rate": "0.05", "vol": "0.2",
        "option_type": "call", "market_price": "10.9506",
    })
    body = r.data.decode()
    assert _has_result(body)
    assert "implied" in body.lower()


def test_order_book_post(client):
    orders = json.dumps([
        {"side": "buy", "type": "limit", "price": 100.0, "quantity": 10},
        {"side": "sell", "type": "limit", "price": 100.0, "quantity": 4},
    ])
    r = client.post("/quant-lab/order-book", data={"orders_json": orders})
    body = r.data.decode()
    assert _has_result(body), _get_error(body)
    assert "trade(s)" in body


def test_order_book_invalid_json_shows_error(client):
    r = client.post("/quant-lab/order-book", data={"orders_json": "not json"})
    body = r.data.decode()
    assert not _has_result(body)
    assert _get_error(body) is not None


def test_regime_selector_post_with_upload(client):
    config = {
        "indicators": [{"type": "ema", "period": 10, "column": "close", "as": "ema_fast"},
                        {"type": "ema", "period": 40, "column": "close", "as": "ema_slow"}],
        "long_entry": "ema_fast > ema_slow", "short_entry": "ema_fast < ema_slow",
    }
    save_strategy_text(json.dumps(config), "web_test_regime.json", "manual", overwrite=True)
    save_strategy_metadata("manual", "web_test_regime.json", {"status": "validated"})
    try:
        r = client.post("/quant-lab/regime-selector", data={
            "data_csv": (io.BytesIO(_synthetic_ohlcv_bytes()), "ohlcv.csv"),
            "dimension": "trend", "min_trades": "5",
        }, content_type="multipart/form-data")
        body = r.data.decode()
        assert _has_result(body), _get_error(body)
    finally:
        delete_saved_strategy("manual", "web_test_regime.json")


def test_portfolio_composer_post_with_upload(client):
    c1 = {"indicators": [{"type": "ema", "period": 10, "column": "close", "as": "ema_fast"},
                          {"type": "ema", "period": 40, "column": "close", "as": "ema_slow"}],
          "long_entry": "ema_fast > ema_slow", "short_entry": "ema_fast < ema_slow"}
    c2 = {"indicators": [{"type": "rsi", "period": 10, "column": "close", "as": "rsi_10"}],
          "long_entry": "rsi_10 < 30", "short_entry": "rsi_10 > 70"}
    save_strategy_text(json.dumps(c1), "web_test_trend.json", "manual", overwrite=True)
    save_strategy_metadata("manual", "web_test_trend.json", {"status": "validated"})
    save_strategy_text(json.dumps(c2), "web_test_meanrev.json", "manual", overwrite=True)
    save_strategy_metadata("manual", "web_test_meanrev.json", {"status": "validated"})
    try:
        r = client.post("/quant-lab/portfolio-composer", data={
            "strategy_names": ["web_test_trend.json", "web_test_meanrev.json"],
            "data_csv": (io.BytesIO(_synthetic_ohlcv_bytes()), "ohlcv.csv"),
            "min_legs": "2", "max_legs": "2", "max_evaluations": "10",
        }, content_type="multipart/form-data")
        body = r.data.decode()
        assert _has_result(body), _get_error(body)
    finally:
        delete_saved_strategy("manual", "web_test_trend.json")
        delete_saved_strategy("manual", "web_test_meanrev.json")


def test_strategy_health_post_with_uploads(client, tmp_path):
    db_path = tmp_path / "journal.db"
    journal = ForwardTestJournal(db_path=db_path)
    session_id = journal.start_session("python", "strat.py", "XAUUSD", 15, "12345", "Demo-Server")
    for i, pnl in enumerate([50.0] * 20):
        trade_id = journal.record_open(session_id, mt5_ticket=i, direction=1, volume=0.1,
                                        entry_price=2000.0, sl_price=1990.0, tp_price=2020.0)
        journal.record_close(trade_id, exit_price=2000.0 + pnl, pnl=pnl)

    mc_data = dict(
        n_simulations=1000, evaluation_pass_probability=0.6, first_payout_probability=0.5,
        failure_before_payout_probability=0.4, multiple_payout_probability=0.2,
        median_days_to_pass=20.0, median_days_to_first_payout=40.0, average_days_to_first_payout=42.0,
        median_return_pct=8.0, mean_return_pct=7.5, expected_payout=1000.0, median_payout=900.0,
        total_simulated_withdrawals=500_000.0, median_drawdown_pct=3.0, p95_drawdown_pct=6.0,
        worst_drawdown_pct=9.0, risk_of_ruin_pct=2.0, median_max_losing_streak=4.0, worst_max_losing_streak=9,
        return_percentiles={"5": 1.0, "25": 5.0, "50": 8.0, "75": 11.0, "95": 15.0},
        drawdown_percentiles={"5": 1.0, "25": 2.0, "50": 3.0, "75": 4.5, "95": 6.0},
        days_to_payout_distribution=[], return_distribution=[1.0, 4.0, 6.0, 8.0, 8.0, 9.0, 11.0, 13.0, 15.0, 15.5] * 100,
        drawdown_distribution=[1.0, 2.0, 3.0, 4.0, 6.0] * 200,
    )
    mc_path = tmp_path / "mc.json"
    mc_path.write_text(json.dumps(mc_data))

    with open(db_path, "rb") as jf, open(mc_path, "rb") as mf:
        r = client.post("/quant-lab/strategy-health", data={
            "journal_db": (jf, "journal.db"), "session_id": str(session_id),
            "strategy_label": "test_strat", "mc_result_json": (mf, "mc.json"), "account_balance": "10000",
        }, content_type="multipart/form-data")
    body = r.data.decode()
    assert _has_result(body), _get_error(body)


def test_pairs_screen_missing_alpaca_keys_shows_error(client, monkeypatch):
    from app.data import alpaca_credentials
    monkeypatch.setattr(alpaca_credentials, "load_credentials", lambda: None)
    r = client.post("/quant-lab/pairs-screen", data={
        "symbols": "AAPL,MSFT", "timeframe": "1Day", "start": "2023-01-01", "end": "2024-01-01",
        "min_correlation": "0.7", "alpaca_key": "", "alpaca_secret": "",
    })
    body = r.data.decode()
    assert not _has_result(body)
    assert _get_error(body) is not None


def test_factor_model_post_with_mocked_factors(client, tmp_path, monkeypatch):
    csv_bytes = _synthetic_ohlcv_bytes(n=4000)
    df = pd.read_csv(io.BytesIO(csv_bytes), parse_dates=["timestamp"])
    rng = np.random.default_rng(3)
    dates = pd.date_range(df["timestamp"].min().normalize(), periods=400, freq="1D")
    factors = pd.DataFrame({
        "mkt_rf": rng.normal(0.0004, 0.01, len(dates)), "smb": rng.normal(0, 0.005, len(dates)),
        "hml": rng.normal(0, 0.005, len(dates)), "rf": 0.00005,
    }, index=dates)

    import app.quant_lab.factor_model as fm
    monkeypatch.setattr(fm, "fetch_fama_french_factors", lambda frequency="daily", **kw: factors)

    r = client.post("/quant-lab/factor-model", data={
        "price_csv": (io.BytesIO(csv_bytes), "ohlcv.csv"), "frequency": "daily",
    }, content_type="multipart/form-data")
    body = r.data.decode()
    assert _has_result(body), _get_error(body)
