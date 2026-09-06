from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

import cli
from app.strategy.library import delete_saved_strategy, save_strategy_metadata, save_strategy_text


def _run(args_list, capsys):
    rc = cli.main(args_list)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def _synthetic_ohlcv_csv(tmp_path, n=1500, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    price = 1900 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({"timestamp": ts, "open": price, "high": price + 0.3, "low": price - 0.3,
                        "close": price, "volume": 100.0})
    path = tmp_path / "ohlcv.csv"
    df.to_csv(path, index=False)
    return path


def test_build_parser_registers_every_subcommand():
    parser = cli.build_parser()
    sub_actions = [a for a in parser._subparsers._group_actions if hasattr(a, "choices")]
    commands = set(sub_actions[0].choices.keys())
    expected = {
        "translate-strategy", "auto-regime-select", "strategy-health", "portfolio-composer",
        "pairs-screen", "pairs-backtest", "options-price", "options-implied-vol", "options-compare",
        "order-book-replay", "sentiment-fetch", "sentiment-correlate", "portfolio-optimize",
        "vol-surface", "factor-model",
    }
    assert expected <= commands


def test_options_price_cli(capsys):
    rc, out, err = _run(
        ["options-price", "--spot", "100", "--strike", "100", "--expiry-years", "1",
         "--rate", "0.05", "--vol", "0.2", "--type", "call"], capsys,
    )
    assert rc == 0
    assert "10.4506" in out


def test_options_price_cli_writes_json(tmp_path, capsys):
    out_path = tmp_path / "opt.json"
    rc, out, err = _run(
        ["options-price", "--spot", "100", "--strike", "100", "--expiry-years", "1",
         "--rate", "0.05", "--vol", "0.2", "--out", str(out_path)], capsys,
    )
    assert rc == 0
    data = json.loads(out_path.read_text())
    assert data["price"] == pytest.approx(10.4506, abs=1e-3)
    assert "delta" in data["greeks"]


def test_options_implied_vol_cli(capsys):
    rc, out, err = _run(
        ["options-implied-vol", "--market-price", "10.4506", "--spot", "100", "--strike", "100",
         "--expiry-years", "1", "--rate", "0.05", "--type", "call"], capsys,
    )
    assert rc == 0
    assert "20.0" in out


def test_translate_strategy_cli(tmp_path, capsys):
    config = {
        "name": "CLI Test Strategy",
        "entry_conditions": {
            "long": [{"left": {"type": "ema", "period": 20}, "operator": "crosses above",
                      "right": {"type": "ema", "period": 50}}],
            "short": [{"left": {"type": "ema", "period": 20}, "operator": "crosses below",
                       "right": {"type": "ema", "period": 50}}],
        },
        "exit_conditions": {}, "risk_management": {},
    }
    config_path = tmp_path / "strat.json"
    config_path.write_text(json.dumps(config))
    out_path = tmp_path / "strat.pine"
    rc, out, err = _run(
        ["translate-strategy", "--config", str(config_path), "--target", "pinescript", "--out", str(out_path)],
        capsys,
    )
    assert rc == 0
    assert out_path.exists()
    assert "//@version=5" in out_path.read_text()


def test_translate_strategy_cli_reports_unsupported_construct(tmp_path, capsys):
    config = {"entry_conditions": {"long": [{"left": {"type": "liquidity_sweep"}, "operator": ">", "right": 0}]}}
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps(config))
    rc, out, err = _run(
        ["translate-strategy", "--config", str(config_path), "--target", "pinescript",
         "--out", str(tmp_path / "out.pine")], capsys,
    )
    assert rc == 1
    assert "liquidity_sweep" in err


def test_order_book_replay_cli(tmp_path, capsys):
    orders = [
        {"side": "buy", "type": "limit", "price": 100.0, "quantity": 10},
        {"side": "sell", "type": "limit", "price": 100.0, "quantity": 4},
    ]
    orders_path = tmp_path / "orders.json"
    orders_path.write_text(json.dumps(orders))
    rc, out, err = _run(["order-book-replay", "--orders", str(orders_path)], capsys)
    assert rc == 0
    assert "1 trades printed" in out or "trades printed" in out


def test_vol_surface_demo_cli(tmp_path, capsys):
    out_path = tmp_path / "surface.html"
    rc, out, err = _run(["vol-surface", "--demo", "--spot", "100", "--out", str(out_path)], capsys)
    assert rc == 0
    assert out_path.exists()
    assert "plot.ly" in out_path.read_text() or "cdn.plot.ly" in out_path.read_text()


def test_auto_regime_select_cli(tmp_path, capsys):
    ohlcv_path = _synthetic_ohlcv_csv(tmp_path)
    manual_config = {
        "indicators": [{"type": "ema", "period": 10, "column": "close", "as": "ema_fast"},
                        {"type": "ema", "period": 40, "column": "close", "as": "ema_slow"}],
        "long_entry": "ema_fast > ema_slow", "short_entry": "ema_fast < ema_slow",
    }
    save_strategy_text(json.dumps(manual_config), "cli_test_regime.json", "manual", overwrite=True)
    save_strategy_metadata("manual", "cli_test_regime.json", {"status": "validated"})
    try:
        out_path = tmp_path / "regime.json"
        rc, out, err = _run(
            ["auto-regime-select", "--data", str(ohlcv_path), "--dimension", "trend",
             "--strategy-type", "manual", "--min-trades", "3", "--out", str(out_path)], capsys,
        )
        assert rc == 0
        data = json.loads(out_path.read_text())
        assert data["regime_dimension"] == "trend"
        assert "router" not in data  # non-serializable Strategy object must never leak into JSON
    finally:
        delete_saved_strategy("manual", "cli_test_regime.json")


def test_pairs_screen_cli_with_mocked_alpaca(tmp_path, capsys, monkeypatch):
    rng = np.random.default_rng(2)
    n = 400
    ts = pd.date_range("2023-01-01", periods=n, freq="1D")
    common = np.cumsum(rng.normal(0, 1, n))
    a = 100 + common + rng.normal(0, 0.3, n)
    b = (a - rng.normal(0, 0.3, n)) / 2 + 20

    def fake_df(price):
        return pd.DataFrame({"timestamp": ts, "open": price, "high": price + 0.2, "low": price - 0.2,
                              "close": price, "volume": 1000.0})

    universe = {"A": fake_df(a), "B": fake_df(b)}

    def fake_fetch_universe(api_key, secret_key, symbols, timeframe_label, start, end, feed="iex"):
        return universe

    import app.quant_lab.pairs_trading as pt
    monkeypatch.setattr(pt, "fetch_universe", fake_fetch_universe)

    rc, out, err = _run(
        ["pairs-screen", "--alpaca-key", "k", "--alpaca-secret", "s", "--symbols", "A,B",
         "--start", "2023-01-01", "--end", "2024-01-01", "--min-correlation", "0.3"], capsys,
    )
    assert rc == 0
    assert "A/B" in out


def test_factor_model_cli_with_mocked_factors(tmp_path, capsys, monkeypatch):
    ohlcv_path = _synthetic_ohlcv_csv(tmp_path, n=4000)
    df = pd.read_csv(ohlcv_path, parse_dates=["timestamp"])

    rng = np.random.default_rng(3)
    dates = pd.date_range(df["timestamp"].min().normalize(), periods=400, freq="1D")
    factors = pd.DataFrame({
        "mkt_rf": rng.normal(0.0004, 0.01, len(dates)), "smb": rng.normal(0, 0.005, len(dates)),
        "hml": rng.normal(0, 0.005, len(dates)), "rf": 0.00005,
    }, index=dates)

    import app.quant_lab.factor_model as fm
    monkeypatch.setattr(fm, "fetch_fama_french_factors", lambda frequency="daily", **kw: factors)

    rc, out, err = _run(
        ["factor-model", "--price-csv", str(ohlcv_path), "--frequency", "daily"], capsys,
    )
    assert rc == 0
    assert "alpha" in out.lower() or "Alpha" in out


def test_missing_required_arg_exits_nonzero():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["options-price", "--spot", "100"])


def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["not-a-real-command"])
