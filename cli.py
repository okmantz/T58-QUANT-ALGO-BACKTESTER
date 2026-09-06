"""
T58 Quant Algo Backtester -- Command-Line Interface.

Wires every backend module added across this project's recent upgrade
rounds (the Universal Strategy Translator, Auto Regime Selector, Strategy
Health monitor, Portfolio Composer, and the whole app/quant_lab/ toolkit)
into scriptable `python cli.py <command> ...` subcommands, so any of them
can be run headlessly -- in a cron job, a CI pipeline, or just from a
terminal -- without opening the desktop app or the web UI.

This file stays at the repository root (same reasoning as run_app.py and
run_web.py's own docstrings: a script inside the `app` package breaks
PyInstaller's import analysis, and this file has the same "must be
runnable both frozen and unfrozen" requirement those two already
document).

Every subcommand is a thin wrapper: it parses arguments, loads whatever
CSV/JSON inputs it needs (reusing app.data.importer.import_csv for OHLCV
data, so a CLI-loaded CSV goes through the exact same column-mapping/
validation path a file dropped into the desktop app would), calls straight
into the already-tested library function, and prints or saves the result.
No business logic lives in this file -- if a number here looks wrong, the
bug is in the underlying module, not here (same principle
app.reports.survival_report's own docstring states for its thin
presentation layer).

Run `python cli.py --help` for the full command list, or
`python cli.py <command> --help` for a specific command's arguments.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd


def _load_ohlcv_csv(path: str) -> pd.DataFrame:
    from app.data.importer import import_csv

    result = import_csv(path)
    if result.dataframe is None:
        errors = "; ".join(i.message for i in result.issues if i.level == "error")
        print(f"error: could not load '{path}' as OHLCV data: {errors}", file=sys.stderr)
        sys.exit(2)
    return result.dataframe


def _resolve_alpaca_keys(args) -> tuple[str, str]:
    if args.alpaca_key and args.alpaca_secret:
        return args.alpaca_key, args.alpaca_secret
    from app.data.alpaca_credentials import load_credentials

    creds = load_credentials()
    if creds is None or not creds.is_usable:
        print(
            "error: no Alpaca API key/secret given (--alpaca-key/--alpaca-secret) and none saved "
            "from the desktop/web app's Data tab either.",
            file=sys.stderr,
        )
        sys.exit(2)
    return creds.api_key, creds.secret_key


def _to_jsonable(obj):
    # Prefer a class's own to_dict() (many result dataclasses in this app
    # deliberately exclude non-serializable fields there -- e.g.
    # AutoRegimeSelectionResult.to_dict() omits its `router` Strategy
    # object) over generic dataclasses.asdict(), which would serialize
    # every raw field including ones never meant to leave the process.
    if hasattr(obj, "to_dict") and not isinstance(obj, (dict, list, tuple)):
        return _to_jsonable(obj.to_dict())
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records", date_format="iso"))
    return obj


def _emit(result, out_path: str | None, summary: str | None = None) -> None:
    if summary:
        print(summary)
    if out_path:
        payload = _to_jsonable(result)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# 1. Universal Strategy Translator
# ---------------------------------------------------------------------------

def cmd_translate_strategy(args):
    from app.strategy.translator import to_mql5, to_pinescript

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    code = to_pinescript(config) if args.target == "pinescript" else to_mql5(config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(code, encoding="utf-8")
    print(f"Wrote {args.out} ({args.target})")


# ---------------------------------------------------------------------------
# 2. Auto Regime Selector
# ---------------------------------------------------------------------------

def cmd_auto_regime_select(args):
    from app.strategy.auto_regime_selector import select_regime_strategies
    from app.strategy.library_loader import load_validated_candidates

    df = _load_ohlcv_csv(args.data)
    candidates = load_validated_candidates(strategy_type=args.strategy_type)
    if not candidates:
        print("error: no validated Strategy Library entries found to consider.", file=sys.stderr)
        sys.exit(1)
    result = select_regime_strategies(
        df, candidates, args.dimension, min_trades_per_cell=args.min_trades,
    )
    _emit(result, args.out, result.render_table())


# ---------------------------------------------------------------------------
# 3. Strategy Health / Drift Monitor
# ---------------------------------------------------------------------------

def cmd_strategy_health(args):
    from app.forward_test.journal import ForwardTestJournal
    from app.monitoring.strategy_health import check_strategy_health
    from app.monte_carlo.engine import MonteCarloResult

    with open(args.mc_result, "r", encoding="utf-8") as f:
        mc_data = json.load(f)
    predicted = MonteCarloResult(**mc_data)
    journal = ForwardTestJournal(db_path=Path(args.journal_db))
    result = check_strategy_health(
        journal, args.session_id, args.strategy_label, predicted, account_balance=args.account_balance,
    )
    _emit(result, args.out, result.render_table())


# ---------------------------------------------------------------------------
# 4. Automated Portfolio Composer
# ---------------------------------------------------------------------------

def cmd_portfolio_composer(args):
    from app.portfolio.composer import compose_portfolio
    from app.portfolio.portfolio import InstrumentLeg, PortfolioConfig
    from app.backtest.risk import RiskConfig
    from app.strategy.library_loader import load_strategy_object
    from app.strategy.library import list_saved_strategies

    with open(args.legs, "r", encoding="utf-8") as f:
        legs_spec = json.load(f)
    stored_by_name = {s.name: s for s in list_saved_strategies()}
    legs = []
    for entry in legs_spec:
        stored = stored_by_name.get(entry["strategy_name"])
        if stored is None:
            print(f"error: no Strategy Library entry named '{entry['strategy_name']}'.", file=sys.stderr)
            sys.exit(1)
        legs.append(InstrumentLeg(
            name=entry.get("name", entry["strategy_name"]),
            df=_load_ohlcv_csv(entry["data_csv"]),
            strategy=load_strategy_object(stored),
            risk=RiskConfig(**entry.get("risk", {})),
        ))
    cfg = PortfolioConfig(**args.portfolio_config) if args.portfolio_config else PortfolioConfig()
    result = compose_portfolio(
        legs, min_legs=args.min_legs, max_legs=args.max_legs,
        max_evaluations=args.max_evaluations, portfolio_config=cfg,
    )
    _emit(result, args.out, result.render_table())


# ---------------------------------------------------------------------------
# 5. Pairs Trading
# ---------------------------------------------------------------------------

def cmd_pairs_screen(args):
    from app.quant_lab.pairs_trading import fetch_universe, screen_pairs

    api_key, secret_key = _resolve_alpaca_keys(args)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    universe = fetch_universe(api_key, secret_key, symbols, args.timeframe, args.start, args.end)
    candidates = screen_pairs(universe, min_correlation=args.min_correlation, zscore_period=args.zscore_period)
    summary = "\n".join(
        f"{c.symbol_a}/{c.symbol_b}: corr={c.correlation:.3f} adf={c.adf_statistic:+.2f} "
        f"({c.stationarity_verdict}) z={c.current_zscore:+.2f}"
        for c in candidates
    ) or "No pairs cleared the correlation threshold."
    _emit([c.to_dict() for c in candidates], args.out, summary)


def cmd_pairs_backtest(args):
    from app.quant_lab.pairs_trading import run_pairs_backtest
    from app.backtest.risk import RiskConfig

    api_key, secret_key = _resolve_alpaca_keys(args)
    result = run_pairs_backtest(
        api_key, secret_key, args.symbol_a, args.symbol_b, args.timeframe, args.start, args.end,
        risk=RiskConfig(pip_size=args.pip_size), zscore_period=args.zscore_period,
        entry_z=args.entry_z, exit_z=args.exit_z,
    )
    stats = result.backtest.statistics
    summary = (
        f"Pair: {args.symbol_a}/{args.symbol_b}\n"
        + (result.pair.stationarity_verdict if result.pair else "n/a") + "\n"
        f"Trades: {stats.total_trades}  Net profit: {stats.net_profit:+.2f}  Win rate: {stats.win_rate:.1f}%"
    )
    _emit({"pair": result.pair, "statistics": stats}, args.out, summary)


# ---------------------------------------------------------------------------
# 6. Options Pricing Calculator
# ---------------------------------------------------------------------------

def cmd_options_price(args):
    from app.quant_lab.options_pricing import black_scholes_greeks, black_scholes_price

    price = black_scholes_price(args.spot, args.strike, args.expiry_years, args.rate, args.vol, args.type, args.dividend)
    greeks = black_scholes_greeks(args.spot, args.strike, args.expiry_years, args.rate, args.vol, args.type, args.dividend)
    summary = f"Price: {price:.4f}\n" + "\n".join(f"{k}: {v:+.4f}" for k, v in greeks.to_dict().items())
    _emit({"price": price, "greeks": greeks}, args.out, summary)


def cmd_options_implied_vol(args):
    from app.quant_lab.options_pricing import implied_volatility

    iv = implied_volatility(args.market_price, args.spot, args.strike, args.expiry_years, args.rate, args.type, args.dividend)
    _emit({"implied_vol": iv}, args.out, f"Implied volatility: {iv:.2%}")


def cmd_options_compare(args):
    from app.quant_lab.options_pricing import compare_to_market

    result = compare_to_market(args.market_price, args.spot, args.strike, args.expiry_years, args.rate, args.vol, args.type, args.dividend)
    _emit(result, args.out, result.render_summary())


# ---------------------------------------------------------------------------
# 7. Order Book Simulator
# ---------------------------------------------------------------------------

def cmd_order_book_replay(args):
    from app.quant_lab.order_book import LimitOrderBook

    with open(args.orders, "r", encoding="utf-8") as f:
        orders = json.load(f)
    book = LimitOrderBook(symbol=args.symbol)
    all_trades = []
    for order in orders:
        if order["type"] == "limit":
            result = book.submit_limit_order(order["side"], order["price"], order["quantity"])
        else:
            result = book.submit_market_order(order["side"], order["quantity"])
        all_trades.extend(result.trades)
        if args.verbose:
            print(f"order {order} -> {len(result.trades)} trade(s), resting={result.resting}")
    summary = (
        f"Processed {len(orders)} orders, {len(all_trades)} trades printed.\n"
        f"Best bid/ask: {book.best_bid()} / {book.best_ask()}   Spread: {book.spread()}\n"
        f"Depth: {book.depth_snapshot(levels=args.depth_levels)}"
    )
    _emit({"trades": all_trades, "depth": book.depth_snapshot(levels=args.depth_levels)}, args.out, summary)


# ---------------------------------------------------------------------------
# 8. Sentiment-Price Correlation
# ---------------------------------------------------------------------------

def cmd_sentiment_fetch(args):
    from app.quant_lab.sentiment_price import fetch_headlines, score_headlines

    headlines = fetch_headlines(args.query, max_results=args.max_results)
    df = score_headlines(headlines)
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv} ({len(df)} headlines)")
    else:
        print(df.to_string(index=False))


def cmd_sentiment_correlate(args):
    from app.quant_lab.sentiment_price import correlate_sentiment_with_price

    sentiment_df = pd.read_csv(args.sentiment_csv, parse_dates=["timestamp"])
    price_df = _load_ohlcv_csv(args.price_csv)
    result = correlate_sentiment_with_price(sentiment_df, price_df)
    _emit(result, args.out, result.render_summary())


# ---------------------------------------------------------------------------
# 9. Portfolio Optimizer (Markowitz)
# ---------------------------------------------------------------------------

def cmd_portfolio_optimize(args):
    from app.quant_lab.portfolio_optimizer import fetch_and_build_inputs, optimize_for_risk_level

    api_key, secret_key = _resolve_alpaca_keys(args)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    inputs = fetch_and_build_inputs(api_key, secret_key, tickers, args.start, args.end, args.timeframe)
    allocation = optimize_for_risk_level(
        inputs, args.risk_level, risk_free_rate=args.risk_free_rate, long_only=args.long_only,
    )
    _emit(allocation, args.out, allocation.render_summary())


# ---------------------------------------------------------------------------
# 10. Volatility Surface Visualizer
# ---------------------------------------------------------------------------

def cmd_vol_surface(args):
    from app.quant_lab.vol_surface import build_iv_surface, export_surface_html, fetch_option_chain, synthetic_demo_chain

    if args.demo:
        chain = synthetic_demo_chain(spot=args.spot)
    else:
        api_key, secret_key = _resolve_alpaca_keys(args)
        chain = fetch_option_chain(api_key, secret_key, args.symbol, option_type=args.option_type)
    surface = build_iv_surface(chain, r=args.rate)
    out_path = export_surface_html(surface, args.out, title=args.title)
    print(f"Wrote {out_path} ({len(surface)} points)")


# ---------------------------------------------------------------------------
# 11. Factor Model
# ---------------------------------------------------------------------------

def cmd_factor_model(args):
    from app.quant_lab.factor_model import (
        compute_factor_exposures,
        compute_returns_from_prices,
        fetch_fama_french_factors,
    )

    price_df = _load_ohlcv_csv(args.price_csv)
    returns = compute_returns_from_prices(price_df)
    factors = fetch_fama_french_factors(frequency=args.frequency)
    periods_per_year = 252 if args.frequency == "daily" else 12
    result = compute_factor_exposures(returns, factors, periods_per_year=periods_per_year)
    _emit(result, args.out, result.render_summary())


# ---------------------------------------------------------------------------
# Argument parser assembly
# ---------------------------------------------------------------------------

def _add_alpaca_args(p):
    p.add_argument("--alpaca-key", default=None, help="Alpaca API key (defaults to the key saved via the desktop/web app's Data tab)")
    p.add_argument("--alpaca-secret", default=None, help="Alpaca API secret (same default behavior as --alpaca-key)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("translate-strategy", help="Convert a Manual Strategy Builder config to PineScript or MQL5")
    p.add_argument("--config", required=True, help="Path to a Manual Strategy Builder JSON config")
    p.add_argument("--target", choices=["pinescript", "mql5"], required=True)
    p.add_argument("--out", required=True, help="Output file path (.pine or .mq5)")
    p.set_defaults(func=cmd_translate_strategy)

    p = sub.add_parser("auto-regime-select", help="Auto-assign the best validated strategy per market regime")
    p.add_argument("--data", required=True, help="OHLCV CSV to fit regimes and backtest candidates on")
    p.add_argument("--dimension", choices=["trend", "volatility", "session", "environment"], required=True)
    p.add_argument("--strategy-type", default=None, choices=[None, "python", "pinescript", "mql5", "manual"])
    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--out", default=None, help="Optional JSON output path")
    p.set_defaults(func=cmd_auto_regime_select)

    p = sub.add_parser("strategy-health", help="Compare a forward-test session against its predicted Monte Carlo distribution")
    p.add_argument("--journal-db", required=True)
    p.add_argument("--session-id", type=int, required=True)
    p.add_argument("--strategy-label", required=True)
    p.add_argument("--mc-result", required=True, help="Path to a JSON dump of the strategy's MonteCarloResult")
    p.add_argument("--account-balance", type=float, required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_strategy_health)

    p = sub.add_parser("portfolio-composer", help="Search the Strategy Library for the best N-strategy combination")
    p.add_argument("--legs", required=True, help="JSON list of {strategy_name, data_csv, name?, risk?}")
    p.add_argument("--min-legs", type=int, default=2)
    p.add_argument("--max-legs", type=int, default=4)
    p.add_argument("--max-evaluations", type=int, default=60)
    p.add_argument("--portfolio-config", type=json.loads, default=None, help="JSON dict of PortfolioConfig overrides")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_portfolio_composer)

    p = sub.add_parser("pairs-screen", help="Screen a symbol universe for correlated, mean-reverting pairs")
    _add_alpaca_args(p)
    p.add_argument("--symbols", required=True, help="Comma-separated tickers")
    p.add_argument("--timeframe", default="1Day")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--min-correlation", type=float, default=0.7)
    p.add_argument("--zscore-period", type=int, default=50)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_pairs_screen)

    p = sub.add_parser("pairs-backtest", help="Backtest a pair's mean-reversion spread end to end")
    _add_alpaca_args(p)
    p.add_argument("--symbol-a", required=True)
    p.add_argument("--symbol-b", required=True)
    p.add_argument("--timeframe", default="1Day")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--zscore-period", type=int, default=50)
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--exit-z", type=float, default=0.5)
    p.add_argument("--pip-size", type=float, default=0.01)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_pairs_backtest)

    p = sub.add_parser("options-price", help="Black-Scholes price + Greeks")
    for name, kw in (("--spot", {}), ("--strike", {}), ("--expiry-years", {}), ("--rate", {}), ("--vol", {})):
        p.add_argument(name, type=float, required=True, **kw)
    p.add_argument("--type", choices=["call", "put"], default="call")
    p.add_argument("--dividend", type=float, default=0.0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_options_price)

    p = sub.add_parser("options-implied-vol", help="Back out implied volatility from a market price")
    p.add_argument("--market-price", type=float, required=True)
    for name in ("--spot", "--strike", "--expiry-years", "--rate"):
        p.add_argument(name, type=float, required=True)
    p.add_argument("--type", choices=["call", "put"], default="call")
    p.add_argument("--dividend", type=float, default=0.0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_options_implied_vol)

    p = sub.add_parser("options-compare", help="Compare a model price/vol against a real market price")
    p.add_argument("--market-price", type=float, required=True)
    for name in ("--spot", "--strike", "--expiry-years", "--rate", "--vol"):
        p.add_argument(name, type=float, required=True)
    p.add_argument("--type", choices=["call", "put"], default="call")
    p.add_argument("--dividend", type=float, default=0.0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_options_compare)

    p = sub.add_parser("order-book-replay", help="Replay a scripted sequence of orders through the matching engine")
    p.add_argument("--orders", required=True, help="JSON list of {side, type: limit|market, price?, quantity}")
    p.add_argument("--symbol", default="SYMBOL")
    p.add_argument("--depth-levels", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_order_book_replay)

    p = sub.add_parser("sentiment-fetch", help="Scrape and score financial headlines for a query")
    p.add_argument("--query", required=True)
    p.add_argument("--max-results", type=int, default=50)
    p.add_argument("--out-csv", default=None)
    p.set_defaults(func=cmd_sentiment_fetch)

    p = sub.add_parser("sentiment-correlate", help="Correlate scored headlines against price movement")
    p.add_argument("--sentiment-csv", required=True, help="Output of sentiment-fetch")
    p.add_argument("--price-csv", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_sentiment_correlate)

    p = sub.add_parser("portfolio-optimize", help="Markowitz mean-variance optimal allocation")
    _add_alpaca_args(p)
    p.add_argument("--tickers", required=True, help="Comma-separated tickers")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--timeframe", default="1Day")
    p.add_argument("--risk-level", type=float, required=True, help="0.0 (conservative) to 1.0 (aggressive)")
    p.add_argument("--risk-free-rate", type=float, default=0.0)
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_portfolio_optimize)

    p = sub.add_parser("vol-surface", help="Build and export an interactive implied-volatility surface")
    p.add_argument("--demo", action="store_true", help="Use a synthetic demo chain instead of live Alpaca options data")
    _add_alpaca_args(p)
    p.add_argument("--symbol", default=None, help="Underlying symbol (required unless --demo)")
    p.add_argument("--spot", type=float, default=100.0, help="Spot price for --demo mode")
    p.add_argument("--option-type", choices=["call", "put"], default="put")
    p.add_argument("--rate", type=float, default=0.04)
    p.add_argument("--title", default="Implied Volatility Surface")
    p.add_argument("--out", required=True, help="Output HTML file path")
    p.set_defaults(func=cmd_vol_surface)

    p = sub.add_parser("factor-model", help="Fama-French 3-factor decomposition of a price series")
    p.add_argument("--price-csv", required=True)
    p.add_argument("--frequency", choices=["daily", "monthly"], default="daily")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_factor_model)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001 -- CLI top level: report cleanly, don't dump a stack trace
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
