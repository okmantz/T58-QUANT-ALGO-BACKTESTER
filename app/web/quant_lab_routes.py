"""
Quant Lab web routes -- wires every backend module added across this
project's recent upgrade rounds (the Universal Strategy Translator, Auto
Regime Selector, Strategy Health monitor, Portfolio Composer, and the
whole app/quant_lab/ toolkit) into the mobile/desktop-browser web app as
real, usable pages.

Registered as a Flask Blueprint (`quant_lab_bp`) rather than added
directly to app/web/server.py's already-3,700-line module, so this whole
feature area lives in one file and app/web/server.py's own change is a
single import + one `app.register_blueprint(...)` line.

Every route is a THIN wrapper, same principle as cli.py's own docstring:
parse the form/upload, call straight into the already-tested library
function, render the result. No business logic lives here.

Every tool page shares one generic template (templates/quant_lab_tool.html)
that takes a form-fields fragment and a result fragment, both rendered as
plain HTML strings by the helper functions below -- one shared template
instead of one per tool, since the 12 tools' pages are structurally
identical (inputs in, one result panel out).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from flask import Blueprint, render_template, request

from app.data import alpaca_credentials

quant_lab_bp = Blueprint("quant_lab", __name__, url_prefix="/quant-lab")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _render(title: str, description: str, form_html: str, result_html: str | None = None, error: str | None = None):
    return render_template(
        "quant_lab_tool.html", title=title, description=description,
        form_html=form_html, result_html=result_html, error=error,
    )


def _field(name, label, kind="text", value="", placeholder="", extra=""):
    if kind == "checkbox":
        checked = "checked" if value else ""
        return (f'<div class="checkbox-row"><input type="checkbox" id="{name}" name="{name}" {checked}>'
                f'<label for="{name}" style="margin:0;">{label}</label></div>')
    return (f'<label for="{name}">{label}</label>'
            f'<input type="{kind}" id="{name}" name="{name}" value="{value}" placeholder="{placeholder}" {extra}>')


def _alpaca_fields() -> str:
    saved = alpaca_credentials.has_saved_credentials()
    note = "Using your saved Alpaca keys (Data tab)." if saved else "No saved Alpaca keys -- enter one below, or save one on the Data tab first."
    return (
        f'<p class="help">{note}</p>'
        + _field("alpaca_key", "Alpaca API key (optional if saved)", placeholder="leave blank to use saved key")
        + _field("alpaca_secret", "Alpaca API secret (optional if saved)", "text", placeholder="leave blank to use saved key")
    )


def _resolve_alpaca_keys(form) -> tuple[str, str]:
    key = (form.get("alpaca_key") or "").strip()
    secret = (form.get("alpaca_secret") or "").strip()
    if key and secret:
        return key, secret
    creds = alpaca_credentials.load_credentials()
    if creds is None or not creds.is_usable:
        raise ValueError("No Alpaca API key/secret given, and none saved on the Data tab.")
    return creds.api_key, creds.secret_key


def _save_upload_to_temp(file_storage, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    Path(path).write_bytes(file_storage.read())
    return path


def _load_ohlcv_upload(form_file_key: str):
    from app.data.importer import import_csv

    upload = request.files.get(form_file_key)
    if upload is None or not upload.filename:
        raise ValueError(f"Please choose a file for '{form_file_key}'.")
    tmp_path = _save_upload_to_temp(upload, Path(upload.filename).suffix or ".csv")
    result = import_csv(tmp_path)
    if result.dataframe is None:
        errors = "; ".join(i.message for i in result.issues if i.level == "error")
        raise ValueError(f"Could not load '{upload.filename}' as OHLCV data: {errors}")
    return result.dataframe


def _library_strategy_options(strategy_type: str | None = None) -> str:
    from app.strategy.library import list_saved_strategies

    options = []
    for s in list_saved_strategies(strategy_type=strategy_type, status="validated"):
        options.append(f'<option value="{s.name}">{s.name} ({s.strategy_type})</option>')
    if not options:
        return '<option value="">No validated Strategy Library entries found</option>'
    return "".join(options)


def _pre(text: str) -> str:
    import html
    return f'<pre class="result-summary">{html.escape(text)}</pre>'


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

_TOOLS = [
    {"url": "/quant-lab/translator", "badge": "Deploy", "title": "Universal Strategy Translator",
     "description": "Manual Strategy Builder config -> clean PineScript v5 / MQL5, ready for TradingView or a live MT5 account."},
    {"url": "/quant-lab/regime-selector", "badge": "Auto", "title": "Auto Regime Selector",
     "description": "Auto-assigns the best validated Strategy Library entry per market regime into a ready-to-run router."},
    {"url": "/quant-lab/strategy-health", "badge": "Monitor", "title": "Strategy Health / Drift Monitor",
     "description": "Compares a forward-test session's realized trades against its own predicted Monte Carlo distribution."},
    {"url": "/quant-lab/portfolio-composer", "badge": "Search", "title": "Automated Portfolio Composer",
     "description": "Searches the Strategy Library for the N-strategy combo maximizing combined eval-pass probability."},
    {"url": "/quant-lab/pairs-screen", "badge": "Stat Arb", "title": "Pairs Screener",
     "description": "Screens a symbol universe (via Alpaca) for correlated, mean-reverting pairs."},
    {"url": "/quant-lab/pairs-backtest", "badge": "Stat Arb", "title": "Pairs Backtest",
     "description": "End-to-end: fetch two instruments, build the z-score reversion strategy, run the real backtest."},
    {"url": "/quant-lab/options-pricing", "badge": "Options", "title": "Options Pricing Calculator",
     "description": "Black-Scholes price + all 5 Greeks, implied volatility, and market-price comparison, from scratch."},
    {"url": "/quant-lab/order-book", "badge": "Microstructure", "title": "Order Book Simulator",
     "description": "Replay a scripted sequence of orders through a real price-time-priority matching engine."},
    {"url": "/quant-lab/sentiment-price", "badge": "NLP", "title": "Sentiment-Price Correlation",
     "description": "Scrapes financial headlines, scores sentiment with a from-scratch lexicon model, correlates against price."},
    {"url": "/quant-lab/portfolio-optimizer", "badge": "Markowitz", "title": "Portfolio Optimizer",
     "description": "Mean-variance optimization: input tickers and a risk level, get the optimal allocation."},
    {"url": "/quant-lab/vol-surface", "badge": "Options", "title": "Volatility Surface",
     "description": "Builds and exports an interactive 3D implied-volatility surface across strikes and expirations."},
    {"url": "/quant-lab/factor-model", "badge": "Factors", "title": "Factor Model",
     "description": "Decomposes returns into market/size/value (Fama-French 3-factor) exposure and tests if alpha is real."},
]


@quant_lab_bp.route("/")
def index():
    return render_template("quant_lab_index.html", tools=_TOOLS)


# ---------------------------------------------------------------------------
# 1. Universal Strategy Translator
# ---------------------------------------------------------------------------

@quant_lab_bp.route("/translator", methods=["GET", "POST"])
def translator():
    from app.strategy.library import list_saved_strategies, load_strategy_text
    from app.strategy.translator import TranslationError, to_mql5, to_pinescript

    form_html = (
        '<label for="strategy_name">Manual Strategy Builder entry</label>'
        f'<select id="strategy_name" name="strategy_name">{_library_strategy_options("manual")}</select>'
        + _field("target", "Target language").replace(
            '<input type="text" id="target" name="target" value="" placeholder="">',
            '<select id="target" name="target"><option value="pinescript">PineScript v5</option>'
            '<option value="mql5">MQL5</option></select>',
        )
    )
    result_html, error = None, None
    if request.method == "POST":
        try:
            name = request.form["strategy_name"]
            target = request.form["target"]
            text = load_strategy_text("manual", name)
            config = json.loads(text)
            code = to_pinescript(config) if target == "pinescript" else to_mql5(config)
            result_html = _pre(code)
        except (TranslationError, json.JSONDecodeError, KeyError, FileNotFoundError) as exc:
            error = str(exc)
    return _render("Universal Strategy Translator",
                    "Convert a saved Manual Strategy Builder config into clean, standalone PineScript or MQL5.",
                    form_html, result_html, error)


# ---------------------------------------------------------------------------
# 2. Auto Regime Selector
# ---------------------------------------------------------------------------

@quant_lab_bp.route("/regime-selector", methods=["GET", "POST"])
def regime_selector():
    form_html = (
        '<label for="data_csv">OHLCV data (CSV)</label><input type="file" id="data_csv" name="data_csv" accept=".csv">'
        + _field("dimension", "Regime dimension").replace(
            '<input type="text" id="dimension" name="dimension" value="" placeholder="">',
            '<select id="dimension" name="dimension">'
            '<option value="trend">Trend</option><option value="volatility">Volatility</option>'
            '<option value="session">Session</option><option value="environment">Environment</option></select>',
        )
        + _field("min_trades", "Minimum trades per regime cell", "number", "20")
    )
    result_html, error = None, None
    if request.method == "POST":
        try:
            from app.strategy.auto_regime_selector import select_regime_strategies
            from app.strategy.library_loader import load_validated_candidates

            df = _load_ohlcv_upload("data_csv")
            candidates = load_validated_candidates()
            if not candidates:
                raise ValueError("No validated Strategy Library entries found to consider.")
            result = select_regime_strategies(
                df, candidates, request.form["dimension"], min_trades_per_cell=int(request.form["min_trades"]),
            )
            result_html = _pre(result.render_table())
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    return _render("Auto Regime Selector",
                    "Auto-assigns the best validated strategy per market regime into a ready-to-run router.",
                    form_html, result_html, error)


# ---------------------------------------------------------------------------
# 3. Strategy Health / Drift Monitor
# ---------------------------------------------------------------------------

@quant_lab_bp.route("/strategy-health", methods=["GET", "POST"])
def strategy_health():
    form_html = (
        '<label for="journal_db">Forward-test journal (.db file)</label><input type="file" id="journal_db" name="journal_db">'
        + _field("session_id", "Session ID", "number")
        + _field("strategy_label", "Strategy label", "text", "", "e.g. my_strategy.json")
        + '<label for="mc_result_json">Predicted Monte Carlo result (JSON)</label><input type="file" id="mc_result_json" name="mc_result_json" accept=".json">'
        + _field("account_balance", "Account balance", "number", "10000")
    )
    result_html, error = None, None
    if request.method == "POST":
        try:
            from app.forward_test.journal import ForwardTestJournal
            from app.monitoring.strategy_health import check_strategy_health
            from app.monte_carlo.engine import MonteCarloResult

            journal_upload = request.files.get("journal_db")
            mc_upload = request.files.get("mc_result_json")
            if not journal_upload or not journal_upload.filename:
                raise ValueError("Please choose a journal .db file.")
            if not mc_upload or not mc_upload.filename:
                raise ValueError("Please choose a Monte Carlo result JSON file.")
            journal_path = _save_upload_to_temp(journal_upload, ".db")
            mc_path = _save_upload_to_temp(mc_upload, ".json")
            with open(mc_path, "r", encoding="utf-8") as f:
                mc_data = json.load(f)
            predicted = MonteCarloResult(**mc_data)
            journal = ForwardTestJournal(db_path=Path(journal_path))
            result = check_strategy_health(
                journal, int(request.form["session_id"]), request.form["strategy_label"], predicted,
                account_balance=float(request.form["account_balance"]),
            )
            result_html = _pre(result.render_table())
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    return _render("Strategy Health / Drift Monitor",
                    "Compares a forward-test session's realized results against its strategy's predicted Monte Carlo distribution.",
                    form_html, result_html, error)


# ---------------------------------------------------------------------------
# 4. Automated Portfolio Composer
# ---------------------------------------------------------------------------

@quant_lab_bp.route("/portfolio-composer", methods=["GET", "POST"])
def portfolio_composer():
    from app.strategy.library import list_saved_strategies

    validated = list_saved_strategies(status="validated")
    checkboxes = "".join(
        f'<div class="checkbox-row"><input type="checkbox" name="strategy_names" value="{s.name}" id="cb_{i}">'
        f'<label for="cb_{i}" style="margin:0;">{s.name} ({s.strategy_type})</label></div>'
        for i, s in enumerate(validated)
    ) or '<p class="help">No validated Strategy Library entries found.</p>'
    form_html = (
        '<p class="help">Choose 2+ validated strategies to search combinations of (all will be backtested on the SAME uploaded data below).</p>'
        + checkboxes
        + '<label for="data_csv">Market data for every leg (CSV)</label><input type="file" id="data_csv" name="data_csv" accept=".csv">'
        + _field("min_legs", "Min legs", "number", "2") + _field("max_legs", "Max legs", "number", "4")
        + _field("max_evaluations", "Max evaluations", "number", "60")
    )
    result_html, error = None, None
    if request.method == "POST":
        try:
            from app.backtest.risk import RiskConfig
            from app.portfolio.composer import compose_portfolio
            from app.portfolio.portfolio import InstrumentLeg, PortfolioConfig
            from app.strategy.library_loader import load_strategy_object

            names = request.form.getlist("strategy_names")
            if len(names) < 2:
                raise ValueError("Choose at least 2 strategies.")
            df = _load_ohlcv_upload("data_csv")
            by_name = {s.name: s for s in validated}
            legs = [InstrumentLeg(name=n, df=df, strategy=load_strategy_object(by_name[n]), risk=RiskConfig()) for n in names]
            result = compose_portfolio(
                legs, min_legs=int(request.form["min_legs"]), max_legs=int(request.form["max_legs"]),
                max_evaluations=int(request.form["max_evaluations"]), portfolio_config=PortfolioConfig(),
            )
            result_html = _pre(result.render_table())
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    return _render("Automated Portfolio Composer",
                    "Searches your validated Strategy Library for the N-strategy combination that maximizes combined performance.",
                    form_html, result_html, error)


# ---------------------------------------------------------------------------
# 5/6. Pairs Trading
# ---------------------------------------------------------------------------

@quant_lab_bp.route("/pairs-screen", methods=["GET", "POST"])
def pairs_screen():
    form_html = (
        _alpaca_fields()
        + _field("symbols", "Symbols (comma-separated)", "text", "", "AAPL,MSFT,GOOG")
        + _field("timeframe", "Timeframe", "text", "1Day")
        + _field("start", "Start date", "text", "2023-01-01") + _field("end", "End date", "text", "2024-01-01")
        + _field("min_correlation", "Minimum correlation", "number", "0.7")
    )
    result_html, error = None, None
    if request.method == "POST":
        try:
            from app.quant_lab.pairs_trading import fetch_universe, screen_pairs

            api_key, secret_key = _resolve_alpaca_keys(request.form)
            symbols = [s.strip() for s in request.form["symbols"].split(",") if s.strip()]
            universe = fetch_universe(api_key, secret_key, symbols, request.form["timeframe"], request.form["start"], request.form["end"])
            candidates = screen_pairs(universe, min_correlation=float(request.form["min_correlation"]))
            summary = "\n".join(
                f"{c.symbol_a}/{c.symbol_b}: corr={c.correlation:.3f} adf={c.adf_statistic:+.2f} "
                f"({c.stationarity_verdict}) z={c.current_zscore:+.2f}" for c in candidates
            ) or "No pairs cleared the correlation threshold."
            result_html = _pre(summary)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    return _render("Pairs Screener", "Screens a symbol universe for correlated, mean-reverting pairs.", form_html, result_html, error)


@quant_lab_bp.route("/pairs-backtest", methods=["GET", "POST"])
def pairs_backtest():
    form_html = (
        _alpaca_fields()
        + _field("symbol_a", "Symbol A", "text", "", "AAPL") + _field("symbol_b", "Symbol B", "text", "", "MSFT")
        + _field("timeframe", "Timeframe", "text", "1Day")
        + _field("start", "Start date", "text", "2023-01-01") + _field("end", "End date", "text", "2024-01-01")
        + _field("entry_z", "Entry z-score", "number", "2.0") + _field("exit_z", "Exit z-score", "number", "0.5")
    )
    result_html, error = None, None
    if request.method == "POST":
        try:
            from app.backtest.risk import RiskConfig
            from app.quant_lab.pairs_trading import run_pairs_backtest

            api_key, secret_key = _resolve_alpaca_keys(request.form)
            result = run_pairs_backtest(
                api_key, secret_key, request.form["symbol_a"], request.form["symbol_b"],
                request.form["timeframe"], request.form["start"], request.form["end"],
                entry_z=float(request.form["entry_z"]), exit_z=float(request.form["exit_z"]),
            )
            stats = result.backtest.statistics
            summary = (
                (result.pair.stationarity_verdict if result.pair else "n/a") + "\n"
                f"Trades: {stats.total_trades}  Net profit: {stats.net_profit:+.2f}  Win rate: {stats.win_rate:.1f}%"
            )
            result_html = _pre(summary)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    return _render("Pairs Backtest", "Fetches two instruments, builds the mean-reversion strategy, and runs the real backtest engine.",
                    form_html, result_html, error)


# ---------------------------------------------------------------------------
# 7. Options Pricing Calculator
# ---------------------------------------------------------------------------

@quant_lab_bp.route("/options-pricing", methods=["GET", "POST"])
def options_pricing():
    form_html = (
        _field("spot", "Spot price", "number", "100") + _field("strike", "Strike", "number", "100")
        + _field("expiry_years", "Time to expiry (years)", "number", "1")
        + _field("rate", "Risk-free rate", "number", "0.05") + _field("vol", "Volatility", "number", "0.20")
        + _field("option_type", "Type").replace(
            '<input type="text" id="option_type" name="option_type" value="" placeholder="">',
            '<select id="option_type" name="option_type"><option value="call">Call</option><option value="put">Put</option></select>',
        )
        + _field("market_price", "Market price (optional, for comparison)", "number", "")
    )
    result_html, error = None, None
    if request.method == "POST":
        try:
            from app.quant_lab.options_pricing import black_scholes_greeks, black_scholes_price, compare_to_market

            spot, strike, T, r, vol = (float(request.form[k]) for k in ("spot", "strike", "expiry_years", "rate", "vol"))
            opt_type = request.form["option_type"]
            market_price = request.form.get("market_price", "").strip()
            if market_price:
                cmp = compare_to_market(float(market_price), spot, strike, T, r, vol, opt_type)
                result_html = _pre(cmp.render_summary())
            else:
                price = black_scholes_price(spot, strike, T, r, vol, opt_type)
                greeks = black_scholes_greeks(spot, strike, T, r, vol, opt_type)
                result_html = _pre(f"Price: {price:.4f}\n" + "\n".join(f"{k}: {v:+.4f}" for k, v in greeks.to_dict().items()))
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    return _render("Options Pricing Calculator", "Black-Scholes price, Greeks, implied volatility, and market comparison -- all from scratch.",
                    form_html, result_html, error)


# ---------------------------------------------------------------------------
# 8. Order Book Simulator
# ---------------------------------------------------------------------------

@quant_lab_bp.route("/order-book", methods=["GET", "POST"])
def order_book():
    form_html = (
        '<label for="orders_json">Orders (JSON list of {side, type, price?, quantity})</label>'
        '<textarea id="orders_json" name="orders_json" placeholder=\'[{"side":"buy","type":"limit","price":100,"quantity":10}]\'>'
        '[\n  {"side": "buy", "type": "limit", "price": 100.0, "quantity": 10},\n'
        '  {"side": "sell", "type": "limit", "price": 101.0, "quantity": 10},\n'
        '  {"side": "sell", "type": "limit", "price": 100.0, "quantity": 4},\n'
        '  {"side": "buy", "type": "market", "quantity": 20}\n]</textarea>'
    )
    result_html, error = None, None
    if request.method == "POST":
        try:
            from app.quant_lab.order_book import LimitOrderBook

            orders = json.loads(request.form["orders_json"])
            book = LimitOrderBook()
            lines = []
            all_trades = 0
            for order in orders:
                fn = book.submit_limit_order if order["type"] == "limit" else book.submit_market_order
                args = (order["side"], order["price"], order["quantity"]) if order["type"] == "limit" else (order["side"], order["quantity"])
                res = fn(*args)
                all_trades += len(res.trades)
                lines.append(f"{order} -> {len(res.trades)} trade(s), resting={res.resting}")
            lines.append(f"\nTotal trades printed: {all_trades}")
            lines.append(f"Best bid/ask: {book.best_bid()} / {book.best_ask()}   Spread: {book.spread()}")
            lines.append(f"Depth: {book.depth_snapshot()}")
            result_html = _pre("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    return _render("Order Book Simulator", "Replays a scripted sequence of orders through a real price-time-priority matching engine.",
                    form_html, result_html, error)


# ---------------------------------------------------------------------------
# 9. Sentiment-Price Correlation
# ---------------------------------------------------------------------------

@quant_lab_bp.route("/sentiment-price", methods=["GET", "POST"])
def sentiment_price():
    form_html = (
        _field("query", "Headline search query", "text", "", "AAPL Apple stock")
        + _field("max_results", "Max headlines", "number", "50")
        + '<label for="price_csv">Price data (CSV, optional -- correlates if given)</label><input type="file" id="price_csv" name="price_csv" accept=".csv">'
    )
    result_html, error = None, None
    if request.method == "POST":
        try:
            from app.quant_lab.sentiment_price import correlate_sentiment_with_price, fetch_headlines, score_headlines

            headlines = fetch_headlines(request.form["query"], max_results=int(request.form["max_results"]))
            sentiment_df = score_headlines(headlines)
            lines = [f"{row.label:<9} {row.sentiment:+.2f}  {row.title}" for row in sentiment_df.itertuples()]
            price_upload = request.files.get("price_csv")
            if price_upload and price_upload.filename:
                price_df = _load_ohlcv_upload("price_csv")
                corr = correlate_sentiment_with_price(sentiment_df, price_df)
                lines.append("")
                lines.append(corr.render_summary())
            result_html = _pre("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    return _render("Sentiment-Price Correlation", "Scrapes financial headlines, scores sentiment with a from-scratch lexicon model, and correlates against price.",
                    form_html, result_html, error)


# ---------------------------------------------------------------------------
# 10. Portfolio Optimizer (Markowitz)
# ---------------------------------------------------------------------------

@quant_lab_bp.route("/portfolio-optimizer", methods=["GET", "POST"])
def portfolio_optimizer():
    form_html = (
        _alpaca_fields()
        + _field("tickers", "Tickers (comma-separated)", "text", "", "AAPL,MSFT,GOOG")
        + _field("start", "Start date", "text", "2022-01-01") + _field("end", "End date", "text", "2024-01-01")
        + _field("risk_level", "Risk level (0.0 conservative -- 1.0 aggressive)", "number", "0.5")
        + _field("risk_free_rate", "Risk-free rate", "number", "0.02")
        + _field("long_only", "Long-only (no short positions)", "checkbox")
    )
    result_html, error = None, None
    if request.method == "POST":
        try:
            from app.quant_lab.portfolio_optimizer import fetch_and_build_inputs, optimize_for_risk_level

            api_key, secret_key = _resolve_alpaca_keys(request.form)
            tickers = [t.strip() for t in request.form["tickers"].split(",") if t.strip()]
            inputs = fetch_and_build_inputs(api_key, secret_key, tickers, request.form["start"], request.form["end"])
            allocation = optimize_for_risk_level(
                inputs, float(request.form["risk_level"]), risk_free_rate=float(request.form["risk_free_rate"]),
                long_only=bool(request.form.get("long_only")),
            )
            result_html = _pre(allocation.render_summary())
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    return _render("Portfolio Optimizer", "Markowitz mean-variance optimization -- input tickers and a risk level, get the optimal allocation.",
                    form_html, result_html, error)


# ---------------------------------------------------------------------------
# 11. Volatility Surface
# ---------------------------------------------------------------------------

@quant_lab_bp.route("/vol-surface", methods=["GET", "POST"])
def vol_surface():
    form_html = (
        _field("demo", "Use synthetic demo chain (no Alpaca options entitlement needed)", "checkbox", True)
        + _alpaca_fields()
        + _field("symbol", "Underlying symbol (if not using demo)", "text", "", "AAPL")
        + _field("spot", "Spot price (for demo mode)", "number", "100")
        + _field("rate", "Risk-free rate", "number", "0.04")
    )
    result_html, error = None, None
    if request.method == "POST":
        try:
            from app.quant_lab.vol_surface import build_iv_surface, fetch_option_chain, render_surface_html, synthetic_demo_chain

            if request.form.get("demo"):
                chain = synthetic_demo_chain(spot=float(request.form["spot"]))
            else:
                api_key, secret_key = _resolve_alpaca_keys(request.form)
                chain = fetch_option_chain(api_key, secret_key, request.form["symbol"])
            surface = build_iv_surface(chain, r=float(request.form["rate"]))
            html = render_surface_html(surface, title="Implied Volatility Surface")
            result_html = f'<iframe srcdoc="{html.replace(chr(34), "&quot;")}" style="width:100%;height:600px;border:0;border-radius:8px;"></iframe>'
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    return _render("Volatility Surface", "Builds an interactive 3D implied-volatility surface across strikes and expirations.",
                    form_html, result_html, error)


# ---------------------------------------------------------------------------
# 12. Factor Model
# ---------------------------------------------------------------------------

@quant_lab_bp.route("/factor-model", methods=["GET", "POST"])
def factor_model():
    form_html = (
        '<label for="price_csv">Price data (CSV)</label><input type="file" id="price_csv" name="price_csv" accept=".csv">'
        + _field("frequency", "Factor data frequency").replace(
            '<input type="text" id="frequency" name="frequency" value="" placeholder="">',
            '<select id="frequency" name="frequency"><option value="daily">Daily</option><option value="monthly">Monthly</option></select>',
        )
    )
    result_html, error = None, None
    if request.method == "POST":
        try:
            from app.quant_lab.factor_model import compute_factor_exposures, compute_returns_from_prices, fetch_fama_french_factors

            price_df = _load_ohlcv_upload("price_csv")
            returns = compute_returns_from_prices(price_df)
            frequency = request.form["frequency"]
            factors = fetch_fama_french_factors(frequency=frequency)
            result = compute_factor_exposures(returns, factors, periods_per_year=252 if frequency == "daily" else 12)
            result_html = _pre(result.render_summary())
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    return _render("Factor Model", "Fama-French 3-factor decomposition -- tests whether apparent outperformance is real alpha or just factor exposure.",
                    form_html, result_html, error)
