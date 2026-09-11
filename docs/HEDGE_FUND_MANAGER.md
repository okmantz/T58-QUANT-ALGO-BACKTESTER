# Hedge Fund Manager

New tab: Research -> Portfolio -> Execution -> Oversight, run end to end
against your own uploaded/stored market data. Commissioned from the
QuantFrame article "The Open-Source Hedge Fund Stack: Four Repos, Four
Desks" (Kronos / skfolio / NautilusTrader / Vibe-Trading) -- rebuilt as
part of this app rather than as four new dependencies.

## Why not just install the four repos

- **Kronos**: an independent test scored it at a Brier of 0.189 on
  5-minute BTC against 0.188 for a plain Brownian-motion baseline --
  statistically indistinguishable from a coin flip. No pip package
  either (repo clone only), which fights this app's "one .exe" goal.
  We kept the *lesson* (sample many forecasts, use the dispersion as
  confidence) and dropped the unproven model.
- **skfolio**: genuinely good, BSD-licensed, but this app already ships
  a closed-form Markowitz optimizer from scratch
  (`app/quant_lab/portfolio_optimizer.py`) in the exact style the rest of
  this codebase uses. Taking a new dependency to re-derive math this app
  already has would be the wrong kind of "reuse."
- **NautilusTrader**: a real production-grade engine, but this app's own
  job (per its README) is backtesting against prop-firm rules, not
  running a live multi-venue execution engine. The one piece worth
  taking -- "diff target weights against current holdings and submit the
  delta" -- is now `app/hedge_fund/rebalancer.py::weights_to_orders`.
- **Vibe-Trading**: the MCP-tool-server architecture is real and good,
  but this app already has a working local-Ollama chat client
  (`app.ai.trading_assistant.TradingAssistantClient`) wired to a
  configured model. Reused directly rather than standing up a second LLM
  integration.

## The four desks, in this codebase

| Desk | Module | Reuses |
|---|---|---|
| Research | `app/hedge_fund/research.py` | `app.strategy.base.Strategy` (any saved strategy can BE a view generator) |
| Portfolio | `app/hedge_fund/black_litterman.py` | `app.quant_lab.portfolio_optimizer` (covariance estimation, closed-form tangency portfolio) |
| Execution | `app/hedge_fund/rebalancer.py` | Nothing forced through `app.backtest.engine` -- different trade model, same reasoning `app/portfolio/portfolio.py` already gives for building its own combine logic |
| Oversight | `app/hedge_fund/oversight.py` | `app.ai.trading_assistant.TradingAssistantClient`, `app.ai.ollama_settings` |

`app/hedge_fund/pipeline.py::run_hedge_fund_manager` wires all four into
one call. The web tab (`/hedge-fund`, `app/web/hedge_fund_routes.py` +
`app/web/templates/hedge_fund.html`) is a thin wrapper over that.

## The safety valve

The most important number in the whole pipeline is the confidence dial
(`RebalanceConfig.confidence`, 0.0-1.0), which maps to the
Black-Litterman Omega term via `confidence_to_omega`. At 1.0 the
optimizer trusts the research desk's own measured view dispersion
exactly; near 0 it ignores the view entirely and the posterior falls
back to an equal-weight equilibrium. `black_litterman.confidence_sweep`
reproduces the article's own diagnostic chart for this, and the web tab
renders it as a table under "The safety valve."

## Known, honestly-scoped limitations

- **No live/paper broker connection.** This runs entirely against
  historical data you upload -- there is no path from this tab to a real
  or demo account. If that's ever wanted, it's a new, separate, and much
  more carefully reviewed piece of work, not an extension of this one.
- **Web form always uses the bootstrap view generator.** The
  strategy-signal view generator (`research.strategy_signal_view`) is
  fully implemented and tested, but the per-asset "use one of my saved
  strategies" picker UI (the same pattern `portfolio.html` already has
  for its legs) isn't wired into `hedge_fund.html` yet. Usable today from
  Python: `run_hedge_fund_manager(price_data, config,
  strategies={"AAPL": my_strategy})`.
- **Market-cap weights default to equal-weight.** This app has no
  market-cap data source, so the Black-Litterman equilibrium prior uses
  an equal-weight proxy -- a documented approximation, same spirit as
  this app's other "optional in spirit" fallbacks.
- **No posterior-covariance update.** `black_litterman_posterior` returns
  posterior mu only; risk uses the original sample covariance. Standard
  simplification -- doesn't change which asset a view favors, only
  slightly understates post-view uncertainty.
- **Turnover cap doesn't (and shouldn't) bind on the very first
  allocation out of cash** -- see the comment in
  `black_litterman.solve_posterior_weights` for why that's a real
  mathematical infeasibility, not a bug, and how it's handled.
- **No desktop (Tkinter) tab yet.** `main_window.py` is a 14,000+ line
  hand-built Tkinter UI with its own panel/threading conventions; adding
  a matching tab there is a substantial, separate follow-up, not
  attempted in this round so it could get the same care as the rest of
  that file's tabs rather than a rushed copy.
- **Rebalance stats are computed directly from the equity curve**, not
  via `app.backtest.statistics.compute_statistics` -- that function
  expects discrete long/flat/short `Trade` objects, which this
  target-weight rebalancing book never produces.

## Files

New:
- `app/hedge_fund/__init__.py`
- `app/hedge_fund/research.py`
- `app/hedge_fund/black_litterman.py`
- `app/hedge_fund/rebalancer.py`
- `app/hedge_fund/oversight.py`
- `app/hedge_fund/pipeline.py`
- `app/web/hedge_fund_routes.py`
- `app/web/templates/hedge_fund.html`
- `tests/test_hedge_fund_research.py`
- `tests/test_hedge_fund_black_litterman.py`
- `tests/test_hedge_fund_rebalancer.py`
- `tests/test_hedge_fund_oversight.py`

Changed:
- `app/web/server.py` (2 lines: import + `register_blueprint`)
- `app/web/templates/_sidebar.html` (1 new nav section)
