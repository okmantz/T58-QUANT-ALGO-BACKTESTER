"""
Single entry point for the Hedge Fund Manager tab: Research -> Portfolio
-> Execution -> Oversight, run end to end over an uploaded/stored
multi-asset dataset. Everything here is a thin orchestrator over the four
desk modules -- no business logic lives in this file, same principle
cli.py's own docstring states for the rest of this app.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.ai.ollama_settings import OllamaSettings
from app.hedge_fund.black_litterman import BlackLittermanError, confidence_sweep, views_to_posterior
from app.hedge_fund.oversight import generate_journal
from app.hedge_fund.rebalancer import HedgeFundBacktestResult, RebalanceConfig, RebalanceError, run_rebalance_backtest
from app.hedge_fund.research import ResearchError, generate_views
from app.quant_lab.portfolio_optimizer import PortfolioOptimizerError, build_inputs_from_prices
from app.strategy.base import Strategy


class HedgeFundManagerError(Exception):
    """Raised when the pipeline cannot produce a result at all."""


@dataclass
class HedgeFundManagerResult:
    backtest: HedgeFundBacktestResult
    journal: str
    journal_note: str | None
    confidence_sweep_diagnostic: list[dict] | None


def _diagnostic_confidence_sweep(
    price_data: dict[str, pd.DataFrame], config: RebalanceConfig, strategies: dict[str, Strategy] | None,
) -> list[dict] | None:
    """A one-shot snapshot (using the FULL window as the 'as of' cutoff,
    i.e. not walk-forward-safe -- diagnostic/illustrative only, same
    caveat as any single end-of-data view) reproducing the article's own
    "weights collapse to the benchmark as confidence falls" chart, so the
    UI can show the same safety-valve visualization for THIS book instead
    of the article's SPY/QQQ/TLT/GLD example.
    """
    try:
        views = generate_views(price_data, config.forecast, strategies)
        inputs = build_inputs_from_prices(price_data)
        return confidence_sweep(inputs, views, long_only=config.long_only)
    except (ResearchError, PortfolioOptimizerError, BlackLittermanError):
        return None


def run_hedge_fund_manager(
    price_data: dict[str, pd.DataFrame],
    config: RebalanceConfig,
    strategies: dict[str, Strategy] | None = None,
    write_journal: bool = True,
    ollama_settings: OllamaSettings | None = None,
    include_confidence_sweep: bool = True,
) -> HedgeFundManagerResult:
    try:
        backtest = run_rebalance_backtest(price_data, config, strategies)
    except RebalanceError as exc:
        raise HedgeFundManagerError(str(exc)) from exc

    journal, note = ("", None)
    if write_journal:
        journal, note = generate_journal(backtest, config, ollama_settings)

    sweep = _diagnostic_confidence_sweep(price_data, config, strategies) if include_confidence_sweep else None

    return HedgeFundManagerResult(backtest=backtest, journal=journal, journal_note=note, confidence_sweep_diagnostic=sweep)
