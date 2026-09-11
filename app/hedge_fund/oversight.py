"""
Oversight desk -- reads the finished run and writes a journal entry.
Nothing else.

The article is explicit about why the LLM layer never touches an order:
Alpha Arena gave six frontier LLMs $10,000 of real money each to trade
crypto perpetuals for 17 days, and four of six lost money (GPT-5 -62.7%,
Gemini 2.5 Pro -56.7%, Claude Sonnet 4.5 -30.8%). This module is built so
that outcome is structurally impossible here, not just discouraged by
convention:

    NO function in this file accepts a broker/execution handle, an order
    factory, or a mutable holdings/weights object it could edit. Every
    function below takes an ALREADY-FINISHED HedgeFundBacktestResult (or
    a single already-decided RebalanceCycle) and returns a string. There
    is no code path from this module back into rebalancer.py's order
    generation. If a future change to this file ever imports
    `weights_to_orders` or anything from app.backtest.execution, that is
    a violation of this module's one job and should be reverted, not
    reviewed as a feature.

Uses this app's own existing local-Ollama client
(app.ai.trading_assistant.TradingAssistantClient) rather than a new LLM
integration -- same model, same settings, same "AI Assistant" tab the
rest of this app already configures. If Ollama isn't configured/running,
`generate_journal` degrades to a deterministic templated summary rather
than failing the whole run, matching this app's existing AI-optional
philosophy (e.g. app.ai.trading_assistant.build_deterministic_outlook).
"""
from __future__ import annotations

from app.ai.ollama_settings import OllamaSettings, load_settings
from app.ai.trading_assistant import TradingAssistantClient
from app.hedge_fund.rebalancer import HedgeFundBacktestResult, RebalanceConfig

SYSTEM_PROMPT = (
    "You are the risk-oversight analyst for a systematic rebalancing book. You are read-only: "
    "you never suggest specific position changes, never tell the trader what to do next, and never "
    "speak as if you can place or veto a trade. Your only job is to summarize what already happened -- "
    "which views drove the largest trades, how concentrated the book got, whether drawdown or turnover "
    "looks unusual -- in plain English, in one short paragraph plus up to 4 bullet points. Do not invent "
    "numbers that were not given to you."
)


def _cycle_summary_lines(result: HedgeFundBacktestResult, max_cycles: int = 6) -> list[str]:
    lines = []
    for cycle in result.cycles[-max_cycles:]:
        top_moves = sorted(cycle.target_weights.items(), key=lambda kv: -abs(kv[1]))[:3]
        weight_str = ", ".join(f"{a}={w:+.1%}" for a, w in top_moves)
        view_str = "; ".join(f"{a}: {v.detail}" for a, v in cycle.views.items())
        lines.append(
            f"{cycle.timestamp.date()}: turnover {cycle.turnover:.1%}, top weights [{weight_str}]. Views -- {view_str}"
        )
    return lines


def build_deterministic_summary(result: HedgeFundBacktestResult, config: RebalanceConfig) -> str:
    """No-LLM fallback: a templated summary built entirely from numbers
    already in `result`, no invention."""
    s = result.stats
    lines = [
        f"Ran {s.num_rebalances} rebalance cycles every {config.rebalance_every_bars} bars "
        f"at confidence={config.confidence:.2f}.",
        f"Total return {s.total_return_pct:+.1f}% (CAGR {s.cagr_pct:+.1f}%), max drawdown {s.max_drawdown_pct:.1f}%, "
        f"Sharpe {s.sharpe_ratio:.2f}.",
        f"Average turnover per cycle {s.avg_turnover_pct:.1f}%, total transaction costs ${s.total_transaction_costs:,.2f}.",
    ]
    if result.warnings:
        lines.append(f"{len(result.warnings)} cycle(s) were skipped -- see run warnings for detail.")
    lines.append("Most recent cycles:")
    lines.extend(f"  - {line}" for line in _cycle_summary_lines(result))
    return "\n".join(lines)


def generate_journal(
    result: HedgeFundBacktestResult, config: RebalanceConfig, settings: OllamaSettings | None = None,
) -> tuple[str, str | None]:
    """Returns (journal_text, note). `note` is set (not raised) when the
    LLM path wasn't used, so callers can surface it as an info line
    rather than an error -- the deterministic summary is a complete,
    correct journal entry on its own, just less readable."""
    deterministic = build_deterministic_summary(result, config)
    settings = settings or load_settings()
    if not settings.is_usable:
        return deterministic, "Ollama isn't configured -- showing the deterministic summary instead (AI Assistant tab has setup)."

    client = TradingAssistantClient(settings)
    user_message = (
        "Here is a finished hedge-fund-manager rebalance backtest run. Summarize it for a risk log:\n\n"
        + deterministic
    )
    reply, error = client._chat(SYSTEM_PROMPT, user_message)
    if error or not reply.strip():
        return deterministic, f"Ollama journal generation failed ({error or 'empty reply'}) -- showing the deterministic summary instead."
    return reply.strip(), None
