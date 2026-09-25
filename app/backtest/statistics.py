"""
Backtest statistics.

Computes the full statistics set required by the product spec: returns,
win/loss, risk, strategy-quality, and risk-adjusted metrics, from a trade
list and an equity curve.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from app.backtest.execution import Trade


@dataclass
class BacktestStatistics:
    net_profit: float
    gross_profit: float
    gross_loss: float
    return_pct: float
    average_trade: float

    win_rate: float
    loss_rate: float
    average_winner: float
    average_loser: float
    largest_winner: float
    largest_loser: float

    max_drawdown: float
    max_drawdown_pct: float
    average_drawdown_pct: float
    max_daily_drawdown_pct: float
    max_weekly_drawdown_pct: float
    max_losing_streak: int
    max_winning_streak: int

    profit_factor: float
    expectancy: float
    average_r: float
    risk_reward: float

    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    total_trades: int
    account_reset_count: int = 0
    # Number of times reset_on_breach caused the raw backtest to
    # mechanically "buy a new account" mid-run (see RiskConfig.
    # reset_on_breach / app.backtest.execution.run_execution). 0 for any
    # backtest that never used reset_on_breach -- byte-identical default,
    # purely additive.

    is_reset_chain: bool = False
    final_segment_net_profit: float = 0.0
    final_segment_trade_count: int = 0
    # FIX (RESET-ACCT-001): `net_profit` above is a straight sum of every
    # trade's pnl across the WHOLE run -- when account_reset_count > 0,
    # that pools together the P&L of account_reset_count+1 DIFFERENT
    # simulated accounts (see RiskConfig.reset_on_breach) into one number,
    # which reads like one account's result but isn't. `final_segment_
    # net_profit`/`final_segment_trade_count` describe ONLY the last
    # (currently active) simulated account -- the one still standing at
    # the end of the run -- so a report can show "this account's own P&L"
    # right next to "cumulative P&L across every account this chain burned
    # through" instead of conflating the two under a single "Net Profit"
    # label. Both are 0/equal-to-net_profit's own trades whenever
    # account_reset_count is 0 (no reset ever happened), so this changes
    # nothing for the vast majority of runs that don't use reset_on_breach.

    avg_intended_risk_dollars: float = 0.0
    avg_actual_stop_risk_dollars: float = 0.0
    avg_realized_loss_on_losers: float = 0.0
    pct_trades_position_capped: float = 0.0
    pct_trades_risk_overshoot: float = 0.0
    # ADAPTIVE-RISK-ATTRIBUTION (2026-09-24): pct_trades_position_capped
    # above previously left "why" as an unresolved three-way guess (max-
    # position-size cap? adaptive-risk throttle? whole-contract rounding?)
    # for anyone reading the report -- these two fields answer the
    # "adaptive-risk throttle" branch of that guess directly from the
    # trades themselves (Trade.adaptive_risk_multiplier / .adaptive_risk_
    # rules_active), instead of leaving it as a hedge. Both are exactly
    # 1.0 / 0.0 whenever no AdaptiveRiskConfig was passed to run_backtest
    # at all (every existing report/test that never used this feature is
    # unaffected).
    avg_adaptive_risk_multiplier: float = 1.0
    pct_trades_adaptive_throttle_active: float = 0.0
    # RISK-RECON: reconciles "how much you told the system you're willing
    # to risk" (RiskConfig.risk_value, e.g. 0.5% of a $50k account = a
    # $250 target -- see Trade.intended_risk_dollars) against what actually
    # happened to that risk in two different ways:
    #   avg_intended_risk_dollars      -- mean of the raw %-of-equity target,
    #                                     BEFORE any max_position_size cap
    #                                     or adaptive-risk throttle.
    #   avg_actual_stop_risk_dollars   -- mean of initial_risk * size, i.e.
    #                                     what a clean stop-out would have
    #                                     cost given the size actually taken
    #                                     (reflects any cap/throttle shrink).
    #   pct_trades_position_capped     -- % of trades where actual stop risk
    #                                     came in materially BELOW the
    #                                     intended target (a cap/throttle
    #                                     engaged) -- this is "the strategy
    #                                     calls for less" than the risk %
    #                                     setting alone would suggest.
    #   avg_realized_loss_on_losers    -- mean realized $ loss on trades that
    #                                     actually lost (same value as
    #                                     abs(average_loser) -- included here
    #                                     so all three numbers sit together).
    #   pct_trades_risk_overshoot      -- % of trades whose REALIZED loss
    #                                     exceeded their own actual stop risk
    #                                     by a material margin (gap-through
    #                                     fills -- see execution.py's
    #                                     gap_loss_count warning) -- i.e. the
    #                                     opposite failure mode, where a
    #                                     trade risked MORE than intended.
    # All 0.0 for a backtest with no trades carrying initial_risk/
    # intended_risk_dollars (e.g. a strategy that defines no stop at all).

    def to_dict(self) -> dict:
        return asdict(self)


def _max_streak(bools: list[bool]) -> int:
    best = cur = 0
    for b in bools:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def _reset_segment_ids(equity_df: pd.DataFrame) -> np.ndarray | None:
    """Integer "which simulated account is this bar part of" id per row of
    `equity_df`, incrementing by one at every account-reset event recorded
    in equity_df.attrs["account_reset_events"] (see
    app.backtest.execution.run_execution / RiskConfig.reset_on_breach).
    Returns None when there are no reset events at all -- callers should
    then fall back to treating the whole curve as a single segment, which
    is exactly today's (pre-reset-on-breach) behavior, unchanged.

    FIX (2026-09-18): a running-max/cummax computed straight across a
    reset would treat the OLD account's peak equity as still being the
    high-water mark for the BRAND NEW account that started fresh at
    initial_balance right after it -- reporting a "drawdown" that is
    really just (old account's peak) - (new account's near-initial-
    balance start), which has nothing to do with how much either actual
    account itself ever drew down. Left unfixed, turning reset_on_breach
    on would have made every drawdown-derived number (max_drawdown_pct,
    average_drawdown_pct, max_daily/weekly_drawdown_pct, calmar_ratio, and
    anything downstream that reads them, e.g. Monte Carlo's risk-of-ruin)
    wildly and misleadingly overstated -- the opposite of what
    reset_on_breach is supposed to model. Segmenting the cummax at each
    reset is the fix: each simulated account's drawdown is measured only
    against its OWN peak.
    """
    events = equity_df.attrs.get("account_reset_events")
    if not events:
        return None
    ts = equity_df["timestamp"]
    seg_id = np.zeros(len(ts), dtype=np.int64)
    for ev in events:
        seg_id += (ts >= ev["reset_at"]).to_numpy().astype(np.int64)
    return seg_id


def _drawdown_series(equity: pd.Series, seg_id: np.ndarray | None = None) -> pd.Series:
    running_max = equity.cummax() if seg_id is None else equity.groupby(seg_id).cummax()
    dd = (equity - running_max) / running_max.replace(0, np.nan)
    return dd.fillna(0.0)


def _periodic_max_drawdown(equity_df: pd.DataFrame, freq: str, seg_id: np.ndarray | None = None) -> float:
    """Worst intra-period drawdown (e.g. worst single day, worst single
    week) across the whole equity curve.

    UPGRADE (2026-09-03, speed): this used to build ``df.resample(freq)``
    and then loop over every resulting period in plain Python, calling
    ``_drawdown_series`` (itself a vectorized-but-small pandas computation)
    once per period. On a year or more of intraday data that's 250-1500+
    tiny pandas calls instead of one -- profiling a real backtest showed
    this function (called twice per run: once for daily, once for weekly)
    as the single largest cost in statistics.py, ahead of everything else
    computed from the trade list combined. Replaced with one groupby-
    cummax pass across the ENTIRE series at once (pandas' own vectorized,
    compiled implementation, not a Python loop calling it 1500 times) --
    same math, computed once instead of once per period.
    """
    df = equity_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    equity = df["equity"]
    if equity.empty:
        return 0.0
    # VAL-006 side-effect fix: equity.index is now correctly tz-aware
    # whenever the source data was (see app.backtest.execution's
    # _restore_tz), but pandas' PeriodIndex has no concept of timezone at
    # all -- .to_period() on a tz-aware index works fine but emits a
    # UserWarning every single call (and this runs once per backtest,
    # so a GA search/Search Lab run could emit it thousands of times).
    # Stripping the tz label (NOT converting to UTC first) keeps the
    # exact same displayed wall-clock time, which is what should
    # determine which calendar period a bar belongs to -- this is a
    # silence-the-warning fix, not a behavior change.
    index_for_period = equity.index
    if getattr(index_for_period, "tz", None) is not None:
        index_for_period = index_for_period.tz_localize(None)
    period_key = index_for_period.to_period(freq)
    group_key = period_key if seg_id is None else [seg_id, period_key]
    running_max = equity.groupby(group_key).cummax()
    dd = (equity - running_max) / running_max.replace(0, np.nan)
    dd = dd.fillna(0.0)
    worst = dd.groupby(group_key).min().min()
    return abs(worst) * 100.0 if pd.notna(worst) else 0.0


def _trade_reset_segment_ids(trades: list[Trade], equity_df: pd.DataFrame) -> np.ndarray | None:
    """Integer "which simulated account is this TRADE part of" id, one per
    entry in `trades` -- the trade-level counterpart to _reset_segment_ids
    above (which operates on the bar-level equity curve). Returns None
    when there are no reset events (the common case), so callers fall
    back to treating every trade as one account, unchanged from before
    this existed.

    A trade's exit_time strictly AFTER a reset's timestamp belongs to the
    new (post-reset) account; a trade exiting exactly AT the reset instant
    is the forced-close that caused the reset (see app.backtest.execution's
    reset_on_breach handling) and so still belongs to the OLD, dying
    account -- hence the strict `>` rather than `>=` used for the bar-level
    equity curve (there, the reset bar's own equity has already been reset
    to initial_balance, so `>=` is correct for that series; a trade is a
    discrete event that either caused the reset or came after it, not a
    continuously-updated row that IS the reset)."""
    events = equity_df.attrs.get("account_reset_events") if equity_df is not None else None
    if not events or not trades:
        return None
    reset_ats = [ev["reset_at"] for ev in events]
    seg_id = np.zeros(len(trades), dtype=np.int64)
    exit_times = pd.to_datetime([t.exit_time for t in trades])
    for reset_at in reset_ats:
        seg_id += (exit_times > pd.Timestamp(reset_at)).astype(np.int64)
    return seg_id


def compute_risk_reconciliation(trades: list[Trade]) -> dict:
    """Reconciles "how much you're willing to risk" (RiskConfig.risk_value,
    captured per-trade as Trade.intended_risk_dollars at sizing time)
    against what a trade's actual configured stop would have cost given
    the size it was actually sized to (Trade.initial_risk * Trade.size),
    and against what it actually realized if it lost. Returns the raw
    dict compute_statistics folds into BacktestStatistics; also usable
    standalone (e.g. for a dedicated report table) since it needs nothing
    but a trade list.

    A position-size cap (RiskConfig.max_position_size) or an adaptive-risk
    throttle can size a trade below its intended target -- "the strategy
    calls for less" than the risk % setting alone would suggest -- which
    shows up here as actual_stop_risk < intended_risk on a meaningful
    share of trades. A gap-through fill shows up as the REALIZED loss
    exceeding that same trade's own actual stop risk -- the opposite
    direction (risking more than intended), tracked separately.
    """
    intended = [t.intended_risk_dollars for t in trades if t.intended_risk_dollars]
    pairs = [
        (t.intended_risk_dollars, t.initial_risk * t.size)
        for t in trades
        if t.intended_risk_dollars and t.initial_risk and t.size
    ]
    losers = [t.pnl for t in trades if t.pnl < 0]

    avg_intended = float(np.mean(intended)) if intended else 0.0
    avg_actual_stop = float(np.mean([p[1] for p in pairs])) if pairs else 0.0
    avg_realized_loss = float(abs(np.mean(losers))) if losers else 0.0

    # "Materially" capped/overshot -- a >1% difference, to avoid flagging
    # ordinary floating-point noise as a cap/overshoot event.
    capped = [p for p in pairs if p[1] < p[0] * 0.99]
    pct_capped = float(len(capped) / len(pairs) * 100) if pairs else 0.0

    overshoot_flags = [
        abs(t.pnl) > (t.initial_risk * t.size) * 1.01
        for t in trades
        if t.pnl < 0 and t.initial_risk and t.size
    ]
    pct_overshoot = float(sum(overshoot_flags) / len(overshoot_flags) * 100) if overshoot_flags else 0.0

    # ADAPTIVE-RISK-ATTRIBUTION: direct evidence (not a guess) of how much
    # of any "sized BELOW target" gap above is explained by the
    # adaptive-risk throttle specifically -- see Trade.adaptive_risk_
    # multiplier's own docstring. Both default to "throttle wasn't a
    # factor" (1.0 / 0.0%) when no AdaptiveRiskConfig was active.
    adaptive_multipliers = [t.adaptive_risk_multiplier for t in trades if t.adaptive_risk_multiplier is not None]
    avg_adaptive_multiplier = float(np.mean(adaptive_multipliers)) if adaptive_multipliers else 1.0
    adaptive_active_flags = [bool(t.adaptive_risk_rules_active) for t in trades]
    pct_adaptive_active = float(sum(adaptive_active_flags) / len(adaptive_active_flags) * 100) if adaptive_active_flags else 0.0

    return {
        "avg_intended_risk_dollars": avg_intended,
        "avg_actual_stop_risk_dollars": avg_actual_stop,
        "avg_realized_loss_on_losers": avg_realized_loss,
        "pct_trades_position_capped": pct_capped,
        "pct_trades_risk_overshoot": pct_overshoot,
        "avg_adaptive_risk_multiplier": avg_adaptive_multiplier,
        "pct_trades_adaptive_throttle_active": pct_adaptive_active,
    }


def format_run_summary_line(
    label: str,
    n_trades: int,
    stats: "BacktestStatistics",
    chain_eval_pass_pct: float,
    chain_payout_pct: float,
    per_attempt_eval_pass_pct: float | None = None,
    per_attempt_payout_pct: float | None = None,
    total_attempts: int | None = None,
) -> str:
    """Shared "Baseline: N trades, net $X, eval pass Y%, payout Z%." console
    line built by every orchestration tool (Full Pipeline, Quick Optimize)
    -- one copy so a fix to how this line reads never has to be applied in
    two (or more) places and drift.

    FIX (RESET-ACCT-002/RESET-ACCT-001): when this run used
    reset_on_breach (stats.account_reset_count > 0), the previous version
    of this line printed `stats.net_profit` (cumulative P&L pooled across
    every simulated account the reset chain burned through -- see
    BacktestStatistics.is_reset_chain's docstring) labeled as plain "net
    $X", and printed `chain_eval_pass_pct`/`chain_payout_pct` (whether
    >=1 attempt anywhere in a, possibly hundreds-long, reset chain ever
    passed/paid out -- see MonteCarloResult.per_attempt_pass_probability's
    docstring) labeled as plain "eval pass Y%" -- both read like ordinary
    single-account numbers but weren't. This still prints those same
    chain-level numbers (some callers/tests read them), but ONLY when a
    reset chain was actually used does it also print the two numbers that
    actually answer "what happens to ONE account attempt": the final
    (currently-standing) account's own P&L, and the true per-attempt
    pass/payout rate pooled across every independent attempt in the
    chain. When account_reset_count is 0 (the default, non-reset case)
    this is byte-identical to the line before this fix existed.
    """
    line = (
        f"{label}: {n_trades} trades, net ${stats.net_profit:,.2f}, "
        f"eval pass {chain_eval_pass_pct:.1f}%, payout {chain_payout_pct:.1f}%."
    )
    return line + reset_chain_note(
        stats, chain_eval_pass_pct, chain_payout_pct,
        per_attempt_eval_pass_pct, per_attempt_payout_pct, total_attempts,
    )


def reset_chain_note(
    stats: "BacktestStatistics",
    chain_eval_pass_pct: float,
    chain_payout_pct: float,
    per_attempt_eval_pass_pct: float | None = None,
    per_attempt_payout_pct: float | None = None,
    total_attempts: int | None = None,
) -> str:
    """The bracketed reset-chain clarification appended by
    format_run_summary_line above -- split out separately so any console
    line that reports net_profit/eval-pass/payout with its own custom
    formatting (e.g. one that also prints win rate) can append the exact
    same clarification without duplicating its wording. Returns "" (no
    change to the line at all) whenever stats.account_reset_count is 0 --
    the default, non-reset case."""
    if stats.account_reset_count <= 0:
        return ""
    note = (
        f" [reset_on_breach: {stats.account_reset_count} account reset(s) occurred -- "
        f"net ${stats.net_profit:,.2f} above is CUMULATIVE P&L across "
        f"{stats.account_reset_count + 1} simulated accounts, not one account's result; "
        f"the CURRENT (final) account's own P&L is ${stats.final_segment_net_profit:,.2f} "
        f"over its {stats.final_segment_trade_count} trades. "
        f"'eval pass {chain_eval_pass_pct:.1f}%'/'payout {chain_payout_pct:.1f}%' above mean "
        f"\"did at least one attempt anywhere in the reset chain\" -- "
    )
    if per_attempt_eval_pass_pct is not None and per_attempt_payout_pct is not None:
        attempts_note = f" across {total_attempts:,} independent attempts" if total_attempts else ""
        note += (
            f"the per-ATTEMPT rate{attempts_note} (the \"will ONE account attempt pass\" "
            f"question) is eval pass {per_attempt_eval_pass_pct:.1f}%, "
            f"payout {per_attempt_payout_pct:.1f}%.]"
        )
    else:
        note += "see the Monte Carlo section for the per-attempt rate.]"
    return note


def net_profit_reset_note(stats: "BacktestStatistics") -> str:
    """A shorter version of reset_chain_note's clarification, for the
    (several) console/log lines around the app that print net_profit
    alone, before any Monte Carlo eval-pass/payout numbers exist yet to
    report (e.g. the raw-backtest-just-finished line printed before the
    prop-firm simulation step runs). Returns "" whenever
    stats.account_reset_count is 0 -- the default, non-reset case."""
    if stats.account_reset_count <= 0:
        return ""
    return (
        f" [reset_on_breach: {stats.account_reset_count} account reset(s) occurred -- "
        f"this net profit is CUMULATIVE P&L across {stats.account_reset_count + 1} simulated "
        f"accounts, not one account's result; the CURRENT (final) account's own P&L is "
        f"${stats.final_segment_net_profit:,.2f} over its {stats.final_segment_trade_count} trades.]"
    )


def compute_statistics(
    trades: list[Trade],
    equity_curve: pd.DataFrame,
    initial_balance: float,
    bars_per_year: float = 252 * 78,  # rough default for intraday FX; overridable
) -> BacktestStatistics:
    if not trades:
        return BacktestStatistics(
            net_profit=0, gross_profit=0, gross_loss=0, return_pct=0, average_trade=0,
            win_rate=0, loss_rate=0, average_winner=0, average_loser=0,
            largest_winner=0, largest_loser=0,
            max_drawdown=0, max_drawdown_pct=0, average_drawdown_pct=0,
            max_daily_drawdown_pct=0, max_weekly_drawdown_pct=0,
            max_losing_streak=0, max_winning_streak=0,
            profit_factor=0, expectancy=0, average_r=0, risk_reward=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, total_trades=0,
        )

    pnls_raw = np.array([t.pnl for t in trades])
    finite_mask = np.isfinite(pnls_raw)
    n_bad = int((~finite_mask).sum())
    pnls = pnls_raw[finite_mask]
    if n_bad:
        # A non-finite trade pnl indicates a data or configuration anomaly
        # (e.g. an indicator warm-up period, a near-zero ATR stop distance,
        # or similar edge case). Excluding it keeps the rest of the report
        # trustworthy instead of letting one bad value poison every
        # aggregate statistic (which otherwise silently turns into NaN).
        import warnings
        warnings.warn(
            f"Excluded {n_bad} trade(s) with a non-finite P&L from the backtest "
            "statistics. Check your risk settings (in particular pip size vs. "
            "the instrument's actual price scale, and any ATR-multiple stop) "
            "if this number is large.",
            RuntimeWarning,
        )
    if not len(pnls):
        return BacktestStatistics(
            net_profit=0, gross_profit=0, gross_loss=0, return_pct=0, average_trade=0,
            win_rate=0, loss_rate=0, average_winner=0, average_loser=0,
            largest_winner=0, largest_loser=0,
            max_drawdown=0, max_drawdown_pct=0, average_drawdown_pct=0,
            max_daily_drawdown_pct=0, max_weekly_drawdown_pct=0,
            max_losing_streak=0, max_winning_streak=0,
            profit_factor=0, expectancy=0, average_r=0, risk_reward=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, total_trades=len(trades),
        )
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    net_profit = float(pnls.sum())
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    return_pct = (net_profit / initial_balance) * 100 if initial_balance else 0.0
    average_trade = float(pnls.mean())

    win_rate = len(wins) / len(pnls) * 100
    loss_rate = len(losses) / len(pnls) * 100
    average_winner = float(wins.mean()) if len(wins) else 0.0
    average_loser = float(losses.mean()) if len(losses) else 0.0
    largest_winner = float(wins.max()) if len(wins) else 0.0
    largest_loser = float(losses.min()) if len(losses) else 0.0

    equity = equity_curve["equity"]
    seg_id = _reset_segment_ids(equity_curve)
    running_max = equity.cummax() if seg_id is None else equity.groupby(seg_id).cummax()
    dd_abs = equity - running_max
    max_drawdown = float(dd_abs.min())
    dd_pct_series = _drawdown_series(equity, seg_id)
    max_drawdown_pct = float(abs(dd_pct_series.min()) * 100)
    average_drawdown_pct = float(abs(dd_pct_series[dd_pct_series < 0].mean()) * 100) if (dd_pct_series < 0).any() else 0.0
    max_daily_dd = _periodic_max_drawdown(equity_curve, "1D", seg_id)
    max_weekly_dd = _periodic_max_drawdown(equity_curve, "1W", seg_id)
    account_reset_count = int(seg_id[-1]) if seg_id is not None and len(seg_id) else 0

    trade_seg_id = _trade_reset_segment_ids(trades, equity_curve)
    if trade_seg_id is not None and len(trade_seg_id):
        last_seg = int(trade_seg_id[-1])
        final_seg_trades = [t for t, seg in zip(trades, trade_seg_id) if seg == last_seg]
        final_seg_pnls = np.array([t.pnl for t in final_seg_trades if np.isfinite(t.pnl)])
        final_segment_net_profit = float(final_seg_pnls.sum())
        final_segment_trade_count = len(final_seg_trades)
    else:
        final_segment_net_profit = net_profit
        final_segment_trade_count = len(pnls)
    risk_recon = compute_risk_reconciliation(trades)

    win_streak = _max_streak(list(pnls > 0))
    loss_streak = _max_streak(list(pnls <= 0))

    profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf") if gross_profit > 0 else 0.0
    expectancy = float(average_trade)

    # Average R: pnl / the trade's own PLANNED risk (|entry - stop| * size)
    # at entry time, when the strategy/engine actually attached one. This is
    # the real definition of "R" (risk-normalized return per trade) used in
    # prop-firm evaluation. Falls back to the old realized-loss approximation
    # only for trades with no initial_risk on record (e.g. a strategy that
    # never defines a stop at all and predates this field).
    r_multiples = [
        t.pnl / (t.initial_risk * t.size)
        for t in trades
        if getattr(t, "initial_risk", None) and t.initial_risk > 0 and t.size
    ]
    if r_multiples:
        average_r = float(np.mean(r_multiples))
    else:
        risk_per_trade = [abs(t.pnl) for t in trades if t.pnl <= 0]
        avg_risk = float(np.mean(risk_per_trade)) if risk_per_trade else (abs(average_loser) or 1.0)
        average_r = float(average_trade / avg_risk) if avg_risk else 0.0
    risk_reward = float(abs(average_winner / average_loser)) if average_loser != 0 else float("inf") if average_winner > 0 else 0.0

    # Risk-adjusted ratios computed on per-trade returns (simple, MVP-appropriate approach)
    trade_returns = pnls / initial_balance if initial_balance else pnls
    mean_ret = trade_returns.mean()
    std_ret = trade_returns.std(ddof=1) if len(trade_returns) > 1 else 0.0
    downside = trade_returns[trade_returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0

    sharpe_ratio = float((mean_ret / std_ret) * np.sqrt(len(trade_returns))) if std_ret else 0.0
    sortino_ratio = float((mean_ret / downside_std) * np.sqrt(len(trade_returns))) if downside_std else 0.0
    calmar_ratio = float(return_pct / max_drawdown_pct) if max_drawdown_pct else 0.0

    return BacktestStatistics(
        net_profit=net_profit, gross_profit=gross_profit, gross_loss=gross_loss,
        return_pct=return_pct, average_trade=average_trade,
        win_rate=win_rate, loss_rate=loss_rate,
        average_winner=average_winner, average_loser=average_loser,
        largest_winner=largest_winner, largest_loser=largest_loser,
        max_drawdown=max_drawdown, max_drawdown_pct=max_drawdown_pct,
        average_drawdown_pct=average_drawdown_pct,
        max_daily_drawdown_pct=max_daily_dd, max_weekly_drawdown_pct=max_weekly_dd,
        max_losing_streak=loss_streak, max_winning_streak=win_streak,
        profit_factor=profit_factor, expectancy=expectancy,
        average_r=average_r, risk_reward=risk_reward,
        sharpe_ratio=sharpe_ratio, sortino_ratio=sortino_ratio, calmar_ratio=calmar_ratio,
        total_trades=len(trades), account_reset_count=account_reset_count,
        is_reset_chain=account_reset_count > 0,
        final_segment_net_profit=final_segment_net_profit,
        final_segment_trade_count=final_segment_trade_count,
        avg_intended_risk_dollars=risk_recon["avg_intended_risk_dollars"],
        avg_actual_stop_risk_dollars=risk_recon["avg_actual_stop_risk_dollars"],
        avg_realized_loss_on_losers=risk_recon["avg_realized_loss_on_losers"],
        avg_adaptive_risk_multiplier=risk_recon["avg_adaptive_risk_multiplier"],
        pct_trades_adaptive_throttle_active=risk_recon["pct_trades_adaptive_throttle_active"],
        pct_trades_position_capped=risk_recon["pct_trades_position_capped"],
        pct_trades_risk_overshoot=risk_recon["pct_trades_risk_overshoot"],
    )


def compute_concentration_stats(trades: list) -> dict:
    """
    "Is one lucky trade or one lucky day carrying the whole result?"

    A real, repeatable edge shouldn't evaporate the moment you remove its
    single best outcome. Reports the net profit and profit factor with the
    single best trade removed, and again with the single best calendar day
    removed, plus what share of total gross profit each represents. If
    removing one trade or one day flips net_profit negative (or profit
    factor below ~1.2), the headline numbers are being carried by an
    outlier, not a repeatable process.
    """
    if not trades:
        return {
            "best_trade_pnl": 0.0,
            "best_trade_pct_of_gross_profit": 0.0,
            "net_profit_excluding_best_trade": 0.0,
            "best_day_pnl": 0.0,
            "best_day_pct_of_gross_profit": 0.0,
            "net_profit_excluding_best_day": 0.0,
        }

    pnls = np.array([t.pnl for t in trades])
    pnls = pnls[np.isfinite(pnls)]
    net_profit = float(pnls.sum())
    gross_profit = float(pnls[pnls > 0].sum()) if len(pnls) else 0.0

    best_trade_pnl = float(pnls.max()) if len(pnls) else 0.0
    net_excl_trade = float(net_profit - best_trade_pnl)
    best_trade_pct = (best_trade_pnl / gross_profit * 100.0) if gross_profit > 0 and best_trade_pnl > 0 else 0.0

    daily_pnl: dict = {}
    for t in trades:
        if not np.isfinite(t.pnl):
            continue
        day = pd.Timestamp(t.exit_time).normalize()
        daily_pnl[day] = daily_pnl.get(day, 0.0) + t.pnl
    best_day_pnl = max(daily_pnl.values()) if daily_pnl else 0.0
    net_excl_day = float(net_profit - best_day_pnl)
    best_day_pct = (best_day_pnl / gross_profit * 100.0) if gross_profit > 0 and best_day_pnl > 0 else 0.0

    return {
        "best_trade_pnl": best_trade_pnl,
        "best_trade_pct_of_gross_profit": float(best_trade_pct),
        "net_profit_excluding_best_trade": net_excl_trade,
        "best_day_pnl": float(best_day_pnl),
        "best_day_pct_of_gross_profit": float(best_day_pct),
        "net_profit_excluding_best_day": net_excl_day,
    }


def compute_cost_ladder(trades: list, rungs_pct: list[float] | None = None) -> list[dict]:
    """
    Re-costs the SAME historical trade sequence at increasing round-turn
    friction levels and reports net profit / profit factor at each rung.

    This is the single most emphasized practice in serious strategy
    validation: a "real" edge should survive some added friction, and
    should die gracefully (not from 3.0 to 0.0 profit factor between two
    adjacent rungs) as costs rise. An edge that only exists at 0% added
    cost is not an edge you can trade — it's the cost model doing the
    lying for you.

    rungs_pct: extra ROUND-TURN cost, as a fraction of notional (entry
    price x size), applied on top of whatever commission/spread/slippage
    the backtest already modeled. Defaults to 0%, 0.05%, 0.10%, 0.25% per
    side (i.e. the exact ladder the falsification-kit methodology uses).
    """
    if rungs_pct is None:
        rungs_pct = [0.0, 0.0005, 0.0010, 0.0025]

    notionals = np.array([abs(t.entry_price * t.size) for t in trades]) if trades else np.array([])
    base_pnls = np.array([t.pnl for t in trades]) if trades else np.array([])
    finite_mask = np.isfinite(base_pnls)
    notionals = notionals[finite_mask]
    base_pnls = base_pnls[finite_mask]

    ladder = []
    for rung in rungs_pct:
        extra_cost = notionals * rung
        pnls = base_pnls - extra_cost
        gross_profit = float(pnls[pnls > 0].sum())
        gross_loss = float(pnls[pnls <= 0].sum())
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else (float("inf") if gross_profit > 0 else 0.0)
        ladder.append({
            "extra_cost_pct_per_trade": rung * 100,
            "net_profit": float(pnls.sum()) if len(pnls) else 0.0,
            "profit_factor": profit_factor,
            "win_rate": float((pnls > 0).sum() / len(pnls) * 100) if len(pnls) else 0.0,
        })
    return ladder
