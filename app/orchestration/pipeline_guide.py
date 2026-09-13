"""
Shared "what to do next" guidance for the strategy lifecycle:

    create (Evolution Lab / Search Lab / Manual Builder / upload)
        -> promote to Strategy Library
        -> validate (Validation Lab: walk-forward, CPCV, sensitivity, ...)
        -> optimize (Quick Optimize / Full Pipeline's own GA step)
        -> Full Pipeline (the all-in-one re-validated verdict)
        -> promote to champion (save_to_library / manual promote)

Both the web app and the desktop app import from HERE rather than each
writing their own "next, go do X" text, so the two UIs can never drift
apart or disagree about what the recommended next step is at a given
stage -- see app.web.server's /evolution/promote, /search/job/.../promote,
Full Pipeline status routes, and app.ui.main_window's matching tabs for
the call sites.

Every function here is a pure string-builder: no I/O, no side effects,
safe to call from a web route's JSON response or a desktop Tkinter log
line equally.
"""
from __future__ import annotations


def _pbo_note(n_tested: int, threshold: int = 2, hard_threshold: int = 20) -> str:
    """Shared multiple-comparisons warning for after_search_complete and
    after_evolution_stop. `n_tested` should be the largest honest count of
    "how many independent candidates were compared to produce this pick"
    that the caller has on hand -- total_candidates/total_evaluated when
    available (the real pool size PBO cares about), falling back to
    leaderboard_size (survivors only, an undercount, but still >1 means a
    selection happened). PBO (compute_pbo() in app.validation.cpcv) is the
    genuine, textbook answer to "is picking the best backtest the same as
    picking noise" -- it needs the pool of candidates as input, which is
    exactly the leaderboard this function already has; it does not run
    itself here, both because it needs the price data + candidate specs
    (this module stays pure-string, no I/O) and because it's expensive
    enough (many backtests per candidate per path) that it should be an
    explicit, opt-in action -- CPCV / PBO tab (route /cpcv, desktop tab
    "09 -- CPCV / PBO") -- not something that runs silently after every
    search.
    """
    if n_tested < threshold:
        return ""
    urgency = (
        f"With {n_tested} candidates tried, treat this pick with real suspicion until PBO says otherwise -- "
        if n_tested >= hard_threshold
        else f"With {n_tested} candidates compared, "
    )
    return (
        f" {urgency}the more variants you test, the more likely the single best one is just the luckiest "
        "roll of the batch rather than a genuine edge (the multiple-comparisons problem). Before trusting "
        "profit factor or Sharpe alone to declare a winner, run this same candidate pool through CPCV / "
        "PBO (VALIDATE -> CPCV / PBO tab, or the app's --pbo CLI flag for a full multi-candidate run): a "
        "Probability of Backtest Overfitting comfortably under 50% means the pick likely reflects a real "
        "edge, not noise; at or above 50% means picking this 'winner' was statistically no better than a "
        "coin flip and you should treat the whole batch as unproven."
    )


def after_evolution_stop(leaderboard_size: int, total_evaluated: int | None = None) -> str:
    """`total_evaluated`: optional, the honest count of candidates this run
    actually tried (e.g. generations completed x population size) --
    pass it when the caller has it (see app.evolution.engine.EvolutionRunner's
    generation count and cfg.population_size) so the multiple-comparisons
    note below reflects the real search size, not just how many survived
    onto the leaderboard. Falls back to leaderboard_size if omitted."""
    if leaderboard_size <= 0:
        return (
            "Next step: no candidate cleared the pre-filter yet. Let it run more generations, or try a "
            "different family selection -- there's nothing to promote until at least one candidate "
            "appears on the leaderboard."
        )
    pbo_note = _pbo_note(total_evaluated if total_evaluated is not None else leaderboard_size)
    return (
        f"Next step: pick your best candidate from the {leaderboard_size}-strategy leaderboard above and "
        "click PROMOTE. That saves it to the Strategy Library -- from there, run it through the "
        "Validation Lab (walk-forward + CPCV) or straight through Full Pipeline for a re-validated "
        f"verdict.{pbo_note}"
    )


def after_promote_to_library(filename: str) -> str:
    return (
        f"Saved '{filename}' to the Strategy Library. Next step: run it through Full Pipeline (recommended -- "
        "baseline, walk-forward GA re-optimization, out-of-sample check, holdout check, and a final "
        "READY/MARGINAL/NOT READY verdict in one pass), or the Validation Lab first if you specifically "
        "want walk-forward/CPCV/sensitivity results on their own before optimizing."
    )


def after_search_complete(
    champion_candidate_id: str | None, leaderboard_size: int, total_candidates: int | None = None,
) -> str:
    """`total_candidates`: optional, the total number of candidates this
    search actually generated/tried (see app.search.batch_runner.SearchSummary
    .total_candidates) -- pass it when available so the multiple-comparisons
    note reflects the real pool size (usually far larger than the
    leaderboard, which is only Stage 3 survivors). Falls back to
    leaderboard_size if omitted."""
    n_tested = total_candidates if total_candidates is not None else leaderboard_size
    if not champion_candidate_id:
        if leaderboard_size > 0:
            return (
                "Next step: no candidate passed every Stage 3 gate, but the leaderboard above has "
                f"{leaderboard_size} candidate(s) that made it partway. Promote the top one and run it "
                "through Full Pipeline anyway, or widen the search (more families / more candidates) and "
                f"run again.{_pbo_note(n_tested)}"
            )
        return (
            "Next step: nothing survived Stage 1 at all. Try a wider family selection, a higher "
            "max-candidates count, or looser Stage 1 filters (min trades / min profit factor) and search again."
        )
    return (
        f"Champion candidate found: {champion_candidate_id}. Next step: click PROMOTE next to it, then run "
        "it through Full Pipeline for a re-validated, walk-forward-optimized final verdict before "
        f"considering it for a live prop-firm evaluation.{_pbo_note(n_tested)}"
    )


def after_full_pipeline(verdict: str | None, saved_to_library: bool) -> str:
    verdict = (verdict or "").upper()
    library_note = " It was saved to the Strategy Library." if saved_to_library else ""
    if verdict == "READY":
        return (
            f"Verdict: READY.{library_note} Next step: this is your strongest evidence yet, but it's still "
            "a backtest -- consider a short forward-test (MT5 Forward Test tab, or paper trading) before "
            "risking real capital on a prop-firm evaluation."
        )
    if verdict == "MARGINAL":
        return (
            f"Verdict: MARGINAL.{library_note} Next step: it's not clearly ready or clearly dead. Try "
            "Quick Optimize on it, or adjust risk settings (position size, daily-loss limit) and re-run "
            "Full Pipeline -- don't take it live as-is."
        )
    if verdict == "NOT READY":
        return (
            "Verdict: NOT READY. Next step: this strategy doesn't hold up under re-validation. Go back to "
            "Evolution Lab or Search Lab for a different candidate rather than trying to rescue this one."
        )
    return "Next step: open the full report above for the detailed verdict and what drove it."


def after_full_pipeline_batch(outcomes: list[dict]) -> str:
    """outcomes: list of {"label", "ok", "verdict", "eval_pass_probability"} dicts,
    same shape as app.orchestration.full_pipeline.FullPipelineBatchOutcome."""
    ready = [o for o in outcomes if o.get("ok") and (o.get("verdict") or "").upper() == "READY"]
    if ready:
        best = max(ready, key=lambda o: o.get("eval_pass_probability") or 0)
        return (
            f"Next step: '{best['label']}' came back READY with the highest eval-pass probability "
            f"({best.get('eval_pass_probability', 0):.1f}%) of the batch -- open its report above, then "
            "consider a short forward-test before a live prop-firm evaluation."
        )
    marginal = [o for o in outcomes if o.get("ok") and (o.get("verdict") or "").upper() == "MARGINAL"]
    if marginal:
        return (
            "Next step: nothing in this batch came back READY, but "
            f"{len(marginal)} strateg{'y is' if len(marginal) == 1 else 'ies are'} MARGINAL -- open their "
            "reports, try Quick Optimize on the best of them, or adjust risk settings and re-run."
        )
    return (
        "Next step: nothing in this batch came back READY or MARGINAL. Go back to Evolution Lab or "
        "Search Lab for different candidates rather than optimizing any of these further."
    )


def after_first_backtest(stats: dict, passed_evaluation: bool | None = None) -> str:
    """Branches on a single manual backtest's own results (Run & Report /
    Step 5) -- the very first decision point in the app, and the one place
    that had NO structured "what next" logic at all before this: a person
    with a strategy that generated zero trades, or a strongly losing one,
    got the same generic "here's your report" as someone with a genuinely
    promising result, with nothing steering them toward what to actually
    do about it.

    `stats` is a dict with at least `trade_count`, `profit_factor`,
    `max_drawdown_pct` (matching app.backtest.engine.BacktestStatistics'
    field names loosely -- callers can pass a dataclass's __dict__ or an
    equivalent plain dict). `passed_evaluation` is the prop-sim result on
    this same historical run, if available (None if not computed).
    """
    trade_count = stats.get("trade_count", stats.get("total_trades", 0)) or 0
    profit_factor = stats.get("profit_factor")
    max_dd = stats.get("max_drawdown_pct")

    if trade_count == 0:
        return (
            "Next step: this strategy generated zero trades on this data. Before touching risk or "
            "prop rules (03 Prop-Firm Rules / 04 Risk & Execution won't matter until trades exist), fix "
            "the entry logic itself: for Manual Builder, open 1 Strategy Configuration and check each "
            "indicator/condition row -- an overly narrow threshold (e.g. RSI < 5 instead of < 30) or two "
            "conditions that can never be true at the same bar are the usual causes; for an uploaded "
            "Python/PineScript/MQL5 strategy, check the entry condition function directly for a logic "
            "bug or a symbol/column name that doesn't match this dataset. Also confirm on 2 Market Data "
            "that the loaded timeframe actually matches what the strategy expects (a strategy written "
            "for M15 checked against daily bars will often just never fire). Nothing downstream (Monte "
            "Carlo, prop simulation, Search Lab, Evolution Lab) can produce a meaningful result until at "
            "least a handful of trades appear -- re-run 5 Run & Report after each change."
        )
    if trade_count < 20:
        return (
            f"Next step: only {trade_count} trade(s) over this data -- too few for Monte Carlo or a "
            "prop-firm simulation to say much with confidence (both resample from whatever trades "
            "exist, so a handful of trades just gets resampled a lot, not made more reliable; treat any "
            "pass-probability number from this run as noise until this number is well past 20-30). Three "
            "concrete ways to get more: on 2 Market Data, load a longer date range or a lower timeframe "
            "(M15 instead of H4, for example) for more bars to trade against; on 1 Strategy Configuration, "
            "loosen the entry condition's threshold (e.g. widen an RSI band, shorten a lookback period) so "
            "it fires more often; or, if the strategy is fundamentally low-frequency by design, accept "
            "that and skip straight to Search Lab/Evolution Lab across a wider instrument/timeframe set "
            "rather than trying to force more signals out of this one config."
        )
    if profit_factor is not None and profit_factor < 1.0:
        return (
            f"Next step: profit factor {profit_factor:.2f} means this version loses money over this "
            "data -- don't chase it with Monte Carlo, Search Lab's parameter perturbations, or Full "
            "Pipeline yet, since optimizing a losing edge just finds the least-bad way to still lose. "
            "Fix the edge first: open 5 Run & Report's trade breakdown and check win rate and average "
            "win/loss size specifically -- a low win rate with small wins/big losses points to the exit "
            "logic (tighten the stop-loss or add a profit target on 1 Strategy Configuration / 4 Risk & "
            "Execution), while a decent win rate that still nets negative points to the entry filter "
            "being too loose (add a confirming condition, e.g. a higher-timeframe trend filter). If it's "
            "a Manual Builder strategy, adjust those indicator settings by hand and re-run; if it's an "
            "uploaded or generated strategy, let Search Lab (structured, three-stage) or Evolution Lab "
            "(open-ended GA) explore parameter variations and other families automatically instead of "
            "hand-tuning code. If none of that moves profit factor above 1.0 after a few honest tries, "
            "try a different instrument or timeframe before spending more time on this exact setup."
        )
    if max_dd is not None and max_dd > 40:
        return (
            f"Next step: max drawdown {max_dd:.1f}% is severe for most prop-firm limits (many cap "
            "overall drawdown around 8-10%). A profitable-on-average strategy with a drawdown this "
            "deep will fail almost every prop-firm simulation despite a decent profit factor -- tighten "
            "risk before anything else: on 4 Risk & Execution, lower the position-size / risk-per-trade "
            "percentage first (the fastest lever), then check whether a stop-loss is set at all and "
            "tighten it if it's wide or missing; if the strategy relies on a wide stop by design, reduce "
            "position size further to compensate rather than removing the stop. Re-run 5 Run & Report "
            "after each change and confirm max drawdown has actually come down before moving on to "
            "Monte Carlo or Full Pipeline -- a good profit factor with an unfixed drawdown problem will "
            "still fail a prop evaluation."
        )
    if passed_evaluation is False:
        return (
            "Next step: solid trade stats, but this run failed the prop-firm simulation under your "
            "current 03 Prop Rules / 04 Risk settings. Check the failure reason shown above -- daily "
            "loss limit and max drawdown are the two most common single-run killers -- then either "
            "loosen risk per trade or re-check that your prop rules actually match your real firm's "
            "terms before assuming the strategy itself is bad."
        )
    return (
        "Next step: this looks like a reasonable first result -- check the evaluation pass "
        "probability and first payout probability just computed above. From here, either promote "
        "straight to Full Pipeline for a re-validated verdict, or try Search Lab / Evolution Lab "
        "first if you want to see whether a nearby variant does meaningfully better."
    )


def strategy_creation_recommendation(has_specific_idea: bool) -> str:
    """Helps decide WHERE to start creating a strategy -- the very first
    fork in the app, before any of the create-> validate->optimize chain
    above even begins. Not tied to any run result (there isn't one yet),
    just the one question that actually determines which of the four
    Create-section tools makes sense to reach for first."""
    if has_specific_idea:
        return (
            "You already have a specific setup in mind: use the Manual Builder if it's rule-based "
            "(indicator crossovers, breakouts, MS-LSD-style structure/liquidity/order-block logic), "
            "or upload it directly if you already have it as Python, PineScript, or MQL5 source. "
            "Search Lab and Evolution Lab are for exploring when you DON'T have a specific idea yet -- "
            "skip them for now and come back to them later to see if a variant beats your own design."
        )
    return (
        "No specific idea yet: Search Lab (a structured, three-stage screen across every registered "
        "family) or Evolution Lab (an open-ended genetic search you can leave running for hours) are "
        "both better starting points than the Manual Builder's blank slate. Use Search Lab for a "
        "faster, bounded first pass; use Evolution Lab if you want to leave it running unattended and "
        "come back to a leaderboard later."
    )


def after_forward_test(result: dict) -> str:
    """Branches on a completed Forward Test (MT5) session -- the real-market
    dry run between Full Pipeline and Deploy Live. `result` is a dict with
    at least `trade_count`, `matched_backtest` (bool | None -- whether live
    fills roughly matched what the backtest would have predicted for the
    same period, if that comparison was computed) and `errors` (list)."""
    errors = result.get("errors") or []
    if errors:
        return (
            f"Next step: {len(errors)} error(s) occurred during the forward test (see the log above) -- "
            "these are almost always broker/connection/symbol-mapping issues, not the strategy itself. "
            "Resolve them and run another forward-test session before considering Deploy Live; a session "
            "with unresolved errors proves nothing either way."
        )
    trade_count = result.get("trade_count", 0) or 0
    if trade_count == 0:
        return (
            "Next step: no trades triggered during this forward-test session. That can be entirely "
            "normal for a low-frequency strategy over a short window -- run a longer session before "
            "concluding anything. If it also generated few trades in the original backtest, this is "
            "consistent, not a red flag."
        )
    if result.get("matched_backtest") is False:
        return (
            "Next step: live fills diverged from what the backtest would have predicted for this same "
            "period -- check slippage/spread/commission assumptions on 04 Risk & Execution against what "
            "the broker actually gave you, and re-run Full Pipeline with updated execution costs before "
            "trusting the original verdict for real capital."
        )
    return (
        "Next step: forward test ran cleanly and roughly matched backtest expectations. This is your "
        "strongest evidence yet -- Deploy Live is reasonable from here, starting at reduced size if "
        "your prop firm allows it, with 12 Monitor (Live Market) open to watch the first stretch."
    )
