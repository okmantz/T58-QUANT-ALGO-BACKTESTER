# Release notes 2026-11-07

Implements the independent code audit (all findings except the LAN warning, which was deliberately skipped), plus the Replay / Live Monitor / prop-simulator overhaul and a payout-forecast layer on validation.

## Read this first: numbers that move

**Equity-curve timing (audit finding 4).** Equity is now marked when a signal-exit is *scheduled*, not one bar earlier, matching how fills actually happen. Reported **max drawdown and Sharpe shift slightly** on strategies with signal exits; trade lists are unchanged. No existing test pinned the old numbers, so none needed re-baselining; new tests cover the timing and the portfolio-engine mirror. Switch: `_EQUITY_SIGNAL_EXIT_TIMING` in `app/backtest/execution.py`; the same ordering is mirrored bit-for-bit in `portfolio_engine.py`. Re-run saved strategies before comparing to old reports.

## Performance and reliability
- **Skip-memo in the bar loop (finding 1).** Repeated identical skipped entries are cached (keyed on direction, equity, stop/target/trail distances, and the raw price for fallback stops); only *skipped* plans are cached and it is disabled under adaptive risk. Parity tests compare trades, equity and counters against the old path on plain, adaptive-risk and prop-account configs. Measured 4.6-5.7x on a workload that was ~99.8% skips; the gain scales with the share of skipped entries, so ordinary strategies will see less.
- **Day-index cache lock + `_LOOKAHEAD_CACHE` (finding 3).** Removes a rare crash in long jobs.
- **`dayofweek` -> `day_of_week` (finding 5)** across 18 files.
- **Process workers for sweeps (finding 2).** Walk-forward, sensitivity and parameter robustness run on a spawn-safe process pool (`app/validation/process_sweep.py`, same initializer pattern as Search Lab) with a serial fallback. Tested on a spawn pool on Linux; **not yet run on the Windows build**.
- **Speed Run Phase 2 stays on threads.** The benchmark (`scripts/bench_speed_run_phase2.py`) did not finish in the time available, so the move was not made without a measurement. Re-run it on a real machine before changing.
- **Build hygiene (findings 6/7).** `config/requirements-core.txt` and `requirements-extras.txt` added (`requirements.txt` remains the full install for CI/release); the web build job now `needs: test`. A fully pinned cross-platform lock must be generated on your machine (`pip freeze > config/requirements.lock`).

## Interactive Replay and Live Monitor
Shared chart kit (`t58-chart-kit.js`) and rewritten drawing tools (`chart_drawing_tools.js`); both pages use it, Live Monitor panes 2-4 included.
- Chart types: candles, hollow, bars, line, area, baseline, Heikin-Ashi.
- Indicators: SMA, EMA, WMA, Bollinger, VWAP, Donchian, Supertrend on price; RSI, MACD, Stochastic, ATR in their own panes; volume.
- Drawing: trend, ray, extended, horizontal/vertical line, arrow, channel, Fib retracement/extension, measure, box, ellipse, path, text, long/short R:R. Select, move, edit handles, colour/width/dash/fill, lock, clone, magnet, undo/redo, hide/lock all, per-symbol persistence.
- Settings: 5 themes, candle colours, grid, crosshair, log/%/indexed price axis, timezone, hi/lo and last-price lines, watermark, legend, full-screen, PNG screenshot. Preferences persist.
- Replay: follow mode, entry-to-exit lines, SL/TP zones, go-to-time, step size, open-trade chip, new shortcuts (Home/End, B/S/C, Shift+arrows). The chart never receives future bars.
- Replay prop panel: trailing / end-of-day / static drawdown, daily loss, min days, consistency rule, sticky pass/fail banner, firm-preset dropdown.

## Prop firm simulator
- New rules: end-of-day trailing drawdown, floor lock at starting balance, daily-loss lock-the-day, eval time limit.
- Choosing a firm preset fills rules, fees and profit split; the preset note shows whether the firm allows algos.
- Rough EV card, plus a **Payout forecast** panel (below).

## Payout forecast (validation)
Answers "will this get paid?", not "does this pass our gates?".
- Per firm preset: Monte Carlo plus a rolling replay from real start dates; Wilson lower bounds; the decision uses the *lower* of the two.
- P(paid | pass) and expected payout, then **ranked by conservative EV per month**. Trades are rescaled to each firm's account size at constant risk %.
- Graded **OUT-OF-SAMPLE** when run on the held-out 30%, otherwise **IN-SAMPLE (optimistic)**.
- Verdicts: PAYS / MARGINAL / DOES NOT PAY / NO FEE DATA. Informational in the full pipeline: it never changes the pipeline verdict.
- Calibration ledger (`payout_forecast_ledger.jsonl`): log real eval outcomes; forecasts are shrunk toward reality when they have been too optimistic, never raised.
- `score_vs_payout`: Spearman check of whether the T58 score actually tracks payouts on your saved strategies, and it will say so when it does not.
- Routes: `POST /api/payout-forecast/run`, `POST /api/payout-forecast/outcome`, `GET /api/payout-forecast/calibration`.

## Known limits
- The forecast ranks firm fit and records real outcomes; it does not trade on live firm accounts. The existing Deploy connectors are unchanged.
- LFS market-data files were unavailable during testing; tests needing them were skipped or fail for that reason.
- Three `test_translator` failures and one 1-minute-data strategy test fail in this checkout for reasons unrelated to these changes (a shipped expression-style strategy; missing 1m data).
