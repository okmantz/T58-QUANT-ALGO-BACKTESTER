# Strategy Discovery Engine (2026-10-25)

Search Lab, Evolution Lab, Forge (the deep gauntlet preset), Speed Run and the hypothesis layer are ONE engine with presets (the Create page now exposes only the engine; the classic pages remain under "More tools") (`app/discovery_engine`).
One objective (`edge_lb_r`), one entry floor (100+ trades, mean-R t >= 1, cost < 0.15R), one lifetime trial ledger clustered to
effective trials, one sealed slice read once, one promotion sequence:
reproduction -> sequence-deflated Sharpe -> random-timing null gate -> pooled cross-market edge -> sealed slice.
A passing champion is a forward-test candidate, never a "winner".

## Run
    python -m scripts.bootstrap_data                      # which pooled markets do you actually have? (needs >= 8)
    python -m scripts.run_surrogate_study --null-runs 60  # false-discovery rate on no-edge data + power on planted +0.10..0.30R edges
    python -m scripts.run_discovery_engine --preset search_lab --data data/raw/NQ_5m.csv --symbol NQ --pool ES=... --retest
    python -m scripts.run_preregistered_families NQ=... ES=... --v13-batch=1

The 100 report indicators are in `app/strategy/indicators_v13*.py`; the 100 families (92 runnable, 8 blocked with reasons) in
`app/search/families_v13.py`, pre-registered with kill conditions in `app/discovery/hypothesis.py`.
New family batches stay CLOSED until `reports/surrogate_gate.json` (written by the surrogate study, valid 90 days, bound to the
current engine code) shows < 5% false discoveries over >= 60 null runs and >= 50% power on +0.20/+0.30R plants measured on >= 10 runs per planted-edge size.
Requires `numba` for speed (pure-Python fallback is ~100x slower for the HMM/analog indicators).

## Known limits
- Surrogate FDR on your real data has not been measured; the shipped gate is closed.
- Ext feeds (VIX term, COT, skew, ...) need your CSV drop-ins; those families are `needs_data`.
- Pooled test needs >= 8 markets and fails closed with fewer.
- Blocked families: 6, 95 (scale-in), 18 (calendar spread), 39 (closing imbalance), 45 (roll calendar), 56, 57, 59 (tick/book).
- No Johansen test (PC-spread + AR(1) gate instead); family 29 uses OLS lines, not Theil-Sen.

## Fast Lab v2 (the default method)

- **Budgets:** Quick 5,200 / Standard 10,400 / Deep 15,600 evaluated configurations (shown in the app as 5,000+ / 10,000+ / 15,000+). Every configuration is optimized the same way (random search, evolution, local refinement) and the count of distinct signal sets tried feeds the luck bar below. Known-dead neighbourhoods from the Strategy Graveyard are skipped, not re-tested.
- **Search:** 30 strategy families (trend, reversion, breakout, session, momentum, plus an any-indicator family that turns about 300 indicators into trend, revert or slope rules), each with optional side / trend / hours / volatility filters, stop, target and hold-time variants. Random search, then evolution, then local refinement - all scored on the training slice only.
- **Splits:** train 60% / validate 20% / sealed test 20%, then neighbour robustness and a 2x-cost stress test. Confidence labels deflate by the number of distinct ideas tried.
- **No leaks:** the average-daily-range indicator now uses only completed days (a lookahead was fixed), and a test (`tests/backtest/test_prefix_invariance_all_kinds.py`) checks that every indicator kind gives the same value on a bar with or without later bars. Saved strategies are causal (the position is carried bar by bar; an unresolved trade stays open at the end of data). Every finalist is re-run in the real backtester, including the same lookahead test Full Pipeline uses, before it can be called a winner.
- **Full history:** the Lab uses all loaded bars (resampled to the bar sizes where costs are small versus risk) instead of a 400k-bar tail.
- A winner is still a lead: run it through Full Pipeline and Validate.
