# Engine Parity Checklist

This app now has more than one code path that can produce a backtest
result for the same strategy + data:

- the real bar-by-bar engine (`app/backtest/execution.py`)
- the vectorized Stage 1 fast path (`app/backtest/vectorized_fastpath.py`)
- the portfolio composer's per-leg solo backtests vs. its combined
  shared-account sequence (`app/portfolio/portfolio.py`)

Every one of this app's worst historical bugs (asymmetric entry/exit
costs, stops filling at an impossible exact level, a pip-size mismatch
silently producing catastrophic position sizes, a daily-loss check that
only looked at realized P&L instead of intrabar floating loss) came from
an assumption drifting apart between two places that were supposed to
agree, without anyone noticing until a real strategy's numbers looked
insane. Adding a second, faster execution path multiplies that risk if
it isn't held to the same standard as the first one.

**The rule: any change to how a fill, cost, or risk-sizing assumption is
computed in one engine must ship with a parity table showing the same
change (or lack of behavior change) reflected in every other engine
that's supposed to agree with it, PLUS the dollar/percentage impact.**

## Single-engine consolidation counts too

The rule above is about assumptions drifting apart ACROSS engines. The
same failure mode can happen WITHIN one engine when the same fill/cost
logic is hand-copied at multiple call sites instead of shared: each copy
is free to quietly drift from the others. `app/backtest/execution.py`'s
`run_execution()` used to apply its exit-settlement logic (spread/
slippage, commission, loss-clamping, Trade construction, adaptive-risk
and daily-P&L bookkeeping) independently at three call sites -- the
normal stop/take/signal exit, the daily-loss-limit forced close, and the
end-of-data close. Consolidating all three into one `_settle_exit()`
helper surfaced exactly this kind of drift: a non-finite pnl on the
forced-close or end-of-data path was silently zeroed WITHOUT the
`_invalid_pnl_skipped` reason suffix the normal exit path already
applied. Now every path gets identical treatment. Full existing test
suite (including `tests/test_account_safety.py` and
`tests/test_reliability_improvements.py`) passes unchanged, since the
divergence only affected the (rare) non-finite-pnl edge case's reason
string, not any trade's price/size/pnl.

## What counts as "an assumption that must stay in sync"

- Fill price on stop-loss, take-profit, and signal exits (including
  gap-through / honest-fill behavior)
- Spread, slippage, and commission application
- Position sizing (risk-amount and units-per-stop-distance formulas)
- Daily-loss-limit / max-trades-per-day / account-blown enforcement
- Any pip-size or ATR-scale sanity-check heuristic

## Required table shape

For every PR/CHANGES_SUMMARY.md entry that touches one of the above,
include a table like this (see `app/backtest/vectorized_fastpath.py`'s
own module docstring and `tests/test_vectorized_fastpath.py` for a
worked example):

| Assumption | Real engine (`execution.py`) | Vectorized fast path | Delta / impact |
|---|---|---|---|
| Stop-loss gap-through fill | `min(stop, open)` for longs, `max(stop, open)` for shorts | identical (`_resolve_intrabar_exit` logic mirrored in `run_vectorized_batch`) | none — verified trade-for-trade in `test_vectorized_matches_scalar_engine_trade_by_trade` |
| max_trades_per_day | enforced via per-day counter | enforced via per-day, per-column counter | none once fixed (an earlier draft of this fast path omitted it entirely — caught by `test_vectorized_batch_multiple_candidates_are_independent` before it shipped, since the default is 10, not unlimited) |
| Daily-loss forced-close | enforced (mark-to-market) | **not implemented** | fast path is a ranking filter only; any candidate this affects gets the real number the moment it reaches Stage 2/3 |

A row that says "not implemented, and here's why that's safe" is a
valid, honest entry — the point is that the gap is STATED, not that
every engine must do everything the real one does.

## Where this lives

- Module docstrings (`vectorized_fastpath.py`, `hrp.py`) state their own
  scope/limitations inline — read those first before assuming a
  fast-path number means what the real engine's number means.
- `tests/test_vectorized_fastpath.py` is the enforcement mechanism: it
  asserts trade-for-trade parity against the real engine across several
  stop/target/signal shapes. If you change fill or cost logic in either
  engine, this test suite should fail before a person notices a wrong
  number in production.
- When you add a THIRD path that can answer "what would this strategy's
  trades have been" (a new optimizer, a new report path, a live/forward-
  test reconciliation), add its own parity section here and its own
  parity test file — don't rely on a person remembering to check by eye.
