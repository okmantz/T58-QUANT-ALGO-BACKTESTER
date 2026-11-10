# 30-Day Prop Evaluation Research Report

**Verdict:** `NOT_ESTABLISHED`

The top discovery candidate does NOT establish a 30-day pass: Development P(pass<=30d) is 4.9%: below half the 65% target, so there is nothing worth validating.; The holdout overlaps a range already observed (bundled data after development: read by earlier experiments / sealed window (cannot be proven unread): 2024-07-01..2026-03-01). A window that was looked at before cannot be reused as independent validation.; No independent holdout: the 30-day claim cannot be established from this data. A forward test (or newer, unseen data) is required.

## Rules simulated

```
{
 "account": 50000,
 "target_pct": 6.0,
 "max_dd_pct": 4.0,
 "daily_loss_pct": 2.4
}
```

## Feasibility frontier (i.i.d. upper bound, best risk size)

| Net edge R/trade | Trades/30d | Best risk % | P(pass<=30d) |
|---|---|---|---|
| 0.1 | 5 | 2.0 | 22% |
| 0.1 | 10 | 2.0 | 33% |
| 0.1 | 20 | 1.25 | 41% |
| 0.1 | 40 | 1.25 | 41% |
| 0.1 | 80 | 0.6 | 46% |
| 0.2 | 5 | 2.0 | 27% |
| 0.2 | 10 | 2.0 | 39% |
| 0.2 | 20 | 1.25 | 52% |
| 0.2 | 40 | 0.75 | 56% |
| 0.2 | 80 | 0.5 | 67% |
| 0.3 | 5 | 2.0 | 32% |
| 0.3 | 10 | 2.0 | 47% |
| 0.3 | 20 | 1.25 | 63% |
| 0.3 | 40 | 0.75 | 72% |
| 0.3 | 80 | 0.5 | 84% |
| 0.5 | 5 | 2.0 | 44% |
| 0.5 | 10 | 1.5 | 63% |
| 0.5 | 20 | 1.25 | 82% |
| 0.5 | 40 | 0.75 | 92% |
| 0.5 | 80 | 0.4 | 99% |

## Opportunity scan (discovery data only, 150 cells tried)

- 150 cells tried on DISCOVERY data only (< 2023-01-01). Survivors are hypotheses: they have not touched development, walk-forward or sealed data and the trial count above is the multiple-testing burden.
- CL: skipped (no data supplied)

| Cell | Trades/30d | Net R | Feasibility UB | P(30d) | 95% lower |
|---|---|---|---|---|---|
| NQ|orb|ny_open|{'or_minutes': 60}|t1.5R | 17.79 | 0.0363 | 0.305 | 8.4% | 3.3% |
| NQ|orb|ny_open|{'or_minutes': 60}|t2R_be | 17.79 | 0.0581 | 0.35 | 8.3% | 2.7% |
| NQ|compression|ny_open|{'lookback': 12, 'ratio': 0.7}|t2R_be | 14.08 | 0.0005 | 0.325 | 5.7% | 1.3% |
| NQ|orb|ny_open|{'or_minutes': 60}|t1R | 17.79 | 0.0466 | 0.315 | 5.6% | 1.3% |
| NQ|orb|ny_open|{'or_minutes': 30}|t2R_be | 19.87 | 0.0062 | 0.35 | 6.6% | 0.9% |
| NQ|gap_fade|london|{'k_atr': 2.0}|t1.5R | 11.64 | 0.0089 | 0.252 | 1.9% | 0.0% |
| NQ|gap_fade|london|{'k_atr': 2.0}|t2R_be | 11.64 | 0.0749 | 0.362 | 0.5% | 0.0% |
| NQ|drive_cont|ny_open|{'drive_minutes': 30, 'k_atr': 2.0}|t2R_be | 8.92 | 0.1607 | 0.36 | 0.4% | 0.0% |
| NQ|drive_cont|ny_open|{'drive_minutes': 30, 'k_atr': 1.0}|t2R_be | 14.27 | 0.0191 | 0.298 | 0.4% | 0.0% |
| NQ|gap_fade|london|{'k_atr': 2.0}|t1R | 11.64 | 0.0273 | 0.278 | 0.0% | 0.0% |

## Exit lab (frozen entries of NQ|orb|ny_open|{'or_minutes': 60}|t1.5R; 13 policies tried)

| Policy | Mean R | P(30d) | 95% lower |
|---|---|---|---|
| fixed_3R | 0.083 | 19.0% | 8.2% |
| be_1R_t3 | 0.057 | 13.4% | 6.0% |
| fixed_2R | 0.088 | 13.8% | 5.6% |
| time_60_t2 | 0.081 | 13.0% | 5.1% |
| dyn_target_3atr | 0.077 | 13.0% | 5.1% |
| regime_t1.5_hi2.5 | 0.045 | 9.9% | 4.3% |
| be_1R_t2 | 0.056 | 9.3% | 3.0% |
| partial_1R_run3 | 0.053 | 8.5% | 3.0% |

## Portfolio (in-sample, 26 weight vectors)

Diversification ratio 1.25

- weights {"NQ|orb|ny_open|{'or_minutes': 60}|t1.5R": 1.0, "NQ|orb|ny_open|{'or_minutes': 60}|t2R_be": 1.0, "NQ|compression|ny_open|{'lookback': 12, 'ratio': 0.7}|t2R_be": 1.0} -> P(30d) 27.7%, lower 16.4%, bust 59.9%
- weights {"NQ|orb|ny_open|{'or_minutes': 60}|t1.5R": 1.0, "NQ|orb|ny_open|{'or_minutes': 60}|t2R_be": 1.0, "NQ|compression|ny_open|{'lookback': 12, 'ratio': 0.7}|t2R_be": 0.5} -> P(30d) 24.3%, lower 14.3%, bust 59.1%
- weights {"NQ|orb|ny_open|{'or_minutes': 60}|t1.5R": 1.0, "NQ|orb|ny_open|{'or_minutes': 60}|t2R_be": 1.0, "NQ|compression|ny_open|{'lookback': 12, 'ratio': 0.7}|t2R_be": 0.0} -> P(30d) 23.5%, lower 13.6%, bust 59.4%

## Validation gate

Verdict: **NOT_ESTABLISHED**

- Development P(pass<=30d) is 4.9%: below half the 65% target, so there is nothing worth validating.
- The holdout overlaps a range already observed (bundled data after development: read by earlier experiments / sealed window (cannot be proven unread): 2024-07-01..2026-03-01). A window that was looked at before cannot be reused as independent validation.
- No independent holdout: the 30-day claim cannot be established from this data. A forward test (or newer, unseen data) is required.
- Walk-forward: only 0% of test spans reach half the target pass rate (need >= 60%).
- Block robustness: the pass rate is concentrated in a minority of blocks, i.e. one regime carries the result.
- discovery: P(30d) 8.4% (95% lower 3.0%, 741 windows, ~35 independent)
- development: P(30d) 4.9% (95% lower 0.0%, 349 windows, ~16 independent)

## Honesty notes

- Pass probability is measured on real calendar time from purchase, using the existing prop rule engine's floating drawdown / daily loss.
- Eventual (chain) Monte Carlo pass is never used as a 30-day number.
- No holdout inside the bundled data can be certified untouched; a forward test is required before any success claim.
