# Council: what is holding the backtesting engine back (2026-10-19)

Read against T58_AUDIT_REPORT.md. Ranked by how much each item blocks a strategy from reaching READY.

## Diagnosis
1. **The 30-day payout gate was the dominant false blocker (fixed).** FTMO's 4-day minimum and 14-day payout cadence make "paid inside 30 days" structurally rare (0-3% on every run). It still tracks, but it is now a soft metric: it costs at most 5 score points and can no longer cap the verdict. The EV model uses a longer-horizon paid probability. Mode is `payout30_mode` = soft | hard | off.
2. **EV could not compute (fixed).** The FTMO preset had no challenge price or reset price, so every verdict said "insufficient evidence".
3. **Strategies are regime dependent and the engine could not say so (added).** All four best strategies weakened in 2025. New tools: a regime/volatility gate and a higher-timeframe bias gate, usable on any strategy type (Python, Pine, MQL5, JSON) from every run page.
4. **Single-timeframe thinking (added).** Strategies can now execute on one timeframe while reading another (e.g. read 1h, execute 15m).
5. **Pine and MQL5 were only partly understood (added).** Real interpreters now run complex scripts (loops, functions, arrays, request.security, handles and CopyBuffer), so the optimizer, validator and pipeline can work on them.
6. **Honest multiple-testing (added).** A persistent trial ledger (`reports/trial_ledger.json`) adds earlier runs of the same strategy to the deflated-Sharpe trial count automatically. This protects against false positives, so any pass is believable.
7. **Cost model inconsistency (partly fixed).** Legacy form defaults (1.0 + 0.5 pip) are wrong for futures. Pick the instrument in "Accuracy settings" first (it fills tick size, contract value, spread, slippage and commission). Do this on every run you compare.

## What did NOT need loosening
Ruin, per-attempt pass, null-test, cost-stress and ICIR gates are doing their job: they rejected thin edges. Loosening them would only make a fake edge look real.

## Where an edge is most likely to be found (evidence-based)
- Gap-Aligned Drive (NQ/MNQ): add the regime gate (try "skip dead/low-volatility periods" and "trending"), develop only on data before 2024-12-03, then check the holdout.
- Gold and FX: use the 2009+ hourly/daily files, with a 4h or daily macro filter on an H1 entry, not the 5-month 1-minute gold file.
- Run the optimizer with fitness `prop_guide_score` (new default) rather than the 30-day payout rate.

## Honest limits
No setting makes a strategy profitable. The engine can now test, filter, deflate and deploy honestly; the edge itself still has to exist in the data.
