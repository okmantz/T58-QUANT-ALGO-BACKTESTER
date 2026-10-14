# 02 - Validation, Overfitting and Data Snooping

Why this folder: the goal is a READY verdict that means something. Report B's validation protocol (log every variant, deflate Sharpe by trial count, walk forward with frozen parameters, charge real costs, random-anchor control, regime splits, resample real trades through the eval rules) rests on these papers. Report A and B both say the main risk is an overstated edge; this is the literature on why edges are overstated.

Links came up in searches on 2026-10-12; open and confirm before relying. [IN REPO] = was in the T58 backtester's previous research/01_Validation and Overfitting folder.

## Previously in the repo (do not re-download)

- Bailey, Lopez de Prado - "The Deflated Sharpe Ratio" (SSRN 2460551)
- Bailey, Borwein, Lopez de Prado, Zhu - "The Probability of Backtest Overfitting" (SSRN 2326253)
- "Backtest Overfitting in Financial Markets"
- "Do Simple Technical Trading Rules Survive Realistic Costs"
- "A Novel Approach to Trading Strategy Parameter Optimization"
- "Structural Limits of OHLCV-Based Intraday Signals in MNQ Futures" and "A Validated Volatility-Volume-Gap Classifier for Regime Identification MNQ Data" (directly on-instrument for NQ/MNQ)

## Additions (not in the repo)

1. Bailey, Borwein, Lopez de Prado, Zhu - "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance", Notices of the AMS 2014.
   - Role: the minimum backtest length argument; how few years of data it takes to overfit. Pairs with Report B step 1.
   - Links: https://scholarworks.wmich.edu/math_pubs/40/
     https://www.carmamaths.org/resources/jon/FTbackfit.pdf

2. Harvey, Liu, Zhu - "... and the Cross-Section of Expected Returns", Review of Financial Studies 2016 (NBER w20592).
   - Role: multiple-testing thresholds; argues a new factor needs a t-statistic well above 2 (about 3) given how many have been tried. Use it to justify the significance bar for your own variant count.
   - Links: https://papers.ssrn.com/abstract=2513152
     https://nber.org/papers/w20592
     https://www.nber.org/papers/w20592.pdf

3. White - "A Reality Check for Data Snooping", Econometrica 2000.
   - Role: bootstrap test of whether the best of many rules beats a benchmark after accounting for the search. Sullivan, Timmermann, White (JF 1999) applied it to technical rules, and Report B cites that result.
   - Links: https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping
     https://users.ssc.wisc.edu/~bhansen/718/White2000.pdf

4. Hansen - "A Test for Superior Predictive Ability", Journal of Business and Economic Statistics 2005 (refinement of White's test). Referenced in a practitioner explainer: https://metricgate.com/docs/reality-check-superior-predictive-ability/ - find the journal version for the paper itself.

5. Brock, Lakonishok, LeBaron - "Simple Technical Trading Rules and the Stochastic Properties of Stock Returns", JF 1992, and Sullivan, Timmermann, White - "Data-Snooping, Technical Trading Rule Performance, and the Bootstrap", JF 1999. Cited in Report B; locate through the journal sites.

## How this feeds the backtester

The app's EV verdict already uses a Bailey/Lopez de Prado style search-bias haircut (expected maximum of n trials) on the holdout pass rate. These papers justify the haircut and tell you how large the trial count is allowed to get before a READY verdict stops meaning anything. Keep a running count of every parameter set and rule variant tried.
