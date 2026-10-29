# 10 - Regimes, Events and Macro

Why this folder: the same rule behaves differently in calm, trending and stressed markets, and around scheduled news. These papers cover regime detection and event effects.

Links were written from the authors' and publishers' public records and have NOT been opened or downloaded by the tool that built this list (its sandbox cannot reach arXiv, SSRN or publishers). arXiv and DOI links point straight at the paper; Google Scholar links are searches that lead to the free copy. Run `python scripts/check_paper_links.py` on your own machine to flag any dead link, and correct what it reports.

## Papers

1. Hamilton - "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle", Econometrica 1989.
   - Role: The regime-switching model most regime detectors descend from.
   - Links: https://doi.org/10.2307/1912559

2. Ang, Bekaert - "International Asset Allocation with Regime Shifts", Review of Financial Studies 2002.
   - Role: Regimes in returns and correlations, and what they imply for allocation.
   - Links: https://doi.org/10.1093/rfs/15.4.1137

3. Guidolin, Timmermann - "Asset Allocation under Multivariate Regime Switching", Journal of Economic Dynamics and Control 2007.
   - Role: Four-regime model for stocks and bonds with allocation results.
   - Links: https://doi.org/10.1016/j.jedc.2006.12.004

4. Nystrup, Madsen, Lindstrom - "Long Memory of Financial Time Series and Hidden Markov Models with Time-Varying Parameters", Journal of Forecasting 2017.
   - Role: Adaptive hidden Markov models for regime detection.
   - Links: https://arxiv.org/pdf/1603.01562

5. Nystrup, Hansen, Madsen, Lindstrom - "Regime-Based Versus Static Asset Allocation: Letting the Data Speak", Journal of Portfolio Management 2015.
   - Role: Out-of-sample test of regime-based allocation.
   - Links: https://scholar.google.com/scholar?q=Nystrup+Regime-Based+Versus+Static+Asset+Allocation+Letting+the+Data+Speak

6. Adams, MacKay - "Bayesian Online Changepoint Detection", arXiv 2007.
   - Role: A real-time change-point detector usable for regime switches.
   - Links: https://arxiv.org/pdf/0710.3742

7. Page - "Continuous Inspection Schemes", Biometrika 1954.
   - Role: CUSUM: the original sequential change detector.
   - Links: https://doi.org/10.1093/biomet/41.1-2.100

8. Lucca, Moench - "The Pre-FOMC Announcement Drift", Journal of Finance 2015.
   - Role: Stocks tend to rise in the 24 hours before FOMC meetings.
   - Links: https://www.newyorkfed.org/research/staff_reports/sr512

9. Savor, Wilson - "How Much Do Investors Care About Macroeconomic Risk? Evidence from Scheduled Economic Announcements", Journal of Financial and Quantitative Analysis 2013.
   - Role: Announcement days carry most of the equity premium.
   - Links: https://doi.org/10.1017/S0022109013000070

10. Andersen, Bollerslev, Diebold, Vega - "Micro Effects of Macro Announcements: Real-Time Price Discovery in Foreign Exchange", American Economic Review 2003.
   - Role: How FX prices jump on data releases.
   - Links: https://www.nber.org/papers/w8959

11. Bernanke, Kuttner - "What Explains the Stock Market's Reaction to Federal Reserve Policy?", Journal of Finance 2005.
   - Role: Policy surprises move stocks mainly through expected returns.
   - Links: https://doi.org/10.1111/j.1540-6261.2005.00760.x

12. Ederington, Lee - "How Markets Process Information: News Releases and Volatility", Journal of Finance 1993.
   - Role: Volatility spikes and decays after scheduled news; sizing and stop implications.
   - Links: https://doi.org/10.1111/j.1540-6261.1993.tb04723.x

13. Heston, Sadka - "Seasonality in the Cross-Section of Stock Returns", Journal of Financial Economics 2008.
   - Role: Calendar effects that repeat annually.
   - Links: https://doi.org/10.1016/j.jfineco.2008.02.002

14. Heston, Korajczyk, Sadka - "Intraday Patterns in the Cross-Section of Stock Returns", Journal of Finance 2010.
   - Role: Returns repeat at the same half-hour of the day; the root of time-of-day filters.
   - Links: https://doi.org/10.1111/j.1540-6261.2010.01573.x

15. Ni, Pearson, Poteshman - "Stock Price Clustering on Option Expiration Dates", Journal of Financial Economics 2005.
   - Role: Option hedging pulls prices toward strikes near expiry.
   - Links: https://doi.org/10.1016/j.jfineco.2004.11.001

16. Barbon, Buraschi - "Gamma Fragility", SSRN 2021.
   - Role: How dealer gamma positions make markets unstable.
   - Links: https://scholar.google.com/scholar?q=Barbon+Buraschi+Gamma+Fragility

17. Bollen, Whaley - "Does Net Buying Pressure Affect the Shape of Implied Volatility Functions?", Journal of Finance 2004.
   - Role: Option order flow moves implied volatility.
   - Links: https://doi.org/10.1111/j.1540-6261.2004.00647.x

18. Whaley - "Derivatives on Market Volatility: Hedging Tools Long Overdue", Journal of Derivatives 1993.
   - Role: The introduction of the VIX concept.
   - Links: https://doi.org/10.3905/jod.1993.407868

19. Bekaert, Hoerova - "The VIX, the Variance Premium and Stock Market Volatility", Journal of Econometrics 2014.
   - Role: What the VIX does and does not say about future risk.
   - Links: https://doi.org/10.1016/j.jeconom.2014.05.008

20. Adrian, Brunnermeier - "CoVaR", American Economic Review 2016.
   - Role: Systemic-risk measure for stress periods.
   - Links: https://www.nber.org/papers/w17454

21. Reinhart, Rogoff - "This Time Is Different (selected papers)", Princeton University Press 2009.
   - Role: Long record of crises; a reminder that stress regimes are rare but real.
   - Links: https://scholar.google.com/scholar?q=Reinhart+Rogoff+This+Time+Is+Different

22. Bouchaud, Potters - "Financial Applications of Random Matrix Theory: a short review", Handbook on Random Matrix Theory 2011.
   - Role: Why correlations jump in stress regimes.
   - Links: https://arxiv.org/pdf/0910.1205

23. Brunnermeier, Pedersen - "Market Liquidity and Funding Liquidity", Review of Financial Studies 2009.
   - Role: Liquidity spirals: why stress regimes are self-reinforcing.
   - Links: https://doi.org/10.1093/rfs/hhn098

24. Kaminsky, Reinhart - "The Twin Crises: The Causes of Banking and Balance-of-Payments Problems", American Economic Review 1999.
   - Role: Early-warning view of crises; background for stress filters.
   - Links: https://doi.org/10.1257/aer.89.3.473
