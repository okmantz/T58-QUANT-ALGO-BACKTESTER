# 11 - Trend, Mean Reversion and Technical Rules

Why this folder: these are the families the Discovery Engine searches. They show what has worked, in which markets, and what the honest out-of-sample numbers look like.

Links were written from the authors' and publishers' public records and have NOT been opened or downloaded by the tool that built this list (its sandbox cannot reach arXiv, SSRN or publishers). arXiv and DOI links point straight at the paper; Google Scholar links are searches that lead to the free copy. Run `python scripts/check_paper_links.py` on your own machine to flag any dead link, and correct what it reports.

## Papers

1. Lempérière, Deremble, Seager, Potters, Bouchaud - "Two Centuries of Trend Following", Journal of Investment Strategies 2014.
   - Role: Trend following has worked for two hundred years across assets, with a persistent, modest Sharpe ratio.
   - Links: https://arxiv.org/pdf/1404.3274

2. Lempérière, Deremble, Nguyen, Seager, Potters, Bouchaud - "Risk Premia: Asymmetric Tail Risks and Excess Returns", Quantitative Finance 2017.
   - Role: Trend following has positive skew; carry-style premia have negative skew.
   - Links: https://arxiv.org/pdf/1409.7720

3. Jegadeesh, Titman - "Returns to Buying Winners and Selling Losers", Journal of Finance 1993.
   - Role: The founding cross-sectional momentum result.
   - Links: https://doi.org/10.1111/j.1540-6261.1993.tb04702.x

4. Asness, Moskowitz, Pedersen - "Value and Momentum Everywhere", Journal of Finance 2013.
   - Role: Value and momentum work across eight markets and are negatively correlated.
   - Links: https://doi.org/10.1111/jofi.12021

5. Hurst, Ooi, Pedersen - "Demystifying Managed Futures", Journal of Investment Management 2013.
   - Role: What trend-following funds do and how to replicate it simply.
   - Links: https://scholar.google.com/scholar?q=Hurst+Ooi+Pedersen+Demystifying+Managed+Futures

6. Clare, Seaton, Smith, Thomas - "Trend Following, Risk Parity and the Influence of Correlation", Journal of Banking and Finance 2016.
   - Role: Trend filters on risk-parity portfolios across assets.
   - Links: https://doi.org/10.1016/j.jbankfin.2016.07.015

7. Zakamulin - "Market Timing with Moving Averages: The Anatomy and Performance of Trading Rules", Palgrave 2017.
   - Role: Which moving-average rules do what and how much of the result is noise.
   - Links: https://scholar.google.com/scholar?q=Zakamulin+Market+Timing+with+Moving+Averages

8. Faber - "A Quantitative Approach to Tactical Asset Allocation", Journal of Wealth Management 2007.
   - Role: The simple 10-month moving-average filter and its drawdown reduction.
   - Links: https://papers.ssrn.com/abstract=962461

9. Brock, Lakonishok, LeBaron - "Simple Technical Trading Rules and the Stochastic Properties of Stock Returns", Journal of Finance 1992.
   - Role: Classic test of moving-average and breakout rules, with a bootstrap.
   - Links: https://doi.org/10.1111/j.1540-6261.1992.tb04681.x

10. Park, Irwin - "What Do We Know About the Profitability of Technical Analysis?", Journal of Economic Surveys 2007.
   - Role: Survey of 95 studies; profits shrink after careful testing and costs.
   - Links: https://doi.org/10.1111/j.1467-6419.2007.00519.x

11. Lo, Mamaysky, Wang - "Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation", Journal of Finance 2000.
   - Role: Pattern recognition done statistically; some patterns carry small incremental information.
   - Links: https://doi.org/10.1111/0022-1082.00265

12. Neely, Weller, Ulrich - "The Adaptive Markets Hypothesis: Evidence from the Foreign Exchange Market", Journal of Financial and Quantitative Analysis 2009.
   - Role: FX technical rules profit in bursts, then fade.
   - Links: https://doi.org/10.1017/S0022109009090127

13. Menkhoff, Taylor - "The Obstinate Passion of Foreign Exchange Professionals: Technical Analysis", Journal of Economic Literature 2007.
   - Role: Why practitioners keep using technical rules, and what evidence supports them.
   - Links: https://doi.org/10.1257/jel.45.4.936

14. Poterba, Summers - "Mean Reversion in Stock Prices: Evidence and Implications", Journal of Financial Economics 1988.
   - Role: Evidence of slow mean reversion in returns over long horizons.
   - Links: https://doi.org/10.1016/0304-405X(88)90021-9

15. Fama, French - "Permanent and Temporary Components of Stock Prices", Journal of Political Economy 1988.
   - Role: Long-horizon reversal in returns.
   - Links: https://doi.org/10.1086/261535

16. Gatev, Goetzmann, Rouwenhorst - "Pairs Trading: Performance of a Relative-Value Arbitrage Rule", Review of Financial Studies 2006.
   - Role: The classic pairs-trading study; profits fell over time.
   - Links: https://doi.org/10.1093/rfs/hhj020

17. Avellaneda, Lee - "Statistical Arbitrage in the U.S. Equities Market", Quantitative Finance 2010.
   - Role: PCA-based residual mean reversion and its performance.
   - Links: https://doi.org/10.1080/14697680903124632

18. Leung, Li - "Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit", International Journal of Theoretical and Applied Finance 2015.
   - Role: Optimal entry/exit thresholds for an Ornstein-Uhlenbeck spread, including stops.
   - Links: https://arxiv.org/pdf/1411.5062

19. Bertram - "Analytic Solutions for Optimal Statistical Arbitrage Trading", Physica A 2010.
   - Role: Closed-form entry/exit levels for mean-reverting spreads.
   - Links: https://doi.org/10.1016/j.physa.2010.02.014

20. Elliott, van der Hoek, Malcolm - "Pairs Trading", Quantitative Finance 2005.
   - Role: State-space model for spread mean reversion.
   - Links: https://doi.org/10.1080/14697680500149370

21. Lo, MacKinlay - "When Are Contrarian Profits Due to Stock Market Overreaction?", Review of Financial Studies 1990.
   - Role: Short-term reversals are partly cross-autocorrelation, not pure overreaction.
   - Links: https://doi.org/10.1093/rfs/3.2.175

22. Nagel - "Evaporating Liquidity", Review of Financial Studies 2012.
   - Role: Short-term reversal profits are payment for providing liquidity and spike in stress.
   - Links: https://doi.org/10.1093/rfs/hhs066

23. Connors, Alvarez - "Short-Term Trading Strategies That Work (summary of RSI(2) tests)", TradingMarkets 2009.
   - Role: Practitioner tests of short-term mean reversion; treat as hypotheses, not proof.
   - Links: https://scholar.google.com/scholar?q=Connors+Alvarez+Short+Term+Trading+Strategies+That+Work

24. Menkhoff, Sarno, Schmeling, Schrimpf - "Carry Trades and Global Foreign Exchange Volatility", Journal of Finance 2012.
   - Role: Carry returns and crash risk tied to volatility.
   - Links: https://doi.org/10.1111/j.1540-6261.2012.01728.x

25. Lustig, Roussanov, Verdelhan - "Common Risk Factors in Currency Markets", Review of Financial Studies 2011.
   - Role: A carry and dollar factor explain currency returns.
   - Links: https://doi.org/10.1093/rfs/hhr068

26. Koijen, Moskowitz, Pedersen, Vrugt - "Carry", Journal of Financial Economics 2018.
   - Role: Carry across asset classes in one framework.
   - Links: https://doi.org/10.1016/j.jfineco.2017.11.002

27. Bessembinder - "Do Stocks Outperform Treasury Bills?", Journal of Financial Economics 2018.
   - Role: Most stocks lose to bills; returns are dominated by a few winners; a skew lesson for single-strategy sizing.
   - Links: https://doi.org/10.1016/j.jfineco.2018.05.003

28. Bender, Sun, Thomas, Zdorovtsov - "The Promises and Pitfalls of Factor Timing", Journal of Portfolio Management 2018.
   - Role: Timing rules rarely beat static exposure after costs.
   - Links: https://scholar.google.com/scholar?q=Bender+Sun+Thomas+Zdorovtsov+Promises+and+Pitfalls+of+Factor+Timing

29. Kaufman - "Trading Systems and Methods (indicator reference)", Wiley 2013.
   - Role: Reference for adaptive moving averages, efficiency ratio and many indicators in the library.
   - Links: https://scholar.google.com/scholar?q=Kaufman+Trading+Systems+and+Methods+adaptive+moving+average

30. Chande, Kroll - "The New Technical Trader (indicator reference)", Wiley 1994.
   - Role: Origin of Chande momentum, VIDYA and several oscillators used here.
   - Links: https://scholar.google.com/scholar?q=Chande+Kroll+The+New+Technical+Trader

31. Ehlers - "Cycle Analytics for Traders (filters reference)", Wiley 2013.
   - Role: Signal-processing filters (super smoother, Hilbert) behind the cycle indicators.
   - Links: https://www.mesasoftware.com/papers/
