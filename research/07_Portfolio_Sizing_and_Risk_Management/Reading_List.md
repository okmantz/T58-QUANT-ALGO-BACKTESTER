# 07 - Portfolio Construction, Position Sizing and Risk

Why this folder: edge only pays if size and portfolio construction let you survive long enough to collect it. These papers cover sizing, diversification and drawdown control.

Links were written from the authors' and publishers' public records and have NOT been opened or downloaded by the tool that built this list (its sandbox cannot reach arXiv, SSRN or publishers). arXiv and DOI links point straight at the paper; Google Scholar links are searches that lead to the free copy. Run `python scripts/check_paper_links.py` on your own machine to flag any dead link, and correct what it reports.

## Papers

1. Lopez de Prado - "Building Diversified Portfolios that Outperform Out of Sample", Journal of Portfolio Management 2016.
   - Role: Hierarchical Risk Parity: diversify without inverting a noisy covariance matrix.
   - Links: https://papers.ssrn.com/abstract=2708678

2. Lopez de Prado - "A Robust Estimator of the Efficient Frontier", SSRN 2019.
   - Role: Nested clustered optimisation: stabler weights for correlated strategies.
   - Links: https://papers.ssrn.com/abstract=3469961

3. Bun, Bouchaud, Potters - "Cleaning Large Correlation Matrices: Tools from Random Matrix Theory", Physics Reports 2017.
   - Role: How to remove noise from a correlation matrix before using it for risk.
   - Links: https://arxiv.org/pdf/1610.08104

4. Laloux, Cizeau, Bouchaud, Potters - "Noise Dressing of Financial Correlation Matrices", Physical Review Letters 1999.
   - Role: Most of a correlation matrix estimated from limited data is noise.
   - Links: https://arxiv.org/pdf/cond-mat/9810255

5. Mantegna - "Hierarchical Structure in Financial Markets", European Physical Journal B 1999.
   - Role: Clustering assets by correlation, the idea behind clustering trial counts.
   - Links: https://arxiv.org/pdf/cond-mat/9802256

6. Ledoit, Wolf - "Honey, I Shrunk the Sample Covariance Matrix", Journal of Portfolio Management 2004.
   - Role: Shrinkage estimator that makes covariance usable with short histories.
   - Links: https://www.ledoit.net/honey.pdf

7. Peters, Gell-Mann - "Evaluating Gambles Using Dynamics", Chaos 2016.
   - Role: Why time-average growth, not the average outcome, decides whether a bet should be taken; the logic behind Kelly sizing.
   - Links: https://arxiv.org/pdf/1405.0585

8. Thorp - "The Kelly Criterion and the Stock Market", American Mathematical Monthly 1971.
   - Role: The original application of Kelly sizing to markets (also in the repo as a PDF).
   - Links: https://www.edwardothorp.com/

9. MacLean, Thorp, Ziemba - "Good and Bad Properties of the Kelly Criterion", Wilmott 2010.
   - Role: Why most people should use a fraction of full Kelly.
   - Links: https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf

10. Kelly - "A New Interpretation of Information Rate", Bell System Technical Journal 1956.
   - Role: The source of the Kelly criterion.
   - Links: https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf

11. Magdon-Ismail, Atiya - "Maximum Drawdown", Risk Magazine 2004.
   - Role: How deep a drawdown to expect for a given edge and volatility; useful for prop-firm drawdown limits.
   - Links: https://www.cs.rpi.edu/~magdon/ps/journal/drawdown_RISK04.pdf

12. Chekhlov, Uryasev, Zabarankin - "Drawdown Measure in Portfolio Optimization", International Journal of Theoretical and Applied Finance 2005.
   - Role: Optimising a portfolio directly against drawdown instead of variance.
   - Links: https://doi.org/10.1142/S0219024905002767

13. Rockafellar, Uryasev - "Optimization of Conditional Value-at-Risk", Journal of Risk 2000.
   - Role: CVaR: a tail-risk measure that can be optimised directly.
   - Links: https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf

14. Artzner, Delbaen, Eber, Heath - "Coherent Measures of Risk", Mathematical Finance 1999.
   - Role: Why VaR can mislead and what a sensible risk measure must satisfy.
   - Links: https://doi.org/10.1111/1467-9965.00068

15. Markowitz - "Portfolio Selection", Journal of Finance 1952.
   - Role: The origin of mean-variance diversification.
   - Links: https://www.jstor.org/stable/2975974

16. DeMiguel, Garlappi, Uppal - "Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy?", Review of Financial Studies 2009.
   - Role: Estimation error means equal weights often beat optimised weights out of sample.
   - Links: https://doi.org/10.1093/rfs/hhm075

17. Maillard, Roncalli, Teiletche - "The Properties of Equally Weighted Risk Contribution Portfolios", Journal of Portfolio Management 2010.
   - Role: Risk parity: weight by risk contribution, not by dollars.
   - Links: https://papers.ssrn.com/abstract=1271972

18. Qian - "Risk Parity Portfolios: Efficient Portfolios Through True Diversification", PanAgora 2005.
   - Role: Practical introduction to risk parity.
   - Links: https://scholar.google.com/scholar?q=Qian+Risk+Parity+Portfolios+Efficient+Portfolios+Through+True+Diversification

19. Asness, Frazzini, Pedersen - "Leverage Aversion and Risk Parity", Financial Analysts Journal 2012.
   - Role: Why leveraged low-risk assets earn more than theory predicts.
   - Links: https://scholar.google.com/scholar?q=Asness+Frazzini+Pedersen+Leverage+Aversion+and+Risk+Parity

20. Vince - "Portfolio Management Formulas: Optimal f (summary)", Wiley 1990.
   - Role: Fixed-fractional sizing from the trade history; read with the Kelly papers for the risks.
   - Links: https://scholar.google.com/scholar?q=Ralph+Vince+optimal+f+fixed+fractional

21. Bailey, Lopez de Prado - "The Sharpe Ratio Efficient Frontier", Journal of Risk 2012.
   - Role: Sharpe ratio confidence intervals and the minimum track record needed.
   - Links: https://papers.ssrn.com/abstract=1821643

22. Lo - "The Statistics of Sharpe Ratios", Financial Analysts Journal 2002.
   - Role: Why annualising a Sharpe ratio by sqrt(12) can overstate it when returns are autocorrelated.
   - Links: https://doi.org/10.2469/faj.v58.n4.2453

23. Jobson, Korkie - "Performance Hypothesis Testing with the Sharpe and Treynor Measures", Journal of Finance 1981.
   - Role: Test whether one Sharpe ratio really beats another.
   - Links: https://doi.org/10.1111/j.1540-6261.1981.tb00444.x

24. Opdyke - "Comparing Sharpe Ratios: So Where Are the p-values?", Journal of Asset Management 2007.
   - Role: Correct standard errors for Sharpe ratios with fat tails and skew.
   - Links: https://doi.org/10.1057/palgrave.jam.2250084

25. Bailey, Lopez de Prado - "Drawdown-Based Stop-Outs and the Triple Penance Rule", Journal of Risk 2015.
   - Role: How long recovery from a drawdown typically takes; relevant to funded-account trailing limits.
   - Links: https://papers.ssrn.com/abstract=2201302

26. Sortino, van der Meer - "Downside Risk", Journal of Portfolio Management 1991.
   - Role: Measuring risk only below a target return.
   - Links: https://doi.org/10.3905/jpm.1991.409343

27. Young - "Calmar Ratio: A Smoother Tool", Futures Magazine 1991.
   - Role: Return divided by maximum drawdown, the metric prop traders actually live by.
   - Links: https://scholar.google.com/scholar?q=Young+Calmar+Ratio+smoother+tool

28. Taleb - "Fooled by Randomness / The Black Swan (selected excerpts)", Random House 2001/2007.
   - Role: Why rare events dominate outcomes for strategies that look smooth.
   - Links: https://scholar.google.com/scholar?q=Taleb+Black+Swan+fat+tails+trading

29. Kritzman, Li - "Skulls, Financial Turbulence, and Risk Management", Financial Analysts Journal 2010.
   - Role: A turbulence measure for when diversification stops working.
   - Links: https://doi.org/10.2469/faj.v66.n5.3
