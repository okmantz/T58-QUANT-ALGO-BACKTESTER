# 06 - Volatility Models and Forecasting

Why this folder: stops, targets and position size are all volatility decisions. These papers cover how volatility behaves, how to forecast it and how to size risk against it.

Links were written from the authors' and publishers' public records and have NOT been opened or downloaded by the tool that built this list (its sandbox cannot reach arXiv, SSRN or publishers). arXiv and DOI links point straight at the paper; Google Scholar links are searches that lead to the free copy. Run `python scripts/check_paper_links.py` on your own machine to flag any dead link, and correct what it reports.

## Papers

1. Andersen, Bollerslev, Diebold, Labys - "Modeling and Forecasting Realized Volatility", Econometrica 2003.
   - Role: Why intraday returns give a far better volatility estimate than daily ones.
   - Links: https://www.nber.org/papers/w8160

2. Corsi - "A Simple Approximate Long-Memory Model of Realized Volatility", Journal of Financial Econometrics 2009.
   - Role: The HAR model: yesterday, last week and last month's volatility predict tomorrow's.
   - Links: https://doi.org/10.1093/jjfinec/nbp001

3. Gatheral, Jaisson, Rosenbaum - "Volatility Is Rough", Quantitative Finance 2018.
   - Role: Volatility is much rougher than classical models assume; changes how it should be forecast.
   - Links: https://arxiv.org/pdf/1410.3394

4. Bennedsen, Lunde, Pakkanen - "Decoupling the Short- and Long-Term Behavior of Stochastic Volatility", Journal of Financial Econometrics 2022.
   - Role: Rough short-term behaviour with a long memory; evidence from many equity indices.
   - Links: https://arxiv.org/pdf/1610.00332

5. Bollerslev - "Generalized Autoregressive Conditional Heteroskedasticity", Journal of Econometrics 1986.
   - Role: GARCH: the standard model for volatility clustering.
   - Links: https://doi.org/10.1016/0304-4076(86)90063-1

6. Engle - "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation", Econometrica 1982.
   - Role: The original ARCH paper (Nobel-winning).
   - Links: https://doi.org/10.2307/1912773

7. Nelson - "Conditional Heteroskedasticity in Asset Returns: A New Approach", Econometrica 1991.
   - Role: EGARCH: volatility reacts more to bad news than good.
   - Links: https://doi.org/10.2307/2938260

8. Parkinson - "The Extreme Value Method for Estimating the Variance of the Rate of Return", Journal of Business 1980.
   - Role: High/low range as a volatility estimator; the basis of range-based measures in the indicator library.
   - Links: https://doi.org/10.1086/296071

9. Garman, Klass - "On the Estimation of Security Price Volatilities from Historical Data", Journal of Business 1980.
   - Role: Open-high-low-close volatility estimator, more efficient than close-to-close.
   - Links: https://doi.org/10.1086/296072

10. Yang, Zhang - "Drift-Independent Volatility Estimation Based on High, Low, Open and Close Prices", Journal of Business 2000.
   - Role: A range-based estimator that stays correct when price trends and gaps.
   - Links: https://doi.org/10.1086/209650

11. Rogers, Satchell - "Estimating Variance from High, Low and Closing Prices", Annals of Applied Probability 1991.
   - Role: Another drift-robust range estimator.
   - Links: https://doi.org/10.1214/aoap/1177005835

12. Harvey, Hoyle, Korgaonkar, Rattray, Sargaison, Van Hemert - "The Impact of Volatility Targeting", Journal of Portfolio Management 2018.
   - Role: When volatility targeting helps (risk assets) and when it does not.
   - Links: https://papers.ssrn.com/abstract=3175538

13. Cont - "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues", Quantitative Finance 2001.
   - Role: The standard list of facts any return model must reproduce: fat tails, clustering, no autocorrelation.
   - Links: https://doi.org/10.1080/713665670

14. Mandelbrot - "The Variation of Certain Speculative Prices", Journal of Business 1963.
   - Role: The original argument that returns have fat tails.
   - Links: https://doi.org/10.1086/294632

15. Chakraborti, Toke, Patriarca, Abergel - "Asset Returns and Volatility Clustering in Financial Time Series", Physica A 2011.
   - Role: Why volatility clusters and how long the effect lasts.
   - Links: https://arxiv.org/pdf/1002.0284

16. Aït-Sahalia, Mykland, Zhang - "How Often to Sample a Continuous-Time Process in the Presence of Market Microstructure Noise", Review of Financial Studies 2005.
   - Role: How sampling frequency trades off noise and information; guides bar-size choice.
   - Links: https://doi.org/10.1093/rfs/hhi016

17. Zhang, Mykland, Aït-Sahalia - "A Tale of Two Time Scales", Journal of the American Statistical Association 2005.
   - Role: Cleaner realized-variance estimates from noisy high-frequency data.
   - Links: https://doi.org/10.1198/016214505000000169

18. Patton - "Volatility Forecast Comparison Using Imperfect Volatility Proxies", Journal of Econometrics 2011.
   - Role: Which loss functions rank volatility forecasts correctly.
   - Links: https://doi.org/10.1016/j.jeconom.2010.03.034

19. Wilder - "New Concepts in Technical Trading Systems (ATR chapter)", Trend Research 1978.
   - Role: Origin of the Average True Range used for stops and sizing throughout this app.
   - Links: https://scholar.google.com/scholar?q=Wilder+New+Concepts+in+Technical+Trading+Systems

20. Lo, MacKinlay - "Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test", Review of Financial Studies 1988.
   - Role: The variance-ratio test used by several indicators here.
   - Links: https://doi.org/10.1093/rfs/1.1.41

21. Christoffersen - "Evaluating Interval Forecasts", International Economic Review 1998.
   - Role: How to check if a risk limit (VaR) is breached as often as it claims.
   - Links: https://doi.org/10.2307/2527341
