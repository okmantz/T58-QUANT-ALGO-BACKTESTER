# 09 - Anomalies, Multiple Testing and Replication

Why this folder: the more ideas you try, the more likely the best one is luck. These papers measure how many published edges are real, how fast they decay and how to correct for the search.

Links were written from the authors' and publishers' public records and have NOT been opened or downloaded by the tool that built this list (its sandbox cannot reach arXiv, SSRN or publishers). arXiv and DOI links point straight at the paper; Google Scholar links are searches that lead to the free copy. Run `python scripts/check_paper_links.py` on your own machine to flag any dead link, and correct what it reports.

## Papers

1. Harvey, Liu - "Backtesting", Journal of Portfolio Management 2015.
   - Role: A haircut for the Sharpe ratio that depends on how many strategies were tested.
   - Links: https://papers.ssrn.com/abstract=2345489

2. Harvey, Liu - "Evaluating Trading Strategies", Journal of Portfolio Management 2014.
   - Role: Why a high Sharpe ratio is not enough once multiple testing is counted.
   - Links: https://papers.ssrn.com/abstract=2474755

3. Harvey, Liu - "Lucky Factors", SSRN 2015.
   - Role: Bootstrap method to find which factors survive when others are tried.
   - Links: https://papers.ssrn.com/abstract=2528780

4. Harvey - "The Scientific Outlook in Financial Economics", Journal of Finance 2017.
   - Role: A call to raise the bar for evidence, with Bayesian arguments.
   - Links: https://papers.ssrn.com/abstract=2893930

5. McLean, Pontiff - "Does Academic Research Destroy Stock Return Predictability?", Journal of Finance 2016.
   - Role: Published anomalies lose about a third to a half of their returns after publication.
   - Links: https://papers.ssrn.com/abstract=2156623

6. Jensen, Kelly, Pedersen - "Is There a Replication Crisis in Finance?", Journal of Finance 2023.
   - Role: Most factors replicate, and a Bayesian view of the evidence.
   - Links: https://www.nber.org/papers/w28432

7. Hou, Xue, Zhang - "Replicating Anomalies", Review of Financial Studies 2020.
   - Role: Many anomalies fail replication once microcaps are handled properly.
   - Links: https://www.nber.org/papers/w23394

8. Harvey, Liu, Zhu - "... and the Cross-Section of Expected Returns (data appendix)", Review of Financial Studies 2016.
   - Role: The catalogue of 300+ factors behind the multiple-testing threshold.
   - Links: https://www.nber.org/papers/w20592

9. Lo, MacKinlay - "Data-Snooping Biases in Tests of Financial Asset Pricing Models", Review of Financial Studies 1990.
   - Role: An early, clear statement of why selecting on the same data biases tests.
   - Links: https://doi.org/10.1093/rfs/3.3.431

10. Romano, Wolf - "Stepwise Multiple Testing as Formalized Data Snooping", Econometrica 2005.
   - Role: Tests that find which of many strategies truly beat a benchmark.
   - Links: https://doi.org/10.1111/j.1468-0262.2005.00615.x

11. Politis, Romano - "The Stationary Bootstrap", Journal of the American Statistical Association 1994.
   - Role: Resampling with random block lengths; the basis of block-bootstrap Monte Carlo.
   - Links: https://doi.org/10.1080/01621459.1994.10476870

12. Kunsch - "The Jackknife and the Bootstrap for General Stationary Observations", Annals of Statistics 1989.
   - Role: Why block resampling is needed for dependent data.
   - Links: https://doi.org/10.1214/aos/1176347265

13. Politis, White - "Automatic Block-Length Selection for the Dependent Bootstrap", Econometric Reviews 2004.
   - Role: How to pick the block size in a block bootstrap; relevant to the Monte Carlo block choice.
   - Links: https://doi.org/10.1081/ETC-120028836

14. Benjamini, Hochberg - "Controlling the False Discovery Rate", Journal of the Royal Statistical Society B 1995.
   - Role: False-discovery-rate control, a less strict alternative to Bonferroni.
   - Links: https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

15. Ioannidis - "Why Most Published Research Findings Are False", PLoS Medicine 2005.
   - Role: The general argument that small samples and many tests produce false positives.
   - Links: https://doi.org/10.1371/journal.pmed.0020124

16. Simmons, Nelson, Simonsohn - "False-Positive Psychology", Psychological Science 2011.
   - Role: How flexible analysis choices make anything look significant (the p-hacking paper).
   - Links: https://doi.org/10.1177/0956797611417632

17. Chordia, Roll, Subrahmanyam - "Liquidity and Market Efficiency", Journal of Financial Economics 2008.
   - Role: Predictability shrinks as liquidity rises.
   - Links: https://doi.org/10.1016/j.jfineco.2007.05.003

18. Lo - "The Adaptive Markets Hypothesis", Journal of Portfolio Management 2004.
   - Role: Edges come and go as markets adapt; why every strategy needs a retirement rule.
   - Links: https://doi.org/10.3905/jpm.2004.442611

19. Timmermann, Granger - "Efficient Market Hypothesis and Forecasting", International Journal of Forecasting 2004.
   - Role: What forecastability means when many people are searching for it.
   - Links: https://doi.org/10.1016/S0169-2070(03)00012-8

20. Bailey, Lopez de Prado - "Finding the Best Strategy Among Many: The Haircut (summary)", Journal of Portfolio Management 2014.
   - Role: Practical overview of Sharpe haircuts after selection.
   - Links: https://scholar.google.com/scholar?q=Bailey+Lopez+de+Prado+haircut+Sharpe+selection+bias

21. Arlot, Celisse - "A Survey of Cross-Validation Procedures for Model Selection", Statistics Surveys 2010.
   - Role: How cross-validation works and fails; background for purged CPCV.
   - Links: https://arxiv.org/pdf/0907.4728

22. Bergmeir, Hyndman, Koo - "A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction", Computational Statistics and Data Analysis 2018.
   - Role: When ordinary cross-validation is safe for time series.
   - Links: https://robjhyndman.com/papers/cvts.pdf

23. Pardo - "The Evaluation and Optimization of Trading Strategies (walk-forward chapters)", Wiley 2008.
   - Role: The classic practitioner text on walk-forward testing.
   - Links: https://scholar.google.com/scholar?q=Pardo+Evaluation+and+Optimization+of+Trading+Strategies+walk-forward

24. Aronson - "Evidence-Based Technical Analysis (data-mining bias chapters)", Wiley 2007.
   - Role: Applies the data-mining correction to technical rules.
   - Links: https://scholar.google.com/scholar?q=Aronson+Evidence-Based+Technical+Analysis+data+mining+bias
