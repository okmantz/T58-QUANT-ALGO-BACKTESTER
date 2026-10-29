# 08 - Machine Learning, Deep Learning and Reinforcement Learning

Why this folder: the AI features and the Discovery Engine borrow ideas from this literature, and so do many bad backtests. Read the evaluation sections first.

Links were written from the authors' and publishers' public records and have NOT been opened or downloaded by the tool that built this list (its sandbox cannot reach arXiv, SSRN or publishers). arXiv and DOI links point straight at the paper; Google Scholar links are searches that lead to the free copy. Run `python scripts/check_paper_links.py` on your own machine to flag any dead link, and correct what it reports.

## Papers

1. Gu, Kelly, Xiu - "Empirical Asset Pricing via Machine Learning", Review of Financial Studies 2020.
   - Role: The benchmark comparison of ML methods for return prediction; trees and neural nets beat linear models modestly.
   - Links: https://www.nber.org/papers/w25398

2. Lopez de Prado - "The 10 Reasons Most Machine Learning Funds Fail", Journal of Portfolio Management 2018.
   - Role: A checklist of how ML backtests go wrong: leakage, overlapping labels, overfitting.
   - Links: https://papers.ssrn.com/abstract=3104816

3. Lopez de Prado - "Backtesting", SSRN 2015.
   - Role: Why walk-forward and cross-validation need purging and embargoes in finance.
   - Links: https://papers.ssrn.com/abstract=2606462

4. Lopez de Prado - "Detection of False Investment Strategies Using Unsupervised Learning Methods", Quantitative Finance 2019.
   - Role: Use clustering to count how many independent strategies were really tested.
   - Links: https://papers.ssrn.com/abstract=3167017

5. Lim, Zohren, Roberts - "Enhancing Time-Series Momentum Strategies Using Deep Neural Networks", Journal of Financial Data Science 2019.
   - Role: Deep nets that output positions directly, trained to maximise Sharpe.
   - Links: https://arxiv.org/pdf/1904.04912

6. Lim, Zohren - "Time-Series Forecasting with Deep Learning: A Survey", Philosophical Transactions of the Royal Society A 2021.
   - Role: A map of architectures for time series, with finance examples.
   - Links: https://arxiv.org/pdf/2004.13408

7. Wood, Roberts, Zohren - "Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection", Journal of Financial Data Science 2022.
   - Role: Adds change-point detection so momentum models react faster to regime shifts.
   - Links: https://arxiv.org/pdf/2105.13727

8. Wood, Giegerich, Roberts, Zohren - "Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture", arXiv 2021.
   - Role: Attention-based momentum model that shows which features matter.
   - Links: https://arxiv.org/pdf/2112.08534

9. Zhang, Zohren, Roberts - "Deep Reinforcement Learning for Trading", Journal of Financial Data Science 2020.
   - Role: Reinforcement learning agents for futures, with costs included.
   - Links: https://arxiv.org/pdf/1911.10107

10. Jiang, Xu, Liang - "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem", arXiv 2017.
   - Role: Policy-gradient portfolio agent on crypto data; note how fragile results are.
   - Links: https://arxiv.org/pdf/1706.10059

11. Liu et al. - "FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance", arXiv 2020.
   - Role: Open-source RL environment for trading research.
   - Links: https://arxiv.org/pdf/2011.09607

12. Hambly, Xu, Yang - "Recent Advances in Reinforcement Learning in Finance", Mathematical Finance 2023.
   - Role: Survey of RL for trading, hedging and execution.
   - Links: https://arxiv.org/pdf/2112.04553

13. Buehler, Gonon, Teichmann, Wood - "Deep Hedging", Quantitative Finance 2019.
   - Role: Neural hedging of derivatives under costs and constraints.
   - Links: https://arxiv.org/pdf/1802.03042

14. Fischer, Krauss - "Deep Learning with Long Short-Term Memory Networks for Financial Market Predictions", European Journal of Operational Research 2018.
   - Role: LSTM on S&P 500 constituents; edge shrinks sharply after 2010.
   - Links: https://www.econstor.eu/bitstream/10419/157808/1/886576210.pdf

15. Krauss, Do, Huck - "Deep Neural Networks, Gradient-Boosted Trees, Random Forests: Statistical Arbitrage on the S&P 500", European Journal of Operational Research 2017.
   - Role: Compares ML models on daily statistical arbitrage.
   - Links: https://www.econstor.eu/bitstream/10419/130166/1/856406309.pdf

16. Chen, Pelger, Zhu - "Deep Learning in Asset Pricing", Management Science 2024.
   - Role: A neural pricing kernel that respects no-arbitrage.
   - Links: https://arxiv.org/pdf/1904.00745

17. Kelly, Malamud, Zhou - "The Virtue of Complexity in Return Prediction", Journal of Finance 2024.
   - Role: Heavily over-parameterised models can still forecast returns out of sample.
   - Links: https://scholar.google.com/scholar?q=Kelly+Malamud+Zhou+The+Virtue+of+Complexity+in+Return+Prediction

18. Dixon, Halperin, Bilokon - "Machine Learning in Finance: From Theory to Practice (selected chapters)", Springer 2020.
   - Role: Textbook; the cross-validation and reinforcement-learning chapters are the useful ones.
   - Links: https://link.springer.com/book/10.1007/978-3-030-41068-1

19. Cartea, Jaimungal, Penalva - "Algorithmic and High-Frequency Trading (companion notes)", Cambridge University Press 2015.
   - Role: Textbook on optimal execution and market making; free lecture notes from the authors are the best start.
   - Links: https://www.cambridge.org/core/books/algorithmic-and-highfrequency-trading/

20. Kakushadze - "101 Formulaic Alphas", Wilmott 2016.
   - Role: A bank of real-world alpha formulas (also in the repo as a PDF).
   - Links: https://arxiv.org/pdf/1601.00991

21. Kakushadze, Serur - "151 Trading Strategies", Z-Capital / SSRN 2018.
   - Role: A catalogue of 151 strategies across asset classes; a source of ideas for the family search.
   - Links: https://papers.ssrn.com/abstract=3247865

22. Hochreiter, Schmidhuber - "Long Short-Term Memory", Neural Computation 1997.
   - Role: The LSTM architecture used in many price-prediction papers.
   - Links: https://www.bioinf.jku.at/publications/older/2604.pdf

23. Vaswani et al. - "Attention Is All You Need", NeurIPS 2017.
   - Role: The transformer architecture behind modern sequence models.
   - Links: https://arxiv.org/pdf/1706.03762

24. Chen, Guestrin - "XGBoost: A Scalable Tree Boosting System", KDD 2016.
   - Role: The gradient-boosting library that dominates tabular finance benchmarks.
   - Links: https://arxiv.org/pdf/1603.02754

25. Bergstra, Bengio - "Random Search for Hyper-Parameter Optimization", Journal of Machine Learning Research 2012.
   - Role: Random search beats grid search; supports the optimizer's sampling design.
   - Links: https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf

26. Hansen - "The CMA Evolution Strategy: A Tutorial", arXiv 2016.
   - Role: How the CMA-ES optimizer works.
   - Links: https://arxiv.org/pdf/1604.00772

27. Bergstra, Bardenet, Bengio, Kegl - "Algorithms for Hyper-Parameter Optimization", NeurIPS 2011.
   - Role: The Tree-structured Parzen Estimator behind the TPE optimizer option.
   - Links: https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf

28. Deb, Pratap, Agarwal, Meyarivan - "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation 2002.
   - Role: The multi-objective search used for Pareto optimisation.
   - Links: https://doi.org/10.1109/4235.996017

29. Prechelt - "Early Stopping - But When?", Neural Networks: Tricks of the Trade 1998.
   - Role: Stopping rules that avoid fitting noise; applies to optimiser patience settings.
   - Links: https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
