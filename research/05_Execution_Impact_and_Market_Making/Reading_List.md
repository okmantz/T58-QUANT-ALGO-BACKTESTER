# 05 - Execution, Market Impact and Market Making

Why this folder: costs and fills decide whether an intraday edge survives. These papers say how large trades move price, how to execute and quote, and how to model the order book. Use them to sanity-check the engine's slippage and fill assumptions.

Links were written from the authors' and publishers' public records and have NOT been opened or downloaded by the tool that built this list (its sandbox cannot reach arXiv, SSRN or publishers). arXiv and DOI links point straight at the paper; Google Scholar links are searches that lead to the free copy. Run `python scripts/check_paper_links.py` on your own machine to flag any dead link, and correct what it reports.

## Papers

1. Almgren, Chriss - "Optimal Execution of Portfolio Transactions", Journal of Risk 2000.
   - Role: The classic trade-off between impact cost and the risk of waiting; the base model behind most execution schedules.
   - Links: https://www.courant.nyu.edu/~almgren/papers/optliq.pdf

2. Avellaneda, Stoikov - "High-Frequency Trading in a Limit Order Book", Quantitative Finance 2008.
   - Role: Optimal bid/ask quoting around a reservation price with inventory risk; the standard market-making model.
   - Links: https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf

3. Cont, Kukanov, Stoikov - "The Price Impact of Order Book Events", Journal of Financial Econometrics 2014.
   - Role: Short-horizon price changes are explained mostly by order-flow imbalance at the best quotes; basis of many intraday signals.
   - Links: https://arxiv.org/pdf/1011.6402

4. Bouchaud, Farmer, Lillo - "How Markets Slowly Digest Changes in Supply and Demand", Handbook of Financial Markets 2009.
   - Role: Why large orders are split, why price impact is concave and why order flow is long-memory.
   - Links: https://arxiv.org/pdf/0809.0822

5. Toth, Lempergue, Lillo, Bouchaud et al. - "Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets", Physical Review X 2011.
   - Role: Impact grows like the square root of order size; the empirical anchor for realistic cost models.
   - Links: https://arxiv.org/pdf/1105.1694

6. Donier, Bonart, Mastromatteo, Bouchaud - "A Fully Consistent, Minimal Model for Non-Linear Market Impact", Quantitative Finance 2015.
   - Role: A tidy model that reproduces square-root impact and its decay.
   - Links: https://arxiv.org/pdf/1412.0141

7. Eisler, Bouchaud, Kockelkoren - "The Price Impact of Order Book Events: Market Orders, Limit Orders and Cancellations", Quantitative Finance 2012.
   - Role: Separates the impact of market orders, limit orders and cancellations.
   - Links: https://arxiv.org/pdf/0904.0900

8. Bouchaud, Mezard, Potters - "Statistical Properties of Stock Order Books: Empirical Results and Models", Quantitative Finance 2002.
   - Role: First careful look at the shape of the order book and the cost of walking through it.
   - Links: https://arxiv.org/pdf/cond-mat/0203511

9. Potters, Bouchaud - "More Statistical Properties of Order Books and Price Impact", Physica A 2003.
   - Role: Follow-up with price-impact measurements from real order books.
   - Links: https://arxiv.org/pdf/cond-mat/0210710

10. Gould, Porter, Williams, McDonald, Fenn, Howison - "Limit Order Books", Quantitative Finance 2013.
   - Role: Survey of how limit order books work and how to model them.
   - Links: https://arxiv.org/pdf/1012.0349

11. Bacry, Mastromatteo, Muzy - "Hawkes Processes in Finance", Market Microstructure and Liquidity 2015.
   - Role: Self-exciting point processes for trade and quote arrivals; good intuition for bursty intraday activity.
   - Links: https://arxiv.org/pdf/1502.04592

12. Bacry, Delattre, Hoffmann, Muzy - "Modelling Microstructure Noise with Mutually Exciting Point Processes", Quantitative Finance 2013.
   - Role: Models tick data with mutually exciting processes and explains jump-like behaviour.
   - Links: https://arxiv.org/pdf/1112.1838

13. Gueant, Lehalle, Fernandez-Tapia - "Dealing with the Inventory Risk: A Solution to the Market Making Problem", Mathematics and Financial Economics 2013.
   - Role: Closed-form quotes for market making with inventory limits.
   - Links: https://arxiv.org/pdf/1105.3115

14. Gueant - "Optimal Market Making", Applied Mathematical Finance 2017.
   - Role: Extends market making to many assets and realistic constraints.
   - Links: https://arxiv.org/pdf/1605.01862

15. Zhang, Zohren, Roberts - "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books", IEEE Transactions on Signal Processing 2019.
   - Role: A deep model for short-horizon mid-price moves from the book; read the caveats on costs.
   - Links: https://arxiv.org/pdf/1808.03668

16. Sirignano - "Deep Learning for Limit Order Books", Quantitative Finance 2019.
   - Role: Neural network models of the full book and what they can predict.
   - Links: https://arxiv.org/pdf/1601.01987

17. Sirignano, Cont - "Universal Features of Price Formation in Financial Markets: Perspectives from Deep Learning", Quantitative Finance 2019.
   - Role: One model trained across many stocks finds shared price-formation structure.
   - Links: https://arxiv.org/pdf/1803.06917

18. Ning, Lin, Jaimungal - "Double Deep Q-Learning for Optimal Execution", Applied Mathematical Finance 2021.
   - Role: Reinforcement learning applied to execution, compared with classical schedules.
   - Links: https://arxiv.org/pdf/1812.06600

19. Alfonsi, Fruth, Schied - "Optimal Execution Strategies in Limit Order Books with General Shape Functions", Quantitative Finance 2010.
   - Role: Execution when the book has a realistic, non-flat shape.
   - Links: https://arxiv.org/pdf/0708.1756

20. Farmer, Patelli, Zovko - "The Predictive Power of Zero Intelligence in Financial Markets", PNAS 2005.
   - Role: A random-order model explains much of the order-book behaviour; a reminder how little 'skill' microstructure needs.
   - Links: https://arxiv.org/pdf/cond-mat/0309233

21. Chakraborti, Toke, Patriarca, Abergel - "Econophysics Review: I. Empirical Facts", Quantitative Finance 2011.
   - Role: A checklist of well-established statistical facts about returns, volatility and order flow.
   - Links: https://arxiv.org/pdf/0909.1974

22. Kyle - "Continuous Auctions and Insider Trading", Econometrica 1985.
   - Role: The foundational model of price impact (Kyle's lambda).
   - Links: https://www.jstor.org/stable/1913210

23. Hasbrouck - "Measuring the Information Content of Stock Trades", Journal of Finance 1991.
   - Role: How much of a trade's price effect is permanent; basis for impact estimation.
   - Links: https://www.jstor.org/stable/2328934

24. Glosten, Milgrom - "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders", Journal of Financial Economics 1985.
   - Role: Why spreads exist: adverse selection from informed traders.
   - Links: https://doi.org/10.1016/0304-405X(85)90044-3

25. Easley, Lopez de Prado, O'Hara - "Flow Toxicity and Liquidity in a High-Frequency World", Review of Financial Studies 2012.
   - Role: VPIN: volume-time order-flow imbalance as a toxicity gauge.
   - Links: https://papers.ssrn.com/abstract=1695596

26. Frazzini, Israel, Moskowitz - "Trading Costs of Asset Pricing Anomalies", AQR / SSRN.
   - Role: Real institutional trading costs are far smaller than academic estimates, and why it matters for which anomalies survive.
   - Links: https://scholar.google.com/scholar?q=Frazzini+Israel+Moskowitz+Trading+Costs

27. Novy-Marx, Velikov - "A Taxonomy of Anomalies and Their Trading Costs", Review of Financial Studies 2016.
   - Role: Which published anomalies survive realistic trading costs.
   - Links: https://www.nber.org/papers/w20721

28. Bouchaud, Bonart, Donier, Gould - "Trades, Quotes and Prices (companion notes)", Cambridge University Press 2018.
   - Role: Book-length treatment of microstructure and impact; start with the free chapters and notes on the authors' pages.
   - Links: https://www.cambridge.org/core/books/trades-quotes-and-prices/
