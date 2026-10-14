# 03 - Volatility Sizing, Kelly and Drawdown Math

Why this folder: both reports show that trailing-drawdown evals are decided by sizing. Report A: adaptive size (cushion above the floor divided by $2,000, clipped 0.35x to 1.25x) takes bust to about 0% for a few points of pass. Report B: the same rule cut bust from 31.5% to 9.4% on the 45%-win profile. These papers are the theory behind that rule.

Links came up in searches on 2026-10-12; open and confirm before relying. [IN REPO] = was in the T58 backtester's previous research folder.

1. Moreira, Muir - "Volatility-Managed Portfolios", Journal of Finance 2017 (NBER w22208).
   - Role: scaling exposure down when volatility is high raises Sharpe. Supports cutting size when the cushion shrinks or the VIX spikes. Report B also cites DeMiguel, Martin-Utrera, Uppal (JF), which finds out-of-sample and cost problems, so read both sides.
   - Links: https://www.nber.org/papers/22208
     https://www.nber.org/papers/w22208.pdf

2. Barroso, Santa-Clara - "Momentum Has Its Moments", JFE 2015: volatility-scaled momentum.
   - Link: https://ciencia.ucp.pt/en/publications/momentum-has-its-moments/

3. Thorp - "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market".
   - Role: the standard reference for optimal bet size, why full Kelly is too aggressive, and why fractional Kelly is used when the edge is uncertain. Directly relevant to an overstated backtest edge.
   - Links: https://sites.oxy.edu/lengyel/M330/thorp/paper.htm
     https://ideas.repec.org/h/wsi/wschap/9789814293501_0054.html
   - [IN REPO: 06_Prop Firm Economics and Risk Sizing had two Kelly-criterion papers, plus "An Actuarial Analysis of Proprietary Trading Firm Challenges" and "Day Trading Adapting to Change"]

4. Magdon-Ismail, Atiya, Pratap, Abu-Mostafa - "On the Maximum Drawdown of a Brownian Motion", Journal of Applied Probability 2004.
   - Role: expected maximum drawdown as a function of drift and volatility; a closed-form check on whether a strategy's per-trade volatility fits inside a $2,000 buffer.
   - Link: https://www.cambridge.org/core/journals/journal-of-applied-probability/article/on-the-maximum-drawdown-of-a-brownian-motion/F9E3B8A454B020DDEBF0AC3390EF7807

5. LuxAlgo open-source "prop-firm-sim" (cited in Report A). Practitioner tool, not a paper. It returns pass probability, which rule kills attempts, expected fees, and a risk-per-trade sweep that separates pass-maximizing from EV-maximizing risk. Worth benchmarking against the T58 Monte Carlo. Find it through LuxAlgo's site.

## Key takeaways to carry into strategy design (from the reports, not from these papers)

- Size so a single loss uses no more than about 25% of the remaining drawdown buffer.
- Below 1:1 payoff the drawdown is the enemy; at 1.5:1 or better the failure mode becomes running out of time.
- Test with the lower bound of your win-rate confidence interval, not the point estimate.
