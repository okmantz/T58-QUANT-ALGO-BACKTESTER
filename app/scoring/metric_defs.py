"""
Plain-English definitions for every headline metric in the app. Powers the clickable-metric popups (static/t58-metrics.js):
click Profit Factor, Max Drawdown, T58 Score ... anywhere and get what it means, the formula, what goes into the
calculation, how to read it, and its history from the run log -- the same idea as the "why NOT READY" verdict popup.

`history_for(key)` reads app.reports.run_history and returns the per-run values of that metric (newest last) so the popup can
show a real historical breakdown (latest, best, worst, average, per-strategy rows) rather than a generic blurb.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# history_field -> key in run_history rows (None = no per-run history on record)
METRICS: dict[str, dict[str, Any]] = {
    "profit_factor": {
        "title": "Profit Factor", "aliases": ["profit factor", "pf", "profit factor (pf)", "p.factor"],
        "meaning": "How many dollars the strategy made for every dollar it lost.",
        "formula": "Profit Factor = gross profit ÷ gross loss",
        "inputs": ["Gross profit: sum of all winning trades", "Gross loss: absolute sum of all losing trades"],
        "reading": ["Below 1.0 loses money.", "1.3–1.5 is a workable edge after costs.", "Above 2.0 on a small sample is often luck — check the trade count."],
        "history_field": "profit_factor", "unit": "x", "higher_is_better": True,
    },
    "max_drawdown": {
        "title": "Max Drawdown", "aliases": ["max drawdown", "max dd", "drawdown", "max drawdown %", "max drawdown (%)", "max dd %", "max dd (%)", "dd %", "max dd%", "drawdown %"],
        "meaning": "The deepest peak-to-trough fall in account equity.",
        "formula": "Max Drawdown = max over time of (running peak equity − current equity); in % = that ÷ the peak",
        "inputs": ["Equity curve after every trade", "Running peak of that curve"],
        "reading": ["Prop-firm trailing drawdowns are usually 4–10% — a strategy whose historical drawdown is close to that will bust accounts.", "Compare it to the firm's limit, not to your gut."],
        "history_field": "max_drawdown_pct", "unit": "%", "higher_is_better": False,
    },
    "win_rate": {
        "title": "Win Rate", "aliases": ["win rate", "win %", "win%", "win rate %", "win rate (%)"],
        "meaning": "Share of trades that closed in profit.",
        "formula": "Win Rate = winning trades ÷ total trades",
        "inputs": ["Count of winning trades", "Count of all closed trades"],
        "reading": ["Meaningless alone — a 35% win rate with 3R winners beats a 70% win rate with tiny winners.", "Read it together with average win / average loss."],
        "history_field": "win_rate", "unit": "%", "higher_is_better": True,
    },
    "expectancy": {
        "title": "Expectancy", "aliases": ["expectancy", "avg realized r", "avg r", "average r"],
        "meaning": "What an average trade is worth. In R it is measured in units of the amount you risked.",
        "formula": "Expectancy = win rate × average win − loss rate × average loss  (equivalently net profit ÷ trades)",
        "inputs": ["Win rate", "Average win", "Average loss", "Number of trades"],
        "reading": ["Must be positive after commissions and slippage.", "0.15R+ is meaningful; the T58 score treats 0.6R as a full mark."],
        "history_field": None, "unit": "", "higher_is_better": True,
    },
    "sharpe_ratio": {
        "title": "Sharpe Ratio", "aliases": ["sharpe", "sharpe ratio", "sharpe (ann.)"],
        "meaning": "Return per unit of volatility — how smooth the equity curve is.",
        "formula": "Sharpe = mean(periodic return) ÷ standard deviation(periodic return) × √(periods per year)",
        "inputs": ["Periodic returns of the equity curve", "Annualisation factor"],
        "reading": ["Above 1 is decent, above 2 is excellent, but only trust it with many trades."],
        "history_field": "sharpe_ratio", "unit": "", "higher_is_better": True,
    },
    "net_profit": {
        "title": "Net Profit", "aliases": ["net profit", "total p&l", "total pnl", "net p&l", "net p/l", "net pnl", "net profit ($)", "total p/l"],
        "meaning": "Total money made after all trades, commissions and slippage.",
        "formula": "Net Profit = sum of every closed trade's P&L (after costs)",
        "inputs": ["Every closed trade's P&L", "Commission and slippage per trade"],
        "reading": ["Size-dependent — compare strategies on profit factor / expectancy, not raw dollars."],
        "history_field": "net_profit", "unit": "$", "higher_is_better": True,
    },
    "t58_score": {
        "title": "T58 Score", "aliases": ["t58 score", "t58 scorecard", "score", "t58", "t58 score (0-100)"],
        "meaning": "One 0–100 number that blends every trusted signal: prop-firm odds, payout odds, ruin risk, out-of-sample stability, expectancy and drawdown.",
        "formula": "Score = Σ(weight × component) ÷ Σ(weights present). Monte Carlo family capped at 30 pts; walk-forward/CPCV, parameter stability, expectancy, drawdown and parsimony fill the rest.",
        "inputs": ["Pass probability (lower confidence bound)", "First-payout probability", "Risk of ruin", "Funded survival", "Walk-forward / CPCV stability", "Parameter stability", "Expectancy lower bound", "Drawdown"],
        "reading": ["READY needs every hard gate passed AND a score ≥ 75.", "85+ is Elite. Open a NOT READY tag to see the exact gates that failed."],
        "history_field": None, "unit": "/100", "higher_is_better": True,
    },
    "robustness": {
        "title": "Robustness Score", "aliases": ["robustness", "robustness score"],
        "meaning": "How well the edge survives data it was not tuned on.",
        "formula": "Mean of: best walk-forward/CPCV efficiency, (100 − PBO %), and the last T58 score — whichever exist.",
        "inputs": ["Walk-forward / CPCV efficiency", "Probability of backtest overfitting (PBO)", "T58 score"],
        "reading": ["Below 50: likely curve-fit.", "Blank means no validation has been run yet."],
        "history_field": None, "unit": "/100", "higher_is_better": True,
    },
    "eval_pass_probability": {
        "title": "Evaluation Pass Probability", "aliases": ["pass rate", "pass probability", "eval pass", "eval pass %", "prop pass rate", "probability of passing", "eval %", "eval", "pass %", "p(pass)", "eval-pass %", "eval pass probability", "pass prob"],
        "meaning": "Chance that one purchased evaluation attempt reaches the profit target without breaking a rule.",
        "formula": "Share of Monte Carlo attempts (resampled from your trades) that hit the target before breaching daily loss / drawdown, meeting the minimum days and the consistency rule.",
        "inputs": ["Your trade list (win rate, payoff, frequency)", "The firm's profit target, daily loss limit, drawdown, min days, consistency rule", "Number of simulated attempts"],
        "reading": ["Use the lower confidence bound, not the point estimate.", "Changing risk per trade moves this number faster than anything else."],
        "history_field": "eval_pass_probability", "unit": "%", "higher_is_better": True,
    },
    "first_payout_probability": {
        "title": "Payout Probability", "aliases": ["payout rate", "payout probability", "payout sim rate", "first payout", "first payout %", "payout simulation rate", "payout %", "payout", "p(payout)", "payout prob", "first-payout %", "first payout probability"],
        "meaning": "Chance an attempt both passes the evaluation AND survives to a first payout.",
        "formula": "P(pass) × P(reach payout threshold without breaching | funded)",
        "inputs": ["Evaluation pass probability", "Funded-stage rules (payout threshold, buffer, winning days, consistency)", "Your trade distribution"],
        "reading": ["It is the product of two hurdles, so it is always ≤ pass probability."],
        "history_field": "first_payout_probability", "unit": "%", "higher_is_better": True,
    },
    "oos_pct": {
        "title": "OOS (Out-of-Sample Score)", "aliases": ["oos", "oos %", "oos validation", "out-of-sample", "out of sample", "oos score", "oos efficiency"],
        "meaning": "How well the strategy holds up on data it was NOT tuned on. A strategy that only works on the data it was fitted to is curve-fit; a high OOS says the edge survives on unseen data.",
        "formula": "Best available validation result, as a percent: walk-forward / CPCV efficiency × 100 (out-of-sample performance ÷ in-sample performance), or 100 − PBO (probability of backtest overfitting) when that is what was run.",
        "inputs": ["Walk-forward optimisation, CPCV or walk-forward GA results", "Probability of backtest overfitting (PBO)", "Shows -- until at least one validation has been run"],
        "reading": ["Below 50: likely curve-fit.", "Champion Candidate needs 50 or more.", "Blank (--) means no validation has been run yet -- run Walk-forward or CPCV from the Strategy Lab."],
        "history_field": None, "unit": "%", "higher_is_better": True,
    },
    "promotion_stage": {
        "title": "Stage", "aliases": ["promotion stage"],
        "meaning": "Where the strategy is on the road from idea to something you would trade for real. It only moves forward when it passes that stage's gate.",
        "formula": "Candidate \u2192 Validated \u2192 Champion Candidate \u2192 Forward Testing \u2192 Production Ready. Champion Candidate needs Eval \u2265 60, Payout \u2265 55, OOS \u2265 50 and a verdict that is not NOT READY. Forward Testing needs a READY Champion Check and Eval \u2265 70. Production Ready needs at least 14 days of forward testing.",
        "inputs": ["Eval pass probability", "Payout probability", "OOS score", "Champion Check verdict", "Days of forward testing"],
        "reading": ["Open the Strategy Library to promote a strategy when it qualifies.", "A strategy can never skip a stage."],
        "history_field": None, "unit": "", "higher_is_better": True,
    },
    "risk_of_ruin": {
        "title": "Risk of Ruin", "aliases": ["risk of ruin", "ruin", "ruin %", "risk of ruin %", "ror"],
        "meaning": "Probability the account breaches a loss rule within the horizon (90 days by default).",
        "formula": "Share of simulated accounts that hit the daily-loss or drawdown floor within the ruin horizon",
        "inputs": ["Trade distribution", "Drawdown / daily loss limits", "Risk per trade"],
        "reading": ["Under 10% is comfortable; above 25% means the sizing is too aggressive for the rules."],
        "history_field": "risk_of_ruin_pct", "unit": "%", "higher_is_better": False,
    },
    "avg_win": {
        "title": "Average Win", "aliases": ["avg win", "average win"],
        "meaning": "Mean profit of winning trades.", "formula": "Average Win = gross profit ÷ winning trades",
        "inputs": ["Winning trades"], "reading": ["Compare with average loss: payoff ratio = avg win ÷ |avg loss|."],
        "history_field": None, "unit": "$", "higher_is_better": True,
    },
    "avg_loss": {
        "title": "Average Loss", "aliases": ["avg loss", "average loss"],
        "meaning": "Mean loss of losing trades.", "formula": "Average Loss = gross loss ÷ losing trades",
        "inputs": ["Losing trades"], "reading": ["If it is much larger than your planned risk, stops are slipping or not being honoured."],
        "history_field": None, "unit": "$", "higher_is_better": False,
    },
    "trailing_drawdown": {
        "title": "Trailing Drawdown", "aliases": ["trailing drawdown", "drawdown floor"],
        "meaning": "A loss limit that rises with your best balance and never falls: the floor trails the peak.",
        "formula": "Floor = highest balance reached − max drawdown allowance (stops rising once it reaches the starting balance on most firms)",
        "inputs": ["Peak balance", "Max drawdown allowance"],
        "reading": ["Remaining loss allowance = current balance − floor.", "Winning raises the floor, so banked profit is not extra room."],
        "history_field": None, "unit": "$", "higher_is_better": False,
    },
    "daily_loss_limit": {
        "title": "Daily Loss Limit", "aliases": ["daily loss limit", "daily loss"],
        "meaning": "The most you may lose in one trading day before the account is breached.",
        "formula": "Limit = account size × daily loss %", "inputs": ["Account size", "Daily loss %", "Today's realised P&L"],
        "reading": ["Two or three max-risk losses in a row can use the whole allowance."],
        "history_field": None, "unit": "$", "higher_is_better": False,
    },
    "consistency_rule": {
        "title": "Consistency Rule", "aliases": ["consistency", "consistency rule"],
        "meaning": "No single day may make up more than a set share of your total profit.",
        "formula": "Best day profit ÷ total profit ≤ rule % (e.g. 30%)", "inputs": ["Best single-day profit", "Total profit"],
        "reading": ["It punishes one lucky home-run day; smoother days pass it."],
        "history_field": None, "unit": "%", "higher_is_better": False,
    },
    "net_gex": {
        "title": "Net Gamma Exposure (GEX)", "aliases": ["net gex", "gex", "gamma exposure"],
        "meaning": "Dollar delta that option dealers must trade for a 1% move: positive = they sell rallies / buy dips (calming), negative = they chase moves (amplifying).",
        "formula": "GEX = Σ over strikes of gamma × open interest × 100 × spot² × 1%  (calls +, puts −)",
        "inputs": ["Option gamma", "Open interest per strike", "Spot price"],
        "reading": ["Sign matters more than size.", "Free CBOE data is delayed ~15 min and describes today only."],
        "history_field": None, "unit": "$", "higher_is_better": None,
    },
    "gamma_flip": {
        "title": "Gamma Flip", "aliases": ["gamma flip", "zero gamma"],
        "meaning": "The price where net dealer gamma crosses zero. Above it dealers dampen moves; below it they amplify them.",
        "formula": "Strike where cumulative net GEX (low→high strike) changes sign nearest spot, linearly interpolated",
        "inputs": ["Per-strike net GEX"], "reading": ["A close below the flip in negative gamma favours trend/expansion tactics."],
        "history_field": None, "unit": "", "higher_is_better": None,
    },
    "call_wall": {
        "title": "Call Wall", "aliases": ["call wall"],
        "meaning": "The strike above spot with the largest call gamma — where dealer hedging tends to cap rallies.",
        "formula": "argmax over strikes ≥ spot of call GEX", "inputs": ["Call open interest", "Call gamma"],
        "reading": ["Breaks through it in negative gamma can accelerate."], "history_field": None, "unit": "", "higher_is_better": None,
    },
    "put_wall": {
        "title": "Put Wall", "aliases": ["put wall"],
        "meaning": "The strike below spot with the largest put gamma — a common support where dealer hedging cushions selloffs.",
        "formula": "argmax over strikes ≤ spot of |put GEX|", "inputs": ["Put open interest", "Put gamma"],
        "reading": ["Losing it can unleash selling."], "history_field": None, "unit": "", "higher_is_better": None,
    },
    'trades': {
        "title": 'Trades', "aliases": ['trades', '# trades', 'total trades', 'trade count', 'number of trades', 'trades (n)'],
        "meaning": 'How many round-trip trades the backtest produced. It is the sample size behind every other number on the page.',
        "formula": 'Trades = count of closed round-trip trades in the test window',
        "inputs": ['Every entry/exit pair the strategy took on the data you tested'],
        "reading": ['Under ~100 trades almost any average (win rate, profit factor, Sharpe) can be luck.', "300+ gives a statistically stable read; the discovery engine's entry floor is 100 trades.", 'More trades from loosening entries is not better: judge the edge per trade (mean R) and what it costs per trade.'],
        "history_field": 'trades', "unit": '', "higher_is_better": True,
    },
    'mean_r': {
        "title": 'Average R per trade', "aliases": ['mean r', 'avg r', 'average r', 'expectancy (r)', 'r per trade'],
        "meaning": 'The average profit per trade measured in units of the risk you took (R = 1 initial stop distance in dollars).',
        "formula": 'Mean R = average of (trade P&L ÷ dollars risked on that trade)',
        "inputs": ["Each trade's P&L after costs", "Each trade's initial risk"],
        "reading": ['0.00R is a coin flip after costs. +0.10R is a small but real edge; +0.30R is strong.', 'The t-statistic divides this by its noise: |t| under 1 means it could easily be zero.'],
        "history_field": None, "unit": 'R', "higher_is_better": True,
    },
    't_stat': {
        "title": 't-statistic (edge)', "aliases": ['t-stat', 't stat', 't statistic', 't-statistic', 'edge t'],
        "meaning": 'How many standard errors the average trade sits above zero. It answers: could this edge just be noise?',
        "formula": 't = mean R ÷ (standard deviation of R ÷ √trades)',
        "inputs": ['Mean R', 'Standard deviation of R', 'Number of trades'],
        "reading": ['Below 1 is indistinguishable from nothing; the discovery entry floor needs t ≥ 1.', "Above 2 is conventionally 'significant' — but if you tried hundreds of variants, the multiple-testing deflation matters more than this number."],
        "history_field": None, "unit": '', "higher_is_better": True,
    },
    'dsr': {
        "title": 'Deflated Sharpe (DSR / PSR)', "aliases": ['dsr', 'psr', 'deflated sharpe', 'deflated sharpe ratio', 'probabilistic sharpe', 'unseen dsr'],
        "meaning": "The probability the strategy's Sharpe is really above zero after correcting for how many variants were tried, how long the sample is, and non-normal returns.",
        "formula": "PSR = Φ((SR − SR*) · √(n−1) ÷ √(1 − skew·SR + (kurt−1)/4 · SR²)); DSR sets the benchmark SR* to the best Sharpe you'd expect from N skill-less trials",
        "inputs": ['Observed Sharpe', 'Number of trials (effective, after clustering near-duplicates)', 'Spread of Sharpes across trials', 'Skew and kurtosis of returns'],
        "reading": ['Above 95% survives the selection effect; 50–95% is suggestive; under 50% means the best result is what luck alone would produce.', "This is the main defence against 'I tried 500 settings and the best one looks great'."],
        "history_field": None, "unit": '%', "higher_is_better": True,
    },
    'pbo': {
        "title": 'Probability of Backtest Overfitting (PBO)', "aliases": ['pbo', 'probability of backtest overfitting', 'overfit probability'],
        "meaning": 'How often the setting that looked best in-sample finished below the median out-of-sample, across many train/test splits.',
        "formula": 'PBO = share of CPCV splits where the in-sample winner ranks in the bottom half out-of-sample',
        "inputs": ['Combinatorial train/test splits of your data', "Every candidate setting's performance on each split"],
        "reading": ["Under 25% is healthy; 25–50% is a caution; 50%+ means picking the 'best' setting is no better than picking at random."],
        "history_field": None, "unit": '%', "higher_is_better": False,
    },
    'cpcv': {
        "title": 'CPCV (Combinatorial Purged Cross-Validation)', "aliases": ['cpcv', 'combinatorial purged cross-validation', 'purged cross-validation'],
        "meaning": 'Tests the strategy on many different train/test arrangements of your history, with a gap (purge) so test data never leaks into training.',
        "formula": 'Split into G groups, test on every combination of k groups, purge/embargo around each boundary',
        "inputs": ['Your price history', 'Group count and test-group count (Advanced)'],
        "reading": ['A strategy that works across most splits is robust to *when* it was tested.', 'One lucky stretch of data will show up as a wide spread between splits.'],
        "history_field": None, "unit": '', "higher_is_better": None,
    },
    'monte_carlo': {
        "title": 'Monte Carlo', "aliases": ['monte carlo', 'monte-carlo', 'mc', 'mc sims', 'monte carlo simulations'],
        "meaning": "Re-plays your strategy's trading days in thousands of re-shuffled orders to see how often the account would pass the evaluation or breach a rule.",
        "formula": 'Day-block bootstrap: resample whole trading days with replacement, run the prop-firm rules on each simulated path',
        "inputs": ['Your daily results', "The prop firm's targets, drawdown and daily-loss rules", 'Number of simulations'],
        "reading": ['It shows luck-of-the-order risk, not whether the edge is real — a strategy with no edge can still look fine in Monte Carlo.', 'Read the pass rate together with the breach rate.'],
        "history_field": None, "unit": '', "higher_is_better": None,
    },
    'holdout': {
        "title": 'Holdout (sealed data)', "aliases": ['holdout', 'hold-out', 'holdout window', 'sealed slice', 'sealed window', 'locked holdout'],
        "meaning": "A slice of your data the optimiser never sees. It is read once at the end to check the result wasn't tuned to the past.",
        "formula": 'Holdout = the last X% of the timeline, excluded from every search and tuning step',
        "inputs": ['The reserved date range', 'The number of times it has been looked at (each look is booked)'],
        "reading": ['Every extra look makes the holdout less clean — the app counts them and demands a higher bar each time.', 'If the holdout result is far worse than in-sample, the edge was likely fitted, not found.'],
        "history_field": None, "unit": '', "higher_is_better": None,
    },
    'eval_breach': {
        "title": 'Evaluation breach (ruin)', "aliases": ['evaluation breach', 'eval breach', 'breach rate', 'ruin', 'evaluation breach (ruin)', 'breach %'],
        "meaning": 'Of the evaluation accounts you would buy, the share that break a firm rule (drawdown, daily loss) before passing.',
        "formula": 'Breach % = simulated accounts that hit a loss rule before the profit target ÷ all simulated attempts',
        "inputs": ['Monte Carlo attempts', "The firm's loss rules"],
        "reading": ['Above 20% fails the READY bar: each breach costs a fee and a reset.', "It is counted over the account's lifetime, not just the first N days."],
        "history_field": None, "unit": '%', "higher_is_better": False,
    },
    'cost_stress': {
        "title": 'Cost stress (2× costs)', "aliases": ['cost stress', '2x costs', '2× costs', 'cost ladder', 'point cost ladder', 'slippage stress'],
        "meaning": 'Re-runs the strategy with costs doubled, then steps slippage up (0.5 / 1 / 2 / 3 points on NQ) to find where the edge disappears.',
        "formula": 'Net profit after (commission + slippage × multiplier), per rung of the ladder',
        "inputs": ["Every trade's gross P&L", 'Commission and slippage assumptions'],
        "reading": ['An edge that still earns at 2× costs is real money; one that dies at 1.5× is fragile to a worse fill.', 'Scalpers (small average win) are hit hardest.'],
        "history_field": None, "unit": '', "higher_is_better": None,
    },
    'n_eff': {
        "title": 'Effective number of trials', "aliases": ['n_eff', 'effective trials', 'effective number of trials', 'effective n', 'trial count'],
        "meaning": 'How many genuinely different strategies you really tested, once near-duplicates (very highly correlated variants) are merged.',
        "formula": 'Cluster trial return streams at |correlation| ≥ 0.90 (average linkage); N_eff = number of clusters',
        "inputs": ['Return stream of every variant tried', 'Clustering threshold'],
        "reading": ['Fewer effective trials means a gentler multiple-testing penalty — but the app also shows it at 0.80 and 0.95 so you can see how sensitive the verdict is.'],
        "history_field": None, "unit": '', "higher_is_better": None,
    },
    'ready_bar': {
        "title": 'READY bar', "aliases": ['ready bar', 'ready requirements', 'ready threshold', 'what ready means'],
        "meaning": 'The minimum a strategy must clear to be called READY: evaluation pass rate, payout rate, trade count, out-of-sample quality, a low breach rate and fresh, firm-verified rules.',
        "formula": 'READY = every gate passes simultaneously (any single gate failing caps the verdict)',
        "inputs": ['Eval pass ≥ 70%', 'Breach ≤ 20%', '≥ 300 trades (a 100-trade preflight floor applies earlier)', 'OOS and holdout confirmation', "Rules verified on the firm's page within 30 days"],
        "reading": ["The app's own verdict ladder (Reject → Lead → Forward-Test Candidate → Evaluate → Funded) is stricter and edge-based; READY is the older eval-odds read."],
        "history_field": None, "unit": '', "higher_is_better": None,
    },
    'verdict_ladder': {
        "title": 'Verdict ladder', "aliases": ['verdict ladder', 'ladder', 'forward-test candidate', 'forward test candidate', 'lead', 'evaluate', 'funded'],
        "meaning": 'Five rungs, from Reject to Funded, that say how much evidence the strategy has earned — and what you may do with it.',
        "formula": 'Reject → Lead → Forward-Test Candidate → Evaluate → Funded; each rung needs more evidence than the one below',
        "inputs": ['Holdout confirmation', 'Null/random-entry test', 'Edge after 2× costs', 'Deflated Sharpe', 'Expected value after fees'],
        "reading": ["Forward-Test Candidate means 'worth paper-trading', not 'worth paying for an evaluation'.", 'Evaluate means the maths says buying a challenge has positive expected value.'],
        "history_field": None, "unit": '', "higher_is_better": None,
    },
    'expected_value': {
        "title": 'Expected value (EV) per attempt', "aliases": ['expected value', 'ev', 'ev per attempt', 'ev verdict'],
        "meaning": "What one evaluation purchase is worth on average: the chance of reaching payouts times what you'd collect, minus fees and resets.",
        "formula": 'EV = P(pass) × E[payouts you keep] − challenge fee − expected resets',
        "inputs": ['Pass probability', 'Payout probability and size', 'Firm fee and reset price', 'Profit split'],
        "reading": ['Positive EV after fees is the bar for buying a challenge.', 'If EV is only positive with generous assumptions, treat it as negative.'],
        "history_field": None, "unit": '$', "higher_is_better": True,
    },
}


# ---------------------------------------------------------------------------------------------------------------------
# 2026-11-12 additions: every Validate-gate check, the failure gates the Champion Check names, and the exact wording the
# Library / Dashboard / Champion tiles use ("Eval pass chance", "Payout chance" ...). Optional keys the popup renders when
# present: "how" (calculation, step by step), "bar" (what passes), "fix" (what to do when it does not).
# ---------------------------------------------------------------------------------------------------------------------
_MORE_ALIASES: dict[str, list[str]] = {
    "eval_pass_probability": ["eval pass chance", "evaluation pass chance", "eval pass prob", "eval pass prob %", "eval pass % (per attempt)",
                              "eval pass probability (per attempt)", "evaluation pass probability", "eval pass rate", "eval-pass probability",
                              "optimized eval pass", "baseline eval pass", "per-attempt eval pass", "per-attempt pass", "pass chance"],
    "first_payout_probability": ["payout chance", "payout % (per attempt)", "payout prob %", "payout probability (per attempt)",
                                 "first payout chance", "payout sim", "payout sim %", "payout probability %", "first-payout probability"],
    "sharpe_ratio": ["best sharpe", "sharpe ratio (per trade)", "sharpe (per trade)"],
    "win_rate": ["win", "winrate", "win % (trades)", "optimized win rate"],
    "net_profit": ["net profit ($)", "net p&l ($)", "optimized net profit"],
    "max_drawdown": ["max dd (%)", "drawdown (max)", "max drawdown (backtest)"],
    "profit_factor": ["pf (profit factor)", "profit factor (gross)"],
    "oos_pct": ["oos (out-of-sample score)", "oos score %"],
    "risk_of_ruin": ["ruin risk", "risk of ruin (ror)"],
    "cpcv": ["cpcv (combinatorial purged cross-validation)", "cpcv pf", "cpcv oos pf", "cpcv / pbo"],
    "pbo": ["pbo (overfitting)", "pbo (overfit)"],
    "walk_forward": ["walk-forward opt", "walk forward optimization", "walk-forward optimisation"],
    "wfga": ["walk-forward ga", "wf ga"],
    "monte_carlo": ["monte carlo (prop rules)"],
}

METRICS.update({
    "walk_forward": {
        "title": "Walk-Forward Optimization", "aliases": ["walk-forward", "walk forward", "walk-forward opt", "walk-forward optimization", "wfo",
                                                           "walk-forward efficiency", "out-of-sample efficiency", "oos efficiency"],
        "meaning": "The closest thing to live trading that history allows: the strategy is re-optimized on one stretch of data, then traded blind on the next stretch it has never seen, over and over.",
        "formula": "OOS efficiency = out-of-sample performance ÷ in-sample performance, averaged over the folds",
        "how": ["Cut the history into rolling (or anchored) folds, each with a training window followed by a test window.",
                "Optimize the settings on the training window only.",
                "Trade the NEXT, unseen test window with those settings and record the result.",
                "Repeat for every fold, then divide the out-of-sample performance by the in-sample performance."],
        "inputs": ["Your price history", "Number of folds and window sizes (Advanced > Walk-forward)", "The optimizer's search ranges"],
        "bar": "Pass: keeps 50% or more of its in-sample edge out-of-sample. Borderline: 25–50%. Fail: under 25%.",
        "reading": ["High efficiency means re-tuning on the past still worked on the future; low means the in-sample result was fit to noise.",
                    "Needs enough trades in each test window; with too few it returns no figure and shows a caution."],
        "fix": ["Use fewer optimizable parameters or narrower ranges.", "Give each window more trades (longer history or a faster timeframe).",
                "Try an anchored window if rolling windows are too short."],
        "history_field": None, "unit": "", "higher_is_better": True,
    },
    "wfga": {
        "title": "Walk-Forward GA (genetic search on unseen folds)", "aliases": ["walk-forward ga", "wfga", "walk forward ga", "walk-forward genetic", "overfitting gap"],
        "meaning": "A genetic search that is scored only on folds it did not train on. The gap between its in-sample and out-of-sample fitness shows how over-fit the chosen settings are. It is the slowest check.",
        "formula": "Overfitting gap = in-sample fitness − out-of-sample fitness; judged as gap ÷ |in-sample fitness|",
        "how": ["Split the history into train and test folds.", "Let a genetic algorithm evolve settings using the training folds.",
                "Score the winning settings on the held-out folds.", "Compare the two fitness values: the shortfall is the overfitting gap."],
        "inputs": ["Population and generations (Advanced > Walk-forward)", "Fitness metric", "Out-of-sample trade count"],
        "bar": "Pass: gap is 15% of in-sample fitness or less. Borderline: 15–30%. Fail: over 30%.",
        "reading": ["A small gap says the GA found something that generalizes; a big gap says it memorized the training folds.",
                    "Check the OOS trade count: a handful of trades makes the gap meaningless."],
        "fix": ["Reduce the parameter ranges the GA can explore.", "Lower generations/population so it cannot memorize noise.",
                "Make sure each test fold has enough trades to be meaningful."],
        "history_field": None, "unit": "", "higher_is_better": None,
    },
    "sensitivity": {
        "title": "Sensitivity (1-D parameter sweep)", "aliases": ["sensitivity", "sensitivity (1-d)", "1-d sensitivity", "parameter sensitivity", "cliff edges", "cliff edge"],
        "meaning": "Wiggles every number in the strategy one at a time and watches what happens. A real edge degrades gently when a setting moves; a curve-fit one falls off a cliff.",
        "formula": "For each parameter: evaluate the strategy at N evenly spaced values around the chosen one (others held fixed); flag a cliff where the metric drops sharply between neighbouring steps",
        "how": ["List every tunable parameter.", "Re-run the backtest at several values either side of the current setting, holding everything else fixed.",
                "Compare neighbouring results: a sharp drop between adjacent steps is a cliff, a gradual slope is a plateau.",
                "Count the parameters that sit on a cliff."],
        "inputs": ["Each tunable parameter and its normal search bounds", "Sweep width and step count (Advanced > Sensitivity)"],
        "bar": "Pass: no cliff edges. Borderline: up to one in three parameters on a cliff. Fail: more than a third.",
        "reading": ["Fragile parameters are named in the result: those are the values to widen or re-pick from the middle of a flat region."],
        "fix": ["Open Sensitivity to see which parameter is fragile.", "Re-optimize and choose the middle of the flat region, not the peak.",
                "Smooth the logic (for example a threshold band instead of an exact level)."],
        "history_field": None, "unit": "", "higher_is_better": None,
    },
    "parameter_robustness": {
        "title": "Parameter Robustness (plateau vs cliff)", "aliases": ["parameter robustness", "robustness score (parameters)", "plateau vs cliff", "parameter stability", "parameter_stability"],
        "meaning": "One 0–100 score for how wide the good area is around your chosen settings. A broad plateau means the edge does not depend on pinpoint values.",
        "formula": "Score = 0–100 summary of the 1-D sweeps and 2-D heatmaps: how much of the neighbourhood stays about as good as the chosen point, minus a penalty for each cliff",
        "how": ["Reuse the Sensitivity sweeps (1-D) and pair heatmaps (2-D).", "Measure how much of the nearby area stays near the chosen result.",
                "Penalize every parameter flagged as a cliff.", "Combine into a single 0–100 score."],
        "inputs": ["Sensitivity curves and grids", "Cliff flags per parameter"],
        "bar": "Pass: 70/100 or more. Borderline: 50–70. Fail: under 50.",
        "reading": ["Below 50 means the strategy only works at a pinpoint setting."],
        "fix": ["Re-optimize preferring broad plateaus.", "Drop parameters that do not add edge.", "Re-run this check after changing."],
        "history_field": None, "unit": "/100", "higher_is_better": True,
    },
    "regime_matrix": {
        "title": "Regime Matrix", "aliases": ["regime matrix", "regime survival matrix", "regime split", "cells to switch off"],
        "meaning": "Shows which kinds of market (trending vs ranging, quiet vs wild, which session) the strategy's trades won or lost in, and flags the regimes it should stand aside from.",
        "formula": "Run the backtest once; classify every bar into a regime on four dimensions (trend, volatility, session, plus a combined view); attribute each trade to the regime active at its entry; flag regime cells whose trades lose money",
        "how": ["Backtest the strategy once over the whole history.", "Label each bar using only past data: trend (5 bins), volatility (5 bins), session (Asia/London/NY…).",
                "Assign every trade to the regime at its entry time.", "Sum each regime's results and flag the losing cells with enough trades to matter."],
        "inputs": ["Your trades", "Price history for the regime labels"],
        "bar": "Pass: no regime loses money. Borderline: one regime to switch off. Fail: two or more.",
        "reading": ["It answers 'where should this strategy NOT trade', not 'is the edge real'.",
                    "A flagged regime is a candidate filter: test it on data the filter never saw before trusting it."],
        "fix": ["Press Auto-fix on the Validate report: it finds the losing regimes on the first 70% of data, adds a stand-aside filter and keeps it only if the last 30% holds up.",
                "Or open the Regime Matrix to inspect the regimes and test an exclusion by hand."],
        "history_field": None, "unit": "", "higher_is_better": None,
    },
    "single_run_pass_rate": {
        "title": "Single-run pass rate", "aliases": ["single-run pass rate", "single run pass rate", "single-run pass", "single run pass"],
        "meaning": "Of your strategies, the share whose ONE historical run (a single account started at the beginning of the data) reached the profit target without breaking a rule.",
        "formula": "Strategies whose latest single historical run passed ÷ strategies with a recorded run",
        "how": ["Take the latest run of each strategy.", "Play the trades once, in order, through the prop-firm rules.", "Count the runs that hit the target before a limit."],
        "inputs": ["Latest run per strategy", "The prop-firm rules used in that run"],
        "reading": ["It is one path, not a probability: 100% here can sit beside a failing Monte Carlo or out-of-sample check.", "Use Eval pass chance for the per-attempt probability."],
        "history_field": None, "unit": "%", "higher_is_better": True,
    },
    "champion_check": {
        "title": "Champion Check (READY / MARGINAL / NOT READY)", "aliases": ["champion check", "champion verdict", "full pipeline verdict"],
        "meaning": "The Full Pipeline's final review. It applies hard safety gates first (they cap the verdict whatever the score) and then the T58 score. Any failed gate is listed by name with the number that failed.",
        "formula": "Verdict = READY if every hard gate passes and the T58 score is 75+; MARGINAL if something was unproven or unmeasured; NOT READY if a hard gate failed or the score is weak",
        "how": ["Hard gates: Monte Carlo risk of ruin, lookahead leak, out-of-sample tests, deflated Sharpe, overfitting probability, holdout.",
                "Score the surviving evidence 0–100 (T58 score).", "Apply the edge ladder and the expected-value test on top."],
        "inputs": ["Development-window trades", "Out-of-sample folds", "Sealed holdout", "Prop-firm rules"],
        "reading": ["UNPROVEN means 'positive but not yet statistically separated from zero' — it caps at MARGINAL, it does not reject.",
                    "Re-running the pipeline on the same data and settings reproduces the same verdict; change an input to change the outcome."],
        "history_field": None, "unit": "", "higher_is_better": None,
    },
    "pooled_oos": {
        "title": "Pooled out-of-sample edge (mean R and lower bound)", "aliases": ["pooled oos", "pooled oos mean r", "oos mean r", "oos lower bound", "lower bound", "pooled out-of-sample"],
        "meaning": "The average result per trade (in R) across ALL out-of-sample trades merged together, and how low it could plausibly be. The lower bound is what decides whether the edge is proven.",
        "formula": "Mean R = average of (trade P&L ÷ risk). Lower bound = 95% day-block bootstrap lower bound of that mean. Need: lower bound > 0 and more than half of the merged folds positive.",
        "how": ["Merge the test windows of every out-of-sample fold.", "Compute the mean R of those trades.",
                "Resample whole trading days thousands of times to get a 95% confidence interval.", "Read the lower end of the interval."],
        "inputs": ["Out-of-sample trades", "Their R multiples", "Number of trading days"],
        "bar": "Proven: lower bound above zero. UNPROVEN: mean positive but lower bound at or below zero (caps at MARGINAL). Fail: mean not positive or under half the folds positive.",
        "reading": ["The lower bound shrinks roughly with the square root of the trade count, so doubling the out-of-sample trades tightens it by about 30%.",
                    "With a positive mean and few trades, the fix is more evidence, not different settings."],
        "fix": ["Add out-of-sample trades: a longer history, a faster timeframe or looser entries (without lowering the average R).",
                "Run Optimize with the 'Make the edge more reliable' goal: it scores on this lower bound."],
        "history_field": None, "unit": "R", "higher_is_better": True,
    },
    "dev_vs_full_window": {
        "title": "Development window vs full history", "aliases": ["development window", "dev window", "full history", "full-history window", "sealed holdout window", "scored window"],
        "meaning": "The Full Pipeline tunes and scores on the first 80% of the history (the development window) and seals the last 20% as a holdout. Reports and charts can show either that 80%, or the full history including the holdout, so trade counts and profit differ between them.",
        "formula": "Full history = development window + sealed holdout. Example: 390 development trades + 73 holdout trades = 463 on the full history.",
        "inputs": ["Development-window trades", "Holdout trades", "Where the holdout starts"],
        "reading": ["Both are correct; they measure different stretches of time. Every number in the app now says which window it was measured on.",
                    "Pass chances and the verdict are scored on the development window only, by design, so the holdout stays unseen."],
        "history_field": None, "unit": "", "higher_is_better": None,
    },
    "prop_fitness": {
        "title": "Prop Fitness", "aliases": ["prop fitness", "propfit"],
        "meaning": "A single score ranking how well a strategy suits the chosen prop-firm rules. It is used to order candidates, never to give a verdict.",
        "formula": "Weighted blend of pass probability, payout probability, drawdown headroom and survival; the verdict still comes from the Champion Check",
        "inputs": ["Pass and payout probabilities", "Drawdown versus the firm's limit", "Survival"],
        "reading": ["Compare it only between candidates run on the same prop rules.", "It never overrides the 70% per-attempt pass gate."],
        "history_field": None, "unit": "", "higher_is_better": True,
    },
})

for _k, _extra in _MORE_ALIASES.items():
    if _k in METRICS:
        METRICS[_k]["aliases"] = list(METRICS[_k].get("aliases", [])) + [a for a in _extra if a not in METRICS[_k].get("aliases", [])]

# "how it is calculated" for the headline pass chances (the two the Library tiles never explained)
METRICS["eval_pass_probability"].update({
    "how": ["Take the strategy's trades from the development window.", "Resample whole trading days with replacement to build thousands of alternative account histories.",
            "Run each through the prop firm's rules: profit target, daily loss, max drawdown, minimum days, consistency.",
            "Count the share that hit the target before breaking a rule. That share is the eval pass chance."],
    "bar": "READY bar: 70% or more per attempt (judged on the lower confidence bound).",
    "fix": ["Halve the risk per trade and re-run: risk size moves this number faster than anything else.",
            "Run Optimize with the 'Pass the evaluation' goal.", "Try a prop-firm preset whose drawdown rules fit the strategy."],
})
METRICS["first_payout_probability"].update({
    "how": ["Start from the simulated accounts that passed the evaluation.", "Continue each one through the funded-stage rules (payout threshold, buffer, winning days, consistency).",
            "Count those that reach a first payout without breaching.", "Payout chance = P(pass) × P(reach payout | funded), so it can never exceed the eval pass chance."],
    "bar": "Compared with the eval pass chance: a big gap means accounts pass but do not last.",
    "fix": ["Lower risk per trade or tighten the stop so funded accounts survive to the first payout.", "Run Optimize with the 'Get to a payout faster' goal."],
})
for _k in ("cpcv",):
    METRICS[_k].update({
        "how": ["Cut the history into G time blocks.", "For every combination of k blocks held out as test, train on the rest (with a purge gap so test data never leaks in).",
                "Evaluate each split out-of-sample.", "Compare out-of-sample with in-sample across all the splits."],
        "bar": "Pass: out-of-sample results keep most of the in-sample edge on the paths tested.",
        "fix": ["Remove parameters until only the few that matter remain.", "Re-optimize on a longer history.", "Prefer values in the middle of a flat region over the single best spike."],
    })
METRICS["pbo"].update({
    "how": ["Split the history into blocks and form many train/test combinations (the same splits CPCV uses).",
            "In each split, pick the variant that looks best in-sample.",
            "Check where that pick ranks out-of-sample versus the other variants.",
            "PBO = share of splits where the in-sample winner lands in the bottom half out-of-sample."],
    "bar": "Pass: under 35%. Borderline: 35–50%. Fail: 50% or more.",
    "fix": ["Test fewer, more distinct ideas instead of many near-identical variants.", "Use a longer history.", "Prefer simpler logic with fewer free parameters."],
})

_ALIAS: dict[str, str] = {}
for _k, _v in METRICS.items():
    _ALIAS[_k] = _k
    for _a in _v.get("aliases", []):
        _ALIAS[_a.lower()] = _k


def resolve(label_or_key: str) -> Optional[str]:
    s = re.sub(r"\s+", " ", str(label_or_key or "").strip().lower())
    return _ALIAS.get(s) or _ALIAS.get(s.replace("_", " "))


def definition(key: str) -> Optional[dict]:
    k = resolve(key)
    if not k:
        return None
    d = dict(METRICS[k])
    d["key"] = k
    return d


def all_labels() -> dict[str, str]:
    """alias (lower-case) -> metric key, for the client-side auto-binder."""
    return dict(_ALIAS)


def history_for(key: str, limit: int = 200) -> dict:
    k = resolve(key)
    d = METRICS.get(k or "")
    out: dict[str, Any] = {"key": k, "rows": [], "summary": None}
    if not d or not d.get("history_field"):
        return out
    from app.reports import run_history

    field = d["history_field"]
    rows = []
    for r in run_history.load_runs():
        v = r.get(field)
        if isinstance(v, (int, float)) and v == v:
            rows.append({"date": str(r.get("timestamp", ""))[:10], "strategy": r.get("strategy_name"),
                         "instrument": r.get("instrument"), "value": round(float(v), 3)})
    rows.sort(key=lambda x: x["date"])
    rows = rows[-limit:]
    out["rows"] = rows
    if rows:
        vals = [x["value"] for x in rows]
        hib = d.get("higher_is_better", True)
        out["summary"] = {"count": len(vals), "latest": vals[-1], "average": round(sum(vals) / len(vals), 3),
                          "best": max(vals) if hib is not False else min(vals),
                          "worst": min(vals) if hib is not False else max(vals)}
    return out
