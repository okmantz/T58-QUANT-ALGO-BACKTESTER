"""
Education tab content -- teaches quantitative / algorithmic trading concepts
AND how this specific app's tools answer them, in the same "write once, both
UIs read it" spirit as app.orchestration.pipeline_guide (that module answers
"what do I do next"; this one answers "what does that even mean").

Deliberately NOT a generic trading course. Every lesson exists because this
app produces a number, a tab, or a decision point that needs it -- a lesson
with no corresponding real feature in this app doesn't belong here. Kept
short on purpose (most lessons are a 2-4 block read, matching the app's own
"2-5 minutes, not a 100-lesson course" design goal): the strongest angle
this app has is "learn quantitative strategy research by actually doing it
here," not a from-scratch trading education product.

Structure:
    EducationSection  (a sidebar/TOC entry, e.g. "Validation Lab")
        -> Lesson      (one short, focused topic)
            -> Block    (a paragraph / bullet list / tip / warning / numbered
                         steps -- kept as structured data, not raw HTML or
                         Tkinter text-tag calls, so app.web.server can render
                         it as HTML and app.ui.main_window can render the
                         EXACT same content into its Text-widget tag system
                         without either one re-authoring the copy.)

Every lesson's `see_also` field names REAL tabs/routes in this app (the
same strings app.orchestration.pipeline_guide and app.ui.main_window's own
sidebar already use) -- never an invented or generic pointer.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Block:
    kind: str                      # "p" | "bullets" | "steps" | "tip" | "warn" | "example"
    text: str = ""                 # for "p", "tip", "warn", "example"
    items: tuple[str, ...] = ()    # for "bullets", "steps"


def p(text: str) -> Block:
    return Block("p", text=text)


def bullets(*items: str) -> Block:
    return Block("bullets", items=items)


def steps(*items: str) -> Block:
    return Block("steps", items=items)


def tip(text: str) -> Block:
    return Block("tip", text=text)


def warn(text: str) -> Block:
    return Block("warn", text=text)


def example(text: str) -> Block:
    """A short worked example or illustrative quote -- rendered visually
    distinct from ordinary body text (a bordered/mono-leaning block) in
    both UIs, e.g. the "70% win rate can still lose money" case."""
    return Block("example", text=text)


@dataclass(frozen=True)
class Lesson:
    id: str
    title: str
    blocks: tuple[Block, ...]
    see_also: tuple[str, ...] = ()


@dataclass(frozen=True)
class EducationSection:
    id: str
    icon: str
    title: str
    subtitle: str
    lessons: tuple[Lesson, ...]


# ---------------------------------------------------------------------------
# 1. Getting Started
# ---------------------------------------------------------------------------

_GETTING_STARTED = EducationSection(
    id="getting-started", icon="\U0001F680", title="Getting Started",
    subtitle="What backtesting actually is, and your first result in about ten minutes.",
    lessons=(
        Lesson(
            id="what-is-backtesting", title="What Is Backtesting?",
            blocks=(
                p(
                    "Backtesting runs a trading strategy's rules against historical price data to see what "
                    "would have happened -- how many trades it would have taken, whether it made or lost "
                    "money, and how deep its losing stretches got. It answers 'would this have worked?', "
                    "never 'will this work?' -- the market that produced your historical data is gone, and "
                    "the future doesn't have to resemble it."
                ),
                p(
                    "That gap between 'worked in the past' and 'will work going forward' is the entire "
                    "reason the rest of this app exists past the first backtest -- Search Lab, Evolution "
                    "Lab, CPCV/PBO, Monte Carlo, and Full Pipeline are all different ways of stress-testing "
                    "whether a historical result is likely to generalize, or whether it's a fluke this one "
                    "dataset happened to produce."
                ),
                tip(
                    "Treat every single backtest as a hypothesis test, not a verdict. One green result is "
                    "the START of the research process here, not the end of it -- see the Validation Lab "
                    "and Prop Firm Evaluation sections below for what comes next."
                ),
            ),
            see_also=("5 Run & Report",),
        ),
        Lesson(
            id="what-makes-a-backtest-trustworthy", title="What Makes a Backtest Trustworthy?",
            blocks=(
                p("Four things separate a backtest you can act on from one that's quietly lying to you:"),
                bullets(
                    "Enough trades -- a handful of trades can't distinguish skill from luck. This app's "
                    "own guidance treats anything under ~20 trades as too few to trust a pass-probability "
                    "number from.",
                    "Realistic costs -- spread, slippage, and commission all eat into an edge that looks "
                    "great with zero friction. The Cost Stress tool re-runs your trades at increasingly "
                    "unfavorable costs so you can see how much edge survives.",
                    "No lookahead bias -- the strategy can only see information that would have actually "
                    "been available at the moment it decided to trade, never a future bar's close.",
                    "Out-of-sample confirmation -- the result holds up on data the strategy (or your own "
                    "eyeballing-and-tweaking) never touched while it was being built.",
                ),
                p(
                    "None of these four require a finance degree to check -- they're exactly what the "
                    "Validation Lab section below, and the tabs it describes, are for."
                ),
            ),
        ),
        Lesson(
            id="first-strategy-ten-minutes", title="Your First Strategy in About Ten Minutes",
            blocks=(
                steps(
                    "Load market data on 2 Market Data -- a CSV of OHLCV candles, or fetch some with a "
                    "free Alpaca key.",
                    "Build a strategy on 1 Strategy Configuration. No idea yet? Try Manual Builder's "
                    "indicator crossovers, or ask CREATE -> Generate Strategies (AI) to draft one from a "
                    "plain-language description.",
                    "Click RUN on 5 Run & Report.",
                    "Read the result using the Reading Your Results section below.",
                    "Not promising? That's normal and useful -- the green 'Next step' note under your "
                    "result tells you exactly what to try next.",
                ),
                tip(
                    "No specific idea at all? Skip straight to Search Lab or Evolution Lab (OPTIMIZE "
                    "section) instead of starting from a blank Manual Builder -- both hand you a tested "
                    "leaderboard to start from rather than a blank page."
                ),
            ),
            see_also=("1 Strategy Configuration", "2 Market Data", "5 Run & Report"),
        ),
        Lesson(
            id="from-backtest-to-validation", title="From One Backtest to a Validated Strategy",
            blocks=(
                p(
                    "A single backtest tells you what happened on one specific run of history. It cannot "
                    "tell you whether that result depended on a lucky parameter value, a specific market "
                    "regime, or a trade sequence that happened not to hit a losing streak at a bad time. "
                    "Those are three separate, real risks -- and this app has a separate tool for each one."
                ),
                bullets(
                    "Lucky parameter value -> Sensitivity / Parameter Stability tabs.",
                    "Depends on one market regime -> Regime Survival Matrix.",
                    "Got a favorable trade sequence -> Monte Carlo, which resamples the sequence thousands "
                    "of times.",
                    "Might just be the best of many tried candidates -> CPCV / PBO.",
                ),
                p(
                    "Full Pipeline (OPTIMIZE section) runs the most important of these together and gives "
                    "you one READY / MARGINAL / NOT READY verdict -- see the T58 Research Process section "
                    "for how all of this fits into one ordered workflow."
                ),
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# 2. Strategy Development
# ---------------------------------------------------------------------------

_STRATEGY_DEVELOPMENT = EducationSection(
    id="strategy-development", icon="\U0001F9E0", title="Strategy Development",
    subtitle="How to build a strategy worth testing, before you touch any statistics.",
    lessons=(
        Lesson(
            id="start-with-a-hypothesis", title="Start With a Hypothesis, Not a Parameter Search",
            blocks=(
                p(
                    "A trading hypothesis is a specific, falsifiable claim about WHY price tends to behave "
                    "a certain way under certain conditions -- 'liquidity sweeps above a prior high tend to "
                    "reverse on the lower timeframe when the higher-timeframe trend is down' is a "
                    "hypothesis. 'Try every combination of these 12 indicator parameters and keep whatever "
                    "scores highest' is not -- it's a search, and a search with no hypothesis behind it is "
                    "the single most common way backtests find noise instead of an edge."
                ),
                p(
                    "This matters practically, not just philosophically: a hypothesis tells you which "
                    "parameter RANGES are even worth searching (see Choosing Parameters Without "
                    "Overfitting below), and it gives you something to fall back on when a backtest looks "
                    "bad -- was the hypothesis wrong, or was the implementation wrong? A pure parameter "
                    "search gives you no way to tell the difference."
                ),
                tip(
                    "Search Lab and Evolution Lab are still useful with no starting hypothesis -- they "
                    "exist specifically for exploring when you don't have one yet. Just expect to spend "
                    "more of the Validation Lab's tools confirming whatever they find, since nothing "
                    "grounds it yet."
                ),
            ),
        ),
        Lesson(
            id="entry-exit-logic", title="Entry & Exit Logic",
            blocks=(
                p(
                    "Manual Builder's four condition sets -- long_entry, long_exit, short_entry, "
                    "short_exit -- are where a hypothesis becomes a rule a computer can act on. It's easy "
                    "to spend all your design effort on the entry and treat the exit as an afterthought, "
                    "but the exit decides how big your losses and wins actually are -- two strategies with "
                    "identical entries and different exits can have completely different profit factors."
                ),
                bullets(
                    "A precise entry with a vague exit ('close when it feels right') can't be backtested "
                    "consistently at all.",
                    "An exit that's too tight relative to normal price noise gets stopped out constantly "
                    "on trades that would have worked.",
                    "An exit with no stop at all turns one bad trade into an account-ending one -- see "
                    "Stops, Targets, and Position Sizing next.",
                ),
            ),
            see_also=("1 Strategy Configuration",),
        ),
        Lesson(
            id="stops-targets-position-sizing", title="Stops, Targets, and Position Sizing",
            blocks=(
                p(
                    "Stop-loss and take-profit distances (stop_loss_pips / take_profit_pips in Manual "
                    "Builder) are part of the STRATEGY -- they change which trades win and lose and by how "
                    "much. Position size and risk-per-trade, on 4 Risk & Execution, are a separate decision "
                    "about how much of the account to risk on a signal the strategy already generated. "
                    "Conflating the two is a common source of confusion: a strategy with a great edge and "
                    "reckless position sizing will still blow up an account, and the fix belongs on 4 Risk "
                    "& Execution, not in the strategy's own rules."
                ),
                tip(
                    "If a strategy's max drawdown looks fine but a Monte Carlo run shows a high "
                    "risk_of_ruin_pct, that's usually a position-sizing problem, not a strategy problem -- "
                    "lower risk per trade before touching the entry/exit logic."
                ),
            ),
            see_also=("4 Risk & Execution",),
        ),
        Lesson(
            id="timeframe-market-selection", title="Timeframe & Market Selection",
            blocks=(
                p(
                    "A hypothesis built around structure that forms over hours won't show up cleanly on "
                    "1-minute bars, and a scalping-style hypothesis tested on daily bars will generate too "
                    "few trades to mean anything (see Trade Count & Statistical Significance). Match the "
                    "data's timeframe to the timeframe the hypothesis actually describes before concluding "
                    "anything from a zero-trades or few-trades result."
                ),
                p(
                    "The same applies across instruments -- a mean-reversion idea built around a "
                    "range-bound currency pair has no reason to transfer to a trending commodity. Test on "
                    "the market the hypothesis is actually about before generalizing a result to others."
                ),
            ),
            see_also=("2 Market Data",),
        ),
        Lesson(
            id="regime-filters", title="Regime Filters",
            blocks=(
                p(
                    "Almost every strategy has a market condition it's genuinely good in and one it's "
                    "genuinely bad in -- a breakout system in a trending market, the same system chopped "
                    "apart in a range. A regime filter is a rule that recognizes which condition is "
                    "currently in play and sits out (or switches behavior) when it's the bad one, instead "
                    "of hoping the edge holds everywhere."
                ),
                p(
                    "The Regime Survival Matrix tab buckets your data by volatility regime and shows "
                    "whether a strategy is profitable in each bucket or only in one -- a strategy that's "
                    "only profitable in one of three regimes isn't broken, but it needs either a filter "
                    "for the other two, or an honest expectation that it will sit idle (or lose) when the "
                    "market isn't in its regime."
                ),
            ),
            see_also=("Regime Survival Matrix",),
        ),
        Lesson(
            id="choosing-parameters-without-overfitting", title="Choosing Parameters Without Overfitting",
            blocks=(
                p(
                    "Every tunable number in a strategy (a lookback period, a threshold, a multiplier) is "
                    "a place overfitting can hide. The defense isn't avoiding parameters entirely -- it's "
                    "choosing their SEARCH RANGE from the hypothesis, not from whatever range happens to "
                    "produce the best backtest, and then checking that nearby values work about as well as "
                    "the chosen one (see The Neighboring-Parameter Test in Validation Lab)."
                ),
                warn(
                    "A parameter search with no hypothesis-driven range and no neighboring-value check is "
                    "just curve-fitting with extra steps -- it will always find SOMETHING that worked on "
                    "this exact dataset."
                ),
            ),
            see_also=("Sensitivity", "Parameter Stability / Robustness Map"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# 3. Backtesting Statistics
# ---------------------------------------------------------------------------

_BACKTESTING_FUNDAMENTALS = EducationSection(
    id="backtesting-fundamentals", icon="\U0001F4CA", title="Backtesting Statistics",
    subtitle="What each number on the Run & Report screen actually means, and what can fool you.",
    lessons=(
        Lesson(
            id="win-rate-avg-win-loss", title="Win Rate & Average Win/Loss",
            blocks=(
                p(
                    "Win rate alone tells you almost nothing about profitability -- it has to be read "
                    "together with the average win and average loss size."
                ),
                example(
                    "A 70% win rate doesn't automatically mean a strategy is good: if the average loss is "
                    "3x the average win, this strategy still loses money overall (7 x 1 win = 7, 3 x 3 "
                    "losses = 9 lost -- net negative despite winning most of the time)."
                ),
                p(
                    "This is what the average_r figure (the average result in units of risk taken, "
                    "sometimes called R-multiple) captures in one number -- a strategy with a low win rate "
                    "but a high average R can still be strongly profitable, and vice versa."
                ),
            ),
        ),
        Lesson(
            id="profit-factor-expectancy", title="Profit Factor & Expectancy",
            blocks=(
                p(
                    "Profit factor is gross profit divided by gross loss -- above 1.0 means the strategy "
                    "made more than it lost overall; below 1.0 means it lost money outright regardless of "
                    "win rate. This app's own scoring treats 1.3+ as a healthy target band, not just "
                    "'above 1.0' -- a strategy sitting right at 1.0-1.1 is fragile to even small changes in "
                    "costs or market conditions."
                ),
                p(
                    "Expectancy is the average profit or loss per trade in dollar or pip terms -- it "
                    "answers 'if I take 100 more trades like these, what do I expect to make per trade on "
                    "average', which is a more direct number to size a position against than profit factor "
                    "alone."
                ),
            ),
        ),
        Lesson(
            id="drawdown", title="Drawdown",
            blocks=(
                p(
                    "Max drawdown is the largest peak-to-trough decline the equity curve experienced. It "
                    "matters more than most new traders expect, for a concrete reason: a 50% drawdown "
                    "needs a 100% gain just to get back to even. Deep drawdowns are also almost always the "
                    "single-run killer in prop-firm evaluations (see Prop Firm Evaluation below) -- a "
                    "strategy can be profitable overall and still fail an evaluation purely on drawdown."
                ),
                warn(
                    "This app's own guidance treats max drawdown above ~40% as severe regardless of how "
                    "good profit factor looks -- most prop firms cap overall drawdown around 8-10%, so a "
                    "backtest drawdown anywhere near 40% will fail almost every prop-firm simulation."
                ),
            ),
            see_also=("4 Risk & Execution", "6 Payout Probability"),
        ),
        Lesson(
            id="sharpe-sortino-calmar", title="Sharpe, Sortino, and Calmar",
            blocks=(
                p("All three are risk-adjusted return measures -- return relative to some kind of risk, not return alone:"),
                bullets(
                    "Sharpe ratio -- return relative to overall volatility (both up and down swings count "
                    "against it).",
                    "Sortino ratio -- return relative to DOWNSIDE volatility only, which is usually the "
                    "more relevant one for a trading strategy since upside swings aren't the risk you're "
                    "trying to avoid.",
                    "Calmar ratio -- return relative to max drawdown specifically, which ties it directly "
                    "back to the capital-survival concern above.",
                ),
                p(
                    "None of these three replace looking at trade count, drawdown, and profit factor "
                    "directly -- they're a useful single-number summary for COMPARING two already-plausible "
                    "candidates, not a first-pass filter on their own."
                ),
            ),
        ),
        Lesson(
            id="trade-count-significance", title="Trade Count & Statistical Significance",
            blocks=(
                p(
                    "A handful of trades can't distinguish a real edge from a coin flip that happened to "
                    "land favorably a few times in a row. This app treats under ~20 trades as too few for "
                    "Monte Carlo or a prop-firm simulation to say anything meaningful -- both tools "
                    "resample from whatever trades exist, so a small sample just gets resampled a lot, not "
                    "made more reliable."
                ),
                tip(
                    "If a promising strategy has too few trades, the fix is a longer data range, a lower "
                    "timeframe, or a slightly looser entry condition -- not trusting the small sample's "
                    "pass-probability number as-is."
                ),
            ),
        ),
        Lesson(
            id="equity-curves-recovery", title="Equity Curves & Recovery Periods",
            blocks=(
                p(
                    "The shape of an equity curve carries information a single ending number doesn't: a "
                    "smooth, steadily rising curve and a curve that spikes once and then grinds sideways "
                    "can end at the exact same net profit while being completely different strategies to "
                    "actually trade."
                ),
                p(
                    "Recovery period -- how long it takes to make a new equity high after a drawdown -- "
                    "matters for the same practical reason drawdown itself does: a strategy that recovers "
                    "in days is a very different psychological and capital-allocation experience than one "
                    "that takes months, even with an identical max drawdown number."
                ),
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# 4. Validation Lab -- is your edge real?
# ---------------------------------------------------------------------------

_VALIDATION_LAB = EducationSection(
    id="validation-lab", icon="\U0001F9EA", title="Validation Lab -- Is Your Edge Real?",
    subtitle="How to tell a genuine edge apart from a backtest that got lucky. Start here if you only read one section.",
    lessons=(
        Lesson(
            id="in-sample-out-of-sample", title="In-Sample vs Out-of-Sample",
            blocks=(
                p(
                    "In-sample data is whatever you looked at, tweaked against, or optimized on while "
                    "building the strategy. Out-of-sample data is data the strategy (and you) never saw "
                    "during that process. A result on in-sample data tells you the strategy can describe "
                    "the past it was built from -- only an out-of-sample result tells you anything about "
                    "whether that description generalizes."
                ),
                p(
                    "Everything else in this section is a different, more rigorous way of drawing that "
                    "in-sample/out-of-sample line -- walk-forward moves it forward in time repeatedly, "
                    "CPCV draws it many different ways at once, and PBO asks whether the very act of "
                    "picking a 'best' result already broke the line."
                ),
            ),
        ),
        Lesson(
            id="walk-forward-testing", title="Walk-Forward Testing",
            blocks=(
                p(
                    "Walk-forward testing splits the data into consecutive chronological folds and checks "
                    "performance on each test fold using ONLY information available before it -- no "
                    "re-tuning against the future. The Walk-Forward Opt tab reports walk_forward_efficiency "
                    "(out-of-sample performance as a fraction of in-sample performance) -- a number close "
                    "to 1.0 means the edge held up walking forward through time; a number well below 1.0 "
                    "means it degraded, a warning sign even if the overall backtest still looks profitable."
                ),
                warn(
                    "Walk-forward only samples ONE specific way of splitting the data into train/test "
                    "periods. A strategy can look walk-forward-stable and still be fragile to a different "
                    "partition of the same data -- that's exactly the gap CPCV closes next."
                ),
            ),
            see_also=("Walk-Forward Opt", "Walk-Forward GA"),
        ),
        Lesson(
            id="neighboring-parameter-test", title="The Neighboring-Parameter Test",
            blocks=(
                p(
                    "Don't just ask 'did my chosen parameter value make money' -- ask 'do nearby values "
                    "also work'. The Sensitivity tab sweeps each tunable parameter near its current value "
                    "and flags a cliff_detected result when a small nearby change collapses performance."
                ),
                example(
                    "If a strategy only works at a 14-period ATR but collapses at 13 or 15, that's a "
                    "strong sign you optimized noise at that one exact value rather than discovering a "
                    "real, robust edge -- a genuine edge usually survives small parameter perturbations."
                ),
                p(
                    "This is one of the highest-value, least-technical checks in the whole app -- it "
                    "directly tests the overfitting risk described in Choosing Parameters Without "
                    "Overfitting above."
                ),
            ),
            see_also=("Sensitivity", "Parameter Stability / Robustness Map"),
        ),
        Lesson(
            id="monte-carlo", title="Monte Carlo -- Testing Many Possible Futures",
            blocks=(
                p(
                    "A single historical backtest gives you ONE trade sequence out of many that could have "
                    "happened. Monte Carlo resamples that same pool of trades thousands of times (shuffled, "
                    "bootstrapped, or block-bootstrapped) and re-runs each resample through the full "
                    "prop-rule account simulation, producing a distribution of outcomes instead of one "
                    "point estimate."
                ),
                p(
                    "That's the difference between eval_pass_probability (how often a prop evaluation "
                    "would be passed across many possible trade orderings) and 'my one backtest passed' "
                    "(true of exactly one ordering, out of many that were just as likely)."
                ),
                tip(
                    "If eval_pass_probability is meaningfully lower than 'this backtest technically "
                    "passed', trust the Monte Carlo number -- it's telling you the pass was closer to luck "
                    "of the draw than the historical run alone would suggest."
                ),
            ),
            see_also=("6 Payout Probability", "Monte Carlo"),
        ),
        Lesson(
            id="cpcv", title="CPCV -- Combinatorial Purged Cross-Validation",
            blocks=(
                p(
                    "CPCV splits the data into several contiguous groups and enumerates every combination "
                    "of groups as the 'test' set for that path, with the remainder as train -- because both "
                    "the test AND train sets differ on every path, this samples far more of the ways the "
                    "data could have been partitioned than one walk-forward split ever does, without "
                    "needing more historical data. A few bars around every train/test boundary are purged "
                    "so a slow indicator's warm-up can't leak information across the split."
                ),
                p(
                    "The CPCV / PBO tab reports the distribution of out-of-sample results across every "
                    "path, plus pct_paths_oos_below_is (how often out-of-sample underperformed in-sample) "
                    "-- a strategy that's robust across most CPCV paths is meaningfully better evidence "
                    "than one that only survived a single walk-forward split."
                ),
            ),
            see_also=("CPCV / PBO",),
        ),
        Lesson(
            id="pbo", title="PBO -- Did You Just Pick the Luckiest One?",
            blocks=(
                p(
                    "Probability of Backtest Overfitting answers a different question than everything "
                    "above: given a POOL of candidates you tried (a Search Lab leaderboard, an Evolution "
                    "Lab run), for every CPCV path it ranks all candidates by in-sample performance, picks "
                    "whichever looked best in-sample, and checks where that SAME candidate ranks "
                    "out-of-sample. PBO is the fraction of paths where the in-sample 'winner' finished in "
                    "the bottom half out-of-sample."
                ),
                warn(
                    "The more candidates you test, the more likely the single best one is just the "
                    "luckiest of the batch rather than a genuine edge -- this is the multiple-comparisons "
                    "problem, and it's the one risk that has nothing to do with any individual candidate's "
                    "own quality. PBO at or above ~50% means picking that 'winner' was statistically no "
                    "better than a coin flip; comfortably under 50% means the pick likely reflects a real "
                    "edge."
                ),
                p(
                    "This is exactly why the app's own 'Next step' note after a large Search Lab or "
                    "Evolution Lab run recommends running the leaderboard through CPCV / PBO before "
                    "trusting profit factor or Sharpe alone to declare a winner."
                ),
            ),
            see_also=("CPCV / PBO",),
        ),
        Lesson(
            id="bias-and-snooping", title="Lookahead Bias, Survivorship Bias, and Data Snooping",
            blocks=(
                bullets(
                    "Lookahead bias -- the strategy uses information that would not actually have been "
                    "available at the moment it traded (e.g. deciding a trade using a bar's own close "
                    "price, then also entering at that same bar's close). See How Backtests Lie and the "
                    "Custom Code section for concrete patterns.",
                    "Survivorship bias -- testing only on instruments/strategies that are still around "
                    "today, silently excluding the ones that failed and disappeared, which inflates "
                    "apparent performance.",
                    "Data snooping -- repeatedly re-using the same dataset across many trial-and-error "
                    "iterations until something looks good, which is a slower-motion version of the same "
                    "multiple-comparisons problem PBO measures directly.",
                ),
                p(
                    "All three produce the same symptom: a backtest that looks better than the strategy "
                    "actually is. None of them are fixed by a bigger dataset or a fancier statistic -- only "
                    "by the specific validation step designed to catch each one."
                ),
            ),
        ),
        Lesson(
            id="robustness-ranking", title="Robustness Ranking -- Putting It Together",
            blocks=(
                p(
                    "No single number above is sufficient on its own -- a strategy can pass CPCV and still "
                    "be fragile to one parameter, or look great on Monte Carlo and still be the accidental "
                    "winner of a hundred-candidate search that PBO would flag. Full Pipeline exists to run "
                    "the important checks TOGETHER and combine them into one READY / MARGINAL / NOT READY "
                    "verdict, rather than leaving you to weigh five separate reports by eye."
                ),
                tip(
                    "READY is still just a backtest-based verdict -- the app's own guidance after a READY "
                    "result is to consider a short Forward Test (MT5 demo) before risking real capital on "
                    "a prop-firm evaluation, since that's the first genuinely live-market check."
                ),
            ),
            see_also=("Full Pipeline (all-in-one)", "Forward Test (MT5)"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# 5. Prop Firm Evaluation
# ---------------------------------------------------------------------------

_PROP_EVALUATION = EducationSection(
    id="prop-evaluation", icon="\U0001F6E1\uFE0F", title="Prop Firm Evaluation",
    subtitle="The rules that decide whether a profitable strategy actually survives an evaluation and gets paid.",
    lessons=(
        Lesson(
            id="profit-target-daily-loss-max-drawdown", title="Profit Targets, Daily Loss Limits, and Max Drawdown",
            blocks=(
                p("Nearly every prop-firm evaluation is built from the same three levers, even though the exact numbers vary firm to firm:"),
                bullets(
                    "Profit target -- the balance gain required to pass, usually 6-10% of account size.",
                    "Daily loss limit -- how much the account can lose in a single day before the "
                    "evaluation fails outright (some firms, like Apex's current evaluation, don't enforce "
                    "one at all).",
                    "Max drawdown -- the overall floor the account can never fall below, for the life of "
                    "the evaluation (and often the funded account too).",
                ),
                p(
                    "3 Prop-Firm Rules is where these get configured for a specific firm's preset (or "
                    "typed by hand), and every downstream tool -- Monte Carlo, Full Pipeline, Speed Run -- "
                    "simulates against exactly these numbers."
                ),
            ),
            see_also=("3 Prop-Firm Rules", "7 Prop-Firm Recommender"),
        ),
        Lesson(
            id="trailing-vs-static-drawdown", title="Trailing vs Static Drawdown",
            blocks=(
                p(
                    "A STATIC max drawdown is fixed once, at the start -- FTMO's 2-Step Challenge, for "
                    "example, sets the floor at 10% below the initial balance and it never moves, so every "
                    "dollar of profit adds pure buffer."
                ),
                p(
                    "A TRAILING max drawdown moves up as the account's balance reaches new highs, and "
                    "(depending on the firm) never moves back down -- Apex and Lucid both work this way. "
                    "This is stricter in a specific way that surprises new traders: a trade that runs to a "
                    "large open profit and then gives much of it back can breach the trailing floor even "
                    "though the account is still profitable overall from its starting balance, because the "
                    "floor already trailed up to reflect the earlier high."
                ),
                warn(
                    "Trailing drawdown accounts punish 'let it run and hope' behavior on open profit much "
                    "more than static drawdown accounts do -- banking profit and tightening stops as "
                    "balance grows matters more under a trailing model."
                ),
            ),
            see_also=("3 Prop-Firm Rules",),
        ),
        Lesson(
            id="consistency-rules", title="Consistency Rules",
            blocks=(
                p(
                    "A consistency rule caps how much of a strategy's total profit can come from a single "
                    "best day (or best trade), so passing on one lucky spike doesn't count the same as "
                    "genuinely repeatable performance. Firms differ a lot on WHERE this gets checked:"
                ),
                bullets(
                    "Some firms gate it at the EVALUATION -- you can't pass at all if one day is too "
                    "dominant.",
                    "Some firms gate it only at PAYOUT on the funded account -- you can pass the eval any "
                    "way you like, but a lopsided funded-stage profit history delays a withdrawal.",
                    "Some firms (Apex's current evaluation, LucidPro's evaluation) have no consistency "
                    "rule at eval stage at all.",
                ),
                tip(
                    "This app's own prop presets (3 Prop-Firm Rules) model consistency_rule_pct as an "
                    "EVALUATION-stage check specifically -- for a firm whose real consistency rule only "
                    "applies at payout, that field is correctly left blank there, since a payout-stage "
                    "rule isn't a reason to block the evaluation pass itself."
                ),
            ),
            see_also=("3 Prop-Firm Rules",),
        ),
        Lesson(
            id="min-trading-days-payout-cycles", title="Minimum Trading Days & Payout Cycles",
            blocks=(
                p(
                    "Minimum trading days is how many separate days must show at least one trade before "
                    "an evaluation can pass, even if the profit target was hit sooner -- it exists so a "
                    "single lucky day can't fully prove a strategy. It ranges widely: some firms require "
                    "zero, some require five or more."
                ),
                p(
                    "Payout cycle (payout_frequency_days) is a completely separate clock: the minimum "
                    "number of trading days on the FUNDED account before the first (and each subsequent) "
                    "payout can be requested. Passing fast and getting paid fast are genuinely different "
                    "questions -- a firm with a 1-day evaluation minimum can still have a 14-day payout "
                    "cycle, or vice versa."
                ),
                tip(
                    "If speed to an actual payout matters for your timeline, check BOTH numbers on 7 "
                    "Prop-Firm Recommender, and consider the fastest_payout fitness metric (OPTIMIZE "
                    "tabs) when searching for a candidate, which optimizes directly for the median days to "
                    "first payout rather than just the probability of eventually getting there."
                ),
            ),
            see_also=("7 Prop-Firm Recommender", "3 Prop-Firm Rules"),
        ),
        Lesson(
            id="path-risk", title="Path Risk -- Same Edge, Different Trade Order, Different Outcome",
            blocks=(
                p(
                    "A strategy can have genuinely positive expectancy overall and still fail a prop "
                    "evaluation, purely because of the ORDER its trades happen to arrive in. A losing "
                    "streak that lands early, before any profit buffer has built up, can breach a daily "
                    "loss limit or max drawdown that the exact same trades -- in a different order -- would "
                    "never have touched."
                ),
                example(
                    "Take a strategy with a 55% win rate and a 1.5R average winner. Its historical trade "
                    "list, played in the order it actually happened, might pass an evaluation easily. "
                    "Monte Carlo reshuffles that same list thousands of times and might show only a 65% "
                    "eval_pass_probability -- the other 35% of orderings hit the drawdown limit before "
                    "recovering, even though every individual trade is identical to the ones that passed."
                ),
                p(
                    "This is precisely why eval_pass_probability (a distribution over many possible "
                    "orderings) is a more honest answer than 'my one backtest passed' (true of exactly one "
                    "ordering) -- see Monte Carlo above."
                ),
            ),
            see_also=("6 Payout Probability", "Monte Carlo"),
        ),
        Lesson(
            id="reading-payout-probability", title="Reading Your Payout Probability",
            blocks=(
                p("Three related but distinct numbers come out of a Monte Carlo run -- know which question each one answers:"),
                bullets(
                    "eval_pass_probability -- how often the EVALUATION is passed, across many possible "
                    "trade orderings.",
                    "first_payout_probability -- how often the account goes on to reach a first FUNDED "
                    "payout, which is stricter than just passing (the funded stage has its own drawdown "
                    "and consistency exposure).",
                    "median_days_to_first_payout -- among the runs that DO reach a payout, how long it "
                    "typically takes -- a speed question, not a probability question, and the one "
                    "fastest_payout as a fitness metric optimizes for directly.",
                ),
                warn(
                    "A high first_payout_probability with a very long median_days_to_first_payout is a "
                    "real, common combination -- 'very likely to eventually pay out' and 'fast' are not "
                    "the same claim, which matters a lot if you're working against a deadline."
                ),
            ),
            see_also=("6 Payout Probability",),
        ),
    ),
)


# ---------------------------------------------------------------------------
# 6. Reading Your Results
# ---------------------------------------------------------------------------

_READING_RESULTS = EducationSection(
    id="reading-results", icon="\U0001F4C8", title="Reading Your Results",
    subtitle="Quick field guides for the screens you'll look at most.",
    lessons=(
        Lesson(
            id="field-guide-run-report", title="Field Guide: Run & Report",
            blocks=(
                p("In order, this is what to check first after any backtest:"),
                steps(
                    "Trade count -- zero means check entry logic before anything else; under ~20 means "
                    "don't trust a pass-probability number yet.",
                    "Profit factor -- below 1.0 means the strategy loses money on this data; fix the edge "
                    "before optimizing it.",
                    "Max drawdown -- above ~40% will fail most prop-firm simulations regardless of profit "
                    "factor.",
                    "Prop-firm simulation result, if computed -- a fail here despite good trade stats "
                    "usually means daily loss limit or max drawdown, not the strategy's core edge.",
                    "Everything above clean? Check eval-pass and first-payout probability, then move to "
                    "Optimize or Validate.",
                ),
                p(
                    "This is exactly the branching logic behind the green 'Next step' note that appears "
                    "under every Run & Report result -- this field guide is that same logic, spelled out."
                ),
            ),
            see_also=("5 Run & Report",),
        ),
        Lesson(
            id="field-guide-full-pipeline", title="Field Guide: Full Pipeline Verdict",
            blocks=(
                bullets(
                    "READY -- the strongest evidence this app produces, but still a backtest-based verdict. "
                    "A short Forward Test before real capital is the recommended next step, not a required "
                    "one.",
                    "MARGINAL -- not clearly ready or clearly dead. Try Quick Optimize, or adjust risk "
                    "settings and re-run, rather than taking it live as-is.",
                    "NOT READY -- doesn't hold up under re-validation. Go back to Search Lab or Evolution "
                    "Lab for a different candidate rather than trying to rescue this specific one.",
                ),
            ),
            see_also=("Full Pipeline (all-in-one)",),
        ),
        Lesson(
            id="field-guide-cpcv-pbo", title="Field Guide: CPCV / PBO Reports",
            blocks=(
                bullets(
                    "degradation / pct_paths_oos_below_is -- how much worse out-of-sample looks than "
                    "in-sample across CPCV paths. Large gaps mean overfitting, not a strong strategy, even "
                    "if the mean out-of-sample number is still positive.",
                    "PBO -- comfortably under 50% is the good outcome; at or above 50% means the pick was "
                    "statistically no better than a coin flip among the candidates tried.",
                ),
                p(
                    "See the PBO and CPCV lessons in Validation Lab above for the full explanation of what "
                    "these numbers are actually testing."
                ),
            ),
            see_also=("CPCV / PBO",),
        ),
    ),
)


# ---------------------------------------------------------------------------
# 7. How Backtests Lie
# ---------------------------------------------------------------------------

_HOW_BACKTESTS_LIE = EducationSection(
    id="how-backtests-lie", icon="\u26A0\uFE0F", title="How Backtests Lie",
    subtitle="A short, practical list of the classic ways a good-looking backtest turns out to be worthless.",
    lessons=(
        Lesson(
            id="common-mistakes", title="Nine Ways to Fool Yourself (And How to Notice)",
            blocks=(
                bullets(
                    "Optimizing until the equity curve looks perfect -- if you kept adjusting rules until "
                    "the chart looked good, you optimized to this exact dataset's noise. Check with the "
                    "Neighboring-Parameter Test.",
                    "Testing many strategies and only reporting the winner -- the multiple-comparisons "
                    "problem. Check with PBO.",
                    "Ignoring transaction costs -- an edge that vanishes under realistic spread/slippage "
                    "was never really there. Check with Cost Stress.",
                    "Using too little data -- see Trade Count & Statistical Significance.",
                    "Changing rules after seeing out-of-sample results -- this silently turns your "
                    "out-of-sample data into in-sample data, and you lose the one honest check you had.",
                    "Over-optimizing parameters -- see Choosing Parameters Without Overfitting.",
                    "Ignoring losing streaks -- a strategy's worst historical losing streak is a floor, "
                    "not a ceiling, for what to expect going forward.",
                    "Confusing backtest performance with live performance -- a backtest has no slippage "
                    "surprises, no platform outages, and no emotional deviation from the rules. Forward "
                    "Test and Deploy Live exist specifically to close this gap.",
                    "Using future information accidentally (lookahead bias) -- see the Custom Code "
                    "section for concrete coding patterns that cause this.",
                ),
                tip(
                    "None of these are exotic -- they're also exactly the failure modes the Validation "
                    "Lab's tools were each built to catch. If a result seems too good, it's worth running "
                    "back through this list before running back through more optimization."
                ),
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# 8. Building Custom Strategies With Code
# ---------------------------------------------------------------------------

_CUSTOM_CODE = EducationSection(
    id="custom-code", icon="\U0001F5A5\uFE0F", title="Building Custom Strategies With Code",
    subtitle="For Python, PineScript, or MQL5 strategies -- when you outgrow Manual Builder's condition rows.",
    lessons=(
        Lesson(
            id="manual-vs-code", title="Manual Builder vs Custom Code -- Which Do You Need?",
            blocks=(
                p(
                    "Manual Builder's long_entry / long_exit / short_entry / short_exit condition rows "
                    "cover most indicator-crossover and structural-condition hypotheses without writing "
                    "any code, and every backtest, Monte Carlo, and validation tool in this app treats a "
                    "Manual Builder strategy identically to a coded one."
                ),
                p(
                    "Reach for custom Python / PineScript / MQL5 when the logic genuinely needs something "
                    "condition rows can't express -- multi-step state machines, machine-learning "
                    "classifiers, or logic you already have written and want to import directly."
                ),
            ),
            see_also=("1 Strategy Configuration",),
        ),
        Lesson(
            id="strategy-anatomy", title="Anatomy of a T58 Strategy",
            blocks=(
                p(
                    "Every strategy type -- manual config, Python, PineScript, or MQL5 -- ultimately "
                    "answers the same question, bar by bar: given everything known UP TO AND INCLUDING "
                    "this bar, should a position be opened or closed right now? The backtest engine walks "
                    "forward bar by bar and never hands the strategy any data from bars that haven't "
                    "happened yet in that walk -- the single most important guarantee to preserve if you "
                    "write your own logic."
                ),
                warn(
                    "The single most common bug in hand-written strategy code is accidentally breaking "
                    "that guarantee -- see Common Coding Mistakes & Lookahead Bias below for concrete "
                    "patterns to check for."
                ),
            ),
        ),
        Lesson(
            id="indicators-and-calculations", title="Indicators & Calculations",
            blocks=(
                p(
                    "Vectorized indicator calculations (rolling means, ATR, RSI, and similar) are usually "
                    "computed once over the whole price series for speed -- which is fine, as long as the "
                    "value used to make a decision at bar i only depends on bars up to and including i, "
                    "never on i+1 or later."
                ),
                warn(
                    "pandas operations like .shift(-1), a rolling window centered rather than trailing, or "
                    "an indicator library that 'repaints' its most recent values as new bars arrive, are "
                    "all easy, easy-to-miss ways to leak future information into a decision without "
                    "meaning to."
                ),
            ),
        ),
        Lesson(
            id="risk-in-code", title="Where Risk Config Lives vs Where Strategy Logic Lives",
            blocks=(
                p(
                    "A coded strategy should decide WHEN to enter and exit and, if relevant, where its own "
                    "stop/target levels are. Account-level risk (position size, risk per trade) stays on 4 "
                    "Risk & Execution rather than being hardcoded into the strategy file -- this keeps a "
                    "single strategy file reusable across different account sizes and risk settings "
                    "without editing code every time."
                ),
            ),
            see_also=("4 Risk & Execution",),
        ),
        Lesson(
            id="testing-debugging", title="Testing & Debugging Your Strategy",
            blocks=(
                steps(
                    "Run it on 5 Run & Report first, exactly like a Manual Builder strategy.",
                    "Zero trades? Check the entry condition's logic directly, and confirm the loaded "
                    "timeframe on 2 Market Data actually matches what the code expects.",
                    "Suspicious results (too good, or a suspiciously smooth equity curve)? Re-read Common "
                    "Coding Mistakes & Lookahead Bias below before trusting the number.",
                    "If a Manual Builder equivalent exists for a simpler version of the idea, build that "
                    "first and compare -- a large, unexplained gap between the two often points at a bug "
                    "in the code version rather than a genuinely better idea.",
                ),
            ),
            see_also=("5 Run & Report",),
        ),
        Lesson(
            id="common-coding-mistakes", title="Common Coding Mistakes & Lookahead Bias",
            blocks=(
                bullets(
                    "Deciding a trade using a bar's own close price, then entering AT that same bar's "
                    "close -- in live trading you wouldn't know the close until the bar had already "
                    "finished. Decide on bar i's close, execute at bar i+1's open.",
                    "An indicator that 'repaints' -- its historical values change retroactively as new "
                    "data arrives (common in some smoothing/zig-zag style indicators). A backtest using "
                    "the CURRENT repainted history looks nothing like what the indicator actually showed "
                    "in real time.",
                    "In an ML-based strategy, a feature or label built using information only knowable "
                    "after the fact (e.g. 'was this the local high of the next 10 bars') is lookahead by a "
                    "different name -- the model learns to predict something it was secretly already told.",
                    "Survivorship bias creeping in through data selection -- testing only on symbols that "
                    "still exist and have a long, clean history quietly excludes whatever failed and was "
                    "delisted along the way.",
                ),
                tip(
                    "When a coded strategy looks unusually good, the Neighboring-Parameter Test and a "
                    "walk-forward run are two of the fastest ways to surface a lookahead bug -- a genuine "
                    "edge degrades gracefully under both; a lookahead-inflated one often collapses sharply."
                ),
            ),
            see_also=("Sensitivity", "Walk-Forward Opt"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# 9. The T58 Research Process (centerpiece)
# ---------------------------------------------------------------------------

_RESEARCH_PROCESS = EducationSection(
    id="research-process", icon="\U0001F3C6", title="The T58 Research Process",
    subtitle="Everything above, as one ordered workflow -- the centerpiece of this whole tab.",
    lessons=(
        Lesson(
            id="the-full-process", title="The Full Process, Start to Finish",
            blocks=(
                p(
                    "This is the same lifecycle app.orchestration.pipeline_guide's live 'Next step' notes "
                    "walk you through after every real action in the app -- laid out here as one ordered "
                    "reference, with the concept lessons above linked at each stage."
                ),
                steps(
                    "Define the hypothesis -- Strategy Development section above.",
                    "Load clean data -- 2 Market Data.",
                    "Build the strategy -- 1 Strategy Configuration (Manual Builder or custom code), or "
                    "Search Lab / Evolution Lab if you don't have a specific idea yet.",
                    "Run a baseline backtest -- 5 Run & Report.",
                    "Diagnose weaknesses -- Backtesting Statistics section + the field guide in Reading "
                    "Your Results.",
                    "Refine carefully -- Quick Optimize / Iterative Refinement, choosing a fitness metric "
                    "(eval_pass_probability, composite_prop_score, or fastest_payout) that matches your "
                    "actual goal.",
                    "Run robustness tests -- Sensitivity / Parameter Stability (the Neighboring-Parameter "
                    "Test) and Regime Survival Matrix.",
                    "Out-of-sample validation -- Walk-Forward Opt, then CPCV / PBO.",
                    "Monte Carlo / path analysis -- 6 Payout Probability, to see a distribution of "
                    "outcomes rather than one historical run.",
                    "Prop evaluation simulation -- Full Pipeline for a combined READY / MARGINAL / NOT "
                    "READY verdict against your chosen 3 Prop-Firm Rules.",
                    "Forward / demo test -- Forward Test (MT5), the first genuinely live-market check.",
                    "Deploy live -- Deploy Live, with Monitor (Live Market) open to watch the first "
                    "stretch.",
                ),
                tip(
                    "In a hurry against a real deadline? Speed Run and Overnight Autopilot chain steps "
                    "3-10 into one automated run instead of walking each tab by hand -- see the User "
                    "Manual for how to use them."
                ),
            ),
            see_also=("Full Pipeline (all-in-one)", "Speed Run", "User Manual"),
        ),
    ),
)


EDUCATION_SECTIONS: tuple[EducationSection, ...] = (
    _GETTING_STARTED,
    _STRATEGY_DEVELOPMENT,
    _BACKTESTING_FUNDAMENTALS,
    _VALIDATION_LAB,
    _PROP_EVALUATION,
    _READING_RESULTS,
    _HOW_BACKTESTS_LIE,
    _CUSTOM_CODE,
    _RESEARCH_PROCESS,
)

_BY_ID: dict[str, EducationSection] = {s.id: s for s in EDUCATION_SECTIONS}


def list_sections() -> list[EducationSection]:
    return list(EDUCATION_SECTIONS)


def get_section(section_id: str) -> EducationSection:
    try:
        return _BY_ID[section_id]
    except KeyError as exc:
        raise KeyError(f"Unknown education section '{section_id}'. Known: {sorted(_BY_ID)}") from exc


def total_lesson_count() -> int:
    return sum(len(s.lessons) for s in EDUCATION_SECTIONS)
