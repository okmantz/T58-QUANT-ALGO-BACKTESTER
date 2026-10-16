"""Manual Strategy Builder tab."""
from __future__ import annotations

from tkinter import END, Frame, Text, ttk
from app.ui.common import _safe_font
from app.ui.theme import MONO
from app.ui import theme

# Palette names below are (re)bound by app.ui.theme on import and on every theme toggle.
PANEL = ""
BORDER = ""
TEXT = ""
TEXT_MUTED = ""
METAL_BRIGHT = ""
GREEN = ""
AMBER = ""
ACCENT_HOVER = ""
NEON_CYAN = ""
NEON_LIME = ""
theme.bind(globals())


class ManualBuilderMixin:
    """Manual Strategy Builder tab."""


    # -----------------------------------------------------------------------
    # User Manual
    # -----------------------------------------------------------------------

    def _build_manual_tab(self):
        f = self.tab_manual
        self._page_header(
            f, "GUIDE", "User Manual",
            "A dummy-proof, step-by-step walkthrough of the whole app -- start to finish, "
            "from importing your first candle of data to reading a Full Pipeline verdict.",
        )

        wrap = Frame(f, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        wrap.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        text = Text(
            wrap, wrap="word", bg=PANEL, fg=TEXT, relief="flat", bd=0,
            highlightthickness=0, font=_safe_font(9), padx=22, pady=18, cursor="arrow",
            spacing1=1, spacing3=1,
        )
        scrollbar = ttk.Scrollbar(wrap, orient="vertical", command=text.yview, style="T58.Vertical.TScrollbar")
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            text.bind(seq, lambda e: self._generic_text_wheel(text, e))

        text.tag_configure("h1", font=_safe_font(17, "bold"), foreground=ACCENT_HOVER, spacing1=4, spacing3=10)
        text.tag_configure("h2", font=_safe_font(12, "bold"), foreground=NEON_CYAN, spacing1=20, spacing3=8)
        text.tag_configure("h3", font=_safe_font(10, "bold"), foreground=METAL_BRIGHT, spacing1=12, spacing3=4)
        text.tag_configure("body", font=_safe_font(9), foreground=TEXT_MUTED, spacing3=5, lmargin1=2, lmargin2=2)
        text.tag_configure("bullet", font=_safe_font(9), foreground=TEXT_MUTED, lmargin1=22, lmargin2=38, spacing3=4)
        text.tag_configure("numstep", font=_safe_font(9, "bold"), foreground=TEXT, lmargin1=4, lmargin2=26, spacing1=8, spacing3=2)
        text.tag_configure("substep", font=_safe_font(9), foreground=TEXT_MUTED, lmargin1=30, lmargin2=46, spacing3=3)
        text.tag_configure("mono", font=(MONO, 9), foreground=NEON_LIME)
        text.tag_configure("warn", font=_safe_font(9, "bold"), foreground=AMBER, lmargin1=4, lmargin2=20, spacing1=8, spacing3=8)
        text.tag_configure("tip", font=_safe_font(9), foreground=GREEN, lmargin1=4, lmargin2=20, spacing1=6, spacing3=8)
        text.tag_configure("divider", font=_safe_font(4), foreground=BORDER, spacing3=14)

        def h1(t):
            text.insert(END, t + "\n", "h1")

        def h2(t):
            text.insert(END, t + "\n", "h2")

        def h3(t):
            text.insert(END, t + "\n", "h3")

        def body(t):
            text.insert(END, t + "\n", "body")

        def bullet(t):
            text.insert(END, "•  " + t + "\n", "bullet")

        def numstep(n, t):
            text.insert(END, f"{n}.  " + t + "\n", "numstep")

        def substep(t):
            text.insert(END, "–  " + t + "\n", "substep")

        def warn(t):
            text.insert(END, "⚠  " + t + "\n", "warn")

        def tip(t):
            text.insert(END, "✓  " + t + "\n", "tip")

        def rule():
            text.insert(END, "―" * 90 + "\n", "divider")

        # -------------------------------------------------------------
        # Quick Start (short version) -- up top, for anyone who just
        # wants the fastest path and already knows their way around a
        # backtester.
        # -------------------------------------------------------------
        h1("QUICK START (SHORT VERSION)")
        body(
            "For anyone in a hurry: this is the whole app in five clicks. If any of these "
            "words don't mean anything yet, skip down to the FULL WALKTHROUGH below -- it "
            "explains every one of these in plain terms."
        )
        numstep(1, "TEST → Market Data — import a CSV, or fetch data with a free Alpaca key, then select it.")
        numstep(2, "CREATE → Strategy Builder — pick Manual / Python / PineScript / MQL5 and set it up.")
        numstep(3, "TEST → Prop-Firm Rules — enter your prop firm's account size, targets, and drawdown rules.")
        numstep(4, "TEST → Risk & Execution — set your risk per trade and starting balance (defaults are fine to start).")
        numstep(5, "OPTIMIZE → Full Pipeline — click RUN FULL PIPELINE and read the READY / MARGINAL / NOT READY verdict.")
        tip(
            "That's it. Everything under OPTIMIZE, VALIDATE, and CHAMPION in the sidebar is all "
            "optional extra tooling for later — you do not need it for a first pass."
        )
        rule()

        # -------------------------------------------------------------
        # Full walkthrough
        # -------------------------------------------------------------
        h1("FULL WALKTHROUGH (STEP BY STEP)")
        body(
            "Follow these in order the first time. The sidebar groups tabs by stage (Create, Test, "
            "Optimize, Validate, Champion, Forward Test, Deploy, Monitor) rather than one long numbered "
            "list -- this walkthrough still goes in a sensible first-pass order regardless of section."
        )

        h2("What's new in this update")
        body(
            "The sidebar was reorganized into three umbrellas -- Strategy Lab (Create through Strategy "
            "Graveyard, the 7-step journey below), Quant Lab (Quant Lab, Options, Hedge Fund Manager), "
            "and Account (Account, Education) -- and several tools that used to be tucked inside another "
            "tab now have their own screen:"
        )
        bullet("CREATE \u2192 Multi-Instrument Speed Run -- the same Speed Run, run concurrently across several instrument/timeframe datasets.")
        bullet("CREATE \u2192 Research Loop (Background) -- split out of Research Agent into its own independent tab (own market data, own AI Assist settings) so it can run as a background companion.")
        bullet("OPTIMIZE \u2192 Overview / Picker -- a guided \"which OPTIMIZE tool should I use\" picker; every tool underneath is still its own full tab.")
        bullet("OPTIMIZE \u2192 Quick Optimize -- the GA search step in isolation (previously only reachable via the Strategy Library's OPTIMIZE SELECTED button, which still works too). \u26a0 NOT OOS validated -- a strong result here means \"worth a real Full Pipeline run,\" not a finished answer.")
        bullet("OPTIMIZE \u2192 Multi-Instrument Search Lab and Multi-Instrument Evolution Lab -- the same Search Lab / Evolution Lab run concurrently across several instrument/timeframe datasets.")
        bullet("OPTIMIZE \u2192 Risk Sweep -- not a strategy-parameter search: finds which risk-per-trade level maximizes prop-survival odds for the strategy as configured.")
        bullet("VALIDATE \u2192 Overview / Checklist -- aggregates what Walk-Forward Opt / WFGA / CPCV / Sensitivity / Regime Matrix have already found for whichever strategy is marked \"current.\"")
        bullet("VALIDATE \u2192 PBO is now its own tab, split out from the combined CPCV / PBO screen.")
        bullet("DEPLOYMENT \u2192 Compare Strategies -- pick 2-4 saved strategies and see them backtested side by side.")
        bullet("ACCOUNT \u2192 Account Settings (display name/email/company) now sits above Notification Settings.")
        tip("If a bullet above doesn't match what you see, the sidebar item names/order win -- this manual "
            "gets updated by hand and can occasionally lag a fast-moving reorganization.")

        h2("STEP 1 — Import your market data (TEST → Market Data)")
        body("The backtester needs historical price candles (open/high/low/close/volume) before anything else can run.")
        numstep(1, "Click Market Data in the sidebar (under TEST).")
        numstep(2, "If you already have a CSV file of price data, click IMPORT CSV(S) and select it.")
        substep("A CSV needs columns for timestamp, open, high, low, and close (volume is optional). Headers are "
                "auto-detected — 'time'/'date', 'o'/'open', etc. all work.")
        substep(
            "You can select more than one timeframe at once (Ctrl/Cmd-click or Shift-click) for multi-timeframe "
            "strategies — e.g. a 60-minute file for bias plus a 5-minute file for entries. Ignore this at first."
        )
        numstep(3, "Don't have a CSV? Use the built-in Alpaca fetch instead — see the box below.")
        numstep(4, "Click a dataset in the list to select it. That's the data every later tab will use.")
        tip("Tip: you can also drop CSV files straight into the app's data/raw/ folder and click REFRESH LIST.")

        h3("Getting a free Alpaca API key (for fetching data without a CSV)")
        body(
            "Alpaca is a brokerage that gives out free market data through an API — you do not need to fund "
            "an account or place any real trades to use this."
        )
        numstep(1, "Go to alpaca.markets and sign up for a free account (choose Paper Trading, not a live account).")
        numstep(2, "In the Alpaca dashboard, generate an API Key ID and a Secret Key (usually under 'API Keys' "
                    "or 'Paper Trading' settings).")
        numstep(3, "Back in this app, on the Market Data tab (under TEST), paste both into the 'Fetch data from Alpaca' section.")
        numstep(4, "Click TEST CONNECTION to confirm the keys work.")
        numstep(5, "Choose an asset class, symbol, and timeframe, then click FETCH & SAVE. It's saved locally as "
                    "a CSV, so you only need to fetch it once.")
        tip("Your keys are saved locally on this computer so you don't have to re-enter them every time.")
        rule()

        h2("STEP 2 — Build or upload a strategy (CREATE → Strategy Builder)")
        body("This is the trading logic that will be tested against your data. Four options:")
        bullet("MANUAL — a visual, no-code builder: pick indicators and entry/exit conditions from dropdowns.")
        bullet("PYTHON — upload your own .py strategy file, or load one from the built-in Strategy Library.")
        bullet("PINESCRIPT — upload a TradingView-style .pine strategy (a supported subset of syntax).")
        bullet("MQL5 — upload a MetaTrader .mq5 Expert Advisor (a supported subset of syntax).")
        numstep(1, "Click Strategy Builder (under CREATE) and pick one of the four buttons at the top.")
        numstep(2, "MANUAL: fill in the visual builder — indicators, then entry/exit rules, then a stop-loss/"
                    "take-profit. PYTHON/PINESCRIPT/MQL5: upload a file, or load a saved one from the Strategy Library.")
        tip(
            "New to this? Start with MANUAL and a simple moving-average crossover — it's the fastest way to "
            "see the whole app work end to end before bringing in your own code."
        )

        h3("The Strategy Library — saving, reusing, and tuning strategies")
        body(
            "Every Python/PineScript/MQL5 strategy you upload can be saved here, so it lives inside the "
            "app/repo instead of only on your computer — you (or the Full Pipeline, Evolution Lab, etc.) "
            "can load it back up any time without re-browsing to the original file."
        )
        bullet("REFRESH LIBRARY — reload the list (e.g. after dropping a file into the library folder by hand).")
        bullet("VIEW CODE / CONFIG — read a saved strategy's actual source (or its manual-builder config) "
               "without leaving this tab.")
        bullet("VIEW DNA — a keyword-based breakdown of a strategy's own entry / exit / risk logic into "
               "plain tags, so you can see at a glance what kind of strategy it actually is.")
        bullet("FIND COMMON PATTERNS (ALL) — mines every saved strategy for genes (DNA tags) that show up "
               "together often, surfacing patterns across your whole library rather than one file at a time.")
        bullet("Check strategies off in the Batch selection panel, then RUN BATCH TEST (CHECKED) or "
               "RUN FULL PIPELINE (BATCH) (CHECKED) — test several library strategies back-to-back, "
               "each getting its own separate report, instead of one at a time.")
        bullet("OPTIMIZE SELECTED — runs Quick Optimize: auto-tunes one strategy's parameters toward pass/"
               "win-rate/payout targets using the same search engine as 06 Refinement, and saves the result "
               "back into the library as a new draft — the original file is left untouched.")
        bullet("OPEN LIBRARY FOLDER / EXPORT LIBRARY AS ZIP — for backing up or moving your library by hand.")
        rule()

        h2("STEP 3 — Enter your prop firm's rules (TEST → Prop-Firm Rules)")
        body(
            "These numbers come straight from your prop firm's rulebook — check their FAQ/PDF, since getting "
            "this wrong makes every result downstream meaningless."
        )
        bullet("Account size, evaluation profit target %, daily loss limit %, maximum drawdown %.")
        bullet("Drawdown type (trailing vs static) and drawdown check mode (intrabar vs end-of-day) — match "
               "these exactly to what your firm documents.")
        bullet("Optional: payout threshold/cap/frequency, minimum trading days, position size limits.")
        rule()

        h2("STEP 4 — Set risk & execution (TEST → Risk & Execution)")
        body("Position sizing, trading costs, and execution assumptions used by the backtest engine.")
        bullet("Initial balance, risk mode (percent of equity vs fixed), and risk per trade.")
        bullet("Spread/slippage/commission — leave at realistic defaults unless your broker publishes different numbers.")
        tip("Not sure what to put here? The defaults are sane starting points — you can always come back and tune them.")

        h3("Adaptive Risk (optional) — graduated, limit-aware position sizing")
        body(
            "Instead of one fixed risk-per-trade for the whole account life, Adaptive Risk throttles position "
            "size down as the account gets closer to its own drawdown limit (and can lock in profit once "
            "ahead), the way a careful trader manages size manually near their firm's hard limits."
        )
        bullet("Add rules by hand (a drawdown_pct trigger + a smaller risk to switch to), or click the "
               "one-click preset button to auto-build a sensible graduated throttle straight off whatever "
               "Prop Rules are already entered on Step 3.")
        bullet("Available everywhere a strategy actually runs: 05 Run & Report, 15 Full Pipeline, "
               "06 Refinement, Quick Optimize, and Evolution Lab all have their own enable checkbox for it.")
        rule()

        h2("STEP 5 — Run a single backtest (TEST → Run & Report)")
        body(
            "This runs your strategy exactly as configured, once, and produces one HTML report: "
            "Backtest → Prop Simulation → Monte Carlo → Report."
        )
        numstep(1, "Click 05 RUN & REPORT, set the number of Monte Carlo simulations (10,000 is a good default).")
        numstep(2, "Click RUN BACKTEST & REPORT on this tab, then OPEN HTML REPORT once it finishes.")
        warn(
            "This tab's button used to be labeled RUN FULL PIPELINE too, which made it easy to confuse "
            "with the 15 FULL PIPELINE tab below — it's now called RUN BACKTEST & REPORT to make the "
            "difference obvious: this one just runs your strategy once, as-is. The 15 FULL PIPELINE tab "
            "(Step 9 in this guide) automatically searches for a better configuration and validates it out-"
            "of-sample before giving you a verdict. For a first pass, most people skip straight to Step 9. "
            "If you're testing several strategies at once, use Batch selection on 01 Strategy Configuration instead of "
            "either single-strategy button."
        )
        body("A few things this report checks automatically, every single run:")
        bullet("Lookahead check — re-runs the strategy on data cut off right after a sample of its own real "
               "signal bars and confirms the signals don't change; a strategy that's secretly peeking at "
               "future data would fail this and get flagged in the run log.")
        bullet("Cost-ladder section — re-prices the exact same trades at several extra levels of trading cost "
               "(0%, 0.05%, 0.10%, 0.25% added friction) so you can see how fragile the edge is to slippage.")
        bullet("Holdout comparison — a chronological 80/20 in-sample/holdout split with no re-tuning on the "
               "holdout portion, so you get an early read on overfitting without running a full Walk-Forward.")
        warn(
            "Watch for pip_scale_mismatch / gap_loss warnings in the run log — they mean the pip size doesn't "
            "match this instrument's actual price scale (a common issue testing an FX-tuned strategy against "
            "gold or a stock), which can produce wildly wrong P&L numbers if ignored."
        )
        rule()

        h2("PAYOUT PROBABILITY — full-lifecycle survival simulator")
        body(
            "Goes beyond a single eval-pass number: simulates thousands of complete account lifecycles "
            "(Start -> Evaluation -> Funded -> Payout #1 -> Payout #2 -> ... -> Reset/Continue) and reports "
            "the funnel — how many simulated accounts actually made it through EACH stage, not just whether "
            "they passed the eval. Uses whatever strategy, data, prop rules, and risk are set on Steps 1-4."
        )
        rule()

        h2("STEP 6 — Iterative Refinement (OPTIMIZE → Iterative Refinement, optional)")
        body(
            "A genetic-algorithm-style parameter search: re-runs your strategy many times with mutated "
            "parameters on the SAME historical data, keeps the best-performing configurations each round, "
            "and converges toward the best-scoring configuration it can find. Produces its own separate "
            "report — your normal 05 Run & Report result is never touched unless you apply the winner back."
        )
        tip("Found a good configuration? Use APPLY BEST CONFIG TO STRATEGY TAB to carry it straight back to "
            "01 Strategy Configuration without retyping any parameter by hand.")
        rule()

        h2("STEP 7 — Search Lab (OPTIMIZE → Search Lab, optional)")
        body(
            "Generates and tests many strategy variations at once instead of one at a time: a fast filter "
            "narrows thousands of candidates down to a shortlist, the same Iterative Refinement engine tunes "
            "each shortlisted candidate, and a strict validation gate (walk-forward, lookahead check, "
            "parameter-neighborhood robustness, a deflated Sharpe ratio that corrects for how many candidates "
            "were tried) decides what actually survives."
        )
        tip(
            "Run a search here first, then check the Family Diversity tab (under CHAMPION) to "
            "see which whole hypothesis families — trend, breakout, mean reversion, liquidity sweep, and so "
            "on — actually held up, not just which single candidate scored highest."
        )
        rule()

        h1("VALIDATE — Walk-Forward, CPCV, Sensitivity, Walk-Forward GA, Regime Survival Matrix — optional deeper robustness checks")
        body(
            "Everything in this group is optional and can be safely skipped on a first pass — the OPTIMIZE → Full "
            "Pipeline tab already runs a solid, automated version of search and out-of-sample "
            "validation on its own. Come back here once you have a strategy worth digging into further."
        )
        h2("VALIDATE \u2192 Walk-Forward Optimization")
        body(
            "Re-optimizes the strategy fresh on each rolling or anchored fold's training window using a "
            "small GA search, applies the winning configuration UNCHANGED to that fold's held-out test "
            "window, and chains every fold's out-of-sample trades into one continuous equity curve — trust "
            "this number over a single in-sample backtest."
        )
        h2("VALIDATE \u2192 CPCV")
        body(
            "Combinatorial Purged Cross-Validation stress-tests the strategy across many different "
            "combinatorial train/test partitions of the same data, instead of one holdout split."
        )
        h2("VALIDATE \u2192 PBO")
        body(
            "Probability of Backtest Overfitting -- its own tab now, separate from CPCV above (they used "
            "to share one screen). Checks a small pool of candidates (this strategy plus a few "
            "automatically perturbed variants) and reports the odds that whichever looks best in-sample is "
            "really just noise."
        )
        h2("VALIDATE \u2192 Sensitivity")
        body(
            "Sweeps every tunable numeric parameter across +/- a percentage of its current value, holding "
            "everything else fixed, and flags a 'cliff' wherever the metric drops sharply between adjacent "
            "steps — the sign of a knife-edge parameter rather than a real, stable plateau. Can also produce "
            "a 2D heatmap for a chosen pair of parameters, since two can interact even when each looks fine alone."
        )
        h2("CHAMPION \u2192 Multi-Asset Portfolio")
        body(
            "Applies the strategy configured on Step 2 to every instrument you list, computes the "
            "correlation matrix of their daily returns, re-weights each instrument's risk (correlated "
            "instruments get sized down), and merges every instrument's trades into one shared account "
            "equity curve — including its own Monte Carlo pass-probability on the combined account."
        )
        tip("ADD LEG FROM LIBRARY lets a portfolio combine genuinely different saved strategies, not just "
            "one strategy run across several instruments.")
        h2("OPTIMIZE \u2192 Multi-Objective Optimization")
        body(
            "Runs a real NSGA-II search across several objectives at once (e.g. Sharpe, max drawdown, "
            "eval-pass probability) instead of collapsing them into one weighted score the way Iterative "
            "Refinement's GA does. Produces a Pareto front — candidates where none is strictly worse than "
            "any other on the front. Picking a final winner from that list is your call."
        )
        h2("VALIDATE \u2192 Walk-Forward GA")
        body(
            "The same crossover/mutation/tournament-selection GA as Iterative Refinement, but every "
            "candidate's fitness is scored ONLY on chained out-of-sample fold data — never on the training "
            "windows or the full dataset. A genome that only fits one historical stretch scores lower and "
            "gets selected against, generation over generation."
        )
        h2("Regime Survival Matrix")
        body(
            "Classifies every bar on trend, volatility, session, and market environment, then attributes "
            "THIS strategy's own trades to whichever regime was active at entry — so you can see exactly "
            "which market conditions to gate the strategy off in, not just one aggregate backtest number."
        )
        rule()

        h1("CHAMPION, CREATE & OPTIMIZE — Ensemble, Generate Strategies, Evolution Lab, Family Diversity (split across those three sections)")
        h2("Multi-Strategy Ensemble (CHAMPION)")
        body(
            "The mirror case of Portfolio: several DIFFERENT strategies combined on the SAME instrument, "
            "instead of one strategy across several instruments. Two modes: Blend keeps every strategy "
            "trading independently at a correlation-adjusted share of the account's risk budget (reuses "
            "Portfolio's own math); Vote combines every strategy's signal into one entry, taken only once "
            "enough of them agree on direction."
        )
        h2("Generate Strategies (AI) (CREATE)")
        body(
            "Drafts a NEW strategy's source code from a plain-language idea, using a local Ollama model — "
            "grounded in whatever papers you've dropped into the research/ folder and in your own best-"
            "performing saved strategies of the same language, so it matches this app's actual code style "
            "instead of guessing. Every strategy this produces is saved tagged DRAFT and has to earn its "
            "way through the normal backtest -> validation ladder like anything else — nothing runs "
            "automatically."
        )
        h2("Evolution Lab (OPTIMIZE)")
        body(
            "Natural-selection-based strategy discovery: each generation generates a fresh population of "
            "strategies across every hypothesis family, runs them through pre-filter -> robustness -> "
            "walk-forward -> Monte Carlo -> prop simulation -> real CPCV/PBO -> a stress test at higher "
            "execution costs -> a correlation-based cluster dedupe, keeps the top performers by PROP FITNESS "
            "(not raw profit), mutates them, and repeats."
        )
        numstep(1, "Set a population size, elite keep, max generations (blank = run until stopped), and the "
                    "other run settings, then click START. It runs in the background — safe to leave running "
                    "for hours while you work in other tabs.")
        numstep(2, "Watch the leaderboard fill in as generations complete. Click a row for its detail (stats "
                    "+ code) and PROMOTE TO STRATEGY LIBRARY once you find one worth keeping.")
        tip("Every candidate evaluated (not just the winners) is logged, so later generations' journal "
            "entries can say what's historically worked before, not just what this generation found.")
        h2("Family Diversity (CHAMPION)")
        body(
            "Groups the most recent Search Lab run's candidates by hypothesis family (trend following, "
            "breakout, mean reversion, liquidity sweep, momentum, VWAP, opening range, market structure, "
            "volatility expansion/contraction, pullback, statistical arbitrage, relative strength, regime "
            "switching) and ranks the FAMILIES themselves — run a search on 07 Search Lab first."
        )
        rule()

        h2("STEP 9 — Run the Full Pipeline (OPTIMIZE → Full Pipeline) — the recommended one-button path")
        body(
            "This is the fastest way to get a trustworthy answer: it backtests your strategy as given, "
            "automatically searches for a configuration that generalizes (scored only on data it wasn't "
            "tuned on), re-validates the winner with a fresh Monte Carlo run, checks it holds up across "
            "several different historical stretches, and gives you one plain verdict."
        )
        numstep(1, "Make sure Steps 1-4 above are filled in (data selected, strategy built, prop rules and risk set).")
        numstep(2, "Click 15 FULL PIPELINE, leave the default settings for a first run, and click RUN FULL PIPELINE.")
        numstep(3, "Watch the live progress log — it can take anywhere from under a minute to several minutes "
                    "depending on your data size and settings.")
        numstep(4, "Read the VERDICT box at the top once it finishes:")
        substep("READY — passed every check: Monte Carlo pass probability, drawdown, out-of-sample stability, "
                "and the signal-quality (ICIR) gate.")
        substep("MARGINAL — passed some but not all checks — worth a closer look before trusting it.")
        substep("NOT READY — failed enough checks that this configuration isn't trustworthy as-is; check the "
                "listed reasons for exactly why.")
        substep("The Verdict box lists the specific reason for every pass/fail — scroll down within the box "
                "(or the page) if the full list runs long.")
        tip(
            "For Python/PineScript/MQL5 strategies, the winning version is automatically saved into the "
            "Strategy Library, tagged 'validated', ready to use again later or take straight into 16 LIVE "
            "DEMO TEST."
        )
        rule()

        h2("STEP 10 — Forward Test on a real broker feed (FORWARD TEST → Forward Test (MT5))")
        body(
            "Once a strategy looks good in the Full Pipeline, this deploys it to a free MetaTrader 5 (MT5) "
            "demo account so you can watch it trade forward against real, live broker prices — still no real "
            "money, no live/funded order path exists on this tab."
        )
        h3("One-time MT5 setup")
        numstep(1, "Download and install the MT5 terminal (64-bit) — any MT5-supporting broker's website offers "
                    "a free download, or your prop firm's own site if they provide one.")
        numstep(2, "Open the terminal and create a free demo account from within it (File → Open an Account → "
                    "choose a demo account) — this gives you a login number, server name, and password.")
        numstep(3, "In this app's Live Demo Test tab, enter that login, server, and password under "
                    "'MT5 Demo Account'.")
        numstep(4, "Click SAVE & TEST CONNECTION.")
        substep(
            "If it fails saying the terminal wasn't found, click AUTO-DETECT first (it searches the common "
            "Windows install locations), or BROWSE... to point directly at terminal64.exe if MT5 is installed "
            "somewhere unusual."
        )
        substep("This tab only works on Windows with the MT5 terminal actually installed and running.")
        h3("Running a session")
        numstep(1, "Pick a saved strategy from the Strategy Library dropdown.")
        numstep(2, "Set the symbol (must match the exact name in your MT5 broker's Market Watch, e.g. 'XAUUSD') "
                    "and your risk settings.")
        numstep(3, "Click START LIVE DEMO TEST. Watch trades appear in the trade journal and live log as they happen.")
        warn(
            "The red KILL SWITCH — FLATTEN & STOP button immediately closes every open position and stops the "
            "session. Use it any time something looks wrong."
        )
        rule()

        h2("STEP 11 — Deploy Live (DEPLOY → Deploy Live) — real money, read this first")
        warn(
            "This is NOT the same as Live Demo Test. An account connected here trades with real capital in a "
            "live or funded prop-firm account. Confirm your prop firm actually permits automated/EA trading "
            "on that account before connecting it — many firms restrict or ban it outright, and violating "
            "that can get a funded account terminated regardless of performance."
        )
        numstep(1, "Pick your prop firm from the dropdown (or 'Other' if it isn't listed) — this fills in which "
                    "platform(s) it uses.")
        substep(
            "Only MT4/MT5 accounts are connectable today. Futures-only firms (Apex, Topstep, MyFundedFutures) "
            "use Tradovate/Rithmic/NinjaTrader instead — a different integration this app doesn't have yet."
        )
        numstep(2, "Enter your account's nickname, login, server, and password, check the confirmation box, "
                    "and click SAVE ACCOUNT.")
        numstep(3, "Click TEST CONNECTION to confirm it reaches your real account and reports its balance.")
        numstep(4, "Pick a validated strategy and click START LIVE TRADING.")
        substep(
            "The same red KILL SWITCH from Live Demo Test works identically here — use it any time something "
            "looks wrong."
        )
        rule()

        h2("STEP 12 — Monitor the live market (MONITOR → Live Market)")
        body(
            "A live-updating candlestick chart for any symbol — live via MT5 if connected, delayed via Alpaca "
            "if you've saved keys, or a steady replay of a local CSV if neither is available. Pick a source, "
            "symbol, and timeframe, then click OPEN LIVE CHART."
        )
        rule()

        h1("CREATE — Research Agent")
        body(
            "Ask a local Ollama model to investigate the strategy configured under TEST. It can call "
            "run_backtest, run_prop_simulation, run_monte_carlo, run_walk_forward, run_regime_analysis, "
            "run_parameter_sensitivity, run_cost_stress, compare_strategies, search_research (your research/ "
            "paper library), and search_experiments (this app's own memory of every past strategy test) — "
            "each one runs this app's real, already-validated engine, never a guess."
        )
        warn(
            "The agent can only RECOMMEND a next step (e.g. a parameter worth testing) — it has no tool that "
            "edits or applies code, so nothing it suggests ever takes effect until you act on it yourself "
            "through OPTIMIZE → Iterative Refinement / Quick Optimize / Full Pipeline."
        )
        bullet("One-time setup: install Ollama (ollama.com/download), enable it on this tab, and enter the "
               "host/model — everything runs locally, nothing is sent to the cloud.")
        bullet("EMBED RESEARCH LIBRARY — indexes whatever papers you've dropped into the research/ folder so "
               "the agent (and Generate Strategies, under CREATE) can search them by meaning, "
               "not just keyword.")
        bullet("VIEW LEADERBOARD — every strategy test recorded from OPTIMIZE → Full Pipeline, Quick Optimize, or a "
               "Batch Test, ranked and diffed against its own parent configuration so you can see exactly "
               "what changed between a KEEP and a DISCARD.")
        bullet("RUN RESEARCH LOOP — a closed loop: hypothesis -> Ollama drafts a strategy -> backtest -> "
               "Monte Carlo -> prop-survival check -> a real computed failure diagnosis if it doesn't hold "
               "up -> Ollama refines the hypothesis -> repeat. Safe to leave running; it dedupes against "
               "strategies with DNA it's already tried and failed.")
        rule()

        # -------------------------------------------------------------
        # Decision guide -- real branching logic, not just a numbered
        # list. Everything above walks the tabs in a sensible first-pass
        # order; this section is organized instead around the decision
        # points that actually come up once you have real results in
        # front of you -- "I got X, now what?" -- including the dynamic
        # "Next step:" messages this app now prints live after Run &
        # Report, Search Lab, Evolution Lab, and Full Pipeline (see
        # app.orchestration.pipeline_guide -- the SAME logic drives both
        # those live messages and this reference section, so they can
        # never disagree with each other).
        # -------------------------------------------------------------
        h1("DECISION GUIDE — what to do depending on your results")
        body(
            "Everything above is the order to click through the FIRST time. This section is different: "
            "it's organized by decision point, not by tab, for when you already have a real result in "
            "front of you and want to know what it means. Every branch below also prints live, as a "
            "green 'Next step:' note, right after the run that produced it -- this section exists so you "
            "can look ahead, or look a decision up again later without re-running anything."
        )

        h2("After your very first backtest (TEST → Run & Report)")
        bullet("Zero trades generated → fix the entry logic before touching risk or prop rules (nothing "
               "downstream matters until trades exist). Manual Builder: open 1 Strategy Configuration and "
               "check each indicator/condition row for an overly narrow threshold or two conditions that "
               "can never both be true. Uploaded code: check the entry function directly. Either way, "
               "confirm on 2 Market Data that the loaded timeframe matches what the strategy expects.")
        bullet("Fewer than ~20 trades → too few for Monte Carlo or a prop simulation to say much with "
               "confidence; treat any pass-probability number as noise until this is well past 20-30. "
               "On 2 Market Data, load a longer date range or a lower timeframe; on 1 Strategy "
               "Configuration, loosen the entry threshold so it fires more often.")
        bullet("Profit factor below 1.0 → fix the edge first, don't optimize a losing strategy. Open 5 "
               "Run & Report's trade breakdown: low win rate + small wins/big losses → tighten the "
               "stop-loss or add a profit target (1 Strategy Configuration / 4 Risk & Execution); decent "
               "win rate but still net-negative → the entry filter is too loose, add a confirming "
               "condition. Manual Builder: adjust by hand. Generated/uploaded: let Search Lab or "
               "Evolution Lab explore variations instead of hand-tuning code.")
        bullet("Max drawdown above ~40% → severe for most prop-firm limits (many cap overall drawdown "
               "around 8-10%). On 4 Risk & Execution, lower position size / risk-per-trade first (the "
               "fastest lever), then tighten or add a stop-loss if one is wide or missing. Re-run 5 Run "
               "& Report and confirm drawdown actually came down before moving on, even if profit "
               "factor looks fine.")
        bullet("Failed the prop-firm simulation despite decent trade stats → check the failure reason "
               "shown in the report. Daily loss limit and max drawdown are the two most common single-run "
               "killers — re-check that 03 Prop Rules actually matches your real firm's terms.")
        bullet("None of the above → reasonable first result. Check the eval-pass / first-payout "
               "probabilities the same run just computed, then either promote to Full Pipeline directly, "
               "or try Search Lab / Evolution Lab first to see if a nearby variant does better.")

        h2("Choosing where to start creating a strategy")
        bullet("Already have a specific setup in mind (rule-based, or an MS-LSD-style structure/liquidity/"
               "order-block idea, or you already have Python/PineScript/MQL5 source) → Manual Builder or "
               "upload it directly. Search Lab and Evolution Lab are for when you DON'T have a specific "
               "idea — skip them for now and come back later to see if a variant beats your own design.")
        bullet("No specific idea yet → Search Lab (bounded, three-stage, faster) or Evolution Lab "
               "(open-ended, meant to run for hours unattended) are both better starting points than a "
               "blank Manual Builder. Use Search Lab for a quick first pass; use Evolution Lab if you "
               "want to leave it running and come back to a leaderboard.")

        h2("Search Lab / Evolution Lab leaderboard, once you have one")
        bullet("Nothing survived Stage 1 at all (Search Lab) → try a wider family selection, a higher "
               "max-candidates count, or looser Stage 1 filters (min trades / min profit factor).")
        bullet("Some candidates made it partway but none passed every gate → promote the top one and run "
               "it through Full Pipeline anyway, or widen the search and try again.")
        bullet("A clear champion / strong leaderboard → click PROMOTE, then run it through Full Pipeline "
               "for a re-validated, walk-forward-optimized verdict before considering a live evaluation.")
        warn(
            "Evolution Lab's own leaderboard 'eval pass probability' is scored IN-SAMPLE, against the "
            "same data the GA searched with — it will usually look better than an honest out-of-sample "
            "number. Check the CPCV OOS pass-probability column next to it; a big gap between the two "
            "means overfitting, not a strong strategy. Full Pipeline re-checks this properly."
        )
        warn(
            "Picking the single best candidate out of a large batch (dozens of Search Lab candidates, "
            "or a whole Evolution Lab run) is exactly the multiple-comparisons problem: the more variants "
            "tried, the more likely the 'best' one just got lucky rather than found a real edge. Before "
            "trusting profit factor/Sharpe alone to declare a winner, run the candidate pool through 09 "
            "CPCV / PBO (Probability of Backtest Overfitting) — a PBO comfortably under 50% means the "
            "pick likely reflects a real edge; at or above 50% means the win was statistically no better "
            "than a coin flip. The live 'Next step' note after a Search Lab or Evolution Lab run now "
            "flags this automatically once enough candidates were tried."
        )

        h2("Full Pipeline verdict")
        bullet("READY → your strongest evidence yet, but still a backtest. Consider a short Forward Test "
               "(MT5) or paper trading before risking real capital on a prop-firm evaluation.")
        bullet("MARGINAL → not clearly ready or clearly dead. Try Quick Optimize on it, or adjust risk "
               "settings (position size, daily-loss limit) and re-run Full Pipeline. Don't take it live "
               "as-is.")
        bullet("NOT READY → doesn't hold up under re-validation. Go back to Evolution Lab or Search Lab "
               "for a different candidate rather than trying to rescue this one.")

        h2("Forward Test (MT5) results")
        bullet("Any errors during the session → almost always broker/connection/symbol-mapping issues, "
               "not the strategy. Resolve them and run another session — a session with unresolved "
               "errors proves nothing either way.")
        bullet("Zero trades triggered → can be entirely normal for a low-frequency strategy over a short "
               "window. Run a longer session before concluding anything, especially if the backtest also "
               "traded infrequently.")
        bullet("Live fills diverged from what the backtest predicted → check slippage/spread/commission "
               "assumptions on 04 Risk & Execution against what the broker actually gave you, then "
               "re-run Full Pipeline with updated execution costs.")
        bullet("Ran cleanly and roughly matched the backtest → this is your strongest evidence yet. "
               "Deploy Live is reasonable from here, starting at reduced size if your firm allows it, "
               "with Monitor (Live Market) open to watch the first stretch.")

        h2("Troubleshooting things that have gone wrong before")
        bullet("Evolution Lab winners pass on its own leaderboard but fail almost every Full Pipeline "
               "re-check → check that Evolution Lab's own 'Prop account & risk' fields (right there on "
               "that tab) actually match your real numbers. They're shared live with 03 Prop Rules / "
               "04 Risk, so if either was left at the generic $100k/8%/5%/10%/1%-risk defaults while "
               "you filled in real numbers somewhere else, Evolution Lab was searching under different "
               "rules than Full Pipeline checks against — not a bug in either tool, a mismatch between "
               "them.")
        bullet("Search Lab or Evolution Lab appears to stall partway through a batch → STOP and START "
               "again; both now detect a genuinely hung worker and recover automatically rather than "
               "waiting forever, but a manual restart is always safe if you don't want to wait it out.")
        bullet("Evolution Lab keeps returning the same 2-3 strategy families every run → click VIEW "
               "FAMILY HEALTH — a family that's had many tests with zero successes gets deprioritized, "
               "with a floor so it's never fully excluded. RESET FAMILY HEALTH if you've since fixed a "
               "bug that was unfairly tanking a family's record.")
        rule()

        # -------------------------------------------------------------
        # Short version recap
        # -------------------------------------------------------------
        h1("SHORT VERSION — RECAP")
        body("Once you've done the full walkthrough once, this is all you need to remember for next time:")
        numstep(1, "TEST → Market Data → select or fetch your data.")
        numstep(2, "CREATE → Strategy Builder → pick/build your strategy (or load/optimize one from the Strategy Library).")
        numstep(3, "TEST → Prop-Firm Rules + Risk & Execution → confirm these still match your firm/settings.")
        numstep(4, "OPTIMIZE → Full Pipeline → RUN FULL PIPELINE → read the verdict.")
        numstep(5, "If READY and you want to see it trade live → FORWARD TEST → Forward Test (MT5) (paper) or "
                    "DEPLOY → Deploy Live (real capital, once you're confident).")
        tip(
            "That's the whole loop. OPTIMIZE, VALIDATE, and CHAMPION in the sidebar are all "
            "there for when you want to dig deeper — none of them are required to get your first result."
        )

        text.config(state="disabled")
