# T58 Quickstart — your first backtest in 10 minutes

Assumes [INSTALL.md](INSTALL.md) is done and the app is open.

## 1. Pick your data (2 min)

1. Open **Test** (the sidebar's *Test a Strategy* button on the Dashboard goes there too).
2. In *Market data*, choose a dataset from the market-data pack (for example ES or NQ, 5-minute).
   The canonical NQ / ES / MNQ / MES history has a sealed final year that the search never sees.
3. Pick your **prop firm** (FTMO, Apex, Topstep, The5%ers, FundedNext or Lucid, including 25k accounts). The app fills in that firm's rules.
   If the rules were last verified more than 30 days ago, a small tick-box appears: read the firm's page, tick it, and it is good for another 30 days.

## 2. Pick a strategy (2 min)

1. Choose a saved strategy from the **Strategy Library**, or go to **Create** to paste one or describe an idea in plain English.
   No strategy yet? **Create → Strategy Discovery Engine** finds candidates for you.
2. Leave *Advanced options* alone for the first run; every section there says what it controls.

> Don't trust a strategy just because its equity curve goes up. The headline on every result is a stage on the
> **verdict ladder**: Reject → Lead → Forward-Test Candidate → Evaluate → Funded. The older READY / MARGINAL / NOT READY label is shown as detail.
> A backtest alone can only reach *Forward-Test Candidate*; *Evaluate* needs a forward-test result and *Funded* needs a real payout.

## 3. Run it (3 min)

1. Press **Run the full test**. It runs the baseline, a walk-forward-aware search, Monte Carlo, an out-of-sample check and the sealed holdout.
2. Read the headline stage and the **verdict waterfall** underneath: it lists each step that moved the verdict and what single change would move it.
3. The page always ends with a *what to do next* box that is specific to your strategy and stage.

## 4. What the numbers mean (3 min)

Click any term in the app (win rate, trades, profit factor, drawdown, T58 score, READY bar) for a plain-language explanation.

- **Per-attempt pass odds** — the honest number: across thousands of simulated eval attempts, how often this strategy would pass.
  READY needs a 70%+ lower bound, at most 20% evaluation breach and **300+ trades**.
- **Walk-forward efficiency** — did the edge survive on data the optimizer never saw? Below ~0.4 is a red flag.
- **Lookahead check** — confirms the strategy never peeked at future bars. If this ever fails, the backtest is untrustworthy, full stop.

## Next steps

- **Strategy Discovery Engine** (Create) — no strategy yet? Pick a market
  and a prop firm and the engine generates, screens and keeps survivors.
  Speed Run, Search Lab and Evolution Lab are presets of it.
- **Optimize** — tunes a strategy's numbers with a walk-forward-aware
  search. It searches the most recent 150,000 bars by default; raise
  "Max bars" under Advanced to use more history.
- **Champion** — promote validated strategies; each row shows its T58 score, a short promotion assessment and a *Promote to Champion* button.
- **Deploy** — connect a prop account (the page walks you through it), then deploy a Champion. The portfolio overview shows each account's firm, live P&L, trades and algo health.
- **Validate** — tick any of the eight checks (Monte Carlo, CPCV, PBO,
  Walk-forward, Walk-forward GA, Sensitivity, Robustness, Regime) and run
  them together or one at a time.

> **[OWEN: DECIDE]** — Point buyers at your onboarding video / docs hub
> here if you have one (URL), or delete this line.

## One honest warning

No backtest — this app's included — can promise future profits. What the
app *can* promise is that it won't lie to you about the past: realistic
costs, no lookahead, and validation gates that say "no" when the edge
isn't there. Trade what survives that, forward-test it on demo, and size
positions like the money matters — because it does.
