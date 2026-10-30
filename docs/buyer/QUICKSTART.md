# T58 Quickstart — your first backtest in 10 minutes

Assumes [INSTALL.md](INSTALL.md) is done and the app is open.

## 1. Pick your data (2 min)

1. Go to the **Backtest** tab.
2. In the dataset picker, choose a dataset from the market-data pack
   (for example EURUSD 5-minute).
3. Leave the instrument on its default — the app fills in realistic
   spread/slippage/commission for you.

## 2. Pick a strategy (2 min)

1. Open the **Strategy Library**.
2. Double-click any strategy marked **READY** (these passed the full
   validation gauntlet: walk-forward, Monte Carlo, lookahead check).
3. Click **Load into Backtest**.

> Don't trust a strategy just because its equity curve goes up. The
> **verdict** line at the top of every report is the part that matters:
> READY / MARGINAL / NOT READY.

## 3. Run it (3 min)

1. Click **Run Backtest**.
2. Read the summary: net profit, profit factor, max drawdown — all **after**
   spread, slippage, and per-contract commissions.
3. Open the **Prop Simulation** section: pick your prop firm and account
   size to see the strategy's simulated pass odds under that firm's real
   rules (drawdown, daily loss, consistency).

## 4. What the numbers mean (3 min)

- **Per-attempt pass odds** — the honest number: across thousands of
  simulated eval attempts, how often this strategy would pass. The app
  gates READY verdicts at 70%+ (lower bound).
- **Walk-forward efficiency** — did the edge survive on data the
  optimizer never saw? Below ~0.4 is a red flag.
- **Lookahead check** — confirms the strategy never peeked at future bars.
  If this ever fails, the backtest is untrustworthy, full stop.

## Next steps

- **Strategy Discovery Engine** (Create) — no strategy yet? Pick a market
  and a prop firm and the engine generates, screens and keeps survivors.
  Speed Run, Search Lab and Evolution Lab are presets of it.
- **Optimize** — tunes a strategy's numbers with a walk-forward-aware
  search. It searches the most recent 150,000 bars by default; raise
  "Max bars" under Advanced to use more history.
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
