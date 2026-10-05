# Installing T58 Quant Algo Backtester (Windows)

The 15-minute path from download to a running app. No Python, no command
line, no Git required.

> **[OWEN: DECIDE]** — Support channel. This doc tells buyers where to get
> help; pick one and put it below (and in the refund/privacy templates):
> a support email (e.g. support@t58trading.com), a Discord channel, or a
> contact form URL. Nothing here works until you choose.

## What you need

- Windows 10 or 11 (64-bit)
- ~500 MB free disk space for the app, plus room for any market-data packs
  you download later (the slim bundle ships with small example files only)
- An internet connection for activation and data downloads

## Step 1 — Download (2 min)

1. Go to the latest release on the T58 Trading store page
   [OWEN: DECIDE — insert your Whop/store download URL here].
2. Download **T58-Quant-Algo-Backtester-Windows.zip**.

## Step 2 — Unzip (2 min)

1. Right-click the zip → **Extract All…** → choose a folder, e.g.
   `C:\T58\`.
2. Open the extracted folder. You should see
   `T58-Quant-Algo-Backtester.exe`.

> **SmartScreen note:** until the builds are code-signed
> [OWEN: DECIDE — you buy the OV certificate; see SENTRY_SETUP.md's
> sibling notes in the build docs], Windows may show "Windows protected
> your PC / Unknown publisher". Click **More info → Run anyway**. This
> warning goes away permanently once signed builds ship.

## Step 3 — First launch & activation (5 min)

1. Double-click `T58-Quant-Algo-Backtester.exe`. The first launch takes
   longer (one-time setup).
2. When prompted, enter the **license key** from your purchase email and
   click **Activate**. Activation needs internet once; afterwards the app
   works offline for up to 3 days between checks.
3. If activation fails, check the troubleshooting section below before
   contacting support.

## Step 4 — Crash-reporting choice (1 min)

On first launch the app asks whether to send **anonymous crash reports**
(this helps fix bugs faster; it never sends strategy code, market data, or
personal info). Choose **Yes** or **No** — you can change it anytime in
**Settings → Privacy**. Default is **off**.

## Step 5 — Get market data (5 min)

The slim bundle ships with small example files only
(`data/examples/`). For real backtests you need history:

1. Open the **Data Center** tab (desktop) or the data page (web app).
2. Pick a data pack (e.g. "ES 1-minute"), click **Download**.
3. Downloads are checksum-verified automatically and resume if your
   connection drops.

That's it — you're ready. **Next:** [QUICKSTART.md](QUICKSTART.md) runs
your first backtest in 10 minutes.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Unknown publisher" SmartScreen warning | More info → Run anyway (see note above) |
| Activation says "no internet" | Connect once; activation needs a single online check |
| "License key invalid" | Copy/paste from the purchase email (no extra spaces); keys are uppercase |
| App won't start / flashes and closes | Right-click → Run as administrator once; if it persists, contact support with the error text |
| Antivirus quarantines the exe | Whitelist the install folder; unsigned exes are commonly flagged heuristically |

**Support:** [OWEN: DECIDE — support email / Discord / contact URL].
