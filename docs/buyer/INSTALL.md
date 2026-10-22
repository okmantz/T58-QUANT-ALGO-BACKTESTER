# Installing T58 Quant Algo Backtester (Windows)

The 15-minute path from download to a running app. No Python, no command
line, no Git required.

> **[OWEN: DECIDE]** — Support channel. This doc tells buyers where to get
> help; pick one and put it below (and in the refund/privacy templates):
> a support email (e.g. support@t58trading.com), a Discord channel, or a
> contact form URL. Nothing here works until you choose.

## What you need

- Windows 10 or 11 (64-bit)
- ~1.5 GB free disk space for the app, plus ~1 GB if you also install the
  market-data pack
- An internet connection for activation and the downloads

## Step 1 — Download (2 min)

1. Go to the latest release on the T58 Trading store page
   [OWEN: DECIDE — insert your Whop/store download URL here].
2. Download **T58-Quant-Algo-Backtester-Windows.zip** (the app).
3. Download **T58-Market-Data.zip** (the historical market data). It is a
   separate file on purpose, so the app download stays small and either
   download can be retried on its own. The app starts without it and ships
   a small sample dataset, but you need it for real backtests.

## Step 2 — Unzip (2–10 min)

1. Right-click **T58-Quant-Algo-Backtester-Windows.zip** → **Extract All…**
   → choose a **short** folder, e.g. `C:\T58\`. (A short path avoids the
   Windows "path too long" error. The app is a few thousand small files,
   so Windows' built-in extractor can take several minutes — the free
   [7-Zip](https://www.7-zip.org) is much faster.)
2. Open the extracted folder. You should see
   `T58-Quant-Algo-Backtester.exe` and a folder called `_internal`. Keep
   them together — the app does not start without `_internal`.
3. Extract **T58-Market-Data.zip** into **that same folder** (the one that
   holds the `.exe`). Windows asks to merge the `data` folder — say yes.
   You end up with `data\raw\EURUSD\…` next to the `.exe`.
4. Never run the `.exe` from inside the zip window: extract first.

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
   (Owner/support: the master key also activates, fully offline — see [../LICENSING.md](../LICENSING.md).)
3. If activation fails, check the troubleshooting section below before
   contacting support.

## Step 4 — Crash-reporting choice (1 min)

On first launch the app asks whether to send **anonymous crash reports**
(this helps fix bugs faster; it never sends strategy code, market data, or
personal info). Choose **Yes** or **No** — you can change it anytime in
**Settings → Privacy**. Default is **off**.

## Step 5 — Get market data (5 min)

If you extracted **T58-Market-Data.zip** in Step 2, you already have the
historical datasets: restart the app and they appear in the dataset lists.
Without it the app opens but has no datasets to test on.

To add your own data at any time: **Market Data tab → Import folder…**
(desktop) or the drag-and-drop folder zone in the **Data Center** (web app).
Each file's SHA-256 is published next to the zips (`*.zip.sha256`) if you
want to verify a download.

That's it — you're ready. **Next:** [QUICKSTART.md](QUICKSTART.md) runs
your first backtest in 10 minutes.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Unknown publisher" SmartScreen warning | More info → Run anyway (see note above) |
| Activation says "no internet" | Connect once; activation needs a single online check |
| "License key invalid" | Copy/paste from the purchase email (no extra spaces); keys are uppercase |
| App won't start / flashes and closes | Right-click → Run as administrator once; if it persists, send support the newest file in the app's `data\logs` folder |
| "Path too long" or "cannot create file" while extracting | Extract to a shorter folder such as `C:\T58\`, or use 7-Zip |
| "Windows cannot complete the extraction" / zip "invalid" | Re-download (compare the zip's SHA-256 with the `.zip.sha256` file), then extract with 7-Zip |
| "Missing _internal" / python DLL not found | You ran the `.exe` from inside the zip, or moved it away from `_internal`. Extract the whole zip and run it from there |
| Antivirus quarantines a file | Restore it and add the install folder to the exclusion list; unsigned exes are commonly flagged heuristically |

**Support:** [OWEN: DECIDE — support email / Discord / contact URL].
