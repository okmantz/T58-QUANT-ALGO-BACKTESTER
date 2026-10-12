# data/real_account — your REAL trades, for the model check

The backtester has never been compared with a real account (`docs/ACCURACY_OVERHAUL.md`). This folder is where that
comparison lives. Until `last_check.json` here says `agrees: true`, treat every pass probability as an untested model.

## What to put here
1. `real_trades.csv` — the trades a real (or paper/sim) account actually took, ~20+ sessions of ONE strategy:
   `entry_time, exit_time, direction, entry_price, exit_price, pnl[, size]` (see `real_trades_TEMPLATE.csv`).
   Times in UTC (or include an offset). `direction`: long/short/buy/sell/1/-1. Use your broker/platform's export.
2. Keep the OHLCV file for the same instrument and period somewhere under `data/raw/`.

## Run
```
python scripts/run_real_account_check.py \
    --real data/real_account/real_trades.csv \
    --data "data/raw/ES1!/futures_ES.F_1m.parquet" \
    --strategy "strategies/python/ES1! Opening Range Breakout 5m.py" \
    --instrument ES --preset lucid_50k
```
It runs the same strategy over the same sessions, matches trades one-to-one, and writes `last_check.json` / `last_check.md`.
`agrees: true` needs >= 80% of real trades matched, median |P&L difference| within the cost tolerance, and daily P&L agreeing
on >= 80% of days. Fewer than 20 real trades is flagged as weak evidence.

`scripts/` and `app/validation/pre_purchase_gate.py` refuse to call a strategy purchase-ready without a passing check.
