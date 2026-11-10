# Data: what ships, what is gated

**This file is a GATE, not a note.** `data/data_manifest.json` is generated from the same scan
(`python scripts/build_data_manifest.py`); the pipeline reads it at the start of every run and
`app.data.data_manifest.check_dataset` refuses a READY verdict (caps at NOT READY / MARGINAL) when the dataset is
synthetic, has under 5 years of history, or is missing from the manifest and unreadable.

## What the download bundles

Only **ES, NQ and GC** 1-minute futures history (2020 onward) ships with the app. Every other market (FX, crypto, oil,
indices, metals) is a separate download so the app stays small. Drop extra files into `data/raw/<instrument>/` and run
`python scripts/build_data_manifest.py` to register them; the Data page also imports CSV / Parquet directly.

The pooled-edge test needs 8 markets (`pooled_min_markets=8`). With the 3 bundled markets it reports **INSUFFICIENT**
rather than guessing; add the extended pack to unlock it.

## T58 market data pack (canonical ES / NQ / GC and their micros)

`data/data_pack.json` (written by `python scripts/freeze_data_pack.py`, checked by `--verify` or `/api/data-pack`) freezes one canonical history with a
**sealed window fixed on a calendar date: 2025-09-01**. Bars before it are development data; bars from it onward are sealed and each read of them is a booked look
(`SealedLedger`). The manifest stores a sha256 of every bar in both windows, so a changed hash means the data moved under earlier results.

| Symbol | Source file | Development window | Sealed window | $ per point |
|---|---|---|---|---|
| ES | `ES1!/futures_ES.F_1m.parquet` | 2020-01-01 to 2025-08-31 | 2025-09-01 to 2026-09-01 | 50 |
| NQ | `NQ1!/futures_NQ.F_1m.parquet` | 2020-01-01 to 2025-08-31 | 2025-09-01 to 2026-03-01 | 20 |
| GC | `GC1!/futures_GC.F_1m.parquet` | 2020-08-02 to 2025-08-31 | 2025-09-01 to 2026-07-31 | 100 |
| MES | same bars as ES | same | same | 5 |
| MNQ | same bars as NQ | same | same | 2 |
| MGC | same bars as GC | same | same | 10 |

The NQ file ends in March 2026, so its sealed window is about 6 months; extend the NQ history, then re-freeze on purpose
(a re-freeze voids earlier sealed-slice evidence). Micros reuse the full-size prices.

## Bundled file audit

| file | symbol | bars | start | end | years | bar | flags |
|---|---|---|---|---|---|---|---|
| ES1!/futures_ES.F_1m.parquet | ES | 2353209 | 2020-01-01 | 2026-09-01 | 6.67 | 0 days 00:01:00 | RAW ROLL GAPS (back-adjusted at import) |
| GC1!/futures_GC.F_1m.parquet | GC | 2039602 | 2020-08-02 | 2026-07-31 | 5.99 | 0 days 00:01:00 | RAW ROLL GAPS (back-adjusted at import) |
| NQ1!/futures_NQ.F_1m.parquet | NQ | 2164819 | 2020-01-01 | 2026-03-01 | 6.16 | 0 days 00:01:00 | RAW ROLL GAPS (back-adjusted at import) |
