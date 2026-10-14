# 04 - Regime, Event Calendar and Data Sources

Why this folder: Report B's gated Noise Area strategy only trades on volatile days (VIX at least 18, or realized range above its 20-day median) and avoids scheduled releases. That needs clean historical regime data and event-day research.

Links came up in searches on 2026-10-12; open and confirm before relying.

## Data

1. FRED - CBOE Volatility Index: VIX (series VIXCLS), daily close, Federal Reserve Bank of St. Louis.
   - Role: history for the volatility gate and for splitting backtest results by VIX bucket (Report B validation step 5). Daily close only, so it cannot gate intraday on its own; use prior-day close as the gate input to avoid lookahead.
   - Link: https://fred.stlouisfed.org/series/VIXCLS

## Event-day research

2. Lucca, Moench - "The Pre-FOMC Announcement Drift", Federal Reserve Bank of New York Staff Report 512 (Journal of Finance 2015).
   - Role: equity returns concentrate in the 24 hours before scheduled FOMC announcements. Report B: calendar filter only (do not fade strength into FOMC); sample ends before 2012 so decay is unknown.
   - Links: https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf
     https://econpapers.repec.org/RePEc:fip:fednsr:512

3. Gamma / 0DTE regime papers cited in Report B (find via the cited titles): Baltussen et al. (folder 01), Adams et al. "Do S&P500 0DTE options increase market volatility?", and the Princeton ORFE senior thesis "0DTE Options and Intraday Volatility" (Apr 2026). Report B notes the evidence is mixed and that GEX is vendor-estimated, so treat any gamma gate as a hypothesis.

## Previously in the repo (regime and microstructure, relevant here)

- "A Validated Volatility-Volume-Gap Classifier for Regime Identification MNQ Data" (01_Validation and Overfitting)
- "Forecasting Intraday Trading Volume" (07_Volume_and_Other_Strategies)
- Order-flow papers in 05_Market Microstructure and Order Flow (Report B says order-flow trading is not feasible for prop evals due to no book data or colocation, so these are background only)

## Gaps I did not fill (find yourself)

- A scheduled-release calendar source you trust for the 2-minute blackout rule (CPI, NFP, FOMC). Use the issuing agencies' own release calendars.
