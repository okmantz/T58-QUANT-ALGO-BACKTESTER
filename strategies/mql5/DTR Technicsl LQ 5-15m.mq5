//+------------------------------------------------------------------+
//| DTR Technical LQ -- T58 Backtester edition (INTRADAY)            |
//|                                                                  |
//| Port of DTR_Technical_LQ_AutoTrader.mq5 to the MQL5 subset the   |
//| T58 backtester parses (strategies/STRATEGY_FORMAT.md, section 4). |
//| Written in the backtester's dialect: it is NOT compiled in        |
//| MetaEditor. The original multi-file EA is the live-trading build. |
//|                                                                  |
//| Chain (same order as the EA):                                     |
//|   regime -> liquidity pool -> sweep + buffer -> reclaim ->        |
//|   1.5x ATR displacement + follow-through -> T1 room guard         |
//|                                                                  |
//| Execution = M5. Context bars (last CLOSED bar only):              |
//|   tf15_* = M15,  tf60_* = H1,  tf1440_* = previous D1 bar.        |
//| LONG  = sweep of BSL (liquidity below lows), reclaim up.          |
//| SHORT = sweep of SSL (liquidity above highs), reclaim down.       |
//|                                                                  |
//| Kept from the EA: 1.5x ATR displacement body; follow-through past |
//|   the last closed M15 extreme; Tier I (prior H1) and Tier E       |
//|   (prior D1) pools; sweep must exceed the pool by a buffer and    |
//|   close back through it; 12-bar sweep window; D-ATR reachability  |
//|   (<1.0); T1 room guard (0.05 x D-range); soft invalidation       |
//|   (reclaim lost); one position at a time; structural-size stop    |
//|   (3.5 x ATR ~ median distance entry -> sweep extreme + 0.1 ATR). |
//|                                                                  |
//| Approximated (the backtester subset has no swing/zone/limit-order |
//| support): H1 swing regime = EMA60/EMA288 + price side; causal PZ  |
//|   not modelled (pool + reclaim stand in); Z4 buffer = 0.25 ATR    |
//|   instead of pips/%; entry = next-bar open instead of a 50%       |
//|   retrace limit; T1 = previous D1 extreme, exited at the close;   |
//|   no partials / breakeven / T2 (engine is single-position, all-in)|
//+------------------------------------------------------------------+
// T58_TIMEFRAME=5m
// T58_HTF=15m,1h,1d
// T58_SL_ATR_MULT=3.5
// T58_ATR_PERIOD=14

void OnTick()
{
   // ---- volatility (execution ATR) --------------------------------
   double atrVal = iATR(_Symbol, PERIOD_CURRENT, 14);

   // ---- technical regime (proxy for H1 swing structure) -----------
   double emaFast = iMA(_Symbol, PERIOD_CURRENT, 60, 0, MODE_EMA, PRICE_CLOSE);
   double emaSlow = iMA(_Symbol, PERIOD_CURRENT, 288, 0, MODE_EMA, PRICE_CLOSE);
   bool regimeLong = emaFast > emaSlow && close > emaSlow;
   bool regimeShort = emaFast < emaSlow && close < emaSlow;

   // ---- sweep window = last 12 execution bars (1 hour) ------------
   double sweepLow = iLowest(_Symbol, PERIOD_CURRENT, MODE_LOW, 12, 0);
   double sweepHigh = iHighest(_Symbol, PERIOD_CURRENT, MODE_HIGH, 12, 0);

   // ---- D-ATR reachability: 24h travelled vs previous day range ---
   double dayHigh = iHighest(_Symbol, PERIOD_CURRENT, MODE_HIGH, 288, 0);
   double dayLow = iLowest(_Symbol, PERIOD_CURRENT, MODE_LOW, 288, 0);
   double dayTravel = dayHigh - dayLow;
   double dayRange = tf1440_high - tf1440_low;
   bool reachable = dayTravel < dayRange;

   // ---- Z4 sweep buffer (ATR-scaled so it works on any symbol) ----
   double buffer = atrVal * 0.25;

   // ---- candle body / displacement --------------------------------
   double body = close - open;
   double bodyDown = open - close;
   double dispMin = atrVal * 1.5;

   // ---- LONG: BSL pools = previous H1 low (tier I), previous D1 low (tier E)
   bool sweptH1Low = sweepLow < (tf60_low - buffer);
   bool sweptD1Low = sweepLow < (tf1440_low - buffer);
   bool reclaimH1Low = close > tf60_low;
   bool reclaimD1Low = close > tf1440_low;
   bool longSweepH1 = sweptH1Low && reclaimH1Low;
   bool longSweepD1 = sweptD1Low && reclaimD1Low && reclaimH1Low;
   bool longSweep = longSweepH1 || longSweepD1;
   bool longDisp = body >= dispMin && close > tf15_high;

   // ---- SHORT: SSL pools = previous H1 high (tier I), previous D1 high (tier E)
   bool sweptH1High = sweepHigh > (tf60_high + buffer);
   bool sweptD1High = sweepHigh > (tf1440_high + buffer);
   bool reclaimH1High = close < tf60_high;
   bool reclaimD1High = close < tf1440_high;
   bool shortSweepH1 = sweptH1High && reclaimH1High;
   bool shortSweepD1 = sweptD1High && reclaimD1High && reclaimH1High;
   bool shortSweep = shortSweepH1 || shortSweepD1;
   bool shortDisp = bodyDown >= dispMin && close < tf15_low;

   // ---- T1 guard: target (opposing D1 pool) must not be consumed and
   //      must sit at least 0.05 x D-range away ("no trade" otherwise)
   double minRoom = dayRange * 0.05;
   double roomLong = tf1440_high - close;
   double roomShort = close - tf1440_low;
   bool t1Long = roomLong > minRoom;
   bool t1Short = roomShort > minRoom;

   // ---- final setup gates ------------------------------------------
   bool longSetup = regimeLong && longSweep && longDisp && reachable && t1Long;
   bool shortSetup = regimeShort && shortSweep && shortDisp && reachable && t1Short;

   // ---- management: soft invalidation + T1 (signal-driven exits) ---
   bool longInvalid = close < tf60_low;
   bool longT1 = close >= tf1440_high;
   bool longOut = longInvalid || longT1;
   bool exitLong = regimeLong && longOut;
   bool shortInvalid = close > tf60_high;
   bool shortT1 = close <= tf1440_low;
   bool shortOut = shortInvalid || shortT1;
   bool exitShort = regimeShort && shortOut;

   if (longSetup)
   {
      trade.Buy(0.1);
   }
   if (shortSetup)
   {
      trade.Sell(0.1);
   }
   if (exitLong)
   {
      trade.PositionClose(0);
   }
   if (exitShort)
   {
      trade.PositionClose(0);
   }
}
