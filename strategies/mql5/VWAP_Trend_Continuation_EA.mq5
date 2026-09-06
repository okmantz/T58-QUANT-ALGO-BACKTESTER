// VWAP Trend Continuation (ema 50/100, rsi7)
//
// IMPORTANT -- written for the T58 Quant Algo Backtester's MQL5 importer
// (app/strategy/mql5.py), which parses a narrow SUBSET of MQL5: direct-
// value iMA(...)/iRSI(...) calls with LITERAL integer periods (not named
// input variables -- the parser cannot resolve a variable name to a
// number), C-style boolean conditions, and trade.Buy()/trade.Sell()/
// trade.PositionClose() calls guarded by an `if`. It does NOT support
// indicator handles + CopyBuffer(), OrderSend() with an MqlTradeRequest
// struct, or OnInit()/OnTick() event structure -- the previous version
// used all three, so the parser found zero recognizable Buy/Sell calls
// and produced no trades.
//
// VWAP SUBSTITUTION: this engine's own manual/JSON VWAP indicator ignores
// the "period" field entirely and uses a session-anchored cumulative
// VWAP -- not expressible in this parser's supported function list (only
// iMA/iRSI on the close price, plus +-*/ arithmetic over already-defined
// series; no volume-weighted cumulative sum, and iMA here always reads
// the close price regardless of the PRICE_* argument). iMA(..., MODE_SMA,
// ...) below is used as the closest available stand-in.
//
// ATR PERIOD LIMIT: this parser supports only ONE T58_ATR_PERIOD shared by
// both the stop and target multipliers (the JSON spec's stop_atr_period=16
// and target_atr_period=11 can't both be expressed here) -- 16 is used
// below for both.
//
// EXIT SCOPE LIMIT: this parser combines every trade.PositionClose(...)
// guard into a SINGLE exit condition applied to whichever side is open
// (there's no separate long-exit/short-exit distinction like the Pine
// adapter's "Long"/"Short" trade IDs) -- both RSI-exit guards below are
// therefore each capable of closing either a long or a short.
//
// MAX-BARS-IN-TRADE: this parser has no max_bars_in_trade equivalent
// (only the "manual" JSON adapter supports it) -- not represented here.

// T58_SL_ATR_MULT=4.806786275425379
// T58_TP_ATR_MULT=5.054720923946611
// T58_ATR_PERIOD=16

double emaLongSlow  = iMA(_Symbol, PERIOD_CURRENT, 294, 0, MODE_EMA, PRICE_CLOSE);
double emaLongFast  = iMA(_Symbol, PERIOD_CURRENT, 58,  0, MODE_EMA, PRICE_CLOSE);
double emaShortFast = iMA(_Symbol, PERIOD_CURRENT, 35,  0, MODE_EMA, PRICE_CLOSE);
double emaShortSlow = iMA(_Symbol, PERIOD_CURRENT, 159, 0, MODE_EMA, PRICE_CLOSE);

double vwapLong  = iMA(_Symbol, PERIOD_CURRENT, 4, 0, MODE_SMA, PRICE_CLOSE);
double vwapShort = iMA(_Symbol, PERIOD_CURRENT, 3, 0, MODE_SMA, PRICE_CLOSE);

double rsiLong  = iRSI(_Symbol, PERIOD_CURRENT, 18, PRICE_CLOSE);
double rsiShort = iRSI(_Symbol, PERIOD_CURRENT, 42, PRICE_CLOSE);
double rsiExit  = iRSI(_Symbol, PERIOD_CURRENT, 5,  PRICE_CLOSE);

if (emaLongSlow > emaLongFast && close > vwapLong && rsiLong < 57.6102629644959) {
    trade.Buy(0.10, _Symbol);
}

if (emaShortFast < emaShortSlow && close < vwapShort && rsiShort > 66.008585331847) {
    trade.Sell(0.10, _Symbol);
}

if (rsiExit > 66.68737207001313) {
    trade.PositionClose(_Symbol);
}

if (rsiExit < 64.42442560822603) {
    trade.PositionClose(_Symbol);
}
