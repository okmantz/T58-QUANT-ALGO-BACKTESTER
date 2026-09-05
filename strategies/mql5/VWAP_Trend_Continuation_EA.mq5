//+------------------------------------------------------------------+
//|                                 VWAP_Trend_Continuation_EA.mq5    |
//| Long : EMA(294) > EMA(58)  AND Close > RollingVWAP(4) AND RSI(18) < 57.6102629644959
//| Short: EMA(35) < EMA(159)  AND Close < RollingVWAP(3) AND RSI(42) > 66.008585331847
//| Exit : RSI(5) crosses exit threshold, ATR stop/target, opposite  |
//|        signal, or max bars in trade.                             |
//+------------------------------------------------------------------+
#property copyright "T58 Trading"
#property version   "1.00"
#property strict

input int    EmaLongSlowLen     = 294;
input int    EmaLongFastLen     = 58;
input int    EmaShortFastLen    = 35;
input int    EmaShortSlowLen    = 159;

input int    RsiLongLen         = 18;
input double RsiLongThresh      = 57.6102629644959;
input int    RsiShortLen        = 42;
input double RsiShortThresh     = 66.008585331847;

input int    RsiExitLen         = 5;
input double RsiExitLongThresh  = 66.68737207001313;
input double RsiExitShortThresh = 64.42442560822603;

input int    VwapLongLen        = 4;
input int    VwapShortLen       = 3;

input int    AtrStopLen         = 16;
input double AtrStopMult        = 4.806786275425379;
input int    AtrTargetLen       = 11;
input double AtrTargetMult      = 5.054720923946611;

input int    MaxBarsInTrade     = 63;
input bool   OppositeSignalExit = true;

input double LotSize            = 0.10;
input ulong  MagicNumber        = 580058;

int hEmaLongSlow, hEmaLongFast, hEmaShortFast, hEmaShortSlow;
int hRsiLong, hRsiShort, hRsiExit;
int hAtrStop, hAtrTarget;

datetime lastBarTime = 0;
int      barsInTrade = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   hEmaLongSlow  = iMA(_Symbol, _Period, EmaLongSlowLen,  0, MODE_EMA, PRICE_CLOSE);
   hEmaLongFast  = iMA(_Symbol, _Period, EmaLongFastLen,  0, MODE_EMA, PRICE_CLOSE);
   hEmaShortFast = iMA(_Symbol, _Period, EmaShortFastLen, 0, MODE_EMA, PRICE_CLOSE);
   hEmaShortSlow = iMA(_Symbol, _Period, EmaShortSlowLen, 0, MODE_EMA, PRICE_CLOSE);

   hRsiLong  = iRSI(_Symbol, _Period, RsiLongLen,  PRICE_CLOSE);
   hRsiShort = iRSI(_Symbol, _Period, RsiShortLen, PRICE_CLOSE);
   hRsiExit  = iRSI(_Symbol, _Period, RsiExitLen,  PRICE_CLOSE);

   hAtrStop   = iATR(_Symbol, _Period, AtrStopLen);
   hAtrTarget = iATR(_Symbol, _Period, AtrTargetLen);

   if(hEmaLongSlow == INVALID_HANDLE || hEmaLongFast == INVALID_HANDLE ||
      hEmaShortFast == INVALID_HANDLE || hEmaShortSlow == INVALID_HANDLE ||
      hRsiLong == INVALID_HANDLE || hRsiShort == INVALID_HANDLE || hRsiExit == INVALID_HANDLE ||
      hAtrStop == INVALID_HANDLE || hAtrTarget == INVALID_HANDLE)
   {
      Print("Failed to create one or more indicator handles");
      return(INIT_FAILED);
   }
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(hEmaLongSlow);
   IndicatorRelease(hEmaLongFast);
   IndicatorRelease(hEmaShortFast);
   IndicatorRelease(hEmaShortSlow);
   IndicatorRelease(hRsiLong);
   IndicatorRelease(hRsiShort);
   IndicatorRelease(hRsiExit);
   IndicatorRelease(hAtrStop);
   IndicatorRelease(hAtrTarget);
}

//+------------------------------------------------------------------+
//| Non-anchored, `len`-bar rolling VWAP, evaluated at `shift`        |
//+------------------------------------------------------------------+
double RollingVWAP(int len, int shift)
{
   double sumPV = 0.0;
   double sumV  = 0.0;
   for(int i = shift; i < shift + len; i++)
   {
      double typical = (iHigh(_Symbol, _Period, i) + iLow(_Symbol, _Period, i) + iClose(_Symbol, _Period, i)) / 3.0;
      long   vol     = iVolume(_Symbol, _Period, i);
      sumPV += typical * (double)vol;
      sumV  += (double)vol;
   }
   if(sumV == 0.0) return(0.0);
   return(sumPV / sumV);
}

//+------------------------------------------------------------------+
double GetBuf(int handle, int shift)
{
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(handle, 0, shift, 1, buf) <= 0) return(EMPTY_VALUE);
   return(buf[0]);
}

//+------------------------------------------------------------------+
bool HasOpenPosition(long &type)
{
   if(PositionSelect(_Symbol) && PositionGetInteger(POSITION_MAGIC) == (long)MagicNumber)
   {
      type = PositionGetInteger(POSITION_TYPE);
      return(true);
   }
   return(false);
}

//+------------------------------------------------------------------+
void OpenTrade(ENUM_ORDER_TYPE type, double price, double sl, double tp)
{
   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action    = TRADE_ACTION_DEAL;
   request.symbol     = _Symbol;
   request.volume     = LotSize;
   request.type       = type;
   request.price      = price;
   request.sl         = sl;
   request.tp         = tp;
   request.deviation  = 10;
   request.magic      = MagicNumber;
   request.comment    = "VWAP_Trend_Continuation";

   if(!OrderSend(request, result))
      Print("OrderSend (open) failed: ", GetLastError());
}

//+------------------------------------------------------------------+
void ClosePosition()
{
   if(!PositionSelect(_Symbol)) return;

   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   long   type = PositionGetInteger(POSITION_TYPE);
   double vol  = PositionGetDouble(POSITION_VOLUME);

   request.action    = TRADE_ACTION_DEAL;
   request.symbol     = _Symbol;
   request.volume     = vol;
   request.type       = (type == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   request.position   = PositionGetInteger(POSITION_TICKET);
   request.price      = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                                                       : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   request.deviation  = 10;
   request.magic       = MagicNumber;

   if(!OrderSend(request, result))
      Print("OrderSend (close) failed: ", GetLastError());
}

//+------------------------------------------------------------------+
void ManagePosition(bool longCondition, bool shortCondition,
                     bool longExitSignal, bool shortExitSignal,
                     double atrStop, double atrTarget)
{
   long posType;
   bool hasPos = HasOpenPosition(posType);

   if(hasPos)
   {
      barsInTrade++;
      bool isLong = (posType == POSITION_TYPE_BUY);

      bool exitNow = false;
      if(isLong  && longExitSignal)                          exitNow = true;
      if(!isLong && shortExitSignal)                          exitNow = true;
      if(OppositeSignalExit && isLong  && shortCondition)     exitNow = true;
      if(OppositeSignalExit && !isLong && longCondition)      exitNow = true;
      if(barsInTrade >= MaxBarsInTrade)                       exitNow = true;

      if(exitNow) ClosePosition();
      return; // SL/TP already attached to the position at open time
   }

   barsInTrade = 0;

   double ask    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   int    digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   if(longCondition)
   {
      double sl = NormalizeDouble(ask - AtrStopMult   * atrStop,   digits);
      double tp = NormalizeDouble(ask + AtrTargetMult * atrTarget, digits);
      OpenTrade(ORDER_TYPE_BUY, ask, sl, tp);
   }
   else if(shortCondition)
   {
      double sl = NormalizeDouble(bid + AtrStopMult   * atrStop,   digits);
      double tp = NormalizeDouble(bid - AtrTargetMult * atrTarget, digits);
      OpenTrade(ORDER_TYPE_SELL, bid, sl, tp);
   }
}

//+------------------------------------------------------------------+
void OnTick()
{
   // Evaluate once per newly closed bar, using shift=1 (last closed bar) values
   // to avoid repainting/look-ahead on the still-forming bar.
   datetime curBarTime = iTime(_Symbol, _Period, 0);
   if(curBarTime == lastBarTime) return;
   lastBarTime = curBarTime;

   double emaLongSlow  = GetBuf(hEmaLongSlow, 1);
   double emaLongFast  = GetBuf(hEmaLongFast, 1);
   double emaShortFast = GetBuf(hEmaShortFast, 1);
   double emaShortSlow = GetBuf(hEmaShortSlow, 1);

   double rsiLong  = GetBuf(hRsiLong, 1);
   double rsiShort = GetBuf(hRsiShort, 1);
   double rsiExit  = GetBuf(hRsiExit, 1);

   double atrStop   = GetBuf(hAtrStop, 1);
   double atrTarget = GetBuf(hAtrTarget, 1);

   double vwapLong  = RollingVWAP(VwapLongLen, 1);
   double vwapShort = RollingVWAP(VwapShortLen, 1);

   double closeBar1 = iClose(_Symbol, _Period, 1);

   bool longCondition  = (emaLongSlow > emaLongFast) && (closeBar1 > vwapLong) && (rsiLong < RsiLongThresh);
   bool shortCondition = (emaShortFast < emaShortSlow) && (closeBar1 < vwapShort) && (rsiShort > RsiShortThresh);

   bool longExitSignal  = (rsiExit > RsiExitLongThresh);
   bool shortExitSignal = (rsiExit < RsiExitShortThresh);

   ManagePosition(longCondition, shortCondition, longExitSignal, shortExitSignal, atrStop, atrTarget);
}
//+------------------------------------------------------------------+
