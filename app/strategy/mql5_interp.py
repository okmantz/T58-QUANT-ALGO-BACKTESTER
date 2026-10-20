"""
MQL5 Expert Advisor interpreter -- the fallback behind app.strategy.mql5.MQL5Strategy.

The fast line-based parser in mql5.py handles single-condition EAs written in the simplified style. Real EAs use handles
(`int h = iMA(...)` + `CopyBuffer`), helper functions, structs (MqlRates, MqlTradeRequest), CTrade, arrays, loops and
several timeframes. This module runs those EAs bar by bar and returns the same +1/0/-1 signal series + stop/target
distances every adapter returns; the backtest engine still does the real fills, stops and costs.

Execution model (documented because EAs are sensitive to it): OnTick runs ONCE per bar, at the open of the next bar --
so `iClose(_Symbol,_Period,1)` is the bar that just closed and Ask/Bid are the next bar's open. Market orders fill at
that open. Netting account semantics (one net position). Lot size is irrelevant to the signal (the engine sizes trades).
Not simulated: pending-order fills (BuyLimit/SellStop... are filled as market orders, with a warning), hedged positions,
trailing logic that modifies stops every tick (stops are fixed at the first level the EA sets), OnTimer/OnTrade events.
"""
from __future__ import annotations

import math
import time as _time

import numpy as np
import pandas as pd

from app.strategy import mql5_lang as L
from app.strategy.base import StrategyError, StrategyResult
from app.strategy.interp_common import (
    NAN, SimBroker, bar_minutes, completed_map, guess_mintick, parse_tf_minutes, resample_to, time_index,
)
from app.strategy.mql5_lang import (
    BREAK, CONT, COERCE, MqlArray, MqlStruct, Parser, Ref, Ret, to_bool, to_float, to_int, to_str, tokenize,
)

DBL_MAX = 1.7976931348623157e308

# ---------------------------------------------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------------------------------------------
_TF_CODES = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 10: 10, 12: 12, 15: 15, 20: 20, 30: 30, 16385: 60, 16386: 120,
             16387: 180, 16388: 240, 16390: 360, 16392: 480, 16396: 720, 16408: 1440, 32769: 10080, 49153: 43200}
CONSTANTS: dict[str, object] = {
    "PERIOD_CURRENT": 0, "PERIOD_M1": 1, "PERIOD_M2": 2, "PERIOD_M3": 3, "PERIOD_M4": 4, "PERIOD_M5": 5, "PERIOD_M6": 6,
    "PERIOD_M10": 10, "PERIOD_M12": 12, "PERIOD_M15": 15, "PERIOD_M20": 20, "PERIOD_M30": 30, "PERIOD_H1": 16385,
    "PERIOD_H2": 16386, "PERIOD_H3": 16387, "PERIOD_H4": 16388, "PERIOD_H6": 16390, "PERIOD_H8": 16392,
    "PERIOD_H12": 16396, "PERIOD_D1": 16408, "PERIOD_W1": 32769, "PERIOD_MN1": 49153,
    "MODE_SMA": 0, "MODE_EMA": 1, "MODE_SMMA": 2, "MODE_LWMA": 3,
    "PRICE_CLOSE": 1, "PRICE_OPEN": 2, "PRICE_HIGH": 3, "PRICE_LOW": 4, "PRICE_MEDIAN": 5, "PRICE_TYPICAL": 6,
    "PRICE_WEIGHTED": 7,
    "MODE_MAIN": 0, "MODE_SIGNAL": 1, "MODE_PLUSDI": 1, "MODE_MINUSDI": 2, "MODE_UPPER": 1, "MODE_LOWER": 2,
    "MODE_GATORJAW": 1, "BASE_LINE": 0, "UPPER_BAND": 1, "LOWER_BAND": 2, "MAIN_LINE": 0, "SIGNAL_LINE": 1,
    "TENKANSEN_LINE": 0, "KIJUNSEN_LINE": 1, "SENKOUSPANA_LINE": 2, "SENKOUSPANB_LINE": 3, "CHIKOUSPAN_LINE": 4,
    "MODE_OPEN": 0, "MODE_LOW": 1, "MODE_HIGH": 2, "MODE_CLOSE": 3, "MODE_VOLUME": 4, "MODE_TIME": 5,
    "STO_LOWHIGH": 0, "STO_CLOSECLOSE": 1, "VOLUME_TICK": 0, "VOLUME_REAL": 1,
    "ORDER_TYPE_BUY": 0, "ORDER_TYPE_SELL": 1, "ORDER_TYPE_BUY_LIMIT": 2, "ORDER_TYPE_SELL_LIMIT": 3,
    "ORDER_TYPE_BUY_STOP": 4, "ORDER_TYPE_SELL_STOP": 5, "ORDER_TYPE_BUY_STOP_LIMIT": 6, "ORDER_TYPE_SELL_STOP_LIMIT": 7,
    "ORDER_TYPE_CLOSE_BY": 8, "OP_BUY": 0, "OP_SELL": 1, "OP_BUYLIMIT": 2, "OP_SELLLIMIT": 3, "OP_BUYSTOP": 4, "OP_SELLSTOP": 5,
    "POSITION_TYPE_BUY": 0, "POSITION_TYPE_SELL": 1,
    "TRADE_ACTION_DEAL": 1, "TRADE_ACTION_PENDING": 5, "TRADE_ACTION_SLTP": 6, "TRADE_ACTION_MODIFY": 7,
    "TRADE_ACTION_REMOVE": 8, "TRADE_ACTION_CLOSE_BY": 10,
    "ORDER_FILLING_FOK": 0, "ORDER_FILLING_IOC": 1, "ORDER_FILLING_RETURN": 2, "ORDER_TIME_GTC": 0,
    "TRADE_RETCODE_DONE": 10009, "TRADE_RETCODE_PLACED": 10008, "TRADE_RETCODE_REJECT": 10006,
    "INIT_SUCCEEDED": 0, "INIT_FAILED": 1, "INIT_PARAMETERS_INCORRECT": 2, "INVALID_HANDLE": -1, "EMPTY_VALUE": DBL_MAX,
    "WRONG_VALUE": -1, "DBL_MAX": DBL_MAX, "DBL_MIN": 2.2250738585072014e-308, "DBL_EPSILON": 2.220446049250313e-16,
    "INT_MAX": 2147483647, "INT_MIN": -2147483648, "UINT_MAX": 4294967295, "LONG_MAX": 9223372036854775807,
    "LONG_MIN": -9223372036854775808, "SHORT_MAX": 32767, "CHAR_MAX": 127, "NULL": 0, "true": True, "false": False,
    "M_PI": math.pi, "M_E": math.e, "CLR_NONE": -1, "TIME_DATE": 1, "TIME_MINUTES": 2, "TIME_SECONDS": 4,
    "MQL_TESTER": "MQL_TESTER", "MQL_OPTIMIZATION": "MQL_OPTIMIZATION", "MQL_VISUAL_MODE": "MQL_VISUAL_MODE",
    "MQL_PROGRAM_TYPE": "MQL_PROGRAM_TYPE", "PROGRAM_EXPERT": 1, "MQL_TRADE_ALLOWED": "MQL_TRADE_ALLOWED",
    "TERMINAL_TRADE_ALLOWED": "TERMINAL_TRADE_ALLOWED", "ACCOUNT_TRADE_ALLOWED": "ACCOUNT_TRADE_ALLOWED",
    "SUNDAY": 0, "MONDAY": 1, "TUESDAY": 2, "WEDNESDAY": 3, "THURSDAY": 4, "FRIDAY": 5, "SATURDAY": 6,
    "ACCOUNT_MARGIN_MODE_RETAIL_NETTING": 0, "ACCOUNT_MARGIN_MODE_EXCHANGE": 1, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING": 2,
    "SERIES_BARS_COUNT": "SERIES_BARS_COUNT", "SERIES_LASTBAR_DATE": "SERIES_LASTBAR_DATE",
}
for _n in ("POSITION_TICKET", "POSITION_TIME", "POSITION_TYPE", "POSITION_MAGIC", "POSITION_IDENTIFIER", "POSITION_VOLUME",
           "POSITION_PRICE_OPEN", "POSITION_PRICE_CURRENT", "POSITION_SL", "POSITION_TP", "POSITION_SWAP", "POSITION_PROFIT",
           "POSITION_SYMBOL", "POSITION_COMMENT", "SYMBOL_ASK", "SYMBOL_BID", "SYMBOL_LAST", "SYMBOL_POINT", "SYMBOL_DIGITS",
           "SYMBOL_SPREAD", "SYMBOL_VOLUME_MIN", "SYMBOL_VOLUME_MAX", "SYMBOL_VOLUME_STEP", "SYMBOL_TRADE_TICK_VALUE",
           "SYMBOL_TRADE_TICK_SIZE", "SYMBOL_TRADE_CONTRACT_SIZE", "SYMBOL_TRADE_STOPS_LEVEL", "SYMBOL_TRADE_FREEZE_LEVEL",
           "SYMBOL_TRADE_TICK_VALUE_PROFIT", "SYMBOL_TRADE_TICK_VALUE_LOSS", "SYMBOL_TIME", "SYMBOL_SELECT",
           "SYMBOL_TRADE_MODE", "SYMBOL_FILLING_MODE", "SYMBOL_SWAP_LONG", "SYMBOL_SWAP_SHORT",
           "ACCOUNT_BALANCE", "ACCOUNT_EQUITY", "ACCOUNT_MARGIN_FREE", "ACCOUNT_FREEMARGIN", "ACCOUNT_MARGIN",
           "ACCOUNT_PROFIT", "ACCOUNT_LEVERAGE", "ACCOUNT_LOGIN", "ACCOUNT_CREDIT", "ACCOUNT_MARGIN_MODE",
           "ACCOUNT_MARGIN_LEVEL", "ACCOUNT_TRADE_MODE"):
    CONSTANTS[_n] = _n
for _c in ("Red", "Green", "Blue", "White", "Black", "Gray", "Yellow", "Orange", "Aqua", "Lime", "Magenta", "Silver",
           "DodgerBlue", "Gold", "Crimson", "Navy", "Teal", "Purple", "Maroon", "Olive"):
    CONSTANTS["clr" + _c] = 0
CONSTANTS["clrNONE"] = -1
CONSTANTS["TERMINAL_BUILD"] = "TERMINAL_BUILD"


class Handle(float):
    """An indicator handle. As an MQL5 handle it feeds CopyBuffer(); as a float it carries the legacy 'direct value'
    (the iMA(...) call style of MQL4 / the simplified style) so both coding styles work."""
    bufs: list
    view: "View"
    kind: str


class View:
    """One timeframe's bars plus, for every base bar i, the last COMPLETE bar of this timeframe once bar i has closed."""

    def __init__(self, df: pd.DataFrame, minutes: float, base_idx=None, base_minutes=None, same=False):
        self.df = df
        self.minutes = minutes
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.v = df["volume"].to_numpy(float) if "volume" in df.columns else np.zeros(len(df))
        tix = time_index(df)
        self.t = (tix.values.astype("datetime64[s]").astype("int64")).astype(np.int64)
        self.pos = np.arange(len(df)) if same else completed_map(base_idx, base_minutes, tix, minutes)
        self.n = len(df)

    def idx(self, shift: int, i: int) -> int:
        p = int(self.pos[i])
        return p - (shift - 1) if shift >= 1 else p + 1 if (p + 1) < self.n else p


# ---------------------------------------------------------------------------------------------------------------
# indicator buffers (vectorized on a View's frame)
# ---------------------------------------------------------------------------------------------------------------
def _applied(df: pd.DataFrame, code: int) -> pd.Series:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    return {1: c, 2: o, 3: h, 4: l, 5: (h + l) / 2, 6: (h + l + c) / 3, 7: (h + l + 2 * c) / 4}.get(int(code), c)


def _ma(series: pd.Series, period: int, method: int) -> pd.Series:
    from app.strategy import indicators as I
    period = max(int(period), 1)
    if method == 1:
        return I.ema(series, period)
    if method == 2:
        return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    if method == 3:
        return I.wma(series, period)
    return I.sma(series, period)


def _shifted(s: pd.Series, shift: int) -> pd.Series:
    return s.shift(int(shift)) if shift else s


def compute_buffers(kind: str, df: pd.DataFrame, a: list) -> list[np.ndarray]:
    """a = the numeric arguments AFTER (symbol, timeframe). Returns the indicator's buffers, oldest -> newest."""
    from app.strategy import indicators as I
    g = lambda ix, d=0: a[ix] if ix < len(a) else d    # noqa: E731
    if kind == "iMA":
        return [_shifted(_ma(_applied(df, g(3, 1)), g(0, 14), int(g(2, 0))), int(g(1, 0))).to_numpy(float)]
    if kind == "iRSI":
        return [I.rsi(_applied(df, g(1, 1)), int(g(0, 14))).to_numpy(float)]
    if kind == "iATR":
        return [I.atr(df, int(g(0, 14))).to_numpy(float)]
    if kind == "iMACD":
        px = _applied(df, g(3, 1))
        fast, slow, sig = int(g(0, 12)), int(g(1, 26)), int(g(2, 9))
        m = I.ema(px, fast) - I.ema(px, slow)
        s = I.sma(m, sig)
        return [m.to_numpy(float), s.to_numpy(float)]
    if kind == "iBands":
        px = _applied(df, g(3, 1))
        per, dev = int(g(0, 20)), float(g(2, 2.0))
        mid = I.sma(px, per)
        sd = px.rolling(per, min_periods=per).std(ddof=0)
        sh = int(g(1, 0))
        return [_shifted(mid, sh).to_numpy(float), _shifted(mid + dev * sd, sh).to_numpy(float),
                _shifted(mid - dev * sd, sh).to_numpy(float)]
    if kind == "iStochastic":
        kp, dp, slow = int(g(0, 5)), int(g(1, 3)), int(g(2, 3))
        hh = df["high"].rolling(kp, min_periods=kp).max()
        ll = df["low"].rolling(kp, min_periods=kp).min()
        raw = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
        k = raw.rolling(max(slow, 1), min_periods=max(slow, 1)).mean()
        d = k.rolling(max(dp, 1), min_periods=max(dp, 1)).mean()
        return [k.to_numpy(float), d.to_numpy(float)]
    if kind == "iADX":
        p = int(g(0, 14))
        up = df["high"].diff()
        dn = -df["low"].diff()
        pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
        mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
        tr = I._wilder_smooth(I.true_range(df), p)
        pdi = 100 * I._wilder_smooth(pdm, p) / tr.replace(0, np.nan)
        mdi = 100 * I._wilder_smooth(mdm, p) / tr.replace(0, np.nan)
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        return [I._wilder_smooth(dx.fillna(0.0), p).to_numpy(float), pdi.to_numpy(float), mdi.to_numpy(float)]
    if kind == "iCCI":
        tp = _applied(df, g(1, 6))
        per = int(g(0, 14))
        m = I.sma(tp, per)
        mad = tp.rolling(per, min_periods=per).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        return [((tp - m) / (0.015 * mad.replace(0, np.nan))).to_numpy(float)]
    if kind == "iMomentum":
        px = _applied(df, g(1, 1))
        per = int(g(0, 14))
        return [(px / px.shift(per) * 100).to_numpy(float)]
    if kind == "iWPR":
        return [I.williams_r(df, int(g(0, 14))).to_numpy(float)]
    if kind == "iSAR":
        sar, _d = I.parabolic_sar(df, float(g(0, 0.02)), float(g(0, 0.02)), float(g(1, 0.2)))
        return [sar.to_numpy(float)]
    if kind == "iStdDev":
        px = _applied(df, g(3, 1))
        return [_shifted(px.rolling(int(g(0, 20)), min_periods=int(g(0, 20))).std(ddof=0), int(g(1, 0))).to_numpy(float)]
    if kind == "iOBV":
        return [I.obv(df).to_numpy(float)]
    if kind == "iMFI":
        return [I.money_flow_index(df, int(g(0, 14))).to_numpy(float)]
    if kind == "iAO":
        return [I.awesome_oscillator(df).to_numpy(float)]
    if kind in ("iBearsPower", "iBullsPower"):
        e = I.ema(df["close"], int(g(0, 13)))
        return [((df["low"] - e) if kind == "iBearsPower" else (df["high"] - e)).to_numpy(float)]
    if kind == "iDEMA":
        return [_shifted(I.dema(_applied(df, g(2, 1)), int(g(0, 14))), int(g(1, 0))).to_numpy(float)]
    if kind == "iTEMA":
        return [_shifted(I.tema(_applied(df, g(2, 1)), int(g(0, 14))), int(g(1, 0))).to_numpy(float)]
    if kind == "iEnvelopes":
        mid = _ma(_applied(df, g(3, 1)), g(0, 14), int(g(2, 0)))
        dev = float(g(4, 0.1)) / 100.0
        sh = int(g(1, 0))
        return [_shifted(mid * (1 + dev), sh).to_numpy(float), _shifted(mid * (1 - dev), sh).to_numpy(float)]
    if kind == "iIchimoku":
        t, k, sa, sb, ch = I.ichimoku(df, int(g(0, 9)), int(g(1, 26)), int(g(2, 52)))
        return [x.to_numpy(float) for x in (t, k, sa, sb, ch)]
    if kind == "iFractals":
        h, l = df["high"], df["low"]
        up = h.where((h > h.shift(1)) & (h > h.shift(2)) & (h > h.shift(-1)) & (h > h.shift(-2)))
        dn = l.where((l < l.shift(1)) & (l < l.shift(2)) & (l < l.shift(-1)) & (l < l.shift(-2)))
        # a fractal is only confirmed two bars later: shift so nothing is visible before it is real
        return [up.shift(2).to_numpy(float), dn.shift(2).to_numpy(float)]
    raise StrategyError(f"MQL5: indicator {kind}() is not supported by the interpreter.")


# arity of the MQL5 (handle) form AFTER the symbol+timeframe pair -> anything longer is the legacy "value" call
_ARITY = {"iMA": 4, "iRSI": 2, "iATR": 1, "iMACD": 4, "iBands": 4, "iStochastic": 5, "iADX": 1, "iCCI": 2,
          "iMomentum": 2, "iWPR": 1, "iSAR": 2, "iStdDev": 4, "iOBV": 1, "iMFI": 2, "iAO": 0, "iBearsPower": 1,
          "iBullsPower": 1, "iDEMA": 3, "iTEMA": 3, "iEnvelopes": 5, "iIchimoku": 3, "iFractals": 0}
_MULTI_BUFFER_MODE = {"iMACD", "iBands", "iStochastic", "iADX", "iEnvelopes", "iIchimoku", "iFractals"}


# ---------------------------------------------------------------------------------------------------------------
# built-in classes
# ---------------------------------------------------------------------------------------------------------------
class CTradeObj:
    def __init__(self, rt):
        self.rt = rt
        self._ret = 10009

    def SetExpertMagicNumber(self, *a): return None
    def SetDeviationInPoints(self, *a): return None
    def SetTypeFilling(self, *a): return None
    def SetAsyncMode(self, *a): return None
    def SetMarginMode(self): return None
    def LogLevel(self, *a): return None
    def Buy(self, volume=0.0, symbol=None, price=0.0, sl=0.0, tp=0.0, comment=""): return self.rt.open_pos(1, volume, sl, tp)
    def Sell(self, volume=0.0, symbol=None, price=0.0, sl=0.0, tp=0.0, comment=""): return self.rt.open_pos(-1, volume, sl, tp)

    def PositionOpen(self, symbol=None, order_type=0, volume=0.0, price=0.0, sl=0.0, tp=0.0, comment=""):
        return self.rt.open_pos(1 if int(order_type) % 2 == 0 else -1, volume, sl, tp)

    def BuyLimit(self, volume=0.0, price=0.0, symbol=None, sl=0.0, tp=0.0, *a): return self.rt.open_pending(1, volume, sl, tp)
    def BuyStop(self, volume=0.0, price=0.0, symbol=None, sl=0.0, tp=0.0, *a): return self.rt.open_pending(1, volume, sl, tp)
    def SellLimit(self, volume=0.0, price=0.0, symbol=None, sl=0.0, tp=0.0, *a): return self.rt.open_pending(-1, volume, sl, tp)
    def SellStop(self, volume=0.0, price=0.0, symbol=None, sl=0.0, tp=0.0, *a): return self.rt.open_pending(-1, volume, sl, tp)
    def OrderOpen(self, symbol=None, order_type=0, volume=0.0, limit=0.0, price=0.0, sl=0.0, tp=0.0, *a):
        t = int(order_type)
        return self.rt.open_pending(1 if t % 2 == 0 else -1, volume, sl, tp) if t > 1 else self.rt.open_pos(1 if t == 0 else -1, volume, sl, tp)

    def PositionClose(self, ticket_or_symbol=None, deviation=0): return self.rt.close_pos()
    def PositionClosePartial(self, *a): return self.rt.close_pos()
    def PositionModify(self, ticket_or_symbol=None, sl=0.0, tp=0.0): return self.rt.modify_pos(sl, tp)
    def OrderDelete(self, *a): return True
    def OrderModify(self, *a): return True
    def ResultRetcode(self): return 10009
    def ResultRetcodeDescription(self): return "done"
    def ResultOrder(self): return 1
    def ResultDeal(self): return 1
    def ResultVolume(self): return 1.0
    def ResultPrice(self): return float(self.rt.ask())
    def ResultComment(self): return ""
    def RequestVolume(self): return 1.0


class CPositionObj:
    def __init__(self, rt):
        self.rt = rt

    def Select(self, symbol=None): return self.rt.broker.pos != 0
    def SelectByTicket(self, t): return self.rt.broker.pos != 0
    def SelectByIndex(self, i): return self.rt.broker.pos != 0 and int(i) == 0
    def PositionType(self): return 0 if self.rt.broker.pos > 0 else 1
    def Volume(self): return abs(self.rt.broker.pos)
    def PriceOpen(self): return float(self.rt.broker.avg)
    def StopLoss(self): return self.rt.pos_level("sl")
    def TakeProfit(self): return self.rt.pos_level("tp")
    def PriceCurrent(self): return float(self.rt.cur_price())
    def Profit(self): return self.rt.pos_profit()
    def Symbol(self): return "SYMBOL"
    def Magic(self): return 0
    def Ticket(self): return 1000
    def Time(self): return int(self.rt.broker_open_time())
    def Comment(self): return ""
    def Swap(self): return 0.0
    def Commission(self): return 0.0
    def Identifier(self): return 1000


class CSymbolObj:
    def __init__(self, rt):
        self.rt = rt

    def Name(self, *a): return "SYMBOL"
    def Refresh(self): return True
    def RefreshRates(self): return True
    def Select(self, *a): return True
    def Ask(self): return self.rt.ask()
    def Bid(self): return self.rt.ask()
    def Last(self): return self.rt.ask()
    def Point(self): return self.rt.point
    def Digits(self): return self.rt.digits
    def Spread(self): return 0
    def TickSize(self): return self.rt.point
    def TickValue(self): return 1.0
    def LotsMin(self): return 0.01
    def LotsMax(self): return 100.0
    def LotsStep(self): return 0.01
    def StopsLevel(self): return 0
    def ContractSize(self): return 100000.0
    def NormalizePrice(self, p): return round(float(p), self.rt.digits)


class CAccountObj:
    def __init__(self, rt):
        self.rt = rt

    def Balance(self): return 10000.0
    def Equity(self): return 10000.0
    def FreeMargin(self): return 10000.0
    def Margin(self): return 0.0
    def Profit(self): return 0.0
    def Leverage(self): return 100
    def Login(self): return 1


# ---------------------------------------------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------------------------------------------
class MqlRuntime:
    def __init__(self, df: pd.DataFrame, parser: Parser, compiled, global_inits, overrides):
        self.df = df
        self.parser = parser
        self.funcs = compiled
        self.overrides = overrides or {}
        self.globals: dict = {}
        self.statics: dict = {}
        self.warnings: list[str] = []
        self.i = 0
        self.n = len(df)
        self.base_minutes = bar_minutes(df)
        self.base_idx = time_index(df)
        self.base = View(df, self.base_minutes, same=True)
        self.views: dict[float, View] = {}
        self.ind_cache: dict = {}
        self.handles: list[Handle] = []
        self.point = guess_mintick(df)
        self.digits = max(0, int(round(-math.log10(self.point)))) if self.point > 0 else 5
        self.broker: SimBroker | None = None
        self.builtins = _builtins()
        self.depth = 0
        self.in_tick = False
        for typ, name, dims, init, is_input, is_static in parser.globals:
            pass

    # -- misc runtime services used by compiled closures
    def constant(self, name: str):
        if name in CONSTANTS:
            return CONSTANTS[name]
        if name in self.parser.enums:
            return self.parser.enums[name]
        sp = {"_Symbol": "SYMBOL", "_Point": self.point, "_Digits": self.digits, "_Period": int(self.base_minutes),
              "Point": self.point, "Digits": self.digits, "Ask": self.ask(), "Bid": self.ask(), "_LastError": 0,
              "_StopFlag": False, "_UninitReason": 0}
        if name in sp:
            return sp[name]
        if name in self.df.columns:                     # backtester dialect: dataframe columns (tf60_close, ...) = last closed bar
            return float(self.df[name].iloc[self.i])
        raise StrategyError(f"MQL5: unknown name '{name}'.")

    def warn_once(self, msg: str):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def default_value(self, typ: str):
        if typ in L.COERCE:
            return {"string": "", "bool": False, "double": 0.0, "float": 0.0}.get(typ, 0)
        return self.construct(typ, [])

    def new_array(self, typ: str, sizes: list[int]):
        if len(sizes) > 1:
            return [self.new_array(typ, sizes[1:]) for _ in range(sizes[0])]
        arr = MqlArray()
        arr.elem = typ
        for _ in range(sizes[0] if sizes else 0):
            arr.append(self.default_value(typ))
        return arr

    def construct(self, typ: str, args: list):
        if typ == "CTrade":
            return CTradeObj(self)
        if typ == "CPositionInfo":
            return CPositionObj(self)
        if typ == "CSymbolInfo":
            return CSymbolObj(self)
        if typ == "CAccountInfo":
            return CAccountObj(self)
        if typ in self.parser.structs:
            out = MqlStruct()
            for ftype, fname, dims in self.parser.structs[typ]:
                out._f[fname] = self.default_value(ftype) if not dims else MqlArray()
            return out
        if typ in L.BUILTIN_CLASS_TYPES or typ.startswith("Mql"):
            return MqlStruct()
        return 0

    def call_user(self, name: str, vals: list):
        body, defaults = self.funcs[name]
        spec = self.parser.funcs[name]
        env: dict = {}
        for ix, (ptype, pname, by_ref, is_arr, dflt) in enumerate(spec["params"]):
            if ix < len(vals):
                v = vals[ix]
                if not by_ref or is_arr:
                    if isinstance(v, Ref):
                        v = v.get()
                    co = COERCE.get(ptype)
                    if co is not None and not is_arr and v.__class__.__name__ != "Handle":
                        v = co(v)
                env[pname] = v
            elif defaults[ix] is not None:
                env[pname] = defaults[ix](self, env)
            else:
                env[pname] = self.default_value(ptype)
        self.depth += 1
        if self.depth > 200:
            raise StrategyError("MQL5: recursion deeper than 200 calls.")
        try:
            r = body(self, env)
        finally:
            self.depth -= 1
        return r.v if isinstance(r, Ret) else None

    # -- market data helpers
    def view(self, tf) -> View:
        m = _TF_CODES.get(int(tf), None) if isinstance(tf, (int, float)) and not isinstance(tf, bool) else parse_tf_minutes(tf)
        if not m or m <= self.base_minutes * 1.0001:
            return self.base
        v = self.views.get(m)
        if v is None:
            htf = resample_to(self.df, m)
            if len(htf) < 2:
                raise StrategyError("MQL5: not enough data to build the requested higher timeframe.")
            v = self.views[m] = View(htf, m, self.base_idx, self.base_minutes)
        return v

    def ask(self):
        j = self.i + 1
        return float(self.base.o[j]) if j < self.n else float(self.base.c[-1])

    def cur_price(self):
        return self.ask()

    # -- trading
    def open_pos(self, direction: int, volume, sl, tp):
        if self.broker is None or to_float(volume) <= 0:
            return False
        if self.broker.pos != 0 and (1 if self.broker.pos > 0 else -1) == direction:
            return True              # already in that direction (netting)
        self.broker.entry(direction, to_float(volume))
        self._stops(sl, tp, reset=True)
        return True

    def open_pending(self, direction: int, volume, sl, tp):
        self.warn_once("pending orders (Buy/SellLimit, Buy/SellStop) are filled as market orders at the next open")
        return self.open_pos(direction, volume, sl, tp)

    def _stops(self, sl, tp, reset=False):
        sl, tp = to_float(sl), to_float(tp)
        b = self.broker
        if reset:
            b.stop_price = b.limit_price = b.stop_ticks = b.limit_ticks = NAN
        b.set_exit(stop=sl if sl > 0 else None, limit=tp if tp > 0 else None)

    def close_pos(self):
        if self.broker is None or self.broker.pos == 0:
            return False
        self.broker.close()
        return True

    def modify_pos(self, sl, tp):
        if self.broker is None:
            return False
        sl, tp = to_float(sl), to_float(tp)
        b = self.broker
        if sl > 0:
            b.stop_price = sl
        elif sl == 0:
            b.stop_price = NAN
        if tp > 0:
            b.limit_price = tp
        elif tp == 0:
            b.limit_price = NAN
        return True

    def pos_level(self, which):
        b = self.broker
        v = b.stop_price if which == "sl" else b.limit_price
        return 0.0 if v != v else float(v)

    def pos_profit(self):
        b = self.broker
        if b.pos == 0:
            return 0.0
        return float((self.base.o[min(self.i + 1, self.n - 1)] - b.avg) * b.pos)

    def broker_open_time(self):
        b = self.broker
        return int(self.base.t[b.last_entry_bar]) if b.last_entry_bar >= 0 else 0


# ---------------------------------------------------------------------------------------------------------------
# builtin functions  fn(rt, *args)
# ---------------------------------------------------------------------------------------------------------------
def _builtins() -> dict:
    B: dict = {}

    def reg(name):
        def d(fn):
            B[name] = fn
            return fn
        return d

    # ---- math / strings
    for nm, fn in {"MathAbs": abs, "MathSqrt": math.sqrt, "MathExp": math.exp, "MathFloor": math.floor,
                   "MathCeil": math.ceil, "MathSin": math.sin, "MathCos": math.cos, "MathTan": math.tan,
                   "MathArctan": math.atan, "MathArcsin": math.asin, "MathArccos": math.acos, "MathLog": math.log,
                   "MathLog10": math.log10}.items():
        def mk(fn=fn, nm=nm):
            def f(rt, x):
                try:
                    r = fn(to_float(x))
                    return float(r) if nm in ("MathFloor", "MathCeil") else r
                except (ValueError, OverflowError):
                    return NAN
            return f
        B[nm] = mk()
    B["MathAbs"] = lambda rt, x: abs(x)
    B["MathMax"] = lambda rt, a, b: max(a, b)
    B["MathMin"] = lambda rt, a, b: min(a, b)
    B["MathPow"] = lambda rt, a, b: _safe_pow(a, b)
    B["MathMod"] = lambda rt, a, b: math.fmod(a, b) if b else NAN
    B["MathRound"] = lambda rt, x: float(math.floor(to_float(x) + 0.5))
    B["MathIsValidNumber"] = lambda rt, x: not (isinstance(x, float) and (x != x or x in (math.inf, -math.inf)))
    B["MathRand"] = lambda rt: 16384
    B["MathSrand"] = lambda rt, s: None
    B["NormalizeDouble"] = lambda rt, x, d: round(to_float(x) + 0.0, int(d))
    B["DoubleToString"] = lambda rt, x, d=8: f"{to_float(x):.{int(d)}f}"
    B["IntegerToString"] = lambda rt, x, *a: str(to_int(x))
    B["StringToDouble"] = lambda rt, s: to_float(s)
    B["StringToInteger"] = lambda rt, s: to_int(s)
    B["StringLen"] = lambda rt, s: len(to_str(s))
    B["StringSubstr"] = lambda rt, s, start, length=-1: to_str(s)[int(start): (int(start) + int(length)) if int(length) >= 0 else None]
    B["StringFind"] = lambda rt, s, sub, start=0: to_str(s).find(to_str(sub), int(start))
    B["StringToUpper"] = lambda rt, s: to_str(s).upper()
    B["StringToLower"] = lambda rt, s: to_str(s).lower()
    B["StringFormat"] = lambda rt, fmt, *a: _fmt(fmt, a)
    B["StringConcatenate"] = lambda rt, *a: "".join(to_str(x) for x in a)
    B["StringTrimLeft"] = lambda rt, s: 0
    B["StringTrimRight"] = lambda rt, s: 0
    B["StringCompare"] = lambda rt, a, b, cs=True: (a > b) - (a < b)
    for nm in ("Print", "PrintFormat", "Comment", "Alert", "Sleep", "EventSetTimer", "EventKillTimer", "ChartRedraw",
               "ObjectsDeleteAll", "ObjectDelete", "ObjectCreate", "ObjectSetInteger", "ObjectSetDouble", "ObjectSetString",
               "SetIndexBuffer", "IndicatorSetInteger", "IndicatorRelease", "ResetLastError", "ZeroMemory_", "ExpertRemove",
               "SendNotification", "SendMail", "ChartSetInteger", "GlobalVariableSet", "DebugBreak", "MathSrand",
               "TesterStop", "ArrayPrint", "Print_"):
        B[nm] = lambda rt, *a: None
    B["IndicatorRelease"] = lambda rt, h: True
    B["GetLastError"] = lambda rt: 0
    B["IsStopped"] = lambda rt: False
    B["MQLInfoInteger"] = lambda rt, p: {"MQL_TESTER": 1, "MQL_OPTIMIZATION": 0, "MQL_VISUAL_MODE": 0, "MQL_TRADE_ALLOWED": 1,
                                         "MQL_PROGRAM_TYPE": 1}.get(p, 0)
    B["TerminalInfoInteger"] = lambda rt, p: 1
    B["TerminalInfoString"] = lambda rt, p: ""
    B["MQLInfoString"] = lambda rt, p: ""
    B["AccountInfoInteger"] = lambda rt, p: {"ACCOUNT_LEVERAGE": 100, "ACCOUNT_LOGIN": 1, "ACCOUNT_TRADE_ALLOWED": 1,
                                             "ACCOUNT_MARGIN_MODE": 0}.get(p, 0)
    B["AccountInfoDouble"] = lambda rt, p: {"ACCOUNT_BALANCE": 10000.0, "ACCOUNT_EQUITY": 10000.0, "ACCOUNT_MARGIN_FREE": 10000.0,
                                            "ACCOUNT_FREEMARGIN": 10000.0, "ACCOUNT_MARGIN": 0.0, "ACCOUNT_PROFIT": 0.0,
                                            "ACCOUNT_CREDIT": 0.0, "ACCOUNT_MARGIN_LEVEL": 0.0}.get(p, 0.0)
    B["AccountInfoString"] = lambda rt, p: "USD"
    B["GlobalVariableCheck"] = lambda rt, n: False
    B["Symbol"] = lambda rt: "SYMBOL"
    B["Period"] = lambda rt: int(rt.base_minutes)
    B["Point"] = lambda rt: rt.point
    B["Digits"] = lambda rt: rt.digits
    B["SymbolSelect"] = lambda rt, *a: True
    B["SymbolsTotal"] = lambda rt, *a: 1
    B["SymbolName"] = lambda rt, *a: "SYMBOL"

    # ---- arrays
    @reg("ArraySize")
    def _(rt, a): return len(a)

    @reg("ArrayResize")
    def _(rt, a, n, reserve=0):
        n = int(n)
        cur = len(a)
        if n < cur:
            del a[n:]
        else:
            typ = getattr(a, "elem", None) or "double"
            for _ in range(n - cur):
                a.append(rt.default_value(typ))
        return n

    @reg("ArrayInitialize")
    def _(rt, a, v):
        for k in range(len(a)):
            a[k] = v
        return len(a)

    B["ArrayFill"] = lambda rt, a, s, c, v: [a.__setitem__(k, v) for k in range(int(s), min(len(a), int(s) + int(c)))] and None
    B["ArrayFree"] = lambda rt, a: a.clear()

    @reg("ArraySetAsSeries")
    def _(rt, a, flag):
        if isinstance(a, MqlArray):
            a.series = bool(flag)
        return True

    B["ArrayIsSeries"] = lambda rt, a: bool(getattr(a, "series", False))

    @reg("ArrayCopy")
    def _(rt, dst, src, dstart=0, sstart=0, count=-1):
        count = len(src) - int(sstart) if int(count) < 0 else int(count)
        part = list(src[int(sstart): int(sstart) + count])
        while len(dst) < int(dstart) + len(part):
            dst.append(0.0)
        dst[int(dstart): int(dstart) + len(part)] = part
        return len(part)

    @reg("ArrayMaximum")
    def _(rt, a, start=0, count=-1):
        seg = list(a)[int(start): None if int(count) < 0 else int(start) + int(count)]
        if getattr(a, "series", False):
            seg = seg
        return int(start) + int(np.argmax(seg)) if seg else -1

    @reg("ArrayMinimum")
    def _(rt, a, start=0, count=-1):
        seg = list(a)[int(start): None if int(count) < 0 else int(start) + int(count)]
        return int(start) + int(np.argmin(seg)) if seg else -1

    @reg("ArraySort")
    def _(rt, a, *args):
        a.sort()
        return True

    @reg("ZeroMemory")
    def _(rt, x):
        if isinstance(x, MqlStruct):
            x.clear()
        elif isinstance(x, list):
            for k in range(len(x)):
                x[k] = 0.0 if not isinstance(x[k], MqlStruct) else x[k]
        return None

    # ---- time
    B["TimeCurrent"] = lambda rt: int(rt.base.t[min(rt.i + 1, rt.n - 1)])
    B["TimeLocal"] = B["TimeCurrent"]
    B["TimeGMT"] = B["TimeCurrent"]
    B["TimeTradeServer"] = B["TimeCurrent"]

    @reg("TimeToStruct")
    def _(rt, dt, st):
        ts = pd.Timestamp(int(dt), unit="s")
        st.year, st.mon, st.day, st.hour, st.min, st.sec = ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second
        st.day_of_week, st.day_of_year = (ts.dayofweek + 1) % 7, ts.dayofyear
        return True

    @reg("StructToTime")
    def _(rt, st):
        return int(pd.Timestamp(year=int(st.year), month=int(st.mon), day=int(st.day), hour=int(st.hour), minute=int(st.min),
                                second=int(st.sec)).value // 10 ** 9)

    @reg("TimeToString")
    def _(rt, dt, mode=5):
        ts = pd.Timestamp(int(dt), unit="s")
        m = int(mode)
        if m & 1 and m & 2:
            return ts.strftime("%Y.%m.%d %H:%M")
        if m & 1:
            return ts.strftime("%Y.%m.%d")
        return ts.strftime("%H:%M")

    B["TimeDayOfWeek"] = lambda rt, dt: (pd.Timestamp(int(dt), unit="s").dayofweek + 1) % 7
    B["TimeHour"] = lambda rt, dt: pd.Timestamp(int(dt), unit="s").hour
    B["TimeMinute"] = lambda rt, dt: pd.Timestamp(int(dt), unit="s").minute
    B["DayOfWeek"] = lambda rt: (pd.Timestamp(int(B["TimeCurrent"](rt)), unit="s").dayofweek + 1) % 7
    B["Hour"] = lambda rt: pd.Timestamp(int(B["TimeCurrent"](rt)), unit="s").hour
    B["Minute"] = lambda rt: pd.Timestamp(int(B["TimeCurrent"](rt)), unit="s").minute

    # ---- price series
    def _px(rt, tf, shift, which):
        v = rt.view(tf)
        shift = int(shift)
        if v is rt.base:
            j = rt.i + 1 - shift
            if shift == 0:
                if j >= v.n:
                    j = v.n - 1
                return {"o": v.o, "h": v.o, "l": v.o, "c": v.o, "v": np.ones(v.n), "t": v.t.astype(float)}[which][j]
        else:
            j = v.idx(shift, rt.i)
            if shift == 0:
                return {"o": v.o, "h": v.o, "l": v.o, "c": v.o, "v": np.ones(v.n), "t": v.t.astype(float)}[which][min(max(j, 0), v.n - 1)]
        if j < 0 or j >= v.n:
            return 0.0 if which != "t" else 0
        arr = {"o": v.o, "h": v.h, "l": v.l, "c": v.c, "v": v.v, "t": v.t}[which]
        return int(arr[j]) if which == "t" else float(arr[j])

    B["iOpen"] = lambda rt, s, tf, sh: _px(rt, tf, sh, "o")
    B["iHigh"] = lambda rt, s, tf, sh: _px(rt, tf, sh, "h")
    B["iLow"] = lambda rt, s, tf, sh: _px(rt, tf, sh, "l")
    B["iClose"] = lambda rt, s, tf, sh: _px(rt, tf, sh, "c")
    B["iVolume"] = lambda rt, s, tf, sh: _px(rt, tf, sh, "v")
    B["iTime"] = lambda rt, s, tf, sh: _px(rt, tf, sh, "t")
    B["iBars"] = lambda rt, s, tf: int(rt.view(tf).pos[rt.i]) + 1 if rt.view(tf) is not rt.base else rt.i + 2
    B["Bars"] = lambda rt, s, tf=0, *a: B["iBars"](rt, s, tf)
    B["iBarShift"] = lambda rt, s, tf, t, exact=False: _bar_shift(rt, tf, t)

    def _bar_shift(rt, tf, t):
        v = rt.view(tf)
        j = int(np.searchsorted(v.t, int(t), side="right")) - 1
        if v is rt.base:
            return rt.i + 1 - j
        return int(v.pos[rt.i]) - j + 1

    @reg("iHighest")
    def _(rt, s, tf, mode=2, count=0, start=0):
        return _extreme(rt, tf, mode, count, start, True)

    @reg("iLowest")
    def _(rt, s, tf, mode=1, count=0, start=0):
        return _extreme(rt, tf, mode, count, start, False)

    def _extreme(rt, tf, mode, count, start, hi):
        v = rt.view(tf)
        arr = {0: v.o, 1: v.l, 2: v.h, 3: v.c, 4: v.v}.get(int(mode), v.h if hi else v.l)
        count = int(count) or v.n
        best, bs = None, -1
        for sh in range(int(start), int(start) + count):
            j = (rt.i + 1 - sh) if v is rt.base else v.idx(sh, rt.i)
            if j < 0:
                break
            if j >= v.n:
                continue
            x = arr[j]
            if best is None or (x > best if hi else x < best):
                best, bs = x, sh
        return bs

    def _copy_series(rt, tf, start, count, arr, which):
        v = rt.view(tf)
        start, count = int(start), int(count)
        vals = []
        for sh in range(start, start + count):
            j = (rt.i + 1 - sh) if v is rt.base else v.idx(sh, rt.i)
            if j < 0 or j >= v.n:
                if sh > start and j < 0:
                    break
                continue
            if sh == 0:
                vals.append(float(v.o[j]) if which != "t" else int(v.t[j]))
            else:
                a = {"o": v.o, "h": v.h, "l": v.l, "c": v.c, "v": v.v, "t": v.t}[which]
                vals.append(int(a[j]) if which == "t" else float(a[j]))
        if not getattr(arr, "series", False):
            vals = vals[::-1]
        arr[:] = vals
        return len(vals)

    B["CopyClose"] = lambda rt, s, tf, st, c, arr: _copy_series(rt, tf, st, c, arr, "c")
    B["CopyOpen"] = lambda rt, s, tf, st, c, arr: _copy_series(rt, tf, st, c, arr, "o")
    B["CopyHigh"] = lambda rt, s, tf, st, c, arr: _copy_series(rt, tf, st, c, arr, "h")
    B["CopyLow"] = lambda rt, s, tf, st, c, arr: _copy_series(rt, tf, st, c, arr, "l")
    B["CopyTime"] = lambda rt, s, tf, st, c, arr: _copy_series(rt, tf, st, c, arr, "t")
    B["CopyTickVolume"] = lambda rt, s, tf, st, c, arr: _copy_series(rt, tf, st, c, arr, "v")
    B["CopyRealVolume"] = lambda rt, s, tf, st, c, arr: _copy_series(rt, tf, st, c, arr, "v")

    @reg("CopyRates")
    def _(rt, s, tf, start, count, arr):
        v = rt.view(tf)
        out = []
        for sh in range(int(start), int(start) + int(count)):
            j = (rt.i + 1 - sh) if v is rt.base else v.idx(sh, rt.i)
            if j < 0 or j >= v.n:
                continue
            st = MqlStruct()
            if sh == 0:
                o = float(v.o[j])
                st.open = st.high = st.low = st.close = o
            else:
                st.open, st.high, st.low, st.close = float(v.o[j]), float(v.h[j]), float(v.l[j]), float(v.c[j])
            st.time, st.tick_volume, st.spread, st.real_volume = int(v.t[j]), int(v.v[j]), 0, int(v.v[j])
            out.append(st)
        if not getattr(arr, "series", False):
            out = out[::-1]
        arr[:] = out
        return len(out)

    @reg("SymbolInfoDouble")
    def _(rt, sym, prop):
        return {"SYMBOL_ASK": rt.ask(), "SYMBOL_BID": rt.ask(), "SYMBOL_LAST": rt.ask(), "SYMBOL_POINT": rt.point,
                "SYMBOL_VOLUME_MIN": 0.01, "SYMBOL_VOLUME_MAX": 100.0, "SYMBOL_VOLUME_STEP": 0.01,
                "SYMBOL_TRADE_TICK_VALUE": 1.0, "SYMBOL_TRADE_TICK_VALUE_PROFIT": 1.0, "SYMBOL_TRADE_TICK_VALUE_LOSS": 1.0,
                "SYMBOL_TRADE_TICK_SIZE": rt.point, "SYMBOL_TRADE_CONTRACT_SIZE": 100000.0, "SYMBOL_SWAP_LONG": 0.0,
                "SYMBOL_SWAP_SHORT": 0.0}.get(prop, 0.0)

    B["SymbolInfoInteger"] = lambda rt, sym, prop: {"SYMBOL_DIGITS": rt.digits, "SYMBOL_SPREAD": 0, "SYMBOL_TRADE_STOPS_LEVEL": 0,
                                                    "SYMBOL_TRADE_FREEZE_LEVEL": 0, "SYMBOL_TIME": B["TimeCurrent"](rt),
                                                    "SYMBOL_SELECT": 1, "SYMBOL_TRADE_MODE": 4,
                                                    "SYMBOL_FILLING_MODE": 3}.get(prop, 0)
    B["SymbolInfoString"] = lambda rt, sym, prop: "SYMBOL"

    @reg("SymbolInfoTick")
    def _(rt, sym, tick):
        p = rt.ask()
        tick.ask = tick.bid = tick.last = p
        tick.time = B["TimeCurrent"](rt)
        tick.volume = 1
        return True

    # ---- positions / orders
    B["PositionsTotal"] = lambda rt: 1 if rt.broker.pos != 0 else 0
    B["OrdersTotal"] = lambda rt: 0
    B["HistoryOrdersTotal"] = lambda rt: 0
    B["HistoryDealsTotal"] = lambda rt: 0
    B["HistorySelect"] = lambda rt, *a: True
    B["PositionSelect"] = lambda rt, sym=None: rt.broker.pos != 0
    B["PositionSelectByTicket"] = lambda rt, t: rt.broker.pos != 0
    B["PositionGetTicket"] = lambda rt, i: 1000 if rt.broker.pos != 0 and int(i) == 0 else 0
    B["PositionGetSymbol"] = lambda rt, i: "SYMBOL" if rt.broker.pos != 0 and int(i) == 0 else ""
    B["OrderGetTicket"] = lambda rt, i: 0

    @reg("PositionGetInteger")
    def _(rt, prop):
        b = rt.broker
        return {"POSITION_TYPE": 0 if b.pos > 0 else 1, "POSITION_TICKET": 1000, "POSITION_IDENTIFIER": 1000,
                "POSITION_MAGIC": 0, "POSITION_TIME": rt.broker_open_time()}.get(prop, 0)

    @reg("PositionGetDouble")
    def _(rt, prop):
        b = rt.broker
        return {"POSITION_VOLUME": abs(b.pos), "POSITION_PRICE_OPEN": 0.0 if b.avg != b.avg else float(b.avg),
                "POSITION_SL": rt.pos_level("sl"), "POSITION_TP": rt.pos_level("tp"), "POSITION_PRICE_CURRENT": rt.ask(),
                "POSITION_PROFIT": rt.pos_profit(), "POSITION_SWAP": 0.0}.get(prop, 0.0)

    B["PositionGetString"] = lambda rt, prop: "SYMBOL" if prop == "POSITION_SYMBOL" else ""

    @reg("OrderSend")
    def _(rt, req, res=None, *a):
        act = int(req.action)
        typ = int(req.type)
        vol = to_float(req.volume)
        if res is not None:
            res.retcode, res.order, res.deal, res.volume, res.price = 10009, 1, 1, vol, rt.ask()
        if act == 6 or act == 7:                         # SLTP / MODIFY
            return rt.modify_pos(req.sl, req.tp)
        if act == 1:                                     # market deal
            if int(req.position) != 0:                   # closing a specific position
                return rt.close_pos()
            d = 1 if typ == 0 else -1
            if rt.broker.pos != 0 and (1 if rt.broker.pos > 0 else -1) != d and not (abs(rt.broker.pos) < vol):
                return rt.close_pos()
            return rt.open_pos(d, vol, req.sl, req.tp)
        if act == 5:                                     # pending
            return rt.open_pending(1 if typ % 2 == 0 else -1, vol, req.sl, req.tp)
        return True

    B["OrderSendAsync"] = B["OrderSend"]
    B["OrderCheck"] = lambda rt, req, res: True
    B["OrderCalcMargin"] = lambda rt, *a: False
    B["OrderCalcProfit"] = lambda rt, *a: False

    # ---- indicators
    def make_ind(kind):
        arity = _ARITY[kind]

        def f(rt, sym, tf, *args):
            args = list(args)
            legacy = len(args) > arity
            shift = 1
            mode = 0
            if legacy:
                shift = int(args[-1])
                extra = args[arity:-1]
                if kind in _MULTI_BUFFER_MODE and extra:
                    mode = int(extra[-1])
                args = args[:arity]
            view = rt.view(tf)
            key = (kind, tuple(float(x) if isinstance(x, (int, float)) else 0.0 for x in args), view.minutes)
            bufs = rt.ind_cache.get(key)
            if bufs is None:
                bufs = rt.ind_cache[key] = compute_buffers(kind, view.df, args)
            val = NAN
            if rt.in_tick:            # direct-value style: iMA(...) used as a number inside OnTick -> last closed bar
                j = (rt.i + 1 - shift) if view is rt.base else view.idx(shift, rt.i)
                if 0 <= j < view.n:
                    val = float(bufs[min(mode, len(bufs) - 1)][j])
            h = Handle(val)
            h.bufs, h.view, h.kind = bufs, view, kind
            rt.handles.append(h)
            return h
        return f

    for k in _ARITY:
        B[k] = make_ind(k)

    @reg("iCustom")
    def _(rt, *a):
        raise StrategyError("MQL5: iCustom() (custom indicators) is not supported by the interpreter; rebuild the indicator logic "
                            "inside the EA, or use a built-in indicator.")

    B["BarsCalculated"] = lambda rt, h: 10 ** 6

    @reg("CopyBuffer")
    def _(rt, h, buf, start, count, arr):
        if not isinstance(h, Handle):
            return -1
        bufs, view = h.bufs, h.view
        buf = int(buf)
        if buf >= len(bufs):
            return -1
        start, count = int(start), int(count)
        vals = []
        for sh in range(start, start + count):
            j = (rt.i + 1 - sh) if view is rt.base else view.idx(sh, rt.i)
            # a forming bar (shift 0) has no completed indicator value yet -> use the last completed one
            if sh == 0:
                j = min(j, rt.i) if view is rt.base else min(j, int(view.pos[rt.i]))
            if j < 0 or j >= view.n:
                vals.append(DBL_MAX)
                continue
            x = float(bufs[buf][j])
            vals.append(DBL_MAX if x != x else x)
        if not getattr(arr, "series", False):
            vals = vals[::-1]
        arr[:] = vals
        return len(vals)

    return B


def _safe_pow(a, b):
    try:
        return math.pow(a, b)
    except (ValueError, OverflowError):
        return NAN


def _fmt(fmt, args):
    out = to_str(fmt)
    try:
        return out % tuple(args)
    except (TypeError, ValueError):
        return out


# ---------------------------------------------------------------------------------------------------------------
# public interpreter
# ---------------------------------------------------------------------------------------------------------------
import re  # noqa: E402

_SL_ATR = re.compile(r"T58_SL_ATR_MULT\s*=\s*([\d.]+)")
_TP_ATR = re.compile(r"T58_TP_ATR_MULT\s*=\s*([\d.]+)")
_ATR_P = re.compile(r"T58_ATR_PERIOD\s*=\s*(\d+)")
_SL_PIPS = re.compile(r"T58_SL_PIPS\s*=\s*([\d.]+)")
_TP_PIPS = re.compile(r"T58_TP_PIPS\s*=\s*([\d.]+)")
# Any call that can open a position (CTrade methods or a raw OrderSend). An EA with none of these can never trade, so
# running it would silently return all-zero signals instead of telling the user the EA has no entry logic.
_ENTRY_CALL = re.compile(r"\b(?:Buy|Sell|BuyLimit|BuyStop|SellLimit|SellStop|PositionOpen|OrderOpen|OrderSend|OrderSendAsync)\s*\(")


class MQL5Interpreter:
    def __init__(self, source: str, overrides: dict | None = None):
        self.source = source
        self.overrides = overrides or {}
        clean, defines = L.preprocess(source)
        toks = tokenize(clean, defines)
        self.parser = Parser(toks).parse()
        if "OnTick" not in self.parser.funcs:
            if "OnCalculate" in self.parser.funcs:
                raise StrategyError("MQL5: this file is an indicator (OnCalculate), not an Expert Advisor with OnTick().")
            raise StrategyError("MQL5: no OnTick() function found -- the interpreter runs Expert Advisors.")
        comp = L.Compiler(self.parser)
        self.funcs, self.global_inits = comp.compile_all()
        self.warnings: list[str] = []
        self.inputs = [name for (name, is_input, typ, mk) in self.global_inits if is_input]
        self._has_entry_call = bool(_ENTRY_CALL.search(clean))

    def input_defaults(self) -> dict:
        """Name -> default for every `input` parameter (the optimizer uses this to find tunable genes)."""
        out = {}
        rt = MqlRuntime(pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [0.0],
                                      "timestamp": pd.date_range("2020-01-01", periods=1)}), self.parser, self.funcs,
                        self.global_inits, {})
        for name, is_input, typ, mk in self.global_inits:
            if is_input:
                try:
                    out[name] = mk(rt, {})
                except Exception:  # noqa: BLE001
                    pass
        return out

    def run(self, df: pd.DataFrame) -> StrategyResult:
        if len(df) < 20:
            raise StrategyError("MQL5: not enough bars to run the interpreter.")
        rt = MqlRuntime(df, self.parser, self.funcs, self.global_inits, self.overrides)
        rt.broker = SimBroker(df, default_qty=1.0, pyramiding=0)
        try:
            for name, is_input, typ, mk in self.global_inits:
                val = mk(rt, {})
                if is_input and name in self.overrides:
                    co = COERCE.get(typ)
                    val = co(self.overrides[name]) if co else self.overrides[name]
                rt.globals[name] = val
            if "OnInit" in self.funcs:
                rc = rt.call_user("OnInit", [])
                if rc not in (None, 0):
                    raise StrategyError(f"MQL5: OnInit() returned {rc} (initialisation failed) -- check the input parameters.")
            n = len(df)
            for i in range(n):
                rt.i = i
                rt.broker.begin_bar(i, rt.point)
                if i < n - 1:
                    rt.in_tick = True
                    try:
                        rt.call_user("OnTick", [])
                    finally:
                        rt.in_tick = False
                rt.broker.end_bar(i, rt.point)
        except StrategyError:
            raise
        except RecursionError:
            raise StrategyError("MQL5: the EA recursed too deeply.")
        except (TypeError, ValueError, KeyError, AttributeError, IndexError, ZeroDivisionError) as exc:
            raise StrategyError(f"MQL5: runtime error while interpreting the EA at bar {rt.i}: {type(exc).__name__}: {exc}") from exc
        # Checked after the run so a more specific runtime error (unsupported function etc.) always wins.
        if not self._has_entry_call:
            raise StrategyError(
                "MQL5: no Buy/Sell (trade.Buy/trade.Sell/OrderSend) call was found, so this EA can never open a "
                "trade. Add an entry call inside OnTick()."
            )
        self.warnings = list(rt.warnings)
        sig = pd.Series(rt.broker.signal, index=df.index)
        sl, tp = rt.broker.distances(df.index)
        return self._finish(df, sig, sl, tp)

    def _finish(self, df, sig, sl, tp) -> StrategyResult:
        from app.strategy.indicators import atr as atr_fn
        code = self.source
        sl_pips = float(_SL_PIPS.search(code).group(1)) if _SL_PIPS.search(code) else None
        tp_pips = float(_TP_PIPS.search(code).group(1)) if _TP_PIPS.search(code) else None
        m1, m2 = _SL_ATR.search(code), _TP_ATR.search(code)
        if m1 or m2:
            period = int(_ATR_P.search(code).group(1)) if _ATR_P.search(code) else 14
            a = atr_fn(df, period)
            if m1:
                sl, sl_pips = a * float(m1.group(1)), None
            if m2:
                tp, tp_pips = a * float(m2.group(1)), None
        return StrategyResult(name="MQL5 Strategy (interpreted)", source_type="mql5", signals=sig.astype(int),
                              stop_loss_pips=sl_pips if sl is None else None,
                              take_profit_pips=tp_pips if tp is None else None,
                              stop_loss_distance=sl, take_profit_distance=tp)
