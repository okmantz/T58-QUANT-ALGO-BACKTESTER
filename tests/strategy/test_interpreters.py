"""Pine and MQL5 interpreters: parity with the vectorized indicators, routing, HTF alignment and error messages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy import indicators as I
from app.strategy.base import StrategyError
from app.strategy.mql5 import MQL5Strategy
from app.strategy.mql5_interp import MQL5Interpreter
from app.strategy.pine_interp import PineInterpreter
from app.strategy.pinescript import PineScriptStrategy


def _bars(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    close = 1.1 + np.cumsum(rng.normal(0, 0.0006, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 0.0004, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.0004, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(100, 1000, n).astype(float)})


DF = _bars()


def _pine_flip(cond: str) -> pd.Series:
    code = f'strategy("t")\nx = {cond}\nif x\n    strategy.entry("L", strategy.long)\nelse\n    strategy.entry("S", strategy.short)\n'
    return PineInterpreter(code).run(DF).signals


@pytest.mark.parametrize("cond,ref", [
    ("ta.rsi(close,14) > 50", lambda d: I.rsi(d.close, 14).gt(50)),
    ("ta.ema(close,20) > ta.ema(close,50)", lambda d: I.ema(d.close, 20).gt(I.ema(d.close, 50))),
    ("ta.sma(close,20) > close", lambda d: I.sma(d.close, 20).gt(d.close)),
    ("ta.atr(14) > ta.sma(ta.atr(14), 50)", lambda d: I.atr(d, 14).gt(I.sma(I.atr(d, 14), 50))),
    ("close > ta.highest(high,20)[1]", lambda d: d.close.gt(I.highest_high(d.high, 20).shift(1))),
])
def test_pine_ta_matches_vectorized_indicators(cond, ref):
    got = _pine_flip(cond)
    want = pd.Series(np.where(ref(DF), 1, -1), index=DF.index)
    assert (got == want)[300:].mean() == 1.0


def test_pine_state_functions_arrays_and_tuples():
    code = """
strategy("s", overlay=true)
len = input.int(20)
[mid, up, lo] = ta.bb(close, len, 2.0)
var int n = 0
var arr = array.new_float(0)
f(x) =>
    x * 2
n := n + 1
array.push(arr, f(close))
if close < lo and n > 30
    strategy.entry("L", strategy.long)
if close > mid and strategy.position_size > 0
    strategy.close("L")
"""
    res = PineInterpreter(code).run(DF)
    assert set(res.signals.unique()) <= {0, 1}
    assert res.signals.sum() > 0


def test_pine_security_is_lookahead_safe():
    """A higher-timeframe value must never be visible before that bar has completed."""
    code = 'strategy("h")\nh = request.security(syminfo.tickerid, "60", close)\nif h > 0\n    strategy.entry("L", strategy.long)\n'
    rt_vals = []
    res = PineInterpreter(code).run(DF)
    assert res.signals.iloc[0] == 0 and res.signals.iloc[1] == 0      # first hour is not complete yet
    assert res.signals.iloc[-1] == 1


def test_pine_strategy_exit_sets_stop_distance():
    code = 'strategy("e")\natr = ta.atr(14)\nif ta.crossover(ta.ema(close,5), ta.ema(close,20))\n    strategy.entry("L", strategy.long)\n    strategy.exit("x", "L", stop=close - 2*atr, limit=close + 4*atr)\n'
    res = PineInterpreter(code).run(DF)
    assert res.stop_loss_distance is not None and res.take_profit_distance is not None
    assert (res.stop_loss_distance.dropna() > 0).all()


def test_pine_routing_simple_script_uses_fast_parser_complex_uses_interpreter():
    simple = 'strategy("a")\nf = ta.ema(close, 10)\ns = ta.ema(close, 30)\nif ta.crossover(f, s)\n    strategy.entry("L", strategy.long)\n'
    st = PineScriptStrategy(simple)
    st.generate(DF)
    assert st.engine_used == "parser"
    st2 = PineScriptStrategy('strategy("b")\nvar x = 0\nx := x + 1\nif x > 5\n    strategy.entry("L", strategy.long)\n')
    st2.generate(DF)
    assert st2.engine_used == "interpreter"


def test_pine_unknown_function_fails_loudly():
    with pytest.raises(StrategyError, match="not supported"):
        PineInterpreter('strategy("z")\nx = ta.nonexistent(close, 3)\nif x > 1\n    strategy.entry("L", strategy.long)\n').run(DF)


EA = r"""
#include <Trade/Trade.mqh>
input int FastPeriod = 10;
input int SlowPeriod = 30;
CTrade trade;
int hf, hs;
int OnInit(){ hf = iMA(_Symbol, PERIOD_CURRENT, FastPeriod, 0, MODE_EMA, PRICE_CLOSE);
              hs = iMA(_Symbol, PERIOD_CURRENT, SlowPeriod, 0, MODE_EMA, PRICE_CLOSE);
              if(hf == INVALID_HANDLE) return INIT_FAILED; return INIT_SUCCEEDED; }
void OnTick(){
   double a[], b[];
   ArraySetAsSeries(a, true); ArraySetAsSeries(b, true);
   if(CopyBuffer(hf,0,1,1,a) < 1 || CopyBuffer(hs,0,1,1,b) < 1) return;
   bool up = a[0] > b[0];
   long typ = -1;
   if(PositionSelect(_Symbol)) typ = PositionGetInteger(POSITION_TYPE);
   if(up && typ != POSITION_TYPE_BUY)  { trade.PositionClose(_Symbol); trade.Buy(0.1); }
   if(!up && typ != POSITION_TYPE_SELL){ trade.PositionClose(_Symbol); trade.Sell(0.1); }
}
"""


def test_mql5_handles_match_vectorized_ema_cross():
    got = MQL5Interpreter(EA).run(DF).signals
    want = pd.Series(np.where(I.ema(DF.close, 10).gt(I.ema(DF.close, 30)), 1, -1), index=DF.index)
    assert (got == want)[300:-1].mean() == 1.0


def test_mql5_input_override_changes_behaviour():
    a = MQL5Interpreter(EA).run(DF).signals
    b = MQL5Interpreter(EA, overrides={"FastPeriod": 5}).run(DF).signals
    assert not a.equals(b)


def test_mql5_routing_and_input_defaults():
    st = MQL5Strategy(EA)
    st.generate(DF)
    assert st.engine_used == "interpreter"
    assert MQL5Interpreter(EA).input_defaults() == {"FastPeriod": 10, "SlowPeriod": 30}


def test_mql5_indicator_file_is_rejected_clearly():
    with pytest.raises(StrategyError, match="indicator"):
        MQL5Interpreter("int OnCalculate(const int a){ return a; }")


def test_mql5_icustom_is_rejected_clearly():
    code = "int h; int OnInit(){ h = iCustom(_Symbol,0,\"x\"); return 0; } void OnTick(){}"
    with pytest.raises(StrategyError, match="iCustom"):
        MQL5Interpreter(code).run(DF)


def test_optimizer_finds_inputs_in_interpreted_scripts():
    from app.optimize.code_parameter_space import apply_code_genome, discover_code_genes
    st = MQL5Strategy(EA)
    genes = discover_code_genes(st)
    assert {g.name for g in genes} == {"FastPeriod", "SlowPeriod"}
    patched = apply_code_genome(st.code, genes, [7, 21])
    assert "FastPeriod = 7;" in patched and "SlowPeriod = 21;" in patched
