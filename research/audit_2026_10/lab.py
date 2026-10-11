import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import os, sys, warnings, pickle, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from app.data.importer import import_csv
from app.data.timeframe_resample import resample_ohlcv
from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig, with_prop_safety_defaults
from app.data.instrument_specs import apply_any_instrument_spec
from app.prop.presets import get_preset
from app.prop.rolling_evaluation import run_rolling_evaluation
from app.strategy.python import PythonStrategy
CACHE=os.path.join(_ROOT, "research", "audit_2026_10", ".cache"); os.makedirs(CACHE, exist_ok=True)
FILES={"NQ":"data/raw/NQ1!/futures_NQ.F_1m.parquet","ES":"data/raw/ES1!/futures_ES.F_1m.parquet","GC":"data/raw/GC1!/futures_GC.F_1m.parquet"}
def load(sym, tf=None):
    key=f"{CACHE}/{sym}_{tf or '1m'}.pkl"
    if os.path.exists(key): return pickle.load(open(key,"rb"))
    base=f"{CACHE}/{sym}_1m.pkl"
    if os.path.exists(base): df=pickle.load(open(base,"rb"))
    else:
        df=import_csv(FILES[sym]).dataframe; pickle.dump(df,open(base,"wb"))
    if tf: df=resample_ohlcv(df,tf); pickle.dump(df,open(key,"wb"))
    return df
def risk_for(sym, balance=50000.0, risk_dollars=None, pct=None):
    r=RiskConfig(initial_balance=balance, risk_mode="fixed" if risk_dollars else "percent",
                 risk_value=risk_dollars if risk_dollars else (pct or 1.0), pip_size=1.0, contract_size=1.0)
    r=apply_any_instrument_spec(r, sym)
    return r
RULES=get_preset("topstep_50k").to_prop_rules()
def evaluate(strat_path, sym, tf, start=None, end=None, risk_dollars=500.0, window=30, verbose=True):
    df=load(sym,tf)
    if start: df=df[df.timestamp>=pd.Timestamp(start,tz="UTC")]
    if end: df=df[df.timestamp<pd.Timestamp(end,tz="UTC")]
    df=df.reset_index(drop=True)
    st=PythonStrategy(strat_path)
    risk=with_prop_safety_defaults(risk_for(sym,risk_dollars=risk_dollars),RULES)
    from app.strategy.base import apply_strategy_trade_cap
    risk=apply_strategy_trade_cap(risk,st)
    bt=run_backtest(df,st,risk,lookahead_check="skip")
    tr=bt.trades; s=bt.statistics
    out=dict(n=len(tr), net=round(s.net_profit), pf=round(getattr(s,'profit_factor',0) or 0,2), wr=round(s.win_rate,1), mdd=round(s.max_drawdown_pct,1))
    if tr:
        ev=run_rolling_evaluation(tr, RULES, window_trading_days=window, stride=3)
        out.update(pass_rate=round(ev.pass_rate_pct,1), payout_rate=round(ev.first_payout_rate_pct,1), windows=ev.n_windows, med_days=ev.median_days_to_pass)
    if verbose: print(sym,tf,start,end,out)
    return bt,out

import re, itertools, tempfile
def variant(path, **kw):
    src=open(path).read()
    for k,v in kw.items():
        src,nsub=re.subn(rf"^{k}\s*=.*$", f"{k} = {v!r}" if not isinstance(v,tuple) else f"{k} = {v!r}", src, flags=re.M)
        assert nsub==1,(k,nsub)
    f=tempfile.NamedTemporaryFile("w",suffix=".py",delete=False,dir=CACHE); f.write(src); f.close(); return f.name
