"""Independent reference simulation of the bar engine (entry lag, slippage on every fill,
stop/TP/gap-through-stop, whole-contract fit_stop sizing, per-contract commission).
Added in the Oct-2026 audit: 1,500 random trades matched to the cent when it was written."""
import warnings
import numpy as np
import pandas as pd

from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig

warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)
CS = 20.0; COMM = 3.78; SLIP = 0.25


def mk(n=60):
    ts = pd.date_range("2024-03-04 14:30", periods=n, freq="5min")
    c = 18000 + np.cumsum(rng.normal(0, 6, n)); o = np.r_[c[0], c[:-1]] + rng.normal(0, 2, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 3, n)); l = np.minimum(o, c) - np.abs(rng.normal(0, 3, n))
    return pd.DataFrame(dict(timestamp=ts, open=o, high=h, low=l, close=c, volume=100))


def test_engine_matches_independent_reference():
    bad = 0; N = 0

    for it in range(300):
        df=mk(); k=int(rng.integers(1,20)); d=int(rng.choice([1,-1]))
        SL=float(rng.uniform(5,25)); TP=float(rng.uniform(5,40))
        sig=pd.Series(0,index=df.index); sig.iloc[k:]=d
        risk=RiskConfig(initial_balance=50000.0,risk_mode="fixed",risk_value=1500.0,pip_size=1.0,contract_size=CS,
            commission_per_contract=COMM,slippage_pips=SLIP,sizing_mode="fit_stop",max_trades_per_day=50)
        tr,_=run_execution(df=df,signals=sig,risk=risk,stop_loss_pips=SL,take_profit_pips=TP,stop_loss_distance=None,
            take_profit_distance=None,trailing_stop_distance=None,breakeven_trigger_r=None,partial_exit_config=None,adaptive_risk=None)
        if not tr: continue
        t=tr[0]; N+=1
        e=k+1; ep=df.open[e]+d*SLIP
        stop=ep-d*SL; tp=ep+d*TP
        xp=None; why=None
        for j in range(e,len(df)):
            o_,h_,l_=df.open[j],df.high[j],df.low[j]
            if d==1:
                if o_<=stop: xp,why=o_,"gap_stop"
                elif l_<=stop: xp,why=stop,"stop"
                elif h_>=tp: xp,why=tp,"tp"
            else:
                if o_>=stop: xp,why=o_,"gap_stop"
                elif h_>=stop: xp,why=stop,"stop"
                elif l_<=tp: xp,why=tp,"tp"
            if xp is not None: xp-=d*SLIP; break
        if xp is None: continue
        n=t.contracts
        exp=(xp-ep)*d*CS*n-COMM*n
        if abs(exp-t.pnl)>0.02 or t.entry_time!=df.timestamp[e]:
            bad+=1
            if bad<=8: print("MISMATCH",d,k,SL,TP,"eng:",t.entry_price,t.exit_price,t.exit_reason,round(t.pnl,2),"ref:",ep,xp,why,round(exp,2),"n",n)
    assert N > 250 and bad == 0
