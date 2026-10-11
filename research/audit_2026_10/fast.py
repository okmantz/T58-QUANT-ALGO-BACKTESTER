import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import numpy as np, pandas as pd
COST_SIDE={"NQ":0.5,"ES":0.5,"GC":0.2}      # spread+slippage per side in price points (1 tick + 1 tick for NQ/ES = .25+.25; GC .1+.1)
PV={"NQ":20.0,"ES":50.0,"GC":100.0}
COMM={"NQ":4.2,"ES":4.2,"GC":5.1}
class Arr:
    def __init__(s, df):
        s.o=df.open.to_numpy(float); s.h=df.high.to_numpy(float); s.l=df.low.to_numpy(float); s.c=df.close.to_numpy(float)
        ny=pd.to_datetime(df.timestamp).dt.tz_convert("America/New_York")
        s.mins=(ny.dt.hour*60+ny.dt.minute).to_numpy(); s.day=ny.dt.normalize().dt.tz_localize(None).to_numpy()
        s.dow=ny.dt.dayofweek.to_numpy(); s.ts=df.timestamp.to_numpy(); s.n=len(df)
        s.year=ny.dt.year.to_numpy()
        s.dstart=np.flatnonzero(np.r_[True,s.day[1:]!=s.day[:-1]]); s.dend=np.r_[s.dstart[1:],s.n]
def trade(A, sym, k, side, sd, td, x_at, maxbars=10**9):
    r=_trade(A,sym,k,side,sd,td,x_at)
    if r is None: return None
    return (r[0], r[1], r[2], sd)
def _trade(A, sym, k, side, sd, td, x_at):
    """signal on bar k close -> fill at open[k+1]. returns (R, pnl_pts_net, exit_idx)"""
    e=k+1
    if e>=A.n: return None
    cs=COST_SIDE[sym]
    ep=A.o[e]+side*cs
    stop=ep-side*sd; tp=ep+side*td if td else None
    b=A.dend[np.searchsorted(A.dstart,e,side="right")-1]
    for j in range(e,b):
        if j>e and A.mins[j]>=x_at:
            xp=A.o[j]-side*cs; return ((xp-ep)*side/sd, (xp-ep)*side, j)
        if side==1:
            if A.o[j]<=stop: xp=A.o[j]-cs; return ((xp-ep)/sd,xp-ep,j)
            if A.l[j]<=stop: xp=stop-cs; return ((xp-ep)/sd,xp-ep,j)
            if tp is not None and A.h[j]>=tp: xp=tp-cs; return ((xp-ep)/sd,xp-ep,j)
        else:
            if A.o[j]>=stop: xp=A.o[j]+cs; return ((ep-xp)/sd,ep-xp,j)
            if A.h[j]>=stop: xp=stop+cs; return ((ep-xp)/sd,ep-xp,j)
            if tp is not None and A.l[j]<=tp: xp=tp+cs; return ((ep-xp)/sd,ep-xp,j)
    j=b-1; xp=A.c[j]-side*cs; return ((xp-ep)*side/sd,(xp-ep)*side,j)

def dollars(sym, trs, risk=500.0):
    """trs: (k, R, exit_idx, sd, pts) -> list of (k, pnl$) applying fit_stop whole-contract sizing (skip if 1 contract > budget)."""
    out=[]
    for t in trs:
        k,_,xi,sd,pts=t
        n=int(risk//((sd+COST_SIDE[sym])*PV[sym]+COMM[sym]))
        if n<1: continue
        out.append((k, n*pts*PV[sym]-n*COMM[sym], xi, n))
    return out
def summarize(A, sym, d):
    if not d: return None
    ks=np.array([t[0] for t in d]); P=np.array([t[1] for t in d]); yr=A.year[ks+1]
    pf=lambda x: (x[x>0].sum()/max(-x[x<0].sum(),1e-9)) if (x<0).any() else 9.9
    return {"n":len(P),"net":round(P.sum()),"PF":round(pf(P),2),"wr":round((P>0).mean(),2),
            "by_year":{int(y):(round(pf(P[yr==y]),2),int((yr==y).sum())) for y in np.unique(yr)}}
def pass_sim(A, d, target=3000.0, dd=2000.0, window_days=None, stride=2):
    if not d: return None
    days=A.day[[t[0]+1 for t in d]]; P=np.array([t[1] for t in d])
    ud,inv=np.unique(days,return_inverse=True)
    dp=np.bincount(inv,weights=P)
    res=[]
    for si in range(0,len(ud),stride):
        bal=0.0; peak=0.0; status="timeout"; nd=0
        for p in dp[si:]:
            bal+=p; nd+=1; peak=max(peak,bal)
            if bal<=min(peak-dd,0.0): status="fail"; break
            if bal>=target: status="pass"; break
            if window_days and nd>=window_days: break
        res.append((status,nd))
    n=len(res); p=sum(1 for s,_ in res if s=="pass"); f=sum(1 for s,_ in res if s=="fail")
    return dict(windows=n, pass_pct=round(100*p/n,1), fail_pct=round(100*f/n,1), med_days=float(np.median([x for s,x in res if s=="pass"])) if p else None)
