import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import sys; 
import lab, fast, numpy as np, pandas as pd, itertools, time, json
CAP={"NQ":22.0,"ES":8.5,"GC":4.4}          # stop cap (pts) so 1 contract fits a $500 budget with costs
OPEN={"NQ":570,"ES":570,"GC":500}          # RTH open minute (ET): 9:30 ; GC 8:20
CLOSE={"NQ":945,"ES":945,"GC":810}         # time exit 15:45 / GC 13:30
def prep(sym,tf,start,end):
    df=lab.load(sym,tf); df=df[(df.timestamp>=pd.Timestamp(start,tz="UTC"))&(df.timestamp<pd.Timestamp(end,tz="UTC"))].reset_index(drop=True)
    A=fast.Arr(df)
    # prior-session range mean (rolling 14) per day
    dr=[]
    for a,b in zip(A.dstart,A.dend):
        m=A.mins[a:b]; rth=(m>=OPEN[sym])&(m<CLOSE[sym])
        dr.append(A.h[a:b][rth].max()-A.l[a:b][rth].min() if rth.any() else np.nan)
    A.atr=pd.Series(dr).rolling(14,min_periods=5).mean().shift(1).to_numpy()
    return A
def gen(A,sym,fam,p):
    o=OPEN[sym]; x=CLOSE[sym]; cap=CAP[sym]; trs=[]
    for di,(a,b) in enumerate(zip(A.dstart,A.dend)):
        if A.dow[a]>=5 or not np.isfinite(A.atr[di]): continue
        m=A.mins[a:b]; Hh=A.h[a:b]; Ll=A.l[a:b]; Cc=A.c[a:b]
        orm=(m>=o)&(m<o+p["or"])
        if orm.sum()<p["or"]//5: continue
        orh=Hh[orm].max(); orl=Ll[orm].min(); rng=orh-orl
        if rng<=0 or not (p["minr"]*A.atr[di]<=rng<=p["maxr"]*A.atr[di]): continue
        end=o+p["or"]+p["win"]
        ks=[i for i in range(len(m)) if o+p["or"]<=m[i]<=end]
        if fam=="orb":
            for i in ks:
                side=1 if Cc[i]>orh else (-1 if Cc[i]<orl else 0)
                if side==0: continue
                sd=min(max(rng*p["sf"],1.0),cap); r=fast.trade(A,sym,a+i,side,sd,sd*p["rr"],x)
                if r: trs.append((a+i,r[0],r[2],r[3],r[1]))
                break
        elif fam=="fail":      # failed breakout: pierce one side, close back inside -> trade the other way
            pierced=0
            for i in ks:
                if Hh[i]>orh and pierced==0: pierced=1
                if Ll[i]<orl and pierced==0: pierced=-1
                if pierced==1 and Cc[i]<orh and Cc[i]>orl: side=-1
                elif pierced==-1 and Cc[i]>orl and Cc[i]<orh: side=1
                else: continue
                sd=min(max(rng*p["sf"],1.0),cap); r=fast.trade(A,sym,a+i,side,sd,sd*p["rr"],x)
                if r: trs.append((a+i,r[0],r[2],r[3],r[1]))
                break
        elif fam=="drive":     # first OR bar(s) direction + follow-through on close beyond mid
            side=1 if Cc[orm.nonzero()[0][-1]]>(orh+orl)/2+p["bias"]*rng else (-1 if Cc[orm.nonzero()[0][-1]]<(orh+orl)/2-p["bias"]*rng else 0)
            if side==0: continue
            i=orm.nonzero()[0][-1]
            sd=min(max(rng*p["sf"],1.0),cap); r=fast.trade(A,sym,a+i,side,sd,sd*p["rr"],x)
            if r: trs.append((a+i,r[0],r[2],r[3],r[1]))
    return trs
def run(sym,fam,grid,start="2020-01-01",end="2025-01-01",tf="5m"):
    A=prep(sym,tf,start,end); rows=[]
    keys=list(grid); 
    for vals in itertools.product(*[grid[k] for k in keys]):
        p=dict(zip(keys,vals)); trs=gen(A,sym,fam,p); d=fast.dollars(sym,trs)
        s=fast.summarize(A,sym,d)
        if not s or s["n"]<120: continue
        ps=fast.pass_sim(A,d); rows.append((p,s,ps))
    return rows,A
if __name__=="__main__":
    sym=sys.argv[1]; fam=sys.argv[2]
    grid=dict(**{"or":[15,30],"win":[60,120,240],"minr":[0.0,0.3],"maxr":[9.9,1.6],"sf":[0.5,1.0],"rr":[1.0,1.5,2.5],"bias":[0.0,0.25]})
    if fam!="drive": grid.pop("bias")
    else: grid.pop("win"); grid["win"]=[0]
    t=time.time(); rows,A=run(sym,fam,grid); print(sym,fam,"cands",len(rows),"secs",round(time.time()-t))
    rows.sort(key=lambda r:-(min(v[0] for v in r[1]["by_year"].values())*0.5+r[1]["PF"]))
    for p,s,ps in rows[:8]:
        print(p,{k:s[k] for k in("n","net","PF","wr")},{y:v[0] for y,v in s["by_year"].items()},ps)
