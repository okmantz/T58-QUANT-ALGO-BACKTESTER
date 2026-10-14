import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import sys; 
import lab, fast, numpy as np, pandas as pd, itertools, time
import scan2
def gen_rb(A,sym,p):
    trs=[]; cap=scan2.CAP[sym]
    for di,(a,b) in enumerate(zip(A.dstart,A.dend)):
        if A.dow[a]>=5: continue
        m=A.mins[a:b]; Hh=A.h[a:b]; Ll=A.l[a:b]; Cc=A.c[a:b]
        rm=(m>=p["r0"])&(m<p["r1"])
        if rm.sum()<(p["r1"]-p["r0"])//5*0.8: continue
        rh=Hh[rm].max(); rl=Ll[rm].min(); rng=rh-rl
        if rng<=0: continue
        # vol filter: range relative to trailing mean of same-window ranges (causal via shift)
        A_.append(rng) if False else None
        for i in range(len(m)):
            if m[i]<p["r1"] or m[i]>p["r1"]+p["win"]: continue
            side=1 if Cc[i]>rh else (-1 if Cc[i]<rl else 0)
            if side==0: continue
            sd=min(max(rng*p["sf"],1.0),cap); r=fast.trade(A,sym,a+i,side,sd,sd*p["rr"],p["x"])
            if r: trs.append((a+i,r[0],r[2],r[3],r[1]))
            break
    return trs
if __name__=="__main__":
    sym=sys.argv[1]
    A=scan2.prep(sym,"5m","2020-01-01","2025-01-01"); rows=[]
    grid=dict(r0=[0,60,120],r1=[180,240,300],win=[60,180,300],sf=[0.5,1.0],rr=[1.0,2.0],x=[scan2.CLOSE[sym],690])
    keys=list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        p=dict(zip(keys,vals))
        if p["r1"]<=p["r0"]+60: continue
        trs=gen_rb(A,sym,p); d=fast.dollars(sym,trs); s=fast.summarize(A,sym,d)
        if not s or s["n"]<120: continue
        rows.append((p,s,fast.pass_sim(A,d)))
    rows.sort(key=lambda r:-(min(v[0] for v in r[1]["by_year"].values())*0.5+r[1]["PF"]))
    print(sym,"cands",len(rows))
    for p,s,ps in rows[:5]: print(p,{k:s[k] for k in("n","net","PF","wr")},{y:v[0] for y,v in s["by_year"].items()},ps)
