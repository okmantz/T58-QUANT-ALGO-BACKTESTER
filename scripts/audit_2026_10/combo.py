import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import sys; 
import scan2, fast, numpy as np
pd_={"or":30,"win":0,"minr":0.3,"maxr":9.9,"sf":0.5,"rr":2.5,"bias":0.25}
pf_={"or":15,"win":60,"minr":0.3,"maxr":9.9,"sf":1.0,"rr":2.5}
for lab,(s,e) in {"DEV":("2020-01-01","2025-01-01"),"OOS":("2025-01-01","2026-10-01")}.items():
    A=scan2.prep("NQ","5m",s,e)
    t1=scan2.gen(A,"NQ","drive",pd_); t2=scan2.gen(A,"NQ","fail",pf_)
    byday={}
    for t in t1+t2:
        d=A.day[t[0]]
        if d not in byday or t[0]<byday[d][0]: byday[d]=t
    tr=sorted(byday.values())
    d=fast.dollars("NQ",tr); sm=fast.summarize(A,"NQ",d)
    print(lab,len(t1),len(t2),{k:sm[k] for k in("n","net","PF","wr")},{y:v for y,v in sm["by_year"].items()},fast.pass_sim(A,d))
